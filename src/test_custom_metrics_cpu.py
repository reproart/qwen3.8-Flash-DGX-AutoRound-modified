"""CPU test for vllm_custom_metrics: sources -> refreshed gauges/counters.

No network and no vLLM: start() is never called (port 0), the PLE module is a
fake injected into sys.modules, and the exposition is generated straight from
the registry. Checks that every metric appears with the right value and that
counters accumulate by DELTA across refreshes (not by absolute value).
"""
import os
import sys
import types

os.environ["VLLM_CUSTOM_METRICS_PORT"] = "0"  # never start the HTTP server here
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vllm_custom_metrics as cm  # noqa: E402

# Fake PLE source: lifetime counters, as the real vllm_ple_mmap._STATS_LIFE.
ple = types.ModuleType("vllm_ple_mmap")
ple._STATS_LIFE = {
    "calls": 100, "op_ms": 250.0, "gather_ms": 120.0,
    "rows": 1000, "bytes": 160_000, "pf_hit": 7, "pf_miss": 3,
}
sys.modules["vllm_ple_mmap"] = ple


# Fake never-evict source: a Scheduler-ish object with a BlockPool-ish pool.
class _Pool:
    def get_pin_stats(self):
        return 5, 2, {0: 3, 1: 2}  # reserved, in-queue, per-group blocks


class _Sched:
    kv_cache_manager = types.SimpleNamespace(block_pool=_Pool())
    _pin_page_sizes = [160, 3200]


cm.set_guard_hits(4)
cm.set_pin_source(_Sched())
cm._refresh_once()

from prometheus_client import generate_latest  # noqa: E402

text = generate_latest(cm._registry).decode()
expected = [
    ("vllm:ple_mmap_ops_total", 100.0),
    ("vllm:ple_mmap_op_ms_total", 250.0),
    ("vllm:ple_mmap_gather_ms_total", 120.0),
    ("vllm:ple_mmap_rows_total", 1000.0),
    ("vllm:ple_mmap_bytes_total", 160_000.0),
    ("vllm:ple_mmap_prefetch_hit_total", 7.0),
    ("vllm:ple_mmap_prefetch_miss_total", 3.0),
    ("vllm:mamba_state_copy_guard_total", 4.0),
    ("vllm:never_evict_blocks_reserved", 5.0),
    ("vllm:never_evict_pin_queue_blocks", 2.0),
    ("vllm:never_evict_pin_bytes", 3 * 160 + 2 * 3200),
]
for name, val in expected:
    assert f"{name} {val}" in text, f"{name} {val} not in exposition:\n{text}"
print("first refresh: all", len(expected), "metrics present with expected values")

# Second refresh: counters must grow by the DELTA only (100 -> 150 = +50),
# not by the new absolute value.
ple._STATS_LIFE["calls"] = 150
cm._refresh_once()
text = generate_latest(cm._registry).decode()
assert "vllm:ple_mmap_ops_total 150.0" in text, text
assert "vllm:ple_mmap_op_ms_total 250.0" in text  # unchanged source -> unchanged
print("second refresh: delta accounting OK (100 -> 150)")

# A source that disappears (e.g. no PLE module) must not raise: the refresh
# reads getattr(..., None) paths. Simulate by dropping _STATS_LIFE.
del ple._STATS_LIFE
cm._refresh_once()
print("missing-source refresh: no exception")

# set_pin_source(None): pin gauges keep their last value, nothing raises.
cm.set_pin_source(None)
cm._refresh_once()
text = generate_latest(cm._registry).decode()
assert "vllm:never_evict_blocks_reserved 5.0" in text, text
print("pin source detached: gauges hold last value, no exception")

# _announce must NEVER be silent: a sidecar that starts without saying so is
# the failure mode this endpoint exists to avoid.
import contextlib  # noqa: E402
import io  # noqa: E402
import logging  # noqa: E402

# 1) No logging configured at all (no handler anywhere): stderr fallback.
_buf = io.StringIO()
_root_handlers = logging.getLogger().handlers[:]
logging.getLogger().handlers.clear()
try:
    with contextlib.redirect_stderr(_buf):
        cm._announce()
finally:
    logging.getLogger().handlers[:] = _root_handlers
assert "sidecar on" in _buf.getvalue(), f"announce was silent: {_buf.getvalue()!r}"
print("announce: non-silent fallback OK")

# 2) vLLM-style setup: a handler on the "vllm" logger tree only, propagation
# off. The announce must land there (it used to go to a logger outside the
# tree and vanish from the server log).
_vl = logging.getLogger("vllm")
_cap = io.StringIO()
_h = logging.StreamHandler(_cap)
_vl.addHandler(_h)
_vl.setLevel(logging.INFO)
_vl.propagate = False
try:
    cm._announce()
finally:
    _vl.removeHandler(_h)
    _vl.propagate = True
    _vl.setLevel(logging.NOTSET)
assert "sidecar on" in _cap.getvalue(), f"announce missed the vllm logger: {_cap.getvalue()!r}"
print("announce: reaches the vllm logger tree OK")

print("ALL OK")
