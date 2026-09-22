"""Engine-side Prometheus sidecar for this fork's gauges (IMPROVEMENTS-RU B3/B4).

vLLM serves /metrics from the API-server process, but the fork's interesting
counters live in the EngineCore process: the PLE mmap gather, the mamba
state-copy guard tripwire and the never-evict pin. Rather than plumbing them
through the SchedulerStats IPC channel (which is serialized and drops unknown
fields), this module serves them from the engine process itself with
prometheus_client — the same library vLLM uses — on a separate port:

  VLLM_CUSTOM_METRICS_PORT     inside-container port, default 18400; 0 = off.
                               scripts/serve-intel-ar.sh publishes it as
                               METRICS_PORT (host side, 0 = not published).
  VLLM_CUSTOM_METRICS_INTERVAL refresher period in seconds, default 5.

Metrics (all created up front, so they are visible as 0 even before traffic):
  vllm:ple_mmap_ops_total            lookup ops (counter)
  vllm:ple_mmap_op_ms_total          op wall time; rate()/rate(ops) = ms/op
  vllm:ple_mmap_gather_ms_total      disk-read share of the above
  vllm:ple_mmap_rows_total / vllm:ple_mmap_bytes_total
  vllm:ple_mmap_prefetch_hit_total / vllm:ple_mmap_prefetch_miss_total
  vllm:mamba_state_copy_guard_total  gauge; tripwire, expect a constant 0
  vllm:never_evict_blocks_reserved / vllm:never_evict_pin_queue_blocks
  vllm:never_evict_pin_bytes

Data sources register themselves from the engine process: vllm_ple_mmap keeps
lifetime counters (_STATS_LIFE, never reset — the log-window _STATS still
resets every VLLM_PLE_MMAP_STATS_SEC), mamba_utils_guarded pushes its guard
mirror every 512 steps, and the never-evict scheduler hook stores a Scheduler
reference whose BlockPool is read at refresh time.

The sidecar starts lazily, on the FIRST push from any source, and idempotently:
the engine process is the one that has data (the scheduler's __init__ alone
starts it at engine boot), whereas a frontend process that merely imports the
model module has nothing to report — if it grabbed the port first, every
metric would read 0 forever. Assumes the single-process deployment this fork
targets (one DGX Spark, no tensor parallelism); with several worker processes
the first one to push owns the endpoint. Every failure is logged once and
swallowed — telemetry must never take the server down.
"""
import errno
import os
import threading

_PORT = int(os.environ.get("VLLM_CUSTOM_METRICS_PORT", "18400") or 0)
_INTERVAL = min(60.0, max(1.0, float(os.environ.get("VLLM_CUSTOM_METRICS_INTERVAL", "5") or 5)))

_started = False
_ensured = False
_lock = threading.Lock()

# Engine-side sources (same process as this module; see module docstring).
# Every push also kicks start() (lazy, idempotent) so the endpoint is bound by
# the process that actually has the data.
_guard_hits = [0]      # int, pushed by mamba_utils_guarded's telemetry refresh
_last_guard = [0]      # last value exported to Prometheus (counter delta)
_pin_source = [None]   # Scheduler, set by patch_never_evict's injected init
_last_life = {}        # last vllm_ple_mmap._STATS_LIFE snapshot (counter deltas)
_warned = [False]


def set_guard_hits(hits: int) -> None:
    _guard_hits[0] = int(hits)
    start()


def set_pin_source(scheduler) -> None:
    _pin_source[0] = scheduler
    start()


def _log_once(msg: str) -> None:
    if _warned[0]:
        return
    _warned[0] = True
    try:
        from vllm.logger import init_logger
        init_logger(__name__).warning("%s", msg)
    except Exception:
        import logging
        logging.getLogger(__name__).warning("%s", msg)


# --- metric objects ---------------------------------------------------------
_registry = None
_m = {}


def _ensure_metrics() -> None:
    """Create the registry and metric objects (idempotent; no threads)."""
    global _registry, _ensured
    if _ensured:
        return
    from prometheus_client import CollectorRegistry, Counter, Gauge

    _registry = CollectorRegistry()
    c = lambda name, doc: Counter(name, doc, registry=_registry)  # noqa: E731
    g = lambda name, doc: Gauge(name, doc, registry=_registry)  # noqa: E731
    _m.update({
        "ops": c("vllm:ple_mmap_ops", "PLE mmap lookup ops (engine-side, lifetime)."),
        "op_ms": c("vllm:ple_mmap_op_ms",
                   "PLE mmap lookup wall time in ms (hash+gather+H2D), lifetime. "
                   "rate(ple_mmap_op_ms_total)/rate(ple_mmap_ops_total) = ms/op."),
        "gather_ms": c("vllm:ple_mmap_gather_ms",
                       "PLE mmap disk-read time in ms, lifetime."),
        "rows": c("vllm:ple_mmap_rows", "PLE table rows gathered, lifetime."),
        "bytes": c("vllm:ple_mmap_bytes", "PLE table bytes read, lifetime."),
        "pf_hit": c("vllm:ple_mmap_prefetch_hit", "Batch-assembly prefetch hits."),
        "pf_miss": c("vllm:ple_mmap_prefetch_miss", "Batch-assembly prefetch misses."),
        "guard": c("vllm:mamba_state_copy_guard",
                   "Out-of-range mamba state-copy block ids skipped by the "
                   "bounds guard (tripwire; expected to stay 0 — any growth "
                   "is a new bug, it is also ERROR-logged). Sampled every "
                   "512 engine steps."),
        "pin_reserved": g("vllm:never_evict_blocks_reserved",
                              "KV blocks held by the never-evict pin."),
        "pin_queue": g("vllm:never_evict_pin_queue_blocks",
                           "KV blocks still in the never-evict pin queue "
                           "(rest of the reserved set is on loan in caches)."),
        "pin_bytes": g("vllm:never_evict_pin_bytes",
                           "Bytes pinned by the never-evict pin."),
    })
    _ensured = True


def _refresh_once() -> None:
    """Pull every source once and update the metrics. Never raises."""
    try:
        _ensure_metrics()
        # PLE mmap: lifetime counters -> prometheus counter deltas.
        import vllm_ple_mmap as _pm
        life = getattr(_pm, "_STATS_LIFE", None)
        if life:
            for key, metric in (("calls", "ops"), ("op_ms", "op_ms"),
                                ("gather_ms", "gather_ms"), ("rows", "rows"),
                                ("bytes", "bytes"), ("pf_hit", "pf_hit"),
                                ("pf_miss", "pf_miss")):
                now = life.get(key, 0)
                delta = now - _last_life.get(key, 0)
                if delta > 0:
                    _m[metric].inc(delta)
            _last_life.clear()
            _last_life.update(life)

        # Mamba state-copy guard tripwire (mirror refreshed every 512 steps;
        # monotonic, so it exports as a counter).
        if _guard_hits[0] > _last_guard[0]:
            _m["guard"].inc(_guard_hits[0] - _last_guard[0])
            _last_guard[0] = _guard_hits[0]

        # Never-evict pin: read the scheduler's BlockPool at refresh time.
        sched = _pin_source[0]
        pool = getattr(getattr(sched, "kv_cache_manager", None), "block_pool", None)
        get_pin_stats = getattr(pool, "get_pin_stats", None)
        if sched is not None and callable(get_pin_stats):
            reserved, held, per_group = get_pin_stats()
            _m["pin_reserved"].set(reserved)
            _m["pin_queue"].set(held)
            page_sizes = getattr(sched, "_pin_page_sizes", None)
            num_bytes = sum(count * page_sizes[gid]
                            for gid, count in per_group.items()) if page_sizes else 0
            _m["pin_bytes"].set(num_bytes)
    except Exception as exc:  # never let telemetry break anything
        _log_once(f"vllm_custom_metrics: refresh failed once and will keep "
                  f"retrying silently: {exc!r}")


def _refresher() -> None:
    import time
    while True:
        _refresh_once()
        time.sleep(_INTERVAL)


def start() -> None:
    """Idempotently start the refresher thread and the HTTP endpoint."""
    global _started
    if _started or _PORT <= 0:
        return
    with _lock:
        if _started:
            return
        _started = True
        try:
            _ensure_metrics()
            from prometheus_client import start_wsgi_server
            threading.Thread(target=_refresher, name="vllm-custom-metrics",
                             daemon=True).start()
            start_wsgi_server(_PORT, addr="0.0.0.0", registry=_registry)
            _announce()
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                # Another process of this container got there first (expected
                # only with multi-process executors): not an error, but the
                # endpoint then reports that process's view.
                _log_once(f"vllm_custom_metrics: port {_PORT} already bound by "
                          f"another process; metrics are served from there")
            else:
                _log_once(f"vllm_custom_metrics: sidecar not started: {exc!r}")
        except Exception as exc:
            _log_once(f"vllm_custom_metrics: sidecar not started: {exc!r}")


def _announce() -> None:
    """Log that the sidecar is up. Never silent: if vLLM's logger is unusable,
    fall back to stderr — a sidecar that starts without saying so is exactly
    the failure mode this endpoint must not have."""
    msg = ("vllm_custom_metrics: engine-side Prometheus sidecar on :%d "
           "(refresh every %.0fs)" % (_PORT, _INTERVAL))
    try:
        from vllm.logger import init_logger
        init_logger(__name__).info(msg)
    except Exception:
        import sys
        print(msg, file=sys.stderr, flush=True)
