#!/usr/bin/env python3
"""On-demand torch.profiler around GPUModelRunner.execute_model.

This build has no VLLM_TORCH_PROFILER_DIR / /start_profile. With
VLLM_STEP_PROFILE=1, touching /tmp/profile_trigger inside the container makes
the next 24 engine steps run under torch.profiler (CPU+CUDA) and writes a
chrome trace to /tmp/step_profile_<n>.json (docker cp it out). Traces are
tens of MB each: starting a new capture prunes all but the 3 newest, so /tmp
cannot fill up across repeated captures. Zero cost when idle: one
os.path.exists per step.
"""

import ast
import os
import sys

# Site-packages dir: the Dockerfile's ARG SP (build ARGs are visible to RUN).
SP = os.environ.get("SP", "/usr/local/lib/python3.12/dist-packages")
MR = f"{SP}/vllm/v1/worker/gpu/model_runner.py"

src = open(MR).read()
assert "class GPUModelRunner" in src and "_step_profile" not in src

src += '''

# --- appended by patch_step_profile.py (VLLM_STEP_PROFILE=1) ---------------
import glob as _sp_glob
import os as _sp_os

if _sp_os.environ.get("VLLM_STEP_PROFILE", "0") == "1":
    _sp_orig_execute = GPUModelRunner.execute_model
    def _sp_trace_num(path):
        try:
            return int(path.rsplit("_", 1)[1].split(".", 1)[0])
        except ValueError:
            return -1

    def _sp_traces():
        # Numeric order: a string sort puts step_profile_10 before _2 and
        # would prune the NEWEST traces once there are ten or more.
        return sorted(_sp_glob.glob("/tmp/step_profile_*.json"), key=_sp_trace_num)

    # Continue numbering after a restart instead of overwriting old traces.
    _sp_state = {"prof": None, "steps": 0,
                 "n": max([0] + [_sp_trace_num(p) for p in _sp_traces()])}
    _SP_TRIGGER = "/tmp/profile_trigger"
    _SP_STEPS = int(_sp_os.environ.get("VLLM_STEP_PROFILE_STEPS", "24"))
    _SP_KEEP = 3  # old traces to keep around (each is tens of MB)

    def _step_profile_execute(self, *args, **kwargs):
        st = _sp_state
        if st["prof"] is None and _sp_os.path.exists(_SP_TRIGGER):
            try:
                _sp_os.remove(_SP_TRIGGER)
            except OSError:
                pass
            for _p in _sp_traces()[:-_SP_KEEP]:
                try:
                    _sp_os.remove(_p)
                except OSError:
                    pass
            import torch.profiler as _tp

            st["prof"] = _tp.profile(
                activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA]
            )
            st["prof"].__enter__()
            st["steps"] = 0
            logger.info("[step-profile] capturing %d steps", _SP_STEPS)
        out = _sp_orig_execute(self, *args, **kwargs)
        if st["prof"] is not None:
            st["steps"] += 1
            if st["steps"] >= _SP_STEPS:
                st["prof"].__exit__(None, None, None)
                st["n"] += 1
                path = f"/tmp/step_profile_{st['n']}.json"
                st["prof"].export_chrome_trace(path)
                logger.info("[step-profile] trace written to %s", path)
                st["prof"] = None
        return out

    GPUModelRunner.execute_model = _step_profile_execute
'''
open(MR, "w").write(src)
ast.parse(open(MR).read())
print("patch_step_profile.py applied OK", file=sys.stderr)
