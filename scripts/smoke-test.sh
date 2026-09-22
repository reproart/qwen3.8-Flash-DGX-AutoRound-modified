#!/usr/bin/env bash
# Quick check that the server is up, coherent, and measure prefill + decode.
#   scripts/smoke-test.sh [host:port]
# MODEL (default qwen3.8-flash-next) and API_KEY (optional bearer token,
# matching serve.sh's API_KEY) come from the environment:
#   MODEL=qwen scripts/smoke-test.sh localhost:8000
set -euo pipefail
EP="${1:-localhost:18300}"
BASE="http://$EP"
MODEL="${MODEL:-qwen3.8-flash-next}"
API_KEY="${API_KEY:-}"
export MODEL API_KEY   # read by the python snippets below

echo ">> health"   # /health stays unauthenticated even with --api-key
curl -sf -m 5 "$BASE/health" >/dev/null && echo "   OK" || { echo "   not ready"; exit 1; }

echo ">> coherence"
python3 - "$BASE" <<'PY'
import json, os, sys, urllib.request
base = sys.argv[1]
hdr = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    hdr["Authorization"] = "Bearer " + os.environ["API_KEY"]
data = json.dumps({"model": os.environ.get("MODEL", "qwen3.8-flash-next"),
                   "prompt": "The capital of France is",
                   "max_tokens": 12, "temperature": 0}).encode()
req = urllib.request.Request(base + "/v1/completions", data=data, headers=hdr)
print("  ", repr(json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["text"]))
PY

echo ">> prefill (TTFT on a ~8k-token prompt)"
python3 - "$BASE" <<'PY'
import json, os, sys, time, urllib.request
base = sys.argv[1]; prompt = "word " * 8000
hdr = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    hdr["Authorization"] = "Bearer " + os.environ["API_KEY"]
t = time.time()
req = urllib.request.Request(base + "/v1/completions",
    data=json.dumps({"model": os.environ.get("MODEL", "qwen3.8-flash-next"),
                     "prompt": prompt, "max_tokens": 1,
                     "temperature": 0}).encode(), headers=hdr)
u = json.load(urllib.request.urlopen(req, timeout=300))["usage"]; dt = time.time() - t
print(f"   {u['prompt_tokens']} tok in {dt:.2f}s  =>  {u['prompt_tokens']/dt:.0f} tok/s prefill")
PY

echo ">> decode (256 tokens, short prompt)"
python3 - "$BASE" <<'PY'
import json, os, sys, time, urllib.request
base = sys.argv[1]
hdr = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    hdr["Authorization"] = "Bearer " + os.environ["API_KEY"]
t = time.time()
req = urllib.request.Request(base + "/v1/completions",
    data=json.dumps({"model": os.environ.get("MODEL", "qwen3.8-flash-next"),
                     "prompt": "Hello", "max_tokens": 256, "temperature": 0,
                     "ignore_eos": True}).encode(), headers=hdr)
n = json.load(urllib.request.urlopen(req, timeout=300))["usage"]["completion_tokens"]; dt = time.time() - t
print(f"   {n} tok in {dt:.2f}s  =>  {n/dt:.1f} tok/s decode")
PY
