#!/usr/bin/env bash
# Quick check that the server is up, coherent, and measure prefill + decode.
#   scripts/smoke-test.sh [host:port]
# Defaults match serve.sh (localhost:8000). BASE (full URL, e.g.
# http://spark:8000) may be given instead of the host:port argument; MODEL
# defaults to whatever the server reports in /v1/models; API_KEY is the
# optional bearer token, matching serve.sh's API_KEY:
#   scripts/smoke-test.sh                       # serve.sh defaults
#   API_KEY=... scripts/smoke-test.sh spark:18300
set -euo pipefail
if [ -n "${1:-}" ]; then
  BASE="http://$1"
else
  BASE="${BASE:-http://localhost:8000}"
fi
BASE="${BASE%/}"
API_KEY="${API_KEY:-}"
AUTH=()
[ -n "$API_KEY" ] && AUTH=(-H "Authorization: Bearer $API_KEY")

echo ">> health"   # /health stays unauthenticated even with --api-key
if curl -sf -m 5 "$BASE/health" >/dev/null; then echo "   OK"; else echo "   not ready ($BASE)"; exit 1; fi

if [ -z "${MODEL:-}" ]; then
  MODEL=$(curl -sf -m 10 ${AUTH[@]+"${AUTH[@]}"} "$BASE/v1/models" \
          | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null) \
    || { echo "ERROR: cannot read /v1/models (wrong API_KEY?) — set MODEL explicitly" >&2; exit 1; }
fi
echo ">> model: $MODEL"
export MODEL API_KEY   # read by the python snippets below

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
