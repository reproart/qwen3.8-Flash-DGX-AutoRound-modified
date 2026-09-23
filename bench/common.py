"""Shared client helpers for bench/perf.py and bench/longctx.py.

Env (same names as the other benches; defaults match serve.sh):
  BASE     server root, default http://localhost:8000 (a trailing /v1 is fine;
           GB10_BASE_URL is accepted too)
  MODEL    served model name; default: whatever /v1/models reports
  API_KEY  optional bearer token (serve.sh's API_KEY; GB10_API_KEY accepted)
  METRICS_PORT  engine-side metrics sidecar port on the same host (serve.sh
           METRICS_PORT, default 18400; 0 = don't read it)

Everything goes through /v1/chat/completions — the path real clients use, so
the chat template and the reasoning parser are part of the measurement.
Token counts always come from the server's `usage`, never from counting SSE
events: with MTP each event can carry several tokens.
"""
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

BASE = (os.environ.get("BASE") or os.environ.get("GB10_BASE_URL")
        or "http://localhost:8000").rstrip("/")
if BASE.endswith("/v1"):
    BASE = BASE[:-3]
API_KEY = os.environ.get("API_KEY") or os.environ.get("GB10_API_KEY") or ""
HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = "Bearer " + API_KEY

# The canonical decode prompt (code: MTP drafts it well, like real coding use).
CODE_PROMPT = (
    "Write a complete Python implementation of an LRUCache class with get and put "
    "in O(1), using a dict and a doubly linked list. Include docstrings."
)

_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
          "november oscar papa quebec romeo sierra tango uniform victor whiskey xray "
          "yankee zulu scheduler kernel latency throughput quantize tensor gradient cache "
          "pointer buffer register pipeline parallel entropy manifold").split()


def _get(path, timeout=30):
    req = urllib.request.Request(BASE + path, headers=HEADERS)
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def _model():
    if os.environ.get("MODEL") or os.environ.get("GB10_MODEL"):
        return os.environ.get("MODEL") or os.environ.get("GB10_MODEL")
    try:
        return json.loads(_get("/v1/models"))["data"][0]["id"]
    except Exception as e:  # noqa: BLE001 - report the actual cause
        raise SystemExit(f"cannot read {BASE}/v1/models ({e}); is the server up, "
                         f"is API_KEY right? Or set MODEL explicitly.") from None


MODEL = _model()


def unique_prompt(seed, approx_tokens, tail="Reply with only: OK"):
    """A prompt nothing else shares — not even its first block.

    The seed is the very first text, so the prefix cache cannot reuse a single
    block across prompts, and the body is pseudo-random words, so the PLE
    n-gram table sees realistic row diversity instead of one repeated
    sentence (repetitive filler makes the mmap gather look cheaper than it is).
    """
    rng = random.Random(seed)
    body = " ".join(rng.choice(_WORDS) for _ in range(int(approx_tokens / 1.3)))
    return f"Document {seed}. Below is a log excerpt.\n\n{body}\n\n{tail}"


def chat(prompt, max_tokens, thinking=False, stream=False, timeout=3600):
    """One chat completion -> dict(e2e, ttft, completion_tokens, prompt_tokens,
    cached_tokens, finish_reason). ttft is None when not streaming."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 json.dumps(body).encode(), HEADERS)
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None

    ttft, usage, finish = None, {}, None
    with resp:
        if not stream:
            d = json.loads(resp.read())
            usage = d.get("usage") or {}
            finish = d["choices"][0].get("finish_reason")
        else:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except ValueError:
                    continue
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if ttft is None and (delta.get("content") or delta.get("reasoning_content")
                                         or delta.get("reasoning")):
                        ttft = time.perf_counter() - t0
                    finish = ch.get("finish_reason") or finish
                if d.get("usage"):
                    usage = d["usage"]
    details = usage.get("prompt_tokens_details") or {}
    return {
        "e2e": time.perf_counter() - t0,
        "ttft": ttft,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        # Only reported with --enable-prompt-tokens-details; the prefix-cache
        # counters from metrics() are the reliable source.
        "cached_tokens": details.get("cached_tokens"),
        "finish_reason": finish,
    }


def decode_rate(r):
    """Generation speed after the first token (tok/s), None if unknown."""
    if r["ttft"] is None or r["e2e"] <= r["ttft"] or r["completion_tokens"] < 2:
        return None
    return (r["completion_tokens"] - 1) / (r["e2e"] - r["ttft"])


_METRICS = {
    "queue_sum": "vllm:request_queue_time_seconds_sum",
    "queue_count": "vllm:request_queue_time_seconds_count",
    "pc_queries": "vllm:prefix_cache_queries_total",
    "pc_hits": "vllm:prefix_cache_hits_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "preempted": "vllm:num_preemptions_total",
}


def metrics():
    """Selected vLLM counters (summed over label sets); {} if /metrics is
    unreachable — the benches then just omit the derived columns."""
    try:
        text = _get("/metrics")
    except Exception:  # noqa: BLE001 - metrics are optional
        return {}
    out = {}
    for key, name in _METRICS.items():
        vals = re.findall(rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)$", text, re.M)
        if vals:
            out[key] = sum(float(v) for v in vals)
    return out


_SIDECAR_PORT = int(os.environ.get("METRICS_PORT", "18400") or 0)
_SIDECAR = (f"http://{urlparse(BASE).hostname}:{_SIDECAR_PORT}/metrics"
            if _SIDECAR_PORT else None)
_PLE = {"ple_ops": "vllm:ple_mmap_ops_total", "ple_op_ms": "vllm:ple_mmap_op_ms_total",
        "ple_gather_ms": "vllm:ple_mmap_gather_ms_total"}


def sidecar():
    """PLE gather counters from the engine-side sidecar; {} when unreachable."""
    if not _SIDECAR:
        return {}
    try:
        text = urllib.request.urlopen(_SIDECAR, timeout=5).read().decode()
    except Exception:  # noqa: BLE001 - optional
        return {}
    out = {}
    for key, name in _PLE.items():
        m = re.search(rf"^{re.escape(name)}\s+([0-9.eE+-]+)$", text, re.M)
        if m:
            out[key] = float(m.group(1))
    return out


def ple_summary(s0, s1):
    """'PLE 3.1 ms/op (gather 1.2)' over a sidecar() pair — per-workload cost,
    unlike the lifetime average that mixes decode steps and prefill chunks.
    op = hash + gather + H2D, and it includes waiting for the GPU to finish
    the preceding layer (the lookup is a sync point)."""
    ops = delta(s0, s1, "ple_ops")
    if not ops:
        return ""
    return (f"PLE {delta(s0, s1, 'ple_op_ms') / ops:.1f} ms/op "
            f"(gather {delta(s0, s1, 'ple_gather_ms') / ops:.1f})")


def delta(m0, m1, key):
    if key in m0 and key in m1:
        return m1[key] - m0[key]
    return None


def spec_summary(m0, m1):
    """'tok/step 2.61, accept 54%' from a metrics() pair, or '' without MTP."""
    d, a, dt = delta(m0, m1, "drafts"), delta(m0, m1, "accepted"), delta(m0, m1, "draft_tokens")
    if not d or a is None or not dt:
        return ""
    return f"MTP {1 + a / d:.2f} tok/step, accept {100 * a / dt:.0f}%"
