"""Shared client helpers for bench/perf.py and bench/longctx.py.

Env (same names as the other benches; defaults match serve.sh):
  BASE     server root, default http://localhost:8000 (a trailing /v1 is fine;
           GB10_BASE_URL is accepted too)
  MODEL    served model name; default: whatever /v1/models reports
  API_KEY  optional bearer token (serve.sh's API_KEY; GB10_API_KEY accepted)
  METRICS_PORT  engine-side metrics sidecar port on the same host (serve.sh
           METRICS_PORT, default 18400; 0 = don't read it)
  VLLM_CUSTOM_METRICS_INTERVAL  the sidecar's refresh period (default 5 s, as
           on the server); settled reads wait one period + 1 s

Everything goes through /v1/chat/completions — the path real clients use, so
the chat template and the reasoning parser are part of the measurement.
Token counts always come from the server's `usage`, never from counting SSE
events: with MTP each event can carry several tokens.
"""
import itertools
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



def _make_vocab(n=30000, seed=1234):
    """A fixed synthetic vocabulary of n pronounceable pseudo-words."""
    rng = random.Random(seed)
    cons, vows = "bcdfghjklmnprstvwz", "aeiou"
    words = set()
    while len(words) < n:
        k = rng.choice((1, 2, 2, 3, 3, 4))
        words.add("".join(rng.choice(cons) + rng.choice(vows) for _ in range(k)))
    return sorted(words)


# Zipf-distributed word draws (weight 1/rank^1.07, like natural language): a
# realistic spread of distinct n-grams for the PLE table. A few dozen fixed
# words — the earlier generator — have so few distinct n-grams that every
# table row they touch fits in a few hundred MB of page cache, hiding exactly
# the disk reads that a smaller page cache (a bigger KV_BYTES) causes.
_VOCAB = _make_vocab()
_ZIPF_CUM = list(itertools.accumulate(1.0 / (r + 1) ** 1.07 for r in range(len(_VOCAB))))
_TOK_PER_WORD = [None]


def _tokens_per_word():
    """Calibrate words -> tokens once with the server's /tokenize (fallback 2.0)."""
    if _TOK_PER_WORD[0] is None:
        sample = " ".join(random.Random(7).choices(_VOCAB, cum_weights=_ZIPF_CUM, k=2000))
        try:
            req = urllib.request.Request(BASE + "/tokenize", json.dumps(
                {"model": MODEL, "prompt": sample}).encode(), HEADERS)
            n = json.loads(urllib.request.urlopen(req, timeout=30).read())["count"]
            _TOK_PER_WORD[0] = max(0.5, n / 2000)
        except Exception:  # noqa: BLE001 - calibration is best-effort
            _TOK_PER_WORD[0] = 2.0
    return _TOK_PER_WORD[0]


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
    block across prompts, and the body is Zipf-drawn words from a 30k-word
    vocabulary, so the PLE n-gram table sees a realistic spread of rows
    (repetitive filler makes the mmap gather look cheaper than it is).
    """
    rng = random.Random(seed)
    words = int(approx_tokens / _tokens_per_word())
    body = " ".join(rng.choices(_VOCAB, cum_weights=_ZIPF_CUM, k=words))
    return f"Document {seed}. Below is a log excerpt.\n\n{body}\n\n{tail}"


def chat(prompt, max_tokens, thinking=False, stream=False, timeout=3600, ignore_eos=False):
    """One chat completion -> dict(e2e, ttft, completion_tokens, prompt_tokens,
    cached_tokens, finish_reason). ttft is None when not streaming.
    ignore_eos=True (vLLM extension) forces exactly max_tokens tokens."""
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
    if ignore_eos:
        body["ignore_eos"] = True
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
    # gauges (current value, not counters)
    "kv_usage": "vllm:kv_cache_usage_perc",
    "kv_usage_old": "vllm:gpu_cache_usage_perc",  # name in older vLLM builds
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
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


_SIDECAR_LAG = float(os.environ.get("VLLM_CUSTOM_METRICS_INTERVAL", "5") or 5) + 1.0
_SIDECAR_SEEN = [False]


def sidecar(settle=False):
    """PLE gather counters from the engine-side sidecar; {} when unreachable.

    The sidecar copies the engine's counters into Prometheus only every few
    seconds, so a read right after a short request can miss its ops (they
    then land in the NEXT measurement). settle=True waits one refresh period
    first — use it for every read that closes (or opens, after other
    traffic) a measured interval."""
    if not _SIDECAR:
        return {}
    if settle and _SIDECAR_SEEN[0]:
        time.sleep(_SIDECAR_LAG)
    try:
        text = urllib.request.urlopen(_SIDECAR, timeout=5).read().decode()
    except Exception:  # noqa: BLE001 - optional
        return {}
    out = {}
    for key, name in _PLE.items():
        m = re.search(rf"^{re.escape(name)}\s+([0-9.eE+-]+)$", text, re.M)
        if m:
            out[key] = float(m.group(1))
    _SIDECAR_SEEN[0] = _SIDECAR_SEEN[0] or bool(out)
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
