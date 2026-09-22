#!/usr/bin/env python3
"""Concurrency bench for the qwen38-flash stack: N simultaneous streams,
per-stream decode tok/s + TTFT, aggregate throughput — and, critically, the
vllm:request_queue_time delta from /metrics, so an aggregate number is never
mistaken for model speed when it is actually the admission queue
(--max-num-seqs / SEQS too small; the README warns about exactly this).
Also reports MTP acceptance per level.

Usage:
  python3 bench/concurrency_bench.py                          # levels 1 2 4 8
  python3 bench/concurrency_bench.py --levels 8 16 --max-tokens 128 --runs 2
  python3 bench/concurrency_bench.py --json report.json       # also dump raw

Env: BASE (default http://localhost:8000), MODEL (default qwen), API_KEY
(optional bearer token, matches serve.sh's API_KEY).

Workload: unique ~65-token prompts — no prefix-cache overlap between requests
or levels, so every wave is cold — with forced greedy generation
(ignore_eos). Compare per-stream tok/s across levels: how it degrades is the
real concurrency story of the box.
"""
from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import os
import re
import statistics
import time
import urllib.request
from urllib.parse import urlparse

BASE = os.environ.get("BASE", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "qwen")
API_KEY = os.environ.get("API_KEY", "")

TOPICS = [
    "the history of computing", "how hurricanes form", "the sociology of cities",
    "deep-sea exploration", "the economics of coffee", "orbital mechanics",
    "the printing press", "photosynthesis", "roman aqueducts", "machine translation",
    "the invention of bicycles", "tectonic plates", "jazz harmony", "vaccine development",
    "arctic exploration", "the silk road", "combustion engines", "optical fibers",
    "language families", "the water cycle",
]

_METRIC_KEYS = (
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_count",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = "Bearer " + API_KEY
    return h


def metrics() -> dict:
    """The handful of vLLM counters this bench cares about (missing -> None)."""
    req = urllib.request.Request(BASE + "/metrics", headers=_headers())
    out = urllib.request.urlopen(req, timeout=30).read().decode()
    vals = {}
    for key in _METRIC_KEYS:
        m = re.search(rf"^{re.escape(key)}(?:{{[^}}]*}})?\s+([0-9.e+-]+)$", out, re.M)
        vals[key] = float(m.group(1)) if m else None
    return vals


def one_stream(lvl: int, run: int, idx: int, max_tokens: int) -> tuple:
    """Blocking SSE POST /v1/completions -> (ttft_s, total_s, completion_tokens)."""
    u = urlparse(BASE)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=3600)
    prompt = (f"[cb-l{lvl}-r{run}-q{idx}-{time.time():.0f}] Write a very long, "
              f"detailed essay about {TOPICS[idx % len(TOPICS)]}. ")
    payload = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
    })
    t0 = time.perf_counter()
    try:
        conn.request("POST", "/v1/completions", body=payload, headers=_headers())
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:200]!r}")
        ttft, chunks, usage_tok = None, 0, None
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                ev = json.loads(body)
            except json.JSONDecodeError:
                continue
            if ev.get("usage") and ev["usage"].get("completion_tokens") is not None:
                usage_tok = ev["usage"]["completion_tokens"]
                continue
            if ev.get("choices") and ev["choices"][0].get("text"):
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
        total = time.perf_counter() - t0
    finally:
        conn.close()
    return ttft, total, usage_tok if usage_tok is not None else chunks


async def run_level(level: int, run: int, max_tokens: int, pool) -> tuple:
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    results = await asyncio.gather(*[
        loop.run_in_executor(pool, one_stream, level, run, i, max_tokens)
        for i in range(level)
    ])
    return time.perf_counter() - t0, results


def _fmt(vals, unit=""):
    if not vals:
        return "n/a"
    return f"{statistics.mean(vals):.2f}{unit} (min {min(vals):.2f}, max {max(vals):.2f})"


def report(level: int, run: int, wall: float, results: list, m0: dict, m1: dict) -> dict:
    n_req = len(results)
    toks = [c for _, _, c in results]
    ttfts = [t for t, _, _ in results if t is not None]
    rates = [c / (d - t) for (t, d, c) in results if t is not None and d > t]
    total_tok = sum(toks)

    q0, q1 = (m0.get("vllm:request_queue_time_seconds_sum"),
              m1.get("vllm:request_queue_time_seconds_sum"))
    queue_per_req = ((q1 - q0) / n_req) if q0 is not None and q1 is not None else None

    d0, d1 = m0.get("vllm:spec_decode_num_drafts_total"), m1.get("vllm:spec_decode_num_drafts_total")
    a0, a1 = (m0.get("vllm:spec_decode_num_accepted_tokens_total"),
              m1.get("vllm:spec_decode_num_accepted_tokens_total"))
    dt0, dt1 = (m0.get("vllm:spec_decode_num_draft_tokens_total"),
                m1.get("vllm:spec_decode_num_draft_tokens_total"))
    tok_step = ((1 + (a1 - a0) / (d1 - d0))
                if None not in (d0, d1, a0, a1) and d1 > d0 else None)
    accept = (((a1 - a0) / (dt1 - dt0) * 100)
              if None not in (a0, a1, dt0, dt1) and dt1 > dt0 else None)

    print(f"=== concurrency {level} (run {run}) ===")
    print(f"  wall {wall:6.1f}s | req {n_req} | tokens {total_tok} | "
          f"aggregate {total_tok / wall:6.1f} tok/s")
    print(f"  per-stream: {_fmt(rates)} tok/s | ttft mean/max "
          f"{(statistics.mean(ttfts) if ttfts else float('nan')):.2f}s/"
          f"{(max(ttfts) if ttfts else float('nan')):.2f}s")
    if queue_per_req is not None:
        flag = ""
        if queue_per_req > 0.5:
            flag = "   <-- the queue dominates: aggregate measures admission, not the model"
        print(f"  queue: {queue_per_req:.2f} s/req{flag}")
    else:
        print("  queue: n/a (no request_queue_time metric)")
    if tok_step is not None and accept is not None:
        print(f"  mtp: {tok_step:.2f} tok/step, accept {accept:.0f}%")
    print()

    return {"level": level, "run": run, "wall": wall, "requests": n_req,
            "tokens": total_tok, "aggregate_tok_s": total_tok / wall,
            "per_stream_tok_s": rates, "ttft_s": ttfts,
            "queue_s_per_req": queue_per_req,
            "mtp_tok_step": tok_step, "mtp_accept_pct": accept}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 4, 8],
                    help="concurrency levels to sweep (default: 1 2 4 8)")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="generated tokens per request (default: 256)")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeats per level (default: 1)")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="also dump the raw per-run report as JSON")
    args = ap.parse_args()

    print(f"endpoint {BASE}  model {MODEL}  levels {args.levels}  "
          f"max_tokens {args.max_tokens}  runs {args.runs}\n")
    reports = []
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.levels)) as pool:
        for level in args.levels:
            for run in range(1, args.runs + 1):
                m0 = metrics()
                wall, results = await run_level(level, run, args.max_tokens, pool)
                m1 = metrics()
                reports.append(report(level, run, wall, results, m0, m1))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"raw report: {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
