#!/usr/bin/env python3
"""One-command performance overview over the chat API: TTFT, single-stream
decode, a concurrency sweep and long-context prefill.

    python3 bench/perf.py
    BASE=http://spark:8000 API_KEY=... python3 bench/perf.py --levels 1 4 8 16

Env: BASE / MODEL / API_KEY, see bench/common.py. Run it against an otherwise
idle server — competing traffic skews every number here.

What the columns mean (and the traps they avoid):
  * decode tok/s = tokens after the first one / time after the first one, so
    TTFT and queueing do not leak into the generation speed; `e2e tok/s`
    (tokens / whole request) is shown next to it for comparison.
  * queue s/req comes from vLLM's request_queue_time counter. A level above
    the server's --max-num-seqs (serve.sh SEQS, default 8) queues requests in
    waves: TTFT and aggregate then measure admission, not the model — such
    rows are flagged.
  * prefill prompts are unique from their very first token and made of varied
    words, so the prefix cache cannot serve part of them and the PLE table
    sees realistic row diversity; the prefix-cache hit counter is checked and
    the rate is computed over uncached tokens only.
"""
import argparse
import statistics
import sys
import threading
import time
import uuid

import common
from common import CODE_PROMPT, chat, decode_rate, delta, metrics, spec_summary, unique_prompt


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _fmt(x, spec, unit=""):
    return "n/a" if x is None else f"{x:{spec}}{unit}"


def ttft_probe(n=5):
    print("## 1. Time to first token (streaming, short prompt)")
    vals = [chat("Say hello.", 16, stream=True)["ttft"] for _ in range(n)]
    vals = [v for v in vals if v is not None]
    if not vals:
        print("   no content in any stream — cannot measure TTFT\n")
        return
    print(f"   median {statistics.median(vals) * 1000:.0f} ms   "
          f"min {min(vals) * 1000:.0f}   max {max(vals) * 1000:.0f}\n")


def single_stream(n=5, max_tokens=700):
    print(f"## 2. Single-stream decode (code, thinking off, up to {max_tokens} tokens)")
    rows = []
    for _ in range(n):
        m0 = metrics()
        r = chat(CODE_PROMPT, max_tokens, stream=True)
        rows.append((r, spec_summary(m0, metrics())))
    for r, spec in rows:
        print(f"   {r['completion_tokens']:>4} tok  decode {_fmt(decode_rate(r), '.1f')} tok/s  "
              f"e2e {r['completion_tokens'] / r['e2e']:.1f} tok/s  ttft {_fmt(r['ttft'], '.2f', 's')}"
              f"{'  ' + spec if spec else ''}")
    short = [r for r, _ in rows if r["completion_tokens"] < max_tokens * 0.5]
    if short:
        print(f"   note: {len(short)} run(s) stopped early (<50% of max_tokens) — "
              "short runs overweight the fixed per-request cost")
    print(f"   MEDIAN decode {_fmt(_med(decode_rate(r) for r, _ in rows), '.1f')} tok/s   "
          f"e2e {statistics.median(r['completion_tokens'] / r['e2e'] for r, _ in rows):.1f} tok/s\n")


def concurrency(levels, tokens=300):
    print(f"## 3. Concurrency sweep ({tokens} tokens per stream)")
    print(f"   {'streams':>7} {'wall(s)':>8} {'aggregate':>11} {'decode/stream':>14} "
          f"{'TTFT p50':>9} {'TTFT max':>9} {'queue':>9}")
    print("   " + "-" * 74)
    peak = (0, 0.0)
    for n in levels:
        out, errors = [None] * n, []

        def worker(i):
            try:
                # uuid first: unique from the first token, no prefix reuse
                out[i] = chat(f"[{uuid.uuid4().hex[:8]}] {CODE_PROMPT} Variant {i}.",
                              tokens, stream=True)
            except Exception as e:  # noqa: BLE001 - reported below
                errors.append(repr(e))

        m0 = metrics()
        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        m1 = metrics()

        done = [r for r in out if r]
        if errors:
            print(f"   {n:>7}  {len(errors)} request(s) failed, e.g. {errors[0][:120]}")
            if not done:
                continue
        agg = sum(r["completion_tokens"] for r in done) / wall
        ttfts = [r["ttft"] for r in done if r["ttft"] is not None]
        q_sum, q_cnt = delta(m0, m1, "queue_sum"), delta(m0, m1, "queue_count")
        queue = q_sum / q_cnt if q_sum is not None and q_cnt else None
        flag = "  <- queued: above --max-num-seqs" if queue is not None and queue > 0.5 else ""
        if agg > peak[1] and not flag:
            peak = (n, agg)
        print(f"   {n:>7} {wall:>8.1f} {agg:>7.1f} t/s {_fmt(_med(decode_rate(r) for r in done), '>9.1f')} t/s "
              f"{_fmt(_med(ttfts), '>8.2f', 's')} {_fmt(max(ttfts) if ttfts else None, '>8.2f', 's')} "
              f"{_fmt(queue, '>8.2f', 's')}{flag}")
        time.sleep(3)
    if peak[0]:
        print(f"\n   peak aggregate without queueing: {peak[1]:.1f} tok/s at {peak[0]} streams\n")
    else:
        print()


def prefill(targets):
    print("## 4. Long-context prefill (unique prompts, prefix cache checked)")
    for target in targets:
        m0 = metrics()
        r = chat(unique_prompt(uuid.uuid4().hex, target), 8, stream=True)
        hits = delta(m0, metrics(), "pc_hits")
        fresh = r["prompt_tokens"] - (hits or 0)
        rate = fresh / r["ttft"] if r["ttft"] else None
        cached = "n/a" if hits is None else f"{hits:.0f}"
        print(f"   prompt {r['prompt_tokens']:>7} tok (cached {cached:>5}) -> "
              f"TTFT {_fmt(r['ttft'], '>6.2f', 's')}   prefill ~{_fmt(rate, '>6.0f')} tok/s")
        if hits:
            print("   !! prefix-cache hit on a unique prompt — the rate above excludes it")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 4, 8, 16],
                    help="concurrency levels (default: 1 2 4 8 16)")
    ap.add_argument("--prefill", nargs="+", type=int, default=[2000, 8000, 32000, 100000],
                    help="approx. prompt sizes for the prefill section")
    ap.add_argument("--only", choices=["ttft", "decode", "concurrency", "prefill"],
                    help="run a single section")
    args = ap.parse_args()

    print(f"endpoint {common.BASE}  model {common.MODEL}")
    if not metrics():
        print("note: /metrics unreachable — queue / MTP / prefix-cache columns show n/a")
    print("warmup...", flush=True)
    try:
        chat("hi", 8)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"warmup request failed: {e}")
    print("ok\n")
    sections = {
        "ttft": ttft_probe,
        "decode": single_stream,
        "concurrency": lambda: concurrency(args.levels),
        "prefill": lambda: prefill(args.prefill),
    }
    for name, fn in sections.items():
        if args.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
