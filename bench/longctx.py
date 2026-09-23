#!/usr/bin/env python3
"""Concurrent long-context test: N streams, each with its own ~CTX-token prompt.

    python3 bench/longctx.py
    python3 bench/longctx.py --ctx 60000 --streams 1 2 4 8

Env: BASE / MODEL / API_KEY, see bench/common.py.

Every prompt is unique from its first token (and made of varied words), so
the prefix cache cannot deduplicate them: you measure prefill cost and KV
capacity, not cache hits. The prefix-cache hit counter is checked per row
and a hit is reported loudly instead of being assumed away.

What to expect on this stack: vLLM chunks prefill (8192 tokens per step,
shared across requests), so N concurrent long prompts roughly cost N times
one — TTFT grows linearly, s/stream stays flat. Two limits bend that line:
  * --max-num-seqs (serve.sh SEQS, default 8): more streams queue ("queue").
  * the KV pool (serve.sh KV_BYTES, default 20g: roughly 650-700k tokens at
    ~30 KB/token): when N x ctx exceeds it, vLLM preempts running requests and
    recomputes them later ("preempt" > 0) — wall time jumps.
"""
import argparse
import statistics
import threading
import time
import uuid

import common
from common import chat, delta, metrics, unique_prompt


def run_level(n, ctx):
    out, errors = [None] * n, []
    tag = uuid.uuid4().hex[:8]  # fresh per row: no cross-row reuse either

    def worker(i):
        try:
            out[i] = chat(unique_prompt(f"{tag}-{i}", ctx), 24, stream=True)
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
    return [r for r in out if r], errors, wall, m0, metrics()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctx", type=int, default=80000,
                    help="approx. prompt tokens per stream (default 80000)")
    ap.add_argument("--streams", nargs="+", type=int, default=[1, 2, 4, 8],
                    help="concurrency levels (default: 1 2 4 8)")
    args = ap.parse_args()

    print(f"endpoint {common.BASE}  model {common.MODEL}  ctx ~{args.ctx} tok/stream")
    print("warmup...", flush=True)
    chat(unique_prompt(uuid.uuid4().hex, 400), 16)
    print("ok\n")
    print(f"{'streams':>7} {'ctx each':>9} {'total ctx':>10} {'cached':>7} {'wall':>8} "
          f"{'TTFT p50':>9} {'TTFT max':>9} {'s/stream':>9} {'queue':>7} {'preempt':>8}")
    print("-" * 92)

    for n in args.streams:
        done, errors, wall, m0, m1 = run_level(n, args.ctx)
        if errors:
            print(f"{n:>7}  {len(errors)} request(s) failed, e.g. {errors[0][:120]}"
                  " — check `docker logs` (OOM / restart?)")
            if not done:
                continue
        ttfts = [r["ttft"] for r in done if r["ttft"] is not None]
        total = sum(r["prompt_tokens"] for r in done)
        hits = delta(m0, m1, "pc_hits")
        q_sum, q_cnt = delta(m0, m1, "queue_sum"), delta(m0, m1, "queue_count")
        queue = q_sum / q_cnt if q_sum is not None and q_cnt else None
        pre = delta(m0, m1, "preempted")

        def f(x, spec):
            return "n/a" if x is None else f"{x:{spec}}"

        print(f"{n:>7} {done[0]['prompt_tokens']:>9} {total:>10} {f(hits, '>7.0f')} {wall:>7.1f}s "
              f"{f(statistics.median(ttfts) if ttfts else None, '>8.1f')}s "
              f"{f(max(ttfts) if ttfts else None, '>8.1f')}s {wall / n:>8.1f}s "
              f"{f(queue, '>6.1f')}s {f(pre, '>8.0f')}")
        if hits:
            print(f"        !! {hits:.0f} prefix-cache hit tokens on unique prompts — "
                  "the numbers above are optimistic")
        time.sleep(4)


if __name__ == "__main__":
    main()
