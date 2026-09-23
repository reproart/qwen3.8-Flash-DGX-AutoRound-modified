#!/usr/bin/env python3
"""Concurrent long-context test: N streams, each with its own ~CTX-token prompt.

    python3 bench/longctx.py
    python3 bench/longctx.py --ctx 60000 --streams 1 2 4 8

Env: BASE / MODEL / API_KEY, see bench/common.py.

Every prompt is unique from its first token (and made of varied words), so
the prefix cache cannot deduplicate them: you measure prefill cost and KV
capacity, not cache hits. The prefix-cache hit counter is checked per row
and a hit is reported loudly instead of being assumed away.

What to expect on this stack: vLLM chunks prefill (8192 tokens per step),
and one long prefill fills that budget, so concurrent long prompts are
prefilled one after another — TTFT grows linearly, s/stream stays flat, and
the waiting shows up as "queue" (not scheduled yet) even below
--max-num-seqs (serve.sh SEQS, default 8, the other reason to queue).

KV capacity: by default each request answers "OK" right after its prefill
and frees its KV before the next one is admitted, so total ctx can exceed the
pool without preemption — that tests prefill, not capacity. --gen N forces
exactly N generated tokens per stream (ignore_eos), so earlier streams stay
resident while later ones prefill: a real capacity test. N ~ 1000 is enough —
while another stream prefills, each engine step carries an 8192-token chunk
(~4 s), so a resident stream only decodes a few hundred tokens during the
others' prefill. The "gen" column shows what was actually generated. Limits:
  * the KV pool (serve.sh KV_BYTES; ~31 KB/token, so 30g ~ 966k tokens —
    the boot log prints "GPU KV cache size"): when N x ctx exceeds it, vLLM preempts running requests and
    recomputes them later ("preempt" > 0) — wall time jumps.
"""
import argparse
import statistics
import threading
import time
import uuid

import common
from common import chat, delta, metrics, unique_prompt


def _poll(stop, peak):
    """Sample KV-pool usage and running requests every 2 s until stop is set."""
    while not stop.wait(2.0):
        m = metrics()
        u = m.get("kv_usage", m.get("kv_usage_old"))
        if u is not None:
            peak["kv"] = max(peak.get("kv", 0.0), u)
        if m.get("running") is not None:
            peak["running"] = max(peak.get("running", 0), int(m["running"]))


def run_level(n, ctx, gen):
    forced = gen is not None
    out, errors = [None] * n, []
    peak, stop = {}, threading.Event()
    poller = threading.Thread(target=_poll, args=(stop, peak), daemon=True)
    tag = uuid.uuid4().hex[:8]  # fresh per row: no cross-row reuse either

    def worker(i):
        try:
            out[i] = chat(unique_prompt(f"{tag}-{i}", ctx), gen if forced else 24,
                          stream=True, ignore_eos=forced)
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append(repr(e))

    m0 = metrics()
    t0 = time.perf_counter()
    poller.start()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    stop.set()
    poller.join()
    return [r for r in out if r], errors, wall, m0, metrics(), peak


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctx", type=int, default=80000,
                    help="approx. prompt tokens per stream (default 80000)")
    ap.add_argument("--gen", type=int, default=None,
                    help="force exactly N generated tokens per stream (ignore_eos) to keep "
                         "KV resident — a capacity test; default: a short 'OK' answer")
    ap.add_argument("--streams", nargs="+", type=int, default=[1, 2, 4, 8],
                    help="concurrency levels (default: 1 2 4 8)")
    args = ap.parse_args()

    print(f"endpoint {common.BASE}  model {common.MODEL}  ctx ~{args.ctx} tok/stream")
    print("warmup...", flush=True)
    chat(unique_prompt(uuid.uuid4().hex, 400), 16)
    print("ok\n")
    print(f"{'streams':>7} {'ctx each':>9} {'total ctx':>10} {'gen':>6} {'cached':>7} {'wall':>8} "
          f"{'TTFT p50':>9} {'TTFT max':>9} {'s/stream':>9} {'queue':>7} {'preempt':>8}")
    print("-" * 99)

    for n in args.streams:
        done, errors, wall, m0, m1, peak = run_level(n, args.ctx, args.gen)
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

        gen_avg = statistics.mean(r["completion_tokens"] for r in done)
        print(f"{n:>7} {done[0]['prompt_tokens']:>9} {total:>10} {gen_avg:>6.0f} {f(hits, '>7.0f')} {wall:>7.1f}s "
              f"{f(statistics.median(ttfts) if ttfts else None, '>8.1f')}s "
              f"{f(max(ttfts) if ttfts else None, '>8.1f')}s {wall / n:>8.1f}s "
              f"{f(queue, '>6.1f')}s {f(pre, '>8.0f')}")
        if "kv" in peak:
            # Only meaningful when every stream was resident at the same time
            # (--gen): then usage = their tokens / what the pool really holds.
            tokens = total + sum(r["completion_tokens"] for r in done)
            line = (f"        peak KV pool usage {100 * peak['kv']:.1f}%, "
                    f"up to {peak.get('running', '?')} requests running")
            if peak.get("running") == n and peak["kv"] > 0.05:
                line += (f" -> ~{tokens / 1e3:.0f}k tokens resident, so the pool holds "
                         f"~{tokens / peak['kv'] / 1e6:.2f}M such tokens")
            print(line)
        if hits:
            print(f"        !! {hits:.0f} prefix-cache hit tokens on unique prompts — "
                  "the numbers above are optimistic")
        time.sleep(4)


if __name__ == "__main__":
    main()
