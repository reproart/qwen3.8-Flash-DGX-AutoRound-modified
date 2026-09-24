# Qwen3.8-Flash-Next on a single DGX Spark (GB10) — int4 + int8 + fp8 hybrid

Run **Qwen3.8-Flash-Next** — a ~176B-parameter model (125B main + 51B n-gram, 6B
active) — on **one NVIDIA DGX Spark / ASUS GX10** with **vLLM**: **~50–60 tok/s
single-stream decode with MTP=3** (~2,100 tok/s prefill), working **prefix
caching**, and a **never-evict pin** that keeps your system prompt's KV resident
through arbitrary traffic.

Forked from **[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)**,
which established the foundation this fork stands on: the ~49 GiB fp8 n-gram ("PLE")
table is a pure lookup that a token only touches 16 rows of, so it is served
**from NVMe via `mmap`** instead of living in the 128 GB unified pool
(full story: [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md)). Upstream serves the
official **NVFP4** checkpoint at 25–28 tok/s; for that path and its tuning
guide, use upstream. This fork replaces the checkpoint with
**[Intel's W4A16 AutoRound int4](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)**
plus an int8 GPTQ lm_head and blockwise-fp8 side layers (all prepared by
CPU-only tools in `tools/`), an **fp8** PLE table, and a set of GB10/vLLM
patches — roughly **1.8× faster decode** than the NVFP4 recipe on the same box.

> **This repository** is a modified copy of
> [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)
> (all of the above and every patch below). On top of it: an engine-side
> Prometheus sidecar (PLE gather cost, mamba guard, pin gauges), API-key auth,
> `BIND_ADDR`, a container healthcheck and log rotation, a resumable
> `prepare.sh`, chat-API benchmarks (`bench/perf.py`, `bench/longctx.py`) and CI.

> **Upstream's NVFP4 recipe independently reproduced** on a DGX Spark by
> [@jschmied](https://github.com/jschmied) — see
> [blazux#1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1) and their
> [write-up](https://github.com/jschmied/qwen38-flash-next-gb10), which also
> contributed the concurrency findings below.

| | **this fork (int4/int8/fp8)** |
|---|---|
| Prefill | **~2,100–2,200 tok/s** |
| Decode, single stream |**~50–60 tok/s (MTP=3)** |
| Prefix caching | **on** (+ never-evict pin) |
| Context | 262k native / 500k YaRN |
| Weights resident | **~68 GiB** |

## Throughput and concurrency

Measured on this stack with the default `serve.sh` config (DGX Spark, default GPU
clocks; int4 MTP draft experts, `DRAFT_VOCAB=1`, MTP=3, prefix caching on, `SEQS=16`,
8192-token prefill chunks). One caveat: on this box the PLE table is served from
another machine's RAM over RDMA (the `magi` branch), the fastest row source there is —
with the table on a regular local NVMe, as in the Quickstart, expect decode about
2–3% slower than the numbers below. Single-stream decode by workload, **thinking enabled** —
reproduce with
[bench_qwen35.sh](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4/blob/master/bench_qwen35.sh)
(from albond's 122B recipe) pointed at your endpoint, on a freshly started container:

| Task | Prompt Tokens | Gen Tokens | Time (s) | Speed (tok/s) |
| --- | --- | --- | --- | --- |
| **[Q&A]** | 65 | 57.5 ± 5.2 | 1.18 ± 0.17 | 49.1 ± 2.6 |
| **[Code]** | 72 | 275.8 ± 2.2 | 4.89 ± 0.24 | 56.5 ± 3.1 |
| **[JSON]** | 90 | 841.8 ± 20.8 | 13.54 ± 0.46 | 62.1 ± 0.7 |
| **[Math]** | 71 | 64.0 ± 0.0 | 1.15 ± 0.01 | 55.5 ± 0.2 |
| **[LongCode]** | 79 | 2048.0 ± 0.0 | 42.33 ± 2.53 | 48.5 ± 2.8 |

*Note: arithmetic mean ± sample standard deviation over 4 runs (2 scripts × 2 runs, 2026-09-09). Draft acceptance with thinking on is ~70% (2.1 of 3); with `enable_thinking: false` it is ~87% and decode runs 5–10% faster than the table.*


### Measured here with `bench/perf.py` (local NVMe table)

DGX Spark / GX10, this repo's `serve.sh` (MTP=3, `SEQS=8`, `KV_BYTES=30g`,
`DRAFT_VOCAB=1`, table on local NVMe), second run after a restart (see
*Warmup* below), code prompt with thinking off, 2026-09-23:

| | |
|---|---|
| TTFT, short prompt | 129 ms |
| Single-stream decode | **65.8 tok/s** (MTP 3.5 tok/step, 84% of drafts accepted) |
| Aggregate, 8 streams × 300 tokens | **259 tok/s** (35 tok/s per stream, TTFT 0.47 s) |
| 16 streams | requests queue above `SEQS=8` (4.7 s/req): aggregate measures admission |
| Prefill, 2k / 8k / 33k / 105k unique tokens | 2,103 / 2,302 / 2,244 / 2,115 tok/s |

Prefill prompts are unique from their first token: shared-prefix filler lets
the prefix cache serve part of every longer prompt and inflated this same box
to "2,860 tok/s at 120k". Run `python3 bench/perf.py` and
`python3 bench/longctx.py` (N concurrent long prompts; flags queueing and KV
preemption) after any change.

Under concurrency, TTFT grows with MTP's prefill cost (see the next
section), not with the paged table. On upstream's NVFP4 path
[@jschmied](https://github.com/jschmied) measured aggregate throughput
scaling to ~267 tok/s at 48 streams
([load-and-waits.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/load-and-waits.md));
two portable takeaways: per-token page-fault cost *falls* with concurrency
(batched tokens share n-gram rows), and a low `--max-num-seqs` silently
queues requests — check `vllm:request_queue_time_seconds_sum` before quoting
an aggregate number.

## Requirements

- An **NVIDIA DGX Spark or compatible GB10 (sm_121)** box, 128 GB unified memory,
  aarch64, recent NVIDIA driver, Docker with the NVIDIA container runtime.
- **~130 GB free disk** for the checkpoint + fp8 PLE table, on reasonably fast
  storage (the table is read at runtime — NVMe strongly recommended).
- The base image is multi-arch, so `docker build` also works on x86 Blackwell
  (sm_120) for testing, though this is tuned for the Spark.

## Quickstart

```bash
git clone https://github.com/reproart/qwen3.8-Flash-DGX-AutoRound-modified.git
cd qwen3.8-Flash-DGX-AutoRound-modified

docker build -t qwen38-flash-dgx .   # official image + this fork's patches

# The prepared checkpoint + PLE table (one-time, ~116 GiB):
hf download Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN --local-dir /models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN
hf download Saren/Qwen3.8-Flash-Next-ple-table-fp8 --local-dir /models/ple-table-fp8
# option: the same checkpoint with the MTP draft experts left in bf16 (+3.5 GiB, ~3% slower decode):
#   hf download Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid --local-dir /models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid
# (or build them yourself from Intel's release: ./prepare.sh — see below)

# Point serve.sh at your checkpoint + table dirs, then:
./serve.sh                           # boots on :8000 (~5 min with fastsafetensors)
docker logs -f qwen38-flash          # wait for "Application startup complete"
python3 bench/perf.py --only warmup  # optional: compile first-use kernels now (see Warmup)
```

Then hit the OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen",
  "messages": [{"role":"user","content":"Write a haiku about a desktop supercomputer."}],
  "max_tokens": 512
}'
```

`serve.sh` is a thin example config over `scripts/serve-intel-ar.sh` — every
knob is an env var. The defaults below are `serve.sh`'s (the recommended
entry point); where the bare `scripts/serve-intel-ar.sh` falls back to
something else, the note says so. Keep your machine's real settings as an
edited copy or a local-only commit on top.

## Modify the weights yourself

The quickstart's two `hf download` repos are the finished artifacts — hashes
verified against the local originals. If you'd rather build (or audit) them
yourself from
[Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound),
one script runs the whole pipeline (CPU-only — a NAS box is fine):

```bash
./prepare.sh /models/Qwen3.8-Flash-Next-W4A16-AutoRound /models/ple-table-fp8
```

Each step is explained in `prepare.sh`'s header comments: int8 lm_head repack,
fp8 side-layer conversion, n-gram index strip, fp8 table fetch, the
`quantization_config` rewrite, and the int4 MTP-draft-experts variant. On that last one: this vLLM build has no
auto-round loader, but its GPTQ config (`AutoGPTQConfig` → Marlin kernels)
reads the same packed tensors — the GPTQModel-style `dynamic` rules exclude
the families that are not int4-packed and flip the head to 8-bit. The original
AutoRound config is kept as `config.json.autoround`.

The script is resumable: if a step fails (network, disk), fix the cause and
run the same command again — finished steps are skipped, and no tool ever
re-quantizes its own output or overwrites a backup (`tools/test_tools_cpu.py`
checks exactly that). The download also skips shards that hold only the
n-gram table (dropped from the index anyway) when the `huggingface_hub`
Python package is importable.

## Serving

```bash
docker build -t qwen38-flash-dgx .
MODEL_DIR=/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN \
TABLE_DIR=/models/ple-table-fp8 \
PREFIX_CACHE=1 PIN_PROMPT="You are HomeBot, the household assistant." \
scripts/serve-intel-ar.sh
```

or edit the paths in `serve.sh` (the example config used above) and run it.

| Var | `serve.sh` default | Notes |
|---|---|---|
| `MODEL_DIR` / `TABLE_DIR` | `/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN` / `/models/ple-table-fp8` — edit to your paths | Prepared checkpoint / fp8 PLE table dirs (must exist; checked before the running container is touched) |
| `PORT` | `8000` | API port (bare script: `18300`) |
| `CTX` | `262144` | Max context |
| `YARN` | `0` | `1` = Qwen's YaRN rope scaling (factor 4) past the native 262144 — set `CTX` too (500k was upstream's validated ceiling). Also forces the MTP draft's `max_model_len`, which `--hf-overrides` alone doesn't reach |
| `SEQS` | `8` | Max concurrent sequences (don't benchmark with 1–2, see below) |
| `GPU_MEM` | `0.01` | Near-zero pool fraction, paired with `KV_BYTES`: deterministic sizing, so the driver never oversubscribes the unified pool (`NV_ERR_NO_MEMORY` / Xid 31 freezes). Bare script: a `0.85` fraction — avoid on unified-memory boxes. |
| `KV_BYTES` | `30g` | Explicit KV pool size, passed as `--kv-cache-memory-bytes` (bare script: unset). vLLM reads `g` as 10⁹ bytes: `30g` = 27.9 GiB, measured to hold ~4 concurrent 127k contexts. When to raise it to `35g`/`40g`: see [Sizing the KV pool](#sizing-the-kv-pool-30g-35g-or-40g) |
| `MTP` | `3` | Speculative tokens from the MTP head (`0` = off; bare script: `2`) |
| `PREFIX_CACHE` | `1` | Prefix caching — fixed and recommended on this fork (bare script: `0`) |
| `DET_TOPK` | `1` | Deterministic QSA top-k **kernel** (patch 9; @jschmied, vllm#55122): identical output at T=0 at full prefill speed. `0` = stock kernel (non-deterministic, may drop attention candidates) |
| `EXACT_TOPK` | `0` | `1` = exact `torch.topk` fallback (patch 9; deterministic, −20–40% on long prefill). Wins over `DET_TOPK` when set |
| `DRAFT_VOCAB` | `1` | The MTP drafter scores only the 65,536 most frequent tokens (patch 10; from upstream blazux): +3–5% decode on English/code, draft acceptance unchanged there (thinking on or off). The shipped id set is English/code-weighted — **CJK-heavy output loses acceptance and decode speed with it**, so set `0` (full vocabulary) or build your own set with `tools/build_draft_vocab.py` over your traffic |
| `DRAFT_HEAD` | `int8` | `int4` gives the drafter a private int4 g128 RTN GPTQ-Marlin copy of the full-vocabulary head (built at first use, ~313 MiB vs 616 MiB read per draft step). Measured a wash: tg unchanged within noise, draft acceptance 1–8 points lower (see the MTP options section), so the shared int8 head stays the default |
| `PIN_PROMPT` / `PIN_MAX_FRACTION` | unset / `0.25` | Never-evict pin (patch 6); needs `PREFIX_CACHE=1` (the launcher warns otherwise). Use a distinctive substring of a few dozen characters — a word or two matches unrelated prompts |
| `API_KEY` | unset | Non-empty → Bearer auth on the API (vLLM `--api-key`; generate with `openssl rand -hex 32`). `/health` stays open; the key is visible in `docker inspect` |
| `BIND_ADDR` | unset | Host address for the published ports. Unset = all interfaces; `127.0.0.1` = this machine only (e.g. behind a reverse proxy). The metrics sidecar has no auth even with `API_KEY` |
| `FP8_HYBRID` | `1` | int4+fp8 hybrid dispatch (patch 4) |
| `PLE_MADV_RANDOM` | `1` | `MADV_RANDOM` on the table mmap (patch 1): no readahead around 160-byte row faults — upstream (blazux `0c6df7e`) measured 4–8% faster cold prefill and a cleaner page cache, now the default |
| `METRICS_PORT` | `18400` | Engine-side Prometheus sidecar (`src/vllm_custom_metrics.py`): `vllm:ple_mmap_*` counters (ops, op/gather ms, rows, bytes, prefetch hit/miss), `vllm:mamba_state_copy_guard_total` (tripwire, expect 0) and `vllm:never_evict_{blocks_reserved,pin_queue_blocks,pin_bytes}` gauges. vLLM's own `/metrics` runs in the API-server process and cannot see EngineCore counters, so these are served from the engine process on this separate port; `0` = off. The launcher probes the image for both the module and its start hooks, and warns when the sidecar cannot come up (the port would just stay closed) |
| `PLE_PREFETCH` | `0` | Batch-assembly prefetch — measured not worth enabling (see appendix) |
| `PLE_CHUNK` / `PLE_FAST_ROWS` / `PLE_STATS_SEC` | `2048` / `512` / `0`* | Gather rows per task / decode fast-path threshold / stats-log period. The log lines are off by default since the numbers moved to the metrics sidecar (`METRICS_PORT`); set `30` to bring them back. *On an image without the sidecar the launcher keeps them at `30` |
| `HIT_DEBUG` | `0` | Prefix-cache tracing (patch 8) |
| `HIT_DEBUG_N` | `0` | Event budget for `HIT_DEBUG=1`; tracing auto-disables with a warning once spent |
| `STEP_PROFILE` | `0` | `1` — on-demand torch profiler, triggered by `touch /tmp/profile_trigger` inside the container (patch 11) |
| `PREWARM` | `1` | Stream the table once at boot to warm the page cache |
| `WORKERS` | `32` | Threads for the mmap gather |
| `LOAD_FORMAT` | `fastsafetensors` | Noticeably faster cold boots |
| `TOOL_PARSER` | `qwen3_xml` | Tool-call parser (bare script: `qwen3_coder`) |
| `SERVED_NAME` | `qwen` | Model id on the API (bare script: `qwen3.8-flash-next`) |
| `ITER_DETAILS` | `0` | `1` = per-step prefill metrics: `vllm:scheduled_ctx_tokens_total` updates every engine step (stock `prompt_tokens_total` only moves when a prefill finishes). Live view: `bench/ppwatch.sh` |
| `FLASHINFER_AUTOTUNE` | `0` | `1` — enable flashinfer kernel autotuning (longer warmup, possibly faster kernels) |
| `CUDA_LAUNCH_BLOCKING` | `0` | `1` — synchronous CUDA errors, for debugging Xid 31 (much slower; not for production) |
| `RESTART` | `unless-stopped` | Container restart policy — survives reboots and crashes; `no` = manual start only. The next `./serve.sh` re-creates the container with this value |
| `LOG_MAX_SIZE` / `LOG_MAX_FILE` | `10m` / `3` | Docker log rotation — `PLE mmap stats` logs a line every 30 s, so an unbounded log grows to GBs |
| `NAME` / `IMAGE` | `qwen38-flash` / `qwen38-flash-dgx` | Container / image names (`IMAGE` is not overwritten by serve.sh — point it at a backup tag to roll back) |
| `EXTRA` | | Extra vLLM flags, passed verbatim |

The launcher probes the image before touching the running container and warns
when a knob needs a patch the image lacks (`DET_TOPK`, `EXACT_TOPK`,
`ITER_DETAILS`, `DRAFT_VOCAB`/`DRAFT_HEAD`, `METRICS_PORT`) — rebuild with
`docker build -t qwen38-flash-dgx .` rather than trusting a silently ignored
env var.

The launcher also fail-fasts on unedited `/path/to` placeholders or missing
`MODEL_DIR`/`TABLE_DIR` *before* it removes a running container, and creates
the container with a `/health` HEALTHCHECK (a 10-minute start period covers
the weight load; 10 consecutive misses → `unhealthy` in `docker ps` — a
restart on unhealthy needs an external watcher such as `willfarrell/autoheal`)
and json-file log rotation. The bench and smoke-test scripts read `BASE` /
`MODEL` / `PIN` / `API_KEY` from the environment (where applicable), so they
work unchanged with a non-default port or with auth on. Their defaults match
`serve.sh` (`http://localhost:8000`, model `qwen`); `scripts/smoke-test.sh`
also takes `host:port` as its argument and, without `MODEL`, uses whatever
`/v1/models` reports.

For graphs, scrape **two** endpoints: vLLM's own `/metrics` on `PORT` (request
throughput, cache usage, queue time) and the engine-side sidecar on
`METRICS_PORT` (default 18400) — PLE gather cost
(`rate(vllm:ple_mmap_op_ms_total)/rate(vllm:ple_mmap_ops_total)` = ms/op), the
mamba state-copy guard tripwire and the never-evict pin footprint. With the
sidecar on, the `PLE mmap stats` log lines default off (`PLE_STATS_SEC=0`).

### Sizing the KV pool: 30g, 35g or 40g

**Keep the default `KV_BYTES=30g` unless you run five or more long (100k+
token) sessions at the same time.** Measured on a GX10 (local NVMe table,
MTP=3, `SEQS=8`; `bench/longctx.py --ctx 120000 --streams 8 --gen 1000` and
`bench/perf.py --only prefill`, warmed up, 2026-09-24):

| | `30g` (default) | `35g` (estimate) | `40g` |
|---|---|---|---|
| KV pool | 27.9 GiB | 32.6 GiB | 37.3 GiB |
| Long contexts resident at once (~127k each) | **~4** (0.55M tokens) | ~5 (≈0.65M) | **~6** (0.77M) |
| Preemptions, 8 × 127k | 0 | ? | 2 |
| Long-prefill TTFT vs 30g (32k / 101k prompt) | 14.6 s / 50.0 s | ≈ +2–3.5% | +6.7% / +4.3% |
| Single-stream decode | ~66 tok/s | ~66 tok/s | ~66 tok/s |

The 35g column is interpolated from the two measured ones, not measured —
run the two commands above if you pick it.

Why it is a trade-off:

- **What more KV buys.** vLLM admits a request only when its KV blocks fit,
  so on a full pool extra long requests *wait* (`queue` in `longctx.py`)
  rather than fail. More KV = more long sessions decoding at once (agents
  with big contexts), and more room for the prefix cache to keep several long
  conversations warm between turns. It does **not** speed up a batch of long
  prompts: prefill runs one prompt at a time either way (8 × 127k took
  548 s at 30g and 564–576 s at 40g).
- **What it costs.** KV memory is pinned by the driver, so every GiB comes
  out of the page cache that holds the ~48 GiB PLE table. At 40g ~2.5 GB of
  page cache was left, n-gram row reads per 8k prefill chunk rose from
  ~184 to ~273 ms at 101k, and long prefill slowed 4–7%. Decode is
  unaffected (its hot rows stay cached). At 40g the pool also runs at ~99%
  under load, so admitted requests occasionally get preempted and recomputed.
- **Memory safety.** 40g ran for hours of mixed load without hangs; the
  kernel moved ~6 GB of cold vLLM startup memory to swap once and did not
  page it back (`vmstat` `si` ≈ 0). Check the same on your box — `vmstat 5`
  during load (`si`/`so` should stay near 0) and `cat /proc/pressure/memory`
  (`some avg10` near 0) — especially if other programs share the machine.
- **Ignore the boot log's capacity.** "GPU KV cache size: 966,390 tokens" at
  30g assumes ~31 KB/token; on long contexts the measured cost is ~53 KB/token
  (the mamba/GDN state groups hold more than that estimate assumes), i.e.
  ~0.55M tokens. `bench/longctx.py` prints the real figure for your settings.

So: **30g** for chat, coding assistants and up to ~4 concurrent long
contexts (fastest prefill, no preemption); **35g** if you regularly have
exactly one more long session than that; **40g** for 5–6 concurrent long
sessions, accepting ~5% slower long prefill.


## Limitations & notes

- **One big model at a time** — and on a Spark the OS and GPU share the pool;
  prefer the deterministic `GPU_MEM=0.01` + `KV_BYTES` sizing over a large
  fraction (an OOM inside the unified pool can freeze the box).
- **1M context is out of reach on one box**: the QSA layers refuse an fp8 KV
  cache, and in bf16 a single 1M request needs ~30 GiB of KV. 500k with YaRN
  was upstream's validated ceiling.
- **Weights are not included** and the checkpoint carries Qwen's license (with
  a MAU/revenue clause) — review it before production use.

## What runs in what precision

| Component | Precision | How |
|---|---|---|
| 512-expert MoE, 48 main layers | **int4** GPTQ-Marlin g128 | Intel checkpoint as-is |
| MTP draft layer's own 512 experts | **int4** GPTQ-Marlin g128 RTN (the default `-MTP_int4RTN` checkpoint) | Intel leaves layer 48 in bf16 (~4.7 GiB, `-:.*layers\.48\..*`); `tools/quantize_mtp_experts_int4.py` (patch 10) makes it int4 on the Marlin path: −3.5 GiB, +2–4% decode, acceptance unchanged. The plain `-hybrid` repo keeps them bf16 |
| lm_head (shared with MTP draft head) | **int8** GPTQ-Marlin (uint8b128) | `tools/quantize_lm_head_int8.py` + `"lm_head": true` |
| GDN in/out projections, QSA q/k/v/o, shared expert | **fp8** blockwise e4m3 (128×128) | `tools/fp8_convert.py` + `src/vllm_fp8_hybrid.py` |
| Embeddings, hyper-connections, norms, MoE gates, fc_hidden | bf16 | excluded via `dynamic` rules |
| PLE n-gram table (51B params, layer 1) | **fp8** rows, mmapped from disk | `tools/fetch-ple-table-fp8.sh` + the mmap patch |
| KV cache | bf16 | QSA refuses fp8 KV |

## The patches

Everything is applied at image build time (see the `Dockerfile`); each patch is
independent and gated by an env var where it changes behavior. The build is
fail-fast — patch scripts assert their anchors — and a successful build prints
one confirmation per step: `ple_layer.py patched OK`, `fla shmem gate patched`,
`fla num_warps pinned`, `auto_gptq.py patched OK`, `never-evict pin patched OK`,
`lm_head patched OK in model.py + mtp.py`, `mamba_utils.py guarded OK`,
`patch_hit_debug.py applied OK`, `patch_mamba_align_split.py applied OK`,
`qsa.py: top-k variants (1|fill) added OK`, `qsadet INSTALLED`, `qsadet wired OK`,
`patch_step_profile.py applied OK`, `loggers.py: per-step prefill metrics added
OK` and `draft-head hook INSTALLED`. A missing line means that step did not run.

### 1. PLE mmap upgrades (`src/vllm_ple_mmap.py`, extends upstream's patch)

- **Any table dtype**: bf16/f16 tables and fp8 (with `weight_scale`) are all
  accepted; row size is derived from the safetensors headers. The fp8 table
  halves the bytes read per token vs bf16.
- **`VLLM_PLE_MMAP_DIR`**: the table no longer has to live inside the
  checkpoint dir — point it at any directory of safetensors shards (NFS, local
  NVMe, a RAM-backed device...). The backend matters: the ~49 GiB table
  outgrows what the page cache can keep warm next to the model, so gathers
  cost ~1.3 ms/op from a RAM-backed source vs 5–9 ms from local NVMe vs
  30–50 ms over NFS — decode impact in the appendix below.
- **Hot path**: per-step dedup of row ids on CPU (`np.unique`), gather of
  unique rows only, staging through a persistent pinned buffer with an async
  H2D copy, and GPU-side expansion via the inverse index. A decode fast path
  (`VLLM_PLE_MMAP_FAST_ROWS`, default 512) skips the thread pool entirely for
  small gathers. Net effect: ~7.2 → ~2.5–3.8 ms per lookup op on a RAM-backed table.
- **`VLLM_PLE_MMAP_MADV_RANDOM=1`**: `madvise(MADV_RANDOM)` the mmap so faults
  stay single-page — for tables on remote RAM or boxes with no page-cache
  headroom.
- **Stats**: lifetime counters go to the metrics sidecar (patch 13); with
  `VLLM_PLE_MMAP_STATS_SEC=30` the engine also logs
  `PLE mmap stats (last Ns): calls, op ms, gather ms, rows, MB` per period.
- **Fail-fast table check**: every shard `0..parts-1` must be present at load
  time — a missing one used to surface mid-serving as a `TypeError`.

### 2. FLA shared-memory gate (`Dockerfile` sed)

sm_121 reports 99 KiB of shared memory per block — the same as ADA, where the
flash-linear-attention Triton kernels use their big GDN tiles — but the gate in
`vllm/third_party/flash_linear_attention/ops/utils.py` demands 100 KiB, so all
36 GDN layers silently fell back to small tiles. Lowering the gate to 99 KiB
(101376) lets GB10 take the big-tile path. (Found the hard way in the
Qwen3.5-122B Spark recipe — ported from
[Entrpi/qwen3.5-122B-A10B-on-spark](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark)'s
`patch_fla_shmem.py`.)

### 3. int8 lm_head enablement (`Dockerfile` sed on `model.py` / `mtp.py`)

Upstream constructs `ParallelLMHead` without `quant_config`, forcing a bf16
head (1.27 GiB, and a bf16 GEMV per token over a 248320 vocab). One added
kwarg in both the main model and the MTP draft lets the head pick up the
checkpoint's int8 GPTQ packing. Without the `mtp.py` half, MTP ≥ 3 crashes at
load ("no module or parameter named 'lm_head.qweight'").

### 4. int4+fp8 hybrid dispatch (`src/vllm_fp8_hybrid.py`, `VLLM_FP8_HYBRID=1`)

vLLM's GPTQ config quantizes listed layers and leaves the rest to
`UnquantizedLinearMethod` — it has no notion of "this excluded layer is
actually fp8 in the checkpoint". This shim wraps `AutoGPTQConfig`: it scans the
checkpoint metadata for `F8_E4M3` weights with a `weight_scale_inv` sibling and
routes exactly those layers to vLLM's blockwise-`Fp8Config`
(`weight_block_size=[128,128]`, dynamic activation scheme) while everything
else keeps the GPTQ path. `VLLM_USE_DEEP_GEMM=0` is required on sm_121
(DeepGEMM hits `CUDA_ERROR_LAUNCH_FAILED`); the triton fallback is fine.

### 5. Prefix caching: on, and fixed (`src/patch_mamba_align_split.py`)

Upstream runs `--no-enable-prefix-caching` because of a CUBLAS error in the
GDN `in_proj` GEMM on the cached-block path. With the fp8 side layers that
GEMM runs a different kernel, and prefix caching is stable in our serving
(`PREFIX_CACHE=1`).

It also *works properly* now. On this hybrid model the reconciled cache hit is
the **minimum across all KV cache groups** (full attention + four mamba/GDN
state groups), and mamba "align" mode can only cache a state at a prefill
chunk end on a 1600-token boundary. The image's scheduler aligned those chunk
ends to `cache_config.block_size` — which the engine rewrites to the *minimum*
group block size (8, the MTP draft granularity) — so chunks ended where no
mamba state was cacheable, a cold request published **zero** reusable mamba
states, and a repeated prompt only got fast on the **3rd** try (the miss
triggers junction machinery that rebuilds the boundary one request late).
Long prompts effectively never hit. The patch makes the split use
`cache_config.mamba_block_size` (1600). Verified: an 8k-token repeat goes
10.1 s → **0.90 s on the 2nd request**.

The same rewritten `block_size` also poisoned the **worker** side: the
align-mode state-slot seed (`mamba_hybrid.py`) divided by it too, so a prefix
hit at 6400 tokens seeded state column 799 instead of 3, read past the
block-table row and restored a wrong (often all-zero / stale) mamba state —
greedy outputs visibly changed on cache hits. Root-caused upstream by
[blazux](https://github.com/blazux/qwen3.8-Flash-DGX/issues/2#issuecomment-546252046)
(his fix: `8347e7c`); the same one-line seed fix is folded into this patch.
Verified: cold-vs-hit first-token logprobs now agree within the stack's
normal run-to-run jitter (Marlin atomic-add nondeterminism), where before the
fix greedy outputs diverged within the first few tokens.

Notes: the prefix-cache granularity is large (1600 tokens; shorter prefixes
get no reuse), and a repeat hit tops out at `round_down(P,1600) − 1600` — MTP
(eagle-style) always recomputes the last matched block.

### 6. Never-evict prompt pinning (`src/patch_never_evict.py`)

`--never-evict-kv-cache-prompt-includes "<substring of your system prompt>"`
pins the KV blocks of any prompt containing that marker: they are held in a
side queue on `BlockPool`, excluded from the free count, and thus never handed
out for eviction — your assistant's system prompt stays cached no matter what
other traffic does. `--never-evict-kv-cache-max-fraction` (default 0.25) caps
the pin. The pin set is *replaced* on each matching request, so an updated
system prompt releases the old blocks automatically.

Verified end-to-end: a pinned 8k prompt still answers in **0.94 s after 2M
tokens of unique traffic** (3.1× full KV-pool turnover).

Implementation notes: the marker is tokenized once and matched as a token-id
subsequence (first/last token dropped — BPE merges at the boundaries); the pin
is keyed on block *hashes*, not block ids, because this hybrid model frees
mamba/GDN state blocks mid-request — each freed block is re-claimed by the pin
the moment `free_blocks()` sees it. Only prefix-cacheable KV groups
participate (the MTP draft layer's group is not, and must be skipped). This is
a pin-only port of our `arc_pin2` patch from the Qwen3.5-122B Spark stack,
which built the pin on top of the ARC GPU-eviction work in
[vllm#40270](https://github.com/vllm-project/vllm/pull/40270); the ARC/2Q
policies themselves were deliberately dropped — the stock free queue is
C-speed on a path this model hits every step.

Self-check (no GPU): `docker run --rm -v "$PWD/src:/t" --entrypoint python3
qwen38-flash-dgx /t/test_never_evict_pin.py`.

### 7. Mamba state-copy guard (`src/mamba_utils_guarded.py`)

Hardens the align-mode state-copy kernels against the "CUDA illegal memory
access / Xid 31 under load" crash class (also
[blazux#2](https://github.com/blazux/qwen3.8-Flash-DGX/issues/2)): backports
[vllm#50729](https://github.com/vllm-project/vllm/pull/50729) (overlapping
state-copy race) and bounds-checks every block id against its state pool
before dereferencing — an out-of-range id skips the copy and bumps a counter
(logged as `mamba state-copy guard`) instead of taking down the CUDA context.

With the block-size seed fix (patch 5) the out-of-range ids the guard was
absorbing are gone at the root: the counter is expected to stay at **0**, and
a nonzero count is logged as an *error* — it now indicates a new bug worth
reporting, not a known quirk.

### 8. Prefix-cache tracing (`src/patch_hit_debug.py`, `HIT_DEBUG=1`)

Set `HIT_DEBUG=1` (→ `VLLM_HIT_DEBUG=1` in the container) to log, per request:
the per-KV-group hit reconciliation (which group truncated the hit), mamba
boundary-state publication (which slots were real/null/hashed), cached-block
evictions, and prefill chunk-stop decisions. This is what found the bug in
patch 5; costs nothing when off.

### 9. Deterministic QSA top-k (`src/patch_qsa_exact_topk.py` + a kernel built in the `Dockerfile`)

Taken from upstream [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
(`8347e7c`, `4b723de`, `0022e36`). The sparse attention (QSA) picks its top-k key
blocks with vLLM's `persistent_topk` kernel, which is non-deterministic on GB10 and
can drop real candidates ([vllm#51782](https://github.com/vllm-project/vllm/issues/51782),
diagnosed by [@k3dani](https://github.com/k3dani) in
[blazux#3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3)). Two fixes, both opt-in
env vars, `DET_TOPK=1` the default:

- **`DET_TOPK=1` — deterministic kernel** by [@jschmied](https://github.com/jschmied)
  ([vllm#55122](https://github.com/vllm-project/vllm/pull/55122)): index-ordered output
  slots and bit-exact tie resolution (signed zero canonicalised), plus a deterministic
  low-shared-memory fallback. The sources are fetched at a pinned commit (sha256-checked)
  and compiled with the image's nvcc at build time as a standalone `_C_det.so` — no vLLM
  rebuild; `qsadet_patch.py` (also @jschmied's) wires the QSA block selection to it.
  Upstream measured the whole prefill penalty of the exact path gone (GX10, 32k:
  1,794 → 2,996 tok/s) with decode unchanged.
- **`EXACT_TOPK=1` — exact `torch.topk`** over the visible columns (blazux's first fix):
  also deterministic, −20–40% on long prefill; kept as the fallback and wins over
  `DET_TOPK` when set. Self-check (no GPU): `docker run --rm -v "$PWD/src:/t" -w /t
  --entrypoint python3 qwen38-flash-dgx test_qsa_exact_topk_cpu.py`.

### 10. MTP drafter options: int4 draft experts + reduced draft vocabulary

Both on by default (the Quickstart checkpoint + `DRAFT_VOCAB=1`); the plain
`-hybrid` checkpoint and `DRAFT_VOCAB=0` remain options. Both leave outputs
unchanged — the target verifies every drafted token, only the draft's cost and
acceptance move. Measured on a DGX Spark (`bench_qwen35.sh` with
`enable_thinking: false`, T=0, second run):

| | bf16 draft (`-hybrid` repo) | int4 draft experts | int4 + `DRAFT_VOCAB=1` (default) |
|---|---|---|---|
| Code / JSON / LongCode tok/s | 57.3 / 61.3 / 57.9 | 59.1 / 62.6 / 60.3 | **62.0 / 66.0 / 62.4** |
| draft acceptance (of 3) | 88.0% (2.64) | 88.5% (2.65) | 87.3% (2.62) |
| weights resident | ~71.4 GiB | 67.9 GiB | 67.9 GiB |

With thinking **on** (albond's original script) draft acceptance is ~70% in all three
configs — only 1.0% of reasoning tokens fall outside the 65k set — and the default
config is still the fastest (Code 56.5, JSON 62.1, LongCode 48.5 tok/s, see the
Throughput table).

- **int4 draft experts** — Intel's checkpoint leaves the MTP layer's 512 routed
  experts in bf16 (~4.7 GiB on the unquantized MoE path). `tools/quantize_mtp_experts_int4.py`
  writes a variant checkpoint (hardlinks for the untouched shards) with those
  experts as int4 g128 RTN in Intel's exact GPTQ layout and drops the `layers.48`
  exclusion, so the draft runs on GPTQ-Marlin like the main layers. Prebuilt:
  [Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN](https://huggingface.co/Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN)
  — point `MODEL_DIR` at it. Idea from upstream's `hybrid-mtp` mode
  ([blazux#11](https://github.com/blazux/qwen3.8-Flash-DGX/pull/11) by @pfy, after
  thavoc's graft write-up), done here as RTN int4 instead of an NVFP4 graft so no
  new kernel path is needed.
- **`DRAFT_VOCAB=1`** (`src/patch_mtp_draft_vocab.py`, from upstream blazux `0c6df7e`,
  idea from MiaAI-Lab's recipe reimplemented there) — vLLM shares the target's
  lm_head with the drafter, so every draft pass scores all 248,320 rows. With the
  flag the draft scores a private 65,536-row slice (`src/draft_vocab_65536.npy`:
  corpus frequency + BPE order + all special tokens; `tools/build_draft_vocab.py`
  rebuilds it) and every other id is −inf. This fork's head is int8 GPTQ-Marlin,
  so the slice is dequantized from the checkpoint's GPTQ tensors at first use
  (616 → 320 MiB read per draft step).
- **`DRAFT_HEAD=int4`** (same patch) — the other way to halve the head read
  without touching the vocabulary: a private int4 g128 RTN copy of the full head,
  quantized with vLLM's own `marlin_quantize` at first use (~23 s) from the
  checkpoint's int8 GPTQ tensors, run through the same GPTQ-Marlin kernel
  (616 → 313 MiB per draft step; +313 MB resident). Outputs cannot change (the
  target verifies every token), only acceptance can, and it does: int4 RTN puts
  ~11% relative error on the head (int8: ~0.5%). Measured 2026-09-10, thinking
  off, 600-token answers, second run of each prompt:

  | | int8 shared head (default) | `DRAFT_HEAD=int4` |
  |---|---|---|
  | Traditional Chinese prose | 34.6 tok/s, 31% accept | 35.5 tok/s, 30% |
  | Cantonese prose | 32.8 tok/s, 28% | 32.3 tok/s, 25% |
  | English prose | 40.4 tok/s, 44% | 39.6 tok/s, 39% |
  | Python + tests | 62.5 tok/s, 86% | 60.7 tok/s, 78% |

  The bandwidth saved is handed back in rejected drafts, so it stays off. A
  calibrated int4 (AutoRound/GPTQ on real MTP hidden states) might keep more
  acceptance; not tried. Context: krisbailey.com's "shortlist MTP" write-up found
  the same ~4% ceiling for head-shrinking tricks once the head kernel is
  bandwidth-bound, which Marlin already is here.

### 11. On-demand step profiling (`src/patch_step_profile.py`, `STEP_PROFILE=1`)

This vLLM build predates `VLLM_TORCH_PROFILER_DIR` and `/start_profile`. With
`STEP_PROFILE=1`, `docker exec qwen38-flash touch /tmp/profile_trigger` makes
the next 24 engine steps run under torch.profiler (CPU+CUDA) and writes a
chrome trace to `/tmp/step_profile_<n>.json` — `docker cp` it out and open at
ui.perfetto.dev. Old traces are pruned to the 3 newest; when idle the patch
costs one `os.path.exists` per step.

### 12. Per-step prefill metrics (`src/patch_prefill_metrics.py`, `ITER_DETAILS=1`)

vLLM credits `vllm:prompt_tokens_total` only when a request's prefill
*finishes*, so a 100k-token prompt shows 0 tok/s for a minute and then a spike.
With `ITER_DETAILS=1` (→ `--enable-logging-iteration-details`) this patch feeds
`vllm:scheduled_ctx_tokens_total` and `vllm:scheduled_iterations_total` on
vLLM's own `/metrics` every engine step and mutes the stock one-line-per-step
log. `bench/ppwatch.sh` prints live prefill tok/s from them. From
Saren-Arterius/qwen3.8-Flash-DGX-AutoRound. Self-check: `docker run --rm -v
"$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx test_prefill_metrics_cpu.py`.

### 13. Engine-side metrics sidecar (`src/vllm_custom_metrics.py`, `METRICS_PORT`)

vLLM's own `/metrics` is served by the API-server process, which cannot see the
EngineCore-side counters this fork cares about; plumbing them through the
serialized `SchedulerStats` IPC would mean patching vLLM internals blindly. So
the engine process serves them itself with `prometheus_client` (the library
vLLM already uses) on its own port: `vllm:ple_mmap_*` (ops, op/gather ms, rows,
bytes, prefetch hit/miss — `rate(op_ms)/rate(ops)` is ms/op), the
`vllm:mamba_state_copy_guard_total` tripwire and the `vllm:never_evict_*` pin
gauges. It starts lazily from the first data push (i.e. in the process that
actually has data) and every failure is swallowed — telemetry never takes the
server down. With it on, the `PLE mmap stats` log lines default off
(`PLE_STATS_SEC=0`); on an image without the module the launcher keeps them at
`30` instead of going silent.

## Speculative decoding and TTFT

MTP's draft layer is a stateful autoregressive transformer: vLLM's v1 engine
only emits the first token after the drafter has run, and on every prefill
chunk the drafter runs a full-chunk-width forward (always eager: above the
cudagraph capture sizes) to sync its own KV/GDN state, plus k−1 sequential
single-token passes. Earlier versions of this README called that a ~0.8 s TTFT
floor. On this stack, warmed up (`bench/perf.py`, MTP=3), it is not a floor
but part of the prefill rate:

| prompt | 1 line | 2,194 | 8,481 | 33,612 | 105,290 tokens |
|---|---|---|---|---|---|
| TTFT | 0.13 s | 1.04 s | 3.68 s | 14.98 s | 49.77 s |

— i.e. TTFT ≈ 0.1 s + prompt / ~2,200–2,400 tok/s. A cold server does show
second-long stalls, from first-use kernel compiles (next section), not from
MTP. Cross-attention drafters like DFlash skip the per-chunk drafter forward,
but Flash-Next has no such drafter; `MTP=0` removes that cost from prefill at
the price of decode speed.

### Warmup

After every restart, Triton compiles several kernels on first use *during
inference* — vLLM's `jit_monitor` logs each one as `Triton kernel JIT
compilation during inference`: the spec-decode sampling kernels
(`_rejection_kernel`, `_resample_kernel`, ...) on the first request, the QSA
split-k/merge kernels at the first few concurrent streams, and the QSA indexer
(`_qsa_pre_indexer_kernel`, `_expand_qsa_indices_kernel`) on the first ~2k-token
prompt. Each is a stall of about a second for whoever hits it first — on a
fresh container the first 4-stream wave saw TTFT 1.94 s instead of 0.39 s.
`python3 bench/perf.py --only warmup` touches all of those shapes in ~20 s;
`bench/perf.py` runs it before measuring (`--no-warmup` to measure a cold
server).

## What's in here

```
Dockerfile                    official vLLM Flash-Next image + the patches above
serve.sh                      example launcher config (edit paths, run)
prepare.sh                    build the checkpoint + table from Intel's release
src/vllm_ple_mmap.py          mmap PLE table (any dtype, relocatable dir, fast gather)
src/vllm_custom_metrics.py    engine-side Prometheus sidecar: PLE/guard/pin (METRICS_PORT)
src/vllm_fp8_hybrid.py        int4+fp8 hybrid dispatch on the GPTQ config
src/patch_never_evict.py      never-evict system-prompt KV pinning
src/patch_mamba_align_split.py  prefix-cache chunk-alignment fix
src/patch_hit_debug.py        prefix-cache tracing (VLLM_HIT_DEBUG)
src/patch_qsa_exact_topk.py   exact, deterministic QSA top-k (VLLM_QSA_EXACT_TOPK=1; from blazux)
(Dockerfile, _C_det.so)       @jschmied's deterministic persistent_topk kernel, built at docker build
src/test_qsa_exact_topk_cpu.py  CPU unit test for the exact top-k (no GPU needed)
src/patch_mtp_draft_vocab.py  private MTP draft head: reduced vocabulary (from blazux) / int4 (VLLM_MTP_DRAFT_VOCAB, VLLM_MTP_DRAFT_HEAD)
src/draft_vocab_65536.npy     the default 65,536-token draft id set (from blazux)
tools/build_draft_vocab.py    rebuild the draft id set (from blazux)
tools/quantize_mtp_experts_int4.py  int4 RTN the MTP draft experts -> -MTP_int4RTN checkpoint
src/mamba_utils_guarded.py    hardened align-mode state copy (vllm#50729 + guard)
src/test_ple_mmap_cpu.py      CPU unit test for the gather (no GPU needed)
src/test_never_evict_pin.py   CPU unit test for the pin (no GPU needed)
src/test_custom_metrics_cpu.py  CPU unit test for the metrics sidecar (no GPU, no vLLM)
src/test_prefill_metrics_cpu.py  check for the prefill-metrics patch (inside the image)
src/patch_step_profile.py     on-demand torch.profiler around engine steps (VLLM_STEP_PROFILE)
scripts/serve-intel-ar.sh     the docker run behind serve.sh
scripts/smoke-test.sh         health + coherence + prefill/decode numbers
src/patch_prefill_metrics.py  per-step prefill counters for Prometheus (ITER_DETAILS=1)
bench/ppwatch.sh              live prefill tok/s from those counters
bench/decode_bench.py         batch-1 decode / TTFT / spec-acceptance bench
bench/concurrency_bench.py    N-stream throughput / TTFT / queue-time bench (completions API)
bench/perf.py                 one-command overview over the chat API: warmup, TTFT, decode, concurrency, prefill
bench/longctx.py              concurrent long-context test (queue / preemption / cache checks)
bench/common.py               shared chat-API client for perf.py / longctx.py
tools/eval_quality.py         ppl + greedy-facts quality check against the API
tools/test_tools_cpu.py       CPU test for the preparation tools (correctness, safe re-runs)
tools/                        CPU-only checkpoint preparation
docs/HOW-IT-WORKS.md          upstream's mmap-PLE story (NVFP4-era numbers)
docs/OPTIMIZATIONS.md         stub (kept for old links; the recipe lives in this README)
.github/workflows/ci.yml      CI: ruff, shellcheck, the CPU tests
```

## Credits

- Model: **Qwen team, Alibaba** — Qwen3.8-Flash-Next.
- This repository is based on
  **[Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)**,
  the int4/int8/fp8 AutoRound fork this README describes.
- **This is a fork of [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)** —
  the original mmap-PLE idea, the GB10 serving recipe, the NVFP4 path, and
  docs/HOW-IT-WORKS.md are theirs.
- int4 checkpoint this fork builds on: **[Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)**
  (AutoRound); fp8 PLE table from **[Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)**.
- Several pieces originate in the Qwen3.5-122B-A10B Spark recipes:
  the int4 AutoRound + int8 lm_head serving approach from
  **[albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)**;
  the FLA shared-memory gate fix and the int4+fp8 hybrid idea from
  **[Entrpi/qwen3.5-122B-A10B-on-spark](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark)**.
  The never-evict pin was built for that 122B stack on top of the ARC
  GPU-eviction work in [vllm#40270](https://github.com/vllm-project/vllm/pull/40270)
  and re-ported here.
- Serving engine and base image: **vLLM** (`vllm/vllm-openai:qwen38-flash-next`,
  the `release/qwen38next` recipe / PR #53896).
- MTP drafter options: the reduced draft vocabulary (patch, id set, build tool) from
  upstream **[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)**
  (`0c6df7e`; idea from MiaAI-Lab's recipe, reimplemented there); the quantized draft
  experts follow upstream's `hybrid-mtp` mode
  ([blazux#11](https://github.com/blazux/qwen3.8-Flash-DGX/pull/11) by **@pfy**).
- Deterministic QSA top-k: the exact `torch.topk` fix and the kernel wiring/pins from
  upstream **[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)**;
  the deterministic `persistent_topk` kernel itself by **[@jschmied](https://github.com/jschmied)**
  ([vllm#55122](https://github.com/vllm-project/vllm/pull/55122)); the bug diagnosed by
  **[@k3dani](https://github.com/k3dani)** ([blazux#3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3)).
- Independent reproduction of the upstream recipe, the native-offload fixes and
  the concurrency measurements: **[@jschmied](https://github.com/jschmied)**
  ([blazux#1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1),
  [qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10)).
- License: [Apache-2.0](LICENSE).

## Appendix: where to put the PLE table (and why not to bother with `PLE_PREFETCH`)

Batch-1 decode bench (`bench/decode_bench.py`, medians of 3; W1 = fresh-prompt
1000-token decode, W2 = pinned ~8k prefix hit + 256 tokens; DGX Spark GB10,
MTP=3). Only the table location and the prefetch flag change between rows (all local NVMe is Gen 4):

| table source              | prefetch | W1 tok/s | W2 tok/s | gather      |
|---------------------------|----------|----------|----------|-------------|
| RDMA row daemon           | on       | 42.4     | 47.8     | ~1.3 ms/op  |
| RDMA row daemon           | off      | 41.6     | 46.7     | ~1.3 ms/op  |
| local NVMe                | off      | 36.5     | 45.7     | 5–9 ms/op   |
| local NVMe                | on       | 34.5     | 42.6     | 9–17 ms/op  |
| local NVMe, warm cache    | on       | 33.4     | 41.5     | 9–16 ms/op  |
| NFS (RDMA mount, btrfs)   | on       | 26.2     | 36.5     | 26–56 ms/op |
| NFS (RDMA mount, btrfs)   | off      | 24.2     | 32.9     | 30–42 ms/op |

Takeaways:

- **Put the table on local NVMe.** It costs ~15% decode vs an exotic fast row
  source, and it is the simple recipe. Serving straight off a NAS works but
  costs ~40% (and that was NFS-over-RDMA to a btrfs box — a plain GbE NAS
  will be worse).
- **Don't bother with `PLE_PREFETCH=1`** (`VLLM_PLE_MMAP_PREFETCH`). The
  batch-assembly prefetch only wins where the row source is very slow (NFS,
  +2–3 tok/s, within run-to-run variance), is a wash on fast sources, and is
  **net-negative on local NVMe**: the handoff/wait in `consume()` costs more
  than the inline gather it replaces (confirmed on a warm page cache). It
  stays experimental and default-off.
- TTFT is unaffected by any of this (prefill compute dominates), and
  between-restart variance on this bench is >10% — treat small deltas above
  accordingly.

About those "RDMA row daemon" rows: that is a custom one-sided RDMA READ
server that pins the whole table in another box's RAM — the fastest row
source measured, at the price of a second machine and an ibverbs science
project. If you have a NAS with **≥64 GB of RAM** and a **≥100 Gbit RDMA
link** to your Spark — and no second DGX Spark to put to better use — the
[`magi` branch](../../tree/magi) ships the tool (`src/ple_rdma/`) and setup
notes ("PLE table over RDMA"). Everyone else: local NVMe is the recipe.
