#!/usr/bin/env bash
# Example launcher for scripts/serve-intel-ar.sh — point the paths at your
# machine (or keep your real settings as a local-only commit on top of this).
# Every knob here is passed through to serve-intel-ar.sh's docker run;
# anything unset falls back to that script's defaults.
cd "$(dirname "$0")" || exit 1

# Required: the prepared checkpoint (int4 experts + int8 lm_head + fp8 side
# layers — see tools/) and the stripped fp8 ngram/PLE table directory.
# The -MTP_int4RTN variant (int4 draft experts, -3.5 GiB, the default) is
# built by prepare.sh step 8 / tools/quantize_mtp_experts_int4.py; the plain
# -hybrid dir (bf16 draft) works too.
export MODEL_DIR="/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN"
export TABLE_DIR="/models/ple-table-fp8"

export PORT=8000
export SERVED_NAME=qwen
export TOOL_PARSER=qwen3_xml
export SEQS=8
export MTP=3
export PREFIX_CACHE=1

# API key (Bearer token) for the OpenAI-compatible endpoint:
#   non-empty -> clients must send "Authorization: Bearer <key>" (vLLM --api-key)
#   empty     -> no auth, as before (closed-network deployment)
# Generate: openssl rand -hex 32  (hex: no spaces/quotes, safe to inline here).
# To keep the secret out of git, read it from a file outside the repo instead:
#   export API_KEY="$(cat "$HOME/.qwen-api-key")"
# /health stays open (container HEALTHCHECK unaffected); curl examples and
# bench/decode_bench.py will need the Authorization header while this is set.
export API_KEY=''

# Host address the API/metrics ports are published on. Empty = all
# interfaces; 127.0.0.1 = this machine only (e.g. behind a reverse proxy).
# The metrics sidecar (METRICS_PORT) has no auth even when API_KEY is set.
export BIND_ADDR=''

# Deterministic memory sizing for unified-memory boxes (GB10 / DGX Spark):
# near-zero utilization fraction plus an explicit KV pool, so the driver
# never oversubscribes the unified pool (NV_ERR_NO_MEMORY / Xid 31 crashes).
export GPU_MEM=0.01
export KV_BYTES=30g

# Context: 262144 native. For 500k via YaRN set CTX=500000 YARN=1 (the
# upstream-validated ceiling). KV_BYTES=30g (vLLM reads "g" as 10^9 bytes =
# 27.9 GiB) held ~4 concurrent 127k contexts (~0.55M tokens, ~53 KB/token
# measured) on a GX10 — the boot log's "~966k tokens" overstates it. 35g/40g
# fit ~5/~6 such contexts but cost 2-7% long-prefill speed: every GiB of KV
# is a GiB less page cache for the ~48 GiB PLE table. See the README's
# "Sizing the KV pool" before raising it. The launcher pins
# the MTP draft model's length to CTX so speculative decoding still boots.
export CTX=262144
export YARN=0

# Restart policy of the container: unless-stopped = авто-запуск после ребута
# машины и перезапуск после падений (остановленный вручную не поднимется);
# no = только ручной запуск через ./serve.sh.
export RESTART=unless-stopped

# madvise(MADV_RANDOM) the PLE mmap: no readahead around 160-byte row faults.
# Upstream (blazux 0c6df7e) measured 4-8% faster cold prefill and a cleaner
# page cache; on by default. 0 = kernel readahead (worth trying when the
# table sits on remote RAM with no page-cache headroom).
export PLE_MADV_RANDOM=1

# QSA top-k (sparse-attention block selection): the stock kernel is
# non-deterministic on GB10 and can drop candidates (vllm#51782).
# DET_TOPK=1 = @jschmied's deterministic kernel at full speed (default);
# EXACT_TOPK=1 = exact torch.topk fallback (slower long prefill, wins when set).
export DET_TOPK=1
export EXACT_TOPK=0

# Per-step prefill metrics on vLLM's /metrics (vllm:scheduled_ctx_tokens_total;
# watch live prefill tok/s with bench/ppwatch.sh). 0 = off.
export ITER_DETAILS=0

# Engine-side metrics sidecar port (PLE gather counters, the mamba guard
# tripwire, the never-evict pin gauges — vllm_custom_metrics): served from the
# engine process on its own endpoint, because vLLM's /metrics in the API
# process cannot see them. 0 = off. PLE log lines are off by default now
# (PLE_STATS_SEC=0) — the same numbers are on this endpoint.
export METRICS_PORT=18400

# Prefix-cache diagnosis logging (VLLM_HIT_DEBUG=1 in the container):
# per-group hit breakdown, mamba boundary publication, evictions, chunk stops.
export HIT_DEBUG=0

# Never-evict pin: any request whose prompt contains this exact substring has
# its prompt-prefix KV blocks pinned (held out of eviction) — meant for a
# long fixed system prompt. Empty disables it. Needs PREFIX_CACHE=1; use a
# distinctive substring (a few dozen characters), not a word or two.
export PIN_PROMPT=''

exec scripts/serve-intel-ar.sh
