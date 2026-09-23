# Qwen3.8-Flash-Next on a single DGX Spark / GB10, via vLLM.
#
# Starts from the official Qwen3.8-Flash-Next vLLM image and appends one patch:
# it serves the 51B-parameter n-gram ("PLE") table from disk via mmap instead of
# keeping it resident in the 128 GB unified pool. That is the single change that
# lets the ~176B checkpoint (int4/int8/fp8 hybrid, ~71 GiB resident here;
# 122 GiB NVFP4 upstream) fit next to a real KV cache on one box.
#
#   docker build -t qwen38-flash-dgx .
#
# The base image is multi-arch (arm64 for the Spark's Grace CPU). Pinned by digest
# for reproducibility; bump the tag below if the upstream recipe moves.
FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

# Package layout inside the official image (vLLM 0.1.dev20073, torch 2.13 cu130,
# numpy 2.2.6 — the patch needs numpy, already present).
ARG SP=/usr/local/lib/python3.12/dist-packages
ARG PLE=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py
ARG QSA_OPS=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py

COPY src/vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py
# Engine-side Prometheus sidecar: PLE gather counters, the mamba state-copy
# guard tripwire and the never-evict pin gauges on their own port
# (VLLM_CUSTOM_METRICS_PORT, default 18400; published by serve as METRICS_PORT).
# vLLM's own /metrics runs in the API-server process and cannot see
# EngineCore-side counters, so this serves them from the engine process.
COPY src/vllm_custom_metrics.py ${SP}/vllm_custom_metrics.py

# Append the hook to the model file. No-op unless VLLM_PLE_MMAP=1 at runtime, so
# the image still behaves exactly like upstream when the flag is off.
RUN cp ${PLE} ${PLE}.orig \
 && printf '\n\n# --- qwen38-flash-dgx: serve the PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; ast.parse(open('${PLE}').read()); print('ple_layer.py patched OK')"

# spark-fla-shmem (from Saren's 122B recipe): sm121 reports 99 KiB shared mem
# (= ADA, where big tiles fit) but the FLA gate demands 100 KiB -> the 36 GDN
# layers run small Triton tiles. Lower the gate so GB10 gets big tiles.
ARG FLA_UTILS=${SP}/vllm/third_party/flash_linear_attention/ops/utils.py
RUN sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # spark-fla-shmem: GB10 99KiB = ADA, big GDN tiles fit|' ${FLA_UTILS} \
 && grep -q "spark-fla-shmem" ${FLA_UTILS} && echo "fla shmem gate patched"

# spark-fla-warps (fla-org/flash-linear-attention#953): the chunked delta-rule
# state kernel races on Blackwell when autotune picks num_warps=4 — a tl.dot
# recurrence race yielding nondeterministic h/v_new, i.e. corrupt GDN state.
# The USE_INITIAL_STATE (prefix-cache resume) variant autotunes separately, so
# corruption tracks the cached-block path. Upstream pins num_warps=2 on
# Blackwell; this image only ever runs on GB10, so pin unconditionally.
ARG FLA_CDH=${SP}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
RUN sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # spark-fla-warps: fla#953 Blackwell tl.dot race|' ${FLA_CDH} \
 && grep -q "spark-fla-warps" ${FLA_CDH} && echo "fla num_warps pinned"

# int4+fp8 hybrid: dispatch blockwise-fp8 side layers from AutoGPTQConfig
# (no-op unless VLLM_FP8_HYBRID=1 at runtime).
ARG GPTQ_PY=${SP}/vllm/model_executor/layers/quantization/auto_gptq.py
COPY src/vllm_fp8_hybrid.py ${SP}/vllm_fp8_hybrid.py
RUN printf '\n\n# --- qwen38-flash-dgx: int4+fp8 hybrid dispatch (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n' >> ${GPTQ_PY} \
 && python3 -c "import ast; ast.parse(open('${GPTQ_PY}').read()); print('auto_gptq.py patched OK')"

# never-evict prompt pinning (pin-only port of the 122B recipe's arc_pin2):
# --never-evict-kv-cache-prompt-includes pins the HA system prompt's KV blocks
# against eviction. No-op unless the flag is passed at runtime.
COPY src/patch_never_evict.py /tmp/patch_never_evict.py
RUN python3 /tmp/patch_never_evict.py && rm /tmp/patch_never_evict.py

# Let the LM head pick up the checkpoint's quantization (int8 GPTQ head):
# upstream constructs ParallelLMHead without quant_config, forcing bf16.
ARG MODEL_PY=${SP}/vllm/models/qwen3_8_flash_next/nvidia/model.py
ARG MTP_PY=${SP}/vllm/models/qwen3_8_flash_next/nvidia/mtp.py
RUN cp ${MODEL_PY} ${MODEL_PY}.orig && cp ${MTP_PY} ${MTP_PY}.orig \
 && sed -i 's|prefix=maybe_prefix(prefix, "lm_head"),|quant_config=vllm_config.quant_config,\n            prefix=maybe_prefix(prefix, "lm_head"),|' ${MODEL_PY} \
 && sed -i 's|prefix=maybe_prefix(prefix, "lm_head"),|quant_config=vllm_config.quant_config,\n                    prefix=maybe_prefix(prefix, "lm_head"),|' ${MTP_PY} \
 && grep -c 'quant_config=vllm_config.quant_config' ${MODEL_PY} ${MTP_PY} \
 && python3 -c "import ast; [ast.parse(open(p).read()) for p in ('${MODEL_PY}','${MTP_PY}')]; print('lm_head patched OK in model.py + mtp.py')"

# mamba align-mode state-copy hardening (the "Xid 31 / illegal memory access
# under load" crash with PREFIX_CACHE=1 + MTP — also blazux/qwen3.8-Flash-DGX#2):
# CUDA_LAUNCH_BLOCKING=1 caught the fault synchronously inside vLLM's
# precopy_mamba_align_fused_kernel reading a wild address derived from a bad
# block id. src/mamba_utils_guarded.py is the image's stock
# vllm/v1/worker/mamba_utils.py plus:
#   1. upstream a02cfccbc6 "[Bugfix][Mamba] Fix overlapping state copy race"
#      (vllm#50729, landed after this image's vLLM snapshot)
#   2. a bounds guard in _copy_mamba_state_block: block ids are validated
#      against each state pool before dereferencing; an out-of-range id skips
#      the copy and bumps a counter (logged as "mamba state-copy guard")
#      instead of taking down the CUDA context.
ARG MAMBA_UTILS=${SP}/vllm/v1/worker/mamba_utils.py
RUN cp ${MAMBA_UTILS} ${MAMBA_UTILS}.orig
COPY src/mamba_utils_guarded.py ${MAMBA_UTILS}
RUN python3 -c "import ast; ast.parse(open('${MAMBA_UTILS}').read()); print('mamba_utils.py guarded OK')"

# prefix-cache diagnosis logging (VLLM_HIT_DEBUG=1): per-group hit breakdown,
# mamba boundary-state publication, cached-block eviction, prefill chunk stops.
COPY src/patch_hit_debug.py /tmp/patch_hit_debug.py
RUN python3 /tmp/patch_hit_debug.py && rm /tmp/patch_hit_debug.py

# prefill chunks must end at MAMBA block boundaries (1600), not the scheduler
# minimum block size (8) — otherwise cold requests publish no mamba state and
# repeated prompts only hit the prefix cache from the 3rd request on.
COPY src/patch_mamba_align_split.py /tmp/patch_mamba_align_split.py
RUN python3 /tmp/patch_mamba_align_split.py && rm /tmp/patch_mamba_align_split.py

# Exact QSA top-k (VLLM_QSA_EXACT_TOPK=1|fill), from upstream blazux/qwen3.8-Flash-DGX
# (8347e7c) via Saren-Arterius/qwen3.8-Flash-DGX-AutoRound. The stock persistent_topk
# kernel is non-deterministic on GB10 and can drop real top-k candidates (vllm#51782;
# reported by @k3dani, blazux#3). The exact path uses torch.topk over the visible
# columns: deterministic, but -20-40% long prefill. Superseded by the kernel below,
# kept as the fallback (wins over it when set). Inert unless the env is set.
COPY src/patch_qsa_exact_topk.py /tmp/patch_qsa_exact_topk.py
RUN python3 /tmp/patch_qsa_exact_topk.py ${QSA_OPS} && rm /tmp/patch_qsa_exact_topk.py

# Deterministic persistent_topk kernel (VLLM_QSA_DET_TOPK=1, the serve default):
# @jschmied's fix for the same bug at kernel speed (upstream as vllm#55122), built
# here as a standalone extension (_C_det.so) with the image's nvcc — no vLLM rebuild.
# Upstream measured on a GX10, vs the exact path: 8k 1,476 -> 2,488 tok/s, 32k
# 1,794 -> 2,996, decode unchanged. Wiring and pins from Saren-Arterius/
# qwen3.8-Flash-DGX-AutoRound (blazux 4b723de; pin bumped in 0022e36 by @jschmied:
# signed-zero canonicalisation, deterministic low-shared-memory fallback, launcher
# chunk sizing for 24576/49152-wide rows). Sources fetched from
# https://github.com/jschmied/qwen38-flash-next-gb10 at a pinned commit AND sha256
# (Apache-2.0; attribution: @jschmied). ADD --checksum needs BuildKit (the default
# since Docker 23). DET_ARCH=120a for x86 Blackwell (RTX 5090).
ARG KDET_SHA=e0ef69d4f5575dad00d34e05479eaf4c6547bace
ARG KDET=https://raw.githubusercontent.com/jschmied/qwen38-flash-next-gb10/${KDET_SHA}
ARG DET_ARCH=121a
ADD --checksum=sha256:138cacfc5eb117f0922d53c88727e4d0dc26dcfb246c3d401fc280cfc726cc71 ${KDET}/patches/kernel-det/build_det.py /opt/llm/kernel-det/src/build_det.py
ADD --checksum=sha256:b103fbeaf7589b9468471142ad0b30012a076f93d20ba11fc5ff6dcb1ecd32a6 ${KDET}/patches/kernel-det/bindings_det.cpp /opt/llm/kernel-det/src/bindings_det.cpp
ADD --checksum=sha256:19e1d53425ea9a839445722fd1dac1c41727128eebdce84508c1bfb8592afecf ${KDET}/patches/kernel-det/topk_det.cu /opt/llm/kernel-det/src/topk_det.cu
ADD --checksum=sha256:16939700ae389750782ff5c0d5b9caef59aa0ff8b869b64ec94fa72c814910ee ${KDET}/patches/kernel-det/torch_utils.h /opt/llm/kernel-det/src/torch_utils.h
ADD --checksum=sha256:b4ef9ce298d43d6c0e6db9fcca451df20815b2cfe33791919c1ad9c0e84f0ba7 ${KDET}/patches/kernel-det/persistent_topk.cuh /opt/llm/kernel-det/src/persistent_topk.cuh
ADD --checksum=sha256:70905073fe3fa361030bf1cb469b74610766bdfe361419cd7df50af2561322e3 ${KDET}/tools/determinism/qsadet_patch.py /tmp/qsadet_patch.py
RUN cd /opt/llm/kernel-det/src && DET_BUILD_DIR=/opt/llm/kernel-det/build DET_ARCH=${DET_ARCH} python3 build_det.py 2>&1 | tail -2 \
 && cp /opt/llm/kernel-det/build/_C_det.so /opt/llm/kernel-det/_C_det.so \
 && VLLM_QSA_PY=${QSA_OPS} python3 /tmp/qsadet_patch.py && rm /tmp/qsadet_patch.py \
 && python3 -c "import ast; ast.parse(open('${QSA_OPS}').read()); print('qsadet wired OK')"

# On-demand torch.profiler around engine steps (VLLM_STEP_PROFILE=1 +
# touch /tmp/profile_trigger). This vLLM predates VLLM_TORCH_PROFILER_DIR.
COPY src/patch_step_profile.py /tmp/patch_step_profile.py
RUN python3 /tmp/patch_step_profile.py && rm /tmp/patch_step_profile.py

# Per-step prefill metrics (--enable-logging-iteration-details; ITER_DETAILS=1 in the
# launcher), from Saren-Arterius/qwen3.8-Flash-DGX-AutoRound. vllm:prompt_tokens_total
# is credited only when a prefill FINISHES; this adds vllm:scheduled_ctx_tokens_total
# (+ scheduled_iterations_total) fed every engine step, and mutes the stock
# one-line-per-step logger. Watch live prefill tok/s with bench/ppwatch.sh.
COPY src/patch_prefill_metrics.py /tmp/patch_prefill_metrics.py
RUN python3 /tmp/patch_prefill_metrics.py && rm /tmp/patch_prefill_metrics.py

# Private MTP draft head (VLLM_MTP_DRAFT_VOCAB=<ids.npy>, VLLM_MTP_DRAFT_HEAD=int4),
# from upstream blazux/qwen3.8-Flash-DGX 0c6df7e (idea: MiaAI-Lab, reimplemented there),
# adapted for this fork's int8 GPTQ-Marlin lm_head. vLLM shares the target's lm_head
# with the MTP draft, so each draft step reads the whole int8 head (616 MiB); the knobs
# shrink that read. The target verifies every drafted token, so outputs never change —
# only draft acceptance can. DRAFT_VOCAB: score a private 65,536-row slice
# (src/draft_vocab_65536.npy; tools/build_draft_vocab.py rebuilds it over your own
# corpus), every other id -inf. DRAFT_HEAD=int4: a private int4 g128 RTN GPTQ-Marlin
# copy of the full-vocabulary head (vLLM marlin_quantize at first use). Rows are
# dequantized from the checkpoint's GPTQ tensors (VLLM_MTP_DRAFT_VOCAB_CKPT, default
# /model). Inert unless the env is set at runtime — on an image built before this
# section the serve script's DRAFT_* knobs are ignored (the launcher warns).
COPY src/draft_vocab_65536.npy /opt/llm/draft_vocab_65536.npy
COPY src/patch_mtp_draft_vocab.py /tmp/patch_mtp_draft_vocab.py
RUN python3 /tmp/patch_mtp_draft_vocab.py ${MTP_PY} && rm /tmp/patch_mtp_draft_vocab.py
