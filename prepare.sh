#!/usr/bin/env bash
# prepare.sh — build the ready-to-serve checkpoint from Intel's AutoRound
# release. CPU-only (a NAS box is fine); needs python3 with torch +
# safetensors, and the `hf` CLI. One-time, ~30 min + downloads.
#
# You can skip all of this: the finished outputs of exactly this script are
# published at
#   https://huggingface.co/Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN  (default)
#   https://huggingface.co/Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid              (bf16 MTP draft)
#   https://huggingface.co/Saren/Qwen3.8-Flash-Next-ple-table-fp8
# Run this only if you'd rather build (or audit) the artifacts yourself.
#
# What it does, in order:
#   1. Download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound (int4 experts,
#      everything else bf16), then check that every file the index references
#      is actually there: the public Intel repo currently lists 17 shards but
#      serves no model-00002-of-00017.safetensors, so an incomplete mirror must
#      fail here, with file names, rather than inside vLLM at load time.
#      Missing files that hold only the n-gram table are fine (step 4 strips
#      them from the index; the table comes from TABLE_DIR instead).
#      (The upstream fork's script downloaded "…-W4A16-RTN-AutoRound" here;
#      that name answers 401 anonymously — gated or renamed. If you have
#      access to it, substitute it in the hf download line below.)
#   2. quantize_lm_head_int8.py — repack the bf16 lm_head as int8 GPTQ-Marlin
#      (full-range symmetric). Kills a 1.27 GiB bf16 head and its per-token
#      bf16 GEMV; also used by the MTP draft head.
#   3. fp8_convert.py — convert the 300 bf16 side-layer tensors (GDN in/out
#      projections, QSA q/k/v/o, shared expert) to blockwise fp8 e4m3
#      (128x128). Originals kept as .bf16.bak until the cleanup step.
#   4. strip_ngram_index.py — drop the 51B n-gram ("PLE") table from the
#      safetensors index: it is NOT loaded as a weight, it is mmapped from a
#      separate directory at runtime (see the README's PLE mmap patch).
#   5. fetch-ple-table-fp8.sh — download that table's fp8 shards (from
#      Qwen/Qwen3.8-Flash-Next-FP8) into the separate table dir.
#   6. Rewrite config.json's quantization_config for vLLM's GPTQ loader
#      (this vLLM has no auto-round loader; the GPTQ config reads the same
#      packed tensors). The dynamic rules exclude the non-int4 families and
#      flip the head to 8-bit; the original config is kept as
#      config.json.autoround.
#   7. Delete the backups: *.bf16.bak (fp8_convert.py) and *.bf16head.bak
#      (quantize_lm_head_int8.py's shard copy, ~2 GiB incl. the 1.27 GiB bf16
#      head). Rollback = re-download from Intel and re-run; steps 2+3 are
#      deterministic, verified bit-exact.
#   8. quantize_mtp_experts_int4.py — the MTP draft layer's 512 routed experts
#      (bf16, ~4.7 GiB, excluded by Intel) -> int4 g128 RTN in the same GPTQ
#      layout, written as a hardlinked variant <checkpoint-dir>-MTP_int4RTN
#      (no extra space) with the layers.48 exclusion dropped. The default
#      MODEL_DIR; the bf16-draft dir stays usable as an option.
#
# Usage: prepare.sh <checkpoint-dir> <ple-table-dir>
# Then point serve.sh's MODEL_DIR at <checkpoint-dir>-MTP_int4RTN and
# TABLE_DIR at the table dir.
#
# Resumable: if a step fails (network, disk), fix the cause and run the same
# command again. Finished steps are skipped — the download via a marker file
# (<checkpoint-dir>/.prepare-downloaded), everything else because each tool
# detects its own finished work and never re-quantizes or overwrites a backup.
# The download skips shards that hold only the n-gram table (step 4 drops
# them from the index anyway) when the huggingface_hub Python package is
# importable; otherwise it falls back to a plain full `hf download`.
set -euo pipefail
CKPT="${1:?usage: prepare.sh <checkpoint-dir> <ple-table-dir>}"
TABLE="${2:?usage: prepare.sh <checkpoint-dir> <ple-table-dir>}"
cd "$(dirname "$0")"

SRC_REPO=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound

# 1. Download. Once finished, never again: a re-download would compare the
# already-modified shards against the hub and silently restore the originals.
if [ -e "$CKPT/.prepare-downloaded" ]; then
  echo ">> step 1: download already done ($CKPT/.prepare-downloaded), skipping"
else
  hf download "$SRC_REPO" model.safetensors.index.json --local-dir "$CKPT"
  if python3 -c 'import huggingface_hub' 2>/dev/null; then
    python3 - "$SRC_REPO" "$CKPT" <<'EOF'
import json, os, sys
from huggingface_hub import snapshot_download
repo, ckpt = sys.argv[1], sys.argv[2]
wm = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
by_file = {}
for tensor, fname in wm.items():
    by_file.setdefault(fname, []).append(tensor)
table_only = sorted(f for f, ts in by_file.items()
                    if all(".ngram_embedding." in t for t in ts))
print(">> skipping %d table-only shard(s): %s" % (len(table_only), ", ".join(table_only) or "-"))
snapshot_download(repo, local_dir=ckpt, ignore_patterns=table_only)
EOF
  else
    echo ">> huggingface_hub not importable from python3 — downloading the full repo"
    hf download "$SRC_REPO" --local-dir "$CKPT"
  fi
fi

# A download can finish while leaving the checkpoint unusable: the Intel
# release on HF has been seen missing a shard it references. Check here, where
# the cause is obvious, instead of at load time inside vLLM. Missing files that
# hold only the PLE n-gram table are fine — step 4 strips those from the index
# and the table is mmapped from TABLE_DIR instead.
python3 - "$CKPT" <<'EOF'
import json, os, sys
ckpt = sys.argv[1]
idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
by_file = {}
for tensor, fname in idx["weight_map"].items():
    by_file.setdefault(fname, []).append(tensor)
missing, missing_table = [], []
for fname, tensors in sorted(by_file.items()):
    if os.path.exists(os.path.join(ckpt, fname)):
        continue
    (missing_table if all(".ngram_embedding." in t for t in tensors)
     else missing).append(fname)
if missing:
    sys.exit(">> ERROR: checkpoint is incomplete — %d shard(s) referenced by the "
             "index are missing:\n   %s\n   Re-download the source release (the "
             "public Intel repo currently lacks one shard) or use the prebuilt "
             "artifacts from the README Quickstart."
             % (len(missing), "\n   ".join(missing)))
print(">> checkpoint complete: %d files referenced by the index, all present%s"
      % (len(by_file),
         "" if not missing_table else " (%d table-only shard(s) absent, stripped in step 4: %s)"
         % (len(missing_table), ", ".join(missing_table))))
EOF
touch "$CKPT/.prepare-downloaded"

tools/quantize_lm_head_int8.py "$CKPT"
tools/fp8_convert.py "$CKPT"
tools/strip_ngram_index.py "$CKPT"
tools/fetch-ple-table-fp8.sh "$TABLE"

python3 - "$CKPT" <<'EOF'
import json, os, shutil, sys
ckpt = sys.argv[1]
cfg = json.load(open(f"{ckpt}/config.json"))
if cfg.get("quantization_config", {}).get("quant_method") == "gptq":
    print(">> quantization_config already rewritten, skipping")
    sys.exit(0)
# Never overwrite the backup: on a re-run it is the only AutoRound original.
if not os.path.exists(f"{ckpt}/config.json.autoround"):
    shutil.copy2(f"{ckpt}/config.json", f"{ckpt}/config.json.autoround")
cfg["quantization_config"] = {
    "quant_method": "gptq",
    "bits": 4,
    "group_size": 128,
    "desc_act": False,
    "sym": True,
    "lm_head": True,
    "dynamic": {
        "+:.*lm_head$": {"bits": 8},
        "-:.*linear_attn.*": {},
        "-:.*self_attn.*": {},
        "-:.*hyper_connection.*": {},
        "-:.*visual.*": {},
        "-:.*shared_expert.*": {},
        "-:.*\\.ple\\..*": {},
        "-:.*embed.*": {},
        "-:.*fc_hidden.*": {},
        "-:.*layers\\.48\\..*": {},
        "-:.*\\.gate$": {},
    },
}
json.dump(cfg, open(f"{ckpt}/config.json", "w"), indent=2)
print(">> quantization_config rewritten (original: config.json.autoround)")
EOF

rm -f "$CKPT"/*.bf16.bak "$CKPT"/*.bf16head.bak

# 8. int4 MTP draft experts -> hardlinked variant (the default MODEL_DIR)
tools/quantize_mtp_experts_int4.py "$CKPT" "${CKPT}-MTP_int4RTN"
echo ">> done. MODEL_DIR=${CKPT}-MTP_int4RTN (bf16-draft option: $CKPT) TABLE_DIR=$TABLE"
