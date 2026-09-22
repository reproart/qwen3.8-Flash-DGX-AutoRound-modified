#!/usr/bin/env bash
# Download just the PLE n-gram table shards from the official FP8 checkpoint
# (Qwen/Qwen3.8-Flash-Next-FP8, shards 5-37 hold the table, ~52 GB) into a
# standalone dir for VLLM_PLE_MMAP_DIR. The fp8 table halves the table's read
# volume vs bf16 (~24 GiB less per cold 262k prefill) with no measurable
# quality change (the PLE rows feed a dequant that runs anyway).
#
#   tools/fetch-ple-table-fp8.sh /path/to/ple-table-fp8
set -euo pipefail
DST="${1:?usage: fetch-ple-table-fp8.sh <dst_dir>}"
FILES=()
for i in $(seq -w 5 37); do
  FILES+=("model-000${i}-of-00131.safetensors")
done
hf download Qwen/Qwen3.8-Flash-Next-FP8 "${FILES[@]}" --local-dir "$DST"

# Sanity-check the result: if upstream ever renames its shards, hf download
# leaves an empty dir and the failure would only surface at server boot
# ("PLE mmap: no shard tensors").
N=$(find "$DST" -name 'model-*-of-*.safetensors' | wc -l)
if [ "$N" -ne 33 ]; then
  echo "ERROR: expected 33 table shards in $DST, found $N — upstream layout changed?" >&2
  exit 1
fi
echo "done: $(du -sh "$DST" | cut -f1) in $DST ($N shards)"
echo "optional: tools/strip_ple_table.py $DST <lean_dst> to drop non-table tensors"
