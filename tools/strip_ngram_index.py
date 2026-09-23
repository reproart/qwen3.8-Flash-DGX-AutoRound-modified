#!/usr/bin/env python3
"""Remove the PLE n-gram table entries from a checkpoint's safetensors index,
so vLLM does not try to load the 51B-parameter table from the checkpoint —
the mmap patch serves it from VLLM_PLE_MMAP_DIR instead.

Usage: strip_ngram_index.py <checkpoint_dir>
Backup written to model.safetensors.index.json.with-ngram.bak (never
overwritten by a re-run, so it always holds the unstripped index).
Safe to re-run: an already-stripped index is reported and left alone.
The shard files themselves are untouched (delete them separately if you want
the disk back; nothing references them after this).
"""

import json
import os
import shutil
import sys

d = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: strip_ngram_index.py <checkpoint_dir>")
path = f"{d}/model.safetensors.index.json"
idx = json.load(open(path))
removed = [k for k in idx["weight_map"] if ".ngram_embedding.shard_" in k]
if not removed:
    print("no ngram entries in index — already stripped, nothing to do")
    sys.exit(0)
for k in removed:
    del idx["weight_map"][k]
if not os.path.exists(path + ".with-ngram.bak"):
    shutil.copy(path, path + ".with-ngram.bak")
with open(path + ".tmp", "w") as f:
    json.dump(idx, f)
os.replace(path + ".tmp", path)
print(f"stripped {len(removed)} ngram entries (backup: {path}.with-ngram.bak)")
