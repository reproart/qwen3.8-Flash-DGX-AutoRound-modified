#!/usr/bin/env python3
"""CPU test for the checkpoint-preparation tools (no GPU, no vLLM, no download).

Builds a tiny synthetic checkpoint and checks:
  * quantize_lm_head_int8.py — int8 dequant matches, an all-zero group gives
    zeros (not NaN), a re-run is a no-op, an interrupted run recovers;
  * fp8_convert.py — blockwise fp8 roundtrip, a re-run never re-quantizes and
    never overwrites the .bf16.bak original;
  * strip_ngram_index.py — strips once, a re-run exits 0 and changes nothing;
  * quantize_mtp_experts_int4.py — int4 quantize/dequant roundtrip.

  python3 tools/test_tools_cpu.py          # needs numpy, torch, safetensors
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
A, B = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
QKV = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
NORM = "model.language_model.layers.0.input_layernorm.weight"
NGRAM = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"


def run(tool, *args):
    r = subprocess.run([PY, os.path.join(HERE, tool), *args], capture_output=True, text=True)
    assert r.returncode == 0, f"{tool} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}"
    return r.stdout


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def snapshot(d):
    return {f: sha(os.path.join(d, f)) for f in sorted(os.listdir(d))}


def make_ckpt():
    d = tempfile.mkdtemp()
    g = torch.Generator().manual_seed(0)
    head = torch.randn(256, 384, generator=g).to(torch.bfloat16)
    head[5, 128:256] = 0  # one all-zero (group, out) cell
    save_file({"lm_head.weight": head,
               QKV: torch.randn(256, 256, generator=g).to(torch.bfloat16),
               NORM: torch.ones(256, dtype=torch.bfloat16)}, os.path.join(d, A))
    save_file({NGRAM: torch.zeros(64, 160, dtype=torch.bfloat16)}, os.path.join(d, B))
    json.dump({"metadata": {"total_size": 0},
               "weight_map": {"lm_head.weight": A, QKV: A, NORM: A, NGRAM: B}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    return d, head


def dequant_int8(t):
    qw = t["lm_head.qweight"].numpy().view(np.uint8)            # [in/4, out*4] bytes
    in4, out4 = qw.shape
    out_f = out4 // 4
    q = qw.reshape(in4, out_f, 4).transpose(0, 2, 1).reshape(in4 * 4, out_f)
    sc = t["lm_head.scales"].float().numpy()                    # [groups, out]
    w = (q.astype(np.float32) - 128).reshape(sc.shape[0], -1, out_f) * sc[:, None, :]
    return torch.from_numpy(w.reshape(in4 * 4, out_f).T.copy())


# --- full pipeline, then a re-run of every step ------------------------------
d, head = make_ckpt()
run("quantize_lm_head_int8.py", d)
run("fp8_convert.py", d)
run("strip_ngram_index.py", d)

t = load_file(os.path.join(d, A))
assert "lm_head.weight" not in t
deq = dequant_int8(t)
assert torch.isfinite(deq).all(), "NaN/inf in the int8 head"
assert float(deq[5, 128:256].abs().max()) == 0.0, "all-zero group must dequantize to 0"
rel = float((deq - head.float()).abs().max() / head.float().abs().max())
assert rel < 0.01, f"int8 head error {rel}"
assert t[QKV].dtype == torch.float8_e4m3fn and (QKV[:-7] + ".weight_scale_inv") in t
wq = t[QKV].float().reshape(2, 128, 2, 128) * t[QKV[:-7] + ".weight_scale_inv"][:, None, :, None]
orig_qkv = load_file(os.path.join(d, A + ".bf16.bak"))[QKV].float()
assert float((wq.reshape(256, 256) - orig_qkv).abs().max() / orig_qkv.abs().max()) < 0.1
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
assert NGRAM not in idx["weight_map"] and "lm_head.qweight" in idx["weight_map"]
assert QKV[:-7] + ".weight_scale_inv" in idx["weight_map"]
print("pipeline: OK (int8 head, zero group, fp8 side layer, index)")

before = snapshot(d)
out = run("quantize_lm_head_int8.py", d)
assert "nothing to do" in out, out
out = run("fp8_convert.py", d)
assert "0 converted" in out, out
out = run("strip_ngram_index.py", d)
assert "nothing to do" in out, out
assert snapshot(d) == before, "a re-run changed files"
print("re-run: OK (no-op, backups untouched)")

# --- interrupted lm_head run: shard renamed to .bak, swap never happened -----
d2, _ = make_ckpt()
os.rename(os.path.join(d2, A), os.path.join(d2, A + ".bf16head.bak"))
run("quantize_lm_head_int8.py", d2)
assert "lm_head.qweight" in load_file(os.path.join(d2, A))
assert "lm_head.weight" in load_file(os.path.join(d2, A + ".bf16head.bak"))
print("lm_head recovery (only .bak left): OK")

# --- interrupted lm_head run: shard swapped, index update lost --------------
d3, _ = make_ckpt()
idx_path = os.path.join(d3, "model.safetensors.index.json")
orig_idx = open(idx_path).read()
run("quantize_lm_head_int8.py", d3)
open(idx_path, "w").write(orig_idx)
out = run("quantize_lm_head_int8.py", d3)
assert "updating the index only" in out, out
assert "lm_head.qweight" in json.load(open(idx_path))["weight_map"]
print("lm_head recovery (index lost): OK")
for x in (d, d2, d3):
    shutil.rmtree(x)

# --- int4 MTP expert quantizer ---------------------------------------------
spec = importlib.util.spec_from_file_location(
    "qmtp", os.path.join(HERE, "quantize_mtp_experts_int4.py"))
qmtp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qmtp)
w = torch.randn(256, 512, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
w[:, :128] = 0
qw, qz, sc = qmtp.quantize(w)
back = qmtp.dequant(qw, qz, sc)
assert torch.isfinite(back).all() and float(back[:, :128].abs().max()) == 0.0
rel = float((back - w.float()).abs().max() / w.float().abs().max())
assert rel < 0.15, f"int4 error {rel}"
print(f"int4 MTP quantizer: OK (max rel err {rel:.3f})")
print("ALL OK")
