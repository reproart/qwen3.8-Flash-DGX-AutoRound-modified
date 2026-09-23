#!/usr/bin/env python3
"""Repack a bf16 lm_head into 8-bit GPTQ (group 128, full-range symmetric) in
place, inside the shard that holds it. CPU-only, needs numpy + safetensors +
torch. This is what turns the Intel W4A16 checkpoint's bf16 head into the
uint8b128 head that vLLM's GPTQ-Marlin serves (with `"lm_head": true` and a
`"+:.*lm_head$": {"bits": 8}` dynamic rule in quantization_config).

Layout produced (matches auto_round:auto_gptq 8-bit):
  lm_head.qweight  int32 [in/4, out]   4 uint8 per int32, little-endian in-dim
  lm_head.qzeros   int32 [groups, out/4]  all bytes 127 (zp=128, v1 storage)
  lm_head.scales   f16   [groups, out]

Quantization: per group of 128 in-features, scale = w[argmax|w|] / -128, i.e.
the scale carries the OPPOSITE sign of the largest-magnitude element, so that
element lands exactly on the -128 slot (full-range symmetric: the -128 slot is
usable), q = clamp(round(w/scale) + 128, 0, 255). An all-zero group gets
scale 0 and q = 128 (dequantizes to 0) instead of NaN.

Usage:
  quantize_lm_head_int8.py <checkpoint_dir> [--shard model-00001-of-*.safetensors]
The original shard is kept as <shard>.bf16head.bak; the model index json is
updated. Run with --dry-run to write to <shard>.int8head instead.
Safe to re-run: an already-quantized checkpoint is reported and left alone.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import safe_open, save_file

GROUP = 128
BITS = 8
PACK = 32 // BITS  # 4 uint8 per int32


def quantize(w: torch.Tensor):
    """w: [out, in] bf16 -> (qweight int32 [in/4, out], qzeros, scales f16)."""
    out_f, in_f = w.shape
    assert in_f % GROUP == 0
    groups = in_f // GROUP
    wg = w.to(torch.float32).T.reshape(groups, GROUP, out_f)  # [g, 128, out]
    # Full-range symmetric: the largest-magnitude element of each (group, out)
    # maps to the -128 slot, so the scale carries the OPPOSITE sign of it.
    # Ties (|min| == max) resolve to the positive element.
    wmin = wg.amin(dim=1)
    wmax = wg.amax(dim=1)
    picked = torch.where(-wmin > wmax, wmin, wmax)
    scale = (picked / -128).to(torch.float16)  # [g, out]
    s32 = scale.to(torch.float32).unsqueeze(1)
    s32 = torch.where(s32 == 0, torch.ones_like(s32), s32)  # all-zero group guard
    q = torch.clamp(torch.round(wg / s32) + 128, 0, 255).to(torch.uint8)  # [g, 128, out]
    q = q.reshape(in_f, out_f)
    qweight = (
        q.reshape(in_f // PACK, PACK, out_f)
        .permute(0, 2, 1)
        .contiguous()
        .numpy()
        .view(np.int32)
        .reshape(in_f // PACK, out_f)
    )
    qzeros = np.full((groups, out_f // PACK), 0x7F7F7F7F, dtype=np.int32)
    return (
        torch.from_numpy(qweight),
        torch.from_numpy(qzeros),
        scale.contiguous(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--shard", default=None, help="shard holding lm_head.weight")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    index_path = os.path.join(args.ckpt_dir, "model.safetensors.index.json")
    index = json.load(open(index_path))
    wm = index["weight_map"]
    if "lm_head.weight" not in wm and "lm_head.qweight" in wm:
        print("lm_head already int8 (lm_head.qweight in index) — nothing to do")
        return
    shard_name = args.shard or wm.get("lm_head.weight")
    if shard_name is None:
        sys.exit("neither lm_head.weight nor lm_head.qweight in the index")
    shard_path = os.path.join(args.ckpt_dir, shard_name)

    bak_path = shard_path + ".bf16head.bak"
    # An earlier run interrupted between the rename and the swap leaves only
    # the backup: that IS the original shard, read it from there.
    src_path = shard_path if os.path.exists(shard_path) else bak_path
    tensors, w = {}, None
    with safe_open(src_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k == "lm_head.weight":
                w = f.get_tensor(k)
            else:
                tensors[k] = f.get_tensor(k)
    if w is None and "lm_head.qweight" in tensors and not args.dry_run:
        # Shard already repacked, only the index update was lost: finish it.
        print(f"{shard_name} already holds the int8 head — updating the index only")
    else:
        assert w is not None, f"lm_head.weight not in {shard_name}"
        _repack(args, shard_path, bak_path, tensors, w)
        if args.dry_run:
            return
    del wm["lm_head.weight"]
    for k in ("lm_head.qweight", "lm_head.qzeros", "lm_head.scales"):
        wm[k] = shard_name
    index.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(p) - (8 + int.from_bytes(open(p, "rb").read(8), "little"))
        for p in glob.glob(os.path.join(args.ckpt_dir, "model-*-of-*.safetensors"))
    )
    with open(index_path + ".tmp", "w") as f:
        json.dump(index, f, indent=2)
    os.replace(index_path + ".tmp", index_path)
    print(f"done: {shard_name} repacked, original at {shard_name}.bf16head.bak")


def _repack(args, shard_path, bak_path, tensors, w):
    print(f"quantizing lm_head {tuple(w.shape)} bf16 -> int8 g{GROUP} sym")
    qweight, qzeros, scales = quantize(w)
    tensors["lm_head.qweight"] = qweight
    tensors["lm_head.qzeros"] = qzeros
    tensors["lm_head.scales"] = scales

    if args.dry_run:
        save_file(tensors, shard_path + ".int8head")
        print(f"dry run: wrote {shard_path}.int8head")
        return
    # Write first, then swap: an interrupted run never loses the original.
    save_file(tensors, shard_path + ".int8head.tmp")
    if not os.path.exists(bak_path):
        os.rename(shard_path, bak_path)
    elif os.path.exists(shard_path):
        os.remove(shard_path)  # the .bak from an earlier run is the original
    os.replace(shard_path + ".int8head.tmp", shard_path)


if __name__ == "__main__":
    main()
