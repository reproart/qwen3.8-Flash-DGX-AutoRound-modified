#!/usr/bin/env python3
"""Convert dense bf16 side-layers of the Intel Flash-Next checkpoint to
blockwise FP8-e4m3 (DeepSeek format: fp8 `weight` + fp32 `weight_scale_inv`,
block 128x128). In-place: affected shards are rewritten, originals -> .bf16.bak.

Only converts tensors whose both dims are divisible by 128 (all listed families
qualify); anything else is left untouched and reported.

Safe to re-run (e.g. after an interrupted prepare.sh): tensors that are already
fp8 are skipped, never re-quantized, and an existing .bf16.bak (the true
original) is never overwritten. Each shard is written to a temp file and
renamed into place, and the index is saved after every shard.
"""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: fp8_convert.py <checkpoint_dir>")
BLOCK = 128
FP8_MAX = 448.0  # e4m3 max normal

TARGETS = re.compile(
    r"model\.language_model\.layers\.\d+\.("
    r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.shared_expert\.(gate_proj|up_proj|down_proj)"
    r")\.weight$"
)


def block_quant(w: torch.Tensor):
    out, inn = w.shape
    assert out % BLOCK == 0 and inn % BLOCK == 0, w.shape
    wf = w.float().reshape(out // BLOCK, BLOCK, inn // BLOCK, BLOCK)
    absmax = wf.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    scale = absmax / FP8_MAX
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    # roundtrip check on this tensor
    deq = q.float() * scale
    rel = ((deq - wf).abs().amax() / wf.abs().amax()).item()
    return (
        q.reshape(out, inn),
        scale.squeeze(1).squeeze(-1).contiguous(),  # [out/128, in/128] fp32
        rel,
    )


def _save_index(idx: dict, idx_path: str) -> None:
    tmp = idx_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(idx, f)  # same compact format as always (hash-stable)
    os.replace(tmp, idx_path)


def main():
    idx_path = f"{ROOT}/model.safetensors.index.json"
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    targets = {n: f for n, f in wm.items() if TARGETS.search(n)}
    by_file = {}
    for name, fname in targets.items():
        by_file.setdefault(fname, []).append(name)
    print(f"{len(targets)} tensors across {len(by_file)} shards")

    worst, converted, already = 0.0, 0, 0
    for i, (fname, names) in enumerate(sorted(by_file.items())):
        path = f"{ROOT}/{fname}"
        tensors = load_file(path)
        changed = False
        for name in names:
            scale_name = name.replace(".weight", ".weight_scale_inv")
            w = tensors.pop(name)
            if w.dtype == torch.float8_e4m3fn:
                # Converted by an earlier (possibly interrupted) run: keep it,
                # just make sure the index knows about its scale.
                tensors[name] = w
                if scale_name in tensors:
                    wm[scale_name] = fname
                already += 1
                continue
            if w.shape[0] % BLOCK or w.shape[1] % BLOCK:
                print(f"  SKIP (shape) {name} {tuple(w.shape)}")
                tensors[name] = w
                continue
            q, scale, rel = block_quant(w)
            # Check before anything is written: a bad tensor must not leave a
            # half-converted checkpoint behind.
            assert rel < 0.10, f"fp8 roundtrip error unexpectedly large for {name}: {rel:.4f}"
            worst = max(worst, rel)
            tensors[name] = q
            tensors[scale_name] = scale
            wm[scale_name] = fname
            converted += 1
            changed = True
        if changed:
            tmp = path + ".fp8.tmp"
            save_file(tensors, tmp)  # no metadata: byte-identical to earlier runs
            if os.path.exists(path + ".bf16.bak"):
                os.remove(path)  # the .bak from an earlier run is the original
            else:
                os.rename(path, path + ".bf16.bak")
            os.replace(tmp, path)
        _save_index(idx, idx_path)
        print(f"[{i + 1}/{len(by_file)}] {fname}: {len(names)} tensors"
              f"{'' if changed else ' (already fp8, skipped)'}")
    print(f"done: {converted} converted, {already} already fp8; "
          f"worst per-tensor max rel err: {worst:.4f}")


if __name__ == "__main__":
    sys.exit(main())
