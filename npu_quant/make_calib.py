"""生成 hb_compile 校准数据 + 无重叠测试集索引。

校准集：split_random(recs, 0.15, seed=0)[1] 的前 100 张（与训练验证集同口径，seed 固定 0）。
测试集：同一 val 划分的 [800:1600]，与校准集不相交，供 MAE 用。
预处理直接复用 data/transforms.py：RGB、Resize(224,224,BILINEAR)、/255、ImageNet 归一化。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image

from data.transforms import preprocess
from data.utkface import scan, split_random


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="D:/data/utkface/UTKFace")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calib", type=int, default=100)
    p.add_argument("--test-start", type=int, default=800)
    p.add_argument("--test-n", type=int, default=800)
    p.add_argument("--out", default="npu_quant")
    a = p.parse_args()

    out = Path(a.out)
    cdir = out / "calib"
    cdir.mkdir(parents=True, exist_ok=True)
    for f in cdir.glob("*.npy"):
        f.unlink()

    recs = scan(a.data)
    _, val = split_random(recs, 0.15, a.seed)
    print(f"records {len(recs)} | val {len(val)} | seed {a.seed}")

    for i, r in enumerate(val[: a.calib]):
        t = preprocess(Image.open(r["path"]))
        assert t.shape == (3, 224, 224), t.shape
        np.save(cdir / f"{i:04d}.npy", t.numpy().astype(np.float32))
    print(f"calib: {len(list(cdir.glob('*.npy')))} npy -> {cdir}  (shape 3x224x224 float32)")

    test = val[a.test_start: a.test_start + a.test_n]
    (out / "test_set.json").write_text(json.dumps(
        {"seed": a.seed, "slice": [a.test_start, a.test_start + len(test)],
         "rows": [int(r["row"]) for r in test],
         "targets": [float(r["target"]) for r in test]}, ensure_ascii=False))
    print(f"test set: n={len(test)} rows[{a.test_start}:{a.test_start+len(test)}] -> {out/'test_set.json'}")


if __name__ == "__main__":
    main()
