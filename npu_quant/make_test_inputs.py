"""把测试集前 K 张转成 nash runtime 吃的裸 float32 NCHW 二进制（1x1x3x224x224）。

输出：
  npu_quant/testbin/0000.bin ... (602112 字节/张)
  npu_quant/testbin/meta.json  {rows, targets, k}
预处理与训练/校准完全一致（data/transforms.py 的 preprocess）。
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
from data.utkface import scan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="D:/data/utkface/UTKFace")
    p.add_argument("--test-set", default="npu_quant/test_set.json")
    p.add_argument("--out", default="npu_quant/testbin")
    p.add_argument("--k", type=int, default=200)
    a = p.parse_args()

    ts = json.loads(Path(a.test_set).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.bin"):
        f.unlink()

    by_row = {r["row"]: r for r in scan(a.data)}
    rows, targets = ts["rows"][: a.k], ts["targets"][: a.k]
    for i, (row, tgt) in enumerate(zip(rows, targets)):
        x = preprocess(Image.open(by_row[row]["path"]))
        arr = x.numpy().astype(np.float32)[None, None]  # (1,1,3,224,224)
        assert arr.shape == (1, 1, 3, 224, 224), arr.shape
        arr.tofile(out / f"{i:04d}.bin")
    nbytes = (out / "0000.bin").stat().st_size
    (out / "meta.json").write_text(
        json.dumps({"k": len(rows), "rows": rows, "targets": targets, "bytes_per_sample": nbytes})
    )
    print(f"wrote {len(rows)} bins -> {out}  ({nbytes} bytes each)")


if __name__ == "__main__":
    main()
