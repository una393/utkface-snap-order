"""UTKFace 预缩放成 uint8 memmap，消掉重复 JPEG 解码。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
SIZE = (224, 224)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="D:/data/utkface/UTKFace")
    p.add_argument("--out", default="data/utkface_raw224.npy")
    a = p.parse_args()
    files = sorted(Path(a.root).glob("*.jpg"))
    mm = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.uint8,
                                   shape=(len(files), SIZE[1], SIZE[0], 3))
    for i, f in enumerate(files):
        mm[i] = np.asarray(Image.open(f).convert("RGB").resize(SIZE, Image.BILINEAR))
        if (i + 1) % 5000 == 0:
            print(f"{i + 1}/{len(files)}", flush=True)
    mm.flush()
    print(f"{len(files)} -> {a.out} ({len(files)*224*224*3/2**30:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
