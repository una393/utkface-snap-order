"""UTKFace 索引：文件名首段是年龄。

切片形状 (1,3,224,224) 与驾驶工作区的 (T,3,224,384) 不同：
这里用 seq=1，实验变量只有控制头与 EMA 时机。
"""
from __future__ import annotations

import random
from pathlib import Path


def scan(root: str | Path) -> list[dict]:
    rows = []
    for i, p in enumerate(sorted(Path(root).glob("*.jpg"))):
        try:
            age = int(p.name.split("_")[0])
        except ValueError:
            continue
        rows.append({"path": str(p), "group": "utk", "idx": i,
                     "age": age, "target": age / 100.0, "row": i})
    return rows


def split_random(records: list[dict], val_ratio: float = 0.15, seed: int = 0):
    rs = list(records)
    random.Random(seed).shuffle(rs)
    n_val = int(len(rs) * val_ratio)
    return rs[n_val:], rs[:n_val]
