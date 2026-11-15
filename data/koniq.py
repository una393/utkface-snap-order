"""KonIQ-10k 索引：MOS 来自官方打分表，图像按文件名对上。

与 data/utkface.py 同构：排序后逐图建行，row 是 memmap 行号；
split_random 是 random.Random(seed).shuffle 后前 n_val 张做验证。

打分表实际文件名是 koniq10k_scores_and_distributions.tab（DaRUS 的 TSV，
不是官网按钮上的 .csv），列为 image_name / c1..c5 / c_total / MOS / SD /
MOS_zscore；这里只取 image_name 与 MOS，制表符与逗号两种分隔都认。
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

SCORES_NAME = "koniq10k_scores_and_distributions"
SCORES_SUFFIXES = (".tab", ".csv", ".txt")
MOS_SCALE = 100.0


def find_scores(root: str | Path) -> Path:
    """在 root 及其父目录里找 koniq 打分表。"""
    root = Path(root)
    for d in (root, root.parent):
        for suffix in SCORES_SUFFIXES:
            p = d / (SCORES_NAME + suffix)
            if p.is_file():
                return p
    raise FileNotFoundError(
        f"{SCORES_NAME}{{.tab,.csv}} 不在 {root} 或 {root.parent}；"
        f"用 scan(root, scores=...) 显式指定")


def load_scores(scores: str | Path) -> dict[str, float]:
    """{图像文件名 stem: MOS}。表头两列必需：image_name、MOS。"""
    path = Path(scores)
    text = path.read_text(encoding="utf-8-sig")
    header_line = text.split("\n", 1)[0]
    delim = "\t" if "\t" in header_line else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delim))
    if not rows:
        raise ValueError(f"{path} 没有数据行")
    header = set(rows[0])
    missing = {"image_name", "MOS"} - header
    if missing:
        raise ValueError(f"{path} 缺列 {sorted(missing)}；实际列 {sorted(header)}")
    out = {}
    for r in rows:
        name = (r["image_name"] or "").strip()
        mos = (r["MOS"] or "").strip()
        if not name or not mos:
            continue
        out[Path(name).stem] = float(mos)
    return out


def scan(root: str | Path, scores: str | Path | None = None) -> list[dict]:
    """root 下的 *.jpg 逐张建行；没有 MOS 的图像跳过。row/idx = 输出位置。"""
    root = Path(root)
    mos_by_name = load_scores(scores if scores is not None else find_scores(root))
    rows = []
    for p in sorted(root.glob("*.jpg")):
        mos = mos_by_name.get(p.stem)
        if mos is None:
            continue
        rows.append({"path": str(p), "group": "koniq", "idx": len(rows),
                     "mos": mos, "target": mos / MOS_SCALE, "row": len(rows)})
    return rows


def split_random(records: list[dict], val_ratio: float = 0.15, seed: int = 0):
    rs = list(records)
    random.Random(seed).shuffle(rs)
    n_val = int(len(rs) * val_ratio)
    return rs[n_val:], rs[:n_val]
