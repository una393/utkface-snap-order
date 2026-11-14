"""把 hbdk 的 quant_info.json 转成逐节点余弦表。

json 结构随版本变化，parse() 用容错方式抽 (name, calibrated_cosine, quantized_cosine)；
跑之前先打印前几条确认字段对上，不要凭印象写死。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _walk(node, name=""):
    if isinstance(node, dict):
        if {"calibrated_cosine", "quantized_cosine"} & set(node):
            yield (name, node.get("calibrated_cosine"), node.get("quantized_cosine"))
            return
        for k, v in node.items():
            yield from _walk(v, f"{name}/{k}" if name else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{name}[{i}]")


def parse(path: str | Path) -> list[tuple[str, float | None, float | None]]:
    data = json.loads(Path(path).read_text())
    return list(_walk(data))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quant-info", required=True)
    p.add_argument("--out", default="results/nodes.csv")
    args = p.parse_args()
    rows = parse(args.quant_info)
    rows.sort(key=lambda r: (r[2] if r[2] is not None else 2.0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "calibrated_cosine", "quantized_cosine"])
        w.writerows(rows)
    print(f"{len(rows)} nodes -> {out}; worst 10:")
    for r in rows[:10]:
        print(f"  {r[2] if r[2] is not None else float('nan'):+.4f}  {r[0]}")


if __name__ == "__main__":
    main()
