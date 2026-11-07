"""从 hb_compile 的输出里抽逐节点余弦相似度，生成 CSV 与统计摘要。

数据来源：<workdir>/<tag>_quant_info.json，每条记录对应一个量化节点，字段
  type, inputs, outputs, calibrated_outputs, thresholds, cosine_similarity
cosine_similarity 是量化后输出与 float 输出的余弦（字符串形式的浮点；多输出节点为列表）。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

HEAD_PATTERNS = {
    "relu_qat": ["/head/"],
    "gelu": ["/head/"],
    "bilinear_glu": ["/head/"],
}


def load(tag: str, root: Path) -> dict:
    path = root / tag / f"{tag}_quant_info.json"
    return json.loads(path.read_text())


def as_list(v) -> list:
    return v if isinstance(v, list) else [v]


def rows(tag: str, root: Path) -> list[dict]:
    data = load(tag, root)
    out = []
    for name, rec in data.items():
        cos = as_list(rec.get("cosine_similarity", []))
        cos_f = [float(c) for c in cos if c not in ("", None)]
        out.append(
            {
                "node": name,
                "type": rec.get("type", ""),
                "cos_min": min(cos_f) if cos_f else None,
                "cos_n": len(cos_f),
                "thresholds": rec.get("thresholds", ""),
            }
        )
    return out


def summarize(rs: list[dict]) -> dict:
    vals = sorted(r["cos_min"] for r in rs if r["cos_min"] is not None)
    n = len(vals)
    if n == 0:
        return {"n": 0}
    return {
        "n_nodes": n,
        "min": vals[0],
        "p1": vals[max(0, int(n * 0.01) - 1)],
        "p5": vals[max(0, int(n * 0.05) - 1)],
        "median": vals[n // 2],
        "mean": sum(vals) / n,
        "lt_0.99": sum(1 for v in vals if v < 0.99),
        "lt_0.999": sum(1 for v in vals if v < 0.999),
        "lt_0.9999": sum(1 for v in vals if v < 0.9999),
        "eq_1": sum(1 for v in vals if v >= 0.999999),
    }


def head_rows(rs: list[dict]) -> list[dict]:
    # 控制头作用域：节点名里以 /head/ 开头的那段子图
    return [r for r in rs if any(p in r["node"] for p in ["/head/", "/norm/", "RMSNorm"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--top", type=int, default=15, help="打印最差的 N 个节点")
    args = ap.parse_args()
    root = Path(args.root)

    rs = rows(args.tag, root)
    rs.sort(key=lambda r: (r["cos_min"] is None, r["cos_min"]))
    out_csv = root / f"cosine_{args.tag}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
        w.writeheader()
        w.writerows(rs)

    print(f"[{args.tag}] nodes with cosine: {sum(1 for r in rs if r['cos_min'] is not None)}")
    print("global stats:", json.dumps(summarize(rs), ensure_ascii=False))
    print("\nworst {} nodes:".format(args.top))
    for r in rs[: args.top]:
        print(f"  {r['cos_min']:.6f}  {r['type']:<12} {r['node']}")
    print("\ncontrol-head scope:")
    for r in head_rows(rs):
        print(f"  {r['cos_min']:.6f}  {r['type']:<12} {r['node']}")
    print(f"\nCSV -> {out_csv}")

    # 顺带把编译器给的性能信息抽出来
    perf = root / args.tag / f"{args.tag}.json"
    if perf.exists():
        d = json.loads(perf.read_text())["summary"]
        print("\ncompiler summary.performance:", json.dumps(d.get("performance"), ensure_ascii=False))
        print("compiler summary.model info:", json.dumps(d.get("model info"), ensure_ascii=False))


if __name__ == "__main__":
    main()
