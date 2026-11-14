"""两条外推轴的 run 层统计：逐 seed 配对的端点对比 N=1 − N=∞。

    python scripts/round20_ext_runlevel.py --axis data   # KonIQ-10k（数据轴）
    python scripts/round20_ext_runlevel.py --axis arch   # ResNet-18（架构轴）

输入是 scripts/ext_deploy.py 写的两个 csv（first400 / heldout，每行一个 run），输出
quant_runs/{data,arch}_runlevel.{json,md}。

配对单位是 seed：同一 seed、同一协议下 MAE(N=1) − MAE(N=∞)；差值在 seed 上取均值与
t 95 % 区间（df = n − 1）。两种协议在同一 seed 上不是独立重复，故主检验按协议分别做
n = 8，不把 2×8 个格子并成一个 n = 16 的检验；合并只作描述。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BITS = ["mae_p32", "mae_8", "mae_6", "mae_4"]
BITS_NAME = {"mae_p32": "FP32", "mae_8": "8 bit", "mae_6": "6 bit", "mae_4": "4 bit"}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}

AXES = {
    "data": ("KonIQ-10k", "quant_runs/ext_deploy_koniq.csv", "quant_runs/ext_deploy_koniq_heldout.csv"),
    "arch": ("ResNet-18", "quant_runs/ext_deploy_arch.csv", "quant_runs/ext_deploy_arch_heldout.csv"),
}


def load(path: str) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["N"] = int(r["N"]) if r["N"].isdigit() else -1
        for b in BITS:
            r[b] = float(r[b])
    return rows


def contrast(rows: list[dict], mode: str, bit: str) -> dict:
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["seed"], r["mode"]), {})[r["N"]] = r
    seeds = sorted({s for (s, m) in cells if m == mode})
    d, used = [], []
    for s in seeds:
        pair = cells.get((s, mode), {})
        if 1 in pair and 0 in pair:
            d.append(pair[1][bit] - pair[0][bit])
            used.append(s)
    n = len(d)
    if n < 2:
        return {"mode": mode, "bit": BITS_NAME[bit], "n": n, "d": [round(x, 6) for x in d]}
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    half = T975[n - 1] * se
    pos = sum(1 for x in d if x > 0)
    return {"mode": mode, "bit": BITS_NAME[bit], "n": n, "seeds": used,
            "d": [round(x, 6) for x in d], "mean": round(mean, 6), "sd": round(sd, 6),
            "se": round(se, 6), "ci": [round(mean - half, 6), round(mean + half, 6)],
            "positive": pos, "negative": n - pos}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=sorted(AXES), required=True)
    a = p.parse_args()
    name, first_csv, held_csv = AXES[a.axis]

    report = {"axis": a.axis, "label": name,
              "notes": "paired unit = seed; d = MAE(N=1) - MAE(N=inf); t interval df = n-1"}
    for subset, path in (("first400", first_csv), ("heldout", held_csv)):
        rows = load(path)
        report[subset] = {"rows": len(rows), "seeds": sorted({r["seed"] for r in rows})}
        for bit in BITS:
            for mode in ("none", "after_snap"):
                report[subset][f"{bit}:{mode}"] = contrast(rows, mode, bit)
    out_json = Path("quant_runs/%s_runlevel.json" % a.axis)
    out_md = Path("quant_runs/%s_runlevel.md" % a.axis)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# %s（%s 轴）run 层端点对比" % (name, a.axis), "",
             "d = MAE(N=1) − MAE(N=∞)；配对单位 = seed；区间为 t 95 %（df = n − 1）。", ""]
    for subset in ("first400", "heldout"):
        lines += ["## %s（%d runs）" % (subset, report[subset]["rows"]), "",
                  "| bit | 协议 | n | 均值 | sd | 95 % CI | 正/负 |", "|---|---|---|---|---|---|---|"]
        for bit in BITS:
            for mode in ("none", "after_snap"):
                c = report[subset][f"{bit}:{mode}"]
                if c["n"] < 2:
                    lines.append("| %s | %s | %d | — | — | — | — |" % (BITS_NAME[bit], mode, c["n"]))
                else:
                    lines.append("| %s | %s | %d | %+.6f | %.6f | [%+.6f, %+.6f] | %d/%d |"
                                 % (BITS_NAME[bit], mode, c["n"], c["mean"], c["sd"],
                                    c["ci"][0], c["ci"][1], c["positive"], c["negative"]))
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for subset in ("first400", "heldout"):
        c = report[subset]["mae_p32:none"]
        print("%s %s n=%d mean=%s ci=%s %s/%s" % (a.axis, subset, c["n"], c.get("mean"),
                                                 c.get("ci"), c.get("positive"), c.get("negative")))


if __name__ == "__main__":
    main()
