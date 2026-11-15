"""CIFAR-10 迁移实验的统计：四条对比的配对 t、Holm(4)、TOST 与 MDE。

口径与主稿一致：run 是观测单位，配对按 seed；主结局是 8 bit 部署权重的 top-1 错误率。
A = live@1 - live@inf（主）；B = live@20 - live@inf；C = ema_after@20 - live@20；
D = ste@1 - live@1。所有差值单位是**百分点**（误差率乘 100）。
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from scripts.topup_table import _betai, p_two_sided

ROOT = Path(__file__).resolve().parent.parent
DEP = ROOT / "quant_runs" / "cifar_deploy.csv"
OUT_JSON = ROOT / "quant_runs" / "cifar_stats.json"
OUT_CSV = ROOT / "quant_runs" / "cifar_stats.csv"

MARGIN = 0.002          # 0.2 个百分点：预注册的最大可忽略退化
SEEDS = list(range(8))


def arm_of(r: dict) -> str:
    if r["quant_mode"] == "ste":
        return "ste@1"
    if r["ema_mode"] == "after_snap":
        return "ema_after@20"
    n = int(r["snap_every"])
    return {1: "live@1", 5: "live@5", 20: "live@20", 0: "live@inf"}[n]


def cells(bits: int) -> dict[tuple[str, int], float]:
    out = {}
    with open(DEP, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["bits"]) != bits:
                continue
            out[(arm_of(r), int(r["seed"]))] = float(r["top1_err"])
    return out


def sf_t(t: float, df: int) -> float:
    """P(T_df > t)。"""
    return 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))


def inv_t(p_upper: float, df: int) -> float:
    """上尾概率 p_upper 对应的 t 分位数（二分，误差 < 1e-9）。"""
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if sf_t(mid, df) > p_upper:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def summarize(diffs: list[float]) -> dict:
    n = len(diffs)
    df = n - 1
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / df)
    se = sd / math.sqrt(n)
    tc = inv_t(0.025, df)
    t = m / se if se else float("inf")
    p = p_two_sided(diffs)
    # TOST：两个单侧都落在 margin 内
    p_tost = max(sf_t((m - MARGIN) / se, df) if se else 0.0,
                 1.0 - sf_t((m + MARGIN) / se, df) if se else 0.0)
    mde = (inv_t(0.025, df) + inv_t(0.20, df)) * se
    return {"n": n, "mean": m, "sd": sd, "se": se, "t": t, "p": p,
            "ci_low": m - tc * se, "ci_high": m + tc * se,
            "pos": sum(1 for d in diffs if d > 0), "neg": sum(1 for d in diffs if d < 0),
            "p_tost": p_tost, "mde": mde}


def holm(rows: list[dict], key: str = "p") -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    m = len(rows)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, rows[i][key] * (m - rank))
        prev = max(prev, adj)
        rows[i][key + "_holm"] = prev


CONTRASTS = [
    ("A", "live@1 - live@inf", ("live@1", "live@inf")),
    ("B", "live@20 - live@inf", ("live@20", "live@inf")),
    ("C", "ema_after@20 - live@20", ("ema_after@20", "live@20")),
    ("D", "ste@1 - live@1", ("ste@1", "live@1")),
]


def build(bits: int) -> list[dict]:
    c = cells(bits)
    rows = []
    for tag, label, (a, b) in CONTRASTS:
        missing = [(arm, s) for arm in (a, b) for s in SEEDS if (arm, s) not in c]
        if missing:
            raise SystemExit(f"bits={bits} 缺 {len(missing)} 个格子，前几个：{missing[:4]}")
        diffs = [c[(a, s)] - c[(b, s)] for s in SEEDS]
        r = {"basis": f"{bits}bit" if bits else "fp32", "contrast": tag, "definition": label,
             **summarize([d * 100 for d in diffs])}
        rows.append(r)
    holm(rows)
    return rows


def main() -> None:
    rows = build(8) + build(0)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["basis", "contrast", "definition", "n", "mean", "sd", "se", "t", "p", "p_holm",
            "ci_low", "ci_high", "pos", "neg", "p_tost", "mde"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    OUT_JSON.write_text(json.dumps({"margin": MARGIN, "rows": rows}, indent=2), encoding="utf-8")
    for r in rows:
        print(f"{r['basis']:5s} {r['contrast']} {r['mean']:+.3f} pp "
              f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}] t={r['t']:+.2f} p={r['p']:.4g} "
              f"holm={r['p_holm']:.4g} signs {r['pos']}+/{r['neg']}- "
              f"tost={r['p_tost']:.3g} mde={r['mde']:.3f}", flush=True)
    print(f"wrote {OUT_CSV} and {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
