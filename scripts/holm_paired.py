"""七组 before_snap 对 after_snap 的逐图配对比较，附 Holm 与 Benjamini-Hochberg 校正。

数据来自 quant_runs/perimage/*__fp32.npz（deploy_table.py --dump 的产出，键 pred/target/rows），
run 选择口径来自 quant_runs/deploy_table_v2.csv：ema_decay = 0.99 且 split_seed = seed。

符号与稿子 §5.3 一致：d_i = |before_i - y_i| - |after_i - y_i|，正值表示 before_snap 更差。
七组 = 间隔 1 seed 0；间隔 5 seeds 0/1/2；间隔 20 seeds 0/1/2（同间隔、同 seed，逐图配对）。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ALPHA = 0.05
GROUPS = [(1, 0), (5, 0), (5, 1), (5, 2), (20, 0), (20, 1), (20, 2)]
PERIMAGE = Path("quant_runs/perimage")
TABLE = Path("quant_runs/deploy_table_v2.csv")
OUT = Path("quant_runs/holm_paired.json")


def load_index() -> dict:
    """(mode, N, seed, decay, split_seed) -> run，同键多行直接报错（键重复就无法判定配对对象）。"""
    idx: dict[tuple, str] = {}
    with TABLE.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["mode"], int(r["snap_every"]), int(r["seed"]),
                   float(r["ema_decay"]), int(r["split_seed"]))
            if key in idx:
                raise RuntimeError("deploy_table_v2.csv 同键多行：%s -> %s / %s" % (key, idx[key], r["run"]))
            idx[key] = r["run"]
    return idx


def main_run(idx: dict, mode: str, N: int, seed: int) -> str:
    return idx[(mode, N, seed, 0.99, seed)]


def per_image(run: str) -> tuple[np.ndarray, np.ndarray]:
    p = PERIMAGE / ("%s__fp32.npz" % run)
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p)
    return np.asarray(z["rows"], dtype=np.int64), np.abs(z["pred"] - z["target"]).astype(np.float64)


def paired(before: str, after: str) -> dict:
    rb, eb = per_image(before)
    ra, ea = per_image(after)
    if rb.shape != ra.shape or not np.array_equal(rb, ra):
        raise RuntimeError("不是同一批图像，不能配对：%s / %s" % (before, after))
    d = eb - ea
    n = int(d.size)
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / n ** 0.5
    df = n - 1
    t = m / se
    p = float(2.0 * stats.t.sf(abs(t), df))
    half = float(stats.t.ppf(0.975, df) * se)
    return {"n": n, "mean": m, "sd": sd, "se": se, "t": t, "df": df, "p": p,
            "ci_lo": m - half, "ci_hi": m + half,
            "mae_before": float(eb.mean()), "mae_after": float(ea.mean())}


def holm(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * float(p[i]))
        adj[i] = min(1.0, running)
    return adj


def bh(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m)
    running = 1.0
    for k in range(m - 1, -1, -1):
        i = order[k]
        running = min(running, float(p[i]) * m / (k + 1))
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    idx = load_index()
    groups = []
    for N, seed in GROUPS:
        b = main_run(idx, "before_snap", N, seed)
        a = main_run(idx, "after_snap", N, seed)
        s = paired(b, a)
        s.update({"N": N, "seed": seed, "before_run": b, "after_run": a})
        groups.append(s)

    p = np.array([g["p"] for g in groups])
    holm_adj = holm(p)
    bh_adj = bh(p)
    for g, ha, ba in zip(groups, holm_adj, bh_adj):
        g["holm_adj_p"] = float(ha)
        g["holm_reject"] = bool(ha <= ALPHA)
        g["bh_adj_p"] = float(ba)
        g["bh_reject"] = bool(ba <= ALPHA)

    # 参考：同一间隔跨 seed 的二级汇总（稿子 §5.3 的 n = 3 主张就建立在这个层面）
    ref = []
    for N in (5, 20):
        ms = [g["mean"] for g in groups if g["N"] == N]
        if len(ms) < 2:
            continue
        arr = np.array(ms)
        n = len(arr)
        se = float(arr.std(ddof=1) / n ** 0.5)
        mean = float(arr.mean())
        t = mean / se
        df = n - 1
        pv = float(2.0 * stats.t.sf(abs(t), df))
        half = float(stats.t.ppf(0.975, df) * se)
        ref.append({"N": N, "n": n, "mean": mean, "se": se, "t": t, "df": df, "p": pv,
                    "ci_lo": mean - half, "ci_hi": mean + half})
    if ref:
        rp = np.array([r["p"] for r in ref])
        for r, ha, ba in zip(ref, holm(rp), bh(rp)):
            r["holm_adj_p"] = float(ha)
            r["bh_adj_p"] = float(ba)
            r["holm_reject"] = bool(ha <= ALPHA)
            r["bh_reject"] = bool(ba <= ALPHA)

    out = {
        "alpha": ALPHA,
        "m_tests": len(groups),
        "sign_convention": "d_i = |before_snap - y| - |after_snap - y|；正值 = before_snap 更差（与稿子 §5.3 同号）",
        "pairing": "同一 (interval, seed) 的两个 run 在同一批 400 张部署评估图像上逐图配对；rows 必须逐元素相同",
        "source_runs": "quant_runs/deploy_table_v2.csv 的 ema_decay=0.99 且 split_seed=seed 行；逐图预测取自 quant_runs/perimage/*__fp32.npz",
        "groups": groups,
        "reference_interval_aggregate": ref,
        "min_p": float(p.min()),
        "n_groups_p_le_alpha_uncorrected": int((p <= ALPHA).sum()),
        "n_groups_sig_holm": int(sum(g["holm_reject"] for g in groups)),
        "n_groups_sig_bh": int(sum(g["bh_reject"] for g in groups)),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== 七组逐图配对（before - after，正值 = before 更差；n = 400）===")
    print("%-4s %-5s %-32s %10s %9s %10s %7s %11s %11s" % (
        "N", "seed", "before_run", "mean", "se", "t", "p", "holm_adj_p", "bh_adj_p"))
    for g in groups:
        print("%-4d %-5d %-32s %+10.6f %9.6f %10.3f %7.4f %11.4f %11.4f" % (
            g["N"], g["seed"], g["before_run"], g["mean"], g["se"], g["t"], g["p"],
            g["holm_adj_p"], g["bh_adj_p"]))
    print()
    print("=== 逐组 95% CI（df = 399）===")
    for g in groups:
        print("N=%-3d seed=%d  mean=%+.6f  CI=[%+.6f, %+.6f]  p=%.4f  holm=%s (%.4f)  bh=%s (%.4f)" % (
            g["N"], g["seed"], g["mean"], g["ci_lo"], g["ci_hi"], g["p"],
            "sig" if g["holm_reject"] else "ns", g["holm_adj_p"],
            "sig" if g["bh_reject"] else "ns", g["bh_adj_p"]))
    if ref:
        print()
        print("=== 参考：同间隔跨 seed 二级汇总（n = 3）===")
        for r in ref:
            print("N=%-3d n=%d mean=%+.6f se=%.6f CI=[%+.6f, %+.6f] p=%.4f holm_adj=%.4f bh_adj=%.4f" % (
                r["N"], r["n"], r["mean"], r["se"], r["ci_lo"], r["ci_hi"], r["p"],
                r["holm_adj_p"], r["bh_adj_p"]))
    print()
    print("min p = %.4f | uncorrected p<=%.2f: %d/7 | Holm 显著 %d/7 | BH 显著 %d/7" % (
        p.min(), ALPHA, out["n_groups_p_le_alpha_uncorrected"],
        out["n_groups_sig_holm"], out["n_groups_sig_bh"]))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
