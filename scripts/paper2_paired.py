"""逐图配对统计：8 bit 导出代价、间隔效应、顺序效应、影子效应。

读 quant_runs/perimage/*.npz（deploy_table.py --dump 产出，键 pred/target/rows），
配对只在同一批 400 张图像内进行（rows 必须相同），跨 run 的均值以 run 为单位。
输出 quant_runs/paired_report.md。
"""
import glob, io, json, os, re
from pathlib import Path
import numpy as np

ROOT = Path("quant_runs/perimage")
ALPHA = 0.05
SPLIT_SEED_SINCE = 1793111850   # 最早记录 split_seed 的 run；更晚的 run 缺该键即视为漏记
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}


def tcrit(df):
    return T975.get(df, 1.96)


def load():
    runs = {}
    for p in sorted(ROOT.glob("*__*.npz")):
        name, bits = p.stem.rsplit("__", 1)
        d = np.load(p)
        rows = tuple(int(x) for x in d["rows"])
        err = np.abs(d["pred"] - d["target"])
        runs.setdefault(name, {})[bits] = (rows, err)
    return runs


def meta(name):
    rj = Path("results") / name / "result.json"
    a = json.loads(rj.read_text(encoding="utf-8"))["args"]
    if "ema_decay" not in a:
        raise KeyError("%s: result.json 的 args 缺 ema_decay，无法判定与主配置 γ=0.99 是否同组" % name)
    if "split_seed" in a:
        sp = a["split_seed"]
        split = a.get("seed", 0) if sp is None or int(sp) < 0 else int(sp)
    elif int(name.rsplit("-", 1)[-1]) < SPLIT_SEED_SINCE:
        split = a.get("seed", 0)          # --split-seed 之前的 run：划分硬编码成 seed
    else:
        raise KeyError("%s: result.json 的 args 缺 split_seed（--split-seed 之后产生的 run 必须记录）"
                       % name)
    return {
        "mode": a["ema_mode"],
        "N": a.get("snap_every", 1),
        "seed": a.get("seed", 0),
        "decay": a["ema_decay"],
        "split": split,
    }


def main_run(by, mode, N, seed):
    """主配置 = γ=0.99 且 split_seed == seed（split 进键，别只看 seed）；同键多行就是无法区分。"""
    cand = [n for n, m in by.get((mode, N, 0.99, seed), []) if m["seed"] == seed]
    if len(cand) > 1:
        raise RuntimeError("同键多行无法区分：%s N=%s seed=%s -> %s" % (mode, N, seed, cand))
    return cand[0] if cand else None


def paired(e1, r1, e2, r2):
    assert r1 == r2, "不是同一批图像，不能配对"
    d = e1 - e2
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / n ** 0.5 if n > 1 else float("nan")
    half = tcrit(n - 1) * se if n > 1 else float("nan")
    return {"n": n, "mean": m, "se": se, "lo": m - half, "hi": m + half, "d": d}


def fmt(x):
    return "n/a" if x != x else ("%+.5f" % x)


def main():
    runs = load()
    out = ["# 逐图配对统计（quant_runs/perimage）", ""]
    out.append("配对统计均在每批 400 张部署评估图像内逐图计算；"
               "delta 定义为 left - right，负值表示 left 的逐图误差更小。")
    out.append("")
    by = {}
    for name in runs:
        m = meta(name)
        by.setdefault((m["mode"], m["N"], m["decay"], m["split"]), []).append((name, m))

    # 1. 8 bit 导出代价，逐图配对
    out += ["## 1. 8 bit 导出代价（逐图配对，fp32 - 8bit）", "",
            "| run | mode | N | seed | fp32 MAE | 8bit MAE | 配对 Δ | SE | 95% CI |",
            "|---|---|---|---|---|---|---|---|---|"]
    costs = {}
    for name in sorted(runs, key=lambda n: (meta(n)["mode"], meta(n)["N"], meta(n)["seed"])):
        if "fp32" not in runs[name] or "8bit" not in runs[name]:
            continue
        m = meta(name)
        r1, e1 = runs[name]["fp32"]
        r2, e2 = runs[name]["8bit"]
        p = paired(e1, r1, e2, r2)
        costs.setdefault((m["N"],), []).append(abs(p["mean"]))
        out.append("| %s | %s | %s | %s | %.4f | %.4f | %s | %.5f | [%s, %s] |" % (
            name, m["mode"], m["N"], m["seed"], e1.mean(), e2.mean(),
            fmt(p["mean"]), p["se"], fmt(p["lo"]), fmt(p["hi"])))
    out += ["", "按吸附间隔取 |Δ| 的上界：", ""]
    for N, v in sorted(costs.items(), key=lambda kv: kv[0][0]):
        out.append("- N=%s：max |Δ| = %.5f（%d 个 run）" % (N[0], max(v), len(v)))
    out.append("")

    # 2. 间隔效应（同一协议、seed 0，配对）
    out += ["## 2. 间隔效应（同协议、同 seed，间隔两两配对）", "",
            "| 协议 | 比较 | 配对 Δ | SE | 95% CI | n |", "|---|---|---|---|---|---|"]
    for mode in ("none", "before_snap", "after_snap"):
        for N1, N2 in ((1, 5), (5, 20), (20, 0), (1, 20)):
            a = main_run(by, mode, N1, 0)
            b = main_run(by, mode, N2, 0)
            if a is None or b is None:
                continue
            ra, ea = runs[a]["fp32"]
            rb, eb = runs[b]["fp32"]
            p = paired(ea, ra, eb, rb)
            out.append("| %s | N=%s → N=%s | %s | %.5f | [%s, %s] | %d |" % (
                mode, N1, N2, fmt(p["mean"]), p["se"], fmt(p["lo"]), fmt(p["hi"]), p["n"]))
    out.append("")

    # 3. 顺序效应（同 N、同 seed，before 对 after 的导出权重）
    out += ["## 3. 顺序效应（同间隔、同 seed，before_snap 对 after_snap）", "",
            "| N | seed | 配对 Δ | SE | 95% CI |", "|---|---|---|---|---|"]
    for N in (1, 5, 20, 0):
        for seed in (0, 1, 2):
            a = main_run(by, "before_snap", N, seed)
            b = main_run(by, "after_snap", N, seed)
            if a is None or b is None:
                continue
            ra, ea = runs[a]["fp32"]
            rb, eb = runs[b]["fp32"]
            p = paired(ea, ra, eb, rb)
            out.append("| %s | %s | %s | %.5f | [%s, %s] |" % (
                N, seed, fmt(p["mean"]), p["se"], fmt(p["lo"]), fmt(p["hi"])))
    out.append("")

    # 4. 影子有无（none 对两个 EMA 协议，N=20 seed 0）
    out += ["## 4. 影子效应（none 对 EMA 协议，N=20、seed 0）", "",
            "| 比较 | 配对 Δ | SE | 95% CI |", "|---|---|---|---|"]
    ref = main_run(by, "none", 20, 0)
    if ref:
        r0, e0 = runs[ref]["fp32"]
        for mode in ("before_snap", "after_snap"):
            n = main_run(by, mode, 20, 0)
            if n is not None:
                r1, e1 = runs[n]["fp32"]
                p = paired(e1, r1, e0, r0)
                out.append("| %s 对 none | %s | %.5f | [%s, %s] |" % (
                    mode, fmt(p["mean"]), p["se"], fmt(p["lo"]), fmt(p["hi"])))
    out.append("")

    # 5. 各量化位宽下的逐图配对（fp32 对 8/6/4）
    out += ["## 5. 位宽代价（逐图配对，fp32 对各位宽）", "",
            "| run | fp32 | 8bit | 6bit | 4bit | Δ8 | Δ6 | Δ4 |", "|---|---|---|---|---|---|---|---|"]
    for name in sorted(runs, key=lambda n: (meta(n)["N"], meta(n)["mode"], meta(n)["seed"])):
        row = [name]
        if "fp32" not in runs[name]:
            continue
        r0, e0 = runs[name]["fp32"]
        row.append("%.4f" % e0.mean())
        for bits in ("8bit", "6bit", "4bit"):
            if bits in runs[name]:
                r1, e1 = runs[name][bits]
                row.append("%.4f" % e1.mean())
            else:
                row.append("n/a")
        for bits in ("8bit", "6bit", "4bit"):
            if bits in runs[name]:
                r1, e1 = runs[name][bits]
                row.append(fmt(paired(e0, r0, e1, r1)["mean"]))
            else:
                row.append("n/a")
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    Path("quant_runs/paired_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("runs=%d wrote quant_runs/paired_report.md" % len(runs))


if __name__ == "__main__":
    main()
