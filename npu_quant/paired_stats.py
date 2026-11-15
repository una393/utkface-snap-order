"""逐样本配对统计：量化三口径 vs FP32（含 bilinear_glu）→ npu_quant/out/paired_stats.json / .md"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

OUT = "npu_quant/out"
TAGS = ["relu_qat_s0_N1", "relu_qat_s0_N20", "gelu_s0_N1", "bilinear_glu_s0_N1"]
fp = json.load(open(f"{OUT}/mae_fp32.json"))
fq = json.load(open(f"{OUT}/mae_fakequant.json"))
tgt = np.array(fp["targets"], dtype=np.float64)

rows, res = [], {}
for t in TAGS:
    p32 = np.array(fp["models"][t]["preds"], dtype=np.float64)
    pfq = np.array(fq["models"][t]["preds"], dtype=np.float64)
    path = f"{OUT}/mae_runtime_{t}.json"
    if not os.path.exists(path):
        rows.append((t, None)); continue
    rt = json.load(open(path))
    prt = np.array(rt["preds"], dtype=np.float64)
    n = min(len(p32), len(pfq), len(prt), len(tgt))
    a, b, c, y = p32[:n], pfq[:n], prt[:n], tgt[:n]
    ok = ~np.isnan(c)          # 运行时失败样本记为 NaN，按样本下标剔除
    n_rt = int(ok.sum())
    mae = lambda p, t: float(np.mean(np.abs(p - t)))
    d_fq = b - a
    d_rt = (c - a)[ok]
    d_rt_fq = (c - b)[ok]
    rec = {
        "n": n, "n_rt": n_rt,
        "mae_fp32": mae(a, y), "mae_fq": mae(b, y), "mae_rt": mae(c[ok], y[ok]),
        "dmae_fq": mae(b, y) - mae(a, y), "dmae_rt": mae(c[ok], y[ok]) - mae(a[ok], y[ok]),
        "dmean_fq": float(d_fq.mean()), "dmean_fq_se": float(d_fq.std(ddof=1) / np.sqrt(n)),
        "dmean_rt": float(d_rt.mean()), "dmean_rt_se": float(d_rt.std(ddof=1) / np.sqrt(n_rt)),
        "rt_vs_fq_mean": float(d_rt_fq.mean()), "rt_vs_fq_sd": float(d_rt_fq.std(ddof=1)),
        "rt_vs_fq_p95abs": float(np.percentile(np.abs(d_rt_fq), 95)),
        "rt_vs_fq_maxabs": float(np.abs(d_rt_fq).max()),
        "rt_vs_fq_corr": float(np.corrcoef(b[ok], c[ok])[0, 1]),
    }
    res[t] = rec
    rows.append((t, rec))

L = ["# 逐样本配对统计（量化 - FP32；nash - 伪量化）", "",
     "| tag | n | MAE fp32 | MAE 伪量化 | MAE nash | dMAE 伪量化 | dMAE nash | 偏置 伪量化 (SE) | 偏置 nash (SE) | nash-伪量化 sd / p95 / max / corr |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for t, r in rows:
    if r is None:
        L.append(f"| {t} | - | - | - | 待跑 | - | - | - | - | - |")
        continue
    L.append(f"| {t} | {r['n']} ({r['n_rt']} rt) | {r['mae_fp32']:.6f} | {r['mae_fq']:.6f} | {r['mae_rt']:.6f} | "
             f"{r['dmae_fq']:+.6f} | {r['dmae_rt']:+.6f} | "
             f"{r['dmean_fq']:+.5f} ({r['dmean_fq_se']:.5f}) | {r['dmean_rt']:+.5f} ({r['dmean_rt_se']:.5f}) | "
             f"{r['rt_vs_fq_sd']:.5f} / {r['rt_vs_fq_p95abs']:.4f} / {r['rt_vs_fq_maxabs']:.4f} / {r['rt_vs_fq_corr']:.5f} |")
txt = "\n".join(L) + "\n"
open(f"{OUT}/paired_stats.md", "w", encoding="utf-8").write(txt)
json.dump(res, open(f"{OUT}/paired_stats.json", "w"), indent=1)
print(txt)
