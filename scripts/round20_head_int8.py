# -*- coding: utf-8 -*-
"""Round-20 head×int8 paired test: does the per-node cosine ranking predict int8 cost?

Reads only existing artifacts under npu_quant/out (no retraining, no recompilation):
  mae_fp32.json            per-image FP32 ONNX predictions on the 200-image nash test set
  mae_runtime_<tag>.json   per-image nash x86 runtime predictions for the same 200 images

Cost of quantisation for head h on image i:  d_i = |nash_i - y_i| - |fp32_i - y_i|
(positive = int8 worse).  Heads are compared paired, on the images where both ran.
"""
import json, math, os
import numpy as np
from scipy import stats

OUT = "npu_quant/out"
MD = "quant_runs/head_int8_report.md"
JS = "quant_runs/head_int8_report.json"
HEADS = ["relu_qat_s0_N1", "gelu_s0_N1", "bilinear_glu_s0_N1"]

def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)

fp = load("mae_fp32.json")
y = np.asarray(fp["targets"], dtype=np.float64)

cost = {}
for h in HEADS:
    rt = load("mae_runtime_%s.json" % h)
    pn = np.asarray([np.nan if x is None else x for x in rt["preds"]], dtype=np.float64)
    pf = np.asarray(fp["models"][h]["preds"], dtype=np.float64)
    e_nash = np.abs(pn - y)
    e_fp32 = np.abs(pf - y)
    cost[h] = e_nash - e_fp32
    ok = np.isfinite(cost[h])
    print("%-22s n_valid=%3d  mean_d=%+.6f  fp32_mae=%.6f  nash_mae=%.6f  dmae_report=%+.6f"
          % (h, ok.sum(), cost[h][ok].mean(), e_fp32[ok].mean(), e_nash[ok].mean(),
             rt["mae"] - fp["models"][h]["mae"]))

def paired(a, b):
    m = np.isfinite(cost[a]) & np.isfinite(cost[b])
    d = cost[a][m] - cost[b][m]
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    lo, hi = mean - tcrit * se, mean + tcrit * se
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(10000, n))
    boots = d[idx].mean(axis=1)
    blo, bhi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    t_stat = mean / se if se > 0 else 0.0
    p = float(2 * stats.t.sf(abs(t_stat), n - 1))
    wins = int((d < 0).sum())
    zeros = int((d == 0).sum())
    sign_p = float(stats.binomtest(wins, n - zeros, 0.5).pvalue) if n - zeros > 0 else 1.0
    return dict(a=a, b=b, n=int(n), mean=mean, sd=sd, se=se, t=t_stat,
                ci=[lo, hi], boot_ci=[blo, bhi], p=p,
                frac_a_cheaper=wins / max(n - zeros, 1), zeros=zeros, sign_p=sign_p,
                mean_abs_pred_gap=float(np.abs(cost[a][m] + np.abs(np.asarray(fp["models"][a]["preds"])[m] - y[m])
                                              - (np.abs(np.asarray(fp["models"][b]["preds"])[m] - y[m]))).mean()))

pairs = [paired("relu_qat_s0_N1", "gelu_s0_N1"),
         paired("relu_qat_s0_N1", "bilinear_glu_s0_N1"),
         paired("gelu_s0_N1", "bilinear_glu_s0_N1")]

cos = {"relu_qat_s0_N1": 0.997010, "gelu_s0_N1": 0.990574, "bilinear_glu_s0_N1": 0.983052}

lines = ["# 头部 × int8：余弦名次能否预测部署代价（round20）", "",
         "数据：npu_quant/out 既有产物，同一批 200 张（npu_quant/test_set.json，seed 0 的 val[800:1000]）。",
         "单图代价 d_i = |nash_i − y_i| − |fp32_i − y_i|（正= int8 更差）。全部配对，n 为两侧都跑通的图数。", "",
         "## 1 各头的量化代价", "", "| 头 | n | 平均 d | FP32 MAE | nash MAE |", "|---|---|---|---|---|"]
for h in HEADS:
    ok = np.isfinite(cost[h])
    rt = load("mae_runtime_%s.json" % h)
    e_fp32 = np.abs(np.asarray(fp["models"][h]["preds"], dtype=np.float64) - y)
    pn = np.asarray([np.nan if x is None else x for x in rt["preds"]], dtype=np.float64)
    e_nash = np.abs(pn - y)
    lines.append("| %s | %d | %+.6f | %.6f | %.6f |" % (h, ok.sum(), cost[h][ok].mean(), e_fp32[ok].mean(), e_nash[ok].mean()))

lines += ["", "## 2 头对头的配对检验（差值 = d_A − d_B，负 = A 的量化代价更小）", "",
          "| A vs B | n | 均值差 | t 95% CI (df=n−1) | bootstrap 95% CI | p | A 更省的比例 |",
          "|---|---|---|---|---|---|---|"]
for r in pairs:
    lines.append("| %s vs %s | %d | %+.6f | [%+.6f, %+.6f] | [%+.6f, %+.6f] | %.3f | %.1f%% |"
                 % (r["a"], r["b"], r["n"], r["mean"], r["ci"][0], r["ci"][1],
                    r["boot_ci"][0], r["boot_ci"][1], r["p"], 100 * r["frac_a_cheaper"]))

lines += ["", "## 3 名次对照", "",
          "| 头 | /head/ 余弦 min | int8 平均代价 d | 名次（余弦优→劣） | 名次（代价低→高） |",
          "|---|---|---|---|---|"]
order_cos = sorted(HEADS, key=lambda h: -cos[h])
order_cost = sorted(HEADS, key=lambda h: cost[h][np.isfinite(cost[h])].mean())
for h in order_cos:
    lines.append("| %s | %.6f | %+.6f | %d | %d |"
                 % (h, cos[h], cost[h][np.isfinite(cost[h])].mean(),
                    order_cos.index(h) + 1, order_cost.index(h) + 1))
same = order_cos == order_cost
lines += ["", "余弦名次与 int8 代价名次 **%s**。" % ("一致" if same else "相反"),
          "", "bootstrap：10,000 次，numpy default_rng(0)。t 区间为双侧 95%。", ""]

with open(MD, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
with open(JS, "w", encoding="utf-8") as f:
    json.dump(dict(cos=cos, cost_mean={h: float(cost[h][np.isfinite(cost[h])].mean()) for h in HEADS},
                   pairs=pairs, order_cos=order_cos, order_cost=order_cost, same_order=bool(same)),
              f, ensure_ascii=False, indent=2)
print("WROTE", MD, JS, "same_order=", same)
