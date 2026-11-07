
import json, math, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
base = r"C:\Users\24399\Desktop\paper-utkface\npu_quant\out"
fp = json.load(open(base + r"\mae_fp32.json", encoding="utf-8"))
fq = json.load(open(base + r"\mae_fakequant.json", encoding="utf-8"))
tgt = np.array(fp["targets"], dtype=np.float64)

def paired(a, b, m):
    d = np.abs(a[m] - tgt[m]) - np.abs(b[m] - tgt[m])
    o = a[m] - b[m]
    return (float(d.mean()), float(d.std(ddof=1) / math.sqrt(m.sum())),
            float(o.mean()), float(o.std(ddof=1) / math.sqrt(m.sum())), int(m.sum()))

print(f"{'tag':<20}{'n':>4}{'MAE_fp32':>10}{'MAE_fq':>9}{'MAE_nash':>10}{'dMAE(nash-fp)':>16}{'se':>9}{'dOut':>10}{'seOut':>9}")
out = {}
for tag in ("relu_qat_s0_N1", "relu_qat_s0_N20", "gelu_s0_N1", "bilinear_glu_s0_N1"):
    p32 = np.array(fp["models"][tag]["preds"], dtype=np.float64)
    pfq = np.array(fq["models"][tag]["preds"], dtype=np.float64)
    mae32 = float(np.abs(p32 - tgt).mean()); maefq = float(np.abs(pfq - tgt).mean())
    try:
        rt = json.load(open(base + r"\mae_runtime_%s.json" % tag, encoding="utf-8"))
        prt = np.array([np.nan if v is None else v for v in rt["preds"]], dtype=np.float64)
        m = ~np.isnan(prt)
        dm, se, om, ose, n = paired(prt, p32, m)
        maert = float(np.abs(prt[m] - tgt[m]).mean())
        print(f"{tag:<20}{n:>4}{mae32:>10.4f}{maefq:>9.4f}{maert:>10.4f}{dm:>+16.5f}{se:>9.5f}{om:>+10.5f}{ose:>9.5f}")
        out[tag] = dict(n=n, mae_fp32=mae32, mae_fq=maefq, mae_rt=maert, dmae=dm, dmae_se=se, dout=om, dout_se=ose)
    except FileNotFoundError:
        dm, se, om, ose, n = paired(pfq, p32, np.ones(len(p32), bool))
        print(f"{tag:<20}{n:>4}{mae32:>10.4f}{maefq:>9.4f}{float('nan'):>10.4f}   (no nash yet)")
        out[tag] = dict(n=n, mae_fp32=mae32, mae_fq=maefq)
json.dump(out, open(base + r"\paired_stats.json", "w"), indent=1)
print("wrote paired_stats.json")
