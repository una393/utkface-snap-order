
"""Recompute the four main contrasts with the training-side duplicates removed
from the 1,556-image external basis.

Per-image 8-bit deployment errors are cached in quant_runs/leakpred/<run>__8bit.npz
so this script is cheap on the second run.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from data.utkface import scan
from scripts import heldout_eval as he
from scripts import deploy_table as dt
from models.net import DrivingNet

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = argparse.ArgumentParser()
p.add_argument("--bits", choices=["fp32", "8"], default="8")
A = p.parse_args()
SP = 0
CFG = [("none", 0), ("none", 1), ("none", 20), ("after_snap", 20), ("before_snap", 20)]

recs = scan("D:/data/utkface/UTKFace")
arr = np.load("data/utkface_raw224.npy", mmap_mode="r")
first, mon, held, val, mon_names, inter = he.build_views(recs, SP)
assert len(held) == 1556
u8 = he.gather(arr, held)
yt = np.array([float(r["target"]) for r in held])

leak = json.loads(Path("quant_runs/split_leak_held.json").read_text(encoding="utf-8"))
pix = np.array([np.inf if v is None else v for v in leak["pixel_per_image"]])
assert len(pix) == len(held)
mask_near = pix < 4.0
mask_ident = pix < 1.0
print("leak masks: identical %d, near %d of %d" % (mask_ident.sum(), mask_near.sum(), len(held)), flush=True)

tab = list(csv.DictReader(open("quant_runs/heldout_eval_v3.csv", encoding="utf-8")))
ref = {(r["mode"], int(r["N"]), int(r["seed"])): r for r in tab if r["subset"] == "heldout_1556"}

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
mean_t = torch.tensor(dt.MEAN, dtype=torch.float32, device=dev).view(1, 1, 1, 3)
std_t = torch.tensor(dt.STD, dtype=torch.float32, device=dev).view(1, 1, 1, 3)
print("device", dev, flush=True)

out = Path("quant_runs/leakpred"); out.mkdir(parents=True, exist_ok=True)
errs, checks = {}, []
t0 = time.time()
for mode, N in CFG:
    for seed in range(8):
        run = ref[(mode, N, seed)]["run"]
        p = out / ("%s__%s.npz" % (run, A.bits))
        if p.exists():
            e = np.load(p)["err"]
        else:
            net = DrivingNet(head="relu_qat", grid=(14, 14), seq=1).to(dev)
            ck = torch.load(Path("results") / run / "best.pth", map_location="cpu")
            net.load_state_dict(ck[he.weight_key(mode)], strict=False)
            net.eval()
            if A.bits == "8":
                dt.qdq_(net, 8)
            pred = he.forward_all(net, u8, mean_t, std_t, dev, 64)
            e = np.abs(pred - yt)
            np.savez_compressed(p, err=e)
            del ck
        want = float(ref[(mode, N, seed)]["mae_8" if A.bits == "8" else "mae_fp32"])
        checks.append((run, float(e.mean()), want))
        errs[(mode, N, seed)] = e
        print("%-46s mae %.6f vs csv %.6f %s" % (run, e.mean(), want,
              "OK" if abs(e.mean() - want) < 5e-5 else "DIFF"), flush=True)
print("inference %.1f s" % (time.time() - t0), flush=True)

def contrast(a, b, mask=None):
    d = []
    for seed in range(8):
        ea, eb = errs[a + (seed,)], errs[b + (seed,)]
        if mask is not None:
            ea, eb = ea[~mask], eb[~mask]
        d.append(ea.mean() - eb.mean())
    d = np.array(d)
    se = d.std(ddof=1) / np.sqrt(len(d))
    half = 2.3646243 * se
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "se": float(se),
            "lo": float(d.mean() - half), "hi": float(d.mean() + half),
            "neg": int((d < 0).sum()), "pos": int((d > 0).sum())}

res = {}
for name, a, b in [("a_n1_vs_off", ("none", 1), ("none", 0)),
                   ("b_n20_vs_off", ("none", 20), ("none", 0)),
                   ("c_shadow", ("after_snap", 20), ("none", 20)),
                   ("d_order", ("before_snap", 20), ("after_snap", 20))]:
    res[name] = {"full": contrast(a, b), "minus_near": contrast(a, b, mask_near),
                 "minus_ident": contrast(a, b, mask_ident)}
    print(name, json.dumps(res[name]["full"]), flush=True)
    print("   minus_near ", json.dumps(res[name]["minus_near"]), flush=True)

res["_meta"] = {"n_held": len(held), "n_identical": int(mask_ident.sum()), "n_near": int(mask_near.sum()),
                "checks_max_abs_diff": max(abs(c[1] - c[2]) for c in checks),
                "checks_all_within_5e-5": all(abs(c[1] - c[2]) < 5e-5 for c in checks)}
Path("quant_runs/leak_contrast_%s.json" % A.bits).write_text(json.dumps(res, indent=2), encoding="utf-8")
print("wrote quant_runs/leak_contrast_%s.json" % A.bits, "max|mae-csv| =", res["_meta"]["checks_max_abs_diff"])
