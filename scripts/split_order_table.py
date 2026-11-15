"""Cross-split replication of the ordering contrast (before_snap vs after_snap at N=20).

The ordering claim in the paper is measured on split 0 with eight seeds. This script
extends it to a 3 x 3 design (split_seed x init_seed, seeds 1..3 on splits 0..2): every
cell carries both protocols at interval 20, each run rebuilds its own split, and the
paired difference is read on the held-out part of that split (the 1,556 images that
take no part in checkpoint selection).

Reuse, not re-derivation: build_views / gather / forward_all / mae / weight_key from
scripts/heldout_eval.py, qdq constants and MEAN/STD from scripts/deploy_table.py, and
the t distribution from scripts/order_tost.py. Only FP32 deployment weights are scored
(the contrast of interest is a function-space difference between two training orders,
not a bit-width effect).

Gate: the six split-0 cells must reproduce quant_runs/heldout_eval_v3.csv mae_fp32 on
heldout_1556 to 1e-4, otherwise the script stops without writing deliverables.

Usage: D:/tool/Python/python.exe scripts/split_order_table.py [--splits 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import deploy_table as dt          # noqa: E402
import heldout_eval as he          # noqa: E402
from order_tost import t_cdf, t_ppf  # noqa: E402
from data.utkface import scan      # noqa: E402
from models.net import DrivingNet  # noqa: E402

RESULTS = Path("results")
V3 = Path("quant_runs/heldout_eval_v3.csv")
OUT_CSV = Path("quant_runs/split_order_table.csv")
OUT_JSON = Path("quant_runs/split_order_stats.json")

SPLITS = (0, 1, 2)
SEEDS = (1, 2, 3)
MODES = ("before_snap", "after_snap")
MARGIN = 0.0016
N20 = 20
TOL_GATE = 1e-4


def split_seed_of(args: dict) -> int:
    """train.py:211 — split_seed < 0 means 'use the init seed'."""
    sp = args.get("split_seed", -1)
    return int(args["seed"]) if sp is None or int(sp) < 0 else int(sp)


def collect() -> dict:
    idx: dict = {}
    for rj in RESULTS.glob("utk-*/result.json"):
        try:
            a = json.loads(rj.read_text(encoding="utf-8"))["args"]
        except Exception:
            continue
        if a.get("head") != "relu_qat" or int(a.get("epochs", 0)) != 20:
            continue
        if int(a.get("snap_every", 1)) != N20 or int(a.get("val_windows", 0)) != 2000:
            continue
        if round(float(a.get("ema_decay", -1)), 4) != 0.99:
            continue
        if a.get("ema_mode") not in MODES:
            continue
        if a.get("quant_mode", "overwrite") != "overwrite":
            continue
        key = (split_seed_of(a), int(a["seed"]), a["ema_mode"])
        idx.setdefault(key, []).append(rj.parent.name)
    return idx


def ref_names() -> set:
    with V3.open(encoding="utf-8") as f:
        return {r["run"] for r in csv.DictReader(f)}


def pick(cands: list, ref: set) -> str:
    canon = [c for c in cands if c in ref]
    if canon:
        return sorted(canon)[0]
    return max(cands, key=lambda s: int(s.rsplit("-", 1)[1]))


def ci(vals: list) -> dict:
    n = len(vals)
    m = st.mean(vals)
    sd = st.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    df = n - 1
    tc = t_ppf(0.975, df) if n > 1 else 0.0
    lo, hi = (m - tc * se, m + tc * se) if n > 1 else (m, m)
    out = {"n": n, "mean": m, "sd": sd, "se": se, "df": df, "lo": lo, "hi": hi}
    if n > 1 and se > 0:
        out["p_low"] = 1.0 - t_cdf((m + MARGIN) / se, df)
        out["p_high"] = 1.0 - t_cdf((MARGIN - m) / se, df)
        out["p_tost"] = max(out["p_low"], out["p_high"])
    return out


def fmt(d: dict) -> str:
    s = "%+.6f [%+.6f, %+.6f] n=%d" % (d["mean"], d["lo"], d["hi"], d["n"])
    if "p_tost" in d:
        s += "  TOST p=%.3g" % d["p_tost"]
    return s


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/utkface_raw224.npy")
    p.add_argument("--data", default="D:/data/utkface/UTKFace")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--splits", default="0,1,2")
    a = p.parse_args()
    splits = tuple(int(s) for s in a.splits.split(",") if s.strip())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s  splits %s" % (dev, splits), flush=True)

    idx, ref = collect(), ref_names()
    chosen, missing = {}, []
    for sp in SPLITS:
        for sd in SEEDS:
            for m in MODES:
                cands = idx.get((sp, sd, m), [])
                if not cands:
                    missing.append((sp, sd, m))
                else:
                    chosen[(sp, sd, m)] = pick(cands, ref)
    print("cells found: %d/18  missing: %s" % (len(chosen), missing), flush=True)

    want = {(sp, sd, m) for sp in splits for sd in SEEDS for m in MODES}
    todo = [k for k in sorted(chosen) if k[0] in splits]
    if not todo:
        print("nothing to evaluate", flush=True)
        return 2

    recs = scan(a.data)
    arr = np.load(a.raw, mmap_mode="r")
    mean_t = torch.tensor(dt.MEAN, dtype=torch.float32, device=dev).view(1, 1, 1, 3)
    std_t = torch.tensor(dt.STD, dtype=torch.float32, device=dev).view(1, 1, 1, 3)

    mae_by: dict = {}
    t_all = time.time()
    for sp in splits:
        first, mon, held, val, mon_names, inter = he.build_views(recs, sp)
        print("split %d: val=%d monitor=%d heldout=%d overlap=%d" % (
            sp, len(val), len(mon), len(held), len(inter)), flush=True)
        if len(held) != 1556 or inter:
            raise SystemExit("split %d: unexpected held-out size %d" % (sp, len(held)))
        u8 = he.gather(arr, first + held)
        yt_held = np.array([float(r["target"]) for r in held])
        for sd in SEEDS:
            for m in MODES:
                key = (sp, sd, m)
                if key not in chosen:
                    continue
                run = chosen[key]
                t0 = time.time()
                net = DrivingNet(head="relu_qat", grid=(14, 14), seq=1).to(dev)
                ck = torch.load(RESULTS / run / "best.pth", map_location="cpu")
                net.load_state_dict(ck[he.weight_key(m)], strict=False)
                net.eval()
                pred = he.forward_all(net, u8, mean_t, std_t, dev, a.bs)
                mae = float(np.abs(pred[he.N_FIRST:] - yt_held).mean())
                mae_by[key] = (run, mae)
                print("  split %d seed %d %-11s %-44s MAE %.6f (%.1fs)" % (
                    sp, sd, m, run, mae, time.time() - t0), flush=True)
                del ck
        del u8

    # ---- gate: split-0 cells against the published held-out table ----
    ref_rows = {}
    with V3.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["subset"] == "heldout_1556":
                ref_rows[r["run"]] = float(r["mae_fp32"])
    worst = 0.0
    checked = 0
    for (sp, sd, m), (run, mae) in mae_by.items():
        if sp != 0 or run not in ref_rows:
            continue
        checked += 1
        worst = max(worst, abs(mae - ref_rows[run]))
    print("GATE split-0 replay: %d cells, max|d| = %.2e (tol %.0e) %s" % (
        checked, worst, TOL_GATE, "OK" if worst <= TOL_GATE else "FAIL"), flush=True)
    if worst > TOL_GATE:
        print("gate failed -> no deliverables written", flush=True)
        return 3

    # ---- paired differences ----
    per_split, all_d = {}, []
    rows = []
    for sp in SPLITS:
        ds = []
        for sd in SEEDS:
            b, af = mae_by.get((sp, sd, "before_snap")), mae_by.get((sp, sd, "after_snap"))
            if not b or not af:
                continue
            d = b[1] - af[1]
            ds.append(d)
            all_d.append(d)
            rows.append({"split_seed": sp, "seed": sd,
                         "run_before": b[0], "run_after": af[0],
                         "mae_before": "%.6f" % b[1], "mae_after": "%.6f" % af[1],
                         "delta": "%.6f" % d, "n_images": 1556})
        if ds:
            per_split[sp] = ds

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split_seed", "seed", "run_before", "run_after",
                                          "mae_before", "mae_after", "delta", "n_images"])
        w.writeheader()
        w.writerows(rows)

    stats = {"margin": MARGIN, "n_images": 1556, "rows": rows,
             "per_split": {}, "pooled": None}
    for sp, ds in per_split.items():
        stats["per_split"][str(sp)] = dict(ci(ds), deltas=ds, positive=sum(1 for x in ds if x > 0))

    if len(all_d) >= 3:
        pooled = ci(all_d)
        pooled["positive"] = sum(1 for x in all_d if x > 0)
        pooled["negative"] = sum(1 for x in all_d if x < 0)
        stats["pooled"] = pooled

    OUT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\npaired difference  before_snap - after_snap  (FP32, held-out 1,556 per split)")
    for sp in sorted(per_split):
        print("  split %d: %s  positive %d/%d" % (
            sp, fmt(stats["per_split"][str(sp)]), stats["per_split"][str(sp)]["positive"],
            stats["per_split"][str(sp)]["n"]))
    if stats["pooled"]:
        pl = stats["pooled"]
        print("  pooled  : %s  positive %d/%d  negative %d" % (
            fmt(pl), pl["positive"], pl["n"], pl["negative"]))
    # between-split spread of the cell means
    if len(per_split) > 1:
        means = [st.mean(v) for v in per_split.values()]
        print("  per-split means: %s  spread %.6f" % (
            ", ".join("%+.6f" % m for m in means), max(means) - min(means)))
    print("\nwrote %s and %s (%.0fs)" % (OUT_CSV, OUT_JSON, time.time() - t_all), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
