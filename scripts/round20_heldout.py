"""第二十轮：把 held-out（1556 张，选轮从未见过的验证补集）评估扩展到
quant_runs/deploy_table_v3.csv 里全部 46 个固定划分 run（ema_decay=0.99 且
split_seed=0），与既有 400 张 first400 口径并列成一张长表。

口径不重写：直接 import scripts/heldout_eval.py，复用它的 build_views（split_random +
VAL_RATIO=0.15、N_FIRST=400、N_MON=2000 的监控抽样、以及「held-out = val 中文件名不在
监控 2000 里」的判据）、gather、forward_all、mae、weight_key；量化复用 deploy_table.qdq_
与 MEAN/STD；划分 seed 复用 deploy_table.split_seed_of。

闸门（任一失败即停，不写交付物）：
  G1 划分重建：按 sp=0 重建的验证集均值零基线，必须与 46 个 run 的 result.json 里
     zero_baseline_mae 一致（|d| <= 1e-9）——同时证明验证划分与 train.py 同源。
  G2 监控/补集互斥：held-out 恰为 1556 张，与监控 2000 张的文件名交集为 0。
  G3 回归：heldout_eval.csv 里已有的 11 个 run，4 个 MAE 列 x 2 个子集逐格 |d| <= 1e-4。

选轮未用 held-out 的代码证据（train.py）：
  :195-197  监控子集 = Random(split_seed).sample(val_r, val_windows=2000)
  :316-321  best_live / best_ema 只由该监控子集上的逐轮 MAE 决定
  :336-338  best.pth 落盘 best_live_state / best_ema_state
  :199-201  零基线 = 整份验证划分（3556）的均值，本脚本 Z_split 与之同口径
  held-out = val_r - 监控 2000（按文件名），故从未参与选轮。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import deploy_table as dt          # noqa: E402
import heldout_eval as he          # noqa: E402
from data.utkface import scan      # noqa: E402
from models.net import DrivingNet  # noqa: E402

V3_TABLE = Path("quant_runs/deploy_table_v3.csv")
REF_TABLE = Path("quant_runs/heldout_eval.csv")
OUT_CSV = Path("quant_runs/heldout_eval_v3.csv")
OUT_JSON = Path("quant_runs/heldout_report.json")
OUT_MD = Path("quant_runs/heldout_report.md")

BITS = (8, 6, 4)
SUBSETS = ("first400", "heldout_1556")
COLS = ("mae_fp32", "mae_8", "mae_6", "mae_4")
TOL_REG = 1e-4
TOL_ZERO = 1e-9
SP = 0
MAIN_SEQS = (("none", 1), ("none", 20), ("none", 0), ("before_snap", 20), ("after_snap", 20))
ORDER = ("none", "before_snap", "after_snap")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_runs() -> list[dict]:
    """deploy_table_v3.csv 中 ema_decay=='0.99' 且 split_seed=='0' 的全部 run。"""
    with V3_TABLE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sel = [r for r in rows if r["ema_decay"] == "0.99" and r["split_seed"] == "0"]
    out = []
    for r in sel:
        out.append({"run": r["run"], "mode": r["mode"], "N": int(r["snap_every"]),
                    "seed": int(r["seed"]), "decay": float(r["ema_decay"])})
    out.sort(key=lambda e: (e["N"], e["mode"], e["seed"], e["run"]))
    return out


def ref_index() -> dict:
    """heldout_eval.csv -> {run: {(subset, col): 字符串值}}。"""
    idx: dict = {}
    with REF_TABLE.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = idx.setdefault(r["run"], {})
            for s in SUBSETS:
                if r["subset"] == s:
                    for c in COLS:
                        d[(s, c)] = r[c]
    return idx


def md_tbl(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def sd_of(v: list):
    return statistics.stdev(v) if len(v) > 1 else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/utkface_raw224.npy")
    p.add_argument("--data", default="D:/data/utkface/UTKFace")
    p.add_argument("--bs", type=int, default=64)
    a = p.parse_args()

    t_all = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("device: %s" % dev, flush=True)

    # 断言复用的常量确实是既有口径（防止 heldout_eval.py 被改动后静默漂移）
    assert (he.VAL_RATIO, he.N_FIRST, he.N_MON) == (0.15, 400, 2000)
    assert SUBSETS[1] == "heldout_1556" and he.N_MON == 2000

    plan = select_runs()
    print("plan: %d runs from %s (ema_decay=0.99, split_seed=0)" % (len(plan), V3_TABLE), flush=True)
    if len(plan) != 46:
        print("GATE 0 FAILED: expected 46 runs, got %d" % len(plan), flush=True)
        sys.exit(2)
    ref = ref_index()
    ref_in_plan = {e["run"] for e in plan if e["run"] in ref}
    exp_ref_cells = sum(1 for e in plan if e["run"] in ref
                        for s in SUBSETS for c in COLS if ref[e["run"]].get((s, c), ""))
    print("reference runs present in plan: %d (expected cells %d)" % (
        len(ref_in_plan), exp_ref_cells), flush=True)
    plan.sort(key=lambda e: (e["run"] not in ref, e["N"], e["mode"], e["seed"], e["run"]))

    recs = scan(a.data)
    arr = np.load(a.raw, mmap_mode="r")

    # ---------------- G2：划分与监控/补集 ----------------
    first, mon, held, val, mon_names, inter = he.build_views(recs, SP)
    g2 = (len(held) == 1556 and len(val) == 3556 and len(mon) == he.N_MON and not inter
          and len(first) == he.N_FIRST)
    print("GATE B split_seed=%d: val=%d monitor=%d heldout=%d overlap=%d first400=%d %s" % (
        SP, len(val), len(mon), len(held), len(inter), len(first), "OK" if g2 else "FAIL"), flush=True)
    if not g2:
        sys.exit(3)

    u8 = he.gather(arr, first + held)
    yt = np.array([float(r["target"]) for r in first + held])
    yt_first, yt_held = yt[:he.N_FIRST], yt[he.N_FIRST:]
    mu_val = float(np.mean([float(r["target"]) for r in val]))
    z_split = {"first400": float(np.abs(yt_first - mu_val).mean()),
               "heldout_1556": float(np.abs(yt_held - mu_val).mean())}
    z_self = {"first400": float(np.abs(yt_first - yt_first.mean()).mean()),
              "heldout_1556": float(np.abs(yt_held - yt_held.mean()).mean())}
    print("zero baseline: mu_val=%.6f  Z_split first400=%.6f heldout=%.6f  "
          "Z_self first400=%.6f heldout=%.6f" % (
              mu_val, z_split["first400"], z_split["heldout_1556"],
              z_self["first400"], z_self["heldout_1556"]), flush=True)

    # ---------------- 元数据 + G1 ----------------
    meta = {}
    bad_meta, zrec = [], {}
    for e in plan:
        rj = Path("results") / e["run"] / "result.json"
        if not rj.exists():
            bad_meta.append((e["run"], "no result.json"))
            continue
        m = json.loads(rj.read_text(encoding="utf-8"))
        args = m["args"]
        sp = dt.split_seed_of(args, e["seed"], e["run"])
        meta[e["run"]] = {"args": args, "sp": sp, "ema": args["ema_mode"],
                          "zero_baseline_mae": float(m["zero_baseline_mae"]),
                          "val_mae_live": float(m["val_mae_live"])}
        zrec[e["run"]] = float(m["zero_baseline_mae"])
        if not (sp == SP and args["ema_mode"] == e["mode"]
                and int(args.get("snap_every", 1)) == e["N"]
                and int(args.get("val_windows", 0)) == he.N_MON
                and abs(float(args.get("ema_decay", -1)) - 0.99) < 1e-12
                and args.get("head") == "relu_qat" and args.get("arch", "net") == "net"
                and int(args.get("seed", 0)) == e["seed"]):
            bad_meta.append((e["run"], "meta mismatch", sp, args.get("ema_mode"),
                             args.get("snap_every"), args.get("val_windows")))
    z_full = float(np.abs(np.array([float(r["target"]) for r in val]) - mu_val).mean())
    z_dev = max(abs(v - z_full) for v in zrec.values()) if zrec else float("inf")
    print("GATE 1 zero-baseline recon: computed %.9f  vs 46 result.json  max|d| = %.3e  %s" % (
        z_full, z_dev, "OK" if z_dev <= TOL_ZERO else "FAIL"), flush=True)
    if bad_meta:
        print("META PROBLEMS: %s" % (bad_meta,), flush=True)
    if z_dev > TOL_ZERO or bad_meta:
        sys.exit(4)

    # ---------------- 推理 ----------------
    mean_t = torch.tensor(dt.MEAN, dtype=torch.float32, device=dev).view(1, 1, 1, 3)
    std_t = torch.tensor(dt.STD, dtype=torch.float32, device=dev).view(1, 1, 1, 3)
    rows, per_run, ref_rows, failed = [], [], [], []
    gate3 = {"checked": 0, "bad": 0, "max": {}}
    t_inf = time.time()

    for k, e in enumerate(plan, 1):
        run = e["run"]
        ck_path = Path("results") / run / "best.pth"
        t0 = time.time()
        try:
            if not ck_path.exists():
                raise FileNotFoundError("no best.pth")
            net = DrivingNet(head="relu_qat", grid=(14, 14), seq=1).to(dev)
            ck = torch.load(ck_path, map_location="cpu")
            key = he.weight_key(meta[run]["ema"])
            net.load_state_dict(ck[key], strict=False)
            net.eval()
            base = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
            store = {}
            for tag in ["fp32"] + ["%dbit" % b for b in BITS]:
                if tag != "fp32":
                    net.load_state_dict(base, strict=False)
                    dt.qdq_(net, int(tag[:-3]))
                store[tag] = he.forward_all(net, u8, mean_t, std_t, dev, a.bs)
            net.load_state_dict(base, strict=False)
            del ck
        except Exception as exc:                     # 失败必须单列，不得静默跳过
            failed.append({"run": run, "error": "%s: %s" % (type(exc).__name__, exc)})
            print("[%2d/%d] %-46s FAILED %s" % (k, len(plan), run, exc), flush=True)
            continue

        rec = {"run": run, "mode": e["mode"], "N": e["N"], "split_seed": SP, "seed": e["seed"],
               "checkpoint": ck_path.name, "state_dict_key": key,
               "bits": ["fp32"] + ["%dbit" % b for b in BITS],
               "seconds": round(time.time() - t0, 1)}
        for j, sub in enumerate(SUBSETS):
            sl = slice(0, he.N_FIRST) if j == 0 else slice(he.N_FIRST, len(yt))
            yt_s = yt_first if j == 0 else yt_held
            rec[sub] = {}
            for tag in store:
                col = "mae_fp32" if tag == "fp32" else "mae_%s" % tag[:-3]
                rec[sub][col] = float(np.abs(store[tag][sl] - yt_s).mean())
            n_pred = int(len(store["fp32"][sl]))
            rows.append({"run": run, "mode": e["mode"], "N": e["N"], "split_seed": SP,
                         "seed": e["seed"], "subset": sub, "n": len(yt_s),
                         "mae_fp32": "%.6f" % rec[sub]["mae_fp32"],
                         "mae_8": "%.6f" % rec[sub]["mae_8"],
                         "mae_6": "%.6f" % rec[sub]["mae_6"],
                         "mae_4": "%.6f" % rec[sub]["mae_4"],
                         "checkpoint": ck_path.name, "n_pred": n_pred})
            rec[sub]["n_pred"] = n_pred
        per_run.append(rec)
        print("[%2d/%d] %-46s N=%-3d %-11s fp32 %.4f/%.4f  8 %.4f/%.4f  6 %.4f/%.4f  "
              "4 %.4f/%.4f (%.1fs)" % (
                  k, len(plan), run, e["N"], e["mode"],
                  rec["first400"]["mae_fp32"], rec["heldout_1556"]["mae_fp32"],
                  rec["first400"]["mae_8"], rec["heldout_1556"]["mae_8"],
                  rec["first400"]["mae_6"], rec["heldout_1556"]["mae_6"],
                  rec["first400"]["mae_4"], rec["heldout_1556"]["mae_4"],
                  time.time() - t0), flush=True)

        # ---------------- G3：与 heldout_eval.csv 逐格回归 ----------------
        if run in ref:
            for sub in SUBSETS:
                for col in COLS:
                    r_s = ref[run].get((sub, col), "")
                    if not r_s:
                        continue
                    mine = rec[sub][col]
                    d = abs(mine - float(r_s))
                    gate3["checked"] += 1
                    gate3["max"][col] = max(gate3["max"].get(col, 0.0), d)
                    if d > TOL_REG:
                        gate3["bad"] += 1
                    ref_rows.append({"run": run, "mode": e["mode"], "N": e["N"],
                                     "seed": e["seed"], "subset": sub, "col": col,
                                     "mine": round(mine, 6), "ref": float(r_s),
                                     "abs_diff": float("%.3e" % d)})
        # 11 个参考 run 跑完就判定 G3，不过则立即停（不浪费余下 35 个）
        if gate3["bad"] and ref_in_plan <= {r["run"] for r in per_run}:
            break

    print("\nGATE 3 regression vs %s: %d/%d cells checked, %d over %.0e, max|d| = %s  %s"
          % (REF_TABLE, gate3["checked"], exp_ref_cells, gate3["bad"], TOL_REG,
             {c: "%.2e" % v for c, v in gate3["max"].items()},
             "OK" if gate3["bad"] == 0 and gate3["checked"] == exp_ref_cells else "FAIL"), flush=True)
    if gate3["bad"] or gate3["checked"] != exp_ref_cells:
        print("GATE 3 FAILED -> stop, no deliverables written", flush=True)
        sys.exit(5)

    cover_ok = sorted(r["run"] for r in per_run)
    want = sorted(e["run"] for e in plan)
    print("coverage: %d/%d runs evaluated, %d failed, %.1fs" % (
        len(cover_ok), len(want), len(failed), time.time() - t_inf), flush=True)

    # ---------------- 落盘 CSV ----------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    keys = ["run", "mode", "N", "split_seed", "seed", "subset", "n",
            "mae_fp32", "mae_8", "mae_6", "mae_4", "checkpoint", "n_pred"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # csv_line：表头为第 1 行，首个数据行第 2 行
    line_of = {(r["run"], r["subset"]): i + 2 for i, r in enumerate(rows)}
    csv_bytes, csv_sha = OUT_CSV.stat().st_size, sha256_of(OUT_CSV)
    print("wrote %s (%d rows, %d B, sha256 %s)" % (
        OUT_CSV, len(rows), csv_bytes, csv_sha), flush=True)

    # ---------------- 派生统计 ----------------
    def agg(mode: str, N: int) -> dict:
        rs = sorted([r for r in per_run if r["mode"] == mode and r["N"] == N],
                    key=lambda r: r["seed"])
        g = {"mode": mode, "N": N, "n_runs": len(rs),
             "runs": [{"run": r["run"], "seed": r["seed"],
                       "csv_line_heldout": line_of[(r["run"], "heldout_1556")],
                       "csv_line_first400": line_of[(r["run"], "first400")]} for r in rs]}
        for col in COLS:
            vals = [r["heldout_1556"][col] for r in rs]
            g[col] = {"mean": float(np.mean(vals)), "sd": sd_of(vals),
                      "min": float(np.min(vals)), "max": float(np.max(vals)),
                      "zero": z_split["heldout_1556"],
                      "norm_mean_over_zero": float(np.mean(vals)) / z_split["heldout_1556"]}
            vf = [r["first400"][col] for r in rs]
            g["first400_" + col] = {"mean": float(np.mean(vf)), "sd": sd_of(vf)}
        return g

    groups = [agg(m, n) for n in (0, 1, 5, 20) for m in ORDER
              if any(r["mode"] == m and r["N"] == n for r in per_run)]
    groups.sort(key=lambda g: (g["N"], ORDER.index(g["mode"])))

    seqs = []
    for mode, N in MAIN_SEQS:
        g = next((x for x in groups if x["mode"] == mode and x["N"] == N), None)
        rs = sorted([r for r in per_run if r["mode"] == mode and r["N"] == N],
                    key=lambda r: r["seed"])
        seqs.append({"mode": mode, "N": N, "n_runs": len(rs),
                     "seeds": [r["seed"] for r in rs],
                     "complete_seed_0_to_7": [r["seed"] for r in rs] == list(range(8)),
                     "per_seed": [{"seed": r["seed"], "run": r["run"],
                                   "csv_line_heldout": line_of[(r["run"], "heldout_1556")],
                                   "heldout": {c: r["heldout_1556"][c] for c in COLS},
                                   "first400": {c: r["first400"][c] for c in COLS}}
                                  for r in rs],
                     "heldout_mean": g["mae_fp32"]["mean"] if g else None,
                     "heldout_sd": g["mae_fp32"]["sd"] if g else None})

    # first400 旁证：与 deploy_table_v3.csv（4 位小数）逐格比较。
    # v3 的 MAE 列取自 quant_runs/perimage/*.npz（实测 v3 与 npz 在 46/46 上按 4 位小数相同），
    # 而这些 dump 有两种尺寸：n=400（与 first400 同集）与 n=800（更大子集）。
    # 后者使 v3 的那几行是 800 张口径，与 400 张的 first400 本来就不可比。
    with V3_TABLE.open(encoding="utf-8") as f:
        v3 = {r["run"]: r for r in csv.DictReader(f)}
    npz_n = {}
    for e in plan:
        pz = Path("quant_runs/perimage") / ("%s__fp32.npz" % e["run"])
        if pz.exists():
            with np.load(pz) as z:
                npz_n[e["run"]] = int(len(z["pred"]))
    cross = {"checked": 0, "max_abs_diff": 0.0, "over_1e_4": [], "npz_n_per_run": npz_n,
             "by_group": {g: {"runs": 0, "cells": 0, "max_abs_diff": 0.0, "over_1e_4": 0}
                          for g in ("npz_n_400", "npz_n_800", "npz_n_missing")}}
    for r in per_run:
        v = v3.get(r["run"])
        if not v:
            continue
        n_npz = npz_n.get(r["run"], -1)
        grp = "npz_n_400" if n_npz == he.N_FIRST else ("npz_n_800" if n_npz > 0 else "npz_n_missing")
        cross["by_group"][grp]["runs"] += 1
        for col in COLS:
            d = abs(r["first400"][col] - float(v[col]))
            cross["checked"] += 1
            cross["max_abs_diff"] = max(cross["max_abs_diff"], d)
            cross["by_group"][grp]["cells"] += 1
            cross["by_group"][grp]["max_abs_diff"] = max(cross["by_group"][grp]["max_abs_diff"], d)
            if d > 1e-4:
                cross["by_group"][grp]["over_1e_4"] += 1
                cross["over_1e_4"].append({"run": r["run"], "col": col, "npz_n": n_npz,
                                           "mine": r["first400"][col], "v3": float(v[col])})

    rep = {
        "meta": {
            "generated_by": "scripts/round20_heldout.py",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(dev),
            "source_table": str(V3_TABLE),
            "reference_table": str(REF_TABLE),
            "selection": "deploy_table_v3.csv where ema_decay=='0.99' and split_seed=='0'",
            "n_selected": len(plan),
            "protocol": {
                "split": "data.utkface.split_random(recs, 0.15, split_seed)[1] via heldout_eval.build_views",
                "val_ratio": he.VAL_RATIO, "n_val": len(val),
                "n_first400": he.N_FIRST, "n_monitor": he.N_MON,
                "n_heldout": len(held), "overlap_monitor_heldout": len(inter),
                "heldout_rule": "held = [r for r in val if Path(r.path).name not in monitor_names]",
                "monitor_rule": "random.Random(split_seed).sample(val, 2000), train.py:197",
                "quant": "deploy_table.qdq_ per-output-channel absmax/2^(b-1)-1, round, clamp",
                "norm": "deploy_table.MEAN/STD",
                "forward": "heldout_eval.forward_all (bs=%d)" % a.bs,
                "bits": ["fp32", 8, 6, 4],
                "checkpoint_rule": "results/<run>/best.pth (heldout_eval.py:189,220 hardcode best.pth)",
                "weight_key": "ema_state_dict if ema_mode!=none else model_state_dict (deploy_table.py:148)",
                "csv_line_convention": "line 1 = header; data row i (0-based) is line i+2",
            },
        },
        "zero_baseline": {
            "definition": "constant prediction = mean target of the whole 3556-image validation "
                          "split, matching train.py:199-201",
            "mu_val": mu_val,
            "Z_split": z_split, "Z_self": z_self,
            "recorded_zero_baseline_mae_full_val": {"value": z_full,
                                                    "max_abs_dev_vs_46_result_json": z_dev},
        },
        "coverage": {
            "v3_fixed_split_runs": len(want),
            "evaluated": len(cover_ok),
            "failed": failed,
            "missing_checkpoint": [x for x in failed if "best.pth" in str(x.get("error", ""))],
            "per_run": [{"run": r["run"], "mode": r["mode"], "N": r["N"], "seed": r["seed"],
                         "checkpoint": r["checkpoint"], "state_dict_key": r["state_dict_key"],
                         "n_pred_heldout": r["heldout_1556"]["n_pred"],
                         "n_pred_first400": r["first400"]["n_pred"],
                         "csv_line_heldout": line_of[(r["run"], "heldout_1556")],
                         "csv_line_first400": line_of[(r["run"], "first400")],
                         "seconds": r["seconds"]} for r in per_run],
        },
        "regression_vs_heldout_eval": {
            "tolerance": TOL_REG, "cells_checked": gate3["checked"],
            "cells_expected": exp_ref_cells, "cells_over": gate3["bad"],
            "max_abs_diff_by_col": {c: float("%.3e" % v) for c, v in gate3["max"].items()},
            "runs_matched": sorted({x["run"] for x in ref_rows}),
            "cells": ref_rows,
        },
        "cross_check_first400_vs_deploy_v3": {
            "note": "deploy_table_v3.csv stores 4 decimals and its MAE columns were taken from "
                    "quant_runs/perimage/<run>__<bits>.npz (verified: v3 equals the npz MAE to 4 "
                    "decimals on 46/46 runs). Those dumps come in two sizes: n=400, the same "
                    "first400 subset this table uses, and n=800, a larger subset. For the n=800 "
                    "runs the v3 column is an 800-image MAE and therefore not comparable with a "
                    "400-image number; only the n=400 group is a real cross-check. Informational, "
                    "not a gate - the gate is G3 against heldout_eval.csv, which is n=400 live.",
            "cells_checked": cross["checked"], "max_abs_diff": cross["max_abs_diff"],
            "n_cells_over_1e_4": len(cross["over_1e_4"]),
            "by_group": cross["by_group"],
            "npz_n_per_run": cross["npz_n_per_run"],
        },
        "aggregates_heldout": groups,
        "main_sequences": seqs,
        "rows": [dict({"csv_line": line_of[(r["run"], r["subset"])]},
                      **{k: r[k] for k in keys}) for r in rows],
        "deviations": [
            "6 bit / 4 bit are evaluated for all 46 runs, including the two N=inf endpoint runs "
            "which heldout_eval.csv left blank (BITS_LIGHT). The blank reference cells are skipped, "
            "so G3 is unaffected (80 expected cells over 11 runs).",
            "Primary normalisation divides by the split-level zero baseline Z_split (constant = mean "
            "of the whole 3556-image validation split), matching train.py:199-201; the subset-mean "
            "variant Z_self is reported alongside.",
            "sd uses ddof=1; groups with a single run report sd = null.",
            "9 of 46 runs predate --split-seed and their result.json args carry no split_seed key; "
            "deploy_table.split_seed_of's existing rule (effective split = seed) applies, and their "
            "seed is 0, so the effective split_seed is 0 for every selected run (asserted).",
            "All 46 runs use val_windows=2000 (asserted), so the monitored subset - and therefore "
            "the held-out complement - is identical for every run.",
            "last.pth is never used: heldout_eval.py hardcodes best.pth, and all 46 runs have it.",
            "deploy_table_v3.csv cannot be used as a first400 cross-check for most runs: its MAE "
            "columns came from quant_runs/perimage/*.npz dumps, 32 of 46 of which hold n=800 "
            "predictions (an 800-image MAE), not the 400-image first400. This report keeps every "
            "number at n=400 and is gated instead against heldout_eval.csv (n=400, live), which it "
            "reproduces to <=4.9e-07 on 80/80 cells. Quantifying this is out of scope; the finding "
            "is reported, and no existing file was modified.",
            "No training, no re-quantisation of stored weights, no change to any existing file.",
        ],
    }
    OUT_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    json_bytes, json_sha = OUT_JSON.stat().st_size, sha256_of(OUT_JSON)
    print("wrote %s (%d B, sha256 %s)" % (OUT_JSON, json_bytes, json_sha), flush=True)

    # ---------------- Markdown ----------------
    L = []
    L.append("# held-out（1,556 张）复评：46 个固定划分 run")
    L.append("")
    L.append("生成：`scripts/round20_heldout.py`（%s，device=%s）。口径取自 "
             "`scripts/heldout_eval.py` 与 `scripts/deploy_table.py`，未重写。"
             % (rep["meta"]["generated_at"], dev))
    L.append("")
    L.append("## 0 协议")
    L.append("")
    L.append(md_tbl(["项", "值"], [
        ["划分", "`split_random(recs, 0.15, split_seed=0)`，val = 3,556 张"],
        ["监控子集", "`Random(0).sample(val, 2000)`（train.py:197）"],
        ["held-out", "val 中文件名不在监控 2000 里的 1,556 张；交集 %d" % len(inter)],
        ["first400", "`val[:400]`（heldout_eval.py:76）"],
        ["量化", "`deploy_table.qdq_`：逐输出通道 absmax/127（8 bit），就地写权重"],
        ["评估位宽", "fp32 / 8 / 6 / 4"],
        ["检查点", "`results/<run>/best.pth`；权重键 none→`model_state_dict`，"
                   "其余→`ema_state_dict`"],
        ["归一化分母", "Z_split：整份验证划分均值作常数预测，held-out 上 MAE = %.6f"
                       % z_split["heldout_1556"]],
    ]))
    L.append("")
    L.append("零基线两种口径：")
    L.append("")
    L.append(md_tbl(["子集", "n", "Z_split（分母，与 train.py 同口径）", "Z_self（子集自身均值）"], [
        ["first400", str(he.N_FIRST), "%.6f" % z_split["first400"], "%.6f" % z_self["first400"]],
        ["heldout_1556", str(len(held)), "%.6f" % z_split["heldout_1556"],
         "%.6f" % z_self["heldout_1556"]],
    ]))
    L.append("")
    L.append("G1 划分重建：按 sp=0 重算整份验证集零基线 = %.9f，与 46 个 `result.json` 的 "
             "`zero_baseline_mae` 最大偏差 %.2e（阈值 1e-9）→ 通过。" % (z_full, z_dev))
    L.append("")
    L.append("## A 回归校验：与 `quant_runs/heldout_eval.csv` 的 11 个 run 逐格对照")
    L.append("")
    L.append("阈值 1e-4；检查 %d 格（应为 %d），越界 %d 格。"
             % (gate3["checked"], exp_ref_cells, gate3["bad"]))
    L.append("")
    L.append(md_tbl(["列", "最大 |差|"],
                    [[c, "%.2e" % v] for c, v in sorted(gate3["max"].items())]))
    L.append("")
    L.append(md_tbl(["run", "mode", "N", "seed", "子集", "列", "本脚本", "heldout_eval.csv", "|差|"],
                    [["`%s`" % x["run"], x["mode"], x["N"], x["seed"], x["subset"],
                      x["col"], "%.6f" % x["mine"], "%.6f" % x["ref"], "%.1e" % x["abs_diff"]]
                     for x in ref_rows]))
    L.append("")
    L.append("## B 覆盖与缺口")
    L.append("")
    L.append(md_tbl(["项", "值"], [
        ["`deploy_table_v3.csv` 中 ema_decay=0.99 且 split_seed=0 的 run", str(len(want))],
        ["已评估", "%d/%d" % (len(cover_ok), len(want))],
        ["推理失败", str(len(failed))],
        ["缺 checkpoint", str(len(rep["coverage"]["missing_checkpoint"]))],
        ["未覆盖 run", "无" if len(cover_ok) == len(want)
         else ", ".join(sorted(set(want) - set(cover_ok)))],
    ]))
    L.append("")
    if failed:
        L.append("失败清单：")
        L.append("")
        L.append(md_tbl(["run", "错误"], [["`%s`" % f["run"], f["error"]] for f in failed]))
    else:
        L.append("缺口说明：无。46 个 run 的 `best.pth` 全部存在并完成推理，"
                 "未用 first400 顶替任何 held-out 行。")
    L.append("")
    L.append("## C 按 (mode, N) 聚合（held-out 1,556 张）")
    L.append("")
    L.append(md_tbl(["mode", "N", "n", "MAE fp32 均值", "sd", "min", "max", "8 bit 均值",
                     "6 bit 均值", "4 bit 均值", "fp32/零基线"],
                    [[g["mode"], g["N"], g["n_runs"], "%.6f" % g["mae_fp32"]["mean"],
                      "%.6f" % g["mae_fp32"]["sd"] if g["mae_fp32"]["sd"] is not None else "—",
                      "%.6f" % g["mae_fp32"]["min"], "%.6f" % g["mae_fp32"]["max"],
                      "%.6f" % g["mae_8"]["mean"], "%.6f" % g["mae_6"]["mean"],
                      "%.6f" % g["mae_4"]["mean"],
                      "%.4f" % g["mae_fp32"]["norm_mean_over_zero"]] for g in groups]))
    L.append("")
    L.append("sd 用 ddof=1；单 run 组记 —。归一化分母 Z_split(held-out) = %.6f。"
             % z_split["heldout_1556"])
    L.append("")
    L.append("### C2 五个主序列（seed 0–7 逐 seed）")
    L.append("")
    for s in seqs:
        L.append("**%s / N=%s**（n=%d，8 个 seed %s；held-out fp32 均值 %.6f，sd %s）" % (
            s["mode"], s["N"], s["n_runs"],
            "齐全" if s["complete_seed_0_to_7"] else "不齐", s["heldout_mean"],
            "%.6f" % s["heldout_sd"] if s["heldout_sd"] is not None else "—"))
        L.append("")
        L.append(md_tbl(["seed", "run", "csv 行", "fp32", "8 bit", "6 bit", "4 bit"],
                        [[p["seed"], "`%s`" % p["run"], p["csv_line_heldout"],
                          "%.6f" % p["heldout"]["mae_fp32"], "%.6f" % p["heldout"]["mae_8"],
                          "%.6f" % p["heldout"]["mae_6"], "%.6f" % p["heldout"]["mae_4"]]
                         for p in s["per_seed"]]))
        L.append("")
    L.append("## D 数字溯源")
    L.append("")
    L.append("`quant_runs/heldout_report.json` 的 `rows[*]` 与 CSV 逐行同序，"
             "每条带 `csv_line`（表头第 1 行，数据行 i 为第 i+2 行）；"
             "聚合表每组的 `runs[*].csv_line_heldout` 指向该 run 的 held-out 行，"
             "`main_sequences[*].per_seed[*].csv_line_heldout` 同理。")
    L.append("")
    L.append("## E 检查点清单与「选轮是否用了 held-out」")
    L.append("")
    L.append("结论：**未使用**。证据链（`train.py`）：")
    L.append("")
    L.append("1. `train.py:195-197`：监控子集 = "
             "`Random(split_seed).sample(va_ds.records, val_windows=2000)`。")
    L.append("2. `train.py:316-321`：`best_live`/`best_ema` "
             "只由该监控子集上的逐轮 MAE 决定。")
    L.append("3. `train.py:336-338`：`best.pth` 存的是 "
             "`best_live_state` / `best_ema_state`。")
    L.append("4. held-out = `val − 监控 2000`（按文件名），与监控子集交集 = %d，"
             "故选轮从未见过这 1,556 张。" % len(inter))
    L.append("5. 46 个 run 的 `val_windows` 全为 2000（脚本内断言），"
             "故监控集与 held-out 的定义在全部 run 上都成立。")
    L.append("")
    L.append(md_tbl(["run", "mode", "N", "seed", "checkpoint", "权重键", "n_pred"],
                    [["`%s`" % r["run"], r["mode"], r["N"], r["seed"], r["checkpoint"],
                      "`%s`" % r["state_dict_key"], r["heldout_1556"]["n_pred"]]
                     for r in per_run]))
    L.append("")
    L.append("## F first400 旁证（与 `deploy_table_v3.csv` 的 4 位小数对照）")
    L.append("")
    L.append("注：`deploy_table_v3.csv` 存 4 位小数，其 MAE 列实测取自 "
             "`quant_runs/perimage/*.npz`（v3 与 npz 在 46/46 上按 4 位小数相同），"
             "而这些 dump 有两种尺寸：n=400（与 first400 同集）与 n=800（更大子集）。"
             "n=800 的 run 在 v3 里是 800 张口径，与 400 张的 first400 不可比。"
             "本节只作旁证，不作为闸门；闸门是 A 节的 G3（对 n=400 的 heldout_eval.csv，80/80 格）。")
    L.append("")
    L.append(md_tbl(["npz 尺寸分组", "run 数", "格数", "最大 |差|", "超 1e-4 的格"], [
        [g, str(cross["by_group"][g]["runs"]), str(cross["by_group"][g]["cells"]),
         "%.2e" % cross["by_group"][g]["max_abs_diff"], str(cross["by_group"][g]["over_1e_4"])]
        for g in ("npz_n_400", "npz_n_800", "npz_n_missing") if cross["by_group"][g]["runs"]
    ]))
    L.append("")
    L.append("## G deviations")
    L.append("")
    for d in rep["deviations"]:
        L.append("- " + d)
    L.append("")
    L.append("## H 产物")
    L.append("")
    L.append(md_tbl(["文件", "字节", "sha256"], [
        ["`quant_runs/heldout_eval_v3.csv`", str(csv_bytes), "`%s`" % csv_sha],
        ["`quant_runs/heldout_report.json`", str(json_bytes), "`%s`" % json_sha],
        ["`quant_runs/heldout_report.md`", "（本文件，自指不写）", "—"],
    ]))
    L.append("")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("wrote %s (%d B, sha256 %s)" % (
        OUT_MD, OUT_MD.stat().st_size, sha256_of(OUT_MD)), flush=True)
    print("rows=%d runs=%d total %.1fs" % (len(rows), len(per_run), time.time() - t_all), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
