"""第二十轮 R5：把 STE 臂升到 run 层，与「就地覆写 @ 间隔 1」在同一固定划分上配对。

口径
  * 固定划分 split_seed = 0，五个主配置序列与 STE 臂共享同一验证划分；
  * 子集：first400（与监控 2000 重叠）与 heldout_1556（未参与选轮，主口径）；
  * 检查点：results/<run>/best.pth，键 model_state_dict（ema = none）；
  * 量化：deploy_table.qdq_ 8 bit 逐输出通道对称（与表 4 同口径）。

产物
  quant_runs/ste_v3.csv          16 run × 2 子集 × 2 列
  quant_runs/ste_runlevel.json   配对统计
  quant_runs/ste_runlevel.md     人读报告

闸门：G1 划分（val 3556 / monitor 2000 / heldout 1556 / 交集 0）；G2 每个 seed 恰一个 run；
      G3 复算的 first400 fp32 MAE 与 deploy_table_v4.csv 的 none@N=1 行 |d| <= 1e-4。
"""
from __future__ import annotations

import csv
import json
import math
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

SP = 0
BITS = (8,)
SUBSETS = ("first400", "heldout_1556")
OUT_CSV = Path("quant_runs/ste_v3.csv")
OUT_JSON = Path("quant_runs/ste_runlevel.json")
OUT_MD = Path("quant_runs/ste_runlevel.md")
V4 = Path("quant_runs/deploy_table_v4.csv")
TOL_REG = 1e-4


def ste_runs() -> dict:
    """seed -> run 目录名，只取 split_seed=0 的那批（每个 seed 必须唯一）。"""
    out = {}
    for d in sorted(Path("results").glob("utk-relu_qat-none-ste-s*")):
        rj = d / "result.json"
        if not rj.exists() or not (d / "best.pth").exists():
            continue
        j = json.load(open(rj, encoding="utf-8"))
        a = j.get("args", {})
        if a.get("split_seed") != SP:
            continue
        s = int(a["seed"])
        if s in out:
            raise SystemExit("G2 FAILED: seed %d has two runs: %s / %s" % (s, out[s], d.name))
        out[s] = d.name
    return out


def table_arm(snap_every: int) -> dict:
    """从 v4 取 mode=none、指定 snap_every、固定划分的主配置行（seed -> 行）。"""
    rows = list(csv.DictReader(open(V4, newline="", encoding="utf-8")))
    out = {}
    for r in rows:
        if r["mode"] != "none" or int(r["snap_every"]) != snap_every:
            continue
        if int(r["split_seed"]) != SP or r["ema_decay"] != "0.99":
            continue
        out[int(r["seed"])] = r
    return out


def main() -> None:
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("device: %s" % dev, flush=True)

    ste = ste_runs()
    own = table_arm(1)
    off = table_arm(0)
    print("STE runs (split 0): %s" % {k: v for k, v in sorted(ste.items())}, flush=True)
    print("overwrite runs (none@N=1, split 0): %s" % sorted(own), flush=True)
    print("projection-off runs (none@N=inf, split 0): %s" % sorted(off), flush=True)
    seeds = sorted(set(ste) & set(own) & set(off))
    print("paired seeds: %s" % seeds, flush=True)
    if not seeds:
        raise SystemExit("no paired seeds")

    recs = scan("D:/data/utkface/UTKFace")
    arr = np.load("data/utkface_raw224.npy", mmap_mode="r")
    first, mon, held, val, mon_names, inter = he.build_views(recs, SP)
    ok = (len(held) == 1556 and len(val) == 3556 and len(mon) == he.N_MON and not inter
          and len(first) == he.N_FIRST)
    print("GATE split: val=%d mon=%d held=%d inter=%d first=%d %s" % (
        len(val), len(mon), len(held), len(inter), len(first), "OK" if ok else "FAIL"), flush=True)
    if not ok:
        raise SystemExit("G1 FAILED")

    u8 = he.gather(arr, first + held)
    yt = np.array([float(r["target"]) for r in first + held])
    yt_first, yt_held = yt[:he.N_FIRST], yt[he.N_FIRST:]
    mean_t = torch.tensor(dt.MEAN, dtype=torch.float32, device=dev).view(1, 1, 1, 3)
    std_t = torch.tensor(dt.STD, dtype=torch.float32, device=dev).view(1, 1, 1, 3)

    preds = {}
    rows = []
    plans = (("ste", {s: ste[s] for s in seeds}),
             ("overwrite", {s: own[s]["run"] for s in seeds}),
             ("no_projection", {s: off[s]["run"] for s in seeds}))
    for arm, plan in plans:
        for s in seeds:
            run = plan[s]
            ck = Path("results") / run / "best.pth"
            net = DrivingNet(head="relu_qat", grid=(14, 14), seq=1).to(dev)
            st = torch.load(ck, map_location="cpu")
            key = "model_state_dict" if "model_state_dict" in st else "ema_state_dict"
            net.load_state_dict(st[key], strict=False)
            net.eval()
            base = {k: v.detach().clone() for k, v in net.state_dict().items()}
            out = {"fp32": he.forward_all(net, u8, mean_t, std_t, dev, 64)}
            for b in BITS:
                net.load_state_dict(base, strict=False)
                dt.qdq_(net, b)
                out["%dbit" % b] = he.forward_all(net, u8, mean_t, std_t, dev, 64)
            net.load_state_dict(base, strict=False)
            preds[(arm, s)] = out
            row = {"arm": arm, "seed": s, "run": run, "key": key}
            for j, sub in enumerate(SUBSETS):
                sl = slice(0, he.N_FIRST) if j == 0 else slice(he.N_FIRST, len(yt))
                yts = yt_first if j == 0 else yt_held
                for tag in ("fp32",) + tuple("%dbit" % b for b in BITS):
                    col = "mae_fp32" if tag == "fp32" else "mae_%d" % int(tag[:-3])
                    row[(sub, col)] = float(np.abs(out[tag][sl] - yts).mean())
            rows.append(row)
            print("  %-9s s%d %-42s fp32=%.6f 8bit=%.6f" % (
                arm, s, run, row[(SUBSETS[0], "mae_fp32")], row[(SUBSETS[0], "mae_8")]), flush=True)

    # G3 回归：overwrite 臂的 first400 fp32 必须复现 deploy_table_v4.csv
    worst = 0.0
    for arm, tab in (("overwrite", own), ("no_projection", off)):
        for s in seeds:
            v = float(tab[s]["mae_fp32"])
            got = [r for r in rows if r["arm"] == arm and r["seed"] == s][0][("first400", "mae_fp32")]
            worst = max(worst, abs(got - v))
    print("GATE regress: max|d| vs v4 = %.2e %s" % (worst, "OK" if worst <= TOL_REG else "FAIL"), flush=True)
    if worst > TOL_REG:
        raise SystemExit("G3 FAILED")

    def tcrit(df):
        return 2.3646 if df == 7 else 1.96

    def get(arm, s, key):
        return [r for r in rows if r["arm"] == arm and r["seed"] == s][0][key]

    PAIRS = (("ste", "overwrite"), ("ste", "no_projection"), ("overwrite", "no_projection"))
    stats = {}
    for subj in SUBSETS:
        for col in ("mae_fp32", "mae_8"):
            for a_arm, b_arm in PAIRS:
                ds = [get(a_arm, s, (subj, col)) - get(b_arm, s, (subj, col)) for s in seeds]
                n = len(ds)
                m = statistics.fmean(ds)
                sd = statistics.stdev(ds)
                se = sd / math.sqrt(n)
                tc = tcrit(n - 1)
                neg = sum(1 for x in ds if x < 0)
                k = max(neg, n - neg)
                p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)
                tag = "%s|%s|%s-%s" % (subj, col, a_arm, b_arm)
                stats[tag] = {
                    "n": n, "mean": m, "sd": sd, "se": se, "arms": [a_arm, b_arm],
                    "ci": [m - tc * se, m + tc * se], "neg": neg, "sign_p": p,
                    "per_seed": {str(s): d for s, d in zip(seeds, ds)},
                }
                print("%-12s %-8s %-28s n=%d mean=%+.6f sd=%.6f ci=[%+.6f,%+.6f] neg=%d p=%.4f" % (
                    subj, col, a_arm + "-" + b_arm, n, m, sd,
                    m - tc * se, m + tc * se, neg, p), flush=True)

    # 与关闭投影端点的对照：STE 是否也贴着 0.055 附近
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "run", "state_dict_key", "subset", "n", "mae_fp32", "mae_8"])
        for r in rows:
            for sub in SUBSETS:
                w.writerow([r["arm"], r["seed"], r["run"], r["key"], sub,
                            400 if sub == "first400" else 1556,
                            "%.6f" % r[(sub, "mae_fp32")], "%.6f" % r[(sub, "mae_8")]])
    json.dump({"meta": {"split_seed": SP, "seeds": seeds, "ste_runs": {str(k): v for k, v in ste.items()},
                        "overwrite_runs": {str(k): own[k]["run"] for k in seeds},
                        "projection_off_runs": {str(k): off[k]["run"] for k in seeds},
                        "gate_regress_max_abs_d": worst, "seconds": round(time.time() - t0, 1),
                        "mae": "perimage-free direct forward on first400 + heldout_1556"},
               "contrasts": stats,
               "runs": [{"arm": r["arm"], "seed": r["seed"], "run": r["run"],
                         **{("%s|%s" % (sub, col)): r[(sub, col)] for sub in SUBSETS for col in ("mae_fp32", "mae_8")}}
                        for r in rows]},
              open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    L = ["# STE 臂升到 run 层（R5）", "",
         "固定划分 split_seed = 0；STE 与就地覆写（none@N=1）在同一批 seed 上配对；"
         "检查点 best.pth，8 bit 逐输出通道对称量化。", ""]
    L.append("| 子集 | 列 | 对比 | n | 均值 | sd | 95 % CI | 反向 | 符号 p |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, v in stats.items():
        subj, col, pair = k.split("|")
        L.append("| %s | %s | %s | %d | %+.6f | %.6f | [%+.6f, %+.6f] | %d | %.4f |" % (
            subj, col, pair, v["n"], v["mean"], v["sd"], v["ci"][0], v["ci"][1], v["neg"], v["sign_p"]))
    L += ["", "差值方向为第一个臂减第二个臂，负值表示前者更好；no_projection = none@N=inf。", ""]
    Path(OUT_MD).write_text("\n".join(L), encoding="utf-8")
    print("wrote %s / %s / %s  (%.0fs)" % (OUT_CSV, OUT_JSON, OUT_MD, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
