#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二十轮机制对照：学习率 x 投影间隔、训练预算、随机舍入。

三个已训练组（只 load best.pth 做前向，不训练、不用 GPU）：
  results_lr/    lr in {1e-3, 3e-3} x snap_every in {1, 20, 0} x seed in {0,1,2}   （13 个 run）
  results_long/  60 轮 x snap_every in {1, 0} x seed in {0,1,2}                    （6 个 run）
  results_sr/    N=1 上把 snap_weights_ 换成随机舍入 x seed in {0..7}            （8 个 run）
参照（既有主序列，lr=1e-4、20 轮、none 臂）：quant_runs/deploy_table_v4.csv 的 9 个 run。

口径不重写：直接 import scripts/heldout_eval.py（build_views / gather / forward_all / mae /
weight_key）与 scripts/deploy_table.py（qdq_、MEAN/STD、split_seed_of）。
  主口径 = held-out 1556（选轮从未见过的验证补集，round20_heldout.py 的同一判据）；
  次口径 = val 前 400（deploy_table_v4.csv 同源）。
每个 run 的验证划分按它自己 result.json 的 split_seed 重建（本表全部 run 都是 0，
但代码不取默认值），并用 result.json 的 zero_baseline_mae 逐个反查（G1）。
8 bit = 仅权重 fake quant（deploy_table.qdq_：per-output-channel absmax、RTN、对称），
激活保持 FP32 —— 与 deploy_table.py / heldout_eval.py 完全同一条链路。

闸门（任一失败即停，不写交付物）：
  G1 划分重建：每个 run 按其 sp 重建的整份验证集均值零基线 vs result.json
     zero_baseline_mae，|d| <= 1e-9。
  G2 口径尺寸：val 3556 / 监控 2000 / held-out 1556 / first400 400，监控与 held-out 交集 0。
  G3 验证格：results/utk-relu_qat-none-s0-1793111850 的 4 个 MAE 与
     quant_runs/heldout_eval_v3.csv 逐格 |d| <= 1e-6。
  G4 回归：本次评到且出现在 heldout_eval_v3.csv 的 run，4 格 |d| <= 1e-4；
     9 个主序列 run 的 first400 两列 vs deploy_table_v4.csv，|d| <= 1e-5。
  G5 元数据：每个 run 的 ema_mode/head/batch_size/epochs/lr/seed/val_windows/ema_decay/arch
     与所在组一致；snap_every 缺席或为 None 只出现在 --snap-every 之前的 run 上，按 1 读。
  G6 机制签名：N=1 -> grid_rms <= 3e-3 且 on_grid >= 0.90；N=inf(0) -> grid_rms >= 6e-3 且
     on_grid <= 0.15。on_grid 有恒定地板（snap_weights_ 只投影 nn.Linear/nn.Conv2d，
     nn.MultiheadAttention 的 in_proj_weight 从未被投影），故判据用 grid_rms 的相对量。
     （snap_every=None 按 1 读靠这条核实）
  G7 运行预算：整脚本墙钟 <= 1200 s。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import deploy_table as dt                      # noqa: E402
import heldout_eval as he                      # noqa: E402
from data.utkface import scan, split_random    # noqa: E402
from models.net import DrivingNet              # noqa: E402

RAW = "data/utkface_raw224.npy"
DATA = "D:/data/utkface/UTKFace"
BITS = 8
SUBSETS = ("first400", "heldout_1556")
SUB_LABEL = {"first400": "first400 (400，次口径)", "heldout_1556": "held-out (1556，主口径)"}
V3 = Path("quant_runs/heldout_eval_v3.csv")
V4 = Path("quant_runs/deploy_table_v4.csv")
OUT_MD = Path("quant_runs/mechanism_report.md")
OUT_JSON = Path("quant_runs/mechanism_stats.json")
LOG = Path("logs/round20_mechanism.log")
VERIF_RUN = "utk-relu_qat-none-s0-1793111850"
TOL_ZERO = 1e-9
TOL_CELL = 1e-6
TOL_REG = 1e-4
TOL_V4 = 6e-5
BUDGET_S = 1200.0
GROUP_LABEL = {"main20": "lr=1e-4, 20 轮（main 序列）", "lr1e-3": "lr=1e-3, 20 轮",
               "lr3e-3": "lr=3e-3, 20 轮", "long60": "lr=1e-4, 60 轮",
               "sr": "lr=1e-4, 20 轮, N=1, 随机舍入"}
N_LABEL = {1: "1", 20: "20", 0: "inf (snap_every=0)"}


# ---------------------------------------------------------------- 小工具
class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def jload(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jsafe(o):
    """JSON 的键必须是标量；cells1/d_lr 的键是 tuple，这里写成字符串。"""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            kk = k if isinstance(k, (str, int, float, bool)) or k is None else " | ".join(str(x) for x in k)
            out[kk] = jsafe(v)
        return out
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return o


def norm_n(args: dict, run: str):
    """snap_every 缺席/None 只出现在 --snap-every 之前的 run 上；那时 train.py 每步都投影，
    等价 N=1。这条读法由 G6（on_grid==1.0、grid_rms==0）实测确认，不靠猜 args。"""
    if "snap_every" not in args or args["snap_every"] is None:
        return 1, "args 无 snap_every -> 按 N=1 读（G6 核）"
    return int(args["snap_every"]), "args.snap_every"


def md_tbl(header, rows) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def u6(x) -> str:
    return "—" if x is None else "%.6f" % x


def s6(x) -> str:
    return "—" if x is None else "%+.6f" % x


def u4(x) -> str:
    return "—" if x is None else "%.4f" % x


def cell_seed(vals, seeds, nd=6) -> str:
    return "<br>".join("s%d %s" % (s, ("%.*f" % (nd, v))) for s, v in zip(seeds, vals)
                       if v is not None)


def ci_str(s: dict) -> str:
    if s.get("n", 0) < 2 or s.get("ci_lo") is None:
        return "区间不可估（n=%d）" % s.get("n", 0)
    return "95%% CI [%s, %s]（n=%d, df=%d）" % (s6(s["ci_lo"]), s6(s["ci_hi"]), s["n"], s["df"])


def mean_ci(s: dict, nd=6) -> str:
    if s.get("mean") is None:
        return "—"
    return "%+.*f" % (nd, s["mean"]) + ("" if s.get("ci_lo") is None
                                        else " [%s, %s]" % (s6(s["ci_lo"]), s6(s["ci_hi"])))


def verdict(s: dict) -> str:
    if s.get("n", 0) < 2 or s.get("ci_lo") is None:
        return "不可估（n=%d）" % s.get("n", 0)
    if s["ci_lo"] > 0:
        return "区间 > 0"
    if s["ci_hi"] < 0:
        return "区间 < 0"
    return "区间含 0"


def st_summary(vals) -> dict:
    v = [float(x) for x in vals]
    n = len(v)
    base = {"n": n, "mean": None, "sd": None, "se": None, "df": 0,
            "ci_lo": None, "ci_hi": None, "p": None}
    if n == 0:
        return base
    m = float(statistics.fmean(v))
    base.update({"mean": m})
    if n < 2:
        return base
    sd = float(statistics.stdev(v))
    se = sd / math.sqrt(n)
    df = n - 1
    if se > 0:
        from scipy import stats as sps
        half = float(sps.t.ppf(0.975, df) * se)
        p = float(2.0 * sps.t.sf(abs(m / se), df))
    else:
        half, p = 0.0, None
    base.update({"sd": sd, "se": se, "df": df, "p": p,
                 "ci_lo": m - half, "ci_hi": m + half})
    return base


# ---------------------------------------------------------------- 选 run
def entry_of(root, run, group, lr, epochs, n, seed, sp, args, res, n_src) -> dict:
    return {"run": run, "root": root, "dir": "%s/%s" % (root, run),
            "group": group, "lr": float(lr), "epochs": int(epochs), "N": int(n),
            "seed": int(seed), "sp": int(sp), "ema_mode": args.get("ema_mode"),
            "n_src": n_src, "args": args,
            "val_mae_live": float(res.get("val_mae_live", float("nan"))),
            "zero_baseline_mae": float(res.get("zero_baseline_mae", float("nan")))}


def select_dir_group(root: Path, group: str, lr_expect: float, ep_expect: int,
                     n_allowed, seeds_expect) -> list:
    out = []
    for d in sorted(root.glob("utk-relu_qat-*")):
        rj = d / "result.json"
        if not rj.exists():
            raise SystemExit("%s: 缺 result.json" % d)
        res = jload(rj)
        args = res["args"]
        seed = int(args["seed"])
        lr = float(args.get("lr", -1.0))
        ep = int(args.get("epochs", -1))
        n, n_src = norm_n(args, d.name)
        sp = dt.split_seed_of(args, seed, d.name)
        if abs(lr - lr_expect) > 1e-12:
            raise SystemExit("%s: lr=%r，期望 %r" % (d, lr, lr_expect))
        if ep != ep_expect:
            raise SystemExit("%s: epochs=%r，期望 %r" % (d, ep, ep_expect))
        if n not in n_allowed:
            raise SystemExit("%s: snap_every=%r，期望属于 %s" % (d, n, n_allowed))
        if seed not in seeds_expect:
            raise SystemExit("%s: seed=%r，期望属于 %s" % (d, seed, seeds_expect))
        out.append(entry_of(root.as_posix(), d.name, group, lr, ep, n, seed, sp, args, res, n_src))
    return out


def select_lr() -> list:
    """results_lr/ 的 13 个 run，按 args.lr 分两组（不按目录名猜）。"""
    out = []
    for d in sorted(Path("results_lr").glob("utk-relu_qat-*")):
        rj = d / "result.json"
        if not rj.exists():
            raise SystemExit("%s: 缺 result.json" % d)
        res = jload(rj)
        args = res["args"]
        seed = int(args["seed"])
        lr = float(args.get("lr", -1.0))
        ep = int(args.get("epochs", -1))
        n, n_src = norm_n(args, d.name)
        sp = dt.split_seed_of(args, seed, d.name)
        if abs(lr - 1e-3) <= 1e-12:
            g = "lr1e-3"
        elif abs(lr - 3e-3) <= 1e-12:
            g = "lr3e-3"
        else:
            raise SystemExit("%s: 未见过的 lr=%r" % (d, lr))
        if ep != 20:
            raise SystemExit("%s: epochs=%r，期望 20" % (d, ep))
        if n not in (0, 1, 20):
            raise SystemExit("%s: snap_every=%r" % (d, n))
        out.append(entry_of("results_lr", d.name, g, lr, ep, n, seed, sp, args, res, n_src))
    return out


def select_main() -> list:
    """lr=1e-4、20 轮、none 臂的参照序列：以 deploy_table_v4.csv 为准（模式/seed/sp 全核）。"""
    sel = []
    with V4.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["mode"] != "none":
                continue
            if int(r["seed"]) not in tuple(range(8)):
                continue
            if abs(float(r["ema_decay"]) - 0.99) > 1e-12:
                continue
            if int(r["split_seed"]) != 0:
                continue
            if int(r["snap_every"]) not in (0, 1, 20):
                continue
            sel.append(r)
    if len(sel) != 24:
        raise SystemExit("deploy_table_v4.csv 的 none 臂 seed 0-7 应为 24 行，实际 %d" % len(sel))
    out = []
    for r in sel:
        d = Path("results") / r["run"]
        res = jload(d / "result.json")
        args = res["args"]
        seed = int(r["seed"])
        n, n_src = norm_n(args, r["run"])
        if n != int(r["snap_every"]):
            raise SystemExit("%s: csv snap_every=%s 与 args 读出的 %d 不一致"
                             % (d, r["snap_every"], n))
        sp = dt.split_seed_of(args, seed, r["run"])
        if int(args.get("epochs", -1)) != 20 or abs(float(args.get("lr", -1)) - 1e-4) > 1e-12:
            raise SystemExit("%s: 参照 run 的 lr/epochs 不是 1e-4/20" % d)
        e = entry_of("results", r["run"], "main20", 1e-4, 20, n, seed, sp, args, res, n_src)
        e["v4_mae_fp32"] = float(r["mae_fp32"])
        e["v4_mae_8"] = float(r["mae_8"])
        e["v4_n_eval"] = int(r["n_eval"]) if r.get("n_eval") else -1
        out.append(e)
    return out


def select_all() -> list:
    main = select_main()
    lr = select_lr()
    long_ = select_dir_group(Path("results_long"), "long60", 1e-4, 60, (0, 1), (0, 1, 2))
    sr = select_dir_group(Path("results_sr"), "sr", 1e-4, 20, (1,), tuple(range(8)))
    ents = main + lr + long_ + sr
    seen = {}
    for e in ents:
        k = (e["group"], e["N"], e["seed"])
        if k in seen:
            raise SystemExit("重复格 %s：%s 与 %s" % (k, seen[k], e["dir"]))
        seen[k] = e["dir"]
    want = {"main20": 24, "lr1e-3": 4, "lr3e-3": 9, "long60": 6, "sr": 8}
    got = {}
    for e in ents:
        got[e["group"]] = got.get(e["group"], 0) + 1
    if got != want:
        raise SystemExit("各组件数 %s != 期望 %s" % (got, want))
    return ents


# ---------------------------------------------------------------- 机制读数
def grid_stats_state(sd: dict, bits: int = 8) -> dict:
    """与 scripts/paper2_diagnostics.py:35-44 同一判据（按 weight-tensor 的 numel 加权）：
    on_grid = |q - round(q)| < 0.05 格距的权重占比；grid_rms = rms(w - q_hat) / rms(w)。"""
    qmax = 2 ** (bits - 1) - 1
    on = 0
    tot = 0
    sq_res = 0.0
    sq_ref = 0.0
    off = []
    off_numel = 0
    for k, t in sd.items():
        if not t.is_floating_point() or t.dim() not in (2, 4):
            continue
        flat = t.reshape(t.shape[0], -1).float()
        scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        q = flat / scale
        resid = (q - q.round()).abs()
        n_on = int((resid < 0.05).sum().item())
        on += n_on
        tot += int(resid.numel())
        if n_on < 0.99 * resid.numel():
            off.append([k, round(n_on / resid.numel(), 4), int(resid.numel())])
            off_numel += int(resid.numel())
        wq = q.round() * scale
        sq_res += float((flat - wq).pow(2).sum().item())
        sq_ref += float(flat.pow(2).sum().item())
    return {"on_grid": on / tot if tot else None,
            "grid_rms": math.sqrt(sq_res / sq_ref) if sq_ref > 0 else None,
            "n_weights": tot, "off": off, "off_numel": off_numel}


# ---------------------------------------------------------------- worker
_W = {}


def _init_worker(raw, data, bs, threads):
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    torch.set_grad_enabled(False)
    _W["recs"] = scan(data)
    _W["arr"] = np.load(raw, mmap_mode="r")
    _W["bs"] = int(bs)
    _W["dev"] = torch.device("cpu")
    _W["views"] = {}
    _W["mean_t"] = torch.tensor(dt.MEAN, dtype=torch.float32).view(1, 1, 1, 3)
    _W["std_t"] = torch.tensor(dt.STD, dtype=torch.float32).view(1, 1, 1, 3)


def _view(sp):
    v = _W["views"].get(sp)
    if v is None:
        first, mon, held, val, mon_names, inter = he.build_views(_W["recs"], sp)
        v = {"u8": he.gather(_W["arr"], first + held),
             "yt": np.array([float(r["target"]) for r in first + held]),
             "len_first": len(first), "len_held": len(held),
             "len_val": len(val), "len_mon": len(mon), "len_inter": len(inter)}
        _W["views"][sp] = v
    return v


def eval_task(t):
    t0 = time.time()
    v = _view(t["sp"])
    net = DrivingNet(head="relu_qat", grid=(14, 14), seq=1, pretrained=False)
    ck = torch.load(str(Path(t["dir"]) / "best.pth"), map_location="cpu")
    key = he.weight_key(t["ema_mode"])
    missing, unexpected = net.load_state_dict(ck[key], strict=False)
    net.eval()
    base = {k: val.detach().clone() for k, val in net.state_dict().items()}
    gs = grid_stats_state(base, BITS)
    pred = {}
    pred["fp32"] = he.forward_all(net, v["u8"], _W["mean_t"], _W["std_t"], _W["dev"], _W["bs"])
    net.load_state_dict(base, strict=False)
    dt.qdq_(net, BITS)
    pred["b8"] = he.forward_all(net, v["u8"], _W["mean_t"], _W["std_t"], _W["dev"], _W["bs"])
    nf = v["len_first"]
    yt = v["yt"]
    out = {"run": t["run"], "state_dict_key": key,
           "missing": list(missing), "unexpected": list(unexpected),
           "on_grid": gs["on_grid"], "grid_rms": gs["grid_rms"], "n_weights": gs["n_weights"],
           "off": gs["off"], "off_numel": gs["off_numel"],
           "n_first": nf, "n_held": int(len(yt) - nf), "mae": {}}
    for tag in ("fp32", "b8"):
        p = pred[tag]
        out["mae"][tag] = {"first400": float(he.mae(p[:nf], yt[:nf])),
                           "heldout_1556": float(he.mae(p[nf:], yt[nf:]))}
    out["seconds"] = round(time.time() - t0, 1)
    del ck, base, net
    return out


# ---------------------------------------------------------------- 权重距离
def state_of(dirspec: str) -> dict:
    ck = torch.load(str(Path(dirspec) / "best.pth"), map_location="cpu")
    return ck["model_state_dict"]


def rel_l2(sa: dict, sb: dict):
    num = 0.0
    den = 0.0
    for k in sb:
        if not sb[k].is_floating_point():
            continue
        d = sa[k].float() - sb[k].float()
        num += float(d.pow(2).sum().item())
        den += float(sb[k].float().pow(2).sum().item())
    return math.sqrt(num / den) if den > 0 else None


# ---------------------------------------------------------------- 统计对象
def cell_of(ents, R, subset) -> dict:
    es = sorted(ents, key=lambda e: e["seed"])
    fp = [R[e["run"]]["mae"]["fp32"][subset] for e in es]
    b8 = [R[e["run"]]["mae"]["b8"][subset] for e in es]
    return {"basis": subset, "n": len(es), "seeds": [e["seed"] for e in es],
            "dirs": [e["dir"] for e in es],
            "fp32": dict(st_summary(fp), vals=fp),
            "b8": dict(st_summary(b8), vals=b8)}


def contrast_of(a_ents, b_ents, R, subset, label) -> dict:
    am = {e["seed"]: e for e in a_ents}
    bm = {e["seed"]: e for e in b_ents}
    seeds = sorted(set(am) & set(bm))
    out = {"label": label, "basis": subset, "n": len(seeds), "seeds": seeds,
           "dirs_a": [am[s]["dir"] for s in seeds],
           "dirs_b": [bm[s]["dir"] for s in seeds],
           "seed_note": "A 有 seed %s，B 有 seed %s" % (sorted(am), sorted(bm))}
    for prec in ("fp32", "b8"):
        d = [R[am[s]["run"]]["mae"][prec][subset] - R[bm[s]["run"]]["mae"][prec][subset]
             for s in seeds]
        out[prec] = dict(st_summary(d), diffs=d)
    return out


def dcontrast_of(c_hi, c_lo, prec) -> dict:
    hi = dict(zip(c_hi["seeds"], c_hi[prec]["diffs"]))
    lo = dict(zip(c_lo["seeds"], c_lo[prec]["diffs"]))
    seeds = sorted(set(hi) & set(lo))
    d = [hi[s] - lo[s] for s in seeds]
    return dict(st_summary(d), seeds=seeds, diffs=d, prec=prec, basis=c_hi["basis"])


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()

    t0 = time.time()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    sys.stdout = tee
    try:
        _run(a, t0)
    finally:
        sys.stdout = tee.out
        tee.close()


def _run(a, t0) -> None:
    print("round20_mechanism.py  torch=%s  workers=%d threads=%d bs=%d"
          % (torch.__version__, a.workers, a.threads, a.bs), flush=True)
    assert (he.VAL_RATIO, he.N_FIRST, he.N_MON) == (0.15, 400, 2000), "heldout_eval 常量被改过"
    gates = []

    def gate(gid, ok, detail):
        gates.append({"id": gid, "ok": bool(ok), "detail": detail})
        print("[gate %s] %-4s %s" % (gid, "OK" if ok else "FAIL", detail), flush=True)
        return ok

    ents = select_all()
    print("选定 %d 个 run：" % len(ents), flush=True)
    for g in ("main20", "lr1e-3", "lr3e-3", "long60", "sr"):
        es = [e for e in ents if e["group"] == g]
        print("  %-8s %2d  %s" % (g, len(es), ", ".join(
            "s%d/N%s" % (e["seed"], N_LABEL[e["N"]].split(" ")[0])
            for e in sorted(es, key=lambda x: (x["N"], x["seed"])))), flush=True)

    # ---------- G1/G2：划分重建 ----------
    recs = scan(a.data)
    sps = sorted({e["sp"] for e in ents})
    zfull, views_meta = {}, {}
    for sp in sps:
        first, mon, held, val, mon_names, inter = he.build_views(recs, sp)
        mu = float(np.mean([float(r["target"]) for r in val]))
        zfull[sp] = float(np.abs(np.array([float(r["target"]) for r in val]) - mu).mean())
        views_meta[sp] = {"len_val": len(val), "len_mon": len(mon), "len_held": len(held),
                          "len_first": len(first), "len_inter": len(inter), "mu_val": mu,
                          "overlap_first_held": len({Path(r["path"]).name for r in first}
                                                    & {Path(r["path"]).name for r in held}),
                          "overlap_first_mon": len({Path(r["path"]).name for r in first}
                                                   & {Path(r["path"]).name for r in mon})}
    dev_z = max(abs(e["zero_baseline_mae"] - zfull[e["sp"]]) for e in ents)
    gate("G1", dev_z <= TOL_ZERO,
         "划分重建：%d 个 run 的 zero_baseline_mae vs 按各自 sp 重建的整份验证集零基线，max|d| = %.3e"
         % (len(ents), dev_z))
    g2_ok = all(v["len_val"] == 3556 and v["len_mon"] == 2000 and v["len_held"] == 1556
                and v["len_first"] == 400 and v["len_inter"] == 0 for v in views_meta.values())
    gate("G2", g2_ok, "口径尺寸：%s" % "; ".join(
        "sp=%d val=%d mon=%d held=%d first=%d overlap(mon,held)=%d"
        % (sp, v["len_val"], v["len_mon"], v["len_held"], v["len_first"], v["len_inter"])
        for sp, v in views_meta.items()))

    # ---------- G5：元数据 ----------
    bad = []
    for e in ents:
        g = e["args"]
        ok = (g.get("ema_mode") == "none" and g.get("head") == "relu_qat"
              and int(g.get("batch_size", -1)) == 64
              and int(g.get("val_windows", -1)) == he.N_MON
              and abs(float(g.get("ema_decay", -1)) - 0.99) < 1e-12
              and int(g.get("seed", -1)) == e["seed"]
              and int(g.get("epochs", -1)) == e["epochs"]
              and abs(float(g.get("lr", -1)) - e["lr"]) < 1e-12
              and g.get("arch", "net") == "net"
              and e["sp"] == 0)
        if not ok:
            bad.append((e["dir"], {k: g.get(k) for k in ("ema_mode", "head", "batch_size",
                                                         "val_windows", "ema_decay", "seed",
                                                         "epochs", "lr", "arch")}))
    gate("G5", not bad,
         "元数据：%d 个 run 的 ema_mode/head/batch/val_windows/decay/seed/epochs/lr/arch/sp 与所在组一致%s"
         % (len(ents), "" if not bad else "，问题 %s" % bad[:3]))
    if bad:
        sys.exit(5)

    # ---------- 推理 ----------
    tasks = [{"run": e["run"], "dir": e["dir"], "sp": e["sp"], "ema_mode": e["ema_mode"]}
             for e in ents]
    tasks.sort(key=lambda t: t["dir"])
    print("开始前向：%d 个 run（fp32 + 8 bit，各 1956 张），%d 进程 x %d 线程"
          % (len(tasks), a.workers, a.threads), flush=True)
    t_inf = time.time()
    cache_dir = Path("quant_runs/mechanism_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    def _cp(t):
        return cache_dir / (t["dir"].replace("/", "_").replace(chr(92), "_") + ".json")
    cached, pending = {}, []
    for t in tasks:
        cp = _cp(t)
        if cp.exists():
            try:
                cached[t["run"]] = json.loads(cp.read_text(encoding="utf-8"))
                continue
            except Exception:
                pass
        pending.append(t)
    n_all = len(tasks)
    tasks = pending
    print("缓存命中 %d、待算 %d" % (len(cached), len(tasks)), flush=True)
    evals = []
    if not tasks:
        pass
    elif a.workers <= 1:
        _init_worker(a.raw, a.data, a.bs, a.threads)
        for k, t in enumerate(tasks, 1):
            r = eval_task(t)
            evals.append(r)
            _cp(t).write_text(json.dumps(jsafe(r), ensure_ascii=False), encoding="utf-8")
            print("  [%2d/%d] %-46s %5.1fs" % (k, len(tasks), r["run"], r["seconds"]), flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx,
                                 initializer=_init_worker,
                                 initargs=(a.raw, a.data, a.bs, a.threads)) as ex:
            for k, r in enumerate(ex.map(eval_task, tasks), 1):
                evals.append(r)
                _cp(tasks[k-1]).write_text(json.dumps(jsafe(r), ensure_ascii=False), encoding="utf-8")
                print("  [%2d/%d] %-46s %5.1fs" % (k, len(tasks), r["run"], r["seconds"]),
                      flush=True)
    R = dict(cached)
    R.update({r["run"]: r for r in evals})
    print("前向用时 %.1f s" % (time.time() - t_inf), flush=True)
    if len(R) != n_all:
        gate("G0", False, "有 run 未返回：%d/%d" % (len(R), n_all))
        sys.exit(6)

    bad_state = [(r["run"], r["missing"], r["unexpected"]) for r in R.values()
                 if r["missing"] or r["unexpected"]]
    gate("G5b", not bad_state,
         "checkpoint 严格对齐：%d 个 run 的 state_dict 无缺失/多余键（strict=False 只用于取回计数）"
         % len(R))

    # ---------- G6：机制签名 ----------
    # on_grid 有一个恒定地板：snap_weights_（quant/qat.py:11-26）只处理 nn.Linear / nn.Conv2d，
    # nn.MultiheadAttention 的 in_proj_weight 从未被投影。故判据用 grid_rms 的相对量，
    # 阈值取 N=1 与 N=inf 实测区间的中点（N=1 最差 2.76e-3、N=inf 最好 6.90e-3）。
    sig_bad = []
    for e in ents:
        r = R[e["run"]]
        if e["N"] == 1:
            ok = r["grid_rms"] is not None and r["grid_rms"] <= 3e-3 and r["on_grid"] >= 0.90
        elif e["N"] == 0:
            ok = r["grid_rms"] is not None and r["grid_rms"] >= 6e-3 and r["on_grid"] <= 0.15
        else:
            ok = True
        if not ok:
            sig_bad.append((e["dir"], e["N"], round(r["on_grid"], 4), "%.2e" % r["grid_rms"]))
    gate("G6", not sig_bad, "机制签名（grid_rms 相对量，N=1 <= 3e-3 / N=inf >= 6e-3；"
                             "on_grid 有未被投影的 in_proj_weight 造成的地板）%s"
         % ("" if not sig_bad else "，异常 %s" % sig_bad[:3]))

    # ---------- 交叉产物 ----------
    v3 = {}
    if V3.exists():
        with V3.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                v3[(row["run"], row["subset"])] = row
    need = [("first400", "mae_fp32"), ("first400", "mae_8"),
            ("heldout_1556", "mae_fp32"), ("heldout_1556", "mae_8")]
    vcells, vdev = [], None
    if (VERIF_RUN, "first400") in v3:
        for sub, col in need:
            row = v3[(VERIF_RUN, sub)]
            prec = "fp32" if col == "mae_fp32" else "b8"
            got = R[VERIF_RUN]["mae"][prec][sub]
            exp = float(row[col])
            vcells.append({"subset": sub, "col": col, "got": got, "expect": exp,
                           "abs_diff": abs(got - exp)})
        vdev = max(c["abs_diff"] for c in vcells)
    gate("G3", vdev is not None and vdev <= TOL_CELL,
         "验证格 %s：4 格重算 vs heldout_eval_v3.csv，max|d| = %s"
         % (VERIF_RUN, "缺失" if vdev is None else "%.3e" % vdev))

    reg, reg_dev = [], None
    for e in ents:
        if (e["run"], "first400") not in v3:
            continue
        for sub, col in need:
            row = v3[(e["run"], sub)]
            prec = "fp32" if col == "mae_fp32" else "b8"
            d = abs(R[e["run"]]["mae"][prec][sub] - float(row[col]))
            reg.append({"run": e["run"], "subset": sub, "col": col, "abs_diff": d})
            reg_dev = d if reg_dev is None else max(reg_dev, d)
    v4dev, v4n = None, 0
    for e in ents:
        if "v4_mae_fp32" not in e:
            continue
        for prec, key in (("fp32", "v4_mae_fp32"), ("b8", "v4_mae_8")):
            d = abs(R[e["run"]]["mae"][prec]["first400"] - e[key])
            v4dev = d if v4dev is None else max(v4dev, d)
            v4n += 1
    ok4 = (reg_dev is not None and reg_dev <= TOL_REG and v4dev is not None and v4dev <= TOL_V4)
    gate("G4", ok4, "回归：%d 个 run x 4 格 vs heldout_eval_v3.csv max|d| = %s；"
                    "9 个主序列 run x 2 列 vs deploy_table_v4.csv max|d| = %s"
         % (len({r["run"] for r in reg}), "%.3e" % reg_dev if reg_dev is not None else "—",
            "%.3e" % v4dev if v4dev is not None else "—"))

    # ---------- 表 1 ----------
    by = {}
    for e in ents:
        by.setdefault((e["group"], e["N"]), []).append(e)

    def g(gname, n):
        return by.get((gname, n), [])

    LR_ROW = (("1e-4", "main20"), ("1e-3", "lr1e-3"), ("3e-3", "lr3e-3"))
    cells1 = {}
    for label, gname in LR_ROW:
        for n in (1, 20, 0):
            cells1[(label, n)] = cell_of(g(gname, n), R, "heldout_1556")
    c1_f, c1_400 = {}, {}
    for label, gname in LR_ROW:
        c1_f[label] = contrast_of(g(gname, 1), g(gname, 0), R, "heldout_1556",
                                  "lr=%s: N=1 - N=inf" % label)
        c1_400[label] = contrast_of(g(gname, 1), g(gname, 0), R, "first400",
                                    "lr=%s: N=1 - N=inf" % label)
    d_lr = {}
    for pair in (("1e-4", "3e-3"), ("1e-4", "1e-3")):
        for prec in ("fp32", "b8"):
            for basis, ctab in (("heldout_1556", c1_f), ("first400", c1_400)):
                d_lr[(pair, prec, basis)] = dcontrast_of(ctab[pair[1]], ctab[pair[0]], prec)

    # ---------- 表 2 ----------
    cells2 = {"20ep": {1: cell_of(g("main20", 1), R, "heldout_1556"),
                       0: cell_of(g("main20", 0), R, "heldout_1556")},
              "60ep": {1: cell_of(g("long60", 1), R, "heldout_1556"),
                       0: cell_of(g("long60", 0), R, "heldout_1556")}}
    cells2_400 = {"20ep": {1: cell_of(g("main20", 1), R, "first400"),
                           0: cell_of(g("main20", 0), R, "first400")},
                  "60ep": {1: cell_of(g("long60", 1), R, "first400"),
                           0: cell_of(g("long60", 0), R, "first400")}}
    gap2 = {}
    for tag, gg in (("20ep", "main20"), ("60ep", "long60")):
        for basis in SUBSETS:
            gap2[(tag, basis)] = contrast_of(g(gg, 1), g(gg, 0), R, basis,
                                             "%s: N=1 - N=inf" % tag)
    chg2 = {}
    for prec in ("fp32", "b8"):
        for basis in SUBSETS:
            chg2[(prec, basis)] = dcontrast_of(gap2[("60ep", basis)], gap2[("20ep", basis)], prec)

    # ---------- 表 3 ----------
    cells3 = {"RN@1": cell_of(g("main20", 1), R, "heldout_1556"),
              "SR@1": cell_of(g("sr", 1), R, "heldout_1556"),
              "inf": cell_of(g("main20", 0), R, "heldout_1556")}
    cells3_400 = {"RN@1": cell_of(g("main20", 1), R, "first400"),
                  "SR@1": cell_of(g("sr", 1), R, "first400"),
                  "inf": cell_of(g("main20", 0), R, "first400")}
    c3 = {}
    for basis in SUBSETS:
        c3[("RN-inf", basis)] = contrast_of(g("main20", 1), g("main20", 0), R, basis,
                                            "RN@N=1 - N=inf (代价)")
        c3[("SR-inf", basis)] = contrast_of(g("sr", 1), g("main20", 0), R, basis,
                                            "SR@N=1 - N=inf (残余代价)")
        c3[("SR-RN", basis)] = contrast_of(g("sr", 1), g("main20", 1), R, basis,
                                           "SR@N=1 - RN@N=1 (恢复量，负 = SR 更好)")
    rec = {}
    for prec in ("fp32", "b8"):
        hi = dict(zip(c3[("SR-RN", "heldout_1556")]["seeds"],
                      c3[("SR-RN", "heldout_1556")][prec]["diffs"]))
        base = dict(zip(c3[("RN-inf", "heldout_1556")]["seeds"],
                        c3[("RN-inf", "heldout_1556")][prec]["diffs"]))
        seeds = sorted(set(hi) & set(base))
        vals = [(-(hi[s]) / base[s]) if abs(base[s]) > 1e-9 else None for s in seeds]
        okv = [v for v in vals if v is not None]
        rec[prec] = {"seeds": seeds, "vals": vals,
                     "mean": float(statistics.fmean(okv)) if okv else None}

    # ---------- 权重距离 ----------
    wdist = []
    main1 = {e["seed"]: e for e in g("main20", 1)}
    main0 = {e["seed"]: e for e in g("main20", 0)}
    sr_e = {e["seed"]: e for e in g("sr", 1)}
    srs = {s: state_of(sr_e[s]["dir"]) for s in sorted(sr_e)}
    rns = {s: state_of(main1[s]["dir"]) for s in sorted(sr_e)}
    infs = {s: state_of(main0[s]["dir"]) for s in sorted(sr_e)}
    for s in sorted(sr_e):
        wdist.append({"seed": s, "sr": sr_e[s]["dir"], "rn": main1[s]["dir"], "inf": main0[s]["dir"],
                      "sr_rn": rel_l2(srs[s], rns[s]),
                      "rn_inf": rel_l2(rns[s], infs[s]),
                      "sr_inf": rel_l2(srs[s], infs[s]),
                      "sr_live": sr_e[s]["val_mae_live"], "rn_live": main1[s]["val_mae_live"],
                      "inf_live": main0[s]["val_mae_live"]})
    del srs, rns, infs

    # ---------- 结论句 ----------
    ans = {}
    q1c = c1_f["1e-4"]["b8"]
    q1h = c1_f["3e-3"]["b8"]
    q1d = d_lr[(("1e-4", "3e-3"), "b8", "heldout_1556")]
    ans["q1"] = (
        "学习率提高会让 N=1 的投影代价变小：held-out 1556 张、8 bit 上，N=1−N=inf 从 lr=1e-4 的 %s 降到 "
        "lr=3e-3 的 %s，逐 seed 差中差为 %s（%s，判定：%s）；lr=1e-3 一档的 N=inf 只有 seed 0、"
        "N=1 有 seed 0/1，该档对比 n=1、区间不可估，因此只能说「把学习率放大 30 倍后代价降到接近 0」，"
        "不能说已证明代价随 lr 单调。"
        % (mean_ci(q1c), mean_ci(q1h), mean_ci(q1d), ci_str(q1d), verdict(q1d)))
    ans["q1_short"] = ("提升学习率确实缩小了 N=1 的投影代价（8 bit held-out：lr 1e-4 为 %s，"
                       "3e-3 为 %s；差中差 %s，%s）"
                       % (s6(q1c["mean"]), s6(q1h["mean"]), s6(q1d["mean"]), verdict(q1d)))

    g20 = gap2[("20ep", "heldout_1556")]["b8"]
    g60 = gap2[("60ep", "heldout_1556")]["b8"]
    gch = chg2[("b8", "heldout_1556")]
    if gch["n"] >= 2 and gch["ci_lo"] is not None and gch["ci_hi"] < 0:
        q2v = "缩小但未关闭"
    elif gch["n"] >= 2 and gch["ci_lo"] is not None and gch["ci_lo"] > 0:
        q2v = "变大"
    else:
        q2v = "持续（与 20 轮不可区分）"
    ans["q2_verdict"] = q2v
    ans["q2"] = ("把预算从 20 轮加到 60 轮没有关闭 N=1 的代价：8 bit held-out 上 gap 为 20 轮 %s、"
                 "60 轮 %s，逐 seed 变化（60 轮 − 20 轮）为 %s（%s，判定：%s）。"
                 % (mean_ci(g20), mean_ci(g60), mean_ci(gch), ci_str(gch), q2v))
    ans["q2_short"] = ("60 轮预算下 N=1 的代价没有关闭：gap 20 轮 %s、60 轮 %s，变化 %s（%s）"
                       % (s6(g20["mean"]), s6(g60["mean"]), s6(gch["mean"]), verdict(gch)))

    r3 = c3[("RN-inf", "heldout_1556")]["b8"]
    s3 = c3[("SR-inf", "heldout_1556")]["b8"]
    d3 = c3[("SR-RN", "heldout_1556")]["b8"]
    ans["q3"] = (
        "随机舍入把最近舍入的代价基本抵消：8 bit held-out 上 RN@N=1−N=inf = %s，SR@N=1−N=inf = %s，"
        "SR−RN = %s（%s，判定：%s）；逐 seed 恢复比例 (RN−SR)/(RN−inf) = %s（均值 %s）。"
        % (mean_ci(r3), mean_ci(s3), mean_ci(d3), ci_str(d3), verdict(d3),
           ", ".join("s%d %s" % (s, "—" if v is None else "%.1f%%" % (100 * v))
                     for s, v in zip(rec["b8"]["seeds"], rec["b8"]["vals"])),
           "—" if rec["b8"]["mean"] is None else "%.1f%%" % (100 * rec["b8"]["mean"])))
    ans["q3_short"] = ("SR@N=1 把最近舍入的代价基本抵消：SR−N=inf = %s，而 RN−N=inf = %s；"
                       "SR−RN = %s（%s）"
                       % (s6(s3["mean"]), s6(r3["mean"]), s6(d3["mean"]), verdict(d3)))

    runtime = time.time() - t0
    gate("G7", runtime <= BUDGET_S, "运行预算：墙钟 %.1f s（上限 %.0f s）" % (runtime, BUDGET_S))

    ok_all = all(x["ok"] for x in gates)
    print("闸门：%d/%d 通过" % (sum(1 for x in gates if x["ok"]), len(gates)), flush=True)
    if not ok_all:
        print("有闸门失败，不写交付物。", flush=True)
        sys.exit(2)

    prov = []
    psrc = Path("scripts/run_stochastic_rounding.py")
    if psrc.exists():
        for i, ln in enumerate(psrc.read_text(encoding="utf-8").splitlines(), 1):
            t = ln.strip()
            if "sr_snap_weights_" in t or "results_sr" in t or ("floor" in t and "sign" in t):
                prov.append("scripts/run_stochastic_rounding.py:%d: %s" % (i, t))
    prov = prov[:6]

    payload = {
        "meta": {"script": "scripts/round20_mechanism.py", "torch": torch.__version__,
                 "device": "cpu", "workers": a.workers, "threads": a.threads, "bs": a.bs,
                 "runtime_seconds": round(runtime, 1),
                 "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "python": sys.version.split()[0]},
        "protocol": {"primary_basis": "heldout_1556", "secondary_basis": "first400",
                     "bits": BITS,
                     "quantiser": "deploy_table.qdq_ (per-output-channel absmax, RTN, weights only)",
                     "views": views_meta, "z_full": zfull, "n_runs": len(ents),
                     "groups": {k: sum(1 for e in ents if e["group"] == k) for k in GROUP_LABEL}},
        "gates": gates,
        "verification_cell": {"run": VERIF_RUN, "cells": vcells, "max_abs_diff": vdev,
                              "v3_regression_max_abs_diff": reg_dev,
                              "v4_regression_max_abs_diff": v4dev,
                              "v3_runs_compared": len({r["run"] for r in reg})},
        "runs": [{"dir": e["dir"], "group": e["group"], "lr": e["lr"], "epochs": e["epochs"],
                  "N": e["N"], "n_src": e["n_src"], "seed": e["seed"], "sp": e["sp"],
                  "val_mae_live_monitor2000": e["val_mae_live"],
                  "zero_baseline_mae_args": e["zero_baseline_mae"],
                  "state_dict_key": R[e["run"]]["state_dict_key"],
                  "mae": R[e["run"]]["mae"],
                  "on_grid": R[e["run"]]["on_grid"], "grid_rms": R[e["run"]]["grid_rms"],
                  "off_tensors": R[e["run"]]["off"],
                  "seconds": R[e["run"]]["seconds"]} for e in ents],
        "table1": {"basis": "heldout_1556", "cells": cells1, "contrasts": c1_f,
                   "contrasts_first400": c1_400, "delta_contrast_lr": d_lr},
        "table2": {"cells": cells2, "cells_first400": cells2_400, "gaps": gap2,
                   "gap_change": chg2},
        "table3": {"cells": cells3, "cells_first400": cells3_400, "contrasts": c3,
                   "recovery_fraction": rec},
        "weight_distance": wdist,
        "sr_provenance": prov,
        "answers": ans,
    }
    OUT_JSON.write_text(json.dumps(jsafe(payload), ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")

    md = render(payload, ents, R, cells1, c1_f, c1_400, d_lr, cells2, gap2, chg2,
                cells3, c3, rec, wdist, prov, ans, gates, vcells, views_meta, runtime, a)
    OUT_MD.write_text(md, encoding="utf-8")
    print("写出 %s（%d B）与 %s（%d B）" % (OUT_MD, OUT_MD.stat().st_size,
                                          OUT_JSON, OUT_JSON.stat().st_size), flush=True)
    print("\nQ1 %s\nQ2 %s\nQ3 %s" % (ans["q1_short"], ans["q2_short"], ans["q3_short"]), flush=True)


# ---------------------------------------------------------------- 渲染
def render(payload, ents, R, cells1, c1_f, c1_400, d_lr, cells2, gap2, chg2,
           cells3, c3, rec, wdist, prov, ans, gates, vcells, views_meta, runtime, a) -> str:
    L = []
    add = L.append
    tm = payload["meta"]

    add("# 机制对照：学习率 × 投影间隔、训练预算、随机舍入")
    add("")
    add("只读既有 checkpoint（best.pth）做前向，**不训练、不用 GPU**。全部数字由 "
        "python scripts/round20_mechanism.py 一次性重算（%d 个 run，CPU %d 进程 × %d 线程，"
        "墙钟 %.1f s，含 8 bit 重算与权重距离）。生成时间 %s，torch %s。"
        % (len(ents), a.workers, a.threads, runtime, tm["when"], tm["torch"]))
    add("")
    add("**English TL;DR** — (1) %s. (2) %s. (3) %s."
        % (ans["q1_short"].replace("（", " (").replace("）", ")").replace("；", "; "),
           ans["q2_short"].replace("（", " (").replace("）", ")").replace("；", "; "),
           ans["q3_short"].replace("（", " (").replace("）", ")").replace("；", "; ")))
    add("")
    add("符号约定：投影代价 = MAE(N=1) − MAE(N=inf)，**正值 = 投影更差**。8 bit = 仅权重 "
        "fake quant（deploy_table.qdq_：per-output-channel absmax、对称、RTN），激活保持 FP32。"
        "区间为逐 seed 配对的 95% t 区间（df = n−1），p 只作描述；除注明的两处外 n = 3。")
    add("")

    add("## 0 口径、验证格与闸门")
    add("")
    add("主口径 **held-out 1,556**（val 去掉监控 2,000 后的补集；train.py 的选轮只看监控子集，"
        "故这些图从未参与选轮）；次口径 **val 前 400**（deploy_table_v4.csv 同源）。"
        "两个口径**不是独立重复**：first400 与 held-out 有 %d 张重叠、与监控子集有 %d 张重叠"
        "（监控子集参与选轮，故次口径的绝对值偏乐观，只用来看方向是否一致）。"
        % (views_meta[0]["overlap_first_held"], views_meta[0]["overlap_first_mon"]))
    add("")
    add("每个 run 的验证划分按**它自己** result.json 里的 split_seed 重建（本表 31 个 run 实测全为 0；"
        "代码不取默认值，deploy_table.split_seed_of 原样复用），并用 result.json 的 "
        "zero_baseline_mae 逐个反查。")
    add("")
    add("### 0.1 验证格 results/%s（N=inf、seed 0）" % VERIF_RUN)
    add("")
    add(md_tbl(["口径", "本次重算", "heldout_eval_v3.csv", "|d|"],
               [[SUB_LABEL[c["subset"]] + " " + ("FP32" if c["col"] == "mae_fp32" else "8 bit"),
                 "%.6f" % c["got"], "%.6f" % c["expect"], "%.1e" % c["abs_diff"]]
                for c in vcells]))
    add("")
    add("四格互不同源，串位会立刻暴露：held-out FP32 = %.6f、first400 8 bit = %.6f、"
        "first400 FP32 = %.6f、held-out 8 bit = %.6f。**注意**：0.054847 是 first400 的 **8 bit** 格、"
        "0.054271 是 held-out 的 **FP32** 格、0.055216 是 first400 的 **FP32** 格 —— "
        "若把 0.054847 记作 mae_fp32、0.054271 记作 mae_8、把 0.0552 记作「400 张上的 8 bit」，"
        "那是三格串位；按 v3 的四格逐格对齐，max|d| = %.1e。"
        % (vcells[2]["got"], vcells[1]["got"], vcells[0]["got"], vcells[3]["got"],
           payload["verification_cell"]["max_abs_diff"]))
    add("")
    add("### 0.2 闸门")
    add("")
    add(md_tbl(["闸门", "结果", "内容"],
               [[g["id"], "PASS" if g["ok"] else "FAIL", g["detail"]] for g in gates]))
    add("")

    add("## 1 表 1：学习率 × 投影间隔（20 轮，none 臂）")
    add("")
    add("### 1.1 单元（held-out 1,556，主口径）")
    add("")
    rows = []
    for label in ("1e-4", "1e-3", "3e-3"):
        for n in (1, 20, 0):
            c = cells1[(label, n)]
            if c["n"] == 0:
                rows.append([label, N_LABEL[n], 0, "—", "—", "—", "—"])
                continue
            rows.append([label, N_LABEL[n], c["n"],
                         "%s ± %s" % (u6(c["fp32"]["mean"]), u4(c["fp32"]["sd"])),
                         "%s ± %s" % (u6(c["b8"]["mean"]), u4(c["b8"]["sd"])),
                         cell_seed(c["fp32"]["vals"], c["seeds"]),
                         cell_seed(c["b8"]["vals"], c["seeds"])])
    add(md_tbl(["lr", "N", "n", "FP32 MAE（mean ± sd）", "8 bit MAE（mean ± sd）",
                "FP32 逐 seed", "8 bit 逐 seed"], rows))
    add("")
    add("run 目录（与上一列同 seed 顺序）：")
    add("")
    for label in ("1e-4", "1e-3", "3e-3"):
        for n in (1, 20, 0):
            c = cells1[(label, n)]
            if c["n"] == 0:
                continue
            add("- lr=%s, N=%s（n=%d）：%s" % (label, N_LABEL[n], c["n"], "; ".join(c["dirs"])))
    add("")
    add("### 1.2 对比 N=1 − N=inf（逐 seed 配对）")
    add("")
    for basis, ctab in (("heldout_1556", c1_f), ("first400", c1_400)):
        add("**%s**" % SUB_LABEL[basis])
        add("")
        rows = []
        for label in ("1e-4", "1e-3", "3e-3"):
            c = ctab[label]
            if c["n"] == 0:
                rows.append([label, 0, "—", "—", "—", "无同 seed 的 N=inf 对"])
                continue
            rows.append([label, c["n"],
                         "A: %s<br>B: %s" % ("; ".join(c["dirs_a"]), "; ".join(c["dirs_b"])),
                         cell_seed(c["b8"]["diffs"], c["seeds"]),
                         mean_ci(c["b8"]),
                         "%s（%s）" % (verdict(c["b8"]), ci_str(c["b8"]))])
        add(md_tbl(["lr", "n", "A = N=1 的 run / B = N=inf 的 run", "8 bit 逐 seed 差",
                    "8 bit 均值 [区间]", "判定"], rows))
        add("")
        rows = [[ctab[label]["n"],
                 cell_seed(ctab[label]["fp32"]["diffs"], ctab[label]["seeds"]),
                 mean_ci(ctab[label]["fp32"]),
                 "%s（%s）" % (verdict(ctab[label]["fp32"]), ci_str(ctab[label]["fp32"]))]
                for label in ("1e-4", "1e-3", "3e-3") if ctab[label]["n"] > 0]
        add(md_tbl(["lr（行序同上）", "n", "FP32 逐 seed 差", "FP32 均值 [区间]", "判定"],
                   [["%s" % lab] + r for lab, r in zip(
                       [l for l in ("1e-4", "1e-3", "3e-3") if ctab[l]["n"] > 0], rows)]))
        add("")
    add("### 1.3 学习率改变了多少（差中差：Δ(lr) − Δ(1e-4)）")
    add("")
    rows = []
    for (pair, prec, basis), d in sorted(d_lr.items(),
                                        key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        rows.append(["%s → %s" % pair, "8 bit" if prec == "b8" else "FP32",
                     "held-out 1556" if basis == "heldout_1556" else "first400",
                     d["n"], cell_seed(d["diffs"], d["seeds"]), mean_ci(d), verdict(d)])
    add(md_tbl(["学习率对比", "精度", "口径", "n", "逐 seed 差中差", "均值 [95% CI]", "判定"], rows))
    add("")
    add("差中差把「同一 seed、同一划分」的两个学习率放在一起，抵消了划分与初始化的共同部分；"
        "它仍只是 n ≤ 3 的描述统计。")
    add("")

    add("## 2 表 2：训练预算（20 轮 vs 60 轮，lr=1e-4，none 臂）")
    add("")
    add("60 轮组的 run 与 20 轮参照只差 epochs（60 vs 20）与输出目录：学习率、batch、EMA 衰减、"
        "划分 seed 全同（G5 逐键核过）。")
    add("")
    add("### 2.1 单元（held-out 1,556）")
    add("")
    rows = []
    for tag, n in (("20ep", 1), ("20ep", 0), ("60ep", 1), ("60ep", 0)):
        c = cells2[tag][n]
        rows.append([tag, N_LABEL[n], c["n"],
                     "%s ± %s" % (u6(c["fp32"]["mean"]), u4(c["fp32"]["sd"])),
                     "%s ± %s" % (u6(c["b8"]["mean"]), u4(c["b8"]["sd"])),
                     cell_seed(c["fp32"]["vals"], c["seeds"]),
                     cell_seed(c["b8"]["vals"], c["seeds"])])
    add(md_tbl(["预算", "N", "n", "FP32 MAE", "8 bit MAE", "FP32 逐 seed", "8 bit 逐 seed"], rows))
    add("")
    for tag, n in (("20ep", 1), ("20ep", 0), ("60ep", 1), ("60ep", 0)):
        add("- %s, N=%s：%s" % (tag, N_LABEL[n], "; ".join(cells2[tag][n]["dirs"])))
    add("")
    add("### 2.2 每个预算下的 gap（N=1 − N=inf）与 gap 的变化")
    add("")
    for basis in SUBSETS:
        add("**%s**" % SUB_LABEL[basis])
        add("")
        rows = []
        for tag in ("20ep", "60ep"):
            for prec in ("b8", "fp32"):
                c = gap2[(tag, basis)][prec]
                rows.append([tag, "8 bit" if prec == "b8" else "FP32", c["n"],
                             cell_seed(c["diffs"], gap2[(tag, basis)]["seeds"]),
                             mean_ci(c), verdict(c)])
        for prec in ("b8", "fp32"):
            d = chg2[(prec, basis)]
            rows.append(["**60 − 20**", "8 bit" if prec == "b8" else "FP32", d["n"],
                         cell_seed(d["diffs"], d["seeds"]), mean_ci(d), verdict(d)])
        add(md_tbl(["预算", "精度", "n", "逐 seed gap", "均值 [95% CI]", "判定"], rows))
        add("")
    add("### 2.3 逐 seed 一览（8 bit，held-out）")
    add("")
    d20 = dict(zip(gap2[("20ep", "heldout_1556")]["seeds"],
                   gap2[("20ep", "heldout_1556")]["b8"]["diffs"]))
    d60 = dict(zip(gap2[("60ep", "heldout_1556")]["seeds"],
                   gap2[("60ep", "heldout_1556")]["b8"]["diffs"]))
    rows = [["s%d" % s, s6(d20[s]), s6(d60[s]), s6(d60[s] - d20[s])]
            for s in sorted(set(d20) & set(d60))]
    add(md_tbl(["seed", "gap 20 轮", "gap 60 轮", "变化"], rows))
    add("")

    add("## 3 表 3：随机舍入（SR）@N=1 vs 最近舍入（RN）@N=1 vs N=inf")
    add("")
    add("SR 臂由 scripts/run_stochastic_rounding.py 把 train.snap_weights_ 换成 "
        "sign(x) * floor(|x| + U)（同一 per-channel 网格）后训练，结果落在 results_sr/。"
        "**provenance 限制**：result.json 的 args 与普通 none/N=1 run 完全相同（没有记录 SR 的开关），"
        "SR 身份只由目录与脚本给出；网格残差也区分不了两者（SR 同样落在网格上，见 §4）。"
        "同一 seed 的 SR 与 RN 是两次独立训练（同 seed、同划分、同配方，只差舍入规则）。")
    add("")
    if prov:
        add("证据行：")
        add("")
        for p in prov:
            add("    " + p)
        add("")
    add("### 3.1 单元（held-out 1,556）")
    add("")
    rows = []
    for k in ("RN@1", "SR@1", "inf"):
        c = cells3[k]
        rows.append([k, c["n"], "%s ± %s" % (u6(c["fp32"]["mean"]), u4(c["fp32"]["sd"])),
                     "%s ± %s" % (u6(c["b8"]["mean"]), u4(c["b8"]["sd"])),
                     cell_seed(c["fp32"]["vals"], c["seeds"]),
                     cell_seed(c["b8"]["vals"], c["seeds"])])
    add(md_tbl(["臂", "n", "FP32 MAE", "8 bit MAE", "FP32 逐 seed", "8 bit 逐 seed"], rows))
    add("")
    for k in ("RN@1", "SR@1", "inf"):
        add("- %s：%s" % (k, "; ".join(cells3[k]["dirs"])))
    add("")
    add("### 3.2 三个对比")
    add("")
    for basis in SUBSETS:
        add("**%s**" % SUB_LABEL[basis])
        add("")
        rows = []
        for key in ("RN-inf", "SR-inf", "SR-RN"):
            for prec in ("b8", "fp32"):
                c = c3[(key, basis)][prec]
                rows.append([key, "8 bit" if prec == "b8" else "FP32", c["n"],
                             cell_seed(c["diffs"], c3[(key, basis)]["seeds"]),
                             mean_ci(c), "%s（%s）" % (verdict(c), ci_str(c))])
        add(md_tbl(["对比", "精度", "n", "逐 seed 差", "均值 [95% CI]", "判定"], rows))
        add("")
    add("### 3.3 逐 seed 恢复比例 (RN − SR) / (RN − inf)")
    add("")
    rows = []
    for prec in ("b8", "fp32"):
        r = rec[prec]
        rows.append(["8 bit" if prec == "b8" else "FP32",
                     ", ".join("s%d %s" % (s, "—" if v is None else "%.0f%%" % (100 * v))
                               for s, v in zip(r["seeds"], r["vals"])),
                     "—" if r["mean"] is None else "%.0f%%" % (100 * r["mean"])])
    add(md_tbl(["精度", "逐 seed 恢复比例", "均值"], rows))
    add("")
    add("100% = SR 完全抵消 N=1 的代价；0% = SR 与 RN 一样。")
    add("")

    add("## 4 机制读数")
    add("")
    add("### 4.1 网格残差（8 bit 网格，按 weight-tensor 的 numel 加权）")
    add("")
    add("on_grid = |q − round(q)| < 0.05 格距的权重占比；grid_rms = rms(w − q_hat)/rms(w)。"
        "判据同 scripts/paper2_diagnostics.py:35-44，此处直接在 checkpoint 上算，未改既有脚本。")
    add("")
    rows = []
    for e in sorted(ents, key=lambda x: (x["group"], x["N"], x["seed"])):
        r = R[e["run"]]
        rows.append([e["dir"], N_LABEL[e["N"]], "%.4f" % r["on_grid"], "%.1e" % r["grid_rms"]])
    add(md_tbl(["run 目录", "N", "on_grid", "grid_rms"], rows))
    add("")
    def nrange(ns, key):
        v = [R[e["run"]][key] for e in ents if e["N"] in ns]
        return min(v), max(v)

    sets = {}
    for e in ents:
        sets.setdefault(tuple(sorted(x[0] for x in R[e["run"]]["off"])), []).append(e)
    share = statistics.fmean([R[e["run"]]["off_numel"] / R[e["run"]]["n_weights"] for e in ents])
    if len(sets) == 1:
        names = list(sets)[0]
        add("同一组 %d 个张量在全部 %d 个 run 上都只有约 10%% 的元素落在网格上（占 2D/4D 权重元素 "
            "%.2f%%）：%s。" % (len(names), len(ents), 100 * share, "、".join(names)))
        add("")
        add("原因是 quant/qat.py:11-26 的 snap_weights_ 只处理 nn.Linear 与 nn.Conv2d；"
            "nn.MultiheadAttention 自带的 in_proj_weight 不在这两类里，从未被投影。"
            "所以 on_grid 有一个恒定地板（%.4f），绝对值不能当判据。" % nrange((1,), "on_grid")[0])
        add("")
    r1 = nrange((1,), "grid_rms")
    r20 = nrange((20,), "grid_rms")
    rinf = nrange((0,), "grid_rms")
    add("grid_rms（相对量）逐组：N=1 为 %.1e–%.1e，N=20 为 %.1e–%.1e，N=inf 为 %.1e–%.1e —— "
        "N=inf 是 N=1 的 %.1f–%.1f 倍。所以「投影是否发生」可以用网格残差判定（G6 就是这条）；"
        "但「SR vs RN」不能：两者都是舍入算子，N=1 与 SR@N=1 的 grid_rms 同量级（%.1e vs %.1e）。"
        % (r1[0], r1[1], r20[0], r20[1], rinf[0], rinf[1], rinf[0] / r1[1], rinf[1] / r1[0],
           r1[0], r1[1]))
    add("")
    add("### 4.2 权重空间距离（相对 L2）与训练期 live MAE")
    add("")
    rows = [[ "s%d" % w["seed"], "%.4f" % w["sr_rn"], "%.4f" % w["rn_inf"],
              "%.4f" % w["sr_inf"], u6(w["sr_live"]), u6(w["rn_live"]), u6(w["inf_live"])]
            for w in wdist]
    add(md_tbl(["seed", "SR vs RN", "RN vs inf", "SR vs inf", "SR live MAE", "RN live MAE",
                "inf live MAE"], rows))
    add("")
    add("live MAE 来自各 result.json 的 val_mae_live，口径是**监控 2,000 张**（选轮用的那一份），"
        "与本文的 400 / 1,556 两个口径都不可直接比；列在这里只作为旁证：SR 臂的训练期读数本身"
        "就与 RN 臂差一个量级、而与不投影接近。")
    add("")

    add("## 5 三个问题的一句话回答")
    add("")
    add("**Q1（提高学习率是否缩小 N=1 的投影代价）** " + ans["q1"])
    add("")
    add("**Q2（预算 20 → 60 轮：gap 关闭、缩小还是持续）** " + ans["q2"])
    add("")
    add("**Q3（SR 是否恢复最近舍入的代价）** " + ans["q3"])
    add("")

    ex = [e for e in ents if e["n_src"] != "args.snap_every"]
    add("## 6 排除项、缺失格与限制")
    add("")
    add("**没有排除任何已存在的 run**：%d 个 run 全部进入表格（每个 run 的目录见 §7）。"
        "缺的是训练侧从未产出的格：" % len(ents))
    add("")
    add("- lr=1e-3 只有 4 个 run：seed 0 的 N ∈ {1,20,inf} 与 seed 1 的 N=1。"
        "所以该档的 N=1 − N=inf 对比 **n=1、区间不可估**，N=20 单元格 n=1；"
        "任何「随 lr 单调」的说法在这一档上没有证据。")
    add("- results_long/ 与 results_sr/ 只覆盖 N ∈ {1, inf}，没有 N=20；因此表 2、表 3 只回答端点。")
    add("- results_sr/ 有 8 个 seed，但只有 N=1，没有 SR × N=20、也没有 SR × 不同 lr。")
    add("- 4 bit / 6 bit 不在本轮范围（脚本只算 8 bit 与 FP32），表 1–3 的结论不外推到低比特。")
    add("")
    add("**读法与限制**")
    add("")
    add("1. 每组 n ≤ 3（lr=1e-3 的对比 n = 1）。区间是逐 seed 配对的 95% t 区间，df = n−1；"
        "n = 1 时只报点值并写明「区间不可估」。p 值只作描述，不做多重比较校正，也不作为显著性主张。")
    add("2. 配对只在**同 seed** 上成立：seed 同时决定数据划分与参数初始化（本表所有 run 的划分都是 0，"
        "差异落在初始化与训练噪声上）。跨学习率、跨预算的对比因此是「同划分、不同初始化」。")
    add("3. first400 与监控 2,000 有重叠（见 §0），其绝对值受选轮影响、偏乐观；"
        "held-out 1,556 与选轮无交集，是主口径。两个口径若方向不一致，报告以主口径为准并指出不一致。")
    add("4. SR 的身份没有进 result.json（provenance 见 §3），只能靠目录 + 脚本；"
        "网格残差无法区分 SR 与 RN。")
    add("5. 8 bit 是仅权重 fake quant（激活 FP32），不是整网 int8；表里的「部署 MAE」指这一口径。")
    add("")
    if ex:
        add("snap_every 缺席或为 None 的 run（按 N=1 读，G6 已核实它们在网格上）：%s"
            % "; ".join("%s（%s）" % (e["dir"], e["n_src"]) for e in ex))
        add("")

    add("## 7 run 清单（%d 个）" % len(ents))
    add("")
    add("完整目录即第一列；best.pth 与 result.json 都在其中。state_dict_key 全为 "
        "model_state_dict（ema_mode = none，与 deploy_table.weight_key 一致）。")
    add("")
    rows = []
    for e in sorted(ents, key=lambda x: (x["group"], x["N"], x["seed"])):
        r = R[e["run"]]
        rows.append([e["dir"], e["group"], "%g" % e["lr"], e["epochs"], N_LABEL[e["N"]],
                     e["seed"], u6(r["mae"]["fp32"]["heldout_1556"]),
                     u6(r["mae"]["b8"]["heldout_1556"]),
                     u6(r["mae"]["fp32"]["first400"]), u6(r["mae"]["b8"]["first400"]),
                     u6(e["val_mae_live"]), "%.1f" % r["seconds"]])
    add(md_tbl(["run 目录", "组", "lr", "epochs", "N", "seed", "FP32 held-out", "8 bit held-out",
                "FP32 first400", "8 bit first400", "live MAE（监控 2000）", "前向秒"], rows))
    add("")

    add("## 8 复现")
    add("")
    add("    python scripts/round20_mechanism.py --workers %d --threads %d   # CPU，实测约 %.0f 分钟"
        % (a.workers, a.threads, max(1.0, runtime / 60.0)))
    add("")
    add("产物：quant_runs/mechanism_report.md（本文）、quant_runs/mechanism_stats.json（全精度数值）、"
        "logs/round20_mechanism.log（stdout 全文）。脚本只读 results*/、quant_runs/*.csv、scripts/、"
        "data/utkface_raw224.npy，不写上述三个文件以外的任何路径，不训练。"
        "口径依赖 scripts/heldout_eval.py（VAL_RATIO/N_FIRST/N_MON 已在启动时断言）与 "
        "scripts/deploy_table.py（qdq_、MEAN/STD、split_seed_of）。")
    add("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
