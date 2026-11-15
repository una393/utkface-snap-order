"""held-out 1556 张重评：补集从未参与逐轮监控与最佳轮次选择。

train.py:66-68 用 Random(split_seed).sample(va_records, 2000) 从 3556 张验证图像里固定抽 2000 张
做逐轮监控与选轮；部署表用的是同一划分的前 400 张。补集 1556 张是干净的 held-out 集
（选轮过程完全没见过），用来消除「按监控集挑最佳轮次」带来的乐观偏差。

量化与划分口径完全复用 scripts/deploy_table.py：qdq_ / MEAN / STD / split_seed_of，
划分用 data.utkface.scan + split_random(recs, 0.15, split_seed)[1]，不另写一套实现。

闸门 A：utk-relu_qat-none-s0-1793011336 的 FP32 MAE 在 first400 上必须复现 0.0741（±1e-4），
        不通过就退出、不做任何后续评估。
闸门 B：补集恰为 1556 张，且与 2000 张监控子集的文件名交集为 0（按文件名集合验证）。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from data.utkface import scan, split_random          # noqa: E402
from models.net import DrivingNet                    # noqa: E402
import deploy_table as dt                            # noqa: E402

BITS_CORE = (8, 6, 4)          # 九格核心评四个比特宽；端点与 STE 只评 FP32 与 8 bit
BITS_LIGHT = (8,)
GATE_A_RUN = "utk-relu_qat-none-s0-1793011336"
GATE_A_EXPECT = 0.0741
GATE_A_TOL = 1e-4
N_FIRST = 400
N_MON = 2000
VAL_RATIO = 0.15
TABLE = Path("quant_runs/deploy_table_v2.csv")
STE_TABLE = Path("quant_runs/ste_deploy_table.csv")


def table_index() -> dict:
    idx: dict[tuple, list] = {}
    with TABLE.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["mode"], int(r["snap_every"]), int(r["seed"]),
                   float(r["ema_decay"]), int(r["split_seed"]))
            idx.setdefault(key, []).append(r)
    return idx


def pick(idx: dict, mode: str, N: int, seed: int, decay: float = 0.99, split: int | None = None):
    key = (mode, N, seed, decay, seed if split is None else split)
    rows = idx.get(key, [])
    if len(rows) != 1:
        raise RuntimeError("deploy_table_v2.csv 键 %s 命中 %d 行（需要恰好 1 行）" % (key, len(rows)))
    return rows[0]["run"]


def run_args(run: str) -> dict:
    return json.loads((Path("results") / run / "result.json").read_text(encoding="utf-8"))["args"]


def weight_key(ema_mode: str) -> str:
    """与 deploy_table.py 同一条判据：非 none 走影子权重，none 走在线权重。"""
    return "ema_state_dict" if ema_mode != "none" else "model_state_dict"


def build_views(recs: list[dict], sp: int) -> tuple[list[dict], list[dict], list[dict]]:
    """返回 (first400, monitoring 2000, heldout 1556)。监控子集的抽法与 train.py:66-68 逐字一致。"""
    val = split_random(recs, VAL_RATIO, sp)[1]
    first = val[:N_FIRST]
    mon = random.Random(sp).sample(val, N_MON) if len(val) > N_MON else list(val)
    mon_names = {Path(r["path"]).name for r in mon}
    held = [r for r in val if Path(r["path"]).name not in mon_names]
    inter = mon_names & {Path(r["path"]).name for r in held}
    return first, mon, held, val, mon_names, inter


def gather(arr, recs: list[dict]) -> np.ndarray:
    shape = arr.shape[1:]
    out = np.empty((len(recs),) + shape, dtype=np.uint8)
    for i, r in enumerate(recs):
        out[i] = arr[int(r["row"])]
    return out


def forward_all(net, u8: np.ndarray, mean_t, std_t, dev, bs: int = 64) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, len(u8), bs):
            x = torch.from_numpy(np.ascontiguousarray(u8[i:i + bs])).to(dev).float().div_(255.0)
            x = (x - mean_t) / std_t
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
            outs.append(net(x).flatten())
    return torch.cat(outs).double().cpu().numpy()


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(pred - target).mean())


def paired_stats(err_a: np.ndarray, err_b: np.ndarray) -> dict:
    d = err_a.astype(np.float64) - err_b.astype(np.float64)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / n ** 0.5)
    from scipy import stats as st
    df = n - 1
    half = float(st.t.ppf(0.975, df) * se)
    return {"n": n, "mean": m, "se": se, "ci_lo": m - half, "ci_hi": m + half,
            "p": float(2.0 * st.t.sf(abs(m / se), df)) if se > 0 else float("nan")}


def build_plan(idx: dict) -> list[dict]:
    plan: list[dict] = []
    for N in (1, 5, 20):
        for mode in ("none", "before_snap", "after_snap"):
            run = pick(idx, mode, N, 0)
            a = run_args(run)
            plan.append({"run": run, "mode": mode, "N": N, "seed": 0,
                         "sp": dt.split_seed_of(a, 0, run), "ema": a["ema_mode"],
                         "bits": list(BITS_CORE), "role": "core"})
    for mode in ("none", "after_snap"):
        run = pick(idx, mode, 0, 0)
        a = run_args(run)
        plan.append({"run": run, "mode": mode, "N": 0, "seed": 0,
                     "sp": dt.split_seed_of(a, 0, run), "ema": a["ema_mode"],
                     "bits": list(BITS_LIGHT), "role": "endpoint"})
    for d in sorted(Path("results").glob("utk-relu_qat-none-ste-s*")):
        if not (d / "best.pth").exists():
            continue
        a = run_args(d.name)
        seed = int(a.get("seed", 0))
        plan.append({"run": d.name, "mode": "ste", "N": int(a.get("snap_every", 1)), "seed": seed,
                     "sp": dt.split_seed_of(a, seed, d.name), "ema": a["ema_mode"],
                     "bits": list(BITS_LIGHT), "role": "ste"})
    return plan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/utkface_raw224.npy")
    p.add_argument("--data", default="D:/data/utkface/UTKFace")
    p.add_argument("--out", default="quant_runs/heldout_eval.csv")
    p.add_argument("--gate-only", action="store_true")
    p.add_argument("--bs", type=int, default=64)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("device: %s" % dev, flush=True)

    recs = scan(a.data)
    arr = np.load(a.raw, mmap_mode="r")
    mean_t = torch.tensor(dt.MEAN, dtype=torch.float32, device=dev).view(1, 1, 1, 3)
    std_t = torch.tensor(dt.STD, dtype=torch.float32, device=dev).view(1, 1, 1, 3)

    plan = build_plan(table_index())
    sps = sorted({e["sp"] for e in plan})
    print("plan: %d rows over split_seeds %s" % (len(plan), sps), flush=True)

    # ---------------- 闸门 B：划分与监控/补集 ----------------
    views = {}
    gate_b_ok = True
    for sp in sps:
        first, mon, held, val, mon_names, inter = build_views(recs, sp)
        views[sp] = (first, mon, held)
        ok = (len(held) == len(val) - len(mon)) and not inter
        gate_b_ok &= ok
        print("GATE B split_seed=%d: val=%d  monitor=%d  heldout=%d  overlap=%d  %s" % (
            sp, len(val), len(mon), len(held), len(inter), "OK" if ok else "FAIL"), flush=True)
    if not gate_b_ok:
        print("GATE B FAILED", flush=True)
        sys.exit(3)

    # ---------------- 闸门 A：first400 上复现部署表的 FP32 MAE ----------------
    entry = next(e for e in plan if e["run"] == GATE_A_RUN)
    first, mon, held = views[entry["sp"]]
    t0 = time.time()
    u8 = gather(arr, first + held)
    yt = np.array([float(r["target"]) for r in first + held])
    net = DrivingNet(head="relu_qat", grid=(14, 14), seq=1).to(dev)
    ck = torch.load(Path("results") / entry["run"] / "best.pth", map_location="cpu")
    net.load_state_dict(ck[weight_key(entry["ema"])], strict=False)
    net.eval()
    base = {k: v.detach().clone() for k, v in net.state_dict().items()}
    pred = forward_all(net, u8, mean_t, std_t, dev, a.bs)
    got = mae(pred[:N_FIRST], yt[:N_FIRST])
    dt_pass = abs(got - GATE_A_EXPECT) <= GATE_A_TOL
    print("GATE A %s: FP32 MAE(first400) = %.6f  expect %.4f  |diff| = %.2e  %s  (%.1fs)" % (
        entry["run"], got, GATE_A_EXPECT, abs(got - GATE_A_EXPECT),
        "OK" if dt_pass else "FAIL", time.time() - t0), flush=True)
    if not dt_pass:
        print("GATE A FAILED -> stop", flush=True)
        sys.exit(4)
    if a.gate_only:
        return

    preds: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict] = []
    t_all = time.time()
    done: dict[int, tuple] = {}
    for e in plan:
        sp = e["sp"]
        if sp not in done:
            first, mon, held = views[sp]
            done[sp] = (gather(arr, first + held),
                        np.array([float(r["target"]) for r in first + held]),
                        len(first), len(held))
        u8, yt, n_first, n_held = done[sp]

        t0 = time.time()
        net = DrivingNet(head="relu_qat", grid=(14, 14), seq=1).to(dev)
        ck = torch.load(Path("results") / e["run"] / "best.pth", map_location="cpu")
        key = weight_key(e["ema"])
        net.load_state_dict(ck[key], strict=False)
        net.eval()
        base = {k: v.detach().clone() for k, v in net.state_dict().items()}
        store = {}
        out = {}
        for tag in ["fp32"] + ["%dbit" % b for b in e["bits"]]:
            if tag != "fp32":
                net.load_state_dict(base, strict=False)
                dt.qdq_(net, int(tag[:-3]))
            y = forward_all(net, u8, mean_t, std_t, dev, a.bs)
            store[tag] = y
            out[tag] = (mae(y[:n_first], yt[:n_first]),
                        mae(y[n_first:n_first + n_held], yt[n_first:n_first + n_held]))
        net.load_state_dict(base, strict=False)
        preds[e["run"]] = store

        for subset, j in (("first400", 0), ("heldout_1556", 1)):
            rows.append({
                "run": e["run"], "mode": e["mode"], "N": e["N"], "split_seed": sp,
                "seed": e["seed"], "subset": subset, "n": n_first if j == 0 else n_held,
                "mae_fp32": "%.6f" % out["fp32"][j],
                "mae_8": "%.6f" % out["8bit"][j] if "8bit" in out else "",
                "mae_6": "%.6f" % out["6bit"][j] if "6bit" in out else "",
                "mae_4": "%.6f" % out["4bit"][j] if "4bit" in out else "",
            })
        print("%-46s N=%-3d %-10s fp32 %.4f/%.4f  8bit %.4f/%.4f  (%.1fs)" % (
            e["run"], e["N"], e["mode"], out["fp32"][0], out["fp32"][1],
            out["8bit"][0], out["8bit"][1], time.time() - t0), flush=True)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["run", "mode", "N", "split_seed", "seed", "subset", "n",
            "mae_fp32", "mae_8", "mae_6", "mae_4"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s (%d rows, %.1fs)" % (out_path, len(rows), time.time() - t_all), flush=True)

    # ---------------- 与现有部署表并列核对（first400 列） ----------------
    print()
    print("=== first400 核对：本脚本 vs 现有部署表 ===", flush=True)
    deploy = {(r["run"], k): r[k] for r in csv.DictReader(TABLE.open(encoding="utf-8"))
              for k in ("mae_fp32", "mae_8", "mae_6", "mae_4")}
    ste = {}
    if STE_TABLE.exists():
        for r in csv.DictReader(STE_TABLE.open(encoding="utf-8")):
            if r["arm"] == "ste_mean":      # 汇总行把三个 run 写在同一格，不能当逐 run 读数
                continue
            for pth in r["run"].split(";"):
                ste[Path(pth).name] = r
    bad = 0
    for r in rows:
        if r["subset"] != "first400":
            continue
        ref = deploy.get((r["run"], "mae_fp32"))
        for col, k in (("mae_fp32", "mae_fp32"), ("mae_8", "mae_8"), ("mae_6", "mae_6"), ("mae_4", "mae_4")):
            if not r[col]:
                continue
            mine = float(r[col])
            cand = ref if col in ("mae_fp32", "mae_8", "mae_6", "mae_4") and ref is not None else None
            exp = deploy.get((r["run"], col))
            if exp is None and r["run"] in ste:
                exp = ste[r["run"]][col]
            if exp is None:
                continue
            d = abs(mine - float(exp))
            if d > 1e-4:
                bad += 1
            print("  %-46s %-4s mine=%.6f table=%s |d|=%.1e %s" % (
                r["run"], col, mine, exp, d, "OK" if d <= 1e-4 else "MISMATCH"), flush=True)
    print("first400 mismatch(>1e-4): %d" % bad, flush=True)

    # ---------------- 派生统计（held-out 逐图配对） ----------------
    sp0 = 0
    u8_0, yt_all, nf0, nh0 = done[sp0]          # 预测向量是 first400 + heldout 的拼接，取尾部
    yt0 = yt_all[nf0:nf0 + nh0]
    sp_of = {e["run"]: e["sp"] for e in plan}
    held_err = {n: {t: np.abs(v[nf0:nf0 + nh0] - yt0) for t, v in s.items()}
                for n, s in preds.items() if sp_of[n] == sp0}

    def name(mode, N):
        for e in plan:
            if e["mode"] == mode and e["N"] == N and e["sp"] == sp0:
                return e["run"]
        return None

    print()
    print("=== held-out 1556：FP32 MAE 一览（split_seed 0）===", flush=True)
    for N in (1, 5, 20, 0):
        for mode in ("none", "before_snap", "after_snap"):
            n = name(mode, N)
            if n is None:
                continue
            print("  N=%-3d %-11s %-46s %.6f" % (N, mode, n, held_err[n]["fp32"].mean()), flush=True)

    print()
    print("=== 间隔效应（none，held-out FP32，N=1 vs N=inf）===", flush=True)
    n1, ninf = name("none", 1), name("none", 0)
    d = held_err[n1]["fp32"] - held_err[ninf]["fp32"]
    s = paired_stats(held_err[n1]["fp32"], held_err[ninf]["fp32"])
    print("  MAE N=1 %.6f -> N=inf %.6f  diff=%+.6f  paired mean=%+.6f se=%.6f CI=[%+.6f, %+.6f] p=%.3g" % (
        held_err[n1]["fp32"].mean(), held_err[ninf]["fp32"].mean(),
        held_err[n1]["fp32"].mean() - held_err[ninf]["fp32"].mean(),
        s["mean"], s["se"], s["ci_lo"], s["ci_hi"], s["p"]), flush=True)
    for N in (5, 20):
        nx = name("none", N)
        sx = paired_stats(held_err[nx]["fp32"], held_err[ninf]["fp32"])
        print("  MAE N=%d %.6f  diff vs N=inf %+.6f  paired %+.6f CI=[%+.6f, %+.6f]" % (
            N, held_err[nx]["fp32"].mean(),
            held_err[nx]["fp32"].mean() - held_err[ninf]["fp32"].mean(),
            sx["mean"], sx["ci_lo"], sx["ci_hi"]), flush=True)

    print()
    print("=== 顺序效应（N=20 seed 0，before_snap 对 after_snap，held-out 逐图配对，正值 = before 更差）===", flush=True)
    nb, na = name("before_snap", 20), name("after_snap", 20)
    for tag in ("fp32", "8bit", "6bit", "4bit"):
        if tag not in held_err[nb] or tag not in held_err[na]:
            continue
        s = paired_stats(held_err[nb][tag], held_err[na][tag])
        print("  %-5s MAE before %.6f / after %.6f  diff=%+.6f se=%.6f CI=[%+.6f, %+.6f] p=%.3f" % (
            tag, held_err[nb][tag].mean(), held_err[na][tag].mean(),
            s["mean"], s["se"], s["ci_lo"], s["ci_hi"], s["p"]), flush=True)
    for N in (1, 5):
        nb2, na2 = name("before_snap", N), name("after_snap", N)
        s = paired_stats(held_err[nb2]["fp32"], held_err[na2]["fp32"])
        print("  fp32  N=%-2d before %.6f / after %.6f  diff=%+.6f CI=[%+.6f, %+.6f] p=%.3f" % (
            N, held_err[nb2]["fp32"].mean(), held_err[na2]["fp32"].mean(),
            s["mean"], s["ci_lo"], s["ci_hi"], s["p"]), flush=True)

    print()
    print("=== 影子效应（模式 − none，held-out FP32 逐图配对，负值 = EMA 模式更好）===", flush=True)
    for N in (20, 0):
        n0 = name("none", N)
        for mode in ("before_snap", "after_snap"):
            nn = name(mode, N)
            if nn is None:
                continue
            s = paired_stats(held_err[nn]["fp32"], held_err[n0]["fp32"])
            print("  N=%-3d %-11s %+.6f se=%.6f CI=[%+.6f, %+.6f]" % (
                N, mode, s["mean"], s["se"], s["ci_lo"], s["ci_hi"]), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
