"""第 20 轮补跑队列（T2）：串行、可断点续跑、预算封顶、日志落盘。

分组（按此顺序跑；预算耗尽即停，不做半截 run）：
  A0  none         N=20   s0..s7   与 A3 配对，影子对比才能升到 n=8
  A1  none         N=1    s0..s7
  A2  none         N=inf  s0..s7   （--snap-every 0）
  A3  after_snap   N=20   s0..s7
  A4  before_snap  N=20   s0..s7
  B   ste          N=1    s0..s5
  C1a lr=3e-3      N=1/20/inf  s0..s2   --log-mech
  C2  epochs=60    N=1/inf     s0..s2   --log-mech
  C3  随机舍入     N=1    s0..s2        --log-mech（独立脚本注入）
  C1b lr=1e-3      N=1/20/inf  s0..s2   --log-mech
  E   head=gelu / bilinear_glu N=1 s0   （只为逐图预测取检查点）

落盘目录：A/B/E 写 results/（与既有 run 同口径）；C 组写 results_lr/ results_long/
results_sr/ —— 刻意不落 results/，因为 deploy_table.py 会 glob results/utk-relu_qat-*
并按 (mode, N, seed, decay, split) 建键；同键不同 lr/epochs 的 C 组混进去会污染部署表，
也会让 heldout_eval.py 的 pick() 因重复键抛错。

断点续跑：启动时扫描各 out 根的 result.json，按签名（head/mode/quant/snap/seed/epochs/
lr/decay/split/arch/log_mech）判重，已存在的配置直接跳过——所以杀掉进程后用同一命令行
重跑即可从断点继续。没跑完的 run（无 result.json）不算存在，会重跑。

用法：
  python scripts/run_round20_queue.py --dry-run       # 只打印计划与预计 GPU 小时
  python scripts/run_round20_queue.py                 # 正式串行跑
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = r"C:/Users/USER/.conda/envs/e2e/python.exe"
ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "round20_queue.log"
RUNDIR = ROOT / "logs" / "round20"
DONE = ROOT / "logs" / "round20_done.jsonl"
ROOTS = ("results", "results_lr", "results_long", "results_sr")
TIMEOUT_S = 4500            # 75 m 上限：20 轮 ≈ 15 m，60 轮 ≈ 45 m
TIMEOUT_LONG = 6600         # 60 轮那次实测 epoch 0 就要 215 s，稳态 60-85 s/轮，75 m 装不下
EST = {20: 16.0, 60: 55.0}  # 分钟/run 的保守估计（含 epoch 0 的 warmup）


def timeout_of(c: dict) -> int:
    return TIMEOUT_S if int(c["epochs"]) <= 20 else TIMEOUT_LONG


def norm_sig(d: dict) -> tuple:
    """配置签名。split_seed < 0 是 train.py 的「与 seed 相同」语义，归一化后再比较。"""
    seed = int(d.get("seed", 0))
    sp = int(d.get("split_seed", -1))
    return (str(d.get("head")), str(d.get("ema_mode")), str(d.get("quant_mode", "overwrite")),
            int(d.get("snap_every", 1)), seed, int(d.get("epochs", 15)),
            round(float(d.get("lr", 1e-4)), 12), round(float(d.get("ema_decay", 0.99)), 12),
            seed if sp < 0 else sp, str(d.get("arch", "net")), bool(d.get("log_mech", False)))


def existing() -> dict:
    """已存在的 run：签名 -> 目录名（只认有 result.json 的完整 run）。"""
    out = {}
    for root in ROOTS:
        d = ROOT / root
        if not d.is_dir():
            continue
        for rj in d.glob("*/result.json"):
            try:
                args = json.loads(rj.read_text(encoding="utf-8"))["args"]
            except Exception:
                continue
            out.setdefault(norm_sig(args), rj.parent.name)
    return out


def cfg(cid: str, group: str, out: str, head: str = "relu_qat", mode: str = "none",
        snap: int = 1, seed: int = 0, epochs: int = 20, lr: float = 1e-4,
        quant: str = "overwrite", mech: bool = False, sr: bool = False) -> dict:
    sig = norm_sig({"head": head, "ema_mode": mode, "quant_mode": quant, "snap_every": snap,
                    "seed": seed, "epochs": epochs, "lr": lr, "ema_decay": 0.99,
                    "split_seed": 0, "arch": "net", "log_mech": mech})
    return {"id": cid, "group": group, "out": out, "head": head, "mode": mode, "snap": snap,
            "seed": seed, "epochs": epochs, "lr": lr, "quant": quant, "mech": mech,
            "sr": sr, "sig": sig}


def plan() -> list:
    out = []
    for s in range(8):                       # A0 先跑：n=1 的两组主对比靠它补到 n=8
        out.append(cfg(f"A0_none_N20_s{s}", "A0", "results", mode="none", snap=20, seed=s))
    for s in range(8):
        out.append(cfg(f"A1_none_N1_s{s}", "A1", "results", mode="none", snap=1, seed=s))
    for s in range(8):
        out.append(cfg(f"A2_none_Ninf_s{s}", "A2", "results", mode="none", snap=0, seed=s))
    for s in range(8):
        out.append(cfg(f"A3_after_N20_s{s}", "A3", "results", mode="after_snap", snap=20, seed=s))
    for s in range(8):
        out.append(cfg(f"A4_before_N20_s{s}", "A4", "results", mode="before_snap", snap=20, seed=s))
    for s in range(6):
        out.append(cfg(f"B_ste_s{s}", "B", "results", mode="none", snap=1, seed=s, quant="ste"))
    for s in range(3):
        for snap, tag in ((1, "N1"), (20, "N20"), (0, "Ninf")):
            out.append(cfg(f"C1a_lr3e-3_{tag}_s{s}", "C1a", "results_lr", snap=snap, seed=s,
                           lr=3e-3, mech=True))
    for s in range(3):
        for snap, tag in ((1, "N1"), (0, "Ninf")):
            out.append(cfg(f"C2_ep60_{tag}_s{s}", "C2", "results_long", snap=snap, seed=s,
                           epochs=60, mech=True))
    for s in range(3):
        out.append(cfg(f"C3_sr_N1_s{s}", "C3", "results_sr", snap=1, seed=s, mech=True, sr=True))
    for s in range(3):
        for snap, tag in ((1, "N1"), (20, "N20"), (0, "Ninf")):
            out.append(cfg(f"C1b_lr1e-3_{tag}_s{s}", "C1b", "results_lr", snap=snap, seed=s,
                           lr=1e-3, mech=True))
    for head in ("gelu", "bilinear_glu"):    # E：逐图预测要的 N=1 检查点
        out.append(cfg(f"E_{head}_N1_s0", "E", "results", head=head, mode="none", snap=1, seed=0))
    return out


def cmd_of(c: dict) -> list:
    if c["sr"]:
        return [PY, "scripts/run_stochastic_rounding.py", "--seed", str(c["seed"])]
    cmd = [PY, "train.py", "--head", c["head"], "--ema-mode", c["mode"],
           "--snap-every", str(c["snap"]), "--epochs", str(c["epochs"]),
           "--batch-size", "64", "--lr", repr(c["lr"]), "--seed", str(c["seed"]),
           "--split-seed", "0", "--amp", "--grad-clip", "1.0", "--ema-decay", "0.99",
           "--raw", "data/utkface_raw224.npy", "--out", c["out"]]
    if c["quant"] == "ste":
        cmd += ["--quant-mode", "ste"]
    if c["mech"]:
        cmd += ["--log-mech"]
    return cmd


def log(line: str) -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {line}\n")
    print(f"[{stamp}] {line}", flush=True)


def new_runs(root: str, before: set) -> list:
    d = ROOT / root
    now = {p.parent.name for p in d.glob("*/result.json")} if d.is_dir() else set()
    return sorted(now - before)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-hours", type=float, default=23.0, help="墙钟预算（本任务上限 24 GPU 小时）")
    ap.add_argument("--groups", default="", help="只跑这些组，逗号分隔，如 A0,A1")
    ap.add_argument("--retries", type=int, default=2, help="同配置最多重试次数（1 次初跑 + 2 次重试）")
    a = ap.parse_args()

    have = existing()
    items = plan()
    if a.groups:
        keep = {g.strip() for g in a.groups.split(",") if g.strip()}
        items = [c for c in items if c["group"] in keep]
    todo, skip = [], []
    for c in items:
        (skip if c["sig"] in have else todo).append(c)
    sys.path.insert(0, str(ROOT))
    est_h = sum(EST[c["epochs"]] for c in todo) / 60.0
    print(f"plan: {len(items)} configs | skip {len(skip)} (已存在) | todo {len(todo)} "
          f"| 预计 {est_h:.1f} GPU 小时", flush=True)
    for c in skip:
        print(f"  SKIP {c['id']:<26} -> {have[c['sig']]}", flush=True)
    for c in todo:
        print(f"  TODO {c['id']:<26} out={c['out']} epochs={c['epochs']} lr={c['lr']}", flush=True)
    if a.dry_run:
        return

    RUNDIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log(f"START queue todo={len(todo)} skip={len(skip)} est={est_h:.1f}h max={a.max_hours}h")
    t_start = time.time()
    fails = 0
    for c in todo:
        used = (time.time() - t_start) / 3600.0
        if used >= a.max_hours:
            log(f"STOP budget used={used:.2f}h >= {a.max_hours}h, 余下 {len(todo)} 个不再启动")
            break
        before = {p.parent.name for p in (ROOT / c["out"]).glob("*/result.json")} \
            if (ROOT / c["out"]).is_dir() else set()
        rc, wall, attempt = None, 0.0, 0
        while attempt <= a.retries:
            attempt += 1
            t0 = time.time()
            logf = RUNDIR / f"{c['id']}.log"
            log(f"START id={c['id']} attempt={attempt} cmd={' '.join(cmd_of(c))} log={logf}")
            try:
                with logf.open("w", encoding="utf-8") as fh:
                    p = subprocess.run(cmd_of(c), cwd=str(ROOT), stdout=fh,
                                       stderr=subprocess.STDOUT, timeout=timeout_of(c))
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = "TIMEOUT"
            wall = time.time() - t0
            log(f"END id={c['id']} attempt={attempt} rc={rc} wall={wall:.0f}s")
            if rc == 0:
                break
        if rc != 0:
            fails += 1
            log(f"FAILED id={c['id']} attempts={attempt} rc={rc}")
            continue
        runs = new_runs(c["out"], before)
        with DONE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": c["id"], "group": c["group"], "out": c["out"],
                                "runs": runs, "wall_s": round(wall, 1), "ts": time.time()}) + "\n")
        log(f"OK id={c['id']} run={runs[-1] if runs else '?'} wall={wall:.0f}s")
    used = (time.time() - t_start) / 3600.0
    log(f"DONE queue wall={used:.2f}h fails={fails}")


if __name__ == "__main__":
    sys.exit(main())
