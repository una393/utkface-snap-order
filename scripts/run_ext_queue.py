"""两份 16-run 扩展队列（架构轴 / 数据轴），顺序执行、失败即停。

每条都是前台 subprocess 调用训练解释器；returncode 非 0 立即打印那条命令并退出，
不吞掉失败继续跑下一格（否则一张表里会混进「跑挂了但目录还在」的行）。

    python scripts/run_ext_queue.py --axis arch --dry-run
    python scripts/run_ext_queue.py --axis data

两条轴的格子完全相同：--seed × --ema-mode {none,after_snap} × --snap-every {1,0}。
--seeds 默认 1,2（既有 16 run 就是这两列）；数据轴扩到 8 个 seed 时用
--seeds 0,3,4,5,6,7 --skip-existing 续跑，已完成的 (seed, mode, snap) 记在
logs/_ext_queue_done.txt 里。
差别只在数据集/架构、val-windows 与学习率：arch 轴是 ResNet-18 × UTKFace（val-windows 2000，
lr 1e-4，与既有主实验同协议），data 轴是 DrivingNet × KonIQ-10k（val-windows 1110，使 held-out
恰好是剩下的 400 张：1510 - 1110 = 400，lr 1e-3）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PY = "C:/Users/USER/.conda/envs/e2e/python.exe"
ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "logs" / "_ext_queue_done.txt"

# 两条轴共用的固定超参（不许逐 run 改，改了就不是同一批对照）。--lr 不在这里，按轴给，
# 理由见 AXIS_EXTRA。
FIXED = ["--epochs", "20", "--batch-size", "64", "--amp",
         "--grad-clip", "1.0", "--ema-decay", "0.99", "--workers", "6",
         "--quant-mode", "overwrite", "--head", "relu_qat"]

AXIS_EXTRA = {
    # 与 results/ 里那批 UTKFace 主实验同一条数据路径（raw224 缓存）。lr 1e-3：预训练 ResNet-18
    # 在 1e-4 下 20 epoch 只到 live 0.128（3 epoch 时 0.173），明显欠训练；1e-3 的 3 epoch 冒烟
    # 已到 live 0.0662。两轴的学习率都按各自任务设定，轴内只有间隔与协议两个变量。
    "arch": ["--arch", "resnet", "--raw", "data/utkface_raw224.npy", "--val-windows", "2000",
             "--lr", "1e-3"],
    # koniq：--raw 默认由 train.py 换成 data/koniq_raw224.npy。lr 1e-3 —— 该任务 target 是
    # MOS/100（零基线 0.0044），照抄 1e-4 跑满 20 epoch 仍比零基线差（live 0.005234），
    # 而 1e-3 的 3 epoch 冒烟已明显低于零基线（live 0.0033）。
    "data": ["--dataset", "koniq", "--arch", "net", "--val-windows", "1110",
             "--lr", "1e-3"],
}


def jobs(axis: str, seeds=(1, 2)) -> list[list[str]]:
    """len(seeds) × 4 格 = seed × ema-mode × snap-every 的笛卡尔积。"""
    out = []
    for seed in seeds:
        for mode in ("none", "after_snap"):
            for snap in (1, 0):
                out.append([PY, "train.py", *FIXED, "--seed", str(seed),
                            "--ema-mode", mode, "--snap-every", str(snap), *AXIS_EXTRA[axis]])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=["arch", "data"], required=True)
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split(",") if x.strip())

    cmds = jobs(a.axis, seeds)
    print("axis=%s: %d runs%s" % (a.axis, len(cmds), " (dry-run)" if a.dry_run else ""), flush=True)
    done = set()
    if LEDGER.exists():
        done = {ln.strip() for ln in LEDGER.read_text(encoding="utf-8").splitlines() if ln.strip()}
    for i, cmd in enumerate(cmds, 1):
        line = " ".join(cmd)
        print("[%d/%d] %s" % (i, len(cmds), line), flush=True)
        if a.dry_run:
            continue
        cell = "%s,%s,%s,%s" % (a.axis, cmd[cmd.index("--seed") + 1],
                               cmd[cmd.index("--ema-mode") + 1],
                               cmd[cmd.index("--snap-every") + 1])
        if a.skip_existing and cell in done:
            print("skip (done): %s" % cell, flush=True)
            continue
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print("FAILED rc=%d after %.0fs: %s" % (rc, time.time() - t0, line), flush=True)
            sys.exit(rc if rc > 0 else 1)
        print("ok %.0fs" % (time.time() - t0), flush=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(cell + "\n")
        done.add(cell)
    print("DONE axis=%s (%d runs)" % (a.axis, len(cmds)), flush=True)


if __name__ == "__main__":
    main()
