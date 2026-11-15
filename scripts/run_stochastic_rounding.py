"""随机舍入（stochastic rounding）覆写臂：独立脚本注入，不改 train.py 的默认行为。

做法：进程内把 train.py 模块全局的 snap_weights_ 换成随机舍入版，再调用 train.main()。
确定性 round() 取最近网格点；随机舍入按小数部分概率取上/下网格点，对投影是无偏的。
除舍入规则外，超参与 A1（none / N=1 / split 0）逐项一致，故两者只差「投影算子」这一项。

落盘目录 results_sr/：刻意不落 results/，否则 deploy_table.py 的 results/utk-relu_qat-*
glob 会把它们当成与 A1 同键的第二行（同键不同算法），污染部署表并让 heldout_eval 抛重复键。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn


def sr_snap_weights_(model: nn.Module, bits: int = 8) -> float:
    """把 Linear/Conv 权重按随机舍入投到 int8 网格，网格与 quant.qat.snap_weights_ 完全一致。"""
    qmax = 2 ** (bits - 1) - 1
    num = den = 0.0
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, (nn.Linear, nn.Conv2d)):
                continue
            w = m.weight.data
            flat = w.reshape(w.shape[0], -1)
            scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
            x = flat / scale
            r = torch.rand_like(x)                      # 随机舍入：floor(|x| + U[0,1)) 再还原符号
            q = (torch.sign(x) * torch.floor(x.abs() + r)).clamp(-qmax - 1, qmax) * scale
            num += (q - flat).pow(2).sum().item()
            den += flat.pow(2).sum().item()
            m.weight.data = q.reshape(w.shape)
    return (num / den) ** 0.5 if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--snap-every", type=int, default=1)
    ap.add_argument("--out", default="results_sr")
    a = ap.parse_args()

    import train as T
    T.snap_weights_ = sr_snap_weights_          # 只改本进程的绑定，train.py 文件本身不动
    sys.argv = ["train.py", "--head", "relu_qat", "--ema-mode", "none", "--ema-decay", "0.99",
                "--snap-every", str(a.snap_every), "--epochs", str(a.epochs),
                "--batch-size", "64", "--lr", "1e-4", "--seed", str(a.seed),
                "--split-seed", "0", "--amp", "--grad-clip", "1.0",
                "--raw", "data/utkface_raw224.npy", "--out", a.out, "--log-mech"]
    T.main()


if __name__ == "__main__":
    main()
