"""直通伪量化（STE）：参数永远是浮点，前向用网格上的权重算，梯度按恒等直通。

    wq = w + (snap(w) - w).detach()

snap 与 quant/qat.snap_weights_（qat.py:13-22）逐项同式：按输出通道对称、
scale = amax/qmax、qmax = 2**(bits-1)-1、clamp(-qmax-1, qmax)。
区别只有一处：这里**不写回**——w 全程是浮点，只由优化器更新，
没有任何一步把 wq 落到 m.weight.data 上（这正是 torch FakeQuant / Brevitas / LSQ 的训练语义）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

_PARA_KEY = ".parametrizations.weight.original"


def qmax_of(bits: int = 8) -> int:
    return 2 ** (bits - 1) - 1


def fake_quant_weight(w: torch.Tensor, qmax: int = 127) -> torch.Tensor:
    """snap_weights_ 的同式投影，但返回新张量、不改入参。"""
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    q = (flat / scale).round().clamp(-qmax - 1, qmax) * scale
    return q.reshape(w.shape)


class STEWeight(nn.Module):
    """parametrize 模块：(snap(w)-w).detach() 加法残差即直通梯度。"""

    def __init__(self, qmax: int = 127):
        super().__init__()
        self.qmax = qmax

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w + (fake_quant_weight(w, self.qmax) - w).detach()

    def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
        return w


def register_ste_(model: nn.Module, bits: int = 8) -> int:
    """给每个 Linear/Conv2d 挂 STE 伪量化。参数对象本身不变（parametrizations.weight.original）。"""
    qmax = qmax_of(bits)
    n = 0
    for m in list(model.modules()):
        if isinstance(m, (nn.Linear, nn.Conv2d)) and not parametrize.is_parametrized(m, "weight"):
            parametrize.register_parametrization(m, "weight", STEWeight(qmax))
            n += 1
    return n


def raw_weight(m: nn.Module) -> torch.Tensor:
    """参数本身。parametrize 下 m.weight 已是 wq，原始参数在 parametrizations.weight.original。"""
    if parametrize.is_parametrized(m, "weight"):
        return m.parametrizations.weight.original.detach()
    return m.weight.detach()


def plain_keys(sd: dict) -> dict:
    """把 ...parametrizations.weight.original 键还原成标准 ...weight（值 = 浮点原始参数）。"""
    return {k.replace(_PARA_KEY, ".weight"): v for k, v in sd.items()}


def weight_vs_snap(model: nn.Module, bits: int = 8) -> tuple[float, float]:
    """(相对 RMS 残差, 最大绝对残差)：参数到它自己的 snap 的距离。被投影过的模型 ≈ 0。"""
    qmax = qmax_of(bits)
    num = den = 0.0
    mx = 0.0
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, (nn.Linear, nn.Conv2d)):
                continue
            w = raw_weight(m).float()
            d = fake_quant_weight(w, qmax) - w
            num += d.pow(2).sum().item()
            den += w.pow(2).sum().item()
            mx = max(mx, d.abs().max().item())
    return ((num / den) ** 0.5 if den else 0.0), mx
