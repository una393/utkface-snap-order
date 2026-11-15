"""权重量化吸附（weight snapping）与可选的激活伪量化。

snap 是投影：结果必须落在 int8 网格上，且幂等。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def snap_weights_(model: nn.Module, bits: int = 8) -> float:
    """把 Linear/Conv 权重就地吸附到按输出通道对称量化的网格。返回相对量化误差。"""
    qmax = 2 ** (bits - 1) - 1
    num = den = 0.0
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, (nn.Linear, nn.Conv2d)):
                continue
            w = m.weight.data
            flat = w.reshape(w.shape[0], -1)
            scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
            q = (flat / scale).round().clamp(-qmax - 1, qmax) * scale
            num += (q - flat).pow(2).sum().item()
            den += flat.pow(2).sum().item()
            m.weight.data = q.reshape(w.shape)
    return (num / den) ** 0.5 if den else 0.0


class FakeQuantAct(nn.Module):
    """对称激活伪量化；bound 由校准阶段统计后固定，避免每步漂移。"""

    def __init__(self, bits: int = 8):
        super().__init__()
        self.qmax = 2 ** (bits - 1) - 1
        self.register_buffer("bound", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = (self.bound / self.qmax).clamp_min(1e-8)
        return (x / scale).round().clamp(-self.qmax - 1, self.qmax) * scale
