"""控制头三选一。论文的唯二自变量之一。

契约：输入 (B, D)，输出 (B, 1) 的归一化转向量。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SoftClamp(nn.Module):
    def __init__(self, c: float = 3.0):
        super().__init__()
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c * torch.tanh(x / self.c)


class RMSNorm(nn.Module):
    """只做均方根缩放，不减均值——为的是消除 int8 上最脆弱的 Sub 算子。"""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class ControlHead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GELUHead(ControlHead):
    """H1：原始失效版。BN1d + 软裁剪 + GELU，三种算子都不可量化友好。"""

    def __init__(self, d: int, h: int = 256, clamp: float = 3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), SoftClamp(clamp),
            nn.GELU(), nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReLUQATHead(ControlHead):
    """H2：RMSNorm + QAT 期 ReLU 替换。ReLU 是 BPU 原生指令，但训练与部署行为不一致。"""

    def __init__(self, d: int, h: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), RMSNorm(h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BilinearGLUHead(ControlHead):
    """H3：无激活函数控制头。只有 matmul 与逐元素乘，不含任何需 LUT 近似的超越算子。

    注意：这里的"无误差"指不引入 LUT 近似误差源，不是 int8 逐元素乘没有舍入。
    """

    def __init__(self, d: int, h: int = 256):
        super().__init__()
        self.proj = nn.Linear(d, 2 * h)
        self.out = nn.Linear(h, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return self.out(a * b)


HEADS = {"gelu": GELUHead, "relu_qat": ReLUQATHead, "bilinear_glu": BilinearGLUHead}


def build_head(kind: str, d: int, h: int = 256) -> ControlHead:
    if kind not in HEADS:
        raise ValueError(f"unknown head {kind!r}, expected one of {sorted(HEADS)}")
    return HEADS[kind](d, h)


def param_report(d: int, h: int = 256) -> dict[str, int]:
    """参数增量必须是算出来的，不是手写的。"""
    return {k: cls(d, h).param_count() for k, cls in HEADS.items()}
