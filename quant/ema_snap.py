"""三种 EMA 更新时机。

吸附算子 S 与平均算子 E 不可交换：S(E[w]) != E(S[w])。导出链路取的是影子权重，
所以更新时机改变部署权重的具体数值——**但实测在任务指标上不可观测**（见
experiments.md 的部署表：同频率下差值 <=0.0018（FP32，seed 0 十二格），n=3 seed 复核仍为零结果；
权重空间相对 L2 差异 4.3-9.1%，而跨 seed 波动是 64-83%）。
篇 2 的立论因此改成「吸附频率是主旋钮 + 时机零结果」，不是「时机决定语义」。
"""
from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn

Mode = Literal["none", "before_snap", "after_snap"]
MODES: tuple[str, ...] = ("none", "before_snap", "after_snap")
Source = Literal["param", "forward"]
SOURCES: tuple[str, ...] = ("param", "forward")


class SnapEMA(nn.Module):
    """影子权重。mode 决定平均发生在吸附之前还是之后，source 决定平均的是谁。"""

    def __init__(self, model: nn.Module, decay: float = 0.999, mode: Mode = "after_snap",
                 source: Source = "param"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown ema mode {mode!r}, expected one of {MODES}")
        if source not in SOURCES:
            raise ValueError(f"unknown ema source {source!r}, expected one of {SOURCES}")
        self.mode: str = mode
        self.source: str = source
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # source=forward 只额外记一张「模块名 -> 权重键」映射：影子键与数值口径都不变，
        # 所以 state_dict_ema() 与 train.py 的 plain_keys(...) 都不用改。
        self._fwd_map: dict[str, str] = self._map_weights_(model) if source == "forward" else {}

    def _map_weights_(self, model: nn.Module) -> dict[str, str]:
        """模块名 -> 该模块权重在 shadow 里的键（参数化后是 ...parametrizations.weight.original）。"""
        out: dict[str, str] = {}
        for name, m in model.named_modules():
            if not isinstance(m, (nn.Linear, nn.Conv2d)):
                continue
            pre = f"{name}." if name else ""
            for cand in (f"{pre}weight", f"{pre}parametrizations.weight.original"):
                if cand in self.shadow:
                    out[name] = cand
                    break
        return out

    @torch.no_grad()
    def _forward_weights(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """按影子键索引的前向权重：parametrize 下 m.weight 就是 wq，不是原始参数。"""
        out: dict[str, torch.Tensor] = {}
        if not self._fwd_map:
            return out
        for name, m in model.named_modules():
            key = self._fwd_map.get(name)
            if key is not None and hasattr(m, "weight"):
                out[key] = m.weight.detach()
        return out

    @torch.no_grad()
    def update_(self, model: nn.Module, snap: Callable[[nn.Module], float] | None = None) -> float | None:
        """按 mode 执行一个优化步之后的平均/吸附。返回本次吸附误差（none 模式返回 None）。"""
        err = None
        if self.mode == "before_snap":
            self._avg_(model)
            if snap is not None:
                err = snap(model)
        elif self.mode == "after_snap":
            if snap is not None:
                err = snap(model)
            self._avg_(model)
        else:  # none：仍然做权重量化吸附，只是不维护影子权重
            if snap is not None:
                err = snap(model)
        return err

    @torch.no_grad()
    def _avg_(self, model: nn.Module) -> None:
        fwd = self._forward_weights(model)   # source=param 时是空字典，行为与改造前逐字节相同
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if s.is_floating_point():
                s.mul_(self.decay).add_(fwd.get(k, v).detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def state_dict_ema(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}
