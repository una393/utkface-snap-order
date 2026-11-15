"""Conv2d + BatchNorm2d 折叠：折叠后的模型里不再有 BatchNorm2d。

BN 在 eval 下是逐通道仿射 y = (x - mu) / sqrt(var + eps) * gamma + beta，
可以精确吸收进前一个卷积的权重与偏置：

    W' = W * gamma / sqrt(var + eps)
    b' = (b - mu) * gamma / sqrt(var + eps) + beta

这不是近似（fp32 下只剩 1e-6 量级的舍入），只是把同一个函数写成一层。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _fold_conv_bn_(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BatchNorm2d 缺 running 统计（模型没在 eval 下跑过），无法折叠")
    ones = torch.ones(bn.running_var.shape, device=bn.running_var.device,
                      dtype=bn.running_var.dtype)
    gamma = bn.weight.data if bn.weight is not None else ones
    beta = bn.bias.data if bn.bias is not None else torch.zeros_like(ones)
    scale = gamma / torch.sqrt(bn.running_var + bn.eps)          # (C_out,)

    w = conv.weight.data
    conv.weight.data = w * scale.reshape(-1, *([1] * (w.dim() - 1)))
    bias = conv.bias.data if conv.bias is not None else torch.zeros(
        w.shape[0], device=w.device, dtype=w.dtype)
    folded = (bias - bn.running_mean) * scale + beta
    if conv.bias is None:
        conv.bias = nn.Parameter(folded)
    else:
        conv.bias.data = folded


def fold_bn_(net: nn.Module) -> nn.Module:
    """就地折叠 net 里所有 Conv2d→BatchNorm2d 相邻对，返回同一个 net。

    折叠前强制 net.eval() 并关闭梯度；被折叠的 BatchNorm2d 换成 nn.Identity，
    卷积保留原对象（权重/偏置改写）。调用两次是幂等的（第二次找不到 BN）。
    """
    net.eval()
    with torch.no_grad():
        for parent in list(net.modules()):
            prev_name, prev_conv = None, None
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Conv2d):
                    prev_name, prev_conv = name, child
                elif isinstance(child, nn.BatchNorm2d) and prev_conv is not None:
                    _fold_conv_bn_(prev_conv, child)
                    setattr(parent, name, nn.Identity())
                    prev_name, prev_conv = None, None
                else:
                    prev_name, prev_conv = None, None
    return net
