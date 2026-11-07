"""ResNet-18 架构轴：把 torchvision resnet18 收成 [B, 1] 的标量回归器。

接口与 models/net.py 的 DrivingNet 对齐：输入 (B, T, C, H, W)，输出 (B, 1)。
时间维聚合取 **均值**：T = 1 时退化为单帧，与取最后一帧等价；T > 1 时给出
该窗口内的平均预测。输入沿用 data/transforms.py 的 ImageNet 归一化，
与 ImageNet 预训练权重同分布，无需额外包装。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNetReg(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        self.backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[0], x.shape[1]
        y = self.backbone(x.reshape(b * t, *x.shape[2:]))   # (B*T, num_classes)
        return y.reshape(b, t, -1).mean(dim=1)              # (B, num_classes)
