"""简化上游：RegNetY 骨干 + 轻量 FPN + 空间池化 + 时序编码器 + 可插拔控制头。

这里保留张量形状与接口，但不是原 trans_RegNetY.py 的完整 Perceiver 版本。
本仓库的数字只在本仓库内部可比，见 README 的诚实边界。
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads import build_head


class LightweightFPN(nn.Module):
    def __init__(self, in_chs: tuple[int, int] = (320, 768), out: int = 256):
        super().__init__()
        self.lat = nn.ModuleList(nn.Conv2d(c, out, 1) for c in in_chs)
        self.norm = nn.GroupNorm(8, out)

    def forward(self, feats) -> torch.Tensor:
        c3, c4 = feats[-2], feats[-1]
        a = self.lat[0](c3)
        # 用 scale_factor 而不是 size=a.shape[-2:]：后者会让 ONNX 导出把空间维当动态，
        # 进而使 AdaptiveAvgPool2d 无法导出（shape 不可推断）。二级特征恰为 1/2，等价。
        b = F.interpolate(self.lat[1](c4), scale_factor=2, mode="bilinear", align_corners=False)
        if b.shape[-2:] != a.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.norm(a + b)


class DrivingNet(nn.Module):
    def __init__(self, head: str = "bilinear_glu", d: int = 256, h: int = 256,
                 grid: tuple[int, int] = (14, 24), seq: int = 8, layers: int = 2,
                 pretrained: bool = True):
        super().__init__()
        self.seq = seq
        self.backbone = timm.create_model(
            "regnety_008", pretrained=pretrained, features_only=True, out_indices=(2, 3)
        )
        chs = self.backbone.feature_info.channels()[-2:]
        self.fpn = LightweightFPN(tuple(chs), d)
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.frame_token = nn.Linear(d, d)
        enc = nn.TransformerEncoderLayer(d, 8, 4 * d, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(enc, layers)
        self.pos = nn.Parameter(torch.zeros(1, seq, d))
        self.head = build_head(head, d, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, hh, ww = x.shape
        feats = self.backbone(x.reshape(b * t, c, hh, ww))
        f = self.pool(self.fpn(feats)).flatten(2).mean(-1)
        tok = self.frame_token(f).reshape(b, t, -1) + self.pos[:, :t]
        return self.head(self.temporal(tok)[:, -1])
