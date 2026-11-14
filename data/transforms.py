"""UTKFace 预处理：Resize(224,224) + ImageNet 归一化。

人脸图是正方形，不做裁剪；与驾驶工作区的 384×224 不同——那边是宽屏道路。
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")
SIZE = (224, 224)


def preprocess(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img.convert("RGB").resize(SIZE, Image.BILINEAR), dtype="float32") / 255.0
    a = (a - MEAN) / STD
    return torch.from_numpy(a).permute(2, 0, 1)


def raw_to_tensor(arr) -> torch.Tensor:
    """uint8 HWC → 归一化 CHW（memmap 缓存路径）。几何已在缓存阶段做完。"""
    a = (np.asarray(arr, dtype="float32") / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1))
