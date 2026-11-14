"""单图 + 单标量数据集（与驾驶工作区的序列窗口接口对齐，都返回 images / target）。"""
from __future__ import annotations

import os
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import preprocess, raw_to_tensor

_RAW_HANDLES: dict = {}


def open_raw(path: str):
    import numpy as np

    key = (os.getpid(), path)
    if key not in _RAW_HANDLES:
        _RAW_HANDLES[key] = np.load(path, mmap_mode="r")
    return _RAW_HANDLES[key]


class ImageScalarDataset(Dataset):
    def __init__(self, records: list[dict], transform: Callable | None = preprocess,
                 raw: str | None = None):
        self.records = records
        self.transform = transform
        self.raw = raw

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        if self.raw:
            img = raw_to_tensor(open_raw(self.raw)[r["row"]]).unsqueeze(0)
        else:
            t = self.transform(Image.open(r["path"])) if self.transform else \
                preprocess(Image.open(r["path"]))
            img = t.unsqueeze(0)
        return {"images": img, "target": torch.tensor(float(r["target"]))}
