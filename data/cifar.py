"""CIFAR-10：32×32 十类分类，官方 10k test 集从不参与训练。

与 UTKFace 那条链的差别只有三处：标签是 int64、尺寸 32×32、**不划验证集**
（官方 test 就是 held-out）。吸附口径、影子、部署再量化全部复用既有实现。

缓存：D:/data/cifar10/{train32.npy,test32.npy,labels.json}（uint8 HWC，约 147 MB + 29 MB）。
"""
from __future__ import annotations

import json
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path("D:/data/cifar10")
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
RAW = ROOT / "cifar-10-batches-py"
# 官方 tarball 在本机只有 ~7 KB/s（单连接），同一批图像另有逐张 JPEG 的 GitHub 镜像：
# 内容是同一组 60,000 张图（JPEG 再编码），下载快两个数量级。缓存 npy 才是训练读的东西，
# 两条来源产出同一个 (N,32,32,3) uint8 数组，只差 JPEG 重编码的像素噪声。
IMG = ROOT / "CIFAR-10-images-master"
IMG_URL = "https://codeload.github.com/YoongiKim/CIFAR-10-images/tar.gz/refs/heads/master"
TRAIN_NPY = ROOT / "train32.npy"
TEST_NPY = ROOT / "test32.npy"
LABELS = ROOT / "labels.json"

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype="float32")
STD = np.array([0.2470, 0.2435, 0.2616], dtype="float32")
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def ensure_raw() -> Path:
    """解包到 D:/data/cifar10/；tar.gz 不在就先下载（170,498,071 B）。"""
    if RAW.is_dir():
        return RAW
    ROOT.mkdir(parents=True, exist_ok=True)
    tgz = ROOT / URL.rsplit("/", 1)[-1]
    if not tgz.exists():
        urllib.request.urlretrieve(URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(ROOT)
    return RAW


def _load_batch(p: Path) -> tuple[np.ndarray, list[int]]:
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(x), list(d[b"labels"])


def ensure_images() -> Path:
    """GitHub 逐图镜像（train/<class>/*.jpg, test/<class>/*.jpg）。"""
    if IMG.is_dir():
        return IMG
    ROOT.mkdir(parents=True, exist_ok=True)
    tgz = ROOT / "CIFAR-10-images.tar.gz"
    if not tgz.exists():
        urllib.request.urlretrieve(IMG_URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(ROOT)
    return IMG


def _from_images(root: Path, split: str) -> tuple[np.ndarray, list[int]]:
    from PIL import Image

    xs, ys = [], []
    for label, name in enumerate(CLASSES):
        files = sorted((root / split / name).glob("*.jpg"), key=lambda p: int(p.stem))
        xs.append(np.stack([np.asarray(Image.open(p).convert("RGB")) for p in files]))
        ys += [label] * len(files)
    return np.concatenate(xs), ys


def build_cache(force: bool = False) -> None:
    if not force and TRAIN_NPY.exists() and TEST_NPY.exists() and LABELS.exists():
        return
    if RAW.is_dir():
        src = "pickles"
        ensure_raw()
        tr = [_load_batch(RAW / f"data_batch_{i}") for i in range(1, 6)]
        tr_x, tr_y = np.concatenate([x for x, _ in tr]), sum((y for _, y in tr), [])
        te_x, te_y = _load_batch(RAW / "test_batch")
    else:
        src = "jpegs"
        root = ensure_images()
        tr_x, tr_y = _from_images(root, "train")
        te_x, te_y = _from_images(root, "test")
    np.save(TRAIN_NPY, tr_x)
    np.save(TEST_NPY, te_x)
    LABELS.write_text(json.dumps({"train": tr_y, "test": te_y, "source": src}))
    print(f"cache built from {src}: train {tr_x.shape} test {te_x.shape}", flush=True)


def labels_of(split: str) -> list[int]:
    build_cache()
    d = json.loads(LABELS.read_text())
    return d["train" if split == "train" else "test"]


def scan(split: str = "train") -> list[dict]:
    """records 口径与 UTKFace 同形（row/label/idx/group），供 train.py 的日志与计数复用。"""
    return [{"row": i, "label": int(l), "idx": i, "group": "cifar10"}
            for i, l in enumerate(labels_of(split))]


def split_train(n_val: int = 5000, seed: int = 0):
    """按类分层从 50k 训练集留出 n_val 张做选轮，官方 10k test 不参与任何选择。"""
    ly = labels_of("train")
    rng = np.random.default_rng(seed)
    per = [np.flatnonzero(np.asarray(ly) == c) for c in range(10)]
    take = n_val // 10
    val_idx = np.sort(np.concatenate([rng.permutation(q)[:take] for q in per]))
    mask = np.ones(len(ly), dtype=bool)
    mask[val_idx] = False
    return np.flatnonzero(mask), val_idx

class CifarDataset(Dataset):
    """返回 {"images": (1,3,32,32) float, "label": int64}。

    训练期增强：reflect pad 4 + 随机裁 32 + 水平翻转。随机数走 **torch** 的 RNG，
    这样 DataLoader 的 worker 种子（torch 管理）就足以复现同一条增强序列。
    """

    def __init__(self, split: str = "train", augment: bool = False, indices=None):
        build_cache()
        self.split = split
        self.augment = augment
        self.arr = np.load(TRAIN_NPY if split == "train" else TEST_NPY, mmap_mode="r")
        self._labels = labels_of(split)
        self.indices = None if indices is None else [int(i) for i in indices]
        self.labels = (self._labels if self.indices is None
                       else [self._labels[i] for i in self.indices])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        j = i if self.indices is None else self.indices[i]
        a = np.asarray(self.arr[j])
        if self.augment:
            a = np.pad(a, ((4, 4), (4, 4), (0, 0)), mode="reflect")
            y0 = int(torch.randint(0, 9, (1,)))
            x0 = int(torch.randint(0, 9, (1,)))
            a = a[y0:y0 + 32, x0:x0 + 32]
            if float(torch.rand(1)) < 0.5:
                a = a[:, ::-1]
        x = (a.astype("float32") / 255.0 - MEAN) / STD
        img = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)
        return {"images": img, "label": int(self.labels[i])}


if __name__ == "__main__":
    build_cache()
    tr, te = np.load(TRAIN_NPY, mmap_mode="r"), np.load(TEST_NPY, mmap_mode="r")
    ly = labels_of("train")
    print(f"train {tr.shape} test {te.shape} labels {len(ly)} "
          f"classes {len(set(ly))} min/max count "
          f"{min(ly.count(c) for c in range(10))}/{max(ly.count(c) for c in range(10))}")
    d = CifarDataset("train", augment=True)
    it = d[0]
    print("item", tuple(it["images"].shape), it["images"].dtype, it["label"])
