"""CIFAR-10 的部署侧评估：fold_bn_ →（可选）qdq_ → 官方 10k test 的 top-1 错误率。

与 UTKFace 那条链共用同一套口径，只有两处不同：
* 模型是 10 类、32×32 stem 的 ResNet-18（--head 无关）；
* 官方 test 集本身就是 held-out，不需要按 seed 重建验证划分（分类没有 split_seed 语义）。

部署权重取值键与 deploy_table.py / ext_deploy.py 逐字一致：
ema_mode == "none" 取 model_state_dict，其余取 ema_state_dict；缺失就 raise，绝不回退。
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from data import cifar as cifar10
from models.resnet_reg import ResNetReg
from quant.bnfold import fold_bn_
from scripts.deploy_table import qdq_


def load_test(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.load(cifar10.TEST_NPY)                     # (10000, 32, 32, 3) uint8
    x = (x.astype("float32") / 255.0 - cifar10.MEAN) / cifar10.STD
    xt = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).unsqueeze(1).to(device)
    yt = torch.tensor(cifar10.labels_of("test"), device=device)
    return xt, yt


@torch.no_grad()
def top1_err(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
             bs: int = 1000) -> float:
    model.eval()
    bad = 0
    for i in range(0, len(y), bs):
        out = model(x[i:i + bs])
        bad += int((out.argmax(-1) != y[i:i + bs]).sum().item())
    return bad / len(y)


def build(state: dict, device: torch.device) -> torch.nn.Module:
    m = ResNetReg(num_classes=10, pretrained=False, cifar_stem=True)
    m.load_state_dict(state, strict=True)
    return fold_bn_(m).to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="显式 run 目录；缺省扫 results/cifar-*")
    ap.add_argument("--out", default="quant_runs/cifar_deploy.csv")
    ap.add_argument("--bits", type=int, nargs="+", default=[8])
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    runs = [Path(p) for p in a.runs] or sorted(Path("results").glob("cifar-*"))
    if not runs:
        raise SystemExit("没有可评估的 run：results/cifar-* 为空")

    x, y = load_test(device)
    rows = []
    for run in runs:
        rj = json.loads((run / "result.json").read_text(encoding="utf-8"))
        cfg = rj["args"]
        mode = cfg["ema_mode"]
        key = "ema_state_dict" if mode != "none" else "model_state_dict"
        ck = torch.load(run / "best.pth", map_location="cpu", weights_only=False)
        if key not in ck:
            raise SystemExit("%s: best.pth 缺 %s（不回退到另一个键）" % (run.name, key))
        base = build(ck[key], device)
        base_row = {"run": run.name, "seed": cfg["seed"], "ema_mode": mode,
                    "quant_mode": cfg["quant_mode"], "snap_every": cfg["snap_every"]}
        rows.append({**base_row, "bits": 0, "top1_err": top1_err(base, x, y)})
        for bits in a.bits:
            m = copy.deepcopy(base)
            qdq_(m, bits)
            rows.append({**base_row, "bits": bits, "top1_err": top1_err(m, x, y)})
        print(f"{run.name}: fp32 {rows[-1 - len(a.bits)]['top1_err']:.4f} "
              + " ".join(f"{b}bit {r['top1_err']:.4f}" for b, r in
                         zip(a.bits, rows[-len(a.bits):])), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "seed", "ema_mode", "quant_mode",
                                          "snap_every", "bits", "top1_err"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["snap_every"], r["seed"], r["bits"])))
    print(f"wrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
