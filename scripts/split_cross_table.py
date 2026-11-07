"""3x3（数据划分 seed x 初始化 seed）交叉设计的九格部署表：quant_runs/split_cross_table.csv。

四个缺的格（1,2)(1,3)(2,1)(2,3) 是本次补跑的新 run，用与 quant_runs/deploy_table_v2.csv
完全相同的部署量化协议评（直接 import scripts.deploy_table 的 MEAN/STD/qdq_/split_seed_of；
验证集按每个 run 自己的 split_seed 重建 split_random(recs, 0.15, split_seed)[1][:n]，n=400）。
已有五格（0,1)(0,2)(0,3)(1,1)(2,2) 的读数直接从 deploy_table_v2.csv 抄，不重算（避免漂移）。

写盘前一道闸门：既有 run utk-relu_qat-after_snap-s1-1793038259 的 fp32 / 8 bit 复现值
必须与 v2 表的 0.0593 / 0.0593 相差 <= 1e-4。差别更大说明划分或量化协议与 v2 不同源，
此时 exit 1 且不写任何文件。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from data.utkface import scan, split_random
from models.net import DrivingNet
from scripts.deploy_table import MEAN, STD, qdq_, split_seed_of

V2_TABLE = "quant_runs/deploy_table_v2.csv"
OUT = "quant_runs/split_cross_table.csv"
CAL_RUN = "utk-relu_qat-after_snap-s1-1793038259"
CAL_EXPECT = {"fp32": "0.0593", "8": "0.0593"}
CAL_TOL = 1e-4

# 已有五格：v2 表里 snap_every=20 且 ema_decay=0.99 且 mode=after_snap 的行
BASELINE = [
    (0, 1, "utk-relu_qat-after_snap-s1-1793112917"),
    (0, 2, "utk-relu_qat-after_snap-s2-1793113027"),
    (0, 3, "utk-relu_qat-after_snap-s3-1793114152"),
    (1, 1, "utk-relu_qat-after_snap-s1-1793038259"),
    (2, 2, "utk-relu_qat-after_snap-s2-1793040741"),
]
NEW_CELLS = [(1, 2), (1, 3), (2, 1), (2, 3)]
FIXED = {"head": "relu_qat", "ema_mode": "after_snap", "ema_decay": 0.99,
         "epochs": 20, "snap_every": 20, "quant_mode": "overwrite"}
METRICS = ["fp32", "8", "6", "4"]
COLUMNS = ["split_seed", "seed", "run", "source", "val_mae_train",
           "mae_fp32", "mae_8", "mae_6", "mae_4"]
SRC_OLD, SRC_NEW = "from_deploy_table_v2", "new_run"


def meta(run: Path) -> dict:
    return json.loads((run / "result.json").read_text(encoding="utf-8"))


def cell_of(args: dict, run: str) -> tuple[int, int]:
    seed = int(args.get("seed", 0))
    return split_seed_of(args, seed, run), seed


def find_new_runs() -> dict[tuple[int, int], Path]:
    """扫 results/ 找固定配置的四个新格；同一格多于一个目录即 raise（不猜）。"""
    hits: dict[tuple[int, int], list[Path]] = {}
    for d in sorted(Path("results").glob("utk-relu_qat-*")):
        rj = d / "result.json"
        if "-ste-" in d.name or not rj.exists():
            continue
        m = meta(d)
        a = m.get("args", {})
        if any(a.get(k) != v for k, v in FIXED.items()):
            continue
        cell = cell_of(a, d.name)
        if cell in NEW_CELLS:
            hits.setdefault(cell, []).append(d)
    for cell, ds in hits.items():
        if len(ds) > 1:
            raise SystemExit("%s 命中多个 run 目录，无法判定：%s" % (cell, [d.name for d in ds]))
    missing = [c for c in NEW_CELLS if c not in hits]
    if missing:
        raise SystemExit("缺 run 目录：%s（先跑 scripts/run_split_cross.py）" % missing)
    return {c: hits[c][0] for c in NEW_CELLS}


def load_net(run: Path) -> torch.nn.Module:
    key = "ema_state_dict" if meta(run)["args"]["ema_mode"] != "none" else "model_state_dict"
    ck = torch.load(run / "best.pth", map_location="cpu")
    net = DrivingNet(head="relu_qat", grid=(14, 14), seq=1)
    net.load_state_dict(ck[key], strict=False)
    net.eval()
    return net


def val_batch(recs, arr, split_seed: int, n: int, dev) -> tuple:
    sel = split_random(recs, 0.15, split_seed)[1][:n]
    xs, ys = [], []
    for r in sel:
        x = np.asarray(arr[int(r["row"])], dtype="float32") / 255.0
        x = (x - MEAN) / STD
        xs.append(torch.from_numpy(x.transpose(2, 0, 1)[None, None].copy()))
        ys.append(float(r["target"]))
    return torch.cat(xs).to(dev), torch.tensor(ys).to(dev)


def forward_all(net, x) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 64):
            outs.append(net(x[i:i + 64]).flatten())
    return torch.cat(outs)


def deploy_maes(run: Path, x, ys_t, bits) -> dict:
    net = load_net(run).to(x.device)
    with torch.no_grad():
        base = {k: v.detach().clone() for k, v in net.state_dict().items()}
        out = {"fp32": float((forward_all(net, x) - ys_t).abs().mean())}
        for b in bits:
            net.load_state_dict(base, strict=False)
            qdq_(net, b)
            out[str(b)] = float((forward_all(net, x) - ys_t).abs().mean())
        net.load_state_dict(base, strict=False)
    return out


def v2_rows(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return {r["run"]: r for r in csv.DictReader(f)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/utkface_raw224.npy")
    p.add_argument("--data", default="D:/data/utkface/UTKFace")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--out", default=OUT)
    p.add_argument("--v2", default=V2_TABLE)
    p.add_argument("--calib-only", action="store_true")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("device=%s n=%d" % (dev, a.n), flush=True)

    recs = scan(a.data)
    arr = np.load(a.raw, mmap_mode="r")
    v2 = v2_rows(Path(a.v2))
    cache: dict[int, tuple] = {}

    def batch(sp: int) -> tuple:
        if sp not in cache:
            cache[sp] = val_batch(recs, arr, sp, a.n, dev)
        return cache[sp]

    # 闸门：校准（任何新数字落盘之前）
    cal_sp, _ = cell_of(meta(Path("results") / CAL_RUN)["args"], CAL_RUN)
    xc, yc = batch(cal_sp)
    cal = deploy_maes(Path("results") / CAL_RUN, xc, yc, (8,))
    ok = True
    for k, want in CAL_EXPECT.items():
        diff = abs(cal[k] - float(want))
        print("calib %s %s=%.6f expect=%s diff=%.2e %s"
              % (CAL_RUN, k, cal[k], want, diff, "OK" if diff <= CAL_TOL else "FAIL"), flush=True)
        ok = ok and diff <= CAL_TOL
    del cal, xc, yc
    if not ok:
        print("CALIBRATION FAILED: 复现值与 deploy_table_v2.csv 不一致，未写任何文件", flush=True)
        sys.exit(1)
    if a.calib_only:
        print("calibration OK (calib-only)", flush=True)
        return

    rows = []
    for sp, seed, run in BASELINE:
        r = v2.get(run)
        if r is None:
            raise SystemExit("%s: 找不到 run=%s" % (a.v2, run))
        if int(r["snap_every"]) != 20 or r["mode"] != "after_snap" or r["ema_decay"] != "0.99" \
                or int(r["split_seed"]) != sp or int(r["seed"]) != seed:
            raise SystemExit("v2 行与格点不符：%s -> %s" % (run, r))
        rows.append({"split_seed": sp, "seed": seed, "run": run, "source": SRC_OLD,
                     "val_mae_train": r["val_mae_train"], "mae_fp32": r["mae_fp32"],
                     "mae_8": r["mae_8"], "mae_6": r["mae_6"], "mae_4": r["mae_4"]})
        print("copy %-42s sp=%d seed=%d %s" % (run, sp, seed, r["mae_fp32"]), flush=True)

    found = find_new_runs()
    for sp, seed in NEW_CELLS:
        d = found[(sp, seed)]
        x, ys_t = batch(sp)
        m = deploy_maes(d, x, ys_t, (8, 6, 4))
        live = round(float(meta(d)["val_mae_live"]), 4)
        print("run %-42s sp=%d seed=%d live=%.4f " % (d.name, sp, seed, live)
              + " ".join("%s=%.6f" % (k, m[k]) for k in METRICS), flush=True)
        rows.append({"split_seed": sp, "seed": seed, "run": d.name, "source": SRC_NEW,
                     "val_mae_train": live,
                     **{"mae_%s" % k: round(m[k], 4) for k in METRICS}})
        del x, ys_t

    rows.sort(key=lambda r: (r["split_seed"], r["seed"]))
    if len(rows) != 9:
        raise SystemExit("行数 %d != 9" % len(rows))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in COLUMNS})
    print("wrote %s (%d data rows)" % (out, len(rows)), flush=True)


if __name__ == "__main__":
    main()
