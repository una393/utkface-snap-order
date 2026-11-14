"""STE 臂的部署读数：n_val = 400，与论文表 4 同协议（只读复算，不训练）。

加载各 run 的 best.pth（none 模式取 model_state_dict），按该 run 自己的 seed 重建验证
划分 split_random(recs, 0.15, seed)[1][:n]，再复用 scripts/deploy_table.py 的 qdq_
（权重量化-反量化）评 FP32 / 8 / 6 / 4 bit 的部署 MAE。

写盘前有两道闸门（任一不满足即 exit 1 且不写文件）：
1. 校准：既有 run utk-relu_qat-none-s0-1793011336 的 FP32 与 8 bit 必须与
   quant_runs/deploy_table_v2.csv 的 0.0741 / 0.0741 相差 <= 1e-4。
2. 一致性：STE 三 seed 的均值与样本 sd（ddof=1）四舍五入到 4 位必须等于 EXPECT_MEAN / EXPECT_SD。
   均值与 sd 取自表内公开的 4 位小数读数（三行同精度，与 manuscript 的报数口径一致）；
   raw 均值另打印一行供核对——8 bit / 6 bit 的 raw 均值落在 4 位小数边界另一侧（0.055653 / 0.057048）。

表里 N = 1 与 N = infinity 两行直接从 quant_runs/deploy_table_v2.csv 抄，不重算（避免漂移）。
输出固定 9 列（arm, protocol, seed, mae_fp32, mae_8, mae_6, mae_4, n_val, run）：
均值行的 +-sd 不进表（保持 9 列验收口径），由 EXPECT_SD 断言与 stdout 汇总承担。
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
CAL_RUN = "results/utk-relu_qat-none-s0-1793011336"
CAL_EXPECT = {"fp32": 0.0741, "8": 0.0741}
CAL_TOL = 1e-4
STE_RUNS = [
    "results/utk-relu_qat-none-ste-s0-1793166213",
    "results/utk-relu_qat-none-ste-s1-1793167125",
    "results/utk-relu_qat-none-ste-s2-1793168098",
]
# (arm, protocol, run, deploy_table_v2.csv 里的 snap_every) —— 这两行只抄不算
BASELINES = [
    ("in_place_overwrite", "projection_interval_1", "results/utk-relu_qat-none-s0-1793011336", "1"),
    ("no_projection", "N_inf", "results/utk-relu_qat-none-s0-1793111850", "0"),
]
METRICS = ["fp32", "8", "6", "4"]
EXPECT_MEAN = {"fp32": 0.0560, "8": 0.0556, "6": 0.0571, "4": 0.1208}
EXPECT_SD = {"fp32": 0.0035, "8": 0.0030, "6": 0.0030, "4": 0.0152}
LABEL = {"fp32": "fp32", "8": "8bit", "6": "6bit", "4": "4bit"}
COLUMNS = ["arm", "protocol", "seed", "mae_fp32", "mae_8", "mae_6", "mae_4", "n_val", "run"]


def run_meta(run: str) -> tuple[int, int]:
    """(seed, split_seed)：split_seed 的 -1/缺省语义交给 deploy_table.split_seed_of。"""
    d = Path(run)
    args = json.loads((d / "result.json").read_text(encoding="utf-8"))["args"]
    seed = int(args.get("seed", 0))
    return seed, split_seed_of(args, seed, d.name)


def load_net(run: str) -> torch.nn.Module:
    args = json.loads((Path(run) / "result.json").read_text(encoding="utf-8"))["args"]
    key = "ema_state_dict" if args["ema_mode"] != "none" else "model_state_dict"
    ck = torch.load(Path(run) / "best.pth", map_location="cpu")
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


def deploy_maes(run: str, x, ys_t, bits) -> dict:
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


def v2_row(table: Path, run_name: str, snap_every: str) -> dict:
    with table.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["run"] == run_name and r["snap_every"] == snap_every:
                return r
    raise KeyError("%s: 找不到 run=%s snap_every=%s" % (table, run_name, snap_every))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/utkface_raw224.npy")
    p.add_argument("--data", default="D:/data/utkface/UTKFace")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--out", default="quant_runs/ste_deploy_table.csv")
    p.add_argument("--runs", nargs="+", default=STE_RUNS)
    p.add_argument("--v2", default=V2_TABLE)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("device=%s n=%d" % (dev, a.n), flush=True)

    recs = scan(a.data)
    arr = np.load(a.raw, mmap_mode="r")
    cache: dict[int, tuple] = {}

    def batch(split_seed: int) -> tuple:
        if split_seed not in cache:
            cache[split_seed] = val_batch(recs, arr, split_seed, a.n, dev)
        return cache[split_seed]

    # 闸门 1：校准（在任何 STE 数字落盘之前）
    _, cal_sp = run_meta(CAL_RUN)
    xc, yc = batch(cal_sp)
    cal = deploy_maes(CAL_RUN, xc, yc, (8,))
    for k, want in CAL_EXPECT.items():
        diff = abs(cal[k] - want)
        print("calib %s %s=%.6f expect=%.4f diff=%.2e %s"
              % (CAL_RUN, LABEL[k], cal[k], want, diff, "OK" if diff <= CAL_TOL else "FAIL"), flush=True)
        if diff > CAL_TOL:
            print("CALIBRATION FAILED: 与 deploy_table_v2.csv 不一致，未写任何文件", flush=True)
            sys.exit(1)
    del cal, xc, yc

    # STE 三 seed
    ste, cells, ste_rows = [], [], []
    for run in a.runs:
        seed, sp = run_meta(run)
        x, ys_t = batch(sp)
        m = deploy_maes(run, x, ys_t, (8, 6, 4))
        print("run %s seed=%d " % (Path(run).name, seed)
              + " ".join("%s=%.6f" % (LABEL[k], m[k]) for k in METRICS), flush=True)
        ste.append(m)
        cells.append({"mae_%s" % k: round(m[k], 4) for k in METRICS})
        ste_rows.append({"arm": "ste", "protocol": "ste", "seed": seed,
                         **cells[-1], "n_val": a.n, "run": run})
        del x, ys_t

    # 闸门 2：均值 / 样本 sd 的内部一致性（用表内 4 位小数读数，见文件头）
    mean = {k: float(np.mean([c["mae_%s" % k] for c in cells])) for k in METRICS}
    sd = {k: float(np.std([c["mae_%s" % k] for c in cells], ddof=1)) for k in METRICS}
    raw = {k: float(np.mean([s[k] for s in ste])) for k in METRICS}
    print("mean " + " ".join("%.6f" % mean[k] for k in METRICS), flush=True)
    print("sd   " + " ".join("%.6f" % sd[k] for k in METRICS), flush=True)
    print("mean(raw, 仅供参考) " + " ".join("%.6f" % raw[k] for k in METRICS), flush=True)
    for k in METRICS:
        if round(mean[k], 4) != EXPECT_MEAN[k] or round(sd[k], 4) != EXPECT_SD[k]:
            print("CONSISTENCY FAILED: %s mean=%.4f sd=%.4f expect %.4f/%.4f"
                  % (LABEL[k], mean[k], sd[k], EXPECT_MEAN[k], EXPECT_SD[k]), flush=True)
            sys.exit(1)

    def baseline_row(i: int) -> dict:
        arm, proto, run, every = BASELINES[i]
        r = v2_row(Path(a.v2), Path(run).name, every)
        return {"arm": arm, "protocol": proto, "seed": int(r["seed"]),
                **{"mae_%s" % k: float(r["mae_%s" % k]) for k in METRICS},
                "n_val": a.n, "run": run}

    rows = [baseline_row(0)]
    rows.extend(ste_rows)
    rows.append({"arm": "ste_mean", "protocol": "ste_mean", "seed": -1,
                 **{"mae_%s" % k: round(mean[k], 4) for k in METRICS},
                 "n_val": a.n, "run": ";".join(a.runs)})
    rows.append(baseline_row(1))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("%.4f" % row[k] if k.startswith("mae_") else row[k]) for k in COLUMNS})
    for row in rows:
        print("%-18s %s %s" % (row["arm"], row["protocol"],
                               " ".join("%.6f" % row["mae_%s" % k] for k in METRICS)), flush=True)
    print("wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
