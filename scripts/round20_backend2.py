# -*- coding: utf-8 -*-
"""round20_backend2.py — 第二部署后端：ONNX Runtime 静态 QDQ int8（x86 CPU）。

独立实现的量化器：ORT 自带 quantize_static（QDQ、激活/权重 int8、per-channel 权重、
MinMax 校准、100 张 calib 图）。与厂商 hbdk4 链（per-tensor、同一批 100 张校准、
nash x86 指令级仿真）不是同一条实现：两者只能相互印证，不能互相替代。

只回答一个问题：训练侧测到的间隔效应（N=1 相对关闭投影 N=inf 更差）在 ORT int8
上是否仍然存在、保留多少。

两个臂（同一套检查点、同一批 200 张测试图）：
  full — ORT 默认：全图所有可量化算子都 int8（任务预注册口径）
  conv — 只量化 Conv（骨干 66 个卷积），Transformer 头与逐元素算子留 fp32
          （s0 单点诊断显示全图量化在本网络上崩塌，损坏集中在 MatMul/Gemm；
           见 quant_runs/backend2_report.md 的 "为什么有第二个臂"）

只读：results/ 的 16 个 best.pth、npu_quant/testbin（200 张 nash 同批输入）、
npu_quant/calib（100 张校准）、quant_runs/deploy_table_v4.csv（交叉校验）。
只写：quant_runs/backend2_report.json、quant_runs/backend2_preds/*。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)

from models.net import DrivingNet

SIZE = 224
NTHREADS = 4
OUT_DIR = ROOT / "quant_runs"
PRED_DIR = OUT_DIR / "backend2_preds"
JSON_OUT = OUT_DIR / "backend2_report.json"
NPZ_OUT = PRED_DIR / "backend2_preds.npz"
LAT_OUT = PRED_DIR / "backend2_latency.json"

# 固定划分（split_seed = 0）、ema_decay = 0.99、head = relu_qat、ema_mode = none。
# 目录来自 quant_runs/deploy_table_v4.csv 的五条主配置序列，脚本内再用 result.json 复核 args。
RUNS = [
    # series, snap_every, seed, run_dir
    ("none_N1",  1, 0, "utk-relu_qat-none-s0-1793011336"),
    ("none_N1",  1, 1, "utk-relu_qat-none-s1-1793445602"),
    ("none_N1",  1, 2, "utk-relu_qat-none-s2-1793446593"),
    ("none_N1",  1, 3, "utk-relu_qat-none-s3-1793447591"),
    ("none_N1",  1, 4, "utk-relu_qat-none-s4-1793448669"),
    ("none_N1",  1, 5, "utk-relu_qat-none-s5-1793449730"),
    ("none_N1",  1, 6, "utk-relu_qat-none-s6-1793450777"),
    ("none_N1",  1, 7, "utk-relu_qat-none-s7-1793451837"),
    ("none_Ninf", 0, 0, "utk-relu_qat-none-s0-1793111850"),
    ("none_Ninf", 0, 1, "utk-relu_qat-none-s1-1793452884"),
    ("none_Ninf", 0, 2, "utk-relu_qat-none-s2-1793453651"),
    ("none_Ninf", 0, 3, "utk-relu_qat-none-s3-1793454406"),
    ("none_Ninf", 0, 4, "utk-relu_qat-none-s4-1793455163"),
    ("none_Ninf", 0, 5, "utk-relu_qat-none-s5-1793455928"),
    ("none_Ninf", 0, 6, "utk-relu_qat-none-s6-1793456701"),
    ("none_Ninf", 0, 7, "utk-relu_qat-none-s7-1793457541"),
]

QUANT_CFG = {
    "api": "onnxruntime.quantization.quantize_static",
    "quant_format": "QuantFormat.QDQ",
    "per_channel": True,
    "reduce_range": False,
    "activation_type": "QuantType.QInt8",
    "weight_type": "QuantType.QInt8",
    "calibrate_method": "CalibrationMethod.MinMax",
    "extra_options": {"ActivationSymmetric": False, "WeightSymmetric": True},
    "calib_images": 100,
    "calib_source": "npu_quant/calib/*.npy (val[:100], split_random seed 0)",
    "test_images": 200,
    "test_source": "npu_quant/testbin/*.bin (rows[:200] of npu_quant/test_set.json, slice [800,1600])",
    "providers": ["CPUExecutionProvider"],
}

ARMS = {
    "full": {"op_types_to_quantize": None,
             "desc": "全图：ORT 默认可量化算子全部 int8（预注册主口径）"},
    "conv": {"op_types_to_quantize": ["Conv"],
             "desc": "只量化 Conv（骨干），Transformer 头与逐元素算子保留 fp32（部署可行口径）"},
}

# 不许改动的既有产物（脚本前后各算一次 sha256，用于自证未修改）
FROZEN_PATTERNS = [
    ("quant_runs", "*.csv"), ("quant_runs", "*.json"), ("quant_runs", "*.md"),
    ("quant_runs", "*.npz"), ("quant_runs", "perimage/*.npz"),
    ("scripts", "*.py"), ("npu_quant", "*.py"), ("npu_quant", "*.json"), ("npu_quant", "*.md"),
    ("npu_quant/calib", "*.npy"), ("models", "*.py"),
]


def say(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_frozen() -> dict:
    out = {}
    for sub, pat in FROZEN_PATTERNS:
        for p in sorted((ROOT / sub).glob(pat)):
            if p.name.startswith("backend2_"):
                continue
            rel = p.relative_to(ROOT).as_posix()
            out[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return out


def check_frozen(before: dict, after: dict) -> dict:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    return {"n_frozen_before": len(before), "n_frozen_after": len(after),
            "added": added, "removed": removed, "changed": changed,
            "unchanged": len(set(before) & set(after)) - len(changed)}


def verify_run_args(series: str, snap: int, seed: int, run: str) -> dict:
    """读 result.json 的 args 逐项核对；snap_every / split_seed 缺省按 train.py 默认值解释。"""
    p = ROOT / "results" / run / "result.json"
    if not p.exists():
        raise SystemExit(f"missing result.json: {p}")
    meta = json.loads(p.read_text(encoding="utf-8"))
    a = meta.get("args", {})
    got = {
        "head": a.get("head"),
        "ema_mode": a.get("ema_mode"),
        "ema_decay": a.get("ema_decay"),
        "seed": a.get("seed"),
        "snap_every": a.get("snap_every", 1),          # train.py: --snap-every 默认 1
        "split_seed": a.get("split_seed", -1),          # train.py: -1 == seed
        "quant_mode": a.get("quant_mode", "overwrite"), # train.py 默认 overwrite
    }
    split_eff = got["seed"] if got["split_seed"] == -1 else got["split_seed"]
    ok = (got["head"] == "relu_qat" and got["ema_mode"] == "none"
          and abs(float(got["ema_decay"]) - 0.99) < 1e-12
          and int(got["seed"]) == seed and int(got["snap_every"]) == snap
          and int(split_eff) == 0 and got["quant_mode"] == "overwrite")
    if not ok:
        raise SystemExit(f"args mismatch for {run}: {got} (want snap={snap} seed={seed} "
                         f"split=0 head=relu_qat mode=none decay=0.99)")
    ck = ROOT / "results" / run / "best.pth"
    if not ck.exists():
        raise SystemExit(f"missing best.pth: {ck}")
    got["split_seed_effective"] = int(split_eff)
    got["snap_every_from_args"] = "snap_every" in a
    got["val_mae_live"] = meta.get("val_mae_live")
    return got


def load_test_set() -> tuple[np.ndarray, np.ndarray, list[int], dict]:
    """200 张测试图：直接读 nash 同批裸 float32（npu_quant/testbin），并复核 row 列表。"""
    meta = json.loads((ROOT / "npu_quant/testbin/meta.json").read_text(encoding="utf-8"))
    ts = json.loads((ROOT / "npu_quant/test_set.json").read_text(encoding="utf-8"))
    rows, targets = meta["rows"], meta["targets"]
    if rows != ts["rows"][:200] or targets != ts["targets"][:200]:
        raise SystemExit("testbin meta 与 test_set.json rows[:200] 不一致")
    if meta["bytes_per_sample"] != 4 * 3 * SIZE * SIZE:
        raise SystemExit("testbin bytes_per_sample 异常")
    xs = np.stack([np.fromfile(ROOT / "npu_quant/testbin" / f"{i:04d}.bin",
                               dtype=np.float32).reshape(1, 3, SIZE, SIZE)
                   for i in range(200)]).astype(np.float32)   # (200, T=1, C, H, W)
    return xs, np.asarray(targets, dtype=np.float64), rows, meta


def load_calib() -> np.ndarray:
    files = sorted((ROOT / "npu_quant/calib").glob("*.npy"))
    if len(files) != 100:
        raise SystemExit(f"calib count {len(files)} != 100")
    arrs = [np.load(f).reshape(1, 3, SIZE, SIZE).astype(np.float32) for f in files]
    return np.stack(arrs)


def verify_calib_matches_val(calib: np.ndarray) -> dict:
    """100 张 calib 必须等于 split_random(...,0)[1][:100] 的 preprocess 结果。"""
    from PIL import Image
    from data.transforms import preprocess
    from data.utkface import scan, split_random
    recs = scan("D:/data/utkface/UTKFace")
    _, val = split_random(recs, 0.15, 0)
    worst = 0.0
    for i, r in enumerate(val[:100]):
        ref = preprocess(Image.open(r["path"])).numpy().astype(np.float32)
        worst = max(worst, float(np.abs(ref - calib[i, 0]).max()))
    if worst != 0.0:
        raise SystemExit(f"calib 与 val[:100] 不一致 max|diff|={worst:.3e}")
    return {"calib_matches_val_first100": True, "max_abs_diff": worst,
            "val_size": len(val)}


class CalibReader(CalibrationDataReader):
    def __init__(self, arrays: np.ndarray, name: str):
        self.items = [{name: arrays[i][None].astype(np.float32)} for i in range(arrays.shape[0])]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.items):
            return None
        self.i += 1
        return self.items[self.i - 1]

    def rewind(self):
        self.i = 0


def build_net(head: str, ckpt: Path) -> DrivingNet:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict")
    if sd is None:
        raise SystemExit(f"{ckpt}: 没有 model_state_dict")
    net = DrivingNet(head=head, grid=(14, 14), seq=1, pretrained=False)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{ckpt}: state_dict mismatch missing={missing} unexpected={unexpected}")
    net.eval()
    return net


def torch_preds(net: DrivingNet, xs: np.ndarray) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, xs.shape[0], 10):
            t = torch.from_numpy(xs[i:i + 10])
            outs.append(net(t).reshape(-1).numpy().astype(np.float64))
    return np.concatenate(outs)


def _sess(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = NTHREADS
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def ort_preds(sess: ort.InferenceSession, xs: np.ndarray) -> tuple[np.ndarray, float]:
    name = sess.get_inputs()[0].name
    ys, t0 = [], time.perf_counter()
    for i in range(xs.shape[0]):
        ys.append(float(sess.run(None, {name: xs[i:i + 1]})[0].reshape(-1)[0]))
    dt = time.perf_counter() - t0
    return np.asarray(ys, dtype=np.float64), 1000.0 * dt / xs.shape[0]


def run_one(series: str, snap: int, seed: int, run: str, xs: np.ndarray,
            ys_t: np.ndarray, calib: np.ndarray, tmp: Path) -> tuple[dict, np.ndarray, float]:
    """一个检查点：导出 ONNX → FP32 闸门 → 两个臂各自量化+推理。"""
    ckpt = ROOT / "results" / run / "best.pth"
    net = build_net("relu_qat", ckpt)

    fp32_onnx = tmp / f"{run}__fp32.onnx"
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    dummy = torch.zeros(1, 1, 3, SIZE, SIZE)
    with torch.no_grad():
        torch.onnx.export(net, dummy, str(fp32_onnx), opset_version=18,
                          input_names=["input"], output_names=["out"], dynamo=False)

    y_torch = torch_preds(net, xs)
    so32 = _sess(fp32_onnx)
    y32, lat32 = ort_preds(so32, xs)
    mae_t = float(np.abs(y_torch - ys_t).mean())
    mae32 = float(np.abs(y32 - ys_t).mean())
    gate = abs(mae32 - mae_t)
    maxpix = float(np.abs(y32 - y_torch).max())
    if gate > 1e-4:
        raise SystemExit(f"[GATE FAIL] {run}: |MAE_onnx - MAE_torch| = {gate:.3e} > 1e-4")
    del so32

    rows_out = {}
    for arm, spec in ARMS.items():
        int8_onnx = tmp / f"{run}__{arm}.onnx"
        t0 = time.perf_counter()
        quantize_static(str(fp32_onnx), str(int8_onnx), CalibReader(calib, "input"),
                        quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=False,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax,
                        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
                        op_types_to_quantize=spec["op_types_to_quantize"])
        quant_s = time.perf_counter() - t0

        so8 = _sess(int8_onnx)
        y8, lat8 = ort_preds(so8, xs)
        del so8
        mae8 = float(np.abs(y8 - ys_t).mean())
        c32, c8 = y32 - y32.mean(), y8 - y8.mean()
        cos = float(np.dot(y32, y8) / (np.linalg.norm(y32) * np.linalg.norm(y8) + 1e-12))
        cos_c = float(np.dot(c32, c8) / (np.linalg.norm(c32) * np.linalg.norm(c8) + 1e-12))
        rel_l2 = float(np.linalg.norm(y32 - y8) / (np.linalg.norm(y32) + 1e-12))

        row = {
            "arm": arm, "series": series, "snap_every": snap, "seed": seed, "run": run,
            "mae_torch_fp32": mae_t, "mae_onnx_fp32": mae32, "mae_onnx_int8": mae8,
            "gate_abs_diff_mae": gate, "gate_max_abs_pred_diff": maxpix,
            "delta_int8_minus_fp32": mae8 - mae32,
            "cosine": cos, "cosine_centered": cos_c, "rel_l2": rel_l2,
            "max_abs_pred_diff_int8_fp32": float(np.abs(y8 - y32).max()),
            "latency_ms_fp32": lat32, "latency_ms_int8": lat8,
            "quantize_seconds": quant_s,
            "onnx_fp32_bytes": fp32_onnx.stat().st_size, "onnx_int8_bytes": int8_onnx.stat().st_size,
        }
        say(f"[{arm}/{series}/s{seed}] torch32={mae_t:.6f} onnx32={mae32:.6f} int8={mae8:.6f} "
            f"gate={gate:.2e} d8-32={row['delta_int8_minus_fp32']:+.6f} "
            f"cos_c={cos_c:.6f} lat32={lat32:.1f}ms lat8={lat8:.1f}ms q={quant_s:.0f}s")
        int8_onnx.unlink(missing_ok=True)
        rows_out[arm] = {"row": row, "y8": y8}
    fp32_onnx.unlink(missing_ok=True)
    return rows_out, y32, lat32


def t_crit(df: int, p: float = 0.975) -> float:
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(p, df))
    except Exception:
        return {7: 2.364624251}.get(df, 2.0)


def sign_test(diffs) -> dict:
    d = list(diffs)
    n = len(d)
    neg = sum(1 for x in d if x < 0)
    pos = sum(1 for x in d if x > 0)
    zero = n - neg - pos
    nz = neg + pos
    k = min(neg, pos)
    if nz == 0:
        return {"n": n, "n_negative": neg, "n_positive": pos, "n_zero": zero, "p_exact_two_sided": 1.0}
    p_left = sum(math.comb(nz, i) for i in range(0, k + 1)) / float(2 ** nz)
    p_right = sum(math.comb(nz, i) for i in range(k, nz + 1)) / float(2 ** nz)
    return {"n": n, "n_negative": neg, "n_positive": pos, "n_zero": zero,
            "k_min_tail": k, "p_exact_two_sided": min(1.0, 2.0 * min(p_left, p_right))}


def describe(vals) -> dict:
    v = np.asarray(vals, dtype=np.float64)
    n = v.size
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    tc = t_crit(n - 1)
    return {"n": int(n), "mean": m, "sd": sd, "se": se,
            "t_crit_975": tc, "ci95": [m - tc * se, m + tc * se],
            "min": float(v.min()), "max": float(v.max())}


def _worker(task) -> dict:
    global NTHREADS
    series, snap, seed, run, nthreads = task
    NTHREADS = nthreads
    torch.set_num_threads(max(1, nthreads))
    xs, ys_t, _, _ = load_test_set()
    calib = load_calib()
    args_check = verify_run_args(series, snap, seed, run)
    with tempfile.TemporaryDirectory(prefix=f"backend2_{series}_s{seed}_") as td:
        tmp = Path(td)
        rows_by_arm, y32, lat32 = run_one(series, snap, seed, run, xs, ys_t, calib, tmp)
    return {"series": series, "seed": seed, "run": run, "args": args_check,
            "rows": {k: v["row"] for k, v in rows_by_arm.items()},
            "y8": {k: v["y8"] for k, v in rows_by_arm.items()},
            "y32": y32, "lat32": lat32}


def _stats_for_arm(rows_arm: list[dict]) -> dict:
    by = {(r["series"], r["seed"]): r for r in rows_arm}
    seeds = sorted({r["seed"] for r in rows_arm})
    if not all(("none_N1", s) in by and ("none_Ninf", s) in by for s in seeds):
        raise SystemExit("配对不完整：缺少某个 seed 的 N=1 或 N=inf 读数")
    d8 = [by[("none_N1", s)]["mae_onnx_int8"] - by[("none_Ninf", s)]["mae_onnx_int8"] for s in seeds]
    d32 = [by[("none_N1", s)]["mae_onnx_fp32"] - by[("none_Ninf", s)]["mae_onnx_fp32"] for s in seeds]
    dt = [by[("none_N1", s)]["mae_torch_fp32"] - by[("none_Ninf", s)]["mae_torch_fp32"] for s in seeds]
    dq = [by[("none_N1", s)]["delta_int8_minus_fp32"] - by[("none_Ninf", s)]["delta_int8_minus_fp32"]
          for s in seeds]
    ret_seed = [a8 / a32 for a8, a32 in zip(d8, d32)]
    return {
        "int8_interval_effect": {"desc": describe(d8), "sign_test": sign_test(d8),
                                 "per_seed": dict(zip(map(str, seeds), d8))},
        "fp32_onnx_interval_effect": {"desc": describe(d32), "sign_test": sign_test(d32),
                                      "per_seed": dict(zip(map(str, seeds), d32))},
        "fp32_torch_interval_effect": {"desc": describe(dt), "sign_test": sign_test(dt),
                                       "per_seed": dict(zip(map(str, seeds), dt))},
        "delta_int8_minus_fp32_per_seed": {
            "per_seed": {str(s): by[("none_N1", s)]["delta_int8_minus_fp32"] for s in seeds},
            "by_series": {ser: describe([by[(ser, s)]["delta_int8_minus_fp32"] for s in seeds])
                          for ser in ("none_N1", "none_Ninf")},
            "desc_all": describe([by[(ser, s)]["delta_int8_minus_fp32"]
                                  for ser in ("none_N1", "none_Ninf") for s in seeds]),
            "sign_test_all": sign_test([by[(ser, s)]["delta_int8_minus_fp32"]
                                        for ser in ("none_N1", "none_Ninf") for s in seeds]),
            "diff_of_deltas": {"desc": describe(dq), "sign_test": sign_test(dq),
                               "per_seed": dict(zip(map(str, seeds), dq))},
        },
        "retention": {
            "ratio_of_means": float(np.mean(d8)) / float(np.mean(d32)),
            "ratio_of_means_torch": float(np.mean(d8)) / float(np.mean(dt)),
            "per_seed": dict(zip(map(str, seeds), ret_seed)),
            "per_seed_desc": describe(ret_seed),
        },
        "series_summary": {ser: {k: describe([by[(ser, s)][k] for s in seeds])
                                 for k in ("mae_torch_fp32", "mae_onnx_fp32", "mae_onnx_int8",
                                           "delta_int8_minus_fp32", "cosine_centered")}
                           for ser in ("none_N1", "none_Ninf")},
    }


def latency_bench(runs: list[tuple]) -> dict:
    """单线程逐图延迟（CPU 相对参考，不是硬件结论）。只跑 s0 两个检查点。"""
    global NTHREADS
    NTHREADS = 1
    torch.set_num_threads(1)
    xs, ys_t, _, _ = load_test_set()
    calib = load_calib()
    out = {}
    for series, snap, seed, run in runs:
        net = build_net("relu_qat", ROOT / "results" / run / "best.pth")
        with tempfile.TemporaryDirectory(prefix="backend2_lat_") as td:
            tmp = Path(td)
            fp32 = tmp / "m.onnx"
            if hasattr(torch.backends, "mha"):
                torch.backends.mha.set_fastpath_enabled(False)
            with torch.no_grad():
                torch.onnx.export(net, torch.zeros(1, 1, 3, SIZE, SIZE), str(fp32), opset_version=18,
                                  input_names=["input"], output_names=["out"], dynamo=False)
            y32, lat32 = ort_preds(_sess(fp32), xs)
            ent = {"latency_ms_fp32_single_thread": lat32, "arms": {}}
            for arm, spec in ARMS.items():
                q = tmp / f"q_{arm}.onnx"
                quantize_static(str(fp32), str(q), CalibReader(calib, "input"),
                                quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=False,
                                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                                calibrate_method=CalibrationMethod.MinMax,
                                extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
                                op_types_to_quantize=spec["op_types_to_quantize"])
                y8, lat8 = ort_preds(_sess(q), xs)
                ent["arms"][arm] = {"latency_ms_int8_single_thread": lat8,
                                    "mae_int8": float(np.abs(y8 - ys_t).mean()),
                                    "cosine_centered": float(
                                        np.dot(y32 - y32.mean(), y8 - y8.mean()) /
                                        (np.linalg.norm(y32 - y32.mean()) * np.linalg.norm(y8 - y8.mean()) + 1e-12))}
        out[f"{series}/s{seed}"] = ent
        say(f"[lat/{series}/s{seed}] fp32={lat32:.1f}ms " +
            " ".join(f"{k}={v['latency_ms_int8_single_thread']:.1f}ms" for k, v in ent["arms"].items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑包含该子串的 run（调试用）")
    ap.add_argument("--dry", action="store_true", help="不写产物（调试用）")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4, help="每个进程的 ORT/torch 线程数")
    ap.add_argument("--lat", action="store_true", help="只跑 s0 两个检查点的单线程延迟")
    a = ap.parse_args()

    t_start = time.perf_counter()
    if a.lat:
        bench = latency_bench([r for r in RUNS if r[2] == 0 and r[3].endswith(("1793011336", "1793111850"))])
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        LAT_OUT.write_text(json.dumps({"threads": 1, "images": 200, "bench": bench}, indent=2,
                                      ensure_ascii=False), encoding="utf-8")
        say(f"[out] {LAT_OUT} {LAT_OUT.stat().st_size} B")
        return

    say(f"[ort] onnxruntime {ort.__version__} providers={ort.get_available_providers()}")
    say(f"[cfg] {json.dumps(QUANT_CFG, ensure_ascii=False)}")
    say(f"[arms] {json.dumps({k: v['desc'] for k, v in ARMS.items()}, ensure_ascii=False)}")

    frozen_before = snapshot_frozen()
    say(f"[frozen] snapshot {len(frozen_before)} files before")

    xs, ys_t, rows, tb_meta = load_test_set()
    calib = load_calib()
    calib_check = verify_calib_matches_val(calib)
    say(f"[data] test={xs.shape} calib={calib.shape} rows[:5]={rows[:5]} "
        f"targets[:5]={list(ys_t[:5])} calib_check={calib_check}")

    runs = RUNS if not a.only else [r for r in RUNS if a.only in r[3]]
    if not runs:
        raise SystemExit("no run matched --only")

    tasks = [(s, n, sd, rn, a.threads) for s, n, sd, rn in runs]
    if a.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_worker, tasks))
    else:
        results = [_worker(t) for t in tasks]

    rows_out, args_out, preds = [], {}, {}
    for res in results:
        args_out[res["run"]] = res["args"]
        for arm, row in res["rows"].items():
            rows_out.append(row)
            preds[f"{arm}__{res['run']}__fp32"] = res["y32"]
            preds[f"{arm}__{res['run']}__int8"] = res["y8"][arm]

    say(f"[time] {time.perf_counter() - t_start:.1f}s")
    if a.dry:
        print(json.dumps(rows_out, indent=2, ensure_ascii=False))
        return

    stats = {arm: _stats_for_arm([r for r in rows_out if r["arm"] == arm]) for arm in ARMS}

    gate = {r["run"]: {"abs_diff_mae": r["gate_abs_diff_mae"],
                       "max_abs_pred_diff": r["gate_max_abs_pred_diff"]} for r in rows_out}
    gate_max = max(v["abs_diff_mae"] for v in gate.values())

    xcheck = {}
    v4 = ROOT / "quant_runs/deploy_table_v4.csv"
    if v4.exists():
        import csv
        with v4.open(newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                if rec["run"] in args_out:
                    xcheck[rec["run"]] = {"v4_mae_fp32_400img": float(rec["mae_fp32"]),
                                          "v4_mae_8bit_400img": float(rec["mae_8"]),
                                          "v4_snap_every": int(rec["snap_every"]),
                                          "v4_mode": rec["mode"], "v4_split_seed": int(rec["split_seed"])}

    frozen_after = snapshot_frozen()
    frozen = check_frozen(frozen_before, frozen_after)
    say(f"[frozen] {json.dumps(frozen, ensure_ascii=False)}")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ_OUT, targets=ys_t, rows=np.asarray(rows, dtype=np.int64), **preds)
    preds_sha = sha256_file(NPZ_OUT)
    say(f"[out] {NPZ_OUT} {NPZ_OUT.stat().st_size} B sha256={preds_sha}")

    report = {
        "task": "second deployment backend: ONNX Runtime static QDQ int8 (x86 CPU)",
        "question": "does the training-side snap-interval effect (N=1 worse than N=inf) survive on ORT int8, and how much is retained?",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": {"python": sys.version.split()[0], "onnxruntime": ort.__version__,
                     "torch": torch.__version__, "numpy": np.__version__,
                     "onnx": __import__("onnx").__version__,
                     "providers": ort.get_available_providers()},
        "quant_config": QUANT_CFG,
        "arms": {k: {"op_types_to_quantize": v["op_types_to_quantize"], "desc": v["desc"],
                     "is_primary": k == "full"} for k, v in ARMS.items()},
        "data": {"test_rows": rows, "test_targets": [float(t) for t in ys_t],
                 "test_source": "npu_quant/testbin/*.bin == npu_quant/test_set.json rows[:200]",
                 "test_meta": {k: tb_meta[k] for k in ("k", "bytes_per_sample")},
                 "calib": calib_check, "n_calib": int(calib.shape[0])},
        "run_args_check": args_out,
        "fp32_gate": {"criterion": "|MAE_onnx_fp32 - MAE_torch_fp32| <= 1e-4 per checkpoint",
                      "max_abs_diff_mae": gate_max, "passed": bool(gate_max <= 1e-4),
                      "per_run": gate},
        "per_seed": rows_out,
        "stats": stats,
        "preds": {"path": NPZ_OUT.relative_to(ROOT).as_posix(),
                  "bytes": NPZ_OUT.stat().st_size, "sha256": preds_sha,
                  "keys": "targets, rows, <arm>__<run>__fp32, <arm>__<run>__int8"},
        "crosscheck_v4_csv": xcheck,
        "frozen_check": frozen,
        "wall_seconds": time.perf_counter() - t_start,
    }
    JSON_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    say(f"[out] {JSON_OUT} {JSON_OUT.stat().st_size} B")


if __name__ == "__main__":
    main()
