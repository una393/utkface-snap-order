"""容器侧：在 x86 nash 仿真器上真的跑 .hbm，取预测值算 MAE。

每张样本一次 hbrt4-run-model-nash 进程（实测单次 ~35-39 s，按推理次数线性），
因此用多进程并行。用法: python3 eval_mae_runtime.py <tag> [k] [workers]
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = "/work"
RT_LIB = "/usr/local/lib/python3.10/dist-packages/hbdk4/runtime/x86_64_unknown_linux_gnu/nash/lib"
LD_PATH = ":".join([
    RT_LIB,
    "/usr/ucp",
    "/usr/local/lib/python3.10/dist-packages/hbdk4/compiler/_mlir_libs",
])

TAG = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 200
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
HBM = os.path.join(ROOT, "npu_quant", "out", TAG, TAG + ".hbm")
BINDIR = os.path.join(ROOT, "npu_quant", "testbin")
OUTDIR = tempfile.mkdtemp(prefix="rt_" + TAG + "_")


def run_one(idx):
    b = os.path.join(BINDIR, f"{idx:04d}.bin")
    od = os.path.join(OUTDIR, f"{idx:04d}")
    os.makedirs(od, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LD_PATH
    cmd = [
        "hbrt4-run-model-nash", "-f", HBM, "-n", TAG,
        "-i", b, "-o", od, "--numpy", "--log-level", "Error",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    npy = os.path.join(od, "age.npy")
    if not os.path.exists(npy):
        return idx, None, (p.stdout + p.stderr)[-500:]
    return idx, float(np.load(npy).ravel()[0]), None


if __name__ == "__main__":
    meta = json.load(open(os.path.join(BINDIR, "meta.json")))
    targets = np.array(meta["targets"], dtype=np.float64)[:K]
    preds = np.full(K, np.nan)
    with mp.Pool(WORKERS) as pool:
        for idx, val, err in pool.imap_unordered(run_one, range(K)):
            if val is None:
                print(f"[fail] {idx}: {err}", flush=True)
            else:
                preds[idx] = val
    ok = ~np.isnan(preds)
    err_abs = np.abs(preds[ok] - targets[ok])
    res = {
        "kind": "hbm_nash_x86_runtime_measured",
        "tag": TAG,
        "hbm": HBM,
        "n_requested": K,
        "n_ok": int(ok.sum()),
        "n_fail": int((~ok).sum()),
        "mae": float(err_abs.mean()),
        "rmse": float(np.sqrt(((preds[ok] - targets[ok]) ** 2).mean())),
        "max_abs_err": float(err_abs.max()),
        "preds": [None if np.isnan(v) else float(v) for v in preds],
    }
    out = os.path.join(ROOT, "npu_quant", "out", f"mae_runtime_{TAG}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"{TAG}: n_ok={res['n_ok']}/{K} MAE={res['mae']:.6f} RMSE={res['rmse']:.6f} max={res['max_abs_err']:.6f}", flush=True)
    print("wrote", out, flush=True)
