"""在 100 张校准图上复现/−1 值得不行就证伪：mean1 节点的 float vs fake-quant 余弦。"""
import glob, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from hmct.ir import load_model
from hmct.executor.ort2 import ORTExecutor2

ORTExecutor2.get_support_devices = classmethod(lambda cls: ["cpu"])
base = "/temporal/layers.0/norm2/LayerNormalization_mean1_output"
outs = [base, base + "_calibrated"]
ex = ORTExecutor2(load_model("/work/npu_quant/out/_probe_cal.onnx"))
ex.create_session()
iname = ex.get_inputs()[0].name
files = sorted(glob.glob("/work/npu_quant/calib/*.npy"))
print("calib 文件数:", len(files), flush=True)
res = {k: [] for k in outs}
for i, f in enumerate(files):
    x = np.load(f)[None, None].astype(np.float32)
    y = ex.inference({iname: x})
    for k in outs:
        res[k].append(np.asarray(y[k]).reshape(-1))
a = {k: np.concatenate(v) for k, v in res.items()}
f_, q_ = a[outs[0]], a[outs[1]]
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
print(f"float: n={f_.size} mean={f_.mean():.6g} std={f_.std():.6g} min={f_.min():.6g} max={f_.max():.6g}")
print(f"fq   : n={q_.size} mean={q_.mean():.6g} std={q_.std():.6g} min={q_.min():.6g} max={q_.max():.6g}")
print(f"cos(float, fake-quant) = {cos(f_, q_):.6f}")
print("max|d| =", float(np.abs(f_ - q_).max()), "| 唯一值个数 float:", len(np.unique(f_)), "fq:", len(np.unique(q_)))
# 用该节点自己报的阈值自己再量化一遍
for thr in (3.958074,):
    s = thr / 127.0
    dq = np.clip(np.round(f_ / s), -127, 127) * s
    print(f"阈值 {thr}: scale={s:.6g} cos(float, si8-sym)={cos(f_, dq):.6f} 唯一值={len(np.unique(dq))}")
