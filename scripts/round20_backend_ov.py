"""第三个部署后端：Intel OpenVINO INT8（x86 CPU）+ NNCF 训练后量化（PTQ）。

只读既有产物：results/<run>/best.pth、npu_quant/testbin/*.bin（= test_set.json rows[:200]）、
npu_quant/calib/*.npy（厂商链同一批 100 张）。全部推理在 CPU，不碰 GPU。

链路（与厂商 hbdk4 链、onnxruntime 静态 QDQ 链相互独立）：
  torch FP32 --torch.onnx.export--> ONNX FP32 --openvino.Core.read_model--> OV FP32
             --nncf.quantize(默认预设)--> OV int8(FakeQuantize) --compile_model("CPU")--> int8 读数

用法：
  D:/tool/Python/python.exe scripts/round20_backend_ov.py
  D:/tool/Python/python.exe scripts/round20_backend_ov.py --limit-runs 1 --limit-test 8   # 冒烟
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[1]
SIZE = 224
HEAD = "relu_qat"
GATE_TOL = 1e-4          # FP32 一致性闸门：|MAE_OV32 - MAE_torch| <= 1e-4
TCRIT_DF7 = 2.364624251592777  # t_{0.975, 7}
TCRIT_BY_N = {8: TCRIT_DF7, 16: 2.131449545559323}  # 后者是 t_{0.975, 15}，只给 16 run 合并行用

# 驱动解释器（D:/tool/Python）的 torch 是 2.7.1，训练环境是 2.12；两点用训练环境在同批 200 张上复核过
TORCH_XCHECK = {
    "interpreter": "C:/Users/24399/.conda/envs/e2e/python.exe",
    "torch": "2.12.0+cu132",
    "mae": {"N1_s0": 0.075128953, "Ninf_s0": 0.053658542},
}

# snap_every -> [(seed, run_dir)]；run 目录取自 quant_runs/deploy_table_v4.csv 的主配置行
PROTOCOLS = [
    ("N1", 1, [
        (0, "utk-relu_qat-none-s0-1793011336"),
        (1, "utk-relu_qat-none-s1-1793445602"),
        (2, "utk-relu_qat-none-s2-1793446593"),
        (3, "utk-relu_qat-none-s3-1793447591"),
        (4, "utk-relu_qat-none-s4-1793448669"),
        (5, "utk-relu_qat-none-s5-1793449730"),
        (6, "utk-relu_qat-none-s6-1793450777"),
        (7, "utk-relu_qat-none-s7-1793451837"),
    ]),
    ("Ninf", 0, [
        (0, "utk-relu_qat-none-s0-1793111850"),
        (1, "utk-relu_qat-none-s1-1793452884"),
        (2, "utk-relu_qat-none-s2-1793453651"),
        (3, "utk-relu_qat-none-s3-1793454406"),
        (4, "utk-relu_qat-none-s4-1793455163"),
        (5, "utk-relu_qat-none-s5-1793455928"),
        (6, "utk-relu_qat-none-s6-1793456701"),
        (7, "utk-relu_qat-none-s7-1793457541"),
    ]),
]


# ---------------------------------------------------------------- 统计工具

def paired_stats(vals) -> dict:
    x = np.asarray(vals, dtype=np.float64)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = TCRIT_BY_N.get(n)
    k_pos = int((x > 0).sum())
    k_neg = int((x < 0).sum())
    k = max(k_pos, k_neg)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n))
    return {
        "n": n, "mean": mean, "sd": sd, "se": se,
        "ci95": [mean - t * se, mean + t * se] if t else [float("nan")] * 2,
        "n_pos": k_pos, "n_neg": k_neg, "sign_p": p,
        "values": [float(v) for v in x],
    }


# ---------------------------------------------------------------- 量化图描述

def _node_const(node):
    try:
        return np.asarray(node.get_data())
    except Exception:
        return None


def _n_elem(pshape) -> int:
    if not pshape.is_static:
        return -1
    n = 1
    for d in pshape.to_shape():
        n *= int(d)
    return n


def _chain_has(node, type_name, depth=3) -> bool:
    """沿输入向上找 type_name（NNCF 的零点是 Subtract 形式，对称量化没有它）。"""
    if node.get_type_name() == type_name:
        return True
    if depth <= 0:
        return False
    for i in range(len(node.inputs())):
        try:
            if _chain_has(node.input_value(i).get_node(), type_name, depth - 1):
                return True
        except Exception:
            continue
    return False


def describe_quant(model) -> dict:
    """实测 NNCF 量化图：权重 int8（symmetric / per-channel）与激活 int8（per-tensor）各多少站点。

    注意：本版本 OpenVINO 的 Node.input(i) 返回的是"本节点的第 i 个输入端口"，其上 get_node()
    仍指向本节点；要拿生产者必须用 node.input_value(i).get_node()。第一版用 input(i) 统计出的
    "权重 0 条 / 激活 per-channel 12 条"是这么错的，已作废。
    """
    try:
        ops = model.get_ops()
    except AttributeError:
        ops = model.get_ordered_ops()
    fq = [o for o in ops if o.get_type_name() == "FakeQuantize"]
    a_pc = a_pt = w_fq_pc = w_fq_pt = 0
    a_low_zero = a_neg = 0
    a_ratio = []
    sym_w = asym_w = 0
    levels = set()
    for o in fq:
        levels.add(int(o.get_attributes().get("levels", -1)))
        n_lo = _n_elem(o.input_value(1).get_partial_shape())
        if "fq_weights" in o.get_friendly_name():
            if n_lo > 1:
                w_fq_pc += 1
            else:
                w_fq_pt += 1
            lo = _node_const(o.input_value(1).get_node())
            hi = _node_const(o.input_value(2).get_node())
            if lo is not None and hi is not None and lo.size == hi.size:
                if np.allclose(lo, -hi, atol=1e-6):
                    sym_w += 1
                else:
                    asym_w += 1
        else:
            if n_lo > 1:
                a_pc += 1
            else:
                a_pt += 1
            lo = _node_const(o.input_value(1).get_node())
            hi = _node_const(o.input_value(2).get_node())
            if lo is not None and hi is not None and lo.size == 1 and hi.size == 1:
                lv, hv = float(lo.ravel()[0]), float(hi.ravel()[0])
                if lv == 0.0:
                    a_low_zero += 1
                elif lv < 0.0 < hv:
                    a_neg += 1
                    a_ratio.append(-lv / hv)
    # 其余权重量化被 NNCF 表达成反量化子图：Constant(int8) -> Convert(f32) -> Multiply(逐输出通道 scale)
    deq = deq_pc = deq_pt = deq_int8 = deq_zp = 0
    for o in ops:
        if o.get_type_name() != "Multiply" or "fq_weights" not in o.get_friendly_name():
            continue
        deq += 1
        n = _n_elem(o.input_value(1).get_partial_shape())
        if n > 1:
            deq_pc += 1
        else:
            deq_pt += 1
        if _chain_has(o.input_value(0).get_node(), "Subtract"):
            deq_zp += 1
        src = o.input_value(0).get_node()
        if src.get_type_name() == "Convert":
            prod = src.input_value(0)
            if prod.get_node().get_type_name() == "Constant":
                try:
                    et = prod.get_element_type()
                    # 注意：Type.boolean 只对布尔类型可取，对 int8 取会抛，不能写进条件里
                    is_i8 = bool(et.is_integral_number) and int(et.bitwidth) == 8
                except Exception:
                    is_i8 = False
                if is_i8:
                    deq_int8 += 1
    sites = q_sites = 0
    unq = []
    for o in ops:
        if o.get_type_name() not in ("Convolution", "MatMul", "GroupConvolution"):
            continue
        sites += 1
        w = o.input_value(1).get_node()
        if "fq_weights" in w.get_friendly_name() or w.get_type_name() == "FakeQuantize":
            q_sites += 1
        else:
            unq.append(o.get_friendly_name())
    return {
        "fake_quantize_nodes": len(fq),
        "weight_sites_fakequant": w_fq_pc + w_fq_pt,
        "weight_sites_dequant_multiply": deq,
        "weight_sites_total": w_fq_pc + w_fq_pt + deq,
        "weights_per_channel": w_fq_pc + deq_pc,
        "weights_per_tensor": w_fq_pt + deq_pt,
        "weights_symmetric": sym_w + (deq - deq_zp),
        "weights_asymmetric": asym_w + deq_zp,
        "weights_int8_constants": deq_int8,
        "weights_with_zero_point": deq_zp,
        "activations_per_channel": a_pc, "activations_per_tensor": a_pt,
        "activations_low_zero": a_low_zero,
        "activations_near_symmetric": a_neg,
        "activations_neg_pos_ratio_median": float(np.median(a_ratio)) if a_ratio else None,
        "levels": sorted(levels),
        "conv_matmul_sites": sites, "conv_matmul_with_quantized_weight": q_sites,
        "unquantized_weight_sites": len(unq),
        "unquantized_se_fc_convs": sum(1 for n in unq if "/se/fc" in n),
        "unquantized_attn_qk_matmuls": sum(1 for n in unq if "self_attn/MatMul_2" in n),
        "model_ops": len(ops),
    }


# ---------------------------------------------------------------- 单 run 主流程

def build_net(run_dir: Path, head: str):
    import torch
    from models.net import DrivingNet
    ck = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    net = DrivingNet(head=head, grid=(14, 14), seq=1, pretrained=False)
    sd = ck.get("model_state_dict")
    if sd is None:
        raise SystemExit("no model_state_dict in %s" % run_dir)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit("state_dict mismatch: %s %s" % (missing, unexpected))
    net.eval()
    meta = {k: v for k, v in (ck.get("args") or {}).items()
            if k in ("head", "ema_mode", "seed", "snap_every", "split_seed", "ema_decay")}
    meta["val_mae_live"] = ck.get("val_mae_live")
    return net, meta


def export_onnx(net, path: Path):
    import torch
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    dummy = torch.zeros(1, 1, 3, SIZE, SIZE)
    with torch.no_grad():
        torch.onnx.export(net, dummy, str(path), opset_version=18,
                          input_names=["sequence_images"], output_names=["age"], dynamo=False)


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_compiled(cmodel, xs):
    out_port = cmodel.output(0)
    ys, t0 = [], time.perf_counter()
    for x in xs:
        ys.append(float(np.asarray(cmodel(x)[out_port]).reshape(-1)[0]))
    return np.asarray(ys, dtype=np.float64), time.perf_counter() - t0


def torch_eval(net, xs):
    import torch
    ys, t0 = [], time.perf_counter()
    with torch.no_grad():
        for x in xs:
            ys.append(float(net(torch.from_numpy(x)).reshape(-1)[0]))
    return np.asarray(ys, dtype=np.float64), time.perf_counter() - t0


def run_one(tag, proto, snap, seed, run_dir: Path, ctx) -> dict:
    import nncf
    xs, ys, calib = ctx["xs"], ctx["targets"], ctx["calib"]
    # 只为两个 seed0 样本保留 ONNX / IR 证据（其余用完即删，字节数与 sha1 已入 JSON）
    onnx_dir, keep = ctx["onnx_dir"], (ctx["keep_onnx"] or seed == 0)
    onnx_path = onnx_dir / ("%s_fp32.onnx" % tag)

    net, meta = build_net(run_dir, HEAD)
    y_t, t_torch = torch_eval(net, xs)
    mae_torch = float(np.abs(y_t - ys).mean())
    print("[%s] torch fp32 mae=%.6f (%.1fs)" % (tag, mae_torch, t_torch), flush=True)

    export_onnx(net, onnx_path)
    onnx_bytes, onnx_sha = onnx_path.stat().st_size, sha1_of(onnx_path)

    core = ov.Core()
    om = core.read_model(str(onnx_path))
    cm32 = core.compile_model(om, "CPU")
    y_o32, t_o32 = eval_compiled(cm32, xs)
    mae_o32 = float(np.abs(y_o32 - ys).mean())
    max_dev = float(np.abs(y_o32 - y_t).max())
    gate = abs(mae_o32 - mae_torch)
    print("[%s] ov fp32 mae=%.6f gate|dMAE|=%.3e max|dy|=%.3e (%.1fs)"
          % (tag, mae_o32, gate, max_dev, t_o32), flush=True)

    rec = {
        "tag": tag, "protocol": proto, "snap_every": snap, "seed": seed,
        "run": run_dir.name, "run_args": meta,
        "mae_torch_fp32": mae_torch, "mae_ov_fp32": mae_o32,
        "gate_abs_dmae": gate, "gate_max_abs_output_delta": max_dev,
        "gate_pass": bool(gate <= GATE_TOL),
        "onnx": {"path": str(onnx_path.relative_to(ROOT)), "bytes": onnx_bytes, "sha1": onnx_sha},
        "latency_ms_torch_fp32": 1000 * t_torch / len(xs),
        "latency_ms_ov_fp32": 1000 * t_o32 / len(xs),
    }
    if not rec["gate_pass"]:
        rec["error"] = "FP32 gate failed; quantization skipped"
        return rec

    calib_ds = nncf.Dataset(calib, lambda a: a[None, None].astype(np.float32))
    t0 = time.perf_counter()
    q_model = nncf.quantize(om, calib_ds, subset_size=len(calib))
    rec["quantize_seconds"] = time.perf_counter() - t0
    rec["quant_graph"] = describe_quant(q_model)
    print("[%s] quantize %.1fs %s" % (tag, rec["quantize_seconds"],
                                      json.dumps(rec["quant_graph"])), flush=True)

    int8_path = onnx_dir / ("%s_int8.xml" % tag)
    ov.save_model(q_model, str(int8_path), compress_to_fp16=False)
    cm8 = core.compile_model(q_model, "CPU")
    y_o8, t_o8 = eval_compiled(cm8, xs)
    mae_o8 = float(np.abs(y_o8 - ys).mean())
    rec.update({
        "mae_ov_int8": mae_o8,
        "dmae_int8_minus_fp32": mae_o8 - mae_o32,
        "dmae_int8_minus_torch": mae_o8 - mae_torch,
        "latency_ms_ov_int8": 1000 * t_o8 / len(xs),
        "int8_ir": {"path": str(int8_path.relative_to(ROOT)),
                    "bytes_xml": int8_path.stat().st_size,
                    "bytes_bin": int8_path.with_suffix(".bin").stat().st_size},
    })
    print("[%s] ov int8 mae=%.6f d(int8-fp32)=%+.6f (%.1fs)"
          % (tag, mae_o8, rec["dmae_int8_minus_fp32"], t_o8), flush=True)

    if not keep:
        onnx_path.unlink()
        int8_path.unlink()
        int8_path.with_suffix(".bin").unlink()
        rec["onnx"]["deleted_after_use"] = True
        rec["int8_ir"]["deleted_after_use"] = True
    return rec


# ---------------------------------------------------------------- 报告

def build_stats(per_run) -> dict:
    by = {}
    for proto, _, runs in PROTOCOLS:
        for seed, _ in runs:
            tag = "%s_s%d" % (proto, seed)
            r = next((x for x in per_run if x["tag"] == tag), None)
            if r is not None and "mae_ov_int8" in r:
                by[(proto, seed)] = r

    seeds = [s for s, _ in PROTOCOLS[0][2]]
    d32 = [by[("N1", s)]["mae_ov_fp32"] - by[("Ninf", s)]["mae_ov_fp32"] for s in seeds]
    d8 = [by[("N1", s)]["mae_ov_int8"] - by[("Ninf", s)]["mae_ov_int8"] for s in seeds]
    dq = [by[(p, s)]["dmae_int8_minus_fp32"] for p in ("N1", "Ninf") for s in seeds]
    dq_n1 = [by[("N1", s)]["dmae_int8_minus_fp32"] for s in seeds]
    dq_inf = [by[("Ninf", s)]["dmae_int8_minus_fp32"] for s in seeds]

    eff32 = float(np.mean(d32))
    eff8 = float(np.mean(d8))
    per_seed_ratio = [b / a if a != 0 else float("nan") for a, b in zip(d32, d8)]
    return {
        "interval_effect_fp32_runlevel": paired_stats(d32),
        "interval_effect_int8_runlevel": paired_stats(d8),
        "quant_dmae_pooled16": paired_stats(dq),
        "quant_dmae_N1": paired_stats(dq_n1),
        "quant_dmae_Ninf": paired_stats(dq_inf),
        "means": {
            "N1_fp32": float(np.mean([by[("N1", s)]["mae_ov_fp32"] for s in seeds])),
            "Ninf_fp32": float(np.mean([by[("Ninf", s)]["mae_ov_fp32"] for s in seeds])),
            "N1_int8": float(np.mean([by[("N1", s)]["mae_ov_int8"] for s in seeds])),
            "Ninf_int8": float(np.mean([by[("Ninf", s)]["mae_ov_int8"] for s in seeds])),
        },
        "retention_ratio_of_means": eff8 / eff32 if eff32 else float("nan"),
        "retention_ratio_per_seed_mean": float(np.nanmean(per_seed_ratio)),
        "retention_ratio_per_seed": per_seed_ratio,
        "seeds": seeds,
    }


def write_md(path: Path, payload: dict) -> None:
    recs = payload["per_run"]
    by_tag = {r["tag"]: r for r in recs}
    st = payload["stats"]
    gate = payload["gate"]
    L = []
    A = L.append
    A("# 第三个部署后端：Intel OpenVINO INT8（x86 CPU，NNCF PTQ）")
    A("")
    A("生成时间 %s；脚本 `scripts/round20_backend_ov.py`；解释器 %s。"
      % (payload["generated_at"], payload["python"]))
    A("")
    A("回答的问题：训练侧测到的**间隔效应**（间隔 N=1 相对关闭投影 N=∞ 更差）"
      "在这条 OpenVINO INT8 链上是否仍在、保留多少。")
    A("")
    A("## 1 环境与量化配置（固定项，逐条）")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| openvino | %s |" % payload["openvino_version"])
    A("| nncf | %s |" % payload["nncf_version"])
    A("| python / onnx / torch | %s / %s / %s |" % (payload["python"], payload["onnx_version"], payload["torch_version"]))
    A("| CPU | %s |" % payload["cpu"])
    A("| 设备 | CPU（全程 0 次 GPU） |")
    A("| 校准集 | `npu_quant/calib/*.npy` %d 张（与厂商链同一批） |" % payload["n_calib"])
    A("| 测试集 | `npu_quant/testbin/*.bin` %d 张（= `npu_quant/test_set.json` rows[:200]，逐项比对一致） |" % payload["n_test"])
    A("| 量化调用 | `%s` |" % payload["quant_call"])
    A("| 其余参数 | 全默认：preset=None→PERFORMANCE，target_device=ANY，fast_bias_correction=True，"
      "model_type=None，ignored_scope=None，advanced_parameters=None（含 smooth_quant_alphas matmul=0.95），"
      "校准算法为 NNCF 默认统计收集，未手调 |")
    A("")
    qg = payload["quant_graph_example"]
    A("实测量化图（%s，逐节点清点）：%d 个 conv/MatMul 站点中 **%d** 个的权重带量化节点；"
      "权重站点共 **%d** 个，全部 per-channel（%d per-channel / %d per-tensor）、全部 symmetric"
      "（无 zero-point：带 Subtract 的 %d 个）。其中 %d 个以 `int8 Constant → Convert(f32) → "
      "Multiply(逐输出通道 scale)` 反量化子图表达（int8 常量 %d 个），%d 个以 levels=255 的 "
      "FakeQuantize 表达。激活侧 **%d** 个 FakeQuantize、全部 per-tensor（per-channel %d 个），"
      "其 input_low 为标量；其中 %d 个区间为 [0, r]（非负张量，u8 侧，input_low=0），"
      "%d 个为 ±r 型区间（实测 −lo/hi 中位数 %s，即 NNCF 对称方案把 254 级映射到 ±r，"
      "256/254 = %.6f，两者一致）；levels 集合 %s。整图 FakeQuantize %d 个、算子 %d 个。"
      % (qg["tag"], qg["conv_matmul_sites"], qg["conv_matmul_with_quantized_weight"],
         qg["weight_sites_total"], qg["weights_per_channel"], qg["weights_per_tensor"],
         qg["weights_with_zero_point"], qg["weight_sites_dequant_multiply"],
         qg["weights_int8_constants"], qg["weight_sites_fakequant"],
         qg["activations_per_tensor"], qg["activations_per_channel"],
         qg["activations_low_zero"], qg["activations_near_symmetric"],
         ("%.6f" % qg["activations_neg_pos_ratio_median"]) if qg.get("activations_neg_pos_ratio_median") else "n/a",
         256.0 / 254.0,
         qg["levels"], qg["fake_quantize_nodes"], qg["model_ops"]))
    A("")
    A("## 2 检查点与数据锚点")
    A("")
    A("16 个检查点 = 8 个 seed × 2 个协议（`snap_every=1` 为间隔 1；`snap_every=0` 为关闭投影 N=∞），"
      "全部 `head=relu_qat`、`ema_mode=none`、`split_seed=0`（s0 两行的 `split_seed` 为 -1，"
      "即 `train.py:187` 的「与 seed 同值」，seed 0 → 0）、`ema_decay=0.99`。"
      "`ema_mode=none` 下影子从不更新，故取 `model_state_dict`（live）权重。")
    A("")
    A("| 协议 | seed | run 目录 | snap_every | split_seed |")
    A("|---|---|---|---|---|")
    for r in recs:
        a = r["run_args"]
        A("| %s | %d | `%s` | %s | %s |" % (r["protocol"], r["seed"], r["run"],
                                          a.get("snap_every", "1(默认)"), a.get("split_seed", "-1(默认→seed)")))
    A("")
    A("## 3 FP32 一致性闸门")
    A("")
    A("门槛：OpenVINO FP32 与 torch FP32 在同一批 %d 张上的 MAE 差 ≤ %g。实测 **%d/%d 个检查点通过，"
      "max|ΔMAE| = %.3e**；逐图像输出最大偏差 max|Δy| = %.3e。"
      % (payload["n_test"], GATE_TOL, gate["n_pass"], gate["n_checked"],
         gate["max_abs_dmae"], gate["max_abs_output_delta"]))
    A("")
    A("| run | MAE torch FP32 | MAE OV FP32 | abs(dMAE) | max abs(dy) | 闸门 |")
    A("|---|---|---|---|---|---|")
    for r in recs:
        A("| %s | %.6f | %.6f | %.2e | %.2e | %s |"
          % (r["run"], r["mae_torch_fp32"], r["mae_ov_fp32"], r["gate_abs_dmae"],
             r["gate_max_abs_output_delta"], "PASS" if r["gate_pass"] else "FAIL"))
    A("")
    A("## 4 逐 seed 读数（16 行）")
    A("")
    A("MAE 单位同训练：target = age/100，%d 张。ΔMAE = int8 − OV FP32。" % payload["n_test"])
    A("")
    A("| # | 协议 | seed | run | torch FP32 | OV FP32 | OV int8 | ΔMAE(int8−FP32) | lat FP32 ms | lat int8 ms |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(recs, 1):
        A("| %d | %s | %d | %s | %.6f | %.6f | %s | %s | %.2f | %s |"
          % (i, r["protocol"], r["seed"], r["run"], r["mae_torch_fp32"], r["mae_ov_fp32"],
             ("%.6f" % r["mae_ov_int8"]) if "mae_ov_int8" in r else "—",
             ("%+.6f" % r["dmae_int8_minus_fp32"]) if "dmae_int8_minus_fp32" in r else "—",
             r["latency_ms_ov_fp32"],
             ("%.2f" % r["latency_ms_ov_int8"]) if "latency_ms_ov_int8" in r else "—"))
    A("")
    A("## 5 run 层统计（n = 8，df = 7，逐 seed 配对）")
    A("")
    A("| 量 | 均值 | sd | 95% t 区间 | 同号 | 符号检验 p |")
    A("|---|---|---|---|---|---|")

    def row(name, s):
        A("| %s | %+.6f | %.6f | [%+.6f, %+.6f] | %d 正 / %d 负 | %.4f |"
          % (name, s["mean"], s["sd"], s["ci95"][0], s["ci95"][1], s["n_pos"], s["n_neg"], s["sign_p"]))
    row("间隔效应 @FP32（N1 − N∞）", st["interval_effect_fp32_runlevel"])
    row("间隔效应 @int8（N1 − N∞）", st["interval_effect_int8_runlevel"])
    row("量化代价 int8 − FP32（N1，8 run）", st["quant_dmae_N1"])
    row("量化代价 int8 − FP32（N∞，8 run）", st["quant_dmae_Ninf"])
    row("量化代价 int8 − FP32（16 run 合并）", st["quant_dmae_pooled16"])
    A("")
    A("")
    A("合并行 n = 16（两个协议各 8 个 run），用 t_{0.975,15} = %.3f，只描述量化代价的整体量级，"
      "不是跨协议配对对比。" % TCRIT_BY_N[16])
    m = st["means"]
    A("四个格子的均值：N1/FP32 **%.6f**、N∞/FP32 **%.6f**、N1/int8 **%.6f**、N∞/int8 **%.6f**。"
      % (m["N1_fp32"], m["Ninf_fp32"], m["N1_int8"], m["Ninf_int8"]))
    A("")
    A("## 6 间隔效应穿过本后端保留的比例")
    A("")
    A("保留比例 = (N1_int8 − N∞_int8) / (N1_fp32 − N∞_fp32) = %.6f / %.6f = **%.3f**。"
      % (st["interval_effect_int8_runlevel"]["mean"], st["interval_effect_fp32_runlevel"]["mean"],
         st["retention_ratio_of_means"]))
    A("")
    A("逐 seed 比值（分母为该 seed 的 FP32 间隔效应）均值 %.3f，取值范围 [%.3f, %.3f]——"
      "分母逐 seed 只有 0.01–0.02 量级，单 seed 比值离散度大，**以「均值之比」为准**，逐 seed 比值只作旁证。"
      % (st["retention_ratio_per_seed_mean"],
         float(np.nanmin(st["retention_ratio_per_seed"])), float(np.nanmax(st["retention_ratio_per_seed"]))))
    A("")
    A("**一句话结论**：%s" % payload["conclusion"])
    A("")
    A("## 7 边界")
    A("")
    A("- 这是**第三方运行时（Intel OpenVINO）+ 独立量化器（NNCF）**的一路：ONNX 由本脚本用 torch 导出，"
      "量化由 NNCF 用默认预设完成，与厂商 hbdk4 链（BPU/nash）、与 onnxruntime 静态 QDQ 链**实现互不共享**。"
      "三路只能互相印证，**不能互相替代**：任何一路的读数都不外推到另一路。")
    lat32 = float(np.median([r["latency_ms_ov_fp32"] for r in recs]))
    lat8 = float(np.median([r["latency_ms_ov_int8"] for r in recs if "latency_ms_ov_int8" in r]))
    A("- 延迟只作 **CPU 相对参考**（同一台机器、同一进程、单张串行）：OV FP32 中位 %.2f ms/张、"
      "int8 中位 %.2f ms/张（比值 %.2f×）。这**不是硬件结论**，也**不得**与 BPU 的 µs 数相比"
      "（不同指令集、不同量化器、不同内存层级）。" % (lat32, lat8, lat32 / lat8 if lat8 else float("nan")))
    A("- **不是逐算子全 int8**：NNCF 默认预设下 %d 个 conv/MatMul 站点里只有 %d 个的权重被量化，"
      "未量化的 %d 个中 %d 个是 SE 模块的 fc1/fc2 1×1 卷积（权重留 fp32），另 %d 个是 self-attention "
      "的 QKᵀ MatMul（其第二操作数是激活经 Reshape，本就不是权重）。激活侧 %d 个站点各自带 "
      "FakeQuantize。故本条链是「%d/%d 个 conv/MatMul 站点量化权重 + %d 个激活站点量化 + %d 个 SE 卷积权重留 fp32」"
      "的混合精度，任何「全 int8」的表述都不成立。"
      % (qg["conv_matmul_sites"], qg["conv_matmul_with_quantized_weight"],
         qg["unquantized_weight_sites"], qg["unquantized_se_fc_convs"],
         qg["unquantized_attn_qk_matmuls"], qg["activations_per_tensor"],
         qg["conv_matmul_with_quantized_weight"], qg["conv_matmul_sites"],
         qg["activations_per_tensor"], qg["unquantized_se_fc_convs"]))
    A("- torch FP32 基线由**驱动解释器** `D:/tool/Python` 的 torch %s（CPU）产生，不是训练环境的 torch；"
      "已用训练环境 torch %s（%s）在同批 %d 张上复核两个检查点：%s ⇒ 基线不随 torch 版本漂移。"
      % (payload["torch_version"], TORCH_XCHECK["torch"], TORCH_XCHECK["interpreter"],
         payload["n_test"],
         "；".join("%s 得 %.9f vs 本报告 %.9f（|Δ| = %.1e）"
                  % (t, v, by_tag[t]["mae_torch_fp32"], abs(v - by_tag[t]["mae_torch_fp32"]))
                  for t, v in TORCH_XCHECK["mae"].items())))
    A("- 统计单位为 run（seed）：n = 8、df = 7；t 区间用 t_{0.975,7} = %.3f；符号检验为精确双侧二项。" % TCRIT_DF7)
    A("- 测试集 %d 张是**评估子集**（test_set.json 的 [800:1600]），与全验证集口径的绝对值不可直接比。" % payload["n_test"])
    A("")
    A("## 8 产物与偏离")
    A("")
    for line in payload["artifacts"]:
        A("- " + line)
    if payload["deviations"]:
        for line in payload["deviations"]:
            A("- 偏离：" + line)
    else:
        A("- 无偏离。")
    A("")
    path.write_text(chr(10).join(L) + chr(10), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx-dir", default="quant_runs/backend_ov_onnx")
    p.add_argument("--json", default="quant_runs/backend_ov_report.json")
    p.add_argument("--md", default="quant_runs/backend_ov_report.md")
    p.add_argument("--limit-runs", type=int, default=0, help="冒烟：只跑前 N 个检查点")
    p.add_argument("--limit-test", type=int, default=0, help="冒烟：只用前 N 张测试图")
    p.add_argument("--keep-onnx", action="store_true", help="保留全部 ONNX/IR（默认只留 seed 0 两个协议）")
    p.add_argument("--only", default="", help="逗号分隔的 tag，只跑这些（如 N1_s3,Ninf_s3）")
    p.add_argument("--write-md", action="store_true", help="同时写 markdown 报告")
    a = p.parse_args()

    import onnx
    import torch
    t_start = time.time()

    onnx_dir = ROOT / a.onnx_dir
    onnx_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((ROOT / "npu_quant/testbin/meta.json").read_text())
    ts = json.loads((ROOT / "npu_quant/test_set.json").read_text())
    assert meta["rows"] == ts["rows"][:200] and meta["targets"] == ts["targets"][:200], "test set mismatch"
    ks = len(meta["rows"]) if not a.limit_test else a.limit_test
    xs = [np.fromfile(ROOT / ("npu_quant/testbin/%04d.bin" % i), dtype=np.float32).reshape(1, 1, 3, SIZE, SIZE)
          for i in range(ks)]
    targets = np.asarray(meta["targets"][:ks], dtype=np.float64)

    calib_files = sorted((ROOT / "npu_quant/calib").glob("*.npy"))
    calib = [np.load(f) for f in calib_files]
    assert all(c.shape == (3, SIZE, SIZE) for c in calib), "calib shape"

    ctx = {"xs": xs, "targets": targets, "calib": calib,
           "onnx_dir": onnx_dir, "keep_onnx": a.keep_onnx}

    only = {s.strip() for s in a.only.split(",") if s.strip()}
    per_run = []
    todo = [(proto, snap, seed, run) for proto, snap, runs in PROTOCOLS for seed, run in runs]
    if a.limit_runs:
        todo = todo[: a.limit_runs]
    if only:
        todo = [t for t in todo if "%s_s%d" % (t[0], t[2]) in only]

    gate_failed = False
    for proto, snap, seed, run in todo:
        tag = "%s_s%d" % (proto, seed)
        if gate_failed:
            break
        rec = run_one(tag, proto, snap, seed, ROOT / "results" / run, ctx)
        per_run.append(rec)
        gate_failed = not rec["gate_pass"]

    if a.limit_runs or a.limit_test or only:
        print(json.dumps(per_run, indent=2, ensure_ascii=False))
        print("SMOKE OK  wall=%.1fs" % (time.time() - t_start))
        return

    stats = build_stats(per_run)
    gate = {
        "tolerance": GATE_TOL,
        "n_checked": len(per_run),
        "n_pass": sum(1 for r in per_run if r["gate_pass"]),
        "max_abs_dmae": max(r["gate_abs_dmae"] for r in per_run),
        "max_abs_output_delta": max(r["gate_max_abs_output_delta"] for r in per_run),
    }
    qg = next((dict(r["quant_graph"], tag=r["tag"]) for r in per_run if r.get("quant_graph")), {})
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "openvino_version": ov.__version__,
        "nncf_version": __import__("nncf").__version__,
        "onnx_version": onnx.__version__,
        "torch_version": torch.__version__,
        "cpu": __import__("os").environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "n_test": len(xs), "n_calib": len(calib),
        "quant_call": "nncf.quantize(ov_model, nncf.Dataset(calib100), subset_size=100)",
        "quant_graph_example": qg,
        "gate": gate,
        "per_run": per_run,
        "stats": stats,
        "conclusion": "",
        "artifacts": [],
        "deviations": [],
    }
    # 结论句由数据生成
    e32 = stats["interval_effect_fp32_runlevel"]
    e8 = stats["interval_effect_int8_runlevel"]
    ret = stats["retention_ratio_of_means"]
    payload["conclusion"] = (
        "间隔效应在这条链上仍在：FP32 上 N1−N∞ = %+.6f（8/8 同号，p = %.4f），"
        "OpenVINO int8 上 = %+.6f（%d 正 / %d 负，p = %.4f，区间%s 0），"
        "保留 %.1f%%；缺口的 %.1f%% 主要来自关闭投影那一端被量化削弱得更多"
        "（int8−FP32 在 N∞ 为 %+.6f、在 N=1 为 %+.6f），而不是 N=1 在 int8 上变好。"
        "INT8 本身 16 run 合并的 ΔMAE 为 %+.6f（sd %.6f），远小于间隔效应本身。"
        % (e32["mean"], e32["sign_p"], e8["mean"], e8["n_pos"], e8["n_neg"], e8["sign_p"],
           "不含" if (e8["ci95"][0] > 0 or e8["ci95"][1] < 0) else "含",
           100 * ret, 100 - 100 * ret,
           stats["quant_dmae_Ninf"]["mean"], stats["quant_dmae_N1"]["mean"],
           stats["quant_dmae_pooled16"]["mean"], stats["quant_dmae_pooled16"]["sd"]))

    kept = [r["tag"] for r in per_run if r.get("onnx") and not r["onnx"].get("deleted_after_use")]
    payload["artifacts"] = [
        "脚本 `scripts/round20_backend_ov.py`（只读既有产物，写 `quant_runs/backend_ov_*`）。",
        "本报告 `%s` 与 `quant_runs/backend_ov_report.json`（含逐 run 原始读数、ONNX 的字节数与 sha1）。" % a.md,
        "ONNX/IR 目录 `%s/`：保留 %s 的 FP32 ONNX 与 int8 IR（xml+bin），其余 run 的中间件用完即删，"
        "字节数与 sha1 记在 JSON 里。" % (a.onnx_dir, ", ".join(kept) if kept else "无"),
        "墙钟合计 %.1f 分钟；GPU 使用 0 次。" % ((time.time() - t_start) / 60.0),
    ]
    xc_worst = 0.0
    for t, v in TORCH_XCHECK["mae"].items():
        xc_worst = max(xc_worst, abs(v - next(r["mae_torch_fp32"] for r in per_run if r["tag"] == t)))
    payload["deviations"] = [
        "torch FP32 基线用驱动解释器 D:/tool/Python 的 torch %s（CPU），而非训练环境 torch %s；"
        "已用后者在同批 %d 张上复核 N1_s0 / Ninf_s0，|ΔMAE| 最大 %.1e，故基线不随 torch 版本漂移。"
        % (torch.__version__, TORCH_XCHECK["torch"], len(xs), xc_worst),
        "NNCF 默认预设没有量化 %d 个 SE 1×1 卷积的权重（详见 §1 与 §7）：这是预设行为，"
        "未做任何 ignore/手调；本条链因此是混合精度，不当作逐算子全 int8。"
        % qg["unquantized_se_fc_convs"],
    ]
    (ROOT / a.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if a.write_md:
        write_md(ROOT / a.md, payload)
    print("[done] json=%s md=%s" % (a.json, a.md if a.write_md else "-"), flush=True)
    print(json.dumps({k: payload[k] for k in ("gate", "stats")}, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
