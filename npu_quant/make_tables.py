"""把三份证据（逐节点余弦、编译参数、三档 MAE）汇成 markdown 表。

只读已有产物，不做任何推断性计算之外的东西：
  npu_quant/out/<tag>/<tag>_quant_info.json   -> 逐节点余弦
  npu_quant/out/<tag>/<tag>.json              -> 编译器 summary（latency / FPS / DDR）
  npu_quant/out/<tag>/<tag>.hbm               -> 文件大小
  npu_quant/out/log_<tag>.txt 或 _patched.txt -> 编译耗时 "exit=.. elapsed=..s"
  npu_quant/out/mae_fp32.json / mae_fakequant.json / mae_runtime_<tag>.json -> MAE
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "npu_quant", "out")
TAGS = ["relu_qat_s0_N1", "relu_qat_s0_N20", "gelu_s0_N1", "bilinear_glu_s0_N1"]


def as_list(v):
    return v if isinstance(v, list) else [v]


def cosine_map(tag):
    path = os.path.join(OUT, tag, f"{tag}_quant_info.json")
    if not os.path.exists(path):
        return None
    data = json.load(open(path))
    out = {}
    for name, rec in data.items():
        cos = [float(c) for c in as_list(rec.get("cosine_similarity", [])) if c not in ("", None)]
        if cos:
            out[name] = (rec.get("type"), cos)
    return out


def stats_of(cmap):
    vals = [min(c) for _, c in cmap.values()]
    s = {
        "n_nodes": len(vals),
        "min": min(vals),
        "p1": st.quantiles(vals, n=100)[0],
        "p5": st.quantiles(vals, n=100)[4],
        "median": st.median(vals),
        "mean": sum(vals) / len(vals),
        "lt_0.99": sum(1 for v in vals if v < 0.99),
        "lt_0.999": sum(1 for v in vals if v < 0.999),
        "eq_1": sum(1 for v in vals if v >= 0.999999),
    }
    return s


def elapsed(tag):
    for pat in (f"log_{tag}_patched.txt", f"log_{tag}.txt"):
        p = os.path.join(OUT, pat)
        if os.path.exists(p):
            m = re.search(r"exit=(-?\d+) elapsed=(\d+)s", open(p, errors="ignore").read())
            if m:
                return int(m.group(1)), int(m.group(2))
    return None, None


def summary_json(tag):
    p = os.path.join(OUT, tag, f"{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def dig(obj, key):
    """在嵌套 dict/list 里找第一个名为 key 的字段。"""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            got = dig(v, key)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = dig(v, key)
            if got is not None:
                return got
    return None


def hbm_size(tag):
    p = os.path.join(OUT, tag, f"{tag}.hbm")
    return os.path.getsize(p) if os.path.exists(p) else None


print("## 编译产物\n")
print("| tag | .hbm bytes | 编译耗时 s | exit | latency µs | FPS | DDR bytes |")
print("|---|---|---|---|---|---|---|")
for t in TAGS:
    rc, el = elapsed(t)
    s = summary_json(t)
    sz = hbm_size(t)
    print(f"| {t} | {sz} | {el} | {rc} | {dig(s, 'latency (us)')} | {dig(s, 'FPS')} | "
          f"{dig(s, 'DDR bytes per run')} |")

print("\n## 逐节点余弦：全局\n")
print("| tag | nodes | min | p1 | p5 | median | mean | <0.99 | <0.999 | ==1 |")
print("|---|---|---|---|---|---|---|---|---|---|")
cms = {}
for t in TAGS:
    cm = cosine_map(t)
    if not cm:
        print(f"| {t} | - | - | - | - | - | - | - | - | - |")
        continue
    cms[t] = cm
    s = stats_of(cm)
    print(f"| {t} | {s['n_nodes']} | {s['min']:.6f} | {s['p1']:.6f} | {s['p5']:.6f} | "
          f"{s['median']:.6f} | {s['mean']:.6f} | {s['lt_0.99']} | {s['lt_0.999']} | {s['eq_1']} |")

print("\n## 逐节点余弦：控制头 + 头前 /fpn/norm（按后缀对齐）\n")
def key(node):
    for pref in ("/head/net/net.", "/fpn/norm/"):
        if node.startswith(pref):
            return node
    return None

allkeys = []
for node in sorted(set().union(*[set(cm) for cm in cms.values()])) if cms else []:
    if key(node):
        allkeys.append(node)
print("| node | type | " + " | ".join(TAGS) + " |")
print("|---|---|" + "---|" * len(TAGS))
for node in allkeys:
    typ = next((cms[t][node][0] for t in TAGS if t in cms and node in cms[t]), "")
    cells = []
    for t in TAGS:
        if t in cms and node in cms[t]:
            cells.append(f"{min(cms[t][node][1]):.6f}")
        else:
            cells.append("-")
    print(f"| {node} | {typ} | " + " | ".join(cells) + " |")

print("\n## MAE（同一 200 张测试图，seed 0 划分的 val[800:1000]）\n")
fp32 = json.load(open(os.path.join(OUT, "mae_fp32.json")))["models"]
fq = json.load(open(os.path.join(OUT, "mae_fakequant.json")))["models"] if os.path.exists(os.path.join(OUT, "mae_fakequant.json")) else {}
rt = {}
for p in sorted(glob.glob(os.path.join(OUT, "mae_runtime_*.json"))):
    d = json.load(open(p))
    rt[d["tag"]] = d

print("| tag | FP32 ONNX MAE (x86 宿主) | fake-quant MAE (编译器模拟) | nash runtime MAE (x86 实测) |")
print("|---|---|---|---|")
for t in TAGS:
    a = fp32.get(t, {}).get("mae")
    b = fq.get(t, {}).get("mae")
    c = rt.get(t, {}).get("mae")
    fmt = lambda v: "-" if v is None else f"{v:.6f}"
    print(f"| {t} | {fmt(a)} | {fmt(b)} | {fmt(c)} |")

print("\n## 配对差（量化 − FP32，同图同序）\n")
fp32_targets = json.load(open(os.path.join(OUT, "mae_fp32.json")))["targets"]
print("| tag | fake-quant Δmean ± SE | nash Δmean ± SE | fake-quant ΔMAE ± SE | nash ΔMAE ± SE |")
print("|---|---|---|---|---|")
for t in TAGS:
    ref = fp32.get(t, {}).get("preds")
    if ref is None:
        continue
    row = [t]
    for src in (fq.get(t), rt.get(t)):
        if not src:
            row.append("-")
            continue
        d = [x - y for x, y in zip(src["preds"], ref) if x is not None]
        if not d:
            row.append("-")
            continue
        m = sum(d) / len(d)
        se = (st.pstdev(d) / (len(d) ** 0.5)) if len(d) > 1 else float("nan")
        row.append(f"n={len(d)} {m:+.5f} ± {se:.5f}")
    # MAE 变化（逐样本配对：|err_quant| - |err_fp32|，再取均值 ± SE）
    tgt = fp32_targets
    for src in (fq.get(t), rt.get(t)):
        if not src:
            row.append("-")
            continue
        d = []
        for p, q, y in zip(ref, src["preds"], tgt):
            if q is None:
                continue
            d.append(abs(q - y) - abs(p - y))
        m = sum(d) / len(d)
        se = st.pstdev(d) / (len(d) ** 0.5) if len(d) > 1 else float("nan")
        row.append(f"{m:+.6f} ± {se:.6f}")
    print("| " + " | ".join(row) + " |")
