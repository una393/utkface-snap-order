"""汇总最终证据表：编译产物 / 全局与头部余弦 / BPU op_type / MAE / LUT 计数 -> npu_quant/out/final_summary.md"""
import json, os, re, statistics, sys
sys.stdout.reconfigure(encoding="utf-8")

TAGS = ["relu_qat_s0_N1", "relu_qat_s0_N20", "gelu_s0_N1", "bilinear_glu_s0_N1"]
OUT = "npu_quant/out"
COMPILE = {"relu_qat_s0_N1": 395, "relu_qat_s0_N20": 676, "gelu_s0_N1": 2262, "bilinear_glu_s0_N1": 4601}

def load_cos(tag):
    d = json.load(open(f"{OUT}/{tag}/{tag}_quant_info.json"))
    rows = []
    for k, v in d.items():
        c = v.get("cosine_similarity")
        if isinstance(c, list):
            c = c[0] if c else None
        try:
            rows.append((k, v.get("type"), float(c)))
        except (TypeError, ValueError):
            pass
    return rows

def op_types(tag):
    d = json.load(open(f"{OUT}/{tag}/{tag}_advice.json"))
    m = {}
    for e in d:
        mm = re.search(r'layerName = "(.*?)"', e.get("loc", ""))
        nm = mm.group(1) if mm else e.get("op_name", "")
        nm = re.sub(r"_id_\d+$", "", nm)
        m.setdefault(nm, []).append(e["op_type"])
    return m

L = []
def w(s=""):
    L.append(s)

w("# S100 真实工具链编译与量化误差：最终证据表")
w("")
w("生成脚本 npu_quant/summarize_final.py，只读下列产物，不写任何模型文件。")
w("")

w("## A. 编译产物（容器自报）")
w("")
w("| tag | .hbm 字节 | 编译耗时 | 节点数 | latency | FPS | DDR/run |")
w("|---|---|---|---|---|---|---|")
for t in TAGS:
    hb = f"{OUT}/{t}/{t}.hbm"
    sz = os.path.getsize(hb) if os.path.exists(hb) else None
    s = json.load(open(f"{OUT}/{t}/{t}.json"))["summary"]
    p = s["performance"]
    n = len(load_cos(t))
    w(f"| {t} | {sz:,} | {COMPILE[t]} s | {n} | {p['latency (us)']} us | {p['FPS']} | "
      f"{s['DDR access data']['DDR megabytes per run']} MB |")
w("")

w("## B. 逐节点余弦：全局统计（编译器 *_quant_info.json）")
w("")
w("| tag | n | min | min(排除 -1 异常) | p1 | median | mean | <0.99 | <0.999 | ==1 |")
w("|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    rows = load_cos(t)
    xs = sorted(x[2] for x in rows)
    finite = [x for x in xs if x > -0.5]
    w(f"| {t} | {len(xs)} | {xs[0]:.6f} | {finite[0]:.6f} | {xs[max(0,int(len(xs)*0.01))]:.6f} | "
      f"{statistics.median(xs):.6f} | {statistics.mean(xs):.6f} | "
      f"{sum(1 for x in xs if x<0.99)} | {sum(1 for x in xs if x<0.999)} | {sum(1 for x in xs if x>=0.999999)} |")
w("")

w("## C. 控制头作用域逐节点余弦 + BPU 指令算子（advice.json 的 op_type）")
for t in TAGS:
    rows = {k: (ty, c) for k, ty, c in load_cos(t)}
    ops = op_types(t)
    w("")
    w(f"### {t}")
    w("")
    w("| 节点 | 图类型 | 余弦 | BPU op_type |")
    w("|---|---|---|---|")
    for k, (ty, c) in rows.items():
        if "/head/" not in k:
            continue
        op = ops.get(k, [])
        w(f"| {k} | {ty} | {c:.6f} | {', '.join(op) if op else '-'} |")
    head_lut = [k for k in rows if "/head/" in k and any("lut" in o for o in ops.get(k, []))]
    tot_lut = sum(1 for e_ops in ops.values() for o in e_ops if o == "b30.lut")
    w("")
    if head_lut:
        w(f"- 头部 LUT 指令数（b30.lut）：**{len(head_lut)}**（{', '.join(head_lut)}）")
    else:
        w("- 头部 LUT 指令数（b30.lut）：**0**")
    w(f"- 全模型 b30.lut 指令数：{tot_lut}")
    fpn = [(k, c) for k, (ty, c) in rows.items() if "/fpn/norm/" in k]
    if fpn:
        w("- 头前 FPN 归一化（三头共有）：" + "；".join(f"{k.split('/fpn/norm/')[-1]} {c:.6f}" for k, c in fpn[:6]))
w("")

w("## D. MAE（同一 200 张测试图，与校准集不相交；target = age/100）")
w("")
fp32 = json.load(open(f"{OUT}/mae_fp32.json"))["models"]
fq = json.load(open(f"{OUT}/mae_fakequant.json"))["models"]
w("| tag | FP32 (x86 onnxruntime) | 伪量化 (hmct) | nash 运行时 (x86 实测) | dMAE 伪量化 | dMAE nash |")
w("|---|---|---|---|---|---|")
for t in TAGS:
    f = fp32.get(t, {})
    row = [f"| {t}"]
    row.append(f" {f['mae']:.6f} " if f.get("mae") is not None else " - ")
    row.append(f" {fq[t]['mae']:.6f} " if t in fq else " - ")
    rt_path = f"{OUT}/mae_runtime_{t}.json"
    rt = json.load(open(rt_path)) if os.path.exists(rt_path) else None
    if rt:
        row.append(f" {rt['mae']:.6f} (n={rt.get('n_ok', rt.get('n'))}) ")
    else:
        row.append(" 待跑 ")
    row.append(f" {fq[t]['mae']-f['mae']:+.6f} " if t in fq and f.get("mae") is not None else " - ")
    row.append(f" {rt['mae']-f['mae']:+.6f} " if rt and f.get("mae") is not None else " - ")
    w("|".join(row) + "|")
w("")

txt = "\n".join(L) + "\n"
open(f"{OUT}/final_summary.md", "w", encoding="utf-8").write(txt)
print(txt[:6000])
