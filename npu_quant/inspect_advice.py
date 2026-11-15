import json, re, sys, collections, os
sys.stdout.reconfigure(encoding="utf-8")
tags = ["relu_qat_s0_N1","relu_qat_s0_N20","gelu_s0_N1","bilinear_glu_s0_N1"]
for tag in tags:
    p = f"npu_quant/out/{tag}/{tag}_advice.json"
    if not os.path.exists(p):
        print(tag, "MISSING"); continue
    d = json.load(open(p))
    print("="*70); print(tag, "entries:", len(d))
    print("op_type 分布:", dict(sorted(collections.Counter(e["op_type"] for e in d).items(), key=lambda kv:-kv[1])))
    for e in d:
        loc = e.get("loc","")
        m = re.search(r'layerName = "(.*?)"', loc)
        nm = m.group(1) if m else e.get("op_name","")
        if "/head/" in nm or "head" in nm:
            print(f'  {nm:<55} {e["op_type"]:<28} backend={e["backend"]}')
