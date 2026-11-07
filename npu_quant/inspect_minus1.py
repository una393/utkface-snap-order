import json, sys, os
sys.stdout.reconfigure(encoding="utf-8")
tag="gelu_s0_N1"
d=json.load(open(f"npu_quant/out/{tag}/{tag}_quant_info.json"))
bad=[(k,v) for k,v in d.items() if str(v.get("cosine_similarity")) in ("-1.000000","-1","-1.0")]
print("== 余弦为 -1 的节点:", [k for k,_ in bad])
for k,v in bad:
    print(k, {f:(repr(x)[:220]) for f,x in v.items()})
print()
print("== 名字含 mean 的节点:")
for k,v in d.items():
    if "mean" in k.lower():
        print(" ", k, v.get("type"), v.get("cosine_similarity"), "outputs=",v.get("outputs"))
