
import json, collections
base = r"C:/Users/USER\Desktop\paper-utkface\npu_quant\out\relu_qat_s0_N1\relu_qat_s0_N1"
summ = json.load(open(base + ".json"))
print("=== summary json top-level ===")
def walk(o, pre="", depth=0):
    if depth > 1:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                print(f"{pre}{k}: {type(v).__name__} len={len(v)}")
                if isinstance(v, dict):
                    walk(v, pre + "  ", depth + 1)
                elif v and isinstance(v[0], dict):
                    print(f"{pre}  [0] keys: {list(v[0].keys())[:12]}")
            else:
                print(f"{pre}{k} = {v}")
walk(summ)

qi = json.load(open(base + "_quant_info.json"))
print("\n=== quant_info top-level ===")
print(type(qi).__name__, list(qi.keys())[:10] if isinstance(qi, dict) else len(qi))
first = qi[list(qi.keys())[0]] if isinstance(qi, dict) else qi[0]
print("first entry keys:", list(first.keys()) if isinstance(first, dict) else type(first))
print("\nexample entry:")
print(json.dumps(first, ensure_ascii=False)[:1200])
