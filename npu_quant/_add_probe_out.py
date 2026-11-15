import sys, onnx
sys.stdout.reconfigure(encoding="utf-8")
TAG="gelu_s0_N1"
base=f"/temporal/layers.0/norm2/LayerNormalization_mean1_output"
P=f"npu_quant/out/{TAG}/{TAG}_calibrated_model.onnx"
m=onnx.load(P)
names={v.name for v in m.graph.value_info}
want=[base, base+"_calibrated"]
for w in want:
    print(w, "in value_info:", w in names)
    o=m.graph.output.add(); o.name=w
onnx.save(m,"npu_quant/out/_probe_cal.onnx")
print("saved npu_quant/out/_probe_cal.onnx")
