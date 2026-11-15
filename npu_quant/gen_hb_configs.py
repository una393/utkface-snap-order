"""生成 4 份 hb_compile 配置（容器内路径，/work = 工作区挂载点）。

沿用 output/autopilot.yaml 的 S100P 参数，只换 onnx / 前缀 / 输入名 / 形状 / 校准目录。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

TEMPLATE = """model_parameters:
  onnx_model: '{root}/npu_quant/onnx/{tag}.onnx'
  output_model_file_prefix: '{tag}'
  working_dir: '{root}/npu_quant/out/{tag}'
  march: 'nash-m'

input_parameters:
  input_name: 'sequence_images'
  input_type_rt: 'featuremap'
  input_type_train: 'featuremap'
  input_layout_train: 'NCHW'
  input_shape: '1x1x3x224x224'

calibration_parameters:
  cal_data_dir: '{root}/npu_quant/calib'
  cal_data_type: 'float32'

compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O2'
  debug: False
  core_num: 1
"""

TAGS = ["relu_qat_s0_N1", "relu_qat_s0_N20", "gelu_s0_N1", "bilinear_glu_s0_N1",
        "relu_qat_s0_Ninf"]

def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "/work"
    out = Path(__file__).parent / "hb_config"
    out.mkdir(parents=True, exist_ok=True)
    for t in TAGS:
        (out / f"{t}.yaml").write_text(TEMPLATE.format(root=root, tag=t))
        print(f"wrote {out / (t + '.yaml')}")

if __name__ == "__main__":
    main()
