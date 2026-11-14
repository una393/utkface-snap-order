"""容器内驱动：给 hmct 的常量折叠加钩子，定位 ORT 建 session 段错误发生在哪个算子。

不是修训练代码，纯粹是排查工具链崩溃点。用法：
  python3 npu_quant/hb_driver_debug.py -c /work/npu_quant/hb_config/xxx.yaml
"""
from __future__ import annotations

import faulthandler
import sys

import hmct.common.modifier.passes.constant_folding as cf

faulthandler.enable()
_orig = cf.ConstantFolding._fold_const


def _patched(self, onnx_node):
    print(f"[fold] {onnx_node.op_type} :: {onnx_node.name}", flush=True)
    return _orig(self, onnx_node)


cf.ConstantFolding._fold_const = _patched

from horizon_tc_ui.hb_compile import main  # noqa: E402

if __name__ == "__main__":
    main()
