"""绕过 hmct 常量折叠在 onnxruntime 建 session 时的段错误（本容器已实测复现）。

背景（实测）：hmct 2.6.5 的 ConstantFolding._fold_other_node 把每个"所有输入均为常量"
的节点抽成单节点子模型，再用 ORTExecutor 建 onnxruntime session 推理；在本镜像里
onnxruntime 1.19.0 对该子模型建 session 时直接 SIGSEGV（进程 exit 139），崩溃栈为
onnxruntime_inference_collection.py:491 -> hmct/executor/ort2.py:45 -> ort.py:243
-> constant_folding.py:305。首个崩溃节点是 int64 的
/temporal/layers.0/self_attn/Mod（两个 Constant 操作数 2 和 3），与算子语义无关。

本模块只替换「求值手段」：优先用 onnx.reference.ReferenceEvaluator（纯 Python，不建
ORT session），失败再退回不注册自定义算子的裸 ORT，仍失败则放弃该节点的折叠。
max_inc 判定、常量替换、节点销毁等行为与原实现逐行一致，不改变折叠语义。

用法：在调用 horizon_tc_ui.hb_compile.main() 之前 import 本模块（见 hb_compile_patched.py）。
"""

from __future__ import annotations

import os

from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

from hmct.common.modifier.passes.constant_folding import ConstantFolding
from hmct.ir import extract_submodel

LOG_PATH = os.environ.get("HB_FOLD_LOG", "/work/npu_quant/out/fold_patch.log")


def _log(msg: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _feeds_from_initializers(proto) -> dict:
    inits = {i.name: numpy_helper.to_array(i) for i in proto.graph.initializer}
    feeds = {}
    for vi in proto.graph.input:
        if vi.name not in inits:
            raise RuntimeError(f"submodel graph input {vi.name} carries no constant value")
        feeds[vi.name] = inits[vi.name]
    return feeds


def _eval_reference(proto) -> dict:
    out_names = [o.name for o in proto.graph.output]
    sess = ReferenceEvaluator(proto)
    return dict(zip(out_names, sess.run(out_names, _feeds_from_initializers(proto))))


def _eval_ort_plain(proto) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(
        proto.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"]
    )
    out_names = [o.name for o in proto.graph.output]
    return dict(zip(out_names, sess.run(out_names, _feeds_from_initializers(proto))))


def _fold_other_node(self, onnx_node):  # noqa: ANN001 - 签名与 hmct 原实现一致
    single_node_model = extract_submodel(
        onnx_model=onnx_node.owning_model,
        input_vars=onnx_node.inputs,
        output_vars=onnx_node.outputs,
        infer_shapes=False,
        check_model=False,
    )
    proto = single_node_model.proto
    try:
        output_dict = _eval_reference(proto)
        how = "reference"
    except Exception as exc:  # noqa: BLE001
        _log(f"[patch] reference failed {onnx_node.op_type}::{onnx_node.name}: {exc!r}")
        try:
            output_dict = _eval_ort_plain(proto)
            how = "ort-plain"
        except Exception as exc2:  # noqa: BLE001
            _log(f"[patch] SKIP {onnx_node.op_type}::{onnx_node.name}: {exc2!r}")
            return False
    _log(f"[patch] folded via {how}: {onnx_node.op_type}::{onnx_node.name}")

    if (
        self._max_inc is not None
        and self._calc_param_inc(onnx_node, output_dict.values()) > self._max_inc
    ):
        return False
    for output_idx, output_var in enumerate(onnx_node.outputs):
        onnx_node.outputs[output_idx].replace_all_uses_with(
            onnx_node.owning_graph.create_variable(
                name=output_var.name,
                value=output_dict[output_var.name],
                is_param=True,
            )
        )
    onnx_node.destroy()

    return True


ConstantFolding._fold_other_node = _fold_other_node

# ---------------------------------------------------------------------------
# 第二个坑（实测）：本机 docker 跑该镜像时 ort.get_device() == 'GPU'、可用 provider 含
# CUDAExecutionProvider，于是 hmct 的 ORTExecutor2.get_support_devices() 返回
# ['cuda', 'cpu']，之后所有小模型 session 都用 CUDA EP 建，同样 SIGSEGV（崩溃栈
# hmct/converter/optimizer/move/move_reshape_node_back.py:600 ->
# _is_supported_two_features_arithmetic -> ORTExecutor(...).inference -> ort2.py:45）。
# 量化编译链路只需要能跑通的小模型推理，强制走 CPUExecutionProvider。
# ---------------------------------------------------------------------------
from hmct.executor.ort2 import ORTExecutor2  # noqa: E402

_DEVICE = os.environ.get("HB_ORT_DEVICE", "cpu")


def _get_support_devices(cls):  # noqa: ANN001, ARG001
    return [_DEVICE]


ORTExecutor2.get_support_devices = classmethod(_get_support_devices)
_log(f"[patch] ORTExecutor2.get_support_devices forced to ['{_DEVICE}']")
