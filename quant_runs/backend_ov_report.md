# 第三个部署后端：Intel OpenVINO INT8（x86 CPU，NNCF PTQ）

生成时间 2026-11-02 02:40:11；脚本 `scripts/round20_backend_ov.py`；解释器 3.13.1。

回答的问题：训练侧测到的**间隔效应**（间隔 N=1 相对关闭投影 N=∞ 更差）在这条 OpenVINO INT8 链上是否仍在、保留多少。

## 1 环境与量化配置（固定项，逐条）

| 项 | 值 |
|---|---|
| openvino | 2026.4.1-22982-e213a147257-releases/2026/4 |
| nncf | 3.4.0 |
| python / onnx / torch | 3.13.1 / 1.18.0 / 2.7.1+cu128 |
| CPU | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel |
| 设备 | CPU（全程 0 次 GPU） |
| 校准集 | `npu_quant/calib/*.npy` 100 张（与厂商链同一批） |
| 测试集 | `npu_quant/testbin/*.bin` 200 张（= `npu_quant/test_set.json` rows[:200]，逐项比对一致） |
| 量化调用 | `nncf.quantize(ov_model, nncf.Dataset(calib100), subset_size=100)` |
| 其余参数 | 全默认：preset=None→PERFORMANCE，target_device=ANY，fast_bias_correction=True，model_type=None，ignored_scope=None，advanced_parameters=None（含 smooth_quant_alphas matmul=0.95），校准算法为 NNCF 默认统计收集，未手调 |

实测量化图（N1_s0，逐节点清点）：81 个 conv/MatMul 站点中 **55** 个的权重带量化节点；权重站点共 **53** 个，全部 per-channel（52 per-channel / 1 per-tensor）、全部 symmetric（无 zero-point：带 Subtract 的 0 个）。其中 41 个以 `int8 Constant → Convert(f32) → Multiply(逐输出通道 scale)` 反量化子图表达（int8 常量 41 个），12 个以 levels=255 的 FakeQuantize 表达。激活侧 **116** 个 FakeQuantize、全部 per-tensor（per-channel 0 个），其 input_low 为标量；其中 69 个区间为 [0, r]（非负张量，u8 侧，input_low=0），47 个为 ±r 型区间（实测 −lo/hi 中位数 1.007874，即 NNCF 对称方案把 254 级映射到 ±r，256/254 = 1.007874，两者一致）；levels 集合 [255, 256]。整图 FakeQuantize 128 个、算子 2180 个。

## 2 检查点与数据锚点

16 个检查点 = 8 个 seed × 2 个协议（`snap_every=1` 为间隔 1；`snap_every=0` 为关闭投影 N=∞），全部 `head=relu_qat`、`ema_mode=none`、`split_seed=0`（s0 两行的 `split_seed` 为 -1，即 `train.py:187` 的「与 seed 同值」，seed 0 → 0）、`ema_decay=0.99`。`ema_mode=none` 下影子从不更新，故取 `model_state_dict`（live）权重。

| 协议 | seed | run 目录 | snap_every | split_seed |
|---|---|---|---|---|
| N1 | 0 | `utk-relu_qat-none-s0-1793011336` | 1(默认) | -1(默认→seed) |
| N1 | 1 | `utk-relu_qat-none-s1-1793445602` | 1 | 0 |
| N1 | 2 | `utk-relu_qat-none-s2-1793446593` | 1 | 0 |
| N1 | 3 | `utk-relu_qat-none-s3-1793447591` | 1 | 0 |
| N1 | 4 | `utk-relu_qat-none-s4-1793448669` | 1 | 0 |
| N1 | 5 | `utk-relu_qat-none-s5-1793449730` | 1 | 0 |
| N1 | 6 | `utk-relu_qat-none-s6-1793450777` | 1 | 0 |
| N1 | 7 | `utk-relu_qat-none-s7-1793451837` | 1 | 0 |
| Ninf | 0 | `utk-relu_qat-none-s0-1793111850` | 0 | -1 |
| Ninf | 1 | `utk-relu_qat-none-s1-1793452884` | 0 | 0 |
| Ninf | 2 | `utk-relu_qat-none-s2-1793453651` | 0 | 0 |
| Ninf | 3 | `utk-relu_qat-none-s3-1793454406` | 0 | 0 |
| Ninf | 4 | `utk-relu_qat-none-s4-1793455163` | 0 | 0 |
| Ninf | 5 | `utk-relu_qat-none-s5-1793455928` | 0 | 0 |
| Ninf | 6 | `utk-relu_qat-none-s6-1793456701` | 0 | 0 |
| Ninf | 7 | `utk-relu_qat-none-s7-1793457541` | 0 | 0 |

## 3 FP32 一致性闸门

门槛：OpenVINO FP32 与 torch FP32 在同一批 200 张上的 MAE 差 ≤ 0.0001。实测 **16/16 个检查点通过，max|ΔMAE| = 3.802e-08**；逐图像输出最大偏差 max|Δy| = 1.222e-06。

| run | MAE torch FP32 | MAE OV FP32 | abs(dMAE) | max abs(dy) | 闸门 |
|---|---|---|---|---|---|
| utk-relu_qat-none-s0-1793011336 | 0.075129 | 0.075129 | 1.62e-08 | 8.64e-07 | PASS |
| utk-relu_qat-none-s1-1793445602 | 0.078621 | 0.078621 | 4.84e-09 | 8.34e-07 | PASS |
| utk-relu_qat-none-s2-1793446593 | 0.073124 | 0.073124 | 1.06e-08 | 5.66e-07 | PASS |
| utk-relu_qat-none-s3-1793447591 | 0.074694 | 0.074694 | 2.77e-08 | 7.15e-07 | PASS |
| utk-relu_qat-none-s4-1793448669 | 0.079736 | 0.079736 | 1.37e-08 | 8.94e-07 | PASS |
| utk-relu_qat-none-s5-1793449730 | 0.076087 | 0.076087 | 7.99e-09 | 6.85e-07 | PASS |
| utk-relu_qat-none-s6-1793450777 | 0.078409 | 0.078409 | 1.45e-08 | 8.94e-07 | PASS |
| utk-relu_qat-none-s7-1793451837 | 0.079277 | 0.079277 | 3.80e-08 | 7.15e-07 | PASS |
| utk-relu_qat-none-s0-1793111850 | 0.053659 | 0.053659 | 2.79e-09 | 6.56e-07 | PASS |
| utk-relu_qat-none-s1-1793452884 | 0.056845 | 0.056845 | 1.25e-09 | 8.05e-07 | PASS |
| utk-relu_qat-none-s2-1793453651 | 0.057538 | 0.057538 | 1.02e-08 | 6.56e-07 | PASS |
| utk-relu_qat-none-s3-1793454406 | 0.057191 | 0.057191 | 1.33e-08 | 7.75e-07 | PASS |
| utk-relu_qat-none-s4-1793455163 | 0.059360 | 0.059360 | 5.24e-09 | 9.83e-07 | PASS |
| utk-relu_qat-none-s5-1793455928 | 0.057808 | 0.057808 | 1.58e-09 | 1.04e-06 | PASS |
| utk-relu_qat-none-s6-1793456701 | 0.057246 | 0.057246 | 9.95e-09 | 1.22e-06 | PASS |
| utk-relu_qat-none-s7-1793457541 | 0.059181 | 0.059181 | 1.27e-08 | 6.26e-07 | PASS |

## 4 逐 seed 读数（16 行）

MAE 单位同训练：target = age/100，200 张。ΔMAE = int8 − OV FP32。

| # | 协议 | seed | run | torch FP32 | OV FP32 | OV int8 | ΔMAE(int8−FP32) | lat FP32 ms | lat int8 ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | N1 | 0 | utk-relu_qat-none-s0-1793011336 | 0.075129 | 0.075129 | 0.080401 | +0.005272 | 15.45 | 4.01 |
| 2 | N1 | 1 | utk-relu_qat-none-s1-1793445602 | 0.078621 | 0.078621 | 0.079195 | +0.000574 | 19.41 | 4.56 |
| 3 | N1 | 2 | utk-relu_qat-none-s2-1793446593 | 0.073124 | 0.073124 | 0.071716 | -0.001408 | 18.60 | 4.56 |
| 4 | N1 | 3 | utk-relu_qat-none-s3-1793447591 | 0.074694 | 0.074694 | 0.080600 | +0.005907 | 17.03 | 3.93 |
| 5 | N1 | 4 | utk-relu_qat-none-s4-1793448669 | 0.079736 | 0.079736 | 0.082960 | +0.003224 | 18.68 | 4.08 |
| 6 | N1 | 5 | utk-relu_qat-none-s5-1793449730 | 0.076087 | 0.076087 | 0.080987 | +0.004901 | 16.23 | 3.88 |
| 7 | N1 | 6 | utk-relu_qat-none-s6-1793450777 | 0.078409 | 0.078409 | 0.086823 | +0.008414 | 21.77 | 4.43 |
| 8 | N1 | 7 | utk-relu_qat-none-s7-1793451837 | 0.079277 | 0.079277 | 0.083506 | +0.004230 | 17.04 | 4.15 |
| 9 | Ninf | 0 | utk-relu_qat-none-s0-1793111850 | 0.053659 | 0.053659 | 0.062311 | +0.008653 | 14.99 | 3.65 |
| 10 | Ninf | 1 | utk-relu_qat-none-s1-1793452884 | 0.056845 | 0.056845 | 0.067083 | +0.010238 | 15.27 | 3.78 |
| 11 | Ninf | 2 | utk-relu_qat-none-s2-1793453651 | 0.057538 | 0.057538 | 0.068830 | +0.011291 | 12.56 | 4.04 |
| 12 | Ninf | 3 | utk-relu_qat-none-s3-1793454406 | 0.057191 | 0.057191 | 0.060284 | +0.003093 | 11.79 | 3.41 |
| 13 | Ninf | 4 | utk-relu_qat-none-s4-1793455163 | 0.059360 | 0.059360 | 0.061307 | +0.001947 | 12.11 | 3.65 |
| 14 | Ninf | 5 | utk-relu_qat-none-s5-1793455928 | 0.057808 | 0.057808 | 0.062956 | +0.005148 | 12.51 | 3.75 |
| 15 | Ninf | 6 | utk-relu_qat-none-s6-1793456701 | 0.057246 | 0.057246 | 0.062710 | +0.005464 | 11.95 | 3.03 |
| 16 | Ninf | 7 | utk-relu_qat-none-s7-1793457541 | 0.059181 | 0.059181 | 0.065117 | +0.005936 | 12.57 | 3.78 |

## 5 run 层统计（n = 8，df = 7，逐 seed 配对）

| 量 | 均值 | sd | 95% t 区间 | 同号 | 符号检验 p |
|---|---|---|---|---|---|
| 间隔效应 @FP32（N1 − N∞） | +0.019531 | 0.002196 | [+0.017695, +0.021366] | 8 正 / 0 负 | 0.0078 |
| 间隔效应 @int8（N1 − N∞） | +0.016949 | 0.006660 | [+0.011381, +0.022517] | 8 正 / 0 负 | 0.0078 |
| 量化代价 int8 − FP32（N1，8 run） | +0.003889 | 0.003096 | [+0.001301, +0.006477] | 7 正 / 1 负 | 0.0703 |
| 量化代价 int8 − FP32（N∞，8 run） | +0.006471 | 0.003319 | [+0.003696, +0.009246] | 8 正 / 0 负 | 0.0078 |
| 量化代价 int8 − FP32（16 run 合并） | +0.005180 | 0.003375 | [+0.003382, +0.006979] | 15 正 / 1 负 | 0.0005 |


合并行 n = 16（两个协议各 8 个 run），用 t_{0.975,15} = 2.131，只描述量化代价的整体量级，不是跨协议配对对比。
四个格子的均值：N1/FP32 **0.076885**、N∞/FP32 **0.057354**、N1/int8 **0.080774**、N∞/int8 **0.063825**。

## 6 间隔效应穿过本后端保留的比例

保留比例 = (N1_int8 − N∞_int8) / (N1_fp32 − N∞_fp32) = 0.016949 / 0.019531 = **0.868**。

逐 seed 比值（分母为该 seed 的 FP32 间隔效应）均值 0.856，取值范围 [0.185, 1.161]——分母逐 seed 只有 0.01–0.02 量级，单 seed 比值离散度大，**以「均值之比」为准**，逐 seed 比值只作旁证。

**一句话结论**：间隔效应在这条链上仍在：FP32 上 N1−N∞ = +0.019531（8/8 同号，p = 0.0078），OpenVINO int8 上 = +0.016949（8 正 / 0 负，p = 0.0078，区间不含 0），保留 86.8%；缺口的 13.2% 主要来自关闭投影那一端被量化削弱得更多（int8−FP32 在 N∞ 为 +0.006471、在 N=1 为 +0.003889），而不是 N=1 在 int8 上变好。INT8 本身 16 run 合并的 ΔMAE 为 +0.005180（sd 0.003375），远小于间隔效应本身。

## 7 边界

- 这是**第三方运行时（Intel OpenVINO）+ 独立量化器（NNCF）**的一路：ONNX 由本脚本用 torch 导出，量化由 NNCF 用默认预设完成，与厂商 hbdk4 链（BPU/nash）、与 onnxruntime 静态 QDQ 链**实现互不共享**。三路只能互相印证，**不能互相替代**：任何一路的读数都不外推到另一路。
- 延迟只作 **CPU 相对参考**（同一台机器、同一进程、单张串行）：OV FP32 中位 15.36 ms/张、int8 中位 3.90 ms/张（比值 3.94×）。这**不是硬件结论**，也**不得**与 BPU 的 µs 数相比（不同指令集、不同量化器、不同内存层级）。
- **不是逐算子全 int8**：NNCF 默认预设下 81 个 conv/MatMul 站点里只有 55 个的权重被量化，未量化的 26 个中 24 个是 SE 模块的 fc1/fc2 1×1 卷积（权重留 fp32），另 2 个是 self-attention 的 QKᵀ MatMul（其第二操作数是激活经 Reshape，本就不是权重）。激活侧 116 个站点各自带 FakeQuantize。故本条链是「55/81 个 conv/MatMul 站点量化权重 + 116 个激活站点量化 + 24 个 SE 卷积权重留 fp32」的混合精度，任何「全 int8」的表述都不成立。
- torch FP32 基线由**驱动解释器** `D:/tool/Python` 的 torch 2.7.1+cu128（CPU）产生，不是训练环境的 torch；已用训练环境 torch 2.12.0+cu132（C:/Users/USER/.conda/envs/e2e/python.exe）在同批 200 张上复核两个检查点：N1_s0 得 0.075128953 vs 本报告 0.075128960（|Δ| = 6.5e-09）；Ninf_s0 得 0.053658542 vs 本报告 0.053658541（|Δ| = 1.2e-09） ⇒ 基线不随 torch 版本漂移。
- 统计单位为 run（seed）：n = 8、df = 7；t 区间用 t_{0.975,7} = 2.365；符号检验为精确双侧二项。
- 测试集 200 张是**评估子集**（test_set.json 的 [800:1600]），与全验证集口径的绝对值不可直接比。

## 8 产物与偏离

- 脚本 `scripts/round20_backend_ov.py`（只读既有产物，写 `quant_runs/backend_ov_*`）。
- 本报告 `quant_runs/backend_ov_report.md` 与 `quant_runs/backend_ov_report.json`（含逐 run 原始读数、ONNX 的字节数与 sha1）。
- ONNX/IR 目录 `quant_runs/backend_ov_onnx/`：保留 N1_s0, Ninf_s0 的 FP32 ONNX 与 int8 IR（xml+bin），其余 run 的中间件用完即删，字节数与 sha1 记在 JSON 里。
- 墙钟合计 7.1 分钟；GPU 使用 0 次。
- 偏离：torch FP32 基线用驱动解释器 D:/tool/Python 的 torch 2.7.1+cu128（CPU），而非训练环境 torch 2.12.0+cu132；已用后者在同批 200 张上复核 N1_s0 / Ninf_s0，|ΔMAE| 最大 6.5e-09，故基线不随 torch 版本漂移。
- 偏离：NNCF 默认预设没有量化 24 个 SE 1×1 卷积的权重（详见 §1 与 §7）：这是预设行为，未做任何 ignore/手调；本条链因此是混合精度，不当作逐算子全 int8。

