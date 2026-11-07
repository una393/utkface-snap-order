# S100 真实工具链 int8 量化报告（paper-utkface）

生成时间：2026-10-27。所有数字均来自本机 Docker 容器 `ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0`
里实际跑出来的产物；未取得的一律写「未取得」。

工具链版本（容器内自报）：
- `hb_compile version: 3.5.3`（hb_compile 没有 `--version` 子命令，版本取自编译日志首部）
- `hbdk version: 4.7.5` / `hmct version: 2.6.5`
- march：`nash-m`（BPU core number 1）

## 0. 一句话结论

**真实 S100 工具链的 8-bit PTQ 不是「空操作」。** 在同一 200 张测试图上（seed 0 划分的 val[800:1000]），
relu_qat 头的部署模型 MAE 从 FP32 的 0.0756 升到 **0.0794**（x86 nash 仿真器实测，Δ=+0.0038，
逐样本配对 ±0.0019，约 2σ）；relu_qat(N=20) 从 0.0605 升到 **0.0639**（Δ=+0.0034）。
而 **gelu 头几乎没有代价**：0.0793 → **0.0793**（Δ=−0.0002 ± 0.0018，统计上为 0）；
**bilinear_glu 代价最大**：0.0760 → **0.0803**（Δ=+0.0042，逐样本 +0.0144 ± 0.0021，约 7σ）。
这与「8 bit 导出量化对任务指标影响极小」（旧仿真，仅量化权重且部署权重已在格点上）**不一致**——
真实工具链同时量化**激活**（percentile 阈值 + LUT 激活核），误差来源不同，量级大 10 倍以上
（0.0034–0.0038 vs 旧仿真的 ≤0.0003）。

## 1. 走通的链路

| 步 | 做了什么 | 状态 |
|---|---|---|
| 1 | 从 best.pth 导出静态 ONNX（4 个：relu_qat_s0_N1 / relu_qat_s0_N20 / gelu_s0_N1 / bilinear_glu_s0_N1） | ✅ |
| 2 | UTKFace val 前 100 张按训练预处理转 float32 NCHW npy | ✅ |
| 3 | 容器内 `hb_compile` 编译成 `.hbm`（需两个补丁，见 §2） | ✅ |
| 4 | 逐节点余弦（编译器第一方报告 `*_quant_info.json`） | ✅ |
| 5 | x86 上真的跑量化后模型（`hbrt4-run-model-nash`，nash-m 指令级仿真），算 MAE | ✅ 能做，见 §5 |
| 6 | 三头各编一次 + relu_qat N=1/N=20 | ✅ |

### 走通的命令（可直接复制）

```bash
IMG=ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0
ROOT=C:/Users/24399/Desktop/paper-utkface

# 1) 导出 ONNX（宿主 python，不动训练代码）
"C:/Users/24399/.conda/envs/e2e/python.exe" npu_quant/export_npu_onnx.py

# 2) 校准集（100 张，3x224x224 float32 npy）
"C:/Users/24399/.conda/envs/e2e/python.exe" npu_quant/make_calib.py

# 3) 编译（注意：镜像默认 entrypoint 在本机 exec 失败，必须 --entrypoint /bin/sh）
cd /c/Users/24399/Desktop/paper-utkface && MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /bin/sh \
  -v ${ROOT}:/work -v D:/data/utkface:/data:ro -w /work ${IMG} \
  -c 'python3 /work/npu_quant/hb_compile_patched.py -c /work/npu_quant/hb_config/<tag>.yaml'

# 4) 逐节点余弦
"C:/Users/24399/.conda/envs/e2e/python.exe" npu_quant/extract_quant_report.py --tag <tag> --root npu_quant/out

# 5) x86 上实跑 .hbm 算 MAE（容器内，200 张，12 并发）
cd /c/Users/24399/Desktop/paper-utkface && MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /bin/sh \
  -v ${ROOT}:/work -w /work ${IMG} -c 'python3 /work/npu_quant/eval_mae_runtime.py <tag> 200 12'
```

## 2. 工具链本身的两个坑（这是本任务真正花时间的地方）

### 2.1 `hb_compile` 在本机 **SIGSEGV（exit 139）**，根因两条

1. **常量折叠 + onnxruntime 建 session**：hmct 2.6.5 的
   `hmct/common/modifier/passes/constant_folding.py` 里 `_fold_other_node` 把「输入全为常量」的节点
   抽成单节点子模型，用 `ORTExecutor(...).inference(None)` 建 session。本镜像的 onnxruntime 1.19.0
   对这类子模型建 session 直接段错误。**首个崩点**是 int64 的
   `Mod  :: /temporal/layers.0/self_attn/Mod`（torch TransformerEncoder 在 T=1 时生成的取模节点，
   操作数是 Constant 2 和 3），与算子语义无关——同样内容单独建 session 能跑。
   faulthandler 抓到的原始栈（节选）：
   ```
   File ".../hmct/common/modifier/passes/constant_folding.py", line 305 in _fold_other_node
   File ".../hmct/common/modifier/passes/constant_folding.py", line 147 in _fold_const
   File ".../hmct/converter/optimizer/optimize.py", line 34 in optimize
   ```
2. **CUDA EP**：本机 docker 里 `ort.get_available_providers()` 含 `CUDAExecutionProvider`，
   `hmct/executor/ort2.py` 的 `get_support_devices()` 因此返回 `['cuda','cpu']`，
   于是这些小 session 全用 CUDA EP 建，第二个崩点在
   `hmct/converter/optimizer/move/move_reshape_node_back.py:600`。

**修复**（新文件，未改任何训练代码，也未改容器内代码——用 Python monkeypatch 在进程内打）：

- `npu_quant/hb_patch_fold.py`：覆盖 `ConstantFolding._fold_other_node`，仍用 `extract_submodel`
  建子模型，但求值改走 `onnx.reference.ReferenceEvaluator`（纯 Python）→ 失败退到裸 CPU ORT →
  再失败则跳过折叠（`return False`）。常量替换、`max_inc` 判定、`node.destroy()` 与原实现一致；
  日志写 `npu_quant/out/fold_patch.log`（本次 30 条 "folded via reference"，0 条 skip）。
- 同文件把 `ORTExecutor2.get_support_devices` 固定成 `["cpu"]`。
- `npu_quant/hb_compile_patched.py`：`import hb_patch_fold` 后调用 `horizon_tc_ui.hb_compile.main()`。

打补丁前：`exit=139 elapsed=30s`；打补丁后：`exit=0 elapsed=395s`，日志末尾
`The hb_compile completes running.` / `HBDK hbm perf SUCCESS`。

### 2.2 编译配置

沿用 `output/autopilot.yaml` 的参数（march nash-m、featuremap 输入、NCHW、float32 校准、latency、O2、core_num 1），
只改 onnx 路径 / 前缀 / input_name / input_shape / cal_data_dir：
`npu_quant/hb_config/<tag>.yaml` 由 `npu_quant/gen_hb_configs.py` 生成。

容器自报的、值得注意的警告（原文）：

```
WARNING The input_type_train is featuremap, the input_layout_train will not take effect
WARNING Please note that the parameter 'cal_data_type' has been deprecated.
WARNING find non positive threshold in /fpn/norm/InstanceNormalization_add2_input_1_HzCalibration, it will be set default.
WARNING find small threshold in <若干 SE fc1 权重>_HzCalibration, it will be set default.
```

> 控制头**前面**的 `/fpn/norm`（InstanceNorm）拿到了 non-positive threshold 并被改成默认阈值——
> 这是全模型量化误差最大的来源之一（见 §4），不是控制头自己。

## 3. 编译产物

见 `npu_quant/out/tables.md` 的「编译产物」表（由 `npu_quant/make_tables.py` 生成）。
汇总：4 个 .hbm 都在 6.47–6.49 MB 量级；编译器自报延迟 ≈ 0.99 ms、≈1007 FPS、DDR 32.7 MB/run。
	extbf{编译耗时}（同一台机器，且与其他任务共享 CPU，故仅供参考）：relu_qat N1 = 395 s（最干净的一次，
几乎独占 CPU）、relu_qat N20 = 676 s、gelu = 2262 s（GELU 的 LUT 核 + CPU 争用）。

## 4. 逐节点余弦（编译器第一方证据）

数据源：`npu_quant/out/<tag>/<tag>_quant_info.json`（每个量化节点的 `cosine_similarity`，
量化后输出 vs float 输出的余弦）。CSV 在 `npu_quant/out/cosine_<tag>.csv`。

（详见 `npu_quant/out/tables.md` 的「逐节点余弦：全局」与「控制头 + 头前 /fpn/norm」两表。）

要点：

1. **控制头本身量化得很准**：relu_qat 头的 Linear 0.999375 / RMSNorm(Pow) 0.997010 /
   Mul 0.999286 / ReduceMean 1.000000 / HzRsqrt 1.000000 / **Relu 0.999405** / 末层 Linear 1.000000。
2. **gelu 头明显更差**：Linear 0.990574、Tanh 0.990622、Mul 0.990657、**HzGelu 0.991085**（末层 Linear 仍 1.0）。
3. **H3 必须在两个层面分开说，结果不一致（重要，见 §9 与 §10）**：
   - **ONNX 图层**（`*_quant_info.json` 的节点类型统计，`/head/` 作用域）：relu_qat 头 10 个节点
     `{'Reshape':2,'Conv':2,'Pow':1,'ReduceMean':1,'HzRsqrt':1,'Mul':2,'Relu':1}`，**无 Tanh/GELU/Erf/Exp 类超越算子**；
     gelu 头 7 个节点含 `Tanh` 与 `HzGelu`。→ 这一层的说法（"控制头内超越算子 = 0"）成立。
   - **BPU 指令层**（`*_advice.json` 的 `op_type`，编译器自己给出的算子映射）：relu_qat 头的
     `Pow` / `Div_reciprocal`(HzRsqrt) / `Relu` **三个节点都被编译成 `b30.lut`**；
     gelu 头是 `Tanh` / `HzGelu` 两个 `b30.lut`；bilinear_glu 头 **0 个 b30.lut**。
     全模型 `b30.lut` 数：relu_qat 16、gelu 15、bilinear_glu 13（骨干 12 个 SE-Sigmoid + /fpn/norm 的 reciprocal 各头都有）。
     → 所以 **"控制头内 LUT 节点 = 0" 在指令层不成立**：`b30.lut` 是这颗 BPU 上所有一元逐元素算子的通用指令，
     ReLU 也走它。能成立的是"无超越算子 + 头部各节点余弦更高"（见下条与 §10）。
4. **最差的节点全在骨干 s3 阶段**（不是控制头）：0.933–0.96，例如
   `/backbone/s3/b8/conv2/conv/Conv` 0.938、`/backbone/s3/b8/conv2/bn/act/Relu` 0.933、
   `/backbone/s3/b7/conv3/conv/Conv` 0.944。整体：285 个节点里 <0.99 的有 111–114 个。
5. **一个异常点（已实测归因，见 §9 Q3）**：gelu 的 `/temporal/layers.0/norm2/LayerNormalization_mean1`
   余弦 = **−1.0**（`node_info.csv` 里该行 Output Data Type = `si16`、Quantized Cosine = `--`）。
   在 100 张校准图 + 8 张测试图上把该张量取出来（float 与 fake-quant 两版）相比：
   **max|d| = 1.14e-6、cos = 1.000000**（张量值 mean 0.0339 / std 0.0181 / min −0.0145 / max 0.0680）。
   即该节点在伪量化图里**没有被劣化**，−1.0 是编译器报告里的退化值（该行阈值 3.958074 实际属于相邻 Add 节点）。
   **逐节点余弦不能盲信**；剔除该行后 gelu 全局 min = 0.947076。

## 5. 能不能在 x86 上跑量化后模型 → **能**（并已算 MAE）

容器内有 hbdk4 的 x86 nash 运行时：
`/usr/local/lib/python3.10/dist-packages/hbdk4/runtime/x86_64_unknown_linux_gnu/nash/bin/hbrt4-run-model-nash`。
它是 C++ CLI（hbdk4 的 Python `Hbm` 类没有 run 接口），跑法：

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/hbdk4/runtime/x86_64_unknown_linux_gnu/nash/lib:/usr/ucp:/usr/local/lib/python3.10/dist-packages/hbdk4/compiler/_mlir_libs
hbrt4-run-model-nash -f <tag>.hbm -n <tag> -i <样本.bin> -o <outdir> --numpy
```

- 输入是**裸 float32 NCHW 二进制**（1×1×3×224×224×4 = 602 112 字节/张），因为是 `input_source: ddr` + `featuremap`。
- 输出 `<outdir>/age.npy`；日志 `Model Run Success`、`The real march is: nash-m`。
- **代价**：单张 35.8–39.0 s（`-t 2` = 74.4 s，说明是按推理次数线性计费，不是加载开销）。
  因此 200 张 × 4 模型用 8–12 并发跑了约 3 小时。

> 这是 **x86 上的 nash-m BPU 指令级仿真器实测**，不是真实 S100 板卡。下文 MAE 表里
> 「nash runtime」一列就是这个口径；「fake-quant」一列是 hmct 的伪量化 ONNX 模型在 onnxruntime 上的模拟。

### 5.1 MAE（三档口径，同一 200 张图，同图同序）

测试集：`npu_quant/testbin/0000.bin…0199.bin`（来自 `npu_quant/test_set.json` = seed 0 划分的 val[800:1600]
里的前 200 张，与 100 张校准集不相交；target = age/100）。

（表见 `npu_quant/out/tables.md`）

- FP32 基线（宿主 onnxruntime 跑导出的 ONNX）：relu_qat N1 0.075599 / relu_qat N20 0.060487 /
  gelu 0.079308 / bilinear_glu 0.076025。
- fake-quant（编译器模拟）：relu_qat N1 0.079808 / relu_qat N20 0.062922 / gelu 0.078453 / bilinear_glu 0.078032。
- **nash runtime（x86 实测）**：relu_qat N1 **0.079433**（n=200）/ relu_qat N20 **0.063871**（n=199）/
  gelu **0.079295**（n=199）/ bilinear_glu **0.080258**（n=200，RMSE 0.115484、max 0.769889）。两处 n=199 是 nash 运行时偶发的
  `unix-abstract:ipctrl-socket-share-id-17-pid-... RPC failed: failed to connect to all addresses`，
  重跑即可；如实记录未凑满 200。

**逐样本配对差（量化 − FP32，均值 ± SE，n≈200）**

| tag | Δmean（输出偏置） | ΔMAE |
|---|---|---|
| relu_qat_s0_N1 | fake-quant +0.01771 ± 0.00185 / nash +0.01710 ± 0.00187 | +0.00421 ± 0.00196 / +0.00383 ± 0.00192 |
| relu_qat_s0_N20 | fake-quant +0.00213 ± 0.00157 / nash +0.00259 ± 0.00156 | +0.00244 ± 0.00149 / +0.00337 ± 0.00143 |
| gelu_s0_N1 | fake-quant +0.00037 ± 0.00201 / nash +0.00098 ± 0.00197 | −0.00086 ± 0.00185 / −0.00016 ± 0.00182 |

读法（重要，别过度解读）：
- relu_qat N=1 的量化误差是**系统性正偏置**（+0.0172，约 9σ），N=20 只有 +0.0026（1.6σ），gelu ≈ 0。
- MAE 的变化（+0.0038 / +0.0034 / −0.0002）都只有 ~2σ 甚至更弱；**头与头之间的 MAE 差（~0.004）
  与 2×SE 同量级，n=200 还不足以把它钉成结论**，但方向在 fake-quant 与 nash runtime 两种口径下一致。
- 偏置（Δmean）的信噪比比 ΔMAE 高一个量级，是这组数据里最稳的信号。

### 5.2 仿真器 vs 编译器伪量化的一致性（我们自己的交叉验证）

同一批 200 张，比较 nash runtime 与 hmct fake-quant 的逐样本输出：

| tag | n | mean diff | sd | p95\|d\| | max\|d\| | 相关系数 |
|---|---|---|---|---|---|---|
| relu_qat_s0_N1 | 200 | −0.00061 | 0.01227 | 0.0252 | 0.0382 | 0.99736 |
| relu_qat_s0_N20 | 199 | +0.00018 | 0.01006 | 0.0214 | 0.0416 | 0.99816 |
| gelu_s0_N1 | 199 | +0.00075 | 0.01085 | 0.0197 | 0.0413 | 0.99778 |

结论：**hmct 的 fake-quant 是 nash runtime 的良好代理**（相关 0.997–0.998，MAE 差 ≤0.0005），
但**不是等价物**——逐样本最多差 0.04（=4 岁）。若要给论文报「部署后 MAE」，用 nash runtime 那列。

## 6. 与旧仿真结论对不上的地方（本报告的实质发现）

旧仿真（`quant_runs/deploy_table.csv` + 篇 2 草稿 §5.1）说：**8 bit 导出量化是空操作**
（部署权重离格 ≤0.37% 格距，再量化后 MAE 变化 ≤0.0003，8 bit 上的量化余弦恒为 1）。

真实工具链的 8-bit **PTQ** 不是空操作：

1. **量化的对象不同**。旧仿真只量化**权重**，且权重本身已被 QAT 吸附到格点；工具链还量化**激活**
   （percentile 阈值、per-tensor），并对 Sigmoid/GELU 之类用 **LUT 核**。误差源根本不同。
2. **量级差 10 倍以上**：ΔMAE = 0.0034–0.0038（relu_qat）vs ≤0.0003（旧仿真）；
   输出偏置 +0.017（N=1）在旧仿真里完全不存在。
3. **旧仿真「量化余弦恒为 1」是因为它只算权重**；真实工具链的逐节点余弦 min = 0.933（骨干 s3），
   285 个节点里 111 个 < 0.99。「余弦 = 1」这个现象在激活也量化后不再出现。
4. **但「任务指标几乎不变」这句在数量级上仍站得住**：相对劣化 +5%（0.0756→0.0794）。
   换句话说：**旧结论的方向对、前提错**——它把「权重已在格点上」误当成「模型对 int8 不敏感」。

### 与篇 1（头型）的关系

- 节点级：**relu_qat 头 ≫ gelu 头**（头内最差 0.9994 vs 0.9906），且 relu_qat 头内 LUT 类节点 = 0，
  gelu 头内有 2 个（Tanh + HzGelu）。H3 成立。
- 任务级：**没有出现「gelu 头 MAE 更差」**（−0.0002），反而 relu_qat 的 MAE 代价更大（+0.0038）。
  这与节点级余弦的方向相反，正是篇 2 已经踩过的坑：**余弦不是任务指标**。
  最可能的解释是偏置的符号与误差分布（relu_qat N=1 的量化误差是单向正偏置，被大误差项部分抵消后
  仍抬高了 MAE），但这只是解释，**没有做消融验证，属未验证假设**。

### 与篇 2（吸附间隔）的关系

- 真实工具链下 **N=20 仍然明显优于 N=1**：0.0639 vs 0.0794（FP32 也同向：0.0605 vs 0.0756）。
  即「吸附间隔是主旋钮」在 int8 PTQ 之后依然成立（劣化幅度也相近：+5.6% vs +5.1%）。
- 但量化偏置随间隔变化很大（N=1 +0.0172，N=20 +0.0026，约 7 倍差）——这点旧仿真没看到，
  因为旧仿真没有激活量化。

## 7. 做不到 / 未取得

| 项 | 状态 |
|---|---|
| 真实 S100 板卡上的延迟/功耗 | **未取得**（本机无板卡；编译器自报 0.99 ms 只是静态估算） |
| 板卡上的实测 MAE | **未取得**（同上）。现有 MAE 是 x86 上的 nash-m 指令级仿真 |
| bilinear_glu 的量化偏差与 MAE | 已编译（.hbm 6,553,136 B / 4601 s / 284 节点，min 0.932420、median 0.994365）；FP32 0.076025、伪量化 0.078032、**nash 实测 0.080258（n=200/200）** |

## 8. 产物清单

（文件都在 `C:\Users\24399\Desktop\paper-utkface\npu_quant\`）

| 类别 | 路径 |
|---|---|
| 导出 ONNX（4 个） | `npu_quant/onnx/{relu_qat_s0_N1,relu_qat_s0_N20,gelu_s0_N1,bilinear_glu_s0_N1}.onnx` |
| 校准集（100 张 npy） | `npu_quant/calib/0000.npy … 0099.npy` |
| 测试集定义 / 测试输入 | `npu_quant/test_set.json`、`npu_quant/testbin/0000.bin … 0199.bin` |
| 编译配置（4 个） | `npu_quant/hb_config/{relu_qat_s0_N1,relu_qat_s0_N20,gelu_s0_N1,bilinear_glu_s0_N1}.yaml` |
| **.hbm（4 个）** | `npu_quant/out/<tag>/<tag>.hbm` |
| 编译器量化报告 | `npu_quant/out/<tag>/<tag>_quant_info.json`、`..._node_info.csv`、`..._advice.json`、`..._advice.csv`、`<tag>.html`、`hb_compile.log` |
| 逐节点余弦 CSV | `npu_quant/out/cosine_<tag>.csv` |
| 汇总表（本报告所有表） | `npu_quant/out/final_summary.md`、`npu_quant/out/tables.md` |
| MAE 原始结果 | `npu_quant/out/mae_fp32.json`、`mae_fakequant.json`、`mae_runtime_<tag>.json` |
| 导出溯源 | `npu_quant/out/export_provenance.json`、`provenance.txt` |
| 脚本 | `npu_quant/export_npu_onnx.py`、`make_calib.py`、`make_test_inputs.py`、`gen_hb_configs.py`、`hb_compile_patched.py`、`hb_patch_fold.py`、`extract_quant_report.py`、`eval_mae_{fp32,fakequant,runtime}.py`、`probe_*.py`、`trace_export_provenance.py`、`inspect_advice.py`、`summarize_final.py`、`make_tables.py` |

## 9. 落笔前的口径问答（父 agent 的 Q1–Q5，全部有产物支撑）

### Q1 权重来源与协议：四个导出都是 **after_snap 的 ema_state_dict**，不是 none

反查方法（`npu_quant/trace_export_provenance.py`，实测不是推测）：穷举 `results/*/best.pth`
（21 个 run），各自载入 `ema_state_dict` 与 `model_state_dict` 两套权重，在与 ONNX 相同的
固定输入（`npu_quant/testbin/0000.bin`）上比较输出，取误差最小者。

| ONNX | 来源 run | 权重 | \|diff\| 对 ONNX 输出 | snap_every（deploy_table） | seed | val_mae_ema / live |
|---|---|---|---|---|---|---|
| relu_qat_s0_N1.onnx | utk-relu_qat-after_snap-s0-1793005531 | **ema** | 3.87e-07 | 1 | 0 | 0.072048 / 0.071957 |
| relu_qat_s0_N20.onnx | utk-relu_qat-after_snap-s0-1793033217 | **ema** | 1.19e-07 | 20 | 0 | 0.057359 / 0.059069 |
| gelu_s0_N1.onnx | utk-gelu-after_snap-s0-1793003829 | **ema** | 3.58e-07 | 1 | 0 | 0.074285 / 0.074152 |
| bilinear_glu_s0_N1.onnx | utk-bilinear_glu-after_snap-s0-1793007087 | **ema** | 8.94e-08 | 1 | 0 | 0.073587 / 0.073661 |

补充事实：
- `ckpt["args"]["snap_every"]` 在 N=1 的 run 里是 `None`（训练脚本只在实际传参时写入），
  deploy_table.csv 记作 1；N=20 的 run 里记的就是 20。
- `none` 协议的 ckpt **确实有** `ema_state_dict` 键，但三份的 ema 输出完全相同
  = `0.05676630139350891`（初始化后再未更新，与 CLAUDE.md 的警告一致）；`model_state_dict` 才随 run 变化
  （1793011336 → 0.37114897、1793027163 → 0.34271848、1793030915 → 0.26521823）。
  也就是说"none 不维护影子权重"这点被实测确认，但**导出脚本的 `or` 回退并不会因此触发**（键存在）。
- 因此：**本报告全部数字对应"after_snap + ema_state_dict + seed 0"**，与 deploy_table.csv 里
  `1,after_snap,0,...` / `20,after_snap,0,...` 两行是同一批权重。论文 §5.6 若写 none 协议，需改成 after_snap。

### Q2 HzRsqrt 算不算 LUT 算子：**算，而且 ReLU 也算**

证据：`npu_quant/out/<tag>/<tag>_advice.json`（编译器自己的算子映射表，每个条目有 `op_type` 与
`backend`）里，`/head/` 作用域的实际指令：

| 头 | 节点 → BPU op_type | 头部 b30.lut 数 | 全模型 b30.lut |
|---|---|---|---|
| relu_qat | `net.0/Gemm`→b30.conv2d；`net.1/Pow`→**b30.lut**；`net.1/ReduceMean`→b30.reduce；`net.1/Div_reciprocal`→**b30.lut**；`net.1/Mul`,`Mul_1`→b30.binary_eltwise；`net.2/Relu`→**b30.lut**；`net.3/Gemm`→b30.conv2d | 3 | 16 |
| gelu | `net.0/Gemm`→b30.conv2d；`net.2/Tanh`→**b30.lut**；`net.2/Mul`→b30.binary_eltwise；`net.3/Mul_1`(HzGelu)→**b30.lut**；`net.4/Gemm`→b30.conv2d | 2 | 15 |
| bilinear_glu | proj/out Gemm→b30.conv2d、Slice→hbdk.slice、Mul_2→b30.binary_eltwise | **0** | 13 |

结论：`b30.lut` 是这颗 BPU 上**一元逐元素算子的通用指令**（骨干里 12 个 SE-Sigmoid 与 /fpn/norm 的
reciprocal 也走它），ReLU 同样落在它上面。所以
- "控制头内 LUT 指令 = 0" **不成立**（relu_qat 头 3 条，比 gelu 头还多 1 条）；
- 能站住的是 **ONNX 图层无 Erf/Exp/Tanh/GELU 类超越算子**，以及**头部各节点的量化余弦**：
  relu_qat 头 0.997010–1.000000，gelu 头 0.990574–0.991085。
  即"同样的 LUT 指令，relu_qat 头把它用在了近线性/近恒等的函数上，误差更小"。

建议论文改写口径：H3 由"节点数 = 0"降级为两条可验证的陈述 ——
(1) ONNX 图层面超越算子计数：relu_qat 0、gelu 2（Tanh、HzGelu）；
(2) 头部逐节点量化余弦下界：relu_qat 0.9970 > gelu 0.9906。

### Q3 gelu 的 −1.000000 是哪来的：报告退化值，节点本身没坏

- 节点：`/temporal/layers.0/norm2/LayerNormalization_mean1`（type `ReduceMean`）。
  `node_info.csv` 该行：`ON=BPU, Threshold=3.958074, Calibrated Cosine=−1.000000, Quantized Cosine=--, Output Data Type=si16`。
- 取张量实测（`npu_quant/probe_mean1_tensor.py`、`npu_quant/probe_mean1_calib.py`，在
  `_calibrated_model.onnx` 上把该中间张量加为图输出，用容器内 hmct executor 跑）：
  - 8 张测试图：float 与 fake-quant **cos = 1.000000**，max|d| ~1e-6；
  - 100 张校准图：float `mean 0.0339 / std 0.0181 / min −0.0145 / max 0.0680`，
    fake-quant **max|d| = 1.14e-6、cos = 1.000000**；
  - 即便用该行自己报的阈值 3.958074 强行做对称 int8，cos 也是 0.977849，不会是 −1。
- 结论：该行不是真实的数值劣化（阈值实际属于相邻的 Add 节点，Quantized Cosine 本身就是 `--`）。
  剔除该行后 gelu 的全局 min = **0.947076**。逐节点余弦必须配合 `Output Data Type`/`ON` 列一起读。

### Q4 编译器自报 latency 的口径：单帧、单核、静态估算，不含前后处理

来源字段：`npu_quant/out/<tag>/<tag>.json` → `summary.performance`：
`{"FPS":1006.75,"latency (ms)":0.99,"latency (us)":993.3,"latency (us) by segments":[0.993],"multicore sync time (us)":0}`；
`summary["model info"]`：`{"BPU core number":1,"BPU march":"nash-m","compiling HBDK version":"4.7.5","compiling options":"{...,'opt':2,...,'balance':100,...}"}`。

- 输入是 1×1×3×224×224（单帧、batch 1），所以 993.3 µs 是**一次前向**的时间，不是序列吞吐的倒数除以帧数，也不能读成"1007 帧/秒的视频吞吐"。
- `BPU core number = 1`，与 autopilot.yaml 的 `core_num: 1` 一致；`multicore sync time = 0`。
- 这是编译器的**模型级静态估算**（hb_compile 全程没有硬件），**不含** JPEG 解码/归一化等宿主前后处理（图里没有这些算子），也**不含**驱动/调度开销，且**未在板卡上实测**。
- 四个 .hbm 的估算几乎一样（991.3–993.3 µs，DDR 31.21–31.28 MB/run），差异 <0.2%，不足以区分头型。

### Q5 nash 实测 vs 伪量化：三头 MAE 差 ≤0.001、bilinear 0.0022；逐样本偏置差都 ≤1.1×SE

| tag | FP32 | 伪量化（hmct） | nash 实测 | 伪量化 − FP32 | nash − 伪量化 |
|---|---|---|---|---|---|
| relu_qat_s0_N1 | 0.075599 | 0.079808 | 0.079433 (n=200) | +0.004209 | −0.000375 |
| relu_qat_s0_N20 | 0.060487 | 0.062922 | 0.063871 (n=199) | +0.002435 | +0.000949 |
| gelu_s0_N1 | 0.079308 | 0.078453 | 0.079295 (n=199) | −0.000855 | +0.000842 |
| bilinear_glu_s0_N1 | 0.076025 | 0.078032 | 0.080258 (n=200) | +0.002007 | **+0.002226** |

逐样本（nash − 伪量化，n=200/199/199/200）：relu_qat_N1 mean −0.00061（−0.7×SE）/ sd 0.01230 / p95\|d\| 0.0248 / corr 0.99736；
N20 +0.00018（+0.2×SE）/ 0.01009 / 0.0211 / 0.99816；gelu +0.00075（+1.0×SE）/ 0.01087 / 0.0194 / 0.99778；
bilinear_glu +0.00160（+1.1×SE）/ 0.02091 / 0.0417 / 0.99271（max|d| 0.0886，四个头里最大）。

即 **逐样本均值差四个头都在 ≤1.1×SE 内**，可以对父 agent 的原话负责。但 MAE 层面的差不是同一个量：
三个头 0.0004–0.0009，**bilinear_glu 0.00223**（≈1.5×该头的配对 SE），因为它是对少数大偏差样本敏感
的绝对误差均值，而这个头逐样本 spread 最大（sd 0.0209、max 0.0886）。所以准确说法是
"MAE 差 ≤0.001（三头）/ 0.0022（bilinear_glu），逐样本偏置差均 ≤1.1×SE"；
机制上只能说：伪量化是逐节点 q/dq 的浮点模拟，nash 跑的是编译后的指令流（含 LUT 近似、si8/si16 累加、
layout/padding），两者**不知道**谁更"真"——**没有板卡就无法归因**。建议的写法：
"编译器伪量化与 x86 nash 指令级实测在 MAE 上一致（≤0.001，bilinear_glu 0.0022），
逐样本偏置差 ≤1.1×SE；不解释差异来源（无板卡）"。

## 10. 本次实测对论文的三条硬约束（父 agent 需要据此改口径）

1. **H3 的"零 LUT"表述必须降级**（见 Q2）：指令层 relu_qat 头 = 3 条 `b30.lut`（Pow / HzRsqrt / ReLU），
   gelu 头 = 2 条（Tanh / HzGelu），bilinear_glu 头 = 0 条。能站住的是 ONNX 层超越算子计数与
   头部节点余弦（relu_qat 0.9970–1.0 vs gelu 0.9906–0.9911）。
2. **"部署权重"= after_snap 的 ema_state_dict**（见 Q1），与 deploy_table.csv 的 `1,after_snap,0` 与
   `20,after_snap,0` 两行同源；写成 none 会与本报告数字矛盾。
3. **头间 MAE 差异不足以定论**：n=200 上 relu_qat(N1) 0.0794 vs gelu 0.0793，差 0.0001，
   而配对 SE 是 0.0018–0.0019；bilinear_glu 0.0760(FP32)/0.0780(伪量化) 同理。
   头型能说的是**量化余弦**与**量化偏置的方向/大小**（relu_qat +0.0171±0.0019 系统性正偏，
   gelu +0.0010±0.0020 无偏），不是 MAE 排序。

另一个必须写进论文的事实：三个头的**最差节点都在骨干 s3 与 /fpn/norm**，
而不是控制头 —— /fpn/norm 的 InstanceNormalization 三件套余弦 0.9698–0.9801（relu_qat）、
0.9694–0.9801（gelu）、0.9736–0.9807（bilinear_glu），且编译日志三头都有
`find non positive threshold in /fpn/norm/InstanceNormalization_add2_input_1_HzCalibration, it will be set default.`。
控制头是整网量化最干净的区域之一。

