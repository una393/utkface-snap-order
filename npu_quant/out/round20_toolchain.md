TOOLCHAIN: AVAILABLE

# 供应商工具链复查：N=∞ 与 N=20 两格（T3，对应评审 F3）

日期 2026-10-31。问题：能否把 **N=∞（关掉训练期网格投影）** 与 **N=20（after_snap）** 两个检查点送进同一条 hbdk4 厂商链（编译 → nash 指令级仿真推理），拿到部署链上的 ΔMAE 与输出偏置，以判断「关闭训练期投影」这条实践建议在部署链上是否仍然成立。

每个数字都可追到下列文件，正文用「路径:行/键」标注。

---

## 1 结论

**可用（TOOLCHAIN: AVAILABLE）。** 本轮只补了 **1 格**，不是 2 格：

- **N=∞（none, snap_every=0, seed 0）**：本轮新导出 ONNX 并新编 `.hbm`，新跑 200 张。
- **N=20（after_snap, snap_every=20, seed 0）**：**不需要重编**。既有 `npu_quant/onnx/relu_qat_s0_N20.onnx` 的导出源已由 `npu_quant/out/provenance.txt:43` 钉死为 `utk-relu_qat-after_snap-s0-1793033217`（weights=ema，|onnx−net| = 1.19e-07），2026-10-27 已编出 `npu_quant/out/relu_qat_s0_N20/relu_qat_s0_N20.hbm` 并跑过运行时（`npu_quant/out/mae_runtime_relu_qat_s0_N20.json`）。我用同一套脚本在 200 张上重算了这一格，读数与 `npu_quant/out/tables.md:65` 的 N20 行**逐位相同**，证明本轮代码路径与既有口径一致。

一句话结论（供写稿）：

> 关掉投影的收益**在部署链上仍然成立**（FP32 上 N20 比 N∞ 差 +0.006828，nash 上差 +0.006485，保留 95.0%）；但关掉投影**不消除 int8 部署代价**（ΔMAE 点估计 N=1 +0.0038 / N=20 +0.0034 / N=∞ +0.0040，几乎不动）。

---

## 2 引擎、镜像、工具链版本

### 2.1 引擎

任务书给的前提是「守护进程当前未运行」。**开工时它已经在跑**，`docker info` 直接返回 Server 段，没有任何 `failed to connect` 报错：

    Server:
     Containers: 5
      Running: 0
     ...
     Server Version: 29.6.2
     Storage Driver: overlay2
     Images: 7

因此 **Docker 启动尝试用了 0 次**（上限 3 次未动用）。

### 2.2 镜像（`docker images` 原文）

    ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0   476b9d7e5030   29.3GB

同一台机器上另有 `openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8`（21.9GB）与 `crpi-.../rdk_toolchain:v2.0`（28.2GB），本轮**未使用**。

### 2.3 容器内版本（`npu_quant/out/log_relu_qat_s0_Ninf.txt` 首几行原文）

    hbdk version: 4.7.5
    hmct version: 2.6.5
    hb_compile version: 3.5.3
    BPU march           : nash-m

---

## 3 编译

### 3.1 命令（可直接复制）

    cd /c/Users/USER/Desktop/paper/paper-utkface && MSYS_NO_PATHCONV=1 docker run --rm \
      --entrypoint /bin/sh \
      -v C:/Users/USER/Desktop/paper/paper-utkface:/work \
      -v D:/data/utkface:/data:ro -w /work \
      ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0 \
      -c 'python3 /work/npu_quant/hb_compile_patched.py -c /work/npu_quant/hb_config/relu_qat_s0_Ninf.yaml'

`MSYS_NO_PATHCONV=1` 是必要的：不加的话 git-bash 会把 `/work` 的路径改写成 `C:/Users/.../win32-x64/work`。

### 3.2 退出码与日志尾（`npu_quant/out/log_relu_qat_s0_Ninf.txt`）

    The hb_compile completes running.
    HBDK hbm perf SUCCESS
    COMPILE_RC=0

编译尝试 **1 次**（每个检查点上限 2 次），用时 16:53:06 → 17:01:02，退出码 **0**。

### 3.3 产物

| 项 | 值 | 出处 |
|---|---|---|
| `.hbm` 字节 | 6,487,504 | `npu_quant/out/relu_qat_s0_Ninf/relu_qat_s0_Ninf.hbm` |
| latency | 992.9 µs | `npu_quant/out/relu_qat_s0_Ninf/relu_qat_s0_Ninf.json` → `latency (us)` |
| FPS | 1007.19 | 同上 → `FPS` |
| DDR bytes / run | 32,731,392 | 同上 → `DDR bytes per run` |

**一致性旁证**：三格编出来的 `.hbm` 尺寸与延迟几乎相同——N=1 6,487,152 B / 993.3 µs、N=20 6,487,456 B / 993.0 µs（`npu_quant/out/tables.md:5-6`）、N=∞ 6,487,504 B / 992.9 µs（本轮）。也就是说**训练期是否投影，对编译产物本身几乎没有影响**，差别只在权重取值，不在图结构。

---

## 4 运行时（nash x86 指令级仿真，200 张）

    relu_qat_s0_Ninf: n_ok=200/200 MAE=0.057684 RMSE=0.089008 max=0.555492
    RT_RC=0

产物 `npu_quant/out/mae_runtime_relu_qat_s0_Ninf.json`。后端是容器内的 nash-m 指令级仿真，**不是 S100 板卡**，故本文不给板端延迟/功耗、也不给板端 MAE。

---

## 5 两格读数（本轮要求的正产物）

口径：ΔMAE = 逐图像均值 `mean(|pred_quant − y| − |pred_fp32 − y|)`（**正值 = 量化更差**）；区间 = 同一批图像上的逐图像配对 t 区间，df = n−1，t(0.975) = 1.9720；输出偏置 = `mean(pred_quant − pred_fp32)`，SE 同法。全部在**同一批 200 张**上算。

| 格 | checkpoint | FP32 MAE | nash MAE | ΔMAE | 95% CI | 输出偏置 | n |
|---|---|---|---|---|---|---|---|
| **N=∞**（none, s0） | `utk-relu_qat-none-s0-1793111850` | 0.053659 | 0.057684 | **+0.004025** (SE 0.002650) | **[−0.001201, +0.009252]**（含 0） | **−0.00396** (SE 0.00315) | 200/200 |
| **N=20**（after_snap, s0） | `utk-relu_qat-after_snap-s0-1793033217` | 0.060487 | 0.063871 | **+0.003368** (SE 0.001430) | **[+0.000547, +0.006189]**（不含 0） | **+0.00259** (SE 0.00156) | 199/200 |

出处：FP32 列 = `npu_quant/out/mae_fp32.json` → `models[<tag>]["mae"]`；nash 列 = `npu_quant/out/mae_runtime_<tag>.json` → `mae`；差值、区间、偏置由这两份文件的 `preds` 与 `targets` 逐图像重算（2026-10-31）。N=20 行与 `npu_quant/out/tables.md:65` 完全一致。

### 5.1 两协议在部署链上的对比（这是 F3 真正要的那一格）

同一批图像上，`N=20 − N=∞` 的任务 MAE 差：

| 口径 | N20 − N∞ | 逐图像配对 95% CI | n |
|---|---|---|---|
| **FP32** | **+0.006828** | 点估计相减，不给区间 | 200 |
| **nash 部署链** | **+0.006485** (SE 0.002860) | **[+0.000845, +0.012124]** | 199 |

比值 0.006485 / 0.006828 = **0.950**。即**部署链保留了约 95% 的训练侧收益**，方向与量级都没有被量化吃掉。

---

## 6 与既有四格的口径对照

`npu_quant/out/tables.md:64-66` 的既有读数（nash 列，可直接与本轮两格并排）：

| tag | nash 输出偏置 | nash ΔMAE | n |
|---|---|---|---|
| `relu_qat_s0_N1` | +0.01710 ± 0.00187 | +0.003834 ± 0.001919 | 200 |
| `relu_qat_s0_N20` | +0.00259 ± 0.00156 | +0.003368 ± 0.001430 | 199 |
| `gelu_s0_N1` | +0.00098 ± 0.00197 | −0.000157 ± 0.001821 | 199 |
| `bilinear_glu_s0_N1` | — | — | 未编出 `.hbm`（`tables.md:8` 全为 None） |
| **`relu_qat_s0_Ninf`**（本轮新增） | **−0.00396 ± 0.00315** | **+0.004025 ± 0.002650** | 200 |

### 6.1 三处口径必须写清

1. **这 200 张不是论文正文的 400 张，也不是 held-out 子集。** 200 张 = `npu_quant/make_test_inputs.py:31` 的 `rows[:200]`，其来源 `npu_quant/test_set.json` 是 seed-0 划分的 `val[800:1600]`（`npu_quant/make_calib.py:50-54`）；与 2,000 张监控子集重叠 **112/200**（seed 0）。因此本页的数字**只能同口径横向比**，不能与 `quant_runs/deploy_table_v2.csv` 的 400 张列（该表 N=∞ 两行 mae_fp32 = 0.0556 / 0.0552，`deploy_table_v2.csv` 第 5、29 行）直接相减或互相校验——同一条 N=∞ run 在这 200 张上得 0.053659，比 400 张口径低，与 README 已记的「部署表 FP32 列高 0.002–0.003」方向一致。
2. **校准集与测试集对全部检查点共用同一份**：`npu_quant/calib/` 100 个 `.npy`（seed-0 的 `val[:100]`，`npu_quant/make_calib.py:44-47`）与 `npu_quant/testbin/` 200 个 `.bin`，本轮未改、未重采样。所以五格之间的差异不是校准集差异造成的。
3. **N=∞ 的权重取自 live（`model_state_dict`），不是 ema。** `scripts/deploy_table.py:148` 写死 `key = "ema_state_dict" if mode != "none" else "model_state_dict"`：`none` 模式下影子从不更新（该 run `val_mae_ema` = 0.258308 vs live 0.053188），用 ema 会导出未训练的初始化权重。导出命令里显式写了 `--weights live`。

---

## 7 口径警告：这张表的区间是什么单位（评审 F3 的落点）

上面每个 95% 区间都是 **单 checkpoint × 200 张图 × 逐图像配对 t**，随机单位 = **图像**。

而自变量（是否投影、投影间隔）施加在 **run** 上，本页每格只有 **1 个 run**（seed 0，且该 seed 同时决定数据划分与参数初始化）。因此：

- 区间宽度描述的是**图像抽样误差**，**不含 run 间变异**；
- 不能据这些区间说「N=∞ 的量化代价显著小于 N=20」，也不能用它给「关闭投影」这条实践建议做显著性担保；
- **N=∞ 的 ΔMAE 区间含 0，不等于「量化没有代价」**：SE 0.00265 来自这 200 张图像自身的噪声宽度，点估计 +0.0040 与 N=20 的 +0.0034、N=1 的 +0.0038 同量级；把这句读成「不投影就没有量化代价」是误读。

可以主张的是（措辞建议）：

- 「在部署链上，关掉投影把任务误差降低 0.0065（同一批 200 张，n = 1 个 run/格，区间为逐图像配对，不含 run 间变异）」；
- 「int8 部署代价在三个间隔上量级一致（+0.0034 ... +0.0040），关掉投影没有消除它」。

---

## 8 本轮改动与副作用（必须披露）

### 8.1 对 `npu_quant/*.py` 的改动

**只有一处**：`npu_quant/gen_hb_configs.py:36-37`

- 改前：`TAGS = ["relu_qat_s0_N1", "relu_qat_s0_N20", "gelu_s0_N1", "bilinear_glu_s0_N1"]`（单行）
- 改后：同一列表拆成两行，末尾追加 `+ "relu_qat_s0_Ninf"`
- 效果：多生成 `npu_quant/hb_config/relu_qat_s0_Ninf.yaml`（584 B）；其余 4 个旧 yaml 被重写但与改前**逐字节同尺寸**（590 / 566 / 578 / 581 B）。

其它 `npu_quant/*.py` 未改。

### 8.2 对既有数据文件的副作用

- `npu_quant/out/mae_fp32.json`：新增 `relu_qat_s0_Ninf` 键；原有的 4 个模型被重算，`preds` 最大漂移 ≤ **5.96e-07**（onnxruntime float32 的非确定性），4 个 MAE 六位小数完全不变（0.075599 / 0.060487 / 0.079308 / 0.076025）。改前备份在 `C:/Users/USER/AppData/Local/Temp/t3_mae_fp32_before.json`。
- `npu_quant/out/fold_patch.log`：追加了本轮编译的折叠记录（现 160 行，追加在原 8,909 B 之后）。
- 仓库根目录新出现 `hb_compile.log`（150,739 B，17:01）——`hb_compile` 的默认日志落点，非有意新建。

### 8.3 新建文件清单（6 个，上限 6 个）

1. `npu_quant/onnx/relu_qat_s0_Ninf.onnx`
2. `npu_quant/hb_config/relu_qat_s0_Ninf.yaml`
3. `npu_quant/out/relu_qat_s0_Ninf/relu_qat_s0_Ninf.hbm`
4. `npu_quant/out/relu_qat_s0_Ninf/` 下的编译 json 与日志
5. `npu_quant/out/mae_runtime_relu_qat_s0_Ninf.json`
6. `npu_quant/out/round20_toolchain.md`（本文件）

外加 §8.2 那条非有意产生的根目录 `hb_compile.log`。

---

## 9 复现步骤（从零）

    # 1 导出 N=inf 的 ONNX（none 臂必须用 live 权重）
    C:/Users/USER/.conda/envs/e2e/python.exe npu_quant/export_npu_onnx.py \
      --ckpt results/utk-relu_qat-none-s0-1793111850/best.pth \
      --head relu_qat --weights live --out npu_quant/onnx/relu_qat_s0_Ninf.onnx

    # 2 生成编译 yaml（必须带 MSYS_NO_PATHCONV）
    cd /c/Users/USER/Desktop/paper/paper-utkface && MSYS_NO_PATHCONV=1 \
      C:/Users/USER/.conda/envs/e2e/python.exe npu_quant/gen_hb_configs.py /work

    # 3 编译（见 §3.1 的 docker run 命令）

    # 4 FP32 基线（重算全部 onnx 在 200 张上的 preds）
    C:/Users/USER/.conda/envs/e2e/python.exe npu_quant/eval_mae_fp32.py

    # 5 nash 运行时（容器内，12 worker）
    MSYS_NO_PATHCONV=1 docker run --rm -v C:/Users/USER/Desktop/paper/paper-utkface:/work \
      -w /work ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0 -c \
      'python3 /work/npu_quant/eval_mae_runtime.py relu_qat_s0_Ninf 200 12'

导出侧自检（本轮实测）：`[verify] onnxruntime vs torch max|diff| = 4.768e-07`，输入 `sequence_images [1,1,3,224,224]`、输出 `age [1,1]`、params 4.409M。

---

## 10 资源用量

| 项 | 上限 | 本任务实际 |
|---|---|---|
| Docker 引擎启动尝试 | 3 次 | **0 次**（引擎已在运行） |
| 编译尝试 | 每检查点 2 次 | **1 次**（N=∞，退出码 0） |
| GPU | 0.5 h | **0 h**（nash 是 x86 指令级仿真） |
| 新建文件 | 6 个 | 6 个（+1 个非有意的根目录日志） |
