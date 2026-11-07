# Round 20 · 部署网格 vs 训练网格：关掉投影的优势是否只在网格匹配时成立

生成脚本 `scripts/round20_grid.py`（前向 12 次：3 检查点 × 3 网格 + 3 次 FP32 基线）；评估设备 `cuda`。

## 0 口径

- 评估集：`split_random(recs, 0.15, 0)[1][: 400]`，与 `scripts/deploy_table.py:136` 同一函数同一参数；n = 400。rows 前 5 = `[17967, 4943, 8796, 20916, 18926]`（sha256(rows) = `dfa7983511cddbe8af83e0831e5cd3a8c9d62d190488b8c52f085024bb9d72e7`），target 前 3 = `[0.5, 0.24, 0.28]`。
- 权重与激活：**只量化权重（int8），激活保持 FP32**，与 csv 的 `mae_8` 同口径；不重训、不用校准数据、BN 不折（沿用 `deploy_table.py` 的评估路径）。量化对象是 `nn.Conv2d` / `nn.Linear` 的 `weight`，与 `deploy_table.qdq_` 的遍历口径一致（`MultiheadAttention.in_proj_weight` 是普通 Parameter，不在其中；LayerNorm / BatchNorm 参数同样不动）。
- 三种网格：
  - `per_channel_max` — 逐输出通道对称 int8 max（基线，= deploy_table.qdq_）
  - `per_tensor_max` — 逐 tensor 对称 int8 max（每个权重张量一把 scale）
  - `per_channel_p99` — 逐输出通道取 |w| 的 99 百分位裁剪后对称 int8
- 符号：Δ = MAE(N=1) − MAE(N=∞)；**正值 = 关掉投影更优**。逐图配对差 d_i = |ŷ_i^{N=1} − y_i| − |ŷ_i^{N=∞} − y_i|，故 mean(d) 与 Δ 是同一个数的两种写法（同一批图）。

## 1 A · 基线复算（网格 i vs csv `mae_8`）

| run | csv mae_8 | 网格 i 复算 | 差 | ≤1e-4 | csv mae_fp32 | 复算 FP32 |
|---|---|---|---|---|---|---|
| `utk-relu_qat-none-s0-1793011336` | 0.0741 | 0.074067 | -0.000033 | 是 | 0.0741 | 0.074067 |
| `utk-relu_qat-none-s0-1793111850` | 0.0548 | 0.054847 | +0.000047 | 是 | 0.0552 | 0.055216 |
| `utk-relu_qat-after_snap-s0-1793033217` | 0.0593 | 0.059294 | -0.000006 | 是 | 0.0594 | 0.059364 |

三个检查点全部满足 |差| ≤ 0.0001：**是**（最大 |差| = 0.000047）。

## 2 B · 3 检查点 × 3 网格的任务 MAE（n = 400）

| 检查点 | FP32 | (i) per-channel max | (ii) per-tensor max | (iii) per-channel p99 |
|---|---|---|---|---|
| `utk-relu_qat-none-s0-1793011336`（N1_none） | 0.0741 | 0.0741 | 0.0736 | 0.1203 |
| `utk-relu_qat-none-s0-1793111850`（Ninf_none） | 0.0552 | 0.0548 | 0.0555 | 0.0989 |
| `utk-relu_qat-after_snap-s0-1793033217`（N20_after_snap） | 0.0594 | 0.0593 | 0.0601 | 0.1136 |

N=1 减 N=∞（Δ，正值 = 关掉投影更优）：

| 网格 | MAE(N=1) | MAE(N=∞) | Δ |
|---|---|---|---|
| per_channel_max | 0.0741 | 0.0548 | +0.0192 |
| per_tensor_max | 0.0736 | 0.0555 | +0.0181 |
| per_channel_p99 | 0.1203 | 0.0989 | +0.0213 |

相对网格 (i) 的**失配代价**（同一检查点的 MAE 增量）：

| 检查点 | (ii) − (i) | (iii) − (i) |
|---|---|---|
| N1_none | -0.0005 | +0.0462 |
| Ninf_none | +0.0006 | +0.0441 |
| N20_after_snap | +0.0008 | +0.0544 |

量化后权重相对 RMS 误差（全局，反映网格粗到什么程度）：

| 检查点 | (i) | (ii) | (iii) |
|---|---|---|---|
| N1_none | 1.822e-08 | 2.590e-02 | 1.255e-01 |
| Ninf_none | 1.018e-02 | 2.852e-02 | 1.373e-01 |
| N20_after_snap | 3.493e-03 | 2.855e-02 | 1.376e-01 |

## 3 C · 逐图配对检验（N=1 − N=∞，df = 399，n = 400）

| 网格 | mean | sd | se | t | 95 % t 区间 | bootstrap 10,000 mean | bootstrap 95 % | 含 0 |
|---|---|---|---|---|---|---|---|---|
| per_channel_max | +0.019221 | 0.053568 | 0.002678 | 7.176 | [+0.013955, +0.024486] | +0.019245 | [+0.013981, +0.024585] | 否 |
| per_tensor_max | +0.018096 | 0.054919 | 0.002746 | 6.590 | [+0.012698, +0.023494] | +0.018120 | [+0.012719, +0.023619] | 否 |
| per_channel_p99 | +0.021342 | 0.106460 | 0.005323 | 4.009 | [+0.010878, +0.031807] | +0.021345 | [+0.010864, +0.031468] | 否 |

t 临界值 1.9659272（双侧 95 %，df = 399）。bootstrap 用 `numpy.random.default_rng(0)` 重采样 10,000 次、取 2.5 / 97.5 百分位。符号分解（d_i > 0 即 N=1 更差）：

| 网格 | d_i > 0 | d_i < 0 | d_i = 0 | 中位差 |
|---|---|---|---|---|
| per_channel_max | 259 / 400 | 141 / 400 | 0 | +0.015338 |
| per_tensor_max | 251 / 400 | 149 / 400 | 0 | +0.013891 |
| per_channel_p99 | 237 / 400 | 163 / 400 | 0 | +0.016842 |

## 4 D · 结论

**在网格失配（ii / iii）下，「关掉投影更优」仍然成立，没有翻转。** Δ = +0.0192（i）/ +0.0181（ii）/ +0.0213（iii），三条 95 % 区间都不含 0（[+0.0140, +0.0245] / [+0.0127, +0.0235] / [+0.0109, +0.0318]）；相对网格匹配的 (i)，失配下 Δ 变化 -0.0011（ii，-6 %）与 +0.0021（iii，+11 %），量级变化不超过 0.0022，方向不变。

但**网格本身的代价比投影效应大**：网格 (iii) 把三个检查点的 MAE 抬高 +0.0462 / +0.0441 / +0.0544（相对 (i)），是投影效应 0.018–0.021 的 2–3 倍；网格 (ii) 只抬高 -0.0005 / +0.0006 / +0.0008。

## 5 E · 逐图预测产物（同一批图）

| npz | rows 与基准一致 | target 前 3 | sha256 |
|---|---|---|---|
| `quant_runs/grid_preds/utk-relu_qat-after_snap-s0-1793033217__per_channel_max.npz` | 是 | [0.5, 0.24, 0.28] | `02424fd1d5e52ecd63c98dd2de32f0d47622b0e23cf8d3f119ce6cfdcc2a8cb9` |
| `quant_runs/grid_preds/utk-relu_qat-after_snap-s0-1793033217__per_channel_p99.npz` | 是 | [0.5, 0.24, 0.28] | `8673d9f6a9a92faa530b573292c59dd4d1a4153f4637013dc09ac86c30c61980` |
| `quant_runs/grid_preds/utk-relu_qat-after_snap-s0-1793033217__per_tensor_max.npz` | 是 | [0.5, 0.24, 0.28] | `ca04e0f7518f23daf48c7e942f428a9362d652260e28fcd0018e560ea6e45c60` |
| `quant_runs/grid_preds/utk-relu_qat-none-s0-1793011336__per_channel_max.npz` | 是 | [0.5, 0.24, 0.28] | `457088baa2b8be57fcfabe604ff6ecc378ffd356622e93d0d3dd4b54a93ba4ce` |
| `quant_runs/grid_preds/utk-relu_qat-none-s0-1793011336__per_channel_p99.npz` | 是 | [0.5, 0.24, 0.28] | `bedef8ea1a0dfdad28e64dd04dca8c30d143779dab6f274bdb8e91d6bb65cc54` |
| `quant_runs/grid_preds/utk-relu_qat-none-s0-1793011336__per_tensor_max.npz` | 是 | [0.5, 0.24, 0.28] | `ec2644a26124e189bc6350181b86ccf207b0b4078f030b35c0f655996949044e` |
| `quant_runs/grid_preds/utk-relu_qat-none-s0-1793111850__per_channel_max.npz` | 是 | [0.5, 0.24, 0.28] | `3454af80469135e5d69a5a33f52cb4aa4c905d6ba1ebeae04f291365e6dd025b` |
| `quant_runs/grid_preds/utk-relu_qat-none-s0-1793111850__per_channel_p99.npz` | 是 | [0.5, 0.24, 0.28] | `48598cbfe0074555cb7cad7dd362afbe54df4797da5a858acb025f0223f944e3` |
| `quant_runs/grid_preds/utk-relu_qat-none-s0-1793111850__per_tensor_max.npz` | 是 | [0.5, 0.24, 0.28] | `c4334c9694e10f07caf898d1eb77d36ff035801c7d15072c176332ddd9e48652` |

九个 npz 的 `rows` / `target` 与基准逐元素相同（全部来自 `split_random(recs, 0.15, 0)[1][:400]`，故三个检查点用的是同一批图、同一个 y）。

## 6 deviations

1. **检查点 to csv 行**：本报告按任务要求以 `quant_runs/deploy_table_v3.csv` 为准。任务背景里写「after_snap@N=20 8 bit ≈ 0.0556」，但 csv 中 `utk-relu_qat-after_snap-s0-1793033217` 的 mae_8 = 0.0593；0.0554 属于 `utk-relu_qat-after_snap-s0-1793111850`（该 run 的 snap_every = 0，即 N=∞）。本报告用 1793033217（N=20），复算 0.059294，与 csv 一致。
2. **权重来源**：两臂 none 取 `model_state_dict`（live 权重，`deploy_table.py:148` 的 mode!=none 分支），after_snap 取 `ema_state_dict`；strict=False 载入，与基线脚本一致。
3. **激活口径**：csv 的 `mae_8` 实测是「只量化权重、激活 FP32」（本轮网格 (i) 用同一份 `qdq_` 复算出 0.074067 / 0.054847 / 0.059294，与 csv 差 ≤ 5e-05），故任务里「若 csv 含激活量化」的分支不触发，未重算基线。
4. **网格 (iii) 不是轻微失配**：逐通道 p99 阈值仅为该通道 max 的中位 0.86 倍（p10 = 0.54，min = 0.000），饱和（|q| = 127/128）的权重占 1.47 %，量化后权重相对 RMS 误差 0.125（(i) 为 1.8e-8–1.0e-2、(ii) 为 2.6e-2–2.9e-2）。p99 用 `torch.quantile(·, 0.99, dim=1)`（linear 插值，与 `numpy.percentile(·, 99)` 实测最大差 2.1e-6），阈值再按 `scale = p99/127` 反量化，未做反向搜索或 AdaRound 式校准。
5. **p99 的损失集中在少数通道**：相对误差最大的模块都是 SE 的 `backbone.s3.b*.se.fc1`（8 维瓶颈），单模块 rel RMS 0.44–0.74；总 rel RMS 0.126–0.138 主要由此贡献，不是全网格均匀变粗。
6. **配对检验**：同一批 400 张图、同一划分，配对差无需 Welch 修正，df = 399；t 区间与 bootstrap 区间（10,000 次，种子 0）结论一致。未做多重比较校正（三种网格是三条独立对照，不是同一族的三个检验）。
7. **未做**：激活量化、再训练、KL / AdaRound 校准、BN 折叠后的网格、4/6 bit。MAE 为 n = 400 子集口径，与 csv 的 `mae_8` 可比，与全验证集读数不可直接比较。
8. **复现**：`C:/Users/24399/.conda/envs/e2e/python.exe scripts/round20_grid.py`（从零复跑前向）或 `... scripts/round20_grid.py --from-json quant_runs/grid_report.json`（只重写本报告，不做前向）。日志 `logs/round20_grid.log`。

