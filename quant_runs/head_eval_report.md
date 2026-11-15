# 三头逐图配对检验（round20）

数据：`data/utkface_raw224.npy`，`split_random(records, val_ratio=0.15, seed=0)[1][:200]`——取验证子集的函数与参数跟 `scripts/deploy_table.py:136` 完全相同（`split_seed_of` 照抄 `scripts/deploy_table.py:44-56`）。
脚本：`scripts/round20_head_eval.py`（本轮唯一新建脚本）；产物 `quant_runs/head_eval/<run>__200.npz`，键 `pred` / `target`（float32 一维）/ `rows`（int64 行号）。

**这 200 张与编译 `.hbm` 时用的那 200 张不是同一批**：后者是 `npu_quant/test_set.json` 的 seed 0 划分 `val[800:1000]`（`npu_quant/out/tables.md:51`），与本轮实测重叠 **0/200**。第 4 节的 int8 ΔMAE 因此是另一套图、另一套口径，两列不可相减、不可混算。

## 1 三个 run 与同批图断言

| head | run 目录 | 检查点来历 | `val_mae_live`（2000 张） | `zero_baseline_mae` | 训练耗时（history `sec` 求和） |
|---|---|---|---|---|---|
| relu_qat | `results/utk-relu_qat-none-s0-1793011336` | 篇 1 原件（2026-10-30 清理中保留） | 0.07160146 | 0.15526728 | 1362.0 s / 20 epoch（22.7 min） |
| gelu | `results/utk-gelu-none-s0-1793526777` | **重训、非篇 1 原件** | 0.07416731 | 0.15526728 | 1792.5 s / 20 epoch（29.9 min） |
| bilinear_glu | `results/utk-bilinear_glu-none-s0-1793526779` | **重训、非篇 1 原件** | 0.07389735 | 0.15526728 | 1791.2 s / 20 epoch（29.9 min） |

两个重训 run 的 `args` 与既有 relu_qat 主 run 只差 `--head`（relu_qat run 早于 `split_seed` / `snap_every` 选项，按 `split_seed_of` 语义 ≡ seed 0，按 `deploy_table.py` 语义 ≡ N = 1）：

- `results/utk-gelu-none-s0-1793526777`：head=gelu ema_mode=none snap_every=1 seed=0 split_seed=0 epochs=20 batch_size=64 lr=0.0001 ema_decay=0.99 amp=True grad_clip=1.0 arch=net quant_mode=overwrite
- `results/utk-bilinear_glu-none-s0-1793526779`：head=bilinear_glu ema_mode=none snap_every=1 seed=0 split_seed=0 epochs=20 batch_size=64 lr=0.0001 ema_decay=0.99 amp=True grad_clip=1.0 arch=net quant_mode=overwrite

同批图断言（脚本内 `np.array_equal` + `assert`，逐元素）：

```
ASSERT OK: 3 个 run 的 target/rows 逐元素相同（n=200，split_seed 口径一致）
rows[:5] = [17967, 4943, 8796, 20916, 18926]
npz: pred float32 (200,), target float32 (200,), rows int64 (200,)
```

## 2 同一批 200 张上的 MAE

| head | MAE（本轮 200 张） | 名次（小 = 好） |
|---|---|---|
| relu_qat | 0.075218 | 1 |
| gelu | 0.076687 | 2 |
| bilinear_glu | 0.076829 | 3 |

## 3 三组配对检验

配对差 d_i = |e_A,i| − |e_B,i|，e = pred − target，**负值 = A 更好**。n = 200、df = 199、t(0.975, 199) = 1.97196；bootstrap 为 10,000 次配对重采样（`np.random.default_rng(0)`，百分位法）。

| 配对 A vs B | 均值差 | sd | se | t 的 95% CI | bootstrap 95% CI | 差为 0 的图 | A 更差的图 | 双侧 p |
|---|---|---|---|---|---|---|---|---|
| relu_qat − gelu | -0.001469 | 0.039997 | 0.002828 | [-0.007046, +0.004108] | [-0.007092, +0.004054] | 0/200 | 95/200 | 0.604 |
| relu_qat − bilinear_glu | -0.001611 | 0.040653 | 0.002875 | [-0.007280, +0.004058] | [-0.007259, +0.003836] | 0/200 | 96/200 | 0.576 |
| gelu − bilinear_glu | -0.000142 | 0.034910 | 0.002468 | [-0.005010, +0.004726] | [-0.005099, +0.004623] | 0/200 | 103/200 | 0.954 |

三组区间**全部含 0**；点估计符号全为负（A 在均值上更好），量级 0.0001–0.0016，而区间半宽约 0.005。逐图预测本身差得很大：mean |pred(relu_qat) − pred(gelu)| = 0.0355、corr = 0.9774 —— MAE 上的小差是逐图正负抵消后的净值，不是三头给出了相同预测。

## 4 余弦名次 / 本轮 MAE 名次 / int8 部署 ΔMAE

| head | `/head/` 逐节点余弦 min（最差节点） | 整网余弦 min | 本轮 200 张 MAE | int8 部署 ΔMAE（**另一套** 200 张） |
|---|---|---|---|---|
| relu_qat | 0.997010（`/head/net/net.1/Pow`） | 0.942175 | 0.075218 | +0.003834 ± 0.001919 |
| gelu | 0.990574（`/head/net/net.0/Gemm`） | 0.947076 | 0.076687 | −0.000157 ± 0.001821 |
| bilinear_glu | 0.983052（`/head/Mul_2`） | 0.932420 | 0.076829 | 无 nash 读数（`npu_quant/out/tables.md:58` 只有 FP32 ONNX 0.076025） |

- 余弦名次（大 = 好）：**relu_qat < gelu < bilinear_glu**（0.997010 / 0.990574 / 0.983052）
- 本轮 MAE 名次（小 = 好）：**relu_qat < gelu < bilinear_glu**（0.075218 / 0.076687 / 0.076829）
- int8 ΔMAE 名次（小 = 好，只有两头有读数）：**gelu < relu_qat**（−0.000157 / +0.003834）

余弦取自 `npu_quant/out/cosine_<tag>.csv`（篇 1 原件的 int8 编译产物，2026-10-27 生成，**不是**本轮重训权重）；整网 min 已排除 −1.0 哨兵行（gelu 的 CSV 有 1 行，`/temporal/layers.0/norm2/LayerNormalization_mean1`）。int8 ΔMAE 取自 `npu_quant/out/tables.md:64-66`（nash 指令级仿真，n = 200 的另一批图），只作并置，不与 MAE 列相减。

## 5 结论

1. **点估计上余弦名次与本轮 MAE 名次一致**：relu_qat 余弦最高（0.997010）且 MAE 最小（0.075218）；bilinear_glu 余弦最低（0.983052）且 MAE 最大（0.076829）；gelu 两者居中。
2. **但这个名次在 n = 200 上不可分辨**：三组配对差的 t 区间与 bootstrap 区间全部含 0，0.0001–0.0016 的均值差被约 ±0.005 的区间宽度盖住（逐图 |pred| 差异均值 0.0355，是 MAE 净差的 20 倍以上）。故只能说「名次一致」，不能说「排序已被证实」。
3. **int8 部署代价的名次与余弦名次反向**：余弦最好的 relu_qat 部署代价最贵（+0.003834），余弦居中的 gelu 最好（−0.000157）；bilinear_glu 没有 nash 读数，三头的 int8 名次不完整。
4. 对评审那句问话：**厂商逐节点余弦排名与任务代价排名不一致**——在 int8 部署代价上现有两头读数与余弦顺序相反；在 FP32 任务 MAE 上两者名次相同，但差异低于本设计的检验分辨率（单 seed、单划分、n = 200）。

## 6 证据与限制

### 6.1 重训声明

gelu 与 bilinear_glu 的篇 1 原件已于 2026-10-30 的清理中删除（`drafts/_paper1_runs_archived.md` 只留摘要）。本报告这两个头的检查点是**本轮重训、非篇 1 原件**：参数与既有 relu_qat N = 1 主 run 完全一致，只换 `--head`。保真度旁证：本轮 gelu 的 `val_mae_live` 0.074167 与篇 1 原件 `utk-gelu-after_snap-s0-1793003829` 的 0.074285 同带（后者是 after_snap，仅作旁证，不作结果）。

### 6.2 训练启动的失败记录

第一次启动（17:39，`results/utk-gelu-none-s0-1793526003` 与 `results/utk-bilinear_glu-none-s0-1793526004`）的两个进程在 epoch 0 完成前被静默杀死（err.log 只有 HF/UserWarning、无 traceback，无 `best.pth` / `result.json`）；两个空目录保留为失败证据，未用于任何数字。第二次启动（17:52）成功。合计 **4 次启动**，正好用完「总计 ≤ 4 次」的预算，此后未再启训练。

### 6.3 受保护文件未改动

对 389 个受保护文件（`train.py`、`quant/*`、`quant_runs/deploy_table_v2.csv`、`deploy_table_v3.csv`、`quant_runs/perimage/*`、`drafts/*.md`、全仓 `*.csv`）在 17:39:30 与 18:25 各做一次 md5 + size + mtime 快照（`logs/round20_md5_before.log` / `logs/round20_md5_after.log`），比对：**changed 0 / missing 0**。

窗口内另有 46 个落在受保护 glob 内的**新文件**出现，均非本会话写入：`quant_runs/deploy_table_v3.csv`（mtime 17:41:13；本会话从未运行 `deploy_table.py`，同窗口有父代理的 deploy_table dump 作业）与 45 个 `quant_runs/perimage/*.npz`（s4–s7 各 run 的 4/6/8bit/fp32 逐图文件）。本会话对 `quant_runs/perimage/` 与两个 deploy_table csv 零写操作。

### 6.4 未完成 / 边界

- **bilinear_glu 无 int8 nash ΔMAE**：`npu_quant/out/tables.md:58` 只有该头的 FP32 ONNX MAE 0.076025，无 nash 读数，故 int8 名次只能对两头做，不补估。
- **单 seed、单划分**：所有区间都是 n = 200 的图内配对，不含 seed 方差与划分方差，不能外推。
- **4 bit 未做**：本轮只做 FP32 任务 MAE。
- **两重训 run 各是一次抽样**：与原件不共享初始化，`val_mae_live` 的 0.0003–0.0026 差属 run 间变异，本报告不把它当 head 效应。
