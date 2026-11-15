# held-out（1,556 张）复评：46 个固定划分 run

生成：`scripts/round20_heldout.py`（2026-11-01 18:55:45，device=cuda）。口径取自 `scripts/heldout_eval.py` 与 `scripts/deploy_table.py`，未重写。

## 0 协议

| 项 | 值 |
|---|---|
| 划分 | `split_random(recs, 0.15, split_seed=0)`，val = 3,556 张 |
| 监控子集 | `Random(0).sample(val, 2000)`（train.py:197） |
| held-out | val 中文件名不在监控 2000 里的 1,556 张；交集 0 |
| first400 | `val[:400]`（heldout_eval.py:76） |
| 量化 | `deploy_table.qdq_`：逐输出通道 absmax/127（8 bit），就地写权重 |
| 评估位宽 | fp32 / 8 / 6 / 4 |
| 检查点 | `results/<run>/best.pth`；权重键 none→`model_state_dict`，其余→`ema_state_dict` |
| 归一化分母 | Z_split：整份验证划分均值作常数预测，held-out 上 MAE = 0.155526 |

零基线两种口径：

| 子集 | n | Z_split（分母，与 train.py 同口径） | Z_self（子集自身均值） |
|---|---|---|---|
| first400 | 400 | 0.161745 | 0.162518 |
| heldout_1556 | 1556 | 0.155526 | 0.155969 |

G1 划分重建：按 sp=0 重算整份验证集零基线 = 0.155267284，与 46 个 `result.json` 的 `zero_baseline_mae` 最大偏差 5.55e-17（阈值 1e-9）→ 通过。

## A 回归校验：与 `quant_runs/heldout_eval.csv` 的 11 个 run 逐格对照

阈值 1e-4；检查 80 格（应为 80），越界 0 格。

| 列 | 最大 |差| |
|---|---|
| mae_4 | 4.40e-07 |
| mae_6 | 4.75e-07 |
| mae_8 | 4.89e-07 |
| mae_fp32 | 4.13e-07 |

| run | mode | N | seed | 子集 | 列 | 本脚本 | heldout_eval.csv | |差| |
|---|---|---|---|---|---|---|---|---|
| `utk-relu_qat-after_snap-s0-1793111850` | after_snap | 0 | 0 | first400 | mae_fp32 | 0.055641 | 0.055641 | 1.7e-07 |
| `utk-relu_qat-after_snap-s0-1793111850` | after_snap | 0 | 0 | first400 | mae_8 | 0.055413 | 0.055413 | 1.3e-07 |
| `utk-relu_qat-after_snap-s0-1793111850` | after_snap | 0 | 0 | heldout_1556 | mae_fp32 | 0.054617 | 0.054617 | 2.2e-07 |
| `utk-relu_qat-after_snap-s0-1793111850` | after_snap | 0 | 0 | heldout_1556 | mae_8 | 0.055092 | 0.055092 | 3.2e-07 |
| `utk-relu_qat-none-s0-1793111850` | none | 0 | 0 | first400 | mae_fp32 | 0.055216 | 0.055216 | 3.5e-07 |
| `utk-relu_qat-none-s0-1793111850` | none | 0 | 0 | first400 | mae_8 | 0.054847 | 0.054847 | 4.2e-07 |
| `utk-relu_qat-none-s0-1793111850` | none | 0 | 0 | heldout_1556 | mae_fp32 | 0.054271 | 0.054271 | 3.2e-07 |
| `utk-relu_qat-none-s0-1793111850` | none | 0 | 0 | heldout_1556 | mae_8 | 0.054421 | 0.054421 | 4.6e-07 |
| `utk-relu_qat-after_snap-s0-1793005531` | after_snap | 1 | 0 | first400 | mae_fp32 | 0.074816 | 0.074816 | 6.3e-08 |
| `utk-relu_qat-after_snap-s0-1793005531` | after_snap | 1 | 0 | first400 | mae_8 | 0.074816 | 0.074816 | 1.0e-07 |
| `utk-relu_qat-after_snap-s0-1793005531` | after_snap | 1 | 0 | first400 | mae_6 | 0.079401 | 0.079401 | 4.0e-07 |
| `utk-relu_qat-after_snap-s0-1793005531` | after_snap | 1 | 0 | first400 | mae_4 | 0.091024 | 0.091024 | 1.5e-07 |
| `utk-relu_qat-after_snap-s0-1793005531` | after_snap | 1 | 0 | heldout_1556 | mae_fp32 | 0.073806 | 0.073806 | 1.4e-07 |
| `utk-relu_qat-after_snap-s0-1793005531` | after_snap | 1 | 0 | heldout_1556 | mae_8 | 0.073806 | 0.073806 | 2.1e-07 |
| `utk-relu_qat-after_snap-s0-1793005531` | after_snap | 1 | 0 | heldout_1556 | mae_6 | 0.078105 | 0.078105 | 3.0e-07 |
| `utk-relu_qat-after_snap-s0-1793005531` | after_snap | 1 | 0 | heldout_1556 | mae_4 | 0.093036 | 0.093036 | 2.8e-07 |
| `utk-relu_qat-before_snap-s0-1793009776` | before_snap | 1 | 0 | first400 | mae_fp32 | 0.074113 | 0.074113 | 3.5e-07 |
| `utk-relu_qat-before_snap-s0-1793009776` | before_snap | 1 | 0 | first400 | mae_8 | 0.074117 | 0.074117 | 8.7e-08 |
| `utk-relu_qat-before_snap-s0-1793009776` | before_snap | 1 | 0 | first400 | mae_6 | 0.077590 | 0.077590 | 3.6e-07 |
| `utk-relu_qat-before_snap-s0-1793009776` | before_snap | 1 | 0 | first400 | mae_4 | 0.089891 | 0.089891 | 3.2e-07 |
| `utk-relu_qat-before_snap-s0-1793009776` | before_snap | 1 | 0 | heldout_1556 | mae_fp32 | 0.073769 | 0.073769 | 3.0e-07 |
| `utk-relu_qat-before_snap-s0-1793009776` | before_snap | 1 | 0 | heldout_1556 | mae_8 | 0.073772 | 0.073772 | 4.0e-07 |
| `utk-relu_qat-before_snap-s0-1793009776` | before_snap | 1 | 0 | heldout_1556 | mae_6 | 0.077419 | 0.077419 | 3.9e-07 |
| `utk-relu_qat-before_snap-s0-1793009776` | before_snap | 1 | 0 | heldout_1556 | mae_4 | 0.092224 | 0.092224 | 3.9e-07 |
| `utk-relu_qat-none-s0-1793011336` | none | 1 | 0 | first400 | mae_fp32 | 0.074067 | 0.074067 | 4.1e-07 |
| `utk-relu_qat-none-s0-1793011336` | none | 1 | 0 | first400 | mae_8 | 0.074067 | 0.074067 | 4.1e-07 |
| `utk-relu_qat-none-s0-1793011336` | none | 1 | 0 | first400 | mae_6 | 0.078510 | 0.078510 | 1.4e-07 |
| `utk-relu_qat-none-s0-1793011336` | none | 1 | 0 | first400 | mae_4 | 0.089988 | 0.089988 | 4.4e-07 |
| `utk-relu_qat-none-s0-1793011336` | none | 1 | 0 | heldout_1556 | mae_fp32 | 0.074110 | 0.074110 | 1.4e-07 |
| `utk-relu_qat-none-s0-1793011336` | none | 1 | 0 | heldout_1556 | mae_8 | 0.074110 | 0.074110 | 1.4e-07 |
| `utk-relu_qat-none-s0-1793011336` | none | 1 | 0 | heldout_1556 | mae_6 | 0.077764 | 0.077764 | 2.0e-07 |
| `utk-relu_qat-none-s0-1793011336` | none | 1 | 0 | heldout_1556 | mae_4 | 0.092764 | 0.092764 | 1.3e-07 |
| `utk-relu_qat-after_snap-s0-1793029658` | after_snap | 5 | 0 | first400 | mae_fp32 | 0.065664 | 0.065664 | 1.8e-07 |
| `utk-relu_qat-after_snap-s0-1793029658` | after_snap | 5 | 0 | first400 | mae_8 | 0.065667 | 0.065667 | 4.4e-07 |
| `utk-relu_qat-after_snap-s0-1793029658` | after_snap | 5 | 0 | first400 | mae_6 | 0.067140 | 0.067140 | 1.0e-07 |
| `utk-relu_qat-after_snap-s0-1793029658` | after_snap | 5 | 0 | first400 | mae_4 | 0.086400 | 0.086400 | 1.3e-07 |
| `utk-relu_qat-after_snap-s0-1793029658` | after_snap | 5 | 0 | heldout_1556 | mae_fp32 | 0.065306 | 0.065306 | 1.8e-07 |
| `utk-relu_qat-after_snap-s0-1793029658` | after_snap | 5 | 0 | heldout_1556 | mae_8 | 0.065308 | 0.065308 | 4.5e-07 |
| `utk-relu_qat-after_snap-s0-1793029658` | after_snap | 5 | 0 | heldout_1556 | mae_6 | 0.068168 | 0.068168 | 1.9e-07 |
| `utk-relu_qat-after_snap-s0-1793029658` | after_snap | 5 | 0 | heldout_1556 | mae_4 | 0.092701 | 0.092701 | 6.9e-08 |
| `utk-relu_qat-before_snap-s0-1793028380` | before_snap | 5 | 0 | first400 | mae_fp32 | 0.065811 | 0.065811 | 1.1e-07 |
| `utk-relu_qat-before_snap-s0-1793028380` | before_snap | 5 | 0 | first400 | mae_8 | 0.065811 | 0.065811 | 7.9e-08 |
| `utk-relu_qat-before_snap-s0-1793028380` | before_snap | 5 | 0 | first400 | mae_6 | 0.067361 | 0.067361 | 9.9e-08 |
| `utk-relu_qat-before_snap-s0-1793028380` | before_snap | 5 | 0 | first400 | mae_4 | 0.088037 | 0.088037 | 2.0e-07 |
| `utk-relu_qat-before_snap-s0-1793028380` | before_snap | 5 | 0 | heldout_1556 | mae_fp32 | 0.065289 | 0.065289 | 7.6e-08 |
| `utk-relu_qat-before_snap-s0-1793028380` | before_snap | 5 | 0 | heldout_1556 | mae_8 | 0.065290 | 0.065290 | 2.5e-07 |
| `utk-relu_qat-before_snap-s0-1793028380` | before_snap | 5 | 0 | heldout_1556 | mae_6 | 0.067888 | 0.067888 | 3.8e-07 |
| `utk-relu_qat-before_snap-s0-1793028380` | before_snap | 5 | 0 | heldout_1556 | mae_4 | 0.094801 | 0.094801 | 2.2e-07 |
| `utk-relu_qat-none-s0-1793027163` | none | 5 | 0 | first400 | mae_fp32 | 0.066022 | 0.066022 | 3.6e-07 |
| `utk-relu_qat-none-s0-1793027163` | none | 5 | 0 | first400 | mae_8 | 0.066196 | 0.066196 | 2.9e-07 |
| `utk-relu_qat-none-s0-1793027163` | none | 5 | 0 | first400 | mae_6 | 0.067170 | 0.067170 | 4.7e-07 |
| `utk-relu_qat-none-s0-1793027163` | none | 5 | 0 | first400 | mae_4 | 0.085716 | 0.085716 | 1.9e-07 |
| `utk-relu_qat-none-s0-1793027163` | none | 5 | 0 | heldout_1556 | mae_fp32 | 0.065249 | 0.065249 | 1.5e-07 |
| `utk-relu_qat-none-s0-1793027163` | none | 5 | 0 | heldout_1556 | mae_8 | 0.065406 | 0.065406 | 1.7e-07 |
| `utk-relu_qat-none-s0-1793027163` | none | 5 | 0 | heldout_1556 | mae_6 | 0.067650 | 0.067650 | 1.4e-07 |
| `utk-relu_qat-none-s0-1793027163` | none | 5 | 0 | heldout_1556 | mae_4 | 0.089798 | 0.089798 | 3.9e-07 |
| `utk-relu_qat-after_snap-s0-1793033217` | after_snap | 20 | 0 | first400 | mae_fp32 | 0.059364 | 0.059364 | 8.6e-08 |
| `utk-relu_qat-after_snap-s0-1793033217` | after_snap | 20 | 0 | first400 | mae_8 | 0.059294 | 0.059294 | 3.7e-07 |
| `utk-relu_qat-after_snap-s0-1793033217` | after_snap | 20 | 0 | first400 | mae_6 | 0.063093 | 0.063093 | 2.5e-07 |
| `utk-relu_qat-after_snap-s0-1793033217` | after_snap | 20 | 0 | first400 | mae_4 | 0.084868 | 0.084868 | 3.4e-07 |
| `utk-relu_qat-after_snap-s0-1793033217` | after_snap | 20 | 0 | heldout_1556 | mae_fp32 | 0.057257 | 0.057257 | 1.9e-07 |
| `utk-relu_qat-after_snap-s0-1793033217` | after_snap | 20 | 0 | heldout_1556 | mae_8 | 0.057215 | 0.057215 | 3.9e-07 |
| `utk-relu_qat-after_snap-s0-1793033217` | after_snap | 20 | 0 | heldout_1556 | mae_6 | 0.060877 | 0.060877 | 2.5e-08 |
| `utk-relu_qat-after_snap-s0-1793033217` | after_snap | 20 | 0 | heldout_1556 | mae_4 | 0.086812 | 0.086812 | 9.9e-08 |
| `utk-relu_qat-before_snap-s0-1793032011` | before_snap | 20 | 0 | first400 | mae_fp32 | 0.060685 | 0.060685 | 3.4e-07 |
| `utk-relu_qat-before_snap-s0-1793032011` | before_snap | 20 | 0 | first400 | mae_8 | 0.060785 | 0.060785 | 1.2e-07 |
| `utk-relu_qat-before_snap-s0-1793032011` | before_snap | 20 | 0 | first400 | mae_6 | 0.064328 | 0.064328 | 2.0e-07 |
| `utk-relu_qat-before_snap-s0-1793032011` | before_snap | 20 | 0 | first400 | mae_4 | 0.086584 | 0.086584 | 1.2e-07 |
| `utk-relu_qat-before_snap-s0-1793032011` | before_snap | 20 | 0 | heldout_1556 | mae_fp32 | 0.057912 | 0.057912 | 2.8e-07 |
| `utk-relu_qat-before_snap-s0-1793032011` | before_snap | 20 | 0 | heldout_1556 | mae_8 | 0.058034 | 0.058034 | 1.0e-07 |
| `utk-relu_qat-before_snap-s0-1793032011` | before_snap | 20 | 0 | heldout_1556 | mae_6 | 0.061033 | 0.061033 | 1.0e-07 |
| `utk-relu_qat-before_snap-s0-1793032011` | before_snap | 20 | 0 | heldout_1556 | mae_4 | 0.087457 | 0.087457 | 1.4e-07 |
| `utk-relu_qat-none-s0-1793030915` | none | 20 | 0 | first400 | mae_fp32 | 0.061906 | 0.061906 | 6.3e-08 |
| `utk-relu_qat-none-s0-1793030915` | none | 20 | 0 | first400 | mae_8 | 0.062007 | 0.062007 | 4.9e-07 |
| `utk-relu_qat-none-s0-1793030915` | none | 20 | 0 | first400 | mae_6 | 0.064212 | 0.064212 | 2.7e-08 |
| `utk-relu_qat-none-s0-1793030915` | none | 20 | 0 | first400 | mae_4 | 0.088789 | 0.088789 | 6.7e-08 |
| `utk-relu_qat-none-s0-1793030915` | none | 20 | 0 | heldout_1556 | mae_fp32 | 0.059096 | 0.059096 | 6.7e-08 |
| `utk-relu_qat-none-s0-1793030915` | none | 20 | 0 | heldout_1556 | mae_8 | 0.059217 | 0.059217 | 2.4e-07 |
| `utk-relu_qat-none-s0-1793030915` | none | 20 | 0 | heldout_1556 | mae_6 | 0.062324 | 0.062324 | 1.5e-07 |
| `utk-relu_qat-none-s0-1793030915` | none | 20 | 0 | heldout_1556 | mae_4 | 0.091195 | 0.091195 | 3.8e-08 |

## B 覆盖与缺口

| 项 | 值 |
|---|---|
| `deploy_table_v3.csv` 中 ema_decay=0.99 且 split_seed=0 的 run | 46 |
| 已评估 | 46/46 |
| 推理失败 | 0 |
| 缺 checkpoint | 0 |
| 未覆盖 run | 无 |

缺口说明：无。46 个 run 的 `best.pth` 全部存在并完成推理，未用 first400 顶替任何 held-out 行。

## C 按 (mode, N) 聚合（held-out 1,556 张）

| mode | N | n | MAE fp32 均值 | sd | min | max | 8 bit 均值 | 6 bit 均值 | 4 bit 均值 | fp32/零基线 |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 8 | 0.054757 | 0.000910 | 0.053433 | 0.056189 | 0.054869 | 0.058233 | 0.135500 | 0.3521 |
| after_snap | 0 | 1 | 0.054617 | — | 0.054617 | 0.054617 | 0.055092 | 0.056948 | 0.109977 | 0.3512 |
| none | 1 | 8 | 0.073213 | 0.001121 | 0.071326 | 0.074825 | 0.073213 | 0.079660 | 0.207403 | 0.4707 |
| before_snap | 1 | 1 | 0.073769 | — | 0.073769 | 0.073769 | 0.073772 | 0.077419 | 0.092224 | 0.4743 |
| after_snap | 1 | 1 | 0.073806 | — | 0.073806 | 0.073806 | 0.073806 | 0.078105 | 0.093036 | 0.4746 |
| none | 5 | 1 | 0.065249 | — | 0.065249 | 0.065249 | 0.065406 | 0.067650 | 0.089798 | 0.4195 |
| before_snap | 5 | 1 | 0.065289 | — | 0.065289 | 0.065289 | 0.065290 | 0.067888 | 0.094801 | 0.4198 |
| after_snap | 5 | 1 | 0.065306 | — | 0.065306 | 0.065306 | 0.065308 | 0.068168 | 0.092701 | 0.4199 |
| none | 20 | 8 | 0.059479 | 0.000989 | 0.057971 | 0.061078 | 0.059501 | 0.064277 | 0.146829 | 0.3824 |
| before_snap | 20 | 8 | 0.058131 | 0.000632 | 0.057288 | 0.058975 | 0.058422 | 0.061315 | 0.144571 | 0.3738 |
| after_snap | 20 | 8 | 0.058102 | 0.000760 | 0.057179 | 0.059004 | 0.058384 | 0.061605 | 0.143621 | 0.3736 |

sd 用 ddof=1；单 run 组记 —。归一化分母 Z_split(held-out) = 0.155526。

### C2 五个主序列（seed 0–7 逐 seed）

**none / N=1**（n=8，8 个 seed 齐全；held-out fp32 均值 0.073213，sd 0.001121）

| seed | run | csv 行 | fp32 | 8 bit | 6 bit | 4 bit |
|---|---|---|---|---|---|---|
| 0 | `utk-relu_qat-none-s0-1793011336` | 11 | 0.074110 | 0.074110 | 0.077764 | 0.092764 |
| 1 | `utk-relu_qat-none-s1-1793445602` | 39 | 0.073696 | 0.073696 | 0.077835 | 0.108503 |
| 2 | `utk-relu_qat-none-s2-1793446593` | 41 | 0.072886 | 0.072886 | 0.080493 | 0.258074 |
| 3 | `utk-relu_qat-none-s3-1793447591` | 43 | 0.072055 | 0.072055 | 0.079215 | 0.164680 |
| 4 | `utk-relu_qat-none-s4-1793448669` | 45 | 0.073206 | 0.073206 | 0.078170 | 0.215143 |
| 5 | `utk-relu_qat-none-s5-1793449730` | 47 | 0.071326 | 0.071326 | 0.080744 | 0.272238 |
| 6 | `utk-relu_qat-none-s6-1793450777` | 49 | 0.074825 | 0.074825 | 0.084391 | 0.285683 |
| 7 | `utk-relu_qat-none-s7-1793451837` | 51 | 0.073602 | 0.073602 | 0.078669 | 0.262139 |

**none / N=20**（n=8，8 个 seed 齐全；held-out fp32 均值 0.059479，sd 0.000989）

| seed | run | csv 行 | fp32 | 8 bit | 6 bit | 4 bit |
|---|---|---|---|---|---|---|
| 0 | `utk-relu_qat-none-s0-1793030915` | 23 | 0.059096 | 0.059217 | 0.062324 | 0.091195 |
| 1 | `utk-relu_qat-none-s1-1793440012` | 81 | 0.058857 | 0.058668 | 0.061114 | 0.104740 |
| 2 | `utk-relu_qat-none-s2-1793440765` | 83 | 0.060533 | 0.061374 | 0.063232 | 0.216843 |
| 3 | `utk-relu_qat-none-s3-1793441598` | 85 | 0.057971 | 0.057905 | 0.060113 | 0.133539 |
| 4 | `utk-relu_qat-none-s4-1793442496` | 87 | 0.061078 | 0.061083 | 0.065701 | 0.183229 |
| 5 | `utk-relu_qat-none-s5-1793443267` | 89 | 0.059018 | 0.059069 | 0.066741 | 0.154033 |
| 6 | `utk-relu_qat-none-s6-1793444050` | 91 | 0.059421 | 0.059599 | 0.067886 | 0.098951 |
| 7 | `utk-relu_qat-none-s7-1793444825` | 93 | 0.059861 | 0.059091 | 0.067105 | 0.192105 |

**none / N=0**（n=8，8 个 seed 齐全；held-out fp32 均值 0.054757，sd 0.000910）

| seed | run | csv 行 | fp32 | 8 bit | 6 bit | 4 bit |
|---|---|---|---|---|---|---|
| 0 | `utk-relu_qat-none-s0-1793111850` | 5 | 0.054271 | 0.054421 | 0.059014 | 0.158194 |
| 1 | `utk-relu_qat-none-s1-1793452884` | 25 | 0.054170 | 0.054756 | 0.057254 | 0.103615 |
| 2 | `utk-relu_qat-none-s2-1793453651` | 27 | 0.056189 | 0.055611 | 0.057427 | 0.197211 |
| 3 | `utk-relu_qat-none-s3-1793454406` | 29 | 0.054296 | 0.054268 | 0.056250 | 0.132640 |
| 4 | `utk-relu_qat-none-s4-1793455163` | 31 | 0.054849 | 0.055085 | 0.058744 | 0.117033 |
| 5 | `utk-relu_qat-none-s5-1793455928` | 33 | 0.053433 | 0.053689 | 0.058913 | 0.169372 |
| 6 | `utk-relu_qat-none-s6-1793456701` | 35 | 0.055031 | 0.055322 | 0.058863 | 0.082503 |
| 7 | `utk-relu_qat-none-s7-1793457541` | 37 | 0.055815 | 0.055801 | 0.059400 | 0.123432 |

**before_snap / N=20**（n=8，8 个 seed 齐全；held-out fp32 均值 0.058131，sd 0.000632）

| seed | run | csv 行 | fp32 | 8 bit | 6 bit | 4 bit |
|---|---|---|---|---|---|---|
| 0 | `utk-relu_qat-before_snap-s0-1793032011` | 21 | 0.057912 | 0.058034 | 0.061033 | 0.087457 |
| 1 | `utk-relu_qat-before_snap-s1-1793461738` | 67 | 0.057401 | 0.057225 | 0.058982 | 0.130682 |
| 2 | `utk-relu_qat-before_snap-s2-1793462579` | 69 | 0.058226 | 0.059188 | 0.060405 | 0.170970 |
| 3 | `utk-relu_qat-before_snap-s3-1793463426` | 71 | 0.057288 | 0.057400 | 0.059542 | 0.140967 |
| 4 | `utk-relu_qat-before_snap-s4-1793464281` | 73 | 0.058754 | 0.058692 | 0.061355 | 0.182134 |
| 5 | `utk-relu_qat-before_snap-s5-1793465147` | 75 | 0.057814 | 0.058546 | 0.061170 | 0.177182 |
| 6 | `utk-relu_qat-before_snap-s6-1793466050` | 77 | 0.058678 | 0.058928 | 0.064470 | 0.114464 |
| 7 | `utk-relu_qat-before_snap-s7-1793466971` | 79 | 0.058975 | 0.059365 | 0.063560 | 0.152712 |

**after_snap / N=20**（n=8，8 个 seed 齐全；held-out fp32 均值 0.058102，sd 0.000760）

| seed | run | csv 行 | fp32 | 8 bit | 6 bit | 4 bit |
|---|---|---|---|---|---|---|
| 0 | `utk-relu_qat-after_snap-s0-1793033217` | 19 | 0.057257 | 0.057215 | 0.060877 | 0.086812 |
| 1 | `utk-relu_qat-after_snap-s1-1793112917` | 53 | 0.057349 | 0.057293 | 0.060019 | 0.092799 |
| 2 | `utk-relu_qat-after_snap-s2-1793113027` | 55 | 0.058691 | 0.059886 | 0.062371 | 0.148500 |
| 3 | `utk-relu_qat-after_snap-s3-1793114152` | 57 | 0.057179 | 0.057208 | 0.060313 | 0.128173 |
| 4 | `utk-relu_qat-after_snap-s4-1793458331` | 59 | 0.059004 | 0.059266 | 0.061510 | 0.166622 |
| 5 | `utk-relu_qat-after_snap-s5-1793459173` | 61 | 0.057948 | 0.058475 | 0.061100 | 0.175548 |
| 6 | `utk-relu_qat-after_snap-s6-1793460031` | 63 | 0.058597 | 0.058869 | 0.063265 | 0.168136 |
| 7 | `utk-relu_qat-after_snap-s7-1793460887` | 65 | 0.058792 | 0.058864 | 0.063385 | 0.182379 |

## D 数字溯源

`quant_runs/heldout_report.json` 的 `rows[*]` 与 CSV 逐行同序，每条带 `csv_line`（表头第 1 行，数据行 i 为第 i+2 行）；聚合表每组的 `runs[*].csv_line_heldout` 指向该 run 的 held-out 行，`main_sequences[*].per_seed[*].csv_line_heldout` 同理。

## E 检查点清单与「选轮是否用了 held-out」

结论：**未使用**。证据链（`train.py`）：

1. `train.py:195-197`：监控子集 = `Random(split_seed).sample(va_ds.records, val_windows=2000)`。
2. `train.py:316-321`：`best_live`/`best_ema` 只由该监控子集上的逐轮 MAE 决定。
3. `train.py:336-338`：`best.pth` 存的是 `best_live_state` / `best_ema_state`。
4. held-out = `val − 监控 2000`（按文件名），与监控子集交集 = 0，故选轮从未见过这 1,556 张。
5. 46 个 run 的 `val_windows` 全为 2000（脚本内断言），故监控集与 held-out 的定义在全部 run 上都成立。

| run | mode | N | seed | checkpoint | 权重键 | n_pred |
|---|---|---|---|---|---|---|
| `utk-relu_qat-after_snap-s0-1793111850` | after_snap | 0 | 0 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-none-s0-1793111850` | none | 0 | 0 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-after_snap-s0-1793005531` | after_snap | 1 | 0 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-before_snap-s0-1793009776` | before_snap | 1 | 0 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-none-s0-1793011336` | none | 1 | 0 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-after_snap-s0-1793029658` | after_snap | 5 | 0 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-before_snap-s0-1793028380` | before_snap | 5 | 0 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-none-s0-1793027163` | none | 5 | 0 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-after_snap-s0-1793033217` | after_snap | 20 | 0 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-before_snap-s0-1793032011` | before_snap | 20 | 0 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-none-s0-1793030915` | none | 20 | 0 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s1-1793452884` | none | 0 | 1 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s2-1793453651` | none | 0 | 2 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s3-1793454406` | none | 0 | 3 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s4-1793455163` | none | 0 | 4 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s5-1793455928` | none | 0 | 5 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s6-1793456701` | none | 0 | 6 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s7-1793457541` | none | 0 | 7 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s1-1793445602` | none | 1 | 1 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s2-1793446593` | none | 1 | 2 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s3-1793447591` | none | 1 | 3 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s4-1793448669` | none | 1 | 4 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s5-1793449730` | none | 1 | 5 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s6-1793450777` | none | 1 | 6 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s7-1793451837` | none | 1 | 7 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-after_snap-s1-1793112917` | after_snap | 20 | 1 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-after_snap-s2-1793113027` | after_snap | 20 | 2 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-after_snap-s3-1793114152` | after_snap | 20 | 3 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-after_snap-s4-1793458331` | after_snap | 20 | 4 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-after_snap-s5-1793459173` | after_snap | 20 | 5 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-after_snap-s6-1793460031` | after_snap | 20 | 6 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-after_snap-s7-1793460887` | after_snap | 20 | 7 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-before_snap-s1-1793461738` | before_snap | 20 | 1 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-before_snap-s2-1793462579` | before_snap | 20 | 2 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-before_snap-s3-1793463426` | before_snap | 20 | 3 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-before_snap-s4-1793464281` | before_snap | 20 | 4 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-before_snap-s5-1793465147` | before_snap | 20 | 5 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-before_snap-s6-1793466050` | before_snap | 20 | 6 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-before_snap-s7-1793466971` | before_snap | 20 | 7 | best.pth | `ema_state_dict` | 1556 |
| `utk-relu_qat-none-s1-1793440012` | none | 20 | 1 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s2-1793440765` | none | 20 | 2 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s3-1793441598` | none | 20 | 3 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s4-1793442496` | none | 20 | 4 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s5-1793443267` | none | 20 | 5 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s6-1793444050` | none | 20 | 6 | best.pth | `model_state_dict` | 1556 |
| `utk-relu_qat-none-s7-1793444825` | none | 20 | 7 | best.pth | `model_state_dict` | 1556 |

## F first400 旁证（与 `deploy_table_v3.csv` 的 4 位小数对照）

注：`deploy_table_v3.csv` 存 4 位小数，其 MAE 列实测取自 `quant_runs/perimage/*.npz`（v3 与 npz 在 46/46 上按 4 位小数相同），而这些 dump 有两种尺寸：n=400（与 first400 同集）与 n=800（更大子集）。n=800 的 run 在 v3 里是 800 张口径，与 400 张的 first400 不可比。本节只作旁证，不作为闸门；闸门是 A 节的 G3（对 n=400 的 heldout_eval.csv，80/80 格）。

| npz 尺寸分组 | run 数 | 格数 | 最大 |差| | 超 1e-4 的格 |
|---|---|---|---|---|
| npz_n_400 | 14 | 56 | 4.78e-05 | 0 |
| npz_n_800 | 32 | 128 | 5.35e-03 | 123 |

## G deviations

- 6 bit / 4 bit are evaluated for all 46 runs, including the two N=inf endpoint runs which heldout_eval.csv left blank (BITS_LIGHT). The blank reference cells are skipped, so G3 is unaffected (80 expected cells over 11 runs).
- Primary normalisation divides by the split-level zero baseline Z_split (constant = mean of the whole 3556-image validation split), matching train.py:199-201; the subset-mean variant Z_self is reported alongside.
- sd uses ddof=1; groups with a single run report sd = null.
- 9 of 46 runs predate --split-seed and their result.json args carry no split_seed key; deploy_table.split_seed_of's existing rule (effective split = seed) applies, and their seed is 0, so the effective split_seed is 0 for every selected run (asserted).
- All 46 runs use val_windows=2000 (asserted), so the monitored subset - and therefore the held-out complement - is identical for every run.
- last.pth is never used: heldout_eval.py hardcodes best.pth, and all 46 runs have it.
- deploy_table_v3.csv cannot be used as a first400 cross-check for most runs: its MAE columns came from quant_runs/perimage/*.npz dumps, 32 of 46 of which hold n=800 predictions (an 800-image MAE), not the 400-image first400. This report keeps every number at n=400 and is gated instead against heldout_eval.csv (n=400, live), which it reproduces to <=4.9e-07 on 80/80 cells. Quantifying this is out of scope; the finding is reported, and no existing file was modified.
- No training, no re-quantisation of stored weights, no change to any existing file.

## H 产物

| 文件 | 字节 | sha256 |
|---|---|---|
| `quant_runs/heldout_eval_v3.csv` | 10702 | `78612e5f897b67cc313cce09c54b85bbae710d2a977cabb72b2b23c523f416c5` |
| `quant_runs/heldout_report.json` | 115315 | `fbfaa4546a5e9885ab15ff7bd26a0f34222c4d1ec856348625dd385fb589534c` |
| `quant_runs/heldout_report.md` | （本文件，自指不写） | — |
