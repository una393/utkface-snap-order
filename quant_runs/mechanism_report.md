# 机制对照：学习率 × 投影间隔、训练预算、随机舍入

只读既有 checkpoint（best.pth）做前向，**不训练、不用 GPU**。全部数字由 python scripts/round20_mechanism.py 一次性重算（51 个 run，CPU 4 进程 × 8 线程，墙钟 6.5 s，含 8 bit 重算与权重距离）。生成时间 2026-11-02 04:03:30，torch 2.12.0+cu132。

**English TL;DR** — (1) 提升学习率确实缩小了 N=1 的投影代价 (8 bit held-out：lr 1e-4 为 +0.018344，3e-3 为 +0.006591; 差中差 -0.012044，区间含 0). (2) 60 轮预算下 N=1 的代价没有关闭：gap 20 轮 +0.018344、60 轮 +0.013892，变化 -0.004743 (区间 < 0). (3) SR@N=1 把最近舍入的代价基本抵消：SR−N=inf = +0.000038，而 RN−N=inf = +0.018344; SR−RN = -0.018306 (区间 < 0).

符号约定：投影代价 = MAE(N=1) − MAE(N=inf)，**正值 = 投影更差**。8 bit = 仅权重 fake quant（deploy_table.qdq_：per-output-channel absmax、对称、RTN），激活保持 FP32。区间为逐 seed 配对的 95% t 区间（df = n−1），p 只作描述；除注明的两处外 n = 3。

## 0 口径、验证格与闸门

主口径 **held-out 1,556**（val 去掉监控 2,000 后的补集；train.py 的选轮只看监控子集，故这些图从未参与选轮）；次口径 **val 前 400**（deploy_table_v4.csv 同源）。两个口径**不是独立重复**：first400 与 held-out 有 168 张重叠、与监控子集有 232 张重叠（监控子集参与选轮，故次口径的绝对值偏乐观，只用来看方向是否一致）。

每个 run 的验证划分按**它自己** result.json 里的 split_seed 重建（本表 31 个 run 实测全为 0；代码不取默认值，deploy_table.split_seed_of 原样复用），并用 result.json 的 zero_baseline_mae 逐个反查。

### 0.1 验证格 results/utk-relu_qat-none-s0-1793111850（N=inf、seed 0）

| 口径 | 本次重算 | heldout_eval_v3.csv | |d| |
|---|---|---|---|
| first400 (400，次口径) FP32 | 0.055216 | 0.055216 | 3.5e-07 |
| first400 (400，次口径) 8 bit | 0.054847 | 0.054847 | 3.8e-07 |
| held-out (1556，主口径) FP32 | 0.054271 | 0.054271 | 3.2e-07 |
| held-out (1556，主口径) 8 bit | 0.054421 | 0.054421 | 4.5e-07 |

四格互不同源，串位会立刻暴露：held-out FP32 = 0.054271、first400 8 bit = 0.054847、first400 FP32 = 0.055216、held-out 8 bit = 0.054421。**注意**：0.054847 是 first400 的 **8 bit** 格、0.054271 是 held-out 的 **FP32** 格、0.055216 是 first400 的 **FP32** 格 —— 若把 0.054847 记作 mae_fp32、0.054271 记作 mae_8、把 0.0552 记作「400 张上的 8 bit」，那是三格串位；按 v3 的四格逐格对齐，max|d| = 4.5e-07。

### 0.2 闸门

| 闸门 | 结果 | 内容 |
|---|---|---|
| G1 | PASS | 划分重建：51 个 run 的 zero_baseline_mae vs 按各自 sp 重建的整份验证集零基线，max|d| = 5.551e-17 |
| G2 | PASS | 口径尺寸：sp=0 val=3556 mon=2000 held=1556 first=400 overlap(mon,held)=0 |
| G5 | PASS | 元数据：51 个 run 的 ema_mode/head/batch/val_windows/decay/seed/epochs/lr/arch/sp 与所在组一致 |
| G5b | PASS | checkpoint 严格对齐：51 个 run 的 state_dict 无缺失/多余键（strict=False 只用于取回计数） |
| G6 | PASS | 机制签名（grid_rms 相对量，N=1 <= 3e-3 / N=inf >= 6e-3；on_grid 有未被投影的 in_proj_weight 造成的地板） |
| G3 | PASS | 验证格 utk-relu_qat-none-s0-1793111850：4 格重算 vs heldout_eval_v3.csv，max|d| = 4.492e-07 |
| G4 | PASS | 回归：24 个 run x 4 格 vs heldout_eval_v3.csv max|d| = 5.085e-07；9 个主序列 run x 2 列 vs deploy_table_v4.csv max|d| = 4.979e-05 |
| G7 | PASS | 运行预算：墙钟 6.5 s（上限 1200 s） |

## 1 表 1：学习率 × 投影间隔（20 轮，none 臂）

### 1.1 单元（held-out 1,556，主口径）

| lr | N | n | FP32 MAE（mean ± sd） | 8 bit MAE（mean ± sd） | FP32 逐 seed | 8 bit 逐 seed |
|---|---|---|---|---|---|---|
| 1e-4 | 1 | 8 | 0.073213 ± 0.0011 | 0.073213 ± 0.0011 | s0 0.074110<br>s1 0.073696<br>s2 0.072886<br>s3 0.072055<br>s4 0.073206<br>s5 0.071326<br>s6 0.074825<br>s7 0.073602 | s0 0.074110<br>s1 0.073696<br>s2 0.072886<br>s3 0.072055<br>s4 0.073206<br>s5 0.071326<br>s6 0.074825<br>s7 0.073602 |
| 1e-4 | 20 | 8 | 0.059479 ± 0.0010 | 0.059501 ± 0.0012 | s0 0.059096<br>s1 0.058857<br>s2 0.060533<br>s3 0.057971<br>s4 0.061078<br>s5 0.059018<br>s6 0.059421<br>s7 0.059861 | s0 0.059217<br>s1 0.058668<br>s2 0.061374<br>s3 0.057905<br>s4 0.061083<br>s5 0.059069<br>s6 0.059599<br>s7 0.059091 |
| 1e-4 | inf (snap_every=0) | 8 | 0.054757 ± 0.0009 | 0.054869 ± 0.0007 | s0 0.054271<br>s1 0.054170<br>s2 0.056189<br>s3 0.054296<br>s4 0.054849<br>s5 0.053433<br>s6 0.055031<br>s7 0.055815 | s0 0.054421<br>s1 0.054756<br>s2 0.055610<br>s3 0.054268<br>s4 0.055085<br>s5 0.053689<br>s6 0.055322<br>s7 0.055801 |
| 1e-3 | 1 | 2 | 0.052951 ± 0.0011 | 0.052951 ± 0.0011 | s0 0.052180<br>s1 0.053721 | s0 0.052180<br>s1 0.053721 |
| 1e-3 | 20 | 1 | 0.051037 ± — | 0.051000 ± — | s0 0.051037 | s0 0.051000 |
| 1e-3 | inf (snap_every=0) | 1 | 0.052130 ± — | 0.052157 ± — | s0 0.052130 | s0 0.052157 |
| 3e-3 | 1 | 3 | 0.070119 ± 0.0037 | 0.070119 ± 0.0037 | s0 0.069866<br>s1 0.066557<br>s2 0.073935 | s0 0.069866<br>s1 0.066557<br>s2 0.073935 |
| 3e-3 | 20 | 3 | 0.061586 ± 0.0002 | 0.061596 ± 0.0002 | s0 0.061567<br>s1 0.061358<br>s2 0.061832 | s0 0.061475<br>s1 0.061433<br>s2 0.061878 |
| 3e-3 | inf (snap_every=0) | 3 | 0.063528 ± 0.0023 | 0.063528 ± 0.0024 | s0 0.065094<br>s1 0.064619<br>s2 0.060871 | s0 0.065044<br>s1 0.064723<br>s2 0.060818 |

run 目录（与上一列同 seed 顺序）：

- lr=1e-4, N=1（n=8）：results/utk-relu_qat-none-s0-1793011336; results/utk-relu_qat-none-s1-1793445602; results/utk-relu_qat-none-s2-1793446593; results/utk-relu_qat-none-s3-1793447591; results/utk-relu_qat-none-s4-1793448669; results/utk-relu_qat-none-s5-1793449730; results/utk-relu_qat-none-s6-1793450777; results/utk-relu_qat-none-s7-1793451837
- lr=1e-4, N=20（n=8）：results/utk-relu_qat-none-s0-1793030915; results/utk-relu_qat-none-s1-1793440012; results/utk-relu_qat-none-s2-1793440765; results/utk-relu_qat-none-s3-1793441598; results/utk-relu_qat-none-s4-1793442496; results/utk-relu_qat-none-s5-1793443267; results/utk-relu_qat-none-s6-1793444050; results/utk-relu_qat-none-s7-1793444825
- lr=1e-4, N=inf (snap_every=0)（n=8）：results/utk-relu_qat-none-s0-1793111850; results/utk-relu_qat-none-s1-1793452884; results/utk-relu_qat-none-s2-1793453651; results/utk-relu_qat-none-s3-1793454406; results/utk-relu_qat-none-s4-1793455163; results/utk-relu_qat-none-s5-1793455928; results/utk-relu_qat-none-s6-1793456701; results/utk-relu_qat-none-s7-1793457541
- lr=1e-3, N=1（n=2）：results_lr/utk-relu_qat-none-s0-1793516710; results_lr/utk-relu_qat-none-s1-1793520772
- lr=1e-3, N=20（n=1）：results_lr/utk-relu_qat-none-s0-1793518513
- lr=1e-3, N=inf (snap_every=0)（n=1）：results_lr/utk-relu_qat-none-s0-1793519563
- lr=3e-3, N=1（n=3）：results_lr/utk-relu_qat-none-s0-1793472498; results_lr/utk-relu_qat-none-s1-1793475705; results_lr/utk-relu_qat-none-s2-1793479561
- lr=3e-3, N=20（n=3）：results_lr/utk-relu_qat-none-s0-1793474018; results_lr/utk-relu_qat-none-s1-1793477195; results_lr/utk-relu_qat-none-s2-1793481326
- lr=3e-3, N=inf (snap_every=0)（n=3）：results_lr/utk-relu_qat-none-s0-1793474884; results_lr/utk-relu_qat-none-s1-1793478537; results_lr/utk-relu_qat-none-s2-1793483431

### 1.2 对比 N=1 − N=inf（逐 seed 配对）

**held-out (1556，主口径)**

| lr | n | A = N=1 的 run / B = N=inf 的 run | 8 bit 逐 seed 差 | 8 bit 均值 [区间] | 判定 |
|---|---|---|---|---|---|
| 1e-4 | 8 | A: results/utk-relu_qat-none-s0-1793011336; results/utk-relu_qat-none-s1-1793445602; results/utk-relu_qat-none-s2-1793446593; results/utk-relu_qat-none-s3-1793447591; results/utk-relu_qat-none-s4-1793448669; results/utk-relu_qat-none-s5-1793449730; results/utk-relu_qat-none-s6-1793450777; results/utk-relu_qat-none-s7-1793451837<br>B: results/utk-relu_qat-none-s0-1793111850; results/utk-relu_qat-none-s1-1793452884; results/utk-relu_qat-none-s2-1793453651; results/utk-relu_qat-none-s3-1793454406; results/utk-relu_qat-none-s4-1793455163; results/utk-relu_qat-none-s5-1793455928; results/utk-relu_qat-none-s6-1793456701; results/utk-relu_qat-none-s7-1793457541 | s0 0.019688<br>s1 0.018941<br>s2 0.017276<br>s3 0.017787<br>s4 0.018121<br>s5 0.017637<br>s6 0.019503<br>s7 0.017801 | +0.018344 [+0.017583, +0.019105] | 区间 > 0（95% CI [+0.017583, +0.019105]（n=8, df=7）） |
| 1e-3 | 1 | A: results_lr/utk-relu_qat-none-s0-1793516710<br>B: results_lr/utk-relu_qat-none-s0-1793519563 | s0 0.000023 | +0.000023 | 不可估（n=1）（区间不可估（n=1）） |
| 3e-3 | 3 | A: results_lr/utk-relu_qat-none-s0-1793472498; results_lr/utk-relu_qat-none-s1-1793475705; results_lr/utk-relu_qat-none-s2-1793479561<br>B: results_lr/utk-relu_qat-none-s0-1793474884; results_lr/utk-relu_qat-none-s1-1793478537; results_lr/utk-relu_qat-none-s2-1793483431 | s0 0.004822<br>s1 0.001834<br>s2 0.013117 | +0.006591 [-0.007931, +0.021112] | 区间含 0（95% CI [-0.007931, +0.021112]（n=3, df=2）） |

| lr（行序同上） | n | FP32 逐 seed 差 | FP32 均值 [区间] | 判定 |
|---|---|---|---|---|
| 1e-4 | 8 | s0 0.019839<br>s1 0.019527<br>s2 0.016698<br>s3 0.017759<br>s4 0.018357<br>s5 0.017894<br>s6 0.019794<br>s7 0.017787 | +0.018457 [+0.017498, +0.019415] | 区间 > 0（95% CI [+0.017498, +0.019415]（n=8, df=7）） |
| 1e-3 | 1 | s0 0.000051 | +0.000051 | 不可估（n=1）（区间不可估（n=1）） |
| 3e-3 | 3 | s0 0.004773<br>s1 0.001938<br>s2 0.013064 | +0.006591 [-0.007771, +0.020954] | 区间含 0（95% CI [-0.007771, +0.020954]（n=3, df=2）） |

**first400 (400，次口径)**

| lr | n | A = N=1 的 run / B = N=inf 的 run | 8 bit 逐 seed 差 | 8 bit 均值 [区间] | 判定 |
|---|---|---|---|---|---|
| 1e-4 | 8 | A: results/utk-relu_qat-none-s0-1793011336; results/utk-relu_qat-none-s1-1793445602; results/utk-relu_qat-none-s2-1793446593; results/utk-relu_qat-none-s3-1793447591; results/utk-relu_qat-none-s4-1793448669; results/utk-relu_qat-none-s5-1793449730; results/utk-relu_qat-none-s6-1793450777; results/utk-relu_qat-none-s7-1793451837<br>B: results/utk-relu_qat-none-s0-1793111850; results/utk-relu_qat-none-s1-1793452884; results/utk-relu_qat-none-s2-1793453651; results/utk-relu_qat-none-s3-1793454406; results/utk-relu_qat-none-s4-1793455163; results/utk-relu_qat-none-s5-1793455928; results/utk-relu_qat-none-s6-1793456701; results/utk-relu_qat-none-s7-1793457541 | s0 0.019221<br>s1 0.022037<br>s2 0.012767<br>s3 0.014617<br>s4 0.017716<br>s5 0.016599<br>s6 0.021455<br>s7 0.017559 | +0.017746 [+0.015097, +0.020395] | 区间 > 0（95% CI [+0.015097, +0.020395]（n=8, df=7）） |
| 1e-3 | 1 | A: results_lr/utk-relu_qat-none-s0-1793516710<br>B: results_lr/utk-relu_qat-none-s0-1793519563 | s0 0.001998 | +0.001998 | 不可估（n=1）（区间不可估（n=1）） |
| 3e-3 | 3 | A: results_lr/utk-relu_qat-none-s0-1793472498; results_lr/utk-relu_qat-none-s1-1793475705; results_lr/utk-relu_qat-none-s2-1793479561<br>B: results_lr/utk-relu_qat-none-s0-1793474884; results_lr/utk-relu_qat-none-s1-1793478537; results_lr/utk-relu_qat-none-s2-1793483431 | s0 0.006563<br>s1 -0.000047<br>s2 0.013641 | +0.006719 [-0.010286, +0.023724] | 区间含 0（95% CI [-0.010286, +0.023724]（n=3, df=2）） |

| lr（行序同上） | n | FP32 逐 seed 差 | FP32 均值 [区间] | 判定 |
|---|---|---|---|---|
| 1e-4 | 8 | s0 0.018852<br>s1 0.022457<br>s2 0.012326<br>s3 0.014423<br>s4 0.017953<br>s5 0.016891<br>s6 0.021328<br>s7 0.017951 | +0.017773 [+0.014992, +0.020554] | 区间 > 0（95% CI [+0.014992, +0.020554]（n=8, df=7）） |
| 1e-3 | 1 | s0 0.001882 | +0.001882 | 不可估（n=1）（区间不可估（n=1）） |
| 3e-3 | 3 | s0 0.006465<br>s1 -0.000134<br>s2 0.013538 | +0.006623 [-0.010363, +0.023608] | 区间含 0（95% CI [-0.010363, +0.023608]（n=3, df=2）） |

### 1.3 学习率改变了多少（差中差：Δ(lr) − Δ(1e-4)）

| 学习率对比 | 精度 | 口径 | n | 逐 seed 差中差 | 均值 [95% CI] | 判定 |
|---|---|---|---|---|---|---|
| 1e-4 → 1e-3 | 8 bit | first400 | 1 | s0 -0.017223 | -0.017223 | 不可估（n=1） |
| 1e-4 → 3e-3 | 8 bit | first400 | 3 | s0 -0.012658<br>s1 -0.022084<br>s2 0.000874 | -0.011289 [-0.039957, +0.017378] | 区间含 0 |
| 1e-4 → 1e-3 | FP32 | first400 | 1 | s0 -0.016969 | -0.016969 | 不可估（n=1） |
| 1e-4 → 3e-3 | FP32 | first400 | 3 | s0 -0.012387<br>s1 -0.022591<br>s2 0.001212 | -0.011255 [-0.040920, +0.018409] | 区间含 0 |
| 1e-4 → 1e-3 | 8 bit | held-out 1556 | 1 | s0 -0.019665 | -0.019665 | 不可估（n=1） |
| 1e-4 → 3e-3 | 8 bit | held-out 1556 | 3 | s0 -0.014866<br>s1 -0.017107<br>s2 -0.004159 | -0.012044 [-0.029234, +0.005146] | 区间含 0 |
| 1e-4 → 1e-3 | FP32 | held-out 1556 | 1 | s0 -0.019788 | -0.019788 | 不可估（n=1） |
| 1e-4 → 3e-3 | FP32 | held-out 1556 | 3 | s0 -0.015067<br>s1 -0.017589<br>s2 -0.003634 | -0.012096 [-0.030570, +0.006377] | 区间含 0 |

差中差把「同一 seed、同一划分」的两个学习率放在一起，抵消了划分与初始化的共同部分；它仍只是 n ≤ 3 的描述统计。

## 2 表 2：训练预算（20 轮 vs 60 轮，lr=1e-4，none 臂）

60 轮组的 run 与 20 轮参照只差 epochs（60 vs 20）与输出目录：学习率、batch、EMA 衰减、划分 seed 全同（G5 逐键核过）。

### 2.1 单元（held-out 1,556）

| 预算 | N | n | FP32 MAE | 8 bit MAE | FP32 逐 seed | 8 bit 逐 seed |
|---|---|---|---|---|---|---|
| 20ep | 1 | 8 | 0.073213 ± 0.0011 | 0.073213 ± 0.0011 | s0 0.074110<br>s1 0.073696<br>s2 0.072886<br>s3 0.072055<br>s4 0.073206<br>s5 0.071326<br>s6 0.074825<br>s7 0.073602 | s0 0.074110<br>s1 0.073696<br>s2 0.072886<br>s3 0.072055<br>s4 0.073206<br>s5 0.071326<br>s6 0.074825<br>s7 0.073602 |
| 20ep | inf (snap_every=0) | 8 | 0.054757 ± 0.0009 | 0.054869 ± 0.0007 | s0 0.054271<br>s1 0.054170<br>s2 0.056189<br>s3 0.054296<br>s4 0.054849<br>s5 0.053433<br>s6 0.055031<br>s7 0.055815 | s0 0.054421<br>s1 0.054756<br>s2 0.055610<br>s3 0.054268<br>s4 0.055085<br>s5 0.053689<br>s6 0.055322<br>s7 0.055801 |
| 60ep | 1 | 3 | 0.065582 ± 0.0006 | 0.065582 ± 0.0006 | s0 0.064921<br>s1 0.066075<br>s2 0.065750 | s0 0.064921<br>s1 0.066075<br>s2 0.065750 |
| 60ep | inf (snap_every=0) | 3 | 0.051576 ± 0.0007 | 0.051690 ± 0.0007 | s0 0.051455<br>s1 0.050952<br>s2 0.052321 | s0 0.051456<br>s1 0.051188<br>s2 0.052426 |

- 20ep, N=1：results/utk-relu_qat-none-s0-1793011336; results/utk-relu_qat-none-s1-1793445602; results/utk-relu_qat-none-s2-1793446593; results/utk-relu_qat-none-s3-1793447591; results/utk-relu_qat-none-s4-1793448669; results/utk-relu_qat-none-s5-1793449730; results/utk-relu_qat-none-s6-1793450777; results/utk-relu_qat-none-s7-1793451837
- 20ep, N=inf (snap_every=0)：results/utk-relu_qat-none-s0-1793111850; results/utk-relu_qat-none-s1-1793452884; results/utk-relu_qat-none-s2-1793453651; results/utk-relu_qat-none-s3-1793454406; results/utk-relu_qat-none-s4-1793455163; results/utk-relu_qat-none-s5-1793455928; results/utk-relu_qat-none-s6-1793456701; results/utk-relu_qat-none-s7-1793457541
- 60ep, N=1：results_long/utk-relu_qat-none-s0-1793485675; results_long/utk-relu_qat-none-s1-1793492024; results_long/utk-relu_qat-none-s2-1793499457
- 60ep, N=inf (snap_every=0)：results_long/utk-relu_qat-none-s0-1793489886; results_long/utk-relu_qat-none-s1-1793496960; results_long/utk-relu_qat-none-s2-1793505234

### 2.2 每个预算下的 gap（N=1 − N=inf）与 gap 的变化

**first400 (400，次口径)**

| 预算 | 精度 | n | 逐 seed gap | 均值 [95% CI] | 判定 |
|---|---|---|---|---|---|
| 20ep | 8 bit | 8 | s0 0.019221<br>s1 0.022037<br>s2 0.012767<br>s3 0.014617<br>s4 0.017716<br>s5 0.016599<br>s6 0.021455<br>s7 0.017559 | +0.017746 [+0.015097, +0.020395] | 区间 > 0 |
| 20ep | FP32 | 8 | s0 0.018852<br>s1 0.022457<br>s2 0.012326<br>s3 0.014423<br>s4 0.017953<br>s5 0.016891<br>s6 0.021328<br>s7 0.017951 | +0.017773 [+0.014992, +0.020554] | 区间 > 0 |
| 60ep | 8 bit | 3 | s0 0.014398<br>s1 0.016298<br>s2 0.010046 | +0.013581 [+0.005620, +0.021542] | 区间 > 0 |
| 60ep | FP32 | 3 | s0 0.013581<br>s1 0.016375<br>s2 0.009903 | +0.013286 [+0.005223, +0.021350] | 区间 > 0 |
| **60 − 20** | 8 bit | 3 | s0 -0.004823<br>s1 -0.005739<br>s2 -0.002720 | -0.004428 [-0.008273, -0.000582] | 区间 < 0 |
| **60 − 20** | FP32 | 3 | s0 -0.005271<br>s1 -0.006082<br>s2 -0.002423 | -0.004592 [-0.009365, +0.000182] | 区间含 0 |

**held-out (1556，主口径)**

| 预算 | 精度 | n | 逐 seed gap | 均值 [95% CI] | 判定 |
|---|---|---|---|---|---|
| 20ep | 8 bit | 8 | s0 0.019688<br>s1 0.018941<br>s2 0.017276<br>s3 0.017787<br>s4 0.018121<br>s5 0.017637<br>s6 0.019503<br>s7 0.017801 | +0.018344 [+0.017583, +0.019105] | 区间 > 0 |
| 20ep | FP32 | 8 | s0 0.019839<br>s1 0.019527<br>s2 0.016698<br>s3 0.017759<br>s4 0.018357<br>s5 0.017894<br>s6 0.019794<br>s7 0.017787 | +0.018457 [+0.017498, +0.019415] | 区间 > 0 |
| 60ep | 8 bit | 3 | s0 0.013465<br>s1 0.014887<br>s2 0.013325 | +0.013892 [+0.011744, +0.016040] | 区间 > 0 |
| 60ep | FP32 | 3 | s0 0.013466<br>s1 0.015123<br>s2 0.013429 | +0.014006 [+0.011602, +0.016410] | 区间 > 0 |
| **60 − 20** | 8 bit | 3 | s0 -0.006224<br>s1 -0.004054<br>s2 -0.003951 | -0.004743 [-0.007931, -0.001554] | 区间 < 0 |
| **60 − 20** | FP32 | 3 | s0 -0.006373<br>s1 -0.004404<br>s2 -0.003269 | -0.004682 [-0.008583, -0.000780] | 区间 < 0 |

### 2.3 逐 seed 一览（8 bit，held-out）

| seed | gap 20 轮 | gap 60 轮 | 变化 |
|---|---|---|---|
| s0 | +0.019688 | +0.013465 | -0.006224 |
| s1 | +0.018941 | +0.014887 | -0.004054 |
| s2 | +0.017276 | +0.013325 | -0.003951 |

## 3 表 3：随机舍入（SR）@N=1 vs 最近舍入（RN）@N=1 vs N=inf

SR 臂由 scripts/run_stochastic_rounding.py 把 train.snap_weights_ 换成 sign(x) * floor(|x| + U)（同一 per-channel 网格）后训练，结果落在 results_sr/。**provenance 限制**：result.json 的 args 与普通 none/N=1 run 完全相同（没有记录 SR 的开关），SR 身份只由目录与脚本给出；网格残差也区分不了两者（SR 同样落在网格上，见 §4）。同一 seed 的 SR 与 RN 是两次独立训练（同 seed、同划分、同配方，只差舍入规则）。

证据行：

    scripts/run_stochastic_rounding.py:7: 落盘目录 results_sr/：刻意不落 results/，否则 deploy_table.py 的 results/utk-relu_qat-*
    scripts/run_stochastic_rounding.py:22: def sr_snap_weights_(model: nn.Module, bits: int = 8) -> float:
    scripts/run_stochastic_rounding.py:35: q = (torch.sign(x) * torch.floor(x.abs() + r)).clamp(-qmax - 1, qmax) * scale
    scripts/run_stochastic_rounding.py:47: ap.add_argument("--out", default="results_sr")
    scripts/run_stochastic_rounding.py:51: T.snap_weights_ = sr_snap_weights_          # 只改本进程的绑定，train.py 文件本身不动

### 3.1 单元（held-out 1,556）

| 臂 | n | FP32 MAE | 8 bit MAE | FP32 逐 seed | 8 bit 逐 seed |
|---|---|---|---|---|---|
| RN@1 | 8 | 0.073213 ± 0.0011 | 0.073213 ± 0.0011 | s0 0.074110<br>s1 0.073696<br>s2 0.072886<br>s3 0.072055<br>s4 0.073206<br>s5 0.071326<br>s6 0.074825<br>s7 0.073602 | s0 0.074110<br>s1 0.073696<br>s2 0.072886<br>s3 0.072055<br>s4 0.073206<br>s5 0.071326<br>s6 0.074825<br>s7 0.073602 |
| SR@1 | 8 | 0.054908 ± 0.0009 | 0.054908 ± 0.0009 | s0 0.055376<br>s1 0.054516<br>s2 0.056143<br>s3 0.055266<br>s4 0.055009<br>s5 0.053154<br>s6 0.054175<br>s7 0.055620 | s0 0.055376<br>s1 0.054516<br>s2 0.056143<br>s3 0.055266<br>s4 0.055009<br>s5 0.053154<br>s6 0.054175<br>s7 0.055620 |
| inf | 8 | 0.054757 ± 0.0009 | 0.054869 ± 0.0007 | s0 0.054271<br>s1 0.054170<br>s2 0.056189<br>s3 0.054296<br>s4 0.054849<br>s5 0.053433<br>s6 0.055031<br>s7 0.055815 | s0 0.054421<br>s1 0.054756<br>s2 0.055610<br>s3 0.054268<br>s4 0.055085<br>s5 0.053689<br>s6 0.055322<br>s7 0.055801 |

- RN@1：results/utk-relu_qat-none-s0-1793011336; results/utk-relu_qat-none-s1-1793445602; results/utk-relu_qat-none-s2-1793446593; results/utk-relu_qat-none-s3-1793447591; results/utk-relu_qat-none-s4-1793448669; results/utk-relu_qat-none-s5-1793449730; results/utk-relu_qat-none-s6-1793450777; results/utk-relu_qat-none-s7-1793451837
- SR@1：results_sr/utk-relu_qat-none-s0-1793510216; results_sr/utk-relu_qat-none-s1-1793512950; results_sr/utk-relu_qat-none-s2-1793514777; results_sr/utk-relu_qat-none-s3-1793554855; results_sr/utk-relu_qat-none-s4-1793557139; results_sr/utk-relu_qat-none-s5-1793558720; results_sr/utk-relu_qat-none-s6-1793560125; results_sr/utk-relu_qat-none-s7-1793561553
- inf：results/utk-relu_qat-none-s0-1793111850; results/utk-relu_qat-none-s1-1793452884; results/utk-relu_qat-none-s2-1793453651; results/utk-relu_qat-none-s3-1793454406; results/utk-relu_qat-none-s4-1793455163; results/utk-relu_qat-none-s5-1793455928; results/utk-relu_qat-none-s6-1793456701; results/utk-relu_qat-none-s7-1793457541

### 3.2 三个对比

**first400 (400，次口径)**

| 对比 | 精度 | n | 逐 seed 差 | 均值 [95% CI] | 判定 |
|---|---|---|---|---|---|
| RN-inf | 8 bit | 8 | s0 0.019221<br>s1 0.022037<br>s2 0.012767<br>s3 0.014617<br>s4 0.017716<br>s5 0.016599<br>s6 0.021455<br>s7 0.017559 | +0.017746 [+0.015097, +0.020395] | 区间 > 0（95% CI [+0.015097, +0.020395]（n=8, df=7）） |
| RN-inf | FP32 | 8 | s0 0.018852<br>s1 0.022457<br>s2 0.012326<br>s3 0.014423<br>s4 0.017953<br>s5 0.016891<br>s6 0.021328<br>s7 0.017951 | +0.017773 [+0.014992, +0.020554] | 区间 > 0（95% CI [+0.014992, +0.020554]（n=8, df=7）） |
| SR-inf | 8 bit | 8 | s0 0.002251<br>s1 0.003274<br>s2 -0.001368<br>s3 -0.001797<br>s4 -0.001237<br>s5 -0.000617<br>s6 0.000806<br>s7 0.001435 | +0.000343 [-0.001223, +0.001909] | 区间含 0（95% CI [-0.001223, +0.001909]（n=8, df=7）） |
| SR-inf | FP32 | 8 | s0 0.001881<br>s1 0.003693<br>s2 -0.001808<br>s3 -0.001991<br>s4 -0.001000<br>s5 -0.000324<br>s6 0.000679<br>s7 0.001828 | +0.000370 [-0.001309, +0.002049] | 区间含 0（95% CI [-0.001309, +0.002049]（n=8, df=7）） |
| SR-RN | 8 bit | 8 | s0 -0.016970<br>s1 -0.018763<br>s2 -0.014135<br>s3 -0.016415<br>s4 -0.018953<br>s5 -0.017216<br>s6 -0.020649<br>s7 -0.016123 | -0.017403 [-0.019084, -0.015722] | 区间 < 0（95% CI [-0.019084, -0.015722]（n=8, df=7）） |
| SR-RN | FP32 | 8 | s0 -0.016970<br>s1 -0.018763<br>s2 -0.014135<br>s3 -0.016415<br>s4 -0.018953<br>s5 -0.017216<br>s6 -0.020649<br>s7 -0.016123 | -0.017403 [-0.019084, -0.015722] | 区间 < 0（95% CI [-0.019084, -0.015722]（n=8, df=7）） |

**held-out (1556，主口径)**

| 对比 | 精度 | n | 逐 seed 差 | 均值 [95% CI] | 判定 |
|---|---|---|---|---|---|
| RN-inf | 8 bit | 8 | s0 0.019688<br>s1 0.018941<br>s2 0.017276<br>s3 0.017787<br>s4 0.018121<br>s5 0.017637<br>s6 0.019503<br>s7 0.017801 | +0.018344 [+0.017583, +0.019105] | 区间 > 0（95% CI [+0.017583, +0.019105]（n=8, df=7）） |
| RN-inf | FP32 | 8 | s0 0.019839<br>s1 0.019527<br>s2 0.016698<br>s3 0.017759<br>s4 0.018357<br>s5 0.017894<br>s6 0.019794<br>s7 0.017787 | +0.018457 [+0.017498, +0.019415] | 区间 > 0（95% CI [+0.017498, +0.019415]（n=8, df=7）） |
| SR-inf | 8 bit | 8 | s0 0.000955<br>s1 -0.000240<br>s2 0.000533<br>s3 0.000998<br>s4 -0.000076<br>s5 -0.000535<br>s6 -0.001147<br>s7 -0.000181 | +0.000038 [-0.000584, +0.000661] | 区间含 0（95% CI [-0.000584, +0.000661]（n=8, df=7）） |
| SR-inf | FP32 | 8 | s0 0.001105<br>s1 0.000346<br>s2 -0.000045<br>s3 0.000970<br>s4 0.000161<br>s5 -0.000279<br>s6 -0.000856<br>s7 -0.000195 | +0.000151 [-0.000394, +0.000696] | 区间含 0（95% CI [-0.000394, +0.000696]（n=8, df=7）） |
| SR-RN | 8 bit | 8 | s0 -0.018734<br>s1 -0.019180<br>s2 -0.016743<br>s3 -0.016789<br>s4 -0.018196<br>s5 -0.018173<br>s6 -0.020650<br>s7 -0.017981 | -0.018306 [-0.019369, -0.017243] | 区间 < 0（95% CI [-0.019369, -0.017243]（n=8, df=7）） |
| SR-RN | FP32 | 8 | s0 -0.018734<br>s1 -0.019180<br>s2 -0.016743<br>s3 -0.016789<br>s4 -0.018196<br>s5 -0.018173<br>s6 -0.020650<br>s7 -0.017981 | -0.018306 [-0.019369, -0.017243] | 区间 < 0（95% CI [-0.019369, -0.017243]（n=8, df=7）） |

### 3.3 逐 seed 恢复比例 (RN − SR) / (RN − inf)

| 精度 | 逐 seed 恢复比例 | 均值 |
|---|---|---|
| 8 bit | s0 95%, s1 101%, s2 97%, s3 94%, s4 100%, s5 103%, s6 106%, s7 101% | 100% |
| FP32 | s0 94%, s1 98%, s2 100%, s3 95%, s4 99%, s5 102%, s6 104%, s7 101% | 99% |

100% = SR 完全抵消 N=1 的代价；0% = SR 与 RN 一样。

## 4 机制读数

### 4.1 网格残差（8 bit 网格，按 weight-tensor 的 numel 加权）

on_grid = |q − round(q)| < 0.05 格距的权重占比；grid_rms = rms(w − q_hat)/rms(w)。判据同 scripts/paper2_diagnostics.py:35-44，此处直接在 checkpoint 上算，未改既有脚本。

| run 目录 | N | on_grid | grid_rms |
|---|---|---|---|
| results_long/utk-relu_qat-none-s0-1793489886 | inf (snap_every=0) | 0.1082 | 9.7e-03 |
| results_long/utk-relu_qat-none-s1-1793496960 | inf (snap_every=0) | 0.1077 | 9.7e-03 |
| results_long/utk-relu_qat-none-s2-1793505234 | inf (snap_every=0) | 0.1081 | 9.7e-03 |
| results_long/utk-relu_qat-none-s0-1793485675 | 1 | 0.9195 | 1.4e-03 |
| results_long/utk-relu_qat-none-s1-1793492024 | 1 | 0.9194 | 1.4e-03 |
| results_long/utk-relu_qat-none-s2-1793499457 | 1 | 0.9195 | 1.4e-03 |
| results_lr/utk-relu_qat-none-s0-1793519563 | inf (snap_every=0) | 0.1079 | 8.8e-03 |
| results_lr/utk-relu_qat-none-s0-1793516710 | 1 | 0.9195 | 1.8e-03 |
| results_lr/utk-relu_qat-none-s1-1793520772 | 1 | 0.9195 | 2.6e-03 |
| results_lr/utk-relu_qat-none-s0-1793518513 | 20 | 0.6653 | 2.7e-03 |
| results_lr/utk-relu_qat-none-s0-1793474884 | inf (snap_every=0) | 0.1081 | 7.4e-03 |
| results_lr/utk-relu_qat-none-s1-1793478537 | inf (snap_every=0) | 0.1078 | 6.9e-03 |
| results_lr/utk-relu_qat-none-s2-1793483431 | inf (snap_every=0) | 0.1082 | 7.7e-03 |
| results_lr/utk-relu_qat-none-s0-1793472498 | 1 | 0.9194 | 2.8e-03 |
| results_lr/utk-relu_qat-none-s1-1793475705 | 1 | 0.9195 | 2.4e-03 |
| results_lr/utk-relu_qat-none-s2-1793479561 | 1 | 0.9195 | 2.6e-03 |
| results_lr/utk-relu_qat-none-s0-1793474018 | 20 | 0.7148 | 2.7e-03 |
| results_lr/utk-relu_qat-none-s1-1793477195 | 20 | 0.2844 | 6.2e-03 |
| results_lr/utk-relu_qat-none-s2-1793481326 | 20 | 0.8231 | 2.5e-03 |
| results/utk-relu_qat-none-s0-1793111850 | inf (snap_every=0) | 0.1080 | 9.8e-03 |
| results/utk-relu_qat-none-s1-1793452884 | inf (snap_every=0) | 0.1078 | 9.8e-03 |
| results/utk-relu_qat-none-s2-1793453651 | inf (snap_every=0) | 0.1081 | 9.8e-03 |
| results/utk-relu_qat-none-s3-1793454406 | inf (snap_every=0) | 0.1080 | 9.7e-03 |
| results/utk-relu_qat-none-s4-1793455163 | inf (snap_every=0) | 0.1081 | 9.8e-03 |
| results/utk-relu_qat-none-s5-1793455928 | inf (snap_every=0) | 0.1080 | 9.8e-03 |
| results/utk-relu_qat-none-s6-1793456701 | inf (snap_every=0) | 0.1081 | 9.8e-03 |
| results/utk-relu_qat-none-s7-1793457541 | inf (snap_every=0) | 0.1080 | 9.8e-03 |
| results/utk-relu_qat-none-s0-1793011336 | 1 | 0.9194 | 1.3e-03 |
| results/utk-relu_qat-none-s1-1793445602 | 1 | 0.9195 | 1.3e-03 |
| results/utk-relu_qat-none-s2-1793446593 | 1 | 0.9194 | 1.3e-03 |
| results/utk-relu_qat-none-s3-1793447591 | 1 | 0.9194 | 1.3e-03 |
| results/utk-relu_qat-none-s4-1793448669 | 1 | 0.9194 | 1.3e-03 |
| results/utk-relu_qat-none-s5-1793449730 | 1 | 0.9195 | 1.3e-03 |
| results/utk-relu_qat-none-s6-1793450777 | 1 | 0.9195 | 1.3e-03 |
| results/utk-relu_qat-none-s7-1793451837 | 1 | 0.9195 | 1.3e-03 |
| results/utk-relu_qat-none-s0-1793030915 | 20 | 0.9194 | 1.3e-03 |
| results/utk-relu_qat-none-s1-1793440012 | 20 | 0.9091 | 1.3e-03 |
| results/utk-relu_qat-none-s2-1793440765 | 20 | 0.3615 | 3.4e-03 |
| results/utk-relu_qat-none-s3-1793441598 | 20 | 0.2989 | 4.1e-03 |
| results/utk-relu_qat-none-s4-1793442496 | 20 | 0.9190 | 1.3e-03 |
| results/utk-relu_qat-none-s5-1793443267 | 20 | 0.9194 | 1.3e-03 |
| results/utk-relu_qat-none-s6-1793444050 | 20 | 0.9081 | 1.3e-03 |
| results/utk-relu_qat-none-s7-1793444825 | 20 | 0.3510 | 3.3e-03 |
| results_sr/utk-relu_qat-none-s0-1793510216 | 1 | 0.9195 | 1.3e-03 |
| results_sr/utk-relu_qat-none-s1-1793512950 | 1 | 0.9195 | 1.2e-03 |
| results_sr/utk-relu_qat-none-s2-1793514777 | 1 | 0.9194 | 1.2e-03 |
| results_sr/utk-relu_qat-none-s3-1793554855 | 1 | 0.9194 | 1.3e-03 |
| results_sr/utk-relu_qat-none-s4-1793557139 | 1 | 0.9195 | 1.2e-03 |
| results_sr/utk-relu_qat-none-s5-1793558720 | 1 | 0.9195 | 1.2e-03 |
| results_sr/utk-relu_qat-none-s6-1793560125 | 1 | 0.9194 | 1.2e-03 |
| results_sr/utk-relu_qat-none-s7-1793561553 | 1 | 0.9194 | 1.3e-03 |

grid_rms（相对量）逐组：N=1 为 1.2e-03–2.8e-03，N=20 为 1.3e-03–6.2e-03，N=inf 为 6.9e-03–9.8e-03 —— N=inf 是 N=1 的 2.5–7.9 倍。所以「投影是否发生」可以用网格残差判定（G6 就是这条）；但「SR vs RN」不能：两者都是舍入算子，N=1 与 SR@N=1 的 grid_rms 同量级（1.2e-03 vs 2.8e-03）。

### 4.2 权重空间距离（相对 L2）与训练期 live MAE

| seed | SR vs RN | RN vs inf | SR vs inf | SR live MAE | RN live MAE | inf live MAE |
|---|---|---|---|---|---|---|
| s0 | 0.2845 | 0.2705 | 0.1697 | 0.053691 | 0.071601 | 0.053188 |
| s1 | 0.2941 | 0.2808 | 0.1719 | 0.052884 | 0.074243 | 0.052988 |
| s2 | 0.2958 | 0.2865 | 0.1654 | 0.054974 | 0.071651 | 0.055615 |
| s3 | 0.2922 | 0.2780 | 0.1664 | 0.054023 | 0.071176 | 0.053728 |
| s4 | 0.2893 | 0.2818 | 0.1646 | 0.053333 | 0.073864 | 0.053149 |
| s5 | 0.2957 | 0.2872 | 0.1629 | 0.052068 | 0.071891 | 0.052881 |
| s6 | 0.2850 | 0.2787 | 0.1530 | 0.052590 | 0.075820 | 0.052493 |
| s7 | 0.2924 | 0.2807 | 0.1644 | 0.054574 | 0.072190 | 0.053560 |

live MAE 来自各 result.json 的 val_mae_live，口径是**监控 2,000 张**（选轮用的那一份），与本文的 400 / 1,556 两个口径都不可直接比；列在这里只作为旁证：SR 臂的训练期读数本身就与 RN 臂差一个量级、而与不投影接近。

## 5 三个问题的一句话回答

**Q1（提高学习率是否缩小 N=1 的投影代价）** 学习率提高会让 N=1 的投影代价变小：held-out 1556 张、8 bit 上，N=1−N=inf 从 lr=1e-4 的 +0.018344 [+0.017583, +0.019105] 降到 lr=3e-3 的 +0.006591 [-0.007931, +0.021112]，逐 seed 差中差为 -0.012044 [-0.029234, +0.005146]（95% CI [-0.029234, +0.005146]（n=3, df=2），判定：区间含 0）；lr=1e-3 一档的 N=inf 只有 seed 0、N=1 有 seed 0/1，该档对比 n=1、区间不可估，因此只能说「把学习率放大 30 倍后代价降到接近 0」，不能说已证明代价随 lr 单调。

**Q2（预算 20 → 60 轮：gap 关闭、缩小还是持续）** 把预算从 20 轮加到 60 轮没有关闭 N=1 的代价：8 bit held-out 上 gap 为 20 轮 +0.018344 [+0.017583, +0.019105]、60 轮 +0.013892 [+0.011744, +0.016040]，逐 seed 变化（60 轮 − 20 轮）为 -0.004743 [-0.007931, -0.001554]（95% CI [-0.007931, -0.001554]（n=3, df=2），判定：缩小但未关闭）。

**Q3（SR 是否恢复最近舍入的代价）** 随机舍入把最近舍入的代价基本抵消：8 bit held-out 上 RN@N=1−N=inf = +0.018344 [+0.017583, +0.019105]，SR@N=1−N=inf = +0.000038 [-0.000584, +0.000661]，SR−RN = -0.018306 [-0.019369, -0.017243]（95% CI [-0.019369, -0.017243]（n=8, df=7），判定：区间 < 0）；逐 seed 恢复比例 (RN−SR)/(RN−inf) = s0 95.2%, s1 101.3%, s2 96.9%, s3 94.4%, s4 100.4%, s5 103.0%, s6 105.9%, s7 101.0%（均值 99.8%）。

## 6 排除项、缺失格与限制

**没有排除任何已存在的 run**：51 个 run 全部进入表格（每个 run 的目录见 §7）。缺的是训练侧从未产出的格：

- lr=1e-3 只有 4 个 run：seed 0 的 N ∈ {1,20,inf} 与 seed 1 的 N=1。所以该档的 N=1 − N=inf 对比 **n=1、区间不可估**，N=20 单元格 n=1；任何「随 lr 单调」的说法在这一档上没有证据。
- results_long/ 与 results_sr/ 只覆盖 N ∈ {1, inf}，没有 N=20；因此表 2、表 3 只回答端点。
- results_sr/ 有 8 个 seed，但只有 N=1，没有 SR × N=20、也没有 SR × 不同 lr。
- 4 bit / 6 bit 不在本轮范围（脚本只算 8 bit 与 FP32），表 1–3 的结论不外推到低比特。

**读法与限制**

1. 每组 n ≤ 3（lr=1e-3 的对比 n = 1）。区间是逐 seed 配对的 95% t 区间，df = n−1；n = 1 时只报点值并写明「区间不可估」。p 值只作描述，不做多重比较校正，也不作为显著性主张。
2. 配对只在**同 seed** 上成立：seed 同时决定数据划分与参数初始化（本表所有 run 的划分都是 0，差异落在初始化与训练噪声上）。跨学习率、跨预算的对比因此是「同划分、不同初始化」。
3. first400 与监控 2,000 有重叠（见 §0），其绝对值受选轮影响、偏乐观；held-out 1,556 与选轮无交集，是主口径。两个口径若方向不一致，报告以主口径为准并指出不一致。
4. SR 的身份没有进 result.json（provenance 见 §3），只能靠目录 + 脚本；网格残差无法区分 SR 与 RN。
5. 8 bit 是仅权重 fake quant（激活 FP32），不是整网 int8；表里的「部署 MAE」指这一口径。

snap_every 缺席或为 None 的 run（按 N=1 读，G6 已核实它们在网格上）：results/utk-relu_qat-none-s0-1793011336（args 无 snap_every -> 按 N=1 读（G6 核））

## 7 run 清单（51 个）

完整目录即第一列；best.pth 与 result.json 都在其中。state_dict_key 全为 model_state_dict（ema_mode = none，与 deploy_table.weight_key 一致）。

| run 目录 | 组 | lr | epochs | N | seed | FP32 held-out | 8 bit held-out | FP32 first400 | 8 bit first400 | live MAE（监控 2000） | 前向秒 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| results_long/utk-relu_qat-none-s0-1793489886 | long60 | 0.0001 | 60 | inf (snap_every=0) | 0 | 0.051455 | 0.051456 | 0.052840 | 0.052023 | 0.049806 | 138.9 |
| results_long/utk-relu_qat-none-s1-1793496960 | long60 | 0.0001 | 60 | inf (snap_every=0) | 1 | 0.050952 | 0.051188 | 0.052012 | 0.052089 | 0.049171 | 138.7 |
| results_long/utk-relu_qat-none-s2-1793505234 | long60 | 0.0001 | 60 | inf (snap_every=0) | 2 | 0.052321 | 0.052426 | 0.055915 | 0.055772 | 0.050998 | 138.6 |
| results_long/utk-relu_qat-none-s0-1793485675 | long60 | 0.0001 | 60 | 1 | 0 | 0.064921 | 0.064921 | 0.066420 | 0.066420 | 0.063862 | 138.9 |
| results_long/utk-relu_qat-none-s1-1793492024 | long60 | 0.0001 | 60 | 1 | 1 | 0.066075 | 0.066075 | 0.068387 | 0.068387 | 0.065462 | 139.6 |
| results_long/utk-relu_qat-none-s2-1793499457 | long60 | 0.0001 | 60 | 1 | 2 | 0.065750 | 0.065750 | 0.065818 | 0.065818 | 0.063937 | 138.7 |
| results_lr/utk-relu_qat-none-s0-1793519563 | lr1e-3 | 0.001 | 20 | inf (snap_every=0) | 0 | 0.052130 | 0.052157 | 0.051976 | 0.051860 | 0.049838 | 96.8 |
| results_lr/utk-relu_qat-none-s0-1793516710 | lr1e-3 | 0.001 | 20 | 1 | 0 | 0.052180 | 0.052180 | 0.053858 | 0.053858 | 0.051310 | 144.5 |
| results_lr/utk-relu_qat-none-s1-1793520772 | lr1e-3 | 0.001 | 20 | 1 | 1 | 0.053721 | 0.053721 | 0.056341 | 0.056341 | 0.051885 | 98.4 |
| results_lr/utk-relu_qat-none-s0-1793518513 | lr1e-3 | 0.001 | 20 | 20 | 0 | 0.051037 | 0.051000 | 0.050457 | 0.050270 | 0.049626 | 143.5 |
| results_lr/utk-relu_qat-none-s0-1793474884 | lr3e-3 | 0.003 | 20 | inf (snap_every=0) | 0 | 0.065094 | 0.065044 | 0.064969 | 0.064870 | 0.062967 | 143.8 |
| results_lr/utk-relu_qat-none-s1-1793478537 | lr3e-3 | 0.003 | 20 | inf (snap_every=0) | 1 | 0.064619 | 0.064723 | 0.065789 | 0.065702 | 0.062861 | 98.6 |
| results_lr/utk-relu_qat-none-s2-1793483431 | lr3e-3 | 0.003 | 20 | inf (snap_every=0) | 2 | 0.060871 | 0.060818 | 0.061733 | 0.061630 | 0.059752 | 120.8 |
| results_lr/utk-relu_qat-none-s0-1793472498 | lr3e-3 | 0.003 | 20 | 1 | 0 | 0.069866 | 0.069866 | 0.071433 | 0.071433 | 0.069562 | 140.3 |
| results_lr/utk-relu_qat-none-s1-1793475705 | lr3e-3 | 0.003 | 20 | 1 | 1 | 0.066557 | 0.066557 | 0.065655 | 0.065655 | 0.063516 | 96.7 |
| results_lr/utk-relu_qat-none-s2-1793479561 | lr3e-3 | 0.003 | 20 | 1 | 2 | 0.073935 | 0.073935 | 0.075271 | 0.075271 | 0.073070 | 98.4 |
| results_lr/utk-relu_qat-none-s0-1793474018 | lr3e-3 | 0.003 | 20 | 20 | 0 | 0.061567 | 0.061475 | 0.063812 | 0.063681 | 0.059146 | 213.7 |
| results_lr/utk-relu_qat-none-s1-1793477195 | lr3e-3 | 0.003 | 20 | 20 | 1 | 0.061358 | 0.061433 | 0.061889 | 0.061969 | 0.060944 | 110.4 |
| results_lr/utk-relu_qat-none-s2-1793481326 | lr3e-3 | 0.003 | 20 | 20 | 2 | 0.061832 | 0.061878 | 0.061411 | 0.061428 | 0.061409 | 198.9 |
| results/utk-relu_qat-none-s0-1793111850 | main20 | 0.0001 | 20 | inf (snap_every=0) | 0 | 0.054271 | 0.054421 | 0.055216 | 0.054847 | 0.053188 | 135.9 |
| results/utk-relu_qat-none-s1-1793452884 | main20 | 0.0001 | 20 | inf (snap_every=0) | 1 | 0.054170 | 0.054756 | 0.053882 | 0.054301 | 0.052988 | 136.8 |
| results/utk-relu_qat-none-s2-1793453651 | main20 | 0.0001 | 20 | inf (snap_every=0) | 2 | 0.056189 | 0.055610 | 0.060886 | 0.060445 | 0.055615 | 138.7 |
| results/utk-relu_qat-none-s3-1793454406 | main20 | 0.0001 | 20 | inf (snap_every=0) | 3 | 0.054296 | 0.054268 | 0.057307 | 0.057113 | 0.053728 | 79.6 |
| results/utk-relu_qat-none-s4-1793455163 | main20 | 0.0001 | 20 | inf (snap_every=0) | 4 | 0.054849 | 0.055085 | 0.058159 | 0.058396 | 0.053149 | 79.1 |
| results/utk-relu_qat-none-s5-1793455928 | main20 | 0.0001 | 20 | inf (snap_every=0) | 5 | 0.053433 | 0.053689 | 0.055227 | 0.055520 | 0.052881 | 79.4 |
| results/utk-relu_qat-none-s6-1793456701 | main20 | 0.0001 | 20 | inf (snap_every=0) | 6 | 0.055031 | 0.055322 | 0.055750 | 0.055623 | 0.052493 | 79.2 |
| results/utk-relu_qat-none-s7-1793457541 | main20 | 0.0001 | 20 | inf (snap_every=0) | 7 | 0.055815 | 0.055801 | 0.056469 | 0.056862 | 0.053560 | 59.1 |
| results/utk-relu_qat-none-s0-1793011336 | main20 | 0.0001 | 20 | 1 | 0 | 0.074110 | 0.074110 | 0.074067 | 0.074067 | 0.071601 | 136.0 |
| results/utk-relu_qat-none-s1-1793445602 | main20 | 0.0001 | 20 | 1 | 1 | 0.073696 | 0.073696 | 0.076339 | 0.076339 | 0.074243 | 136.7 |
| results/utk-relu_qat-none-s2-1793446593 | main20 | 0.0001 | 20 | 1 | 2 | 0.072886 | 0.072886 | 0.073212 | 0.073212 | 0.071651 | 138.1 |
| results/utk-relu_qat-none-s3-1793447591 | main20 | 0.0001 | 20 | 1 | 3 | 0.072055 | 0.072055 | 0.071730 | 0.071730 | 0.071176 | 79.7 |
| results/utk-relu_qat-none-s4-1793448669 | main20 | 0.0001 | 20 | 1 | 4 | 0.073206 | 0.073206 | 0.076112 | 0.076112 | 0.073864 | 79.0 |
| results/utk-relu_qat-none-s5-1793449730 | main20 | 0.0001 | 20 | 1 | 5 | 0.071326 | 0.071326 | 0.072119 | 0.072119 | 0.071891 | 79.0 |
| results/utk-relu_qat-none-s6-1793450777 | main20 | 0.0001 | 20 | 1 | 6 | 0.074825 | 0.074825 | 0.077078 | 0.077078 | 0.075820 | 79.2 |
| results/utk-relu_qat-none-s7-1793451837 | main20 | 0.0001 | 20 | 1 | 7 | 0.073602 | 0.073602 | 0.074421 | 0.074421 | 0.072190 | 59.1 |
| results/utk-relu_qat-none-s0-1793030915 | main20 | 0.0001 | 20 | 20 | 0 | 0.059096 | 0.059217 | 0.061906 | 0.062008 | 0.059139 | 136.1 |
| results/utk-relu_qat-none-s1-1793440012 | main20 | 0.0001 | 20 | 20 | 1 | 0.058857 | 0.058668 | 0.058861 | 0.058662 | 0.055498 | 136.0 |
| results/utk-relu_qat-none-s2-1793440765 | main20 | 0.0001 | 20 | 20 | 2 | 0.060533 | 0.061374 | 0.063095 | 0.064397 | 0.060011 | 138.6 |
| results/utk-relu_qat-none-s3-1793441598 | main20 | 0.0001 | 20 | 20 | 3 | 0.057971 | 0.057905 | 0.059499 | 0.059764 | 0.057899 | 80.2 |
| results/utk-relu_qat-none-s4-1793442496 | main20 | 0.0001 | 20 | 20 | 4 | 0.061078 | 0.061083 | 0.060326 | 0.060251 | 0.057897 | 80.4 |
| results/utk-relu_qat-none-s5-1793443267 | main20 | 0.0001 | 20 | 20 | 5 | 0.059018 | 0.059069 | 0.059468 | 0.059425 | 0.056313 | 79.2 |
| results/utk-relu_qat-none-s6-1793444050 | main20 | 0.0001 | 20 | 20 | 6 | 0.059421 | 0.059599 | 0.058044 | 0.058241 | 0.056841 | 79.2 |
| results/utk-relu_qat-none-s7-1793444825 | main20 | 0.0001 | 20 | 20 | 7 | 0.059861 | 0.059091 | 0.062163 | 0.061725 | 0.059320 | 59.2 |
| results_sr/utk-relu_qat-none-s0-1793510216 | sr | 0.0001 | 20 | 1 | 0 | 0.055376 | 0.055376 | 0.057097 | 0.057097 | 0.053691 | 132.1 |
| results_sr/utk-relu_qat-none-s1-1793512950 | sr | 0.0001 | 20 | 1 | 1 | 0.054516 | 0.054516 | 0.057575 | 0.057575 | 0.052884 | 132.8 |
| results_sr/utk-relu_qat-none-s2-1793514777 | sr | 0.0001 | 20 | 1 | 2 | 0.056143 | 0.056143 | 0.059078 | 0.059078 | 0.054974 | 77.9 |
| results_sr/utk-relu_qat-none-s3-1793554855 | sr | 0.0001 | 20 | 1 | 3 | 0.055266 | 0.055266 | 0.055316 | 0.055316 | 0.054023 | 81.1 |
| results_sr/utk-relu_qat-none-s4-1793557139 | sr | 0.0001 | 20 | 1 | 4 | 0.055009 | 0.055009 | 0.057159 | 0.057159 | 0.053333 | 81.1 |
| results_sr/utk-relu_qat-none-s5-1793558720 | sr | 0.0001 | 20 | 1 | 5 | 0.053154 | 0.053154 | 0.054903 | 0.054903 | 0.052068 | 81.3 |
| results_sr/utk-relu_qat-none-s6-1793560125 | sr | 0.0001 | 20 | 1 | 6 | 0.054175 | 0.054175 | 0.056429 | 0.056429 | 0.052590 | 81.1 |
| results_sr/utk-relu_qat-none-s7-1793561553 | sr | 0.0001 | 20 | 1 | 7 | 0.055620 | 0.055620 | 0.058297 | 0.058297 | 0.054574 | 34.6 |

## 8 复现

    python scripts/round20_mechanism.py --workers 4 --threads 8   # CPU，实测约 1 分钟

产物：quant_runs/mechanism_report.md（本文）、quant_runs/mechanism_stats.json（全精度数值）、logs/round20_mechanism.log（stdout 全文）。脚本只读 results*/、quant_runs/*.csv、scripts/、data/utkface_raw224.npy，不写上述三个文件以外的任何路径，不训练。口径依赖 scripts/heldout_eval.py（VAL_RATIO/N_FIRST/N_MON 已在启动时断言）与 scripts/deploy_table.py（qdq_、MEAN/STD、split_seed_of）。

