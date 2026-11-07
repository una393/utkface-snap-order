# run 层推断：把稿件里的关键对比从图像层重算到 run 层

生成：`scripts/paper2_runlevel.py`（2026-10-31 17:00:01，耗时 5.71 s）。

输入（全部只读）：`quant_runs/deploy_table_v2.csv`、`quant_runs/perimage/<run>__{fp32,8,6,4}bit.npz`、`results/<run>/result.json 的 zero_baseline_mae`、`results/<run>/best.pth`。

## 0 口径（先读这一节，再读数字）

- **pairing_unit**：seed（每个 seed 同时决定数据划分与参数初始化）
- **run_statistic**：该 run 在 400 张 first400 部署子集上的 MAE，取自 perimage/<run>__fp32.npz 的逐图 |pred-y| 均值（全精度）；与 deploy_table_v2.csv 的 mae_fp32 列一致到 4 位小数，逐对差值见 §1.0 的 max|Δcsv−Δnpz|
- **delta_definition**：left_MAE - right_MAE；负值 = 左侧更准
- **paired_t**：df = n-1，n = 两侧都有 run 的同 seed 配对数
- **bootstrap**：两级：先整 run(=seed) 有放回重采样 n 个，再在每个被抽中的 run 内对 400 张图有放回重采样；10000 次；百分位法 2.5/97.5；rng seed=20261031
- **merged_se**：sqrt(sigma_I^2/M + sigma_R^2/n)，M=400，df 用 Satterthwaite
- **tost**：配对 t 的双单侧检验，df=n-1，alpha=0.05；判定 = p_tost < alpha
- **tost_margin_preset**：0.0016 为预设值（零基线 0.1553 的 1% = 0.001553，取整到 0.0016），不是从本数据反推
- **mde**：MDE = (t_{1-alpha/2,df} + t_{0.8,df}) * SE，df=n-1
- **main_config_filter**：ema_decay==0.99 且 split_seed==seed（沿用 scripts/paper2_figs_add.py:66-68 的 is_main_config）

行筛选：`deploy_table_v2.csv` 共 30 行，主配置 23 行，筛掉 7 行（见 §8）。
有 perimage 的 run：30 个。

**本文档里有两套 n，永远不要混用**：图像层 n = 400 张图（稿件现行区间的单位）；run 层 n = seed 数。

## 1 三个核心对比的 run 层推断

点估计可逐行核对：每个 run 的 MAE 即 `deploy_table_v2.csv` 的 `mae_fp32` 列（400 张 first400）；配对按 seed，Δ_s = 左 MAE − 右 MAE。

### 1.0 同 seed 配对表

| 组 | 对比 | seed | 左 MAE | 右 MAE | Δ_s（perimage，主） | Δ_s（csv） | 共同图像 |
|---|---|---|---|---|---|---|---|
| 1a | 间隔效应：none, N=20 − N=∞ | 0 | 0.0619 | 0.0552 | +0.00669 | +0.0067 | 400 |
| 1b | 间隔效应：none, N=1 − N=∞ | 0 | 0.0741 | 0.0552 | +0.01885 | +0.0189 | 400 |
| 1c | 影子有无：N=20, after_snap − none | 0 | 0.0594 | 0.0619 | -0.00254 | -0.0025 | 400 |
| 1d | 顺序效应：N=20, before_snap − after_snap | 0 | 0.0607 | 0.0594 | +0.00132 | +0.0013 | 400 |
| 1d | 顺序效应：N=20, before_snap − after_snap | 1 | 0.0594 | 0.0593 | +0.00016 | +0.0001 | 400 |
| 1d | 顺序效应：N=20, before_snap − after_snap | 2 | 0.0562 | 0.0570 | -0.00084 | -0.0008 | 400 |
| S | 间隔效应：after_snap, N=1 − N=20 | 0 | 0.0748 | 0.0594 | +0.01545 | +0.0154 | 400 |
| S | 间隔效应：after_snap, N=1 − N=20 | 1 | 0.0773 | 0.0593 | +0.01806 | +0.0180 | 400 |
| S | 间隔效应：after_snap, N=1 − N=20 | 2 | 0.0712 | 0.0570 | +0.01418 | +0.0142 | 400 |
| S | 间隔效应：after_snap, N=5 − N=20 | 0 | 0.0657 | 0.0594 | +0.00630 | +0.0063 | 400 |
| S | 间隔效应：after_snap, N=5 − N=20 | 1 | 0.0648 | 0.0593 | +0.00551 | +0.0055 | 400 |
| S | 间隔效应：after_snap, N=5 − N=20 | 2 | 0.0634 | 0.0570 | +0.00637 | +0.0064 | 400 |
| S | 间隔效应：before_snap, N=5 − N=20 | 0 | 0.0658 | 0.0607 | +0.00513 | +0.0051 | 400 |
| S | 间隔效应：before_snap, N=5 − N=20 | 1 | 0.0643 | 0.0594 | +0.00489 | +0.0049 | 400 |
| S | 间隔效应：before_snap, N=5 − N=20 | 2 | 0.0635 | 0.0562 | +0.00732 | +0.0073 | 400 |
| S | 间隔效应：before_snap, N=1 − N=20 | 0 | 0.0741 | 0.0607 | +0.01343 | +0.0134 | 400 |
| S | 间隔效应：none, N=1 − N=5 | 0 | 0.0741 | 0.0660 | +0.00805 | +0.0081 | 400 |
| S | 间隔效应：none, N=5 − N=20 | 0 | 0.0660 | 0.0619 | +0.00412 | +0.0041 | 400 |
| S | 影子有无：N=5, after_snap − none | 0 | 0.0657 | 0.0660 | -0.00036 | -0.0003 | 400 |
| S | 影子有无：N=5, after_snap − none | 1 | 0.0648 | 0.0647 | +0.00004 | +0.0001 | 400 |
| S | 影子有无：N=5, after_snap − none | 2 | 0.0634 | 0.0639 | -0.00051 | -0.0005 | 400 |
| S | 影子有无：N=20, before_snap − none | 0 | 0.0607 | 0.0619 | -0.00122 | -0.0012 | 400 |
| S | 影子有无：N=5, before_snap − none | 0 | 0.0658 | 0.0660 | -0.00021 | -0.0002 | 400 |
| S | 影子有无：N=5, before_snap − none | 1 | 0.0643 | 0.0647 | -0.00041 | -0.0004 | 400 |
| S | 影子有无：N=5, before_snap − none | 2 | 0.0635 | 0.0639 | -0.00040 | -0.0004 | 400 |

左/右 MAE 列取自 `deploy_table_v2.csv` 的 `mae_fp32`（4 位小数）；**Δ_s 的主列用 `perimage/*.npz` 的全精度逐图 MAE**。两列的最大差 = 0.000064，即 csv 的 4 位小数取整误差，不影响任何结论的量级。

### 1.1 三组核心对比的汇总

| 组 | 对比 | n（seed） | 点估计 | 配对 t 95% CI (df) | 符号检验双侧 p | 两级 bootstrap 95% CI |
|---|---|---|---|---|---|---|
| 1a | 间隔效应：none, N=20 − N=∞ | 1 | +0.0067 | 不可估 (df=0) | 1.0000 | [+0.0035, +0.0099]（n=1 ⇒ 只含 run 内图像重采样） |
| 1b | 间隔效应：none, N=1 − N=∞ | 1 | +0.0189 | 不可估 (df=0) | 1.0000 | [+0.0135, +0.0241]（n=1 ⇒ 只含 run 内图像重采样） |
| 1c | 影子有无：N=20, after_snap − none | 1 | -0.0025 | 不可估 (df=0) | 1.0000 | [-0.0047, -0.0004]（n=1 ⇒ 只含 run 内图像重采样） |
| 1d | 顺序效应：N=20, before_snap − after_snap | 3 | +0.0002 | [-0.0025, +0.0029] (df=2) | 1.0000 | [-0.0011, +0.0017] |

**n 是多少，逐组写清：**

- 1a `间隔效应：none, N=20 − N=∞`：**n = 1 个 seed**（seed = [0]）。　sd 与配对 t 区间在 n=1 时不存在（df=0），只能给点估计；该对比在 run 层**没有任何重复**。
- 1b `间隔效应：none, N=1 − N=∞`：**n = 1 个 seed**（seed = [0]）。　sd 与配对 t 区间在 n=1 时不存在（df=0），只能给点估计；该对比在 run 层**没有任何重复**。
- 1c `影子有无：N=20, after_snap − none`：**n = 1 个 seed**（seed = [0]）。　sd 与配对 t 区间在 n=1 时不存在（df=0），只能给点估计；该对比在 run 层**没有任何重复**。
- 1d `顺序效应：N=20, before_snap − after_snap`：**n = 3 个 seed**（seed = [0, 1, 2]）。　sd = 0.001082，SE = 0.000625，df = 2。

### 1.2 为什么 n 只有这么几个（不是笔误）

主配置里 N=∞（`snap_every=0`）只有 seed 0 的两个 run（none 与 after_snap）；`none` 协议在 N=20 也只有 seed 0。
因此凡是含 N=∞ 的对比、以及 after_snap − none @ N=20，配对后 n 只能是 1。图像层的 400 张图**不能**替代 seed 重复——它们只重复测量同一个 run。

### 1.3 补充：主配置里真正有跨 seed 重复的对比（n≥3）

这几组是稿件可以安全升级到 run 层口径的候选。

| 对比 | n | 点估计 | 95% CI | 符号 p | bootstrap 95% CI |
|---|---|---|---|---|---|
| 间隔效应：after_snap, N=1 − N=20 | 3 | +0.0159 | [+0.0110, +0.0208] | 0.2500 | [+0.0117, +0.0200] |
| 间隔效应：after_snap, N=5 − N=20 | 3 | +0.0061 | [+0.0049, +0.0072] | 0.2500 | [+0.0029, +0.0092] |
| 间隔效应：before_snap, N=5 − N=20 | 3 | +0.0058 | [+0.0025, +0.0091] | 0.2500 | [+0.0025, +0.0092] |
| 间隔效应：before_snap, N=1 − N=20 | 1 | +0.0134 | 不可估（df=0） | 1.0000 | [+0.0080, +0.0185] |
| 间隔效应：none, N=1 − N=5 | 1 | +0.0080 | 不可估（df=0） | 1.0000 | [+0.0051, +0.0111] |
| 间隔效应：none, N=5 − N=20 | 1 | +0.0041 | 不可估（df=0） | 1.0000 | [-0.0009, +0.0090] |
| 影子有无：N=5, after_snap − none | 3 | -0.0003 | [-0.0010, +0.0004] | 1.0000 | [-0.0013, +0.0009] |
| 影子有无：N=20, before_snap − none | 1 | -0.0012 | 不可估（df=0） | 1.0000 | [-0.0031, +0.0006] |
| 影子有无：N=5, before_snap − none | 3 | -0.0003 | [-0.0006, -0.0001] | 0.2500 | [-0.0016, +0.0009] |

## 2 推断单位对照表（评审 F1 的证据）

同一批逐图差值，三种标准误。图像层 = 稿件现行口径（seed 0 那一对、400 张逐图配对 t，df=399）；run 层 = 以 seed 为单位（df=n−1）；合并 = 两层方差相加。

| 对比 | 图像层 SE (n=400) | run 层 SE | 合并 SE | 合并后 95% CI | 含 0？ |
|---|---|---|---|---|---|
| 影子有无：N=20, after_snap − none | 0.001066 | 不可估 (n=1) | 0.001103 | [-0.0047, -0.0004] | 不含 0 |
| 间隔效应：none, N=1 − N=∞ | 0.002697 | 不可估 (n=1) | 0.003346 | [+0.0118, +0.0259] | 不含 0 |
| 顺序效应：N=20, before_snap − after_snap | 0.000662 | 0.000625 (n=3) | 0.000911 | [-0.0007, +0.0034] | **含 0** |

### 影子有无：N=20, after_snap − none

- 图像层：均差 -0.0025，SE = 0.001066（M = 400），df = 399，95% CI = [-0.0046, -0.0004]，不含 0。
- run 层：均差 -0.0025，**SE 不可估**（n = 1，df = 0）。
- 合并：SE_merged = sqrt(sigma_I^2/M + sigma_R^2/n)，M=400 张/run，n=seed 数。σ_R 来源：supp_shadow_after_vs_none_N5（n=1 无法自估 sigma_R，借用 supp_shadow_after_vs_none_N5 的跨 seed sd（df=2））。合并 SE = 0.001103，比图像层放大 **1.03×**，df(Satterthwaite) = 229.71，95% CI = [-0.0047, -0.0004]，**不含 0**。

### 间隔效应：none, N=1 − N=∞

- 图像层：均差 +0.0189，SE = 0.002697（M = 400），df = 399，95% CI = [+0.0135, +0.0242]，不含 0。
- run 层：均差 +0.0189，**SE 不可估**（n = 1，df = 0）。
- 合并：SE_merged = sqrt(sigma_I^2/M + sigma_R^2/n)，M=400 张/run，n=seed 数。σ_R 来源：supp_interval_N1_vs_N20_after_snap（n=1 无法自估 sigma_R，借用 supp_interval_N1_vs_N20_after_snap 的跨 seed sd（df=2））。合并 SE = 0.003346，比图像层放大 **1.24×**，df(Satterthwaite) = 16.04，95% CI = [+0.0118, +0.0259]，**不含 0**。

### 顺序效应：N=20, before_snap − after_snap

- 图像层：均差 +0.0013，SE = 0.000662（M = 400），df = 399，95% CI = [+0.0000, +0.0026]，不含 0。
- run 层：均差 +0.0002，SE = 0.000625（n = 3），df = 2，95% CI = [-0.0025, +0.0029]。
- 合并：SE_merged = sqrt(sigma_I^2/M + sigma_R^2/n)，M=400 张/run，n=seed 数。σ_R 来源：g1d_order_before_vs_after_N20（自估：该对比本身 n=3，sigma_R 由自身跨 seed sd 给出）。合并 SE = 0.000911，比图像层放大 **1.37×**，df(Satterthwaite) = 8.96，95% CI = [-0.0007, +0.0034]，**含 0**——该对比在合并口径下无法排除无效应。

**放大区间的主要不是标准误，而是自由度。** 以末行为例：run 层 SE = 0.000625 与图像层 SE = 0.000662 同量级，但 t 临界从 t_{0.975,399} = 1.966 变成 t_{0.975,2} = 4.303（2.19×），95% 区间半宽因而从 0.001302 变成 0.002689。单位错配的代价主要落在 df 上。

读法：n=1 的对比给不出 run 层区间，图像层区间因此**不能**当作 protocol/interval 效应的不确定性；n≥2 的对比里，run 层 SE 未必更大，但 df 从 399 掉到 n−1，t 临界放大才是区间变宽的主因。

## 3 margin 敏感性与 TOST（顺序效应，N=20）

对比：顺序效应：N=20, before_snap − after_snap；逐 seed Δ = s0: +0.0013，s1: +0.0001，s2: -0.0008。

**margin 0.0016 是预设值**：预设：稿件所用零基线 0.1553（= seed 0 划分的 zero_baseline_mae 0.155267；三 seed 均值 0.154011）的 1% = 0.001553，取 0.0016。该 margin 在本次分析前已确定（沿用稿件既有设定），不是从观测到的顺序效应反推。其余 margin 只是敏感性网格，均已列出以便读者判断结论对 margin 的依赖。

| margin | p_lower | p_upper | p_TOST | 判定（α=0.05） | 90% CI |
|---|---|---|---|---|---|
| 0.0010 | 0.0958 | 0.1677 | 0.1677 | 不能判定等价 | [-0.0016, +0.0020] |
| 0.0016 | 0.0505 | 0.0784 | 0.0784 | 不能判定等价 | [-0.0016, +0.0020] |
| 0.0018 | 0.0421 | 0.0632 | 0.0632 | 不能判定等价 | [-0.0016, +0.0020] |
| 0.0050 | 0.0070 | 0.0083 | 0.0083 | 等价通过 | [-0.0016, +0.0020] |

**0.0016（预设 margin）这一行不通过**：p_TOST = 0.0784 > 0.05，按 run 层口径不能判定顺序效应等价于 0。margin 0.0018 同样不通过（p_TOST = 0.0632），只有 0.005 通过。

**80% 功效下可检出效应（MDE）**：MDE = (t_{1-alpha/2, df} + t_{power, df}) * SE, df = n-1。
代入 SE = 0.000625、df = 2：t_{0.975,2} = 4.3027、t_{0.80,2} = 1.0607 ⇒ **MDE = 0.003352**。
同一 SE 下 95% 区间半宽 = 0.002689（= t_{0.975,2} × SE），即本检验在 run 层的分辨率约 0.0027。

## 4 MAE / MAE₀ 归一化（评审 M1）

ratio = mae_fp32 / zero_baseline_mae（同一 run 自己划分上的恒预测均值基线）。ratio<1 表示 live MAE 低于同划分零基线。

| run | mode | N | seed | MAE | MAE₀ | MAE/MAE₀ |
|---|---|---|---|---|---|---|
| utk-relu_qat-after_snap-s0-1793005531 | after_snap | 1 | 0 | 0.0748 | 0.1553 | 0.4817 |
| utk-relu_qat-after_snap-s0-1793029658 | after_snap | 5 | 0 | 0.0657 | 0.1553 | 0.4231 |
| utk-relu_qat-after_snap-s0-1793033217 | after_snap | 20 | 0 | 0.0594 | 0.1553 | 0.3826 |
| utk-relu_qat-after_snap-s0-1793111850 | after_snap | 0 | 0 | 0.0556 | 0.1553 | 0.3581 |
| utk-relu_qat-after_snap-s1-1793018836 | after_snap | 1 | 1 | 0.0773 | 0.1552 | 0.4981 |
| utk-relu_qat-after_snap-s1-1793038259 | after_snap | 20 | 1 | 0.0593 | 0.1552 | 0.3821 |
| utk-relu_qat-after_snap-s1-1793116221 | after_snap | 5 | 1 | 0.0648 | 0.1552 | 0.4176 |
| utk-relu_qat-after_snap-s2-1793023654 | after_snap | 1 | 2 | 0.0712 | 0.1516 | 0.4697 |
| utk-relu_qat-after_snap-s2-1793040741 | after_snap | 20 | 2 | 0.0570 | 0.1516 | 0.3760 |
| utk-relu_qat-after_snap-s2-1793117387 | after_snap | 5 | 2 | 0.0634 | 0.1516 | 0.4182 |
| utk-relu_qat-before_snap-s0-1793009776 | before_snap | 1 | 0 | 0.0741 | 0.1553 | 0.4772 |
| utk-relu_qat-before_snap-s0-1793028380 | before_snap | 5 | 0 | 0.0658 | 0.1553 | 0.4238 |
| utk-relu_qat-before_snap-s0-1793032011 | before_snap | 20 | 0 | 0.0607 | 0.1553 | 0.3909 |
| utk-relu_qat-before_snap-s1-1793037047 | before_snap | 20 | 1 | 0.0594 | 0.1552 | 0.3828 |
| utk-relu_qat-before_snap-s1-1793115073 | before_snap | 5 | 1 | 0.0643 | 0.1552 | 0.4144 |
| utk-relu_qat-before_snap-s2-1793039497 | before_snap | 20 | 2 | 0.0562 | 0.1516 | 0.3707 |
| utk-relu_qat-before_snap-s2-1793116251 | before_snap | 5 | 2 | 0.0635 | 0.1516 | 0.4189 |
| utk-relu_qat-none-s0-1793011336 | none | 1 | 0 | 0.0741 | 0.1553 | 0.4772 |
| utk-relu_qat-none-s0-1793027163 | none | 5 | 0 | 0.0660 | 0.1553 | 0.4251 |
| utk-relu_qat-none-s0-1793030915 | none | 20 | 0 | 0.0619 | 0.1553 | 0.3987 |
| utk-relu_qat-none-s0-1793111850 | none | 0 | 0 | 0.0552 | 0.1553 | 0.3555 |
| utk-relu_qat-none-s1-1793114048 | none | 5 | 1 | 0.0647 | — | — |
| utk-relu_qat-none-s2-1793115215 | none | 5 | 2 | 0.0639 | 0.1516 | 0.4215 |

三个 seed 的零基线：s0 = 0.155267，s1 = 0.155177，s2 = 0.151589；极差 = **0.003678**，相对均值（0.154011）为 **2.388%**。
归一化 ratio 的中位数 0.4179，范围 0.3555–0.4981。跨 seed 比较一律同时给原始 MAE 与归一化值：seed 不同即划分不同，零基线本身不同。

## 5 符号检验的精确二项 p（评审 M6）

精确二项检验，p0=0.5；双侧 = 2*min(单侧) 截到 1；单侧给 p_greater。

| 同号数/总数 | 双侧 p | 单侧 p（P(X≥k)） |
|---|---|---|
| 4/4 | 0.125000 | 0.062500 |
| 3/4 | 0.625000 | 0.312500 |
| 3/3 | 0.250000 | 0.125000 |
| 2/3 | 1.000000 | 0.500000 |
| 1/1 | 1.000000 | 0.500000 |

读法：n=4 时 4/4 同号的双侧 p 也只有 0.125，n=4 的符号检验没有 5% 分辨率；n=3 时 3/3 同号双侧 p = 0.25。这是符号检验的固有下限，不是样本量问题。

## 6 4 bit 归因（评审 M2 的便宜部分）

口径：主配置 23 个 run（deploy_table_v2.csv 的 mae_4 列）；n = 23；有 checkpoint 的 22 个。
权重来源：results/<run>/best.pth，ema_state_dict（mode!=none）或 model_state_dict（none），与 scripts/deploy_table.py:148 一致。

- **差于该 run 自身零基线的 run 数：7 / 22**（23 个 run 里有零基线可比的 22 个；其余 run 的 `results/` 目录已不存在）
- mae_4 中位数 **0.1027**（均值 0.1267，sd 0.0484，范围 0.0849–0.2337）
- 相对自身 FP32 的退化倍数：中位数 **1.50×**，范围 1.21–3.56×
- 按协议的中位数：after_snap 0.1058、before_snap 0.1027、none 0.0897
- 按间隔的中位数：N=0 0.1288、N=1 0.0910、N=5 0.0969、N=20 0.1267

逐通道统计：每个 Conv2d/Linear 权重按输出通道取 max|w| / mean|w|；全模型所有通道汇成分位数。

- Spearman(mae_4, 各 run 的通道 max/mean 最大值) = **-0.131**（n = 22）
- Spearman(mae_4, p95) = **0.073**
- Spearman(mae_4, 同张量归一化的极端度) = **0.194**；对每个 run、每个参数张量取逐输出通道 max/mean 的最大值，再除以该张量在全部 run 上的中位数，取全部张量里的最大者；=1 表示与同张量的典型 run 一样，>1 表示该 run 在这个张量上更极端

n = 22 时，Spearman 双侧 5% 显著需要 |ρ| ≥ 0.423；实测三个 ρ 都在此之下，即**没有**可判定的单调关系。

读法：全局 max/mean 的 p95 在 22 个 run 上只落在 11.6–12.4，几乎被架构本身决定，没有判别力；换成「同张量跨 run 归一化」后仍然没有与 mae_4 同向的排序。**故本口径下 4 bit 崩溃没有在少数张量的极端 max/mean 上留下痕迹**——这是一条否定性证据，不是缺失数据。

| run | mode | N | seed | mae_4 | MAE₀ | 差于基线 | 退化× | p50 | p95 | max | 极端度 | max 所在张量 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| utk-relu_qat-after_snap-s0-1793005531 | after_snap | 1 | 0 | 0.0910 | 0.1553 | 否 | 1.22 | 4.5 | 12.4 | 78.8 | 2.488 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-after_snap-s0-1793029658 | after_snap | 5 | 0 | 0.0864 | 0.1553 | 否 | 1.32 | 4.3 | 11.9 | 74.2 | 1.268 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-after_snap-s0-1793033217 | after_snap | 20 | 0 | 0.0849 | 0.1553 | 否 | 1.43 | 4.4 | 11.8 | 72.6 | 1.145 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-after_snap-s0-1793111850 | after_snap | 0 | 0 | 0.1040 | 0.1553 | 否 | 1.87 | 4.4 | 11.7 | 53.8 | 1.372 | backbone.s3.b4.se.fc1.weight |
| utk-relu_qat-after_snap-s1-1793018836 | after_snap | 1 | 1 | 0.1076 | 0.1552 | 否 | 1.39 | 4.5 | 12.4 | 78.7 | 2.488 | backbone.s3.b4.se.fc1.weight |
| utk-relu_qat-after_snap-s1-1793038259 | after_snap | 20 | 1 | 0.1267 | 0.1552 | 否 | 2.14 | 4.4 | 11.9 | 71.7 | 1.866 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-after_snap-s1-1793116221 | after_snap | 5 | 1 | 0.0969 | 0.1552 | 否 | 1.50 | 4.3 | 11.9 | 72.0 | 1.047 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-after_snap-s2-1793023654 | after_snap | 1 | 2 | 0.2337 | 0.1516 | 是 | 3.28 | 4.5 | 12.4 | 78.7 | 2.488 | backbone.s3.b4.se.fc1.weight |
| utk-relu_qat-after_snap-s2-1793040741 | after_snap | 20 | 2 | 0.2030 | 0.1516 | 是 | 3.56 | 4.4 | 11.8 | 72.3 | 1.174 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-after_snap-s2-1793117387 | after_snap | 5 | 2 | 0.2117 | 0.1516 | 是 | 3.34 | 4.3 | 11.9 | 73.2 | 1.348 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-before_snap-s0-1793009776 | before_snap | 1 | 0 | 0.0899 | 0.1553 | 否 | 1.21 | 4.5 | 12.4 | 78.8 | 2.488 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-before_snap-s0-1793028380 | before_snap | 5 | 0 | 0.0880 | 0.1553 | 否 | 1.34 | 4.3 | 11.9 | 74.2 | 1.153 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-before_snap-s0-1793032011 | before_snap | 20 | 0 | 0.0866 | 0.1553 | 否 | 1.43 | 4.3 | 11.7 | 72.5 | 1.500 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-before_snap-s1-1793037047 | before_snap | 20 | 1 | 0.1557 | 0.1552 | 是 | 2.62 | 4.4 | 11.9 | 78.7 | 1.845 | backbone.s2.b2.se.fc1.weight |
| utk-relu_qat-before_snap-s1-1793115073 | before_snap | 5 | 1 | 0.1027 | 0.1552 | 否 | 1.60 | 4.3 | 11.8 | 71.9 | 1.046 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-before_snap-s2-1793039497 | before_snap | 20 | 2 | 0.1749 | 0.1516 | 是 | 3.11 | 4.4 | 11.8 | 72.2 | 1.129 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-before_snap-s2-1793116251 | before_snap | 5 | 2 | 0.1816 | 0.1516 | 是 | 2.86 | 4.3 | 11.9 | 73.3 | 2.488 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-none-s0-1793011336 | none | 1 | 0 | 0.0900 | 0.1553 | 否 | 1.21 | 4.5 | 12.4 | 78.8 | 2.488 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-none-s0-1793027163 | none | 5 | 0 | 0.0857 | 0.1553 | 否 | 1.30 | 4.3 | 11.9 | 74.1 | 1.500 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-none-s0-1793030915 | none | 20 | 0 | 0.0888 | 0.1553 | 否 | 1.43 | 4.3 | 11.6 | 73.0 | 1.500 | backbone.s3.b3.se.fc1.weight |
| utk-relu_qat-none-s0-1793111850 | none | 0 | 0 | 0.1535 | 0.1553 | 否 | 2.78 | 4.4 | 11.6 | 55.8 | 1.552 | backbone.s3.b4.se.fc1.weight |
| utk-relu_qat-none-s1-1793114048 | none | 5 | 1 | 0.0894 | — | 无基线 | 1.38 | — | — | — | — | 无 best.pth |
| utk-relu_qat-none-s2-1793115215 | none | 5 | 2 | 0.1807 | 0.1516 | 是 | 2.83 | 4.3 | 11.9 | 73.1 | 2.488 | backbone.s3.b3.se.fc1.weight |

## 7 perimage 里的头（T2 的输入检查）

`quant_runs/perimage/` 共 120 个 npz、30 个 run，位宽 = ['4bit', '6bit', '8bit', 'fp32']，头计数 = {'relu_qat': 120}。

- gelu 头逐图预测：**没有**
- bilinear_glu 头逐图预测：**没有**
- relu_qat 头逐图预测：有

结论：只报告 quant_runs/perimage/ 里有无；未跑任何推理。 缺 gelu、bilinear_glu，头对头的 run 层推断必须先在 T2 里补出这些 npz。

## 8 筛掉的行与已知限制

按 `ema_decay==0.99 且 split_seed==seed（沿用 scripts/paper2_figs_add.py:66-68 的 is_main_config）` 筛掉 7 行：

| run | mode | N | seed | ema_decay | split_seed | mae_fp32 | mae_4 | 原因 |
|---|---|---|---|---|---|---|---|---|
| utk-relu_qat-after_snap-s0-1793118469 | after_snap | 20 | 0 | 0.9 | 0 | 0.0598 | 0.0883 | γ=0.9≠0.99 |
| utk-relu_qat-after_snap-s0-1793119670 | after_snap | 20 | 0 | 0.999 | 0 | 0.0598 | 0.0860 | γ=0.999≠0.99 |
| utk-relu_qat-after_snap-s1-1793112917 | after_snap | 20 | 1 | 0.99 | 0 | 0.0582 | 0.0953 | 固定划分 split_seed=0≠seed |
| utk-relu_qat-after_snap-s2-1793113027 | after_snap | 20 | 2 | 0.99 | 0 | 0.0593 | 0.1452 | 固定划分 split_seed=0≠seed |
| utk-relu_qat-after_snap-s3-1793114152 | after_snap | 20 | 3 | 0.99 | 0 | 0.0588 | 0.1295 | 固定划分 split_seed=0≠seed |
| utk-relu_qat-before_snap-s0-1793117363 | before_snap | 20 | 0 | 0.9 | 0 | 0.0612 | 0.0857 | γ=0.9≠0.99 |
| utk-relu_qat-before_snap-s0-1793118574 | before_snap | 20 | 0 | 0.999 | 0 | 0.0598 | 0.1142 | γ=0.999≠0.99 |

限制：

1. run 层 n 由主配置决定：N=∞ 端点只有 seed 0，故所有含 N=∞ 的对比 n=1，没有 run 层区间。
2. 合并 SE 需要 σ_R；n=1 的对比只能从另一个有复制的对比借用，属**插件式近似**，`sigma_R_source` 已逐条写明。
3. bootstrap 只在 n≥2 时才真正两级；n=1 时退化为 run 内图像重采样，与图像层区间同源（这本身就是 F1 的展示）。
4. seed 同时含划分与初始化两个方差源，本设计与 4 bit 归因都不能把它们分开（需 split_seed × init_seed 交叉设计）。
5. 4 bit 的通道 max/mean 是**相关诊断**，不是因果机制；Spearman 只说明排序同向程度。

6. 以下 run 的 `results/<run>/` 目录**已不存在**（无 `result.json`、无 `best.pth`），故不参与零基线归一化与逐通道统计；它们的 MAE 仍取自 `deploy_table_v2.csv`、逐图预测仍取自 `quant_runs/perimage/`，故 run 层配对不受影响：

   - `utk-relu_qat-none-s1-1793114048`

