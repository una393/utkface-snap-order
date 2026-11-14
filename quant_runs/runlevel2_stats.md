# run 层统计推断：固定划分（split_seed = 0）五条主配置序列，n = 8

由 scripts/round20_runlevel.py 生成。每个数字都能在 quant_runs/runlevel2_stats.json 里按本节标注的键复现。

## 0 产物

| 文件 | 字节 | sha256 |
|---|---|---|
| quant_runs/deploy_table_v3b.csv | 6131 | a9c1a0374b9089cc292b040d03705fb0ef1cf0645b18011a01616a3142e7034f |
| quant_runs/runlevel2_stats.json | 87542 | 5fc28b6e60ad4b0a71be73501760f81ad81da51fc589c3525d4716eda513a66c |
| quant_runs/runlevel2_stats.md | 本文件 | 生成后在运行日志里打印 |

## 1 输入与主口径

- 输入：quant_runs/deploy_table_v3.csv（65 行，sha256 650573471f45a4b9...）、quant_runs/perimage/*.npz（264 个文件，覆盖 66 个 run）。
- 主统计量：fp32 逐图误差 MAE = (1/M) * sum_i |pred_i - target_i|，float64 全精度，M = 400。
- 评价集：**公共固定划分 400 张**（每个序列 run 的 rows 前 400 与它逐元素相同，见第 2 节）。
- 8/6/4 bit 列只用于与 csv 的 4 位小数交叉校验，不进入配对对比。
- 五条序列：ema_decay = 0.99 且 split_seed = 0，各 8 个 seed（0..7）。

| 序列键 | 含义 | run 数 | seeds |
|---|---|---|---|
| none_Ninf | none @ N=inf (snap_every=0) | 8 | 0..7 |
| none_N1 | none @ N=1 | 8 | 0..7 |
| none_N20 | none @ N=20 | 8 | 0..7 |
| after_N20 | after_snap @ N=20 | 8 | 0..7 |
| before_N20 | before_snap @ N=20 | 8 | 0..7 |

校验键：meta、series.<键>.per_seed.<seed>.mae。

## 2 数据完整性（验收 B）

- 五条序列各 8 个 run，seed 恰为 0..7（断言，原始输出见第 8 节 [B] 行）。
- 40 个序列 run 共用同一套验证行号（n = 400，rows 逐元素相等，sha256 dfa7983511cddbe8...）：这是固定划分能配对的前提。
- v3b 中不含 npz 的 run：[]；npz 有但 v3b 无对应行的 run：[]。

### 2.1 perimage 的 n 不一致（重要事实）

- 全部 66 个 npz run（264 个 npz 文件）中：存 400 张的 30 个，存 800 张的 36 个。五个序列的 40 个 run 里，400 张的 8 个、800 张的 32 个。任务书里「每个 npz n = 400」的表述与实物不符。
- 存 800 张的 36 个 run 中有 32 个（含五条序列的全部 800 张 run）其 rows 的前 400 与公共固定划分 400 张**逐元素相同**（前缀，不是子集重排），因此可以无歧义地对齐到同一批 400 张图像；其余 4 个属于别的划分（split_seed 1/2 等），不进入本文件的统计。
- csv 的 4 位小数按各 run 自己的口径算：65 行 x 4 列 = 260 个值，按各 run 存储的全部图像复现时 0 处不一致；在有公共 400 前缀的 50 行（200 个值）上改按 400 张子集复现，则有 124 处会变，最大 |Delta| = 0.0054。
- 本文件的全部统计量统一在这批公共 400 张上计算（与任务书 M = 400 一致）；口径差异对点估计的影响见 5.3 的敏感性表。

校验键：npz_geometry、crosscheck_csv_vs_npz。

## 3 缺失行重建（验收 A）

| 项 | 值 |
|---|---|
| run | utk-relu_qat-none-s1-1793114048 |
| snap_every / mode / seed / ema_decay / split_seed | 5 / none / 1 / 0.99 / 1 |
| mae_fp32（npz 重算 -> 4 位） | 0.0647（全精度 0.064718345748） |
| mae_8 | 0.0648（0.064751338353） |
| mae_6 | 0.0683（0.068323581712） |
| mae_4 | 0.0894（0.089446071899） |
| val_mae_train | 0.0631（照抄 deploy_table_v2.csv 该 run 行；npz 不含该列） |

断言：四个 MAE 列与 0.0647 / 0.0648 / 0.0683 / 0.0894 完全相等（容差 1e-12），通过；v2 同一 run 行的四列也是 0.0647 / 0.0648 / 0.0683 / 0.0894，与 npz 一致。

v3b = v3 的 65 行原始字节（逐字节保留、未重排）+ 末尾追加该行，共 66 个数据行。该 run 的验证划分是 split_seed = 1，与五条序列不同，故只入表、不进入任何配对对比。

校验键：rebuild_row。

## 4 方法：三种口径与公式

记 seed s = 0..7，图像 i = 1..M（M = 400），臂 A、B。

逐图绝对误差 e_{s,i} = |pred_{s,i} - y_i|；序列 MAE_s = (1/M) * sum_i e_{s,i}；配对差 d_s = MAE_s(A) - MAE_s(B)。

1) run 层（验收 C）：把 d_s 当作 8 个独立 run 的观测。
   d_bar = (1/8) * sum_s d_s；sd = sqrt( sum_s (d_s - d_bar)^2 / 7 )；SE_run = sd / sqrt(8)；
   95% 区间 = d_bar -+ t_{0.975,7} * SE_run（df = 7）。
   bootstrap：对 8 个 d_s 有放回重抽 10,000 次（numpy.random.default_rng(0)），取 2.5 / 97.5 百分位。
   精确符号检验：k = 负差值个数，X ~ Bin(n_nonzero, 0.5)，双侧 p = min(1, 2 * min(P(X <= k), P(X >= k)))。
2) 图像层（验收 D）：逐图配对差 delta_{s,i} = e_{s,i}(A) - e_{s,i}(B)，SE_img,s = sd(delta_s) / sqrt(M)；
   图像层 SE = (1/8) * sum_s SE_img,s。
3) 合并（验收 D）：SE_comb = sqrt(SE_img^2 + SE_run^2)（两源独立假设下的勾股和）。

三种口径回答的问题不同：

- 图像层 SE 只含「换一批验证图像」的波动，把 M = 400 张图当唯一随机源；它回答「同一批 run、换一批图像，平均差会漂多少」。
- run 层 SE 只含「换一批 run（新初始化，同一划分）」的波动；它回答「再跑一个新 run，平均差会漂多少」。
- 合并 SE 同时含两源；它回答「一个新 run 在 400 张子样本上测得的平均差，相对总体（新 run + 全部验证图像）会漂多少」。
- 关于间隔与顺序的主张是施加在 run 上的：主区间必须用 run 层，图像层只描述图像抽样这一项。

## 5 四个配对对比（验收 C）

### 5.1 逐 seed 差值（A - B，fp32）

| seed | (a) none@N=1 - none@N=inf | (b) none@N=20 - none@N=inf | (c) after_snap - none @N=20 | (d) before_snap - after_snap @N=20 |
|---|---|---|---|---|
| 0 | +0.018852 | +0.006690 | -0.002542 | +0.001321 |
| 1 | +0.022457 | +0.004979 | -0.000627 | +0.000687 |
| 2 | +0.012326 | +0.002209 | -0.003842 | -0.000369 |
| 3 | +0.014423 | +0.002193 | -0.000727 | +0.000022 |
| 4 | +0.017953 | +0.002167 | -0.000248 | +0.000909 |
| 5 | +0.016892 | +0.004241 | -0.000322 | -0.000602 |
| 6 | +0.021328 | +0.002294 | +0.000388 | -0.000895 |
| 7 | +0.017951 | +0.005694 | -0.001396 | +0.000387 |

### 5.2 位置、离散度与区间

| 对比 | n | 均值 | sd | SE_run | 95% t 区间 (df=7) | 10,000 次 bootstrap 95% | 负差值 seed 数 | 符号检验 p |
|---|---|---|---|---|---|---|---|---|
| (a) none@N=1 - none@N=inf | 8 | +0.017773 | 0.003327 | 0.001176 | [+0.014992, +0.020554] | [+0.015531, +0.019893] | 0 / 8 | 0.0078125 |
| (b) none@N=20 - none@N=inf | 8 | +0.003808 | 0.001835 | 0.000649 | [+0.002274, +0.005343] | [+0.002664, +0.005025] | 0 / 8 | 0.0078125 |
| (c) after_snap@N=20 - none@N=20 | 8 | -0.001165 | 0.001392 | 0.000492 | [-0.002328, -0.000001] | [-0.002130, -0.000354] | 7 / 8 | 0.0703125 |
| (d) before_snap@N=20 - after_snap@N=20 | 8 | +0.000182 | 0.000777 | 0.000275 | [-0.000467, +0.000832] | [-0.000304, +0.000677] | 3 / 8 | 0.726562 |

校验键：contrasts.<a|b|c|d>.{per_seed,diffs,mean,sd,se_run,ci95_t,bootstrap.ci95,sign_test}。

### 5.3 敏感性：改用 csv 的 4 位小数（混合 400/800 口径）

| 对比 | 均值（csv 口径） | sd（csv 口径） | 负差值 seed 数 | 均值（公共 400 张，主口径） |
|---|---|---|---|---|
| (a) | +0.018112 | 0.002606 | 0 / 8 | +0.017773 |
| (b) | +0.004662 | 0.001364 | 0 / 8 | +0.003808 |
| (c) | -0.001037 | 0.000795 | 8 / 8 | -0.001165 |
| (d) | +0.000225 | 0.000604 | 3 / 8 | +0.000182 |

该表只说明口径更换对点估计的影响，不作为任何区间的输入。校验键：sensitivity_csv_basis。

## 6 图像层 / run 层 / 合并（验收 D，M = 400）

| 对比 | SE_run | SE_img（8 个 seed 的平均） | SE_comb | 图像层 SE 的 min ~ max |
|---|---|---|---|---|
| (a) | 0.001176 | 0.002701 | 0.002946 | 0.002566 ~ 0.002973 |
| (b) | 0.000649 | 0.001724 | 0.001842 | 0.001627 ~ 0.001858 |
| (c) | 0.000492 | 0.001103 | 0.001207 | 0.000821 ~ 0.001325 |
| (d) | 0.000275 | 0.000476 | 0.000550 | 0.000338 ~ 0.000676 |

逐 seed 的图像层 SE：

| seed | (a) | (b) | (c) | (d) |
|---|---|---|---|---|
| 0 | 0.002697 | 0.001641 | 0.001066 | 0.000662 |
| 1 | 0.002649 | 0.001632 | 0.001037 | 0.000453 |
| 2 | 0.002646 | 0.001676 | 0.001228 | 0.000343 |
| 3 | 0.002735 | 0.001751 | 0.000821 | 0.000338 |
| 4 | 0.002691 | 0.001858 | 0.001115 | 0.000424 |
| 5 | 0.002566 | 0.001765 | 0.001325 | 0.000348 |
| 6 | 0.002973 | 0.001627 | 0.000968 | 0.000567 |
| 7 | 0.002648 | 0.001843 | 0.001259 | 0.000676 |

校验键：contrasts.<id>.image_level.{per_seed_se,mean_se,min_se,max_se,combined_se}。

## 7 对比 (d) 的 TOST 与 MDE（验收 E）

被测量：d_bar = +0.000182，SE_run = 0.000275，df = 7。TOST：p_low = 1 - F_t((d_bar + Delta)/SE)，p_high = F_t((d_bar - Delta)/SE)，p_TOST = max(p_low, p_high)。

| margin Delta | t_low | p_low | t_high | p_high | p_TOST | alpha = 0.05 下判等效 |
|---|---|---|---|---|---|---|
| 0.0010 | +4.3055 | 0.00177123 | -2.9769 | 0.0103022 | 0.0103022 | 是 |
| 0.0016 | +6.4903 | 0.000168603 | -5.1617 | 0.000653527 | 0.000653527 | 是 |
| 0.0018 | +7.2185 | 8.7313e-05 | -5.8899 | 0.000302809 | 0.000302809 | 是 |
| 0.0050 | +18.8705 | 1.45868e-07 | -17.5418 | 2.40891e-07 | 2.40891e-07 | 是 |

80% 功效的最小可检出差 MDE = (t_{0.975,7} + t_{0.80,7}) * SE_run = (2.3646 + 0.8960) * 0.000275 = 0.000895。

margin = 0.0016 的来源：零基线（seed 0，utk-relu_qat-none-s0-1793011336 的 zero_baseline_mae）= 0.15526728379987373，其 1% = 0.0015526728379987374，取 4 位小数得 0.0016。
它是**预设值**，只由零基线决定，与本节观测到的 d_bar 无关，也与任何观测到的对比值无关。

校验键：tost_d.{mean,se_run,margins,mde80,zero_baseline}。

## 8 自检（验收 F）与断言原始输出

```
[IN] deploy_table_v3.csv data_rows=65 fields=snap_every,mode,seed,ema_decay,split_seed,run,val_mae_train,mae_fp32,mae_8,mae_6,mae_4
[B] series=none_Ninf  n=8 seeds=[0, 1, 2, 3, 4, 5, 6, 7]  OK
[B] series=none_N1    n=8 seeds=[0, 1, 2, 3, 4, 5, 6, 7]  OK
[B] series=none_N20   n=8 seeds=[0, 1, 2, 3, 4, 5, 6, 7]  OK
[B] series=after_N20  n=8 seeds=[0, 1, 2, 3, 4, 5, 6, 7]  OK
[B] series=before_N20 n=8 seeds=[0, 1, 2, 3, 4, 5, 6, 7]  OK
[GEOM] all 66 npz runs: n_full histogram={'400': 30, '800': 36} ; n!=400: 36 ; share common 400-row prefix: 50 (of the n!=400: 32)
[GEOM] series runs share the common 400-row prefix: 40/40 ; n_full histogram over series = {'400': 8, '800': 32}
[XCHK] 260 csv values on each run's own basis: 0 mismatches
[XCHK] 200 values on the 50 v3 rows that share the common 400-prefix: switching to the aligned 400 subset changes 124 values, max |dev| = 0.005399999999999988
[A] rebuild utk-relu_qat-none-s1-1793114048 (n_full=400): npz_round4={'mae_fp32': 0.0647, 'mae_8': 0.0648, 'mae_6': 0.0683, 'mae_4': 0.0894} expected={'mae_fp32': 0.0647, 'mae_8': 0.0648, 'mae_6': 0.0683, 'mae_4': 0.0894} val_mae_train(v2)=0.0631  OK
[A] rebuilt run rows differ from the fixed-split set: True (split_seed=1)
[A] wrote v3b: data_rows=66 (65 original bytes preserved + 1 appended)  OK
[INV] npz runs=66, v3b runs=66, v3b without npz=[], npz without v3b row=[]
[C] (a) mean=0.017773 sd=0.003327 se_run=0.001176 CI=[0.014992, 0.020554] boot=[0.015531, 0.019893] n_neg=0 sign_p=0.0078125
[C] (b) mean=0.003808 sd=0.001835 se_run=0.000649 CI=[0.002274, 0.005343] boot=[0.002664, 0.005025] n_neg=0 sign_p=0.0078125
[C] (c) mean=-0.001165 sd=0.001392 se_run=0.000492 CI=[-0.002328, -0.000001] boot=[-0.002130, -0.000354] n_neg=7 sign_p=0.0703125
[C] (d) mean=0.000182 sd=0.000777 se_run=0.000275 CI=[-0.000467, 0.000832] boot=[-0.000304, 0.000677] n_neg=3 sign_p=0.726562
[SENS] (a) csv-basis mean=0.018112 sd=0.002606 (aligned-400 mean=0.017773)
[SENS] (b) csv-basis mean=0.004662 sd=0.001364 (aligned-400 mean=0.003808)
[SENS] (c) csv-basis mean=-0.001037 sd=0.000795 (aligned-400 mean=-0.001165)
[SENS] (d) csv-basis mean=0.000225 sd=0.000604 (aligned-400 mean=0.000182)
[F] (c) seed 0 diff = -0.0025418582651764166  |diff - (-0.002542)| = 1.417348235834351e-07  <= 1e-06 -> OK
[F] (a) seed 0 diff = 0.01885177811607719  |diff - (0.018852)| = 2.218839228115832e-07  <= 1e-06 -> OK
[E] zero baseline = 0.15526728379987373 from results/utk-relu_qat-none-s0-1793011336/result.json ; 1% = 0.0015526728379987374 ; round4 = 0.0016
[E] margin=0.0010 t_low=4.3055329327518885 p_low=0.00177123 t_up=-2.9769263575346563 p_up=0.0103022 p_tost=0.0103022 equiv=True
[E] margin=0.0016 t_low=6.490270719837852 p_low=0.000168603 t_up=-5.1616641446206195 p_up=0.000653527 p_tost=0.000653527 equiv=True
[E] margin=0.0018 t_low=7.2185166488665065 p_low=8.7313e-05 t_up=-5.889910073649274 p_up=0.000302809 p_tost=0.000302809 equiv=True
[E] margin=0.0050 t_low=18.87045151332498 p_low=1.45868e-07 t_up=-17.541844938107747 p_up=2.40891e-07 p_tost=2.40891e-07 equiv=True
[E] MDE(80pct power, alpha=0.05 two-sided, df=7, SE_run=0.000275) = 0.000895
[G] ste table rows=6 unique ste seeds=[0, 1, 2] split_seed=[-1] distinct zero_baseline=3 pairable=False
```

两条参考值自检：(c) seed 0 差值 -0.0025418582651764166，参考 -0.002542，相差 1.417e-07，通过；(a) seed 0 差值 0.01885177811607719，参考 0.018852，相差 2.219e-07，通过。
参考值只有 6 位小数，实测相差约 2e-7，属参考值自身的舍入，不是不一致。

## 9 STE 臂的配对性（验收 G）

quant_runs/ste_deploy_table.csv 共 6 个数据行；ste 协议只有 3 个 seed（[0, 1, 2]），另有 in_place_overwrite、no_projection 两个单 seed 行与一个 ste_mean 汇总行。

| 行 | seed | run 的 split_seed（result.json） | zero_baseline_mae |
|---|---|---|---|
| in_place_overwrite | 0 | None | 0.15526728379987373 |
| ste | 0 | -1 | 0.15526728379987373 |
| ste | 1 | -1 | 0.15517695657840297 |
| ste | 2 | -1 | 0.1515890758312129 |
| ste_mean | 0 | -1 | 0.15526728379987373 |
| ste_mean | 1 | -1 | 0.15517695657840297 |
| ste_mean | 2 | -1 | 0.1515890758312129 |
| no_projection | 0 | -1 | 0.15526728379987373 |

结论：三个 ste seed 的零基线分别为 0.1515890758312129, 0.15517695657840297, 0.15526728379987373，彼此不同，说明验证划分随 seed 变化（这些 run 的 split_seed 记为 -1，按该版 train.py 语义就是「用 seed 建划分」）。
因此 STE 臂**不可配对**：下表只给非配对描述统计，不给任何配对检验、区间或符号检验。

| 臂 | n | 统计量 | fp32 | 8 bit | 6 bit | 4 bit |
|---|---|---|---|---|---|---|
| in_place_overwrite | 1 | mean | 0.074100 | 0.074100 | 0.078500 | 0.090000 |
| in_place_overwrite | 1 | sd | n/a | n/a | n/a | n/a |
| in_place_overwrite | 1 | se | n/a | n/a | n/a | n/a |
| in_place_overwrite | 1 | min | 0.074100 | 0.074100 | 0.078500 | 0.090000 |
| in_place_overwrite | 1 | max | 0.074100 | 0.074100 | 0.078500 | 0.090000 |
| no_projection | 1 | mean | 0.055200 | 0.054800 | 0.057500 | 0.153500 |
| no_projection | 1 | sd | n/a | n/a | n/a | n/a |
| no_projection | 1 | se | n/a | n/a | n/a | n/a |
| no_projection | 1 | min | 0.055200 | 0.054800 | 0.057500 | 0.153500 |
| no_projection | 1 | max | 0.055200 | 0.054800 | 0.057500 | 0.153500 |
| ste | 3 | mean | 0.056033 | 0.055633 | 0.057067 | 0.120767 |
| ste | 3 | sd | 0.003459 | 0.003024 | 0.002967 | 0.015232 |
| ste | 3 | se | 0.001997 | 0.001746 | 0.001713 | 0.008794 |
| ste | 3 | min | 0.052100 | 0.052200 | 0.053700 | 0.103500 |
| ste | 3 | max | 0.058600 | 0.057900 | 0.059300 | 0.132300 |
| ste_mean | 1 | mean | 0.056000 | 0.055600 | 0.057100 | 0.120800 |
| ste_mean | 1 | sd | n/a | n/a | n/a | n/a |
| ste_mean | 1 | se | n/a | n/a | n/a | n/a |
| ste_mean | 1 | min | 0.056000 | 0.055600 | 0.057100 | 0.120800 |
| ste_mean | 1 | max | 0.056000 | 0.055600 | 0.057100 | 0.120800 |

校验键：ste_arm.{rows,unique_ste_seeds,ste_split_seed_values,ste_distinct_zero_baseline,descriptive_only}。

## 10 deviations

- **perimage 的 n 不一致**：全部 66 个 run 中 30 个存 400 张、36 个存 800 张（五条序列里 8 / 32），与任务书「n = 400」不符。800 张 run 中 32 个的前 400 行与公共 400 张逐元素相同，故主统计量统一取公共 400 张（与 M = 400 要求一致）；csv 的 4 位小数在各自原始口径下仍能逐位复现（全表 260 个值、0 处不一致）。
- 因此「用 csv 4 位小数做配对」会把 400 与 800 两种口径混在一起；5.3 的敏感性表量化了这一影响，主口径不用它。
- 主统计量取 fp32 列：四个对比针对部署 FP32 任务 MAE 的间隔效应与顺序效应；8/6/4 bit 只做交叉校验。
- v3b 行序 = v3 的 65 行原样 + 末尾追加重建行，未重排，因此与 v2 的排序不同；下游按 run 列连接即可。
- 重建行的 split_seed = 1，与五条序列的固定划分不同（rows 不同），只入表、不进入配对对比。
- bootstrap 的 default_rng(0) 是每个对比各新建一个；若跨对比共用同一实例，数值会变而区间宽度基本不变。
- 符号检验中零差值不计入 n（各对比零差值个数：(a) 0, (b) 0, (c) 0, (d) 0）。
- 图像层 SE 取 8 个 seed 的算术平均（按任务书定义），不是合并方差；SE_comb 用勾股和，隐含图像源与 run 源独立。
- SE_comb 只用于量级比较，未给基于它的区间（其自由度无闭式；run 层与合并口径的 df 都取 7）。
- MDE 是 post-hoc 值：用观测到的 SE_run 反推 80% 功效可检出差，描述本次设计的灵敏度，不是先验功效计算。
- 任务书说 STE 表有 6 个 seed，实际是 6 个数据行、3 个 ste seed（0/1/2）+ 2 个单 seed 对照行 + 1 个汇总行；按实际结构报告。
- 本脚本只读 v3/v2/perimage/ste/results，未修改任何既有文件；新增文件仅 3 个产物 + 本脚本。

## 11 验收对照

| 项 | 证据 |
|---|---|
| A | 第 3 节：npz 重算 4 位 = 0.0647 / 0.0648 / 0.0683 / 0.0894；v3b 66 个数据行 |
| B | 第 2 节与第 8 节 [B] 行：五条序列 n = 8、seeds = [0..7] |
| C | 第 5 节：逐 seed 差值 8 行 x 4 列 + 均值/sd/SE/t 区间/bootstrap/符号检验 |
| D | 第 6 节：SE_run、SE_img、SE_comb 与三口径各自回答的问题 |
| E | 第 7 节：四个 margin 的 TOST p 与 80% 功效 MDE，并写明 0.0016 的来源 |
| F | 第 8 节 [F] 行：两条参考值相差约 2e-7 < 1e-6 |
| G | 第 9 节：STE 臂划分随 seed 变 -> 不可配对，只给描述统计 |

