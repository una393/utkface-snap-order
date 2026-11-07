# 两条外推轴的 8-seed 端点重算（KonIQ-10k 数据轴 / ResNet-18 架构轴）
2026-11-02。d = MAE(N=1) − MAE(N=∞)，**正值 = 关闭网格投影更好**（与部署表 Δ 的符号约定相反，见 §3）。

## 1 队列与台账
- `logs/_ext_queue_done.txt` 最终 **48 行 = data 24 + arch 24**（14:26 时 6/5），两轴 **18:06:37 同刻满 24**，早于 20:30 截止 2 h 24 m；实测 9–11 min/run，无 run 崩溃。
- `logs/_ext_analysis.sh` 在 18:07–18:14 自行跑过同一组命令；本报告的 csv 由我随后重跑的四条覆盖（同命令、同输入、确定性），runlevel 于 18:15:36 重跑。
- 旧 2-seed 的 n=2 表备份在 `logs/_ext8_old2_reference.md`，旧 csv 备份在 `logs/_ext8_oldcsv/`。

## 2 部署表自查（闸门 1）
| csv | 数据行 | seeds | n 列 | 缺格 / 重复格 / NaN |
|---|---|---|---|---|
| ext_deploy_koniq.csv | 32 | 0–7 | 400 | 0 / 0 / 0 |
| ext_deploy_koniq_heldout.csv | 32 | 0–7 | 400 | 0 / 0 / 0 |
| ext_deploy_arch.csv | 32 | 0–7 | 400 | 0 / 0 / 0 |
| ext_deploy_arch_heldout.csv | 32 | 0–7 | 1556 | 0 / 0 / 0 |
- 32 行 = 8 seed × 2 协议 × 2 端点，逐格唯一；表头与旧文件逐列同序（`logs/_ext8_check.py` 断言 `header_same_as_old=True` ×4）。**无缺口。**

## 3 四行表（闸门 2：n=8、df=7、t=2.365）
口径：FP32；每 seed 的 d = 两协议 (none, after_snap) 平均；符号检验 = 精确二项双侧（n=8 的地板 p = 2/256 = 0.007812）。
| 轴 | 子集 | n | 均值 | 95 % CI | d>0 / d<0 | 符号检验 p | 旧 2-seed | 新 − 旧 |
|---|---|---|---|---|---|---|---|---|
| data | first400 | 8 | +0.000743 | [+0.000056, +0.001430] | 8 / 0 | 0.007812 | +0.001076 | −0.000333 |
| data | heldout | 8 | +0.000700 | [+0.000067, +0.001334] | 8 / 0 | 0.007812 | +0.001020 | −0.000320 |
| arch | first400 | 8 | +0.007214 | [+0.005782, +0.008646] | 8 / 0 | 0.007812 | +0.006841 | +0.000373 |
| arch | heldout | 8 | +0.007112 | [+0.006168, +0.008056] | 8 / 0 | 0.007812 | +0.006344 | +0.000768 |
- sd 依次 0.000822 / 0.000758 / 0.001713 / 0.001129。data heldout 逐 seed d：+0.001953 +0.000213 +0.001827 +0.000077 +0.000505 +0.000159 +0.000631 +0.000238。
- 旧值来源：data heldout、arch first400 取任务书给定值；其余两格取本次运行前的 n=2 产物（`logs/_ext8_old2_reference.md`）。**无符号翻转、无 3 倍以上量级差（比值 0.69 / 0.69 / 1.05 / 1.12），无需标红。**

## 4 观察
端点读数（每格 = 8 seed × 2 协议直接池化的均值差 d，与 §3 的逐 seed 平均差 ≤ 1e-6）：
| 轴/子集 | FP32 | 8 bit | 6 bit | 4 bit |
|---|---|---|---|---|
| data first400 | +0.000743 | +0.000701 | +0.000539 | −0.001177 |
| data heldout | +0.000701 | +0.000650 | +0.000453 | −0.001168 |
| arch first400 | +0.007214 | +0.007256 | +0.009205 | +0.042402 |
| arch heldout | +0.007112 | +0.007135 | +0.009205 | +0.041281 |
- **8 bit 与 FP32 同量级**（两轴差 ≤ 7e-5），与「8 bit 导出近似空操作」一致。
- **数据轴相对效应更大**：N=1 → N=∞ 为 0.002689 → 0.001946（−27.6 %，first400）、0.002642 → 0.001942（−26.5 %，heldout）；架构轴 0.057952 → 0.050738（−12.5 %）、0.056332 → 0.049220（−12.6 %）。绝对差 10 倍来自任务基数，相对收益数据轴约 2.2 倍。
- **6 bit 两轴反向偏离 FP32**：数据轴减半（逐协议 5/3、4/4，区间跨 0），架构轴放大（+0.009205，8/0 ×4，CI [+0.0083, +0.0119]）。
- **4 bit 只有架构轴可读**：+0.042402 / +0.041281，8/0 ×4，约为 8 bit 的 5.8 倍；数据轴翻负（−0.001177 / −0.001168）但 sd 0.0128、逐协议符号 3/5、4/4、2/6、4/4 混杂 ⇒ 4 bit 由 run 间变异主导，不能据此比较协议。

## 5 限制
- 每轴 8 个 seed 同时充当 `split_seed`（csv 里 `split_seed == seed`），seed 方差混合划分方差与初始化方差，本设计不能分别归因。
- 两轴子集大小不同（data heldout 400 张、arch heldout 1556 张），CI 宽度不可跨轴直接比较；四行表把两协议按 seed 平均（协议内不独立），只作描述，确认性检验见 runlevel md 的按协议 n=8 对比。
- n=8 符号检验地板 p = 0.007812，任何 8/0 都停在此值，不能读作「远小于 0.008」；4 bit 的 t 区间宽达 ±0.01（data）/±0.013（arch），方向之外无信息。

## 6 复现命令（原样）
```
PY=C:/Users/24399/.conda/envs/e2e/python.exe
$PY scripts/ext_deploy.py --runs 'results/koniq-*' --dataset koniq --selection-windows 1110 --subset first400 --out quant_runs/ext_deploy_koniq.csv
$PY scripts/ext_deploy.py --runs 'results/koniq-*' --dataset koniq --selection-windows 1110 --subset heldout --out quant_runs/ext_deploy_koniq_heldout.csv
$PY scripts/ext_deploy.py --runs 'results/utk-*-resnet-*' --selection-windows 2000 --subset first400 --out quant_runs/ext_deploy_arch.csv
$PY scripts/ext_deploy.py --runs 'results/utk-*-resnet-*' --selection-windows 2000 --subset heldout --out quant_runs/ext_deploy_arch_heldout.csv
$PY scripts/round20_ext_runlevel.py --axis data
$PY scripts/round20_ext_runlevel.py --axis arch
$PY logs/_ext8_stats.py     # 四行表 + 符号检验 + 旧 2-seed 对照
$PY logs/_ext8_levels.py    # §4 端点读数
$PY logs/_ext8_check.py     # 32 行 / seed 覆盖 / 表头自查
```
