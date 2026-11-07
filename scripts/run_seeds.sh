#!/usr/bin/env bash
# 补 seed：三个控制头 × seed 1,2。已完成自动跳过，可随时重入。
set -u
PY=/c/Users/24399/.conda/envs/e2e/python.exe
LOG=/d/data/utk_seeds.log
COMMON="--raw data/utkface_raw224.npy --ema-mode after_snap --epochs 20 --batch-size 64 --amp --grad-clip 1.0 --ema-decay 0.99 --val-windows 2000"

for seed in 1 2; do
  for head in gelu relu_qat bilinear_glu; do
    if $PY scripts/done.py --head "$head" --ema after_snap --seed "$seed" >/dev/null 2>&1; then
      continue
    fi
    echo "[$(date +%F' '%T)] START $head seed=$seed" >> "$LOG"
    $PY -u train.py $COMMON --head "$head" --seed "$seed" >> "$LOG" 2>&1
    echo "[$(date +%F' '%T)] DONE $head seed=$seed exit=$?" >> "$LOG"
  done
done
echo "[$(date +%F' '%T)] ALL DONE" >> "$LOG"
