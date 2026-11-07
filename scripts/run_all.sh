#!/usr/bin/env bash
# UTKFace 全组别：H1/H2/H3 + A/B（C 与 H2 同配置，复用）。日志 /d/data/utk_run.log
set -u
PY=/c/Users/24399/.conda/envs/e2e/python.exe
LOG=/d/data/utk_run.log
COMMON="--raw data/utkface_raw224.npy --epochs 20 --batch-size 64 --amp --grad-clip 1.0 --ema-decay 0.99 --val-windows 2000"

run() {
  local name=$1; shift
  echo "[$(date +%F' '%T)] START $name" >> "$LOG"
  $PY -u train.py $COMMON "$@" >> "$LOG" 2>&1
  echo "[$(date +%F' '%T)] DONE $name exit=$?" >> "$LOG"
}

run H1 --head gelu         --ema-mode after_snap
run H2 --head relu_qat     --ema-mode after_snap
run H3 --head bilinear_glu --ema-mode after_snap
run A  --head relu_qat     --ema-mode none
run B  --head relu_qat     --ema-mode before_snap

echo "[$(date +%F' '%T)] ALL DONE" >> "$LOG"
