#!/bin/bash
# eval-only stream: consume .hbm files in pair order, one tag at a time, 12 workers
set -u
cd /c/Users/USER/Desktop/paper/paper-utkface || exit 1
LOG=npu_quant/out/vendor8_logs; IMAGE=ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0
HOSTDIR=C:/Users/USER/Desktop/paper/paper-utkface; T=quant_runs/vendor8_timings.csv
TAGS="vendor8_s0_N1 vendor8_s1_Ninf vendor8_s1_N1 vendor8_s2_Ninf vendor8_s2_N1"
for tag in $TAGS; do
  n=0
  while [ ! -f "npu_quant/out/$tag/$tag.hbm" ]; do sleep 15; n=$((n+1)); [ $n -gt 200 ] && break; done
  if [ ! -f "npu_quant/out/$tag/$tag.hbm" ]; then echo "[skip] $tag"; continue; fi
  if [ -f "npu_quant/out/vendor8_int8_$tag.json" ]; then echo "[done] $tag"; continue; fi
  t0=$(date +%s)
  MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /bin/sh -v "$HOSTDIR":/work -w /work "$IMAGE" \
    -c "python3 /work/npu_quant/vendor8_eval.py $tag 200 12" > "$LOG/eval_$tag.log" 2>&1
  rc=$?; t1=$(date +%s)
  echo "eval,$tag,$t0,$t1,$((t1-t0)),$rc" >> "$T"
  echo "[eval] $tag rc=$rc $((t1-t0))s"
done
echo "EVALLOOP_DONE" >> "$LOG/pipeline_status.txt"
