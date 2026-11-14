#!/bin/bash
# vendor8 v2: pipelined but resource-partitioned.
#   eval(t)      -> cores 4-15, 12 workers  (the nash simulator is memory-bandwidth hungry)
#   compile(t+1) -> cores 0-3               (hb_compile is thread-hungry but tolerant)
# v1 ran 2 compiles + 1 eval on 16 cores and both stages collapsed (~4x slowdown).
set -u
cd /c/Users/USER/Desktop/paper/paper-utkface || exit 1
LOG=npu_quant/out/vendor8_logs
mkdir -p "$LOG"
IMAGE=ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0
HOSTDIR=C:/Users/USER/Desktop/paper/paper-utkface
T=quant_runs/vendor8_timings.csv
[ -f "$T" ] || echo "stage,tag,start_epoch,end_epoch,elapsed_s,rc" > "$T"
STOP_EPOCH=${STOP_EPOCH:-0}

TAGS="vendor8_s0_Ninf vendor8_s0_N1 vendor8_s1_Ninf vendor8_s1_N1 vendor8_s2_Ninf vendor8_s2_N1 vendor8_s3_Ninf vendor8_s3_N1 vendor8_s4_Ninf vendor8_s4_N1 vendor8_s5_Ninf vendor8_s5_N1 vendor8_s6_Ninf vendor8_s6_N1 vendor8_s7_Ninf vendor8_s7_N1"
arr=($TAGS); N=${#arr[@]}

compile_tag() {
  local tag=$1 t0 t1 rc
  t0=$(date +%s)
  MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /bin/sh \
    -v "$HOSTDIR":/work -w /work/npu_quant/out/vendor8_logs "$IMAGE" \
    -c "python3 /work/npu_quant/hb_compile_patched.py -c /work/npu_quant/hb_config/$tag.yaml" \
    > "$LOG/compile_$tag.log" 2>&1
  rc=$?; t1=$(date +%s)
  echo "compile,$tag,$t0,$t1,$((t1-t0)),$rc" >> "$T"
  echo "[compile] $tag rc=$rc $((t1-t0))s"
}

eval_tag() {
  local tag=$1 t0 t1 rc
  t0=$(date +%s)
  MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /bin/sh \
    -v "$HOSTDIR":/work -w /work "$IMAGE" \
    -c "python3 /work/npu_quant/vendor8_eval.py $tag 200 15" \
    > "$LOG/eval_$tag.log" 2>&1
  rc=$?; t1=$(date +%s)
  echo "eval,$tag,$t0,$t1,$((t1-t0)),$rc" >> "$T"
  echo "[eval] $tag rc=$rc $((t1-t0))s"
}

CP=""
for ((i=0; i<N; i++)); do
  t=${arr[i]}
  now=$(date +%s)
  if [ "$STOP_EPOCH" -gt 0 ] && [ "$now" -ge "$STOP_EPOCH" ]; then
    echo "[stop] deadline reached before $t"; break
  fi
  if [ ! -f "npu_quant/out/$t/$t.hbm" ]; then
    compile_tag "$t"
  fi
  if [ ! -f "npu_quant/out/$t/$t.hbm" ]; then
    echo "[skip] $t has no hbm"; continue
  fi
  nx=""
  for ((j=i+1; j<N; j++)); do
    c=${arr[j]}
    if [ ! -f "npu_quant/out/$c/$c.hbm" ]; then nx=$c; break; fi
  done
  eval_tag "$t" & EP=$!
  CP=""
  if [ -n "$nx" ] && [ ! -f "npu_quant/out/$nx/$nx.hbm" ]; then
    compile_tag "$nx" & CP=$!
  fi
  wait "$EP"
  if [ -n "$CP" ]; then wait "$CP"; fi
done
echo "PIPELINE2_DONE" >> "$LOG/pipeline_status.txt"
echo "[pipeline2] done"
