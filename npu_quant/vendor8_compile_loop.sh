#!/bin/bash
# compile-only stream: wait for the in-flight s1_Ninf compile, then compile the rest serially
set -u
cd /c/Users/24399/Desktop/paper/paper-utkface || exit 1
LOG=npu_quant/out/vendor8_logs; IMAGE=ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0
HOSTDIR=C:/Users/24399/Desktop/paper/paper-utkface; T=quant_runs/vendor8_timings.csv
n=0
while [ ! -f "npu_quant/out/vendor8_s1_Ninf/vendor8_s1_Ninf.hbm" ]; do sleep 20; n=$((n+1)); [ $n -gt 90 ] && break; done
echo "[compile] s1_Ninf in-flight finished"
for tag in vendor8_s1_N1 vendor8_s2_Ninf vendor8_s2_N1; do
  [ -f "npu_quant/out/$tag/$tag.hbm" ] && { echo "[done] $tag"; continue; }
  t0=$(date +%s)
  MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /bin/sh -v "$HOSTDIR":/work \
    -w /work/npu_quant/out/vendor8_logs "$IMAGE" \
    -c "python3 /work/npu_quant/hb_compile_patched.py -c /work/npu_quant/hb_config/$tag.yaml" \
    > "$LOG/compile_$tag.log" 2>&1
  rc=$?; t1=$(date +%s)
  echo "compile,$tag,$t0,$t1,$((t1-t0)),$rc" >> "$T"
  echo "[compile] $tag rc=$rc $((t1-t0))s"
done
echo "COMPILELOOP_DONE" >> "$LOG/pipeline_status.txt"
