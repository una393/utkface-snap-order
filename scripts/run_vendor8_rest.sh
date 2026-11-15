#!/bin/bash
# finish the vendor8 grid: compile then eval, one checkpoint at a time (serial, no overlap)
set -u
cd /c/Users/USER/Desktop/paper/paper-utkface
IMG=ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0
export MSYS_NO_PATHCONV=1
while read -r tag; do
  [ -z "$tag" ] && continue
  echo "=== $tag start $(date +%H:%M:%S)"
  docker run --rm --entrypoint /bin/sh -v C:/Users/USER/Desktop/paper/paper-utkface:/work \
    -w /work/npu_quant/out/vendor8_logs "$IMG" \
    -c "python3 /work/npu_quant/hb_compile_patched.py -c /work/npu_quant/hb_config/vendor8_$tag.yaml" \
    > "logs/_v8_compile_$tag.log" 2>&1
  rc=$?
  echo "    compile rc=$rc"
  if [ "$rc" -ne 0 ]; then echo "    SKIP $tag (compile failed)"; continue; fi
  docker run --rm --entrypoint /bin/sh -v C:/Users/USER/Desktop/paper/paper-utkface:/work \
    -w /work "$IMG" -c "python3 /work/npu_quant/vendor8_eval.py $tag 200 12" \
    > "logs/_v8_eval_$tag.log" 2>&1
  echo "    eval rc=$? $(date +%H:%M:%S)"
done < logs/_v8_todo.txt
echo "ALL DONE $(date)"
