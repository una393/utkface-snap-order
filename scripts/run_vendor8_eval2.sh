#!/bin/bash
set -u
cd /c/Users/USER/Desktop/paper/paper-utkface
IMG=ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0
export MSYS_NO_PATHCONV=1
while read -r tag; do
  [ -z "$tag" ] && continue
  full="vendor8_$tag"
  echo "=== $full start $(date +%H:%M:%S)"
  docker run --rm --entrypoint /bin/sh -v C:/Users/USER/Desktop/paper/paper-utkface:/work \
    -w /work "$IMG" -c "python3 /work/npu_quant/vendor8_eval.py $full 200 12" \
    > "logs/_v8_eval2_$tag.log" 2>&1
  echo "    rc=$? $(date +%H:%M:%S)  $(tail -1 logs/_v8_eval2_$tag.log | cut -c1-60)"
done < logs/_v8_todo.txt
echo "EVAL DONE $(date)"
