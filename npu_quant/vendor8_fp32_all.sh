set -u
cd /c/Users/USER/Desktop/paper/paper-utkface || exit 1
LOG=npu_quant/out/vendor8_logs
TAGS="vendor8_s0_Ninf vendor8_s0_N1 vendor8_s1_N1 vendor8_s1_Ninf vendor8_s2_N1 vendor8_s2_Ninf vendor8_s3_N1 vendor8_s3_Ninf vendor8_s4_N1 vendor8_s4_Ninf vendor8_s5_N1 vendor8_s5_Ninf vendor8_s6_N1 vendor8_s6_Ninf vendor8_s7_N1 vendor8_s7_Ninf"
n=0
for t in $TAGS; do
  C:/Users/USER/.conda/envs/e2e/python.exe npu_quant/vendor8_fp32.py "$t" > "$LOG/fp32_$t.log" 2>&1 &
  n=$((n+1))
  if [ $((n % 2)) -eq 0 ]; then wait; fi
done
wait
echo FP32_ALL_DONE
