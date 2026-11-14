# S100 真实工具链编译与量化误差：最终证据表

生成脚本 npu_quant/summarize_final.py，只读下列产物，不写任何模型文件。

## A. 编译产物（容器自报）

| tag | .hbm 字节 | 编译耗时 | 节点数 | latency | FPS | DDR/run |
|---|---|---|---|---|---|---|
| relu_qat_s0_N1 | 6,487,152 | 395 s | 285 | 993.3 us | 1006.75 | 31.215 MB |
| relu_qat_s0_N20 | 6,487,456 | 676 s | 285 | 993 us | 1007.08 | 31.215 MB |
| gelu_s0_N1 | 6,479,024 | 2262 s | 282 | 991.3 us | 1008.81 | 31.21 MB |
| bilinear_glu_s0_N1 | 6,553,136 | 4601 s | 284 | 992.4 us | 1007.65 | 31.276 MB |

## B. 逐节点余弦：全局统计（编译器 *_quant_info.json）

| tag | n | min | min(排除 -1 异常) | p1 | median | mean | <0.99 | <0.999 | ==1 |
|---|---|---|---|---|---|---|---|---|---|
| relu_qat_s0_N1 | 285 | 0.942175 | 0.942175 | 0.950084 | 0.997250 | 0.987536 | 112 | 205 | 26 |
| relu_qat_s0_N20 | 285 | 0.933407 | 0.933407 | 0.943655 | 0.997211 | 0.988228 | 111 | 175 | 26 |
| gelu_s0_N1 | 282 | -1.000000 | 0.947076 | 0.950464 | 0.993649 | 0.979586 | 114 | 207 | 23 |
| bilinear_glu_s0_N1 | 284 | 0.932420 | 0.932420 | 0.937532 | 0.994365 | 0.986035 | 126 | 210 | 24 |

## C. 控制头作用域逐节点余弦 + BPU 指令算子（advice.json 的 op_type）

### relu_qat_s0_N1

| 节点 | 图类型 | 余弦 | BPU op_type |
|---|---|---|---|
| /head/net/net.0/Gemm_transpose_in_reshape | Reshape | 0.998975 | - |
| /head/net/net.0/Gemm | Conv | 0.999375 | b30.conv2d |
| /head/net/net.1/Pow | Pow | 0.997010 | b30.lut |
| /head/net/net.1/ReduceMean | ReduceMean | 1.000000 | b30.reduce |
| /head/net/net.1/Div_reciprocal | HzRsqrt | 1.000000 | b30.lut |
| /head/net/net.1/Mul | Mul | 0.999286 | b30.binary_eltwise, b30.binary_eltwise |
| /head/net/net.1/Mul_1 | Mul | 0.999281 | b30.binary_eltwise, b30.binary_eltwise |
| /head/net/net.2/Relu | Relu | 0.999405 | b30.lut |
| /head/net/net.3/Gemm | Conv | 1.000000 | b30.conv2d |
| /head/net/net.3/Gemm_reshape_output | Reshape | 1.000000 | hbdk.reshape, b30vpu.dequantize |

- 头部 LUT 指令数（b30.lut）：**3**（/head/net/net.1/Pow, /head/net/net.1/Div_reciprocal, /head/net/net.2/Relu）
- 全模型 b30.lut 指令数：16
- 头前 FPN 归一化（三头共有）：Reshape 0.978332；InstanceNormalization_mean1 0.999047；InstanceNormalization_sub1 0.977561；InstanceNormalization_mul1 0.969821；InstanceNormalization_mean2 0.999927；InstanceNormalization_reciprocal 0.999966

### relu_qat_s0_N20

| 节点 | 图类型 | 余弦 | BPU op_type |
|---|---|---|---|
| /head/net/net.0/Gemm_transpose_in_reshape | Reshape | 0.999703 | - |
| /head/net/net.0/Gemm | Conv | 0.999918 | b30.conv2d |
| /head/net/net.1/Pow | Pow | 0.999821 | b30.lut |
| /head/net/net.1/ReduceMean | ReduceMean | 1.000000 | b30.reduce |
| /head/net/net.1/Div_reciprocal | HzRsqrt | 1.000000 | b30.lut |
| /head/net/net.1/Mul | Mul | 0.999907 | b30.binary_eltwise, b30.binary_eltwise |
| /head/net/net.1/Mul_1 | Mul | 0.999899 | b30.binary_eltwise, b30.binary_eltwise |
| /head/net/net.2/Relu | Relu | 0.999864 | b30.lut |
| /head/net/net.3/Gemm | Conv | 1.000000 | b30.conv2d |
| /head/net/net.3/Gemm_reshape_output | Reshape | 1.000000 | hbdk.reshape, b30vpu.dequantize |

- 头部 LUT 指令数（b30.lut）：**3**（/head/net/net.1/Pow, /head/net/net.1/Div_reciprocal, /head/net/net.2/Relu）
- 全模型 b30.lut 指令数：16
- 头前 FPN 归一化（三头共有）：Reshape 0.980374；InstanceNormalization_mean1 0.998012；InstanceNormalization_sub1 0.979753；InstanceNormalization_mul1 0.974600；InstanceNormalization_mean2 0.999849；InstanceNormalization_reciprocal 0.999953

### gelu_s0_N1

| 节点 | 图类型 | 余弦 | BPU op_type |
|---|---|---|---|
| /head/net/net.0/Gemm_transpose_in_reshape | Reshape | 0.993686 | - |
| /head/net/net.0/Gemm | Conv | 0.990574 | b30.conv2d |
| /head/net/net.2/Tanh | Tanh | 0.990622 | b30.lut |
| /head/net/net.2/Mul | Mul | 0.990657 | b30.binary_eltwise, b30.binary_eltwise |
| /head/net/net.3/Mul_1 | HzGelu | 0.991085 | b30.lut |
| /head/net/net.4/Gemm | Conv | 1.000000 | b30.conv2d |
| /head/net/net.4/Gemm_reshape_output | Reshape | 1.000000 | hbdk.reshape, b30vpu.dequantize |

- 头部 LUT 指令数（b30.lut）：**2**（/head/net/net.2/Tanh, /head/net/net.3/Mul_1）
- 全模型 b30.lut 指令数：15
- 头前 FPN 归一化（三头共有）：Reshape 0.976962；InstanceNormalization_mean1 0.997327；InstanceNormalization_sub1 0.976278；InstanceNormalization_mul1 0.969400；InstanceNormalization_mean2 0.999953；InstanceNormalization_reciprocal 0.999975

### bilinear_glu_s0_N1

| 节点 | 图类型 | 余弦 | BPU op_type |
|---|---|---|---|
| /head/proj/Gemm_transpose_in_reshape | Reshape | 0.992762 | - |
| /head/proj/Gemm | Conv | 0.991130 | b30.conv2d |
| /head/proj/Gemm_reshape_output | Reshape | 0.991130 | hbdk.reshape |
| /head/Slice | Slice | 0.992816 | hbdk.slice |
| /head/Slice_1 | Slice | 0.989507 | hbdk.slice |
| /head/Mul_2 | Mul | 0.983052 | b30.binary_eltwise, b30.binary_eltwise |
| /head/out/Gemm_transpose_in_reshape | Reshape | 0.983052 | hbdk.reshape |
| /head/out/Gemm | Conv | 1.000000 | b30.conv2d |
| /head/out/Gemm_reshape_output | Reshape | 1.000000 | hbdk.reshape, b30vpu.dequantize |

- 头部 LUT 指令数（b30.lut）：**0**
- 全模型 b30.lut 指令数：13
- 头前 FPN 归一化（三头共有）：Reshape 0.980677；InstanceNormalization_mean1 0.998967；InstanceNormalization_sub1 0.980091；InstanceNormalization_mul1 0.973647；InstanceNormalization_mean2 0.999843；InstanceNormalization_reciprocal 0.999956

## D. MAE（同一 200 张测试图，与校准集不相交；target = age/100）

| tag | FP32 (x86 onnxruntime) | 伪量化 (hmct) | nash 运行时 (x86 实测) | dMAE 伪量化 | dMAE nash |
|---|---|---|---|---|---|
| relu_qat_s0_N1| 0.075599 | 0.079808 | 0.079433 (n=200) | +0.004209 | +0.003834 |
| relu_qat_s0_N20| 0.060487 | 0.062922 | 0.063871 (n=199) | +0.002435 | +0.003384 |
| gelu_s0_N1| 0.079308 | 0.078453 | 0.079295 (n=199) | -0.000855 | -0.000014 |
| bilinear_glu_s0_N1| 0.076025 | 0.078032 | 0.080258 (n=200) | +0.002007 | +0.004233 |

