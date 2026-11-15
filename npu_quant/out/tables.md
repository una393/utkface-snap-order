## 编译产物

| tag | .hbm bytes | 编译耗时 s | exit | latency µs | FPS | DDR bytes |
|---|---|---|---|---|---|---|
| relu_qat_s0_N1 | 6487152 | 395 | 0 | 993.3 | 1006.75 | 32731520 |
| relu_qat_s0_N20 | 6487456 | 676 | 0 | 993 | 1007.08 | 32731392 |
| gelu_s0_N1 | 6479024 | 2262 | 0 | 991.3 | 1008.81 | 32726144 |
| bilinear_glu_s0_N1 | None | None | None | None | None | None |

## 逐节点余弦：全局

| tag | nodes | min | p1 | p5 | median | mean | <0.99 | <0.999 | ==1 |
|---|---|---|---|---|---|---|---|---|---|
| relu_qat_s0_N1 | 285 | 0.942175 | 0.949735 | 0.958962 | 0.997250 | 0.987536 | 112 | 205 | 26 |
| relu_qat_s0_N20 | 285 | 0.933407 | 0.942874 | 0.958006 | 0.997211 | 0.988228 | 111 | 175 | 26 |
| gelu_s0_N1 | 282 | -1.000000 | 0.949888 | 0.957866 | 0.993649 | 0.979586 | 114 | 207 | 23 |
| bilinear_glu_s0_N1 | 284 | 0.932420 | 0.937488 | 0.956537 | 0.994365 | 0.986035 | 126 | 210 | 24 |

## 逐节点余弦：控制头 + 头前 /fpn/norm（按后缀对齐）

| node | type | relu_qat_s0_N1 | relu_qat_s0_N20 | gelu_s0_N1 | bilinear_glu_s0_N1 |
|---|---|---|---|---|---|
| /fpn/norm/Add | Add | 0.976734 | 0.979040 | 0.974644 | 0.979354 |
| /fpn/norm/InstanceNormalization_add2 | Add | 0.976909 | 0.979052 | 0.975192 | 0.979353 |
| /fpn/norm/InstanceNormalization_mean1 | ReduceMean | 0.999047 | 0.998012 | 0.997327 | 0.998967 |
| /fpn/norm/InstanceNormalization_mean2 | ReduceMean | 0.999927 | 0.999849 | 0.999953 | 0.999843 |
| /fpn/norm/InstanceNormalization_mul1 | Mul | 0.969821 | 0.974600 | 0.969400 | 0.973647 |
| /fpn/norm/InstanceNormalization_mul2 | Mul | 0.976905 | 0.979053 | 0.975190 | 0.979359 |
| /fpn/norm/InstanceNormalization_mul3 | Mul | 0.976909 | 0.979052 | 0.975192 | 0.979353 |
| /fpn/norm/InstanceNormalization_reciprocal | HzRsqrt | 0.999966 | 0.999953 | 0.999975 | 0.999956 |
| /fpn/norm/InstanceNormalization_sub1 | Sub | 0.977561 | 0.979753 | 0.976278 | 0.980091 |
| /fpn/norm/Mul | Mul | 0.976762 | 0.979041 | 0.974860 | 0.979391 |
| /fpn/norm/Reshape | Reshape | 0.978332 | 0.980374 | 0.976962 | 0.980677 |
| /fpn/norm/Reshape_1 | Reshape | 0.976909 | 0.979052 | 0.975192 | 0.979353 |
| /head/net/net.0/Gemm | Conv | 0.999375 | 0.999918 | 0.990574 | - |
| /head/net/net.0/Gemm_transpose_in_reshape | Reshape | 0.998975 | 0.999703 | 0.993686 | - |
| /head/net/net.1/Div_reciprocal | HzRsqrt | 1.000000 | 1.000000 | - | - |
| /head/net/net.1/Mul | Mul | 0.999286 | 0.999907 | - | - |
| /head/net/net.1/Mul_1 | Mul | 0.999281 | 0.999899 | - | - |
| /head/net/net.1/Pow | Pow | 0.997010 | 0.999821 | - | - |
| /head/net/net.1/ReduceMean | ReduceMean | 1.000000 | 1.000000 | - | - |
| /head/net/net.2/Mul | Mul | - | - | 0.990657 | - |
| /head/net/net.2/Relu | Relu | 0.999405 | 0.999864 | - | - |
| /head/net/net.2/Tanh | Tanh | - | - | 0.990622 | - |
| /head/net/net.3/Gemm | Conv | 1.000000 | 1.000000 | - | - |
| /head/net/net.3/Gemm_reshape_output | Reshape | 1.000000 | 1.000000 | - | - |
| /head/net/net.3/Mul_1 | HzGelu | - | - | 0.991085 | - |
| /head/net/net.4/Gemm | Conv | - | - | 1.000000 | - |
| /head/net/net.4/Gemm_reshape_output | Reshape | - | - | 1.000000 | - |

## MAE（同一 200 张测试图，seed 0 划分的 val[800:1000]）

| tag | FP32 ONNX MAE (x86 宿主) | fake-quant MAE (编译器模拟) | nash runtime MAE (x86 实测) |
|---|---|---|---|
| relu_qat_s0_N1 | 0.075599 | 0.079808 | 0.079433 |
| relu_qat_s0_N20 | 0.060487 | 0.062922 | 0.063871 |
| gelu_s0_N1 | 0.079308 | 0.078453 | 0.079295 |
| bilinear_glu_s0_N1 | 0.076025 | - | - |

## 配对差（量化 − FP32，同图同序）

| tag | fake-quant Δmean ± SE | nash Δmean ± SE | fake-quant ΔMAE ± SE | nash ΔMAE ± SE |
|---|---|---|---|---|
| relu_qat_s0_N1 | n=200 +0.01771 ± 0.00185 | n=200 +0.01710 ± 0.00187 | +0.004209 ± 0.001955 | +0.003834 ± 0.001919 |
| relu_qat_s0_N20 | n=200 +0.00213 ± 0.00157 | n=199 +0.00259 ± 0.00156 | +0.002435 ± 0.001493 | +0.003368 ± 0.001430 |
| gelu_s0_N1 | n=200 +0.00037 ± 0.00201 | n=199 +0.00098 ± 0.00197 | -0.000855 ± 0.001851 | -0.000157 ± 0.001821 |
| bilinear_glu_s0_N1 | - | - | - | - |
