# MLX Air Benchmark Report

- Run: `bench2-scheduler-base71`
- Status: `succeeded`
- Suite: `optimization`
- Focus: `scheduler`
- Server mode: `self-launched`

## Results

Lower ms is better; higher tokens/s is better. Statistics are over trial means; p95 pools individual requests. **Bold** rows are each workload's primary metric.

| model | runtime | workload | metric | trials | mean | median | min–max | CV | request p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| moe | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 37.283 | 37.323 | 36.262–38.224 | 2.74% | — |
| moe | overlap | mixed_c8 | ttft (ms) | 4 | 2268.036 | 2276.474 | 2101.994–2417.201 | 5.71% | 4875.523 |
| moe | overlap | mixed_c8 | tpot (ms) | 4 | 146.183 | 145.916 | 141.550–151.349 | 2.75% | 194.773 |
| moe | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 6827.119 | 6820.683 | 6655.454–7011.657 | 2.73% | 9389.745 |
| moe | overlap | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 141.814 | 141.947 | 136.405–146.957 | 3.88% | — |
| moe | overlap | steady_short_c4 | ttft (ms) | 4 | 386.639 | 385.321 | 378.719–397.196 | 2.02% | 439.050 |
| moe | overlap | steady_short_c4 | tpot (ms) | 4 | 15.326 | 15.095 | 14.519–16.596 | 6.47% | 15.791 |
| moe | overlap | steady_short_c4 | end_to_end_latency (ms) | 4 | 901.045 | 902.229 | 862.967–936.755 | 4.13% | 982.303 |
| moe | overlap | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 141.238 | 142.866 | 130.530–148.692 | 6.05% | — |
| moe | overlap | steady_short_c8 | ttft (ms) | 4 | 677.192 | 680.491 | 641.491–706.297 | 3.99% | 996.679 |
| moe | overlap | steady_short_c8 | tpot (ms) | 4 | 35.618 | 35.094 | 33.172–39.112 | 7.71% | 44.571 |
| moe | overlap | steady_short_c8 | end_to_end_latency (ms) | 4 | 1811.019 | 1790.302 | 1713.579–1949.892 | 6.10% | 2122.972 |
| moe | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 37.507 | 37.251 | 36.831–38.695 | 2.35% | — |
| moe | serial | mixed_c8 | ttft (ms) | 4 | 2178.083 | 2200.191 | 2051.016–2260.935 | 4.59% | 4970.690 |
| moe | serial | mixed_c8 | tpot (ms) | 4 | 147.237 | 146.449 | 143.657–152.393 | 2.73% | 192.883 |
| moe | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 6773.095 | 6816.668 | 6561.843–6897.202 | 2.35% | 9370.942 |
| moe | serial | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 135.330 | 136.458 | 131.157–137.246 | 2.12% | — |
| moe | serial | steady_short_c4 | ttft (ms) | 4 | 388.382 | 385.995 | 382.596–398.943 | 1.99% | 416.867 |
| moe | serial | steady_short_c4 | tpot (ms) | 4 | 16.998 | 16.949 | 16.601–17.494 | 2.27% | 18.137 |
| moe | serial | steady_short_c4 | end_to_end_latency (ms) | 4 | 943.364 | 935.059 | 932.151–971.188 | 1.98% | 991.093 |
| moe | serial | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 129.781 | 130.173 | 124.582–134.195 | 3.79% | — |
| moe | serial | steady_short_c8 | ttft (ms) | 4 | 517.992 | 521.765 | 488.607–539.828 | 4.20% | 920.817 |
| moe | serial | steady_short_c8 | tpot (ms) | 4 | 45.627 | 45.729 | 42.856–48.196 | 6.23% | 58.286 |
| moe | serial | steady_short_c8 | end_to_end_latency (ms) | 4 | 1961.314 | 1953.091 | 1896.886–2042.186 | 3.73% | 2454.728 |
| primary | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 21.230 | 21.220 | 20.449–22.031 | 3.99% | — |
| primary | overlap | mixed_c8 | ttft (ms) | 4 | 3994.213 | 3997.360 | 3812.008–4170.123 | 3.78% | 9081.554 |
| primary | overlap | mixed_c8 | tpot (ms) | 4 | 257.884 | 259.084 | 244.650–268.717 | 4.51% | 341.702 |
| primary | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 12018.347 | 12022.369 | 11564.271–12464.381 | 4.01% | 16705.530 |
| primary | overlap | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 87.579 | 87.413 | 86.482–89.006 | 1.20% | — |
| primary | overlap | steady_short_c4 | ttft (ms) | 4 | 602.050 | 597.539 | 591.791–621.330 | 2.21% | 631.053 |
| primary | overlap | steady_short_c4 | tpot (ms) | 4 | 26.665 | 26.565 | 26.456–27.074 | 1.06% | 28.110 |
| primary | overlap | steady_short_c4 | end_to_end_latency (ms) | 4 | 1460.921 | 1464.149 | 1435.450–1479.938 | 1.27% | 1534.154 |
| primary | overlap | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 89.223 | 89.244 | 86.965–91.440 | 2.10% | — |
| primary | overlap | steady_short_c8 | ttft (ms) | 4 | 691.024 | 668.349 | 627.385–800.011 | 10.96% | 1261.281 |
| primary | overlap | steady_short_c8 | tpot (ms) | 4 | 68.733 | 68.108 | 67.477–71.237 | 2.51% | 81.237 |
| primary | overlap | steady_short_c8 | end_to_end_latency (ms) | 4 | 2851.616 | 2848.698 | 2785.309–2923.759 | 2.05% | 3434.311 |
| primary | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 22.173 | 22.093 | 21.475–23.029 | 2.94% | — |
| primary | serial | mixed_c8 | ttft (ms) | 4 | 3761.902 | 3758.332 | 3580.757–3950.187 | 4.24% | 8706.587 |
| primary | serial | mixed_c8 | tpot (ms) | 4 | 248.255 | 249.504 | 240.131–253.883 | 2.34% | 326.308 |
| primary | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 11488.338 | 11522.355 | 11056.165–11852.477 | 2.91% | 15893.234 |
| primary | serial | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 85.071 | 85.354 | 82.871–86.705 | 2.17% | — |
| primary | serial | steady_short_c4 | ttft (ms) | 4 | 607.507 | 609.438 | 574.060–637.091 | 4.99% | 727.419 |
| primary | serial | steady_short_c4 | tpot (ms) | 4 | 27.908 | 27.951 | 27.585–28.146 | 0.84% | 28.798 |
| primary | serial | steady_short_c4 | end_to_end_latency (ms) | 4 | 1503.969 | 1497.704 | 1476.044–1544.424 | 2.15% | 1607.167 |
| primary | serial | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 88.704 | 88.414 | 87.934–90.051 | 1.13% | — |
| primary | serial | steady_short_c8 | ttft (ms) | 4 | 610.724 | 615.419 | 579.303–632.758 | 3.69% | 1236.479 |
| primary | serial | steady_short_c8 | tpot (ms) | 4 | 71.701 | 72.388 | 69.486–72.541 | 2.07% | 83.869 |
| primary | serial | steady_short_c8 | end_to_end_latency (ms) | 4 | 2861.900 | 2869.039 | 2819.093–2890.431 | 1.11% | 3459.924 |

## Trials

| round | model | runtime | workload | trial | load mode | concurrency | succeeded | errors |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | primary | serial | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | primary | serial | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | primary | serial | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | primary | serial | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | primary | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | primary | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | primary | overlap | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | primary | overlap | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | primary | overlap | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | primary | overlap | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | primary | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | primary | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | moe | overlap | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | moe | overlap | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | moe | overlap | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | moe | overlap | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | moe | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | moe | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | moe | serial | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | moe | serial | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | moe | serial | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | moe | serial | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | moe | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | moe | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | primary | overlap | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | primary | overlap | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | primary | overlap | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | primary | overlap | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | primary | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | primary | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | primary | serial | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | primary | serial | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | primary | serial | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | primary | serial | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | primary | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | primary | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | moe | serial | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | moe | serial | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | moe | serial | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | moe | serial | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | moe | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | moe | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | moe | overlap | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | moe | overlap | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | moe | overlap | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | moe | overlap | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | moe | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | moe | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
