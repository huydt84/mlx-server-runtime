# MLX Air Benchmark Report

- Run: `bench2-scheduler-chain`
- Status: `succeeded`
- Suite: `optimization`
- Focus: `scheduler`
- Server mode: `self-launched`

## Results

Lower ms is better; higher tokens/s is better. Statistics are over trial means; p95 pools individual requests. **Bold** rows are each workload's primary metric.

| model | runtime | workload | metric | trials | mean | median | min–max | CV | request p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| moe | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 38.696 | 38.896 | 37.554–39.439 | 2.10% | — |
| moe | overlap | mixed_c8 | ttft (ms) | 4 | 2316.512 | 2317.599 | 2202.152–2428.697 | 4.14% | 4875.870 |
| moe | overlap | mixed_c8 | tpot (ms) | 4 | 136.344 | 135.722 | 134.628–139.304 | 1.51% | 184.521 |
| moe | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 6573.598 | 6535.066 | 6450.102–6774.160 | 2.14% | 9052.754 |
| moe | overlap | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 144.676 | 145.339 | 141.700–146.326 | 1.41% | — |
| moe | overlap | steady_short_c4 | ttft (ms) | 4 | 394.557 | 391.488 | 389.568–405.683 | 1.92% | 430.526 |
| moe | overlap | steady_short_c4 | tpot (ms) | 4 | 14.620 | 14.548 | 14.461–14.923 | 1.42% | 14.986 |
| moe | overlap | steady_short_c4 | end_to_end_latency (ms) | 4 | 883.565 | 878.586 | 874.072–903.018 | 1.52% | 932.681 |
| moe | overlap | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 147.362 | 148.028 | 143.687–149.706 | 1.94% | — |
| moe | overlap | steady_short_c8 | ttft (ms) | 4 | 655.289 | 651.508 | 633.577–684.563 | 3.45% | 947.113 |
| moe | overlap | steady_short_c8 | tpot (ms) | 4 | 33.709 | 33.641 | 33.333–34.220 | 1.10% | 38.296 |
| moe | overlap | steady_short_c8 | end_to_end_latency (ms) | 4 | 1730.173 | 1721.318 | 1701.525–1776.530 | 1.99% | 1802.565 |
| moe | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 38.256 | 38.283 | 37.186–39.273 | 2.33% | — |
| moe | serial | mixed_c8 | ttft (ms) | 4 | 2146.612 | 2164.305 | 2017.038–2240.799 | 4.62% | 4787.442 |
| moe | serial | mixed_c8 | tpot (ms) | 4 | 144.042 | 142.805 | 142.336–148.221 | 1.94% | 187.730 |
| moe | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 6644.236 | 6636.885 | 6469.711–6833.463 | 2.33% | 9461.739 |
| moe | serial | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 135.547 | 135.218 | 134.821–136.931 | 0.70% | — |
| moe | serial | steady_short_c4 | ttft (ms) | 4 | 397.216 | 398.401 | 392.914–399.148 | 0.73% | 422.417 |
| moe | serial | steady_short_c4 | tpot (ms) | 4 | 16.680 | 16.670 | 16.556–16.825 | 0.66% | 17.597 |
| moe | serial | steady_short_c4 | end_to_end_latency (ms) | 4 | 943.179 | 945.478 | 934.301–947.461 | 0.64% | 990.919 |
| moe | serial | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 132.754 | 132.954 | 130.966–134.142 | 1.12% | — |
| moe | serial | steady_short_c8 | ttft (ms) | 4 | 514.625 | 516.656 | 457.999–567.187 | 11.24% | 910.252 |
| moe | serial | steady_short_c8 | tpot (ms) | 4 | 44.253 | 44.247 | 41.994–46.525 | 4.49% | 54.156 |
| moe | serial | steady_short_c8 | end_to_end_latency (ms) | 4 | 1914.444 | 1911.324 | 1893.040–1942.087 | 1.16% | 2360.054 |
| primary | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 21.910 | 21.905 | 21.708–22.122 | 1.04% | — |
| primary | overlap | mixed_c8 | ttft (ms) | 4 | 4142.293 | 4118.361 | 4080.869–4251.584 | 1.92% | 9136.514 |
| primary | overlap | mixed_c8 | tpot (ms) | 4 | 240.505 | 239.859 | 238.864–243.436 | 0.85% | 321.710 |
| primary | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 11630.260 | 11632.982 | 11519.115–11735.962 | 1.04% | 15550.171 |
| primary | overlap | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 88.679 | 88.401 | 87.351–90.563 | 1.58% | — |
| primary | overlap | steady_short_c4 | ttft (ms) | 4 | 602.185 | 602.569 | 587.629–615.972 | 1.96% | 644.449 |
| primary | overlap | steady_short_c4 | tpot (ms) | 4 | 26.190 | 26.133 | 25.826–26.665 | 1.33% | 26.770 |
| primary | overlap | steady_short_c4 | end_to_end_latency (ms) | 4 | 1441.278 | 1445.948 | 1412.889–1460.326 | 1.43% | 1498.151 |
| primary | overlap | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 91.214 | 91.578 | 89.041–92.659 | 1.82% | — |
| primary | overlap | steady_short_c8 | ttft (ms) | 4 | 798.509 | 800.365 | 685.971–907.337 | 11.58% | 1228.829 |
| primary | overlap | steady_short_c8 | tpot (ms) | 4 | 63.290 | 63.817 | 60.073–65.453 | 3.73% | 75.652 |
| primary | overlap | steady_short_c8 | end_to_end_latency (ms) | 4 | 2789.808 | 2778.187 | 2746.666–2856.192 | 1.79% | 3262.496 |
| primary | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 21.455 | 21.579 | 20.914–21.749 | 1.82% | — |
| primary | serial | mixed_c8 | ttft (ms) | 4 | 3916.739 | 3935.056 | 3779.784–4017.061 | 3.05% | 9110.323 |
| primary | serial | mixed_c8 | tpot (ms) | 4 | 255.438 | 253.861 | 252.182–261.847 | 1.75% | 335.663 |
| primary | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 11866.882 | 11799.427 | 11698.409–12170.264 | 1.84% | 16496.446 |
| primary | serial | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 83.951 | 82.978 | 81.019–88.828 | 4.03% | — |
| primary | serial | steady_short_c4 | ttft (ms) | 4 | 603.467 | 605.588 | 555.326–647.365 | 6.30% | 657.173 |
| primary | serial | steady_short_c4 | tpot (ms) | 4 | 28.644 | 28.942 | 27.550–29.140 | 2.57% | 29.902 |
| primary | serial | steady_short_c4 | end_to_end_latency (ms) | 4 | 1521.656 | 1533.496 | 1440.605–1579.029 | 3.82% | 1597.433 |
| primary | serial | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 89.270 | 89.317 | 88.856–89.591 | 0.36% | — |
| primary | serial | steady_short_c8 | ttft (ms) | 4 | 608.784 | 604.966 | 599.298–625.905 | 1.94% | 1235.008 |
| primary | serial | steady_short_c8 | tpot (ms) | 4 | 71.189 | 71.033 | 70.987–71.704 | 0.48% | 82.605 |
| primary | serial | steady_short_c8 | end_to_end_latency (ms) | 4 | 2844.253 | 2843.169 | 2834.508–2856.165 | 0.39% | 3433.030 |

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
