# MLX Air Benchmark Report

- Run: `bench-decode-candidate`
- Status: `succeeded`
- Suite: `optimization`
- Focus: `decode`
- Server mode: `self-launched`

## Results

Lower ms is better; higher tokens/s is better. Statistics are over trial means; p95 pools individual requests. **Bold** rows are each workload's primary metric.

| model | runtime | workload | metric | trials | mean | median | min–max | CV | request p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| moe | overlap | decode_128_c1 | **tpot (ms)** | 4 | 5.884 | 5.886 | 5.870–5.896 | 0.21% | 5.961 |
| moe | overlap | decode_128_c1 | ttft (ms) | 4 | 160.903 | 160.503 | 156.641–165.963 | 2.43% | 177.919 |
| moe | overlap | decode_128_c1 | end_to_end_latency (ms) | 4 | 950.595 | 950.670 | 945.901–955.140 | 0.40% | 962.024 |
| moe | overlap | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 134.642 | 134.631 | 134.000–135.309 | 0.40% | — |
| moe | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 40.542 | 40.554 | 40.438–40.621 | 0.20% | — |
| moe | overlap | mixed_c8 | ttft (ms) | 4 | 2165.654 | 2182.127 | 2100.293–2198.070 | 2.04% | 4693.845 |
| moe | overlap | mixed_c8 | tpot (ms) | 4 | 131.506 | 131.018 | 130.425–133.564 | 1.08% | 176.942 |
| moe | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 6273.316 | 6269.922 | 6268.126–6285.296 | 0.13% | 8718.880 |
| moe | serial | decode_128_c1 | **tpot (ms)** | 4 | 7.626 | 7.632 | 7.452–7.787 | 2.43% | 7.885 |
| moe | serial | decode_128_c1 | ttft (ms) | 4 | 157.197 | 156.525 | 151.911–163.826 | 3.20% | 173.640 |
| moe | serial | decode_128_c1 | end_to_end_latency (ms) | 4 | 1158.329 | 1154.278 | 1137.072–1187.686 | 1.86% | 1195.071 |
| moe | serial | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 110.525 | 110.887 | 107.765–112.562 | 1.85% | — |
| moe | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 39.891 | 40.039 | 39.242–40.245 | 1.18% | — |
| moe | serial | mixed_c8 | ttft (ms) | 4 | 2137.053 | 2131.202 | 2061.787–2224.022 | 3.15% | 4708.772 |
| moe | serial | mixed_c8 | tpot (ms) | 4 | 135.260 | 135.642 | 130.554–139.201 | 2.63% | 178.966 |
| moe | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 6363.076 | 6340.348 | 6306.501–6465.105 | 1.18% | 8762.820 |
| primary | overlap | decode_128_c1 | **tpot (ms)** | 4 | 11.400 | 11.399 | 11.384–11.419 | 0.13% | 11.432 |
| primary | overlap | decode_128_c1 | ttft (ms) | 4 | 196.886 | 194.651 | 189.765–208.477 | 4.11% | 243.507 |
| primary | overlap | decode_128_c1 | end_to_end_latency (ms) | 4 | 1682.253 | 1679.908 | 1669.460–1699.735 | 0.83% | 1724.793 |
| primary | overlap | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 76.089 | 76.192 | 75.302–76.668 | 0.82% | — |
| primary | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 23.576 | 23.574 | 23.543–23.616 | 0.13% | — |
| primary | overlap | mixed_c8 | ttft (ms) | 4 | 3586.068 | 3590.797 | 3482.107–3680.574 | 2.27% | 8382.053 |
| primary | overlap | mixed_c8 | tpot (ms) | 4 | 231.838 | 231.903 | 228.883–234.662 | 1.02% | 305.299 |
| primary | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 10804.382 | 10808.193 | 10784.216–10816.927 | 0.13% | 14752.479 |
| primary | serial | decode_128_c1 | **tpot (ms)** | 4 | 13.129 | 13.132 | 13.085–13.169 | 0.26% | 13.217 |
| primary | serial | decode_128_c1 | ttft (ms) | 4 | 198.658 | 198.904 | 197.805–199.018 | 0.29% | 235.541 |
| primary | serial | decode_128_c1 | end_to_end_latency (ms) | 4 | 1904.127 | 1901.817 | 1896.901–1915.974 | 0.43% | 1944.795 |
| primary | serial | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 67.220 | 67.301 | 66.804–67.476 | 0.43% | — |
| primary | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 23.405 | 23.400 | 23.392–23.429 | 0.07% | — |
| primary | serial | mixed_c8 | ttft (ms) | 4 | 3571.746 | 3595.091 | 3411.129–3685.674 | 3.26% | 8392.508 |
| primary | serial | mixed_c8 | tpot (ms) | 4 | 234.515 | 233.728 | 231.098–239.506 | 1.54% | 306.580 |
| primary | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 10873.207 | 10872.071 | 10866.667–10882.019 | 0.07% | 14758.301 |

## Trials

| round | model | runtime | workload | trial | load mode | concurrency | succeeded | errors |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | primary | serial | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | primary | serial | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | primary | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | primary | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | primary | overlap | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | primary | overlap | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | primary | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | primary | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | moe | overlap | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | moe | overlap | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | moe | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | moe | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | moe | serial | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | moe | serial | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | moe | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | moe | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | primary | overlap | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | primary | overlap | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | primary | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | primary | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | primary | serial | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | primary | serial | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | primary | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | primary | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | moe | serial | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | moe | serial | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | moe | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | moe | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | moe | overlap | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | moe | overlap | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | moe | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | moe | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
