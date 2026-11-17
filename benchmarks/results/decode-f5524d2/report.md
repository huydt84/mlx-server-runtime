# MLX Air Benchmark Report

- Run: `bench3-decode-chain`
- Status: `succeeded`
- Suite: `optimization`
- Focus: `decode`
- Server mode: `self-launched`

## Results

Lower ms is better; higher tokens/s is better. Statistics are over trial means; p95 pools individual requests. **Bold** rows are each workload's primary metric.

| model | runtime | workload | metric | trials | mean | median | min–max | CV | request p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| moe | overlap | decode_128_c1 | **tpot (ms)** | 4 | 6.612 | 5.887 | 5.884–8.791 | 21.97% | 9.818 |
| moe | overlap | decode_128_c1 | ttft (ms) | 4 | 159.363 | 159.498 | 154.592–163.863 | 2.48% | 181.861 |
| moe | overlap | decode_128_c1 | end_to_end_latency (ms) | 4 | 1039.559 | 948.365 | 943.170–1318.335 | 17.88% | 1488.043 |
| moe | overlap | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 125.675 | 134.957 | 97.084–135.701 | 15.17% | — |
| moe | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 40.986 | 40.997 | 40.939–41.009 | 0.08% | — |
| moe | overlap | mixed_c8 | ttft (ms) | 4 | 2195.561 | 2193.509 | 2171.588–2223.640 | 1.01% | 4666.835 |
| moe | overlap | mixed_c8 | tpot (ms) | 4 | 128.261 | 128.229 | 127.393–129.191 | 0.57% | 174.808 |
| moe | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 6204.852 | 6205.419 | 6198.922–6209.647 | 0.08% | 8413.280 |
| moe | serial | decode_128_c1 | **tpot (ms)** | 4 | 7.581 | 7.553 | 7.487–7.731 | 1.41% | 7.807 |
| moe | serial | decode_128_c1 | ttft (ms) | 4 | 159.831 | 159.995 | 153.187–166.147 | 3.40% | 177.653 |
| moe | serial | decode_128_c1 | end_to_end_latency (ms) | 4 | 1157.804 | 1157.177 | 1148.571–1168.289 | 0.70% | 1180.102 |
| moe | serial | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 110.549 | 110.605 | 109.554–111.435 | 0.70% | — |
| moe | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 39.963 | 40.087 | 39.535–40.142 | 0.72% | — |
| moe | serial | mixed_c8 | ttft (ms) | 4 | 2091.511 | 2071.477 | 2040.597–2182.494 | 3.13% | 4702.333 |
| moe | serial | mixed_c8 | tpot (ms) | 4 | 136.497 | 137.211 | 132.838–138.727 | 1.86% | 180.341 |
| moe | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 6353.853 | 6330.223 | 6328.605–6426.359 | 0.76% | 8702.489 |
| primary | overlap | decode_128_c1 | **tpot (ms)** | 4 | 11.386 | 11.392 | 11.349–11.409 | 0.23% | 11.564 |
| primary | overlap | decode_128_c1 | ttft (ms) | 4 | 196.065 | 195.137 | 188.471–205.516 | 3.99% | 238.502 |
| primary | overlap | decode_128_c1 | end_to_end_latency (ms) | 4 | 1683.263 | 1681.930 | 1679.749–1689.444 | 0.27% | 1727.600 |
| primary | overlap | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 76.039 | 76.099 | 75.761–76.198 | 0.27% | — |
| primary | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 23.538 | 23.544 | 23.453–23.611 | 0.28% | — |
| primary | overlap | mixed_c8 | ttft (ms) | 4 | 3865.646 | 3862.774 | 3838.075–3898.959 | 0.78% | 8389.613 |
| primary | overlap | mixed_c8 | tpot (ms) | 4 | 223.405 | 223.678 | 221.849–224.415 | 0.51% | 303.695 |
| primary | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 10822.040 | 10816.499 | 10789.959–10865.202 | 0.30% | 14597.660 |
| primary | serial | decode_128_c1 | **tpot (ms)** | 4 | 14.391 | 13.227 | 13.160–17.950 | 16.49% | 19.562 |
| primary | serial | decode_128_c1 | ttft (ms) | 4 | 204.511 | 201.278 | 188.467–227.022 | 8.34% | 251.873 |
| primary | serial | decode_128_c1 | end_to_end_latency (ms) | 4 | 2061.304 | 1908.761 | 1893.009–2534.684 | 15.31% | 2772.134 |
| primary | serial | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 63.055 | 67.056 | 50.494–67.615 | 13.29% | — |
| primary | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 20.948 | 21.143 | 18.194–23.315 | 12.68% | — |
| primary | serial | mixed_c8 | ttft (ms) | 4 | 3813.430 | 3672.439 | 3429.377–4479.464 | 12.58% | 8517.235 |
| primary | serial | mixed_c8 | tpot (ms) | 4 | 272.651 | 271.754 | 240.666–306.431 | 13.52% | 452.309 |
| primary | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 12293.280 | 12125.519 | 10918.269–14003.812 | 12.80% | 19612.947 |

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
