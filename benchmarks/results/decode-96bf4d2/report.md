# MLX Air Benchmark Report

- Run: `bench3-decode-gap`
- Status: `succeeded`
- Suite: `optimization`
- Focus: `decode`
- Server mode: `self-launched`

## Results

Lower ms is better; higher tokens/s is better. Statistics are over trial means; p95 pools individual requests. **Bold** rows are each workload's primary metric.

| model | runtime | workload | metric | trials | mean | median | min–max | CV | request p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| moe | overlap | decode_128_c1 | **tpot (ms)** | 4 | 5.798 | 5.791 | 5.783–5.825 | 0.33% | 5.875 |
| moe | overlap | decode_128_c1 | ttft (ms) | 4 | 175.126 | 176.104 | 169.063–179.233 | 2.57% | 191.530 |
| moe | overlap | decode_128_c1 | end_to_end_latency (ms) | 4 | 952.984 | 951.882 | 951.069–957.105 | 0.29% | 964.367 |
| moe | overlap | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 134.304 | 134.459 | 133.724–134.573 | 0.29% | — |
| moe | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 41.062 | 41.228 | 40.546–41.246 | 0.84% | — |
| moe | overlap | mixed_c8 | ttft (ms) | 4 | 2224.954 | 2222.762 | 2183.582–2270.712 | 1.97% | 4618.700 |
| moe | overlap | mixed_c8 | tpot (ms) | 4 | 126.935 | 127.283 | 124.673–128.501 | 1.28% | 173.296 |
| moe | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 6193.132 | 6169.869 | 6161.794–6270.995 | 0.84% | 8601.151 |
| moe | serial | decode_128_c1 | **tpot (ms)** | 4 | 7.423 | 7.430 | 7.397–7.436 | 0.24% | 7.489 |
| moe | serial | decode_128_c1 | ttft (ms) | 4 | 172.739 | 172.434 | 168.559–177.529 | 2.14% | 195.918 |
| moe | serial | decode_128_c1 | end_to_end_latency (ms) | 4 | 1159.725 | 1159.956 | 1154.130–1164.858 | 0.49% | 1178.590 |
| moe | serial | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 110.364 | 110.343 | 109.875–110.898 | 0.50% | — |
| moe | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 40.362 | 40.357 | 40.264–40.469 | 0.21% | — |
| moe | serial | mixed_c8 | ttft (ms) | 4 | 2118.470 | 2119.791 | 2056.695–2177.604 | 2.35% | 4687.399 |
| moe | serial | mixed_c8 | tpot (ms) | 4 | 133.406 | 133.252 | 131.840–135.279 | 1.06% | 177.249 |
| moe | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 6288.642 | 6288.463 | 6276.968–6300.674 | 0.16% | 8661.832 |
| primary | overlap | decode_128_c1 | **tpot (ms)** | 4 | 11.035 | 11.037 | 11.022–11.046 | 0.09% | 11.093 |
| primary | overlap | decode_128_c1 | ttft (ms) | 4 | 204.556 | 202.024 | 200.525–213.650 | 3.02% | 247.371 |
| primary | overlap | decode_128_c1 | end_to_end_latency (ms) | 4 | 1640.297 | 1639.449 | 1631.095–1651.197 | 0.52% | 1681.171 |
| primary | overlap | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 78.032 | 78.071 | 77.516–78.471 | 0.52% | — |
| primary | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 23.702 | 23.692 | 23.658–23.766 | 0.22% | — |
| primary | overlap | mixed_c8 | ttft (ms) | 4 | 3829.327 | 3846.430 | 3756.737–3867.710 | 1.30% | 8314.957 |
| primary | overlap | mixed_c8 | tpot (ms) | 4 | 222.227 | 221.794 | 220.063–225.255 | 1.00% | 302.566 |
| primary | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 10750.536 | 10754.698 | 10723.266–10769.482 | 0.20% | 14374.426 |
| primary | serial | decode_128_c1 | **tpot (ms)** | 4 | 13.396 | 13.217 | 12.938–14.212 | 4.30% | 14.711 |
| primary | serial | decode_128_c1 | ttft (ms) | 4 | 210.311 | 212.467 | 194.512–221.798 | 5.43% | 251.994 |
| primary | serial | decode_128_c1 | end_to_end_latency (ms) | 4 | 1957.471 | 1932.550 | 1900.537–2064.249 | 3.73% | 2120.164 |
| primary | serial | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 65.454 | 66.232 | 62.005–67.346 | 3.62% | — |
| primary | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 23.330 | 23.333 | 23.160–23.491 | 0.59% | — |
| primary | serial | mixed_c8 | ttft (ms) | 4 | 3603.539 | 3571.767 | 3554.291–3716.329 | 2.13% | 8332.073 |
| primary | serial | mixed_c8 | tpot (ms) | 4 | 234.622 | 234.107 | 233.490–236.785 | 0.64% | 306.225 |
| primary | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 10908.741 | 10910.017 | 10831.681–10983.251 | 0.58% | 14741.836 |

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
