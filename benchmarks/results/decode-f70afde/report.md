# MLX Air Benchmark Report

- Run: `bench-decode-baseline`
- Status: `succeeded`
- Suite: `optimization`
- Focus: `decode`
- Server mode: `self-launched`

## Results

Lower ms is better; higher tokens/s is better. Statistics are over trial means; p95 pools individual requests. **Bold** rows are each workload's primary metric.

| model | runtime | workload | metric | trials | mean | median | min–max | CV | request p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| moe | overlap | decode_128_c1 | **tpot (ms)** | 4 | 18.948 | 18.930 | 18.793–19.140 | 0.87% | 19.408 |
| moe | overlap | decode_128_c1 | ttft (ms) | 4 | 183.253 | 183.330 | 179.204–187.149 | 2.13% | 205.309 |
| moe | overlap | decode_128_c1 | end_to_end_latency (ms) | 4 | 2626.570 | 2625.480 | 2591.331–2663.989 | 1.37% | 2708.530 |
| moe | overlap | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 48.738 | 48.756 | 48.047–49.394 | 1.37% | — |
| moe | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 36.336 | 36.332 | 36.265–36.413 | 0.17% | — |
| moe | overlap | mixed_c8 | ttft (ms) | 4 | 2185.822 | 2183.268 | 2170.436–2206.317 | 0.72% | 4968.852 |
| moe | overlap | mixed_c8 | tpot (ms) | 4 | 153.176 | 153.461 | 152.257–153.526 | 0.40% | 197.040 |
| moe | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 6963.530 | 6962.593 | 6950.875–6978.061 | 0.17% | 9610.774 |
| moe | serial | decode_128_c1 | **tpot (ms)** | 4 | 18.893 | 18.846 | 18.760–19.119 | 0.85% | 19.169 |
| moe | serial | decode_128_c1 | ttft (ms) | 4 | 180.446 | 180.558 | 175.386–185.284 | 2.85% | 193.937 |
| moe | serial | decode_128_c1 | end_to_end_latency (ms) | 4 | 2618.776 | 2615.754 | 2587.969–2655.628 | 1.07% | 2673.515 |
| moe | serial | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 48.881 | 48.933 | 48.198–49.458 | 1.07% | — |
| moe | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 36.242 | 36.284 | 36.109–36.292 | 0.25% | — |
| moe | serial | mixed_c8 | ttft (ms) | 4 | 2157.866 | 2160.449 | 2108.541–2202.025 | 2.13% | 4982.651 |
| moe | serial | mixed_c8 | tpot (ms) | 4 | 154.242 | 153.948 | 152.484–156.590 | 1.21% | 197.268 |
| moe | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 6971.708 | 6966.182 | 6957.350–6997.117 | 0.25% | 9585.554 |
| primary | overlap | decode_128_c1 | **tpot (ms)** | 4 | 14.518 | 14.514 | 14.466–14.577 | 0.41% | 14.624 |
| primary | overlap | decode_128_c1 | ttft (ms) | 4 | 193.987 | 195.663 | 184.367–200.255 | 3.65% | 228.568 |
| primary | overlap | decode_128_c1 | end_to_end_latency (ms) | 4 | 2082.659 | 2079.473 | 2063.913–2107.778 | 0.89% | 2133.737 |
| primary | overlap | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 61.461 | 61.552 | 60.725–62.016 | 0.88% | — |
| primary | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 21.170 | 21.197 | 20.521–21.765 | 3.12% | — |
| primary | overlap | mixed_c8 | ttft (ms) | 4 | 3712.317 | 3697.922 | 3613.509–3839.917 | 2.53% | 8551.626 |
| primary | overlap | mixed_c8 | tpot (ms) | 4 | 266.864 | 265.429 | 257.421–279.179 | 3.89% | 345.675 |
| primary | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 12018.457 | 12000.095 | 11682.012–12391.626 | 3.13% | 16311.859 |
| primary | serial | decode_128_c1 | **tpot (ms)** | 4 | 14.621 | 14.623 | 14.595–14.643 | 0.14% | 14.690 |
| primary | serial | decode_128_c1 | ttft (ms) | 4 | 191.576 | 191.724 | 186.151–196.707 | 2.68% | 236.300 |
| primary | serial | decode_128_c1 | end_to_end_latency (ms) | 4 | 2073.788 | 2072.061 | 2063.410–2087.618 | 0.56% | 2119.084 |
| primary | serial | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 61.722 | 61.773 | 61.311–62.031 | 0.56% | — |
| primary | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 21.597 | 21.611 | 21.509–21.659 | 0.32% | — |
| primary | serial | mixed_c8 | ttft (ms) | 4 | 3553.148 | 3517.492 | 3498.041–3679.567 | 2.39% | 8595.037 |
| primary | serial | mixed_c8 | tpot (ms) | 4 | 263.747 | 264.814 | 258.965–266.395 | 1.25% | 333.372 |
| primary | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 11760.305 | 11754.509 | 11726.844–11805.355 | 0.30% | 15503.931 |

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
