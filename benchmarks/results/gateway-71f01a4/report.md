# MLX Air Benchmark Report

- Run: `bench2-gateway-base71`
- Status: `succeeded`
- Suite: `optimization`
- Focus: `gateway`
- Server mode: `self-launched`

## Results

Lower ms is better; higher tokens/s is better. Statistics are over trial means; p95 pools individual requests. **Bold** rows are each workload's primary metric.

| model | runtime | workload | metric | trials | mean | median | min–max | CV | request p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| primary | overlap | interactive_stream_c1 | **ttft (ms)** | 4 | 177.971 | 177.157 | 173.925–183.645 | 2.44% | 193.214 |
| primary | overlap | interactive_stream_c1 | tpot (ms) | 4 | 11.514 | 11.461 | 11.418–11.715 | 1.19% | 11.933 |
| primary | overlap | interactive_stream_c1 | end_to_end_latency (ms) | 4 | 577.110 | 576.925 | 574.400–580.191 | 0.53% | 583.773 |
| primary | overlap | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 55.441 | 55.459 | 55.146–55.702 | 0.53% | — |
| primary | overlap | short_nonstream_c1 | **end_to_end_latency (ms)** | 4 | 573.492 | 572.968 | 566.482–581.552 | 1.16% | 585.567 |
| primary | overlap | short_nonstream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 55.795 | 55.842 | 55.016–56.479 | 1.16% | — |
| primary | serial | interactive_stream_c1 | **ttft (ms)** | 4 | 179.538 | 179.480 | 178.027–181.165 | 0.82% | 193.618 |
| primary | serial | interactive_stream_c1 | tpot (ms) | 4 | 13.516 | 13.459 | 13.340–13.807 | 1.53% | 14.361 |
| primary | serial | interactive_stream_c1 | end_to_end_latency (ms) | 4 | 638.269 | 637.981 | 631.268–645.845 | 0.95% | 683.578 |
| primary | serial | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 50.132 | 50.152 | 49.540–50.685 | 0.95% | — |
| primary | serial | short_nonstream_c1 | **end_to_end_latency (ms)** | 4 | 638.839 | 638.474 | 627.124–651.285 | 1.66% | 681.313 |
| primary | serial | short_nonstream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 50.093 | 50.114 | 49.126–51.019 | 1.66% | — |
| runtime_control | overlap | interactive_stream_c1 | **ttft (ms)** | 4 | 151.507 | 150.804 | 143.830–160.589 | 4.70% | 175.141 |
| runtime_control | overlap | interactive_stream_c1 | tpot (ms) | 4 | 3.283 | 3.312 | 3.092–3.416 | 4.16% | 3.731 |
| runtime_control | overlap | interactive_stream_c1 | end_to_end_latency (ms) | 4 | 292.933 | 292.920 | 275.877–310.014 | 4.79% | 328.006 |
| runtime_control | overlap | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 109.395 | 109.217 | 103.190–115.956 | 4.80% | — |
| runtime_control | overlap | short_nonstream_c1 | **end_to_end_latency (ms)** | 4 | 284.342 | 284.451 | 279.689–288.779 | 1.79% | 330.246 |
| runtime_control | overlap | short_nonstream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 112.527 | 112.484 | 110.766–114.374 | 1.80% | — |
| runtime_control | serial | interactive_stream_c1 | **ttft (ms)** | 4 | 158.223 | 155.443 | 149.118–172.888 | 6.58% | 184.369 |
| runtime_control | serial | interactive_stream_c1 | tpot (ms) | 4 | 4.392 | 4.301 | 4.134–4.830 | 7.05% | 5.077 |
| runtime_control | serial | interactive_stream_c1 | end_to_end_latency (ms) | 4 | 328.706 | 321.521 | 318.363–353.417 | 5.03% | 380.624 |
| runtime_control | serial | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 97.501 | 99.501 | 90.523–100.480 | 4.80% | — |
| runtime_control | serial | short_nonstream_c1 | **end_to_end_latency (ms)** | 4 | 324.102 | 323.662 | 321.456–327.629 | 0.91% | 352.922 |
| runtime_control | serial | short_nonstream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 98.713 | 98.844 | 97.644–99.519 | 0.91% | — |

## Trials

| round | model | runtime | workload | trial | load mode | concurrency | succeeded | errors |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | primary | serial | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | primary | serial | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | primary | serial | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | primary | serial | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | primary | overlap | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | primary | overlap | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | primary | overlap | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | primary | overlap | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | overlap | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | overlap | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | overlap | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | overlap | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | serial | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | serial | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | serial | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | serial | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | primary | overlap | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | primary | overlap | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | primary | overlap | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | primary | overlap | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | primary | serial | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | primary | serial | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | primary | serial | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | primary | serial | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | serial | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | serial | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | serial | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | serial | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | overlap | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | overlap | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | overlap | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | overlap | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
