# MLX Air Benchmark Report

- Run: `bench2-gateway-chain`
- Status: `succeeded`
- Suite: `optimization`
- Focus: `gateway`
- Server mode: `self-launched`

## Results

Lower ms is better; higher tokens/s is better. Statistics are over trial means; p95 pools individual requests. **Bold** rows are each workload's primary metric.

| model | runtime | workload | metric | trials | mean | median | min–max | CV | request p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| primary | overlap | interactive_stream_c1 | **ttft (ms)** | 4 | 178.213 | 176.562 | 173.540–186.190 | 3.24% | 195.463 |
| primary | overlap | interactive_stream_c1 | tpot (ms) | 4 | 11.795 | 11.781 | 11.477–12.142 | 3.07% | 12.564 |
| primary | overlap | interactive_stream_c1 | end_to_end_latency (ms) | 4 | 581.661 | 582.454 | 574.073–587.661 | 0.99% | 629.835 |
| primary | overlap | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 55.010 | 54.932 | 54.445–55.733 | 0.99% | — |
| primary | overlap | short_nonstream_c1 | **end_to_end_latency (ms)** | 4 | 575.081 | 575.344 | 572.755–576.881 | 0.31% | 585.241 |
| primary | overlap | short_nonstream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 55.635 | 55.609 | 55.461–55.860 | 0.31% | — |
| primary | serial | interactive_stream_c1 | **ttft (ms)** | 4 | 182.840 | 182.136 | 177.441–189.647 | 2.95% | 199.319 |
| primary | serial | interactive_stream_c1 | tpot (ms) | 4 | 13.635 | 13.680 | 13.162–14.018 | 2.79% | 14.309 |
| primary | serial | interactive_stream_c1 | end_to_end_latency (ms) | 4 | 640.675 | 637.934 | 625.888–660.942 | 2.49% | 685.501 |
| primary | serial | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 49.961 | 50.158 | 48.409–51.120 | 2.48% | — |
| primary | serial | short_nonstream_c1 | **end_to_end_latency (ms)** | 4 | 644.371 | 640.916 | 635.148–660.503 | 1.87% | 685.493 |
| primary | serial | short_nonstream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 49.666 | 49.924 | 48.441–50.374 | 1.85% | — |
| runtime_control | overlap | interactive_stream_c1 | **ttft (ms)** | 4 | 140.056 | 141.170 | 130.038–147.848 | 5.27% | 160.851 |
| runtime_control | overlap | interactive_stream_c1 | tpot (ms) | 4 | 2.943 | 2.931 | 2.851–3.059 | 3.17% | 3.180 |
| runtime_control | overlap | interactive_stream_c1 | end_to_end_latency (ms) | 4 | 269.083 | 270.025 | 263.916–272.367 | 1.35% | 278.887 |
| runtime_control | overlap | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 118.901 | 118.470 | 117.452–121.212 | 1.36% | — |
| runtime_control | overlap | short_nonstream_c1 | **end_to_end_latency (ms)** | 4 | 272.448 | 271.318 | 269.615–277.542 | 1.29% | 280.050 |
| runtime_control | overlap | short_nonstream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 117.429 | 117.903 | 115.260–118.648 | 1.27% | — |
| runtime_control | serial | interactive_stream_c1 | **ttft (ms)** | 4 | 158.794 | 158.299 | 151.055–167.524 | 4.25% | 175.737 |
| runtime_control | serial | interactive_stream_c1 | tpot (ms) | 4 | 4.305 | 4.248 | 4.221–4.503 | 3.08% | 4.909 |
| runtime_control | serial | interactive_stream_c1 | end_to_end_latency (ms) | 4 | 327.520 | 328.874 | 320.272–332.059 | 1.68% | 372.414 |
| runtime_control | serial | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 97.699 | 97.282 | 96.343–99.889 | 1.69% | — |
| runtime_control | serial | short_nonstream_c1 | **end_to_end_latency (ms)** | 4 | 318.070 | 317.192 | 316.398–321.497 | 0.75% | 330.208 |
| runtime_control | serial | short_nonstream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 100.582 | 100.857 | 99.505–101.109 | 0.74% | — |

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
