# MLX Air Benchmark Report

- Run: `run-base`
- Status: `succeeded`
- Suite: `packed_prefill`
- Focus: `None`
- Server mode: `self-launched`

## Results

Lower ms is better; higher tokens/s is better. Statistics are over trial means; p95 pools individual requests. **Bold** rows are each workload's primary metric.

| model | runtime | workload | metric | trials | mean | median | min–max | CV | request p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| packed_small | packed | cold_prefill_2k_c1 | **ttft (ms)** | 6 | 17820.039 | 17819.868 | 17719.821–17953.459 | 0.49% | 17971.967 |
| packed_small | packed | cold_prefill_2k_c1 | tpot (ms) | 6 | 33.120 | 33.118 | 33.026–33.187 | 0.19% | 33.308 |
| packed_small | packed | cold_prefill_2k_c1 | end_to_end_latency (ms) | 6 | 18352.114 | 18358.979 | 18243.129–18477.889 | 0.47% | 18497.491 |
| packed_small | packed | cold_prefill_2k_c1 | aggregate_output_tokens_per_second (tokens/s) | 6 | 0.872 | 0.872 | 0.866–0.877 | 0.47% | — |
| packed_small | packed | decode_128_c1 | **tpot (ms)** | 6 | 31.730 | 31.738 | 31.704–31.746 | 0.06% | 31.780 |
| packed_small | packed | decode_128_c1 | ttft (ms) | 6 | 533.774 | 513.642 | 503.684–600.429 | 7.45% | 648.745 |
| packed_small | packed | decode_128_c1 | end_to_end_latency (ms) | 6 | 4602.892 | 4583.027 | 4560.351–4665.874 | 0.95% | 4735.987 |
| packed_small | packed | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 6 | 27.810 | 27.929 | 27.433–28.067 | 0.95% | — |
| packed_small | packed | interactive_stream_c1 | **ttft (ms)** | 6 | 477.527 | 461.891 | 457.486–538.827 | 6.65% | 586.186 |
| packed_small | packed | interactive_stream_c1 | tpot (ms) | 6 | 31.561 | 31.568 | 31.485–31.606 | 0.14% | 31.686 |
| packed_small | packed | interactive_stream_c1 | end_to_end_latency (ms) | 6 | 1491.813 | 1477.442 | 1470.374–1551.527 | 2.12% | 1592.749 |
| packed_small | packed | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 6 | 21.457 | 21.658 | 20.623–21.762 | 2.07% | — |

## Trials

| round | model | runtime | workload | trial | load mode | concurrency | succeeded | errors |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | packed_small | packed | cold_prefill_2k_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | packed_small | packed | cold_prefill_2k_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | packed_small | packed | cold_prefill_2k_c1 | 2 | sequential | 1 | 4 | 0 |
| 1 | packed_small | packed | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | packed_small | packed | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | packed_small | packed | interactive_stream_c1 | 2 | sequential | 1 | 6 | 0 |
| 1 | packed_small | packed | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | packed_small | packed | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | packed_small | packed | decode_128_c1 | 2 | sequential | 1 | 4 | 0 |
| 2 | packed_small | packed | cold_prefill_2k_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | packed_small | packed | cold_prefill_2k_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | packed_small | packed | cold_prefill_2k_c1 | 2 | sequential | 1 | 4 | 0 |
| 2 | packed_small | packed | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | packed_small | packed | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | packed_small | packed | interactive_stream_c1 | 2 | sequential | 1 | 6 | 0 |
| 2 | packed_small | packed | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | packed_small | packed | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | packed_small | packed | decode_128_c1 | 2 | sequential | 1 | 4 | 0 |
