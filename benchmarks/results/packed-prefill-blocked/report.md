# MLX Air Benchmark Report

- Run: `run-new`
- Status: `succeeded`
- Suite: `packed_prefill`
- Focus: `None`
- Server mode: `self-launched`

## Results

Lower ms is better; higher tokens/s is better. Statistics are over trial means; p95 pools individual requests. **Bold** rows are each workload's primary metric.

| model | runtime | workload | metric | trials | mean | median | min–max | CV | request p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| packed_small | packed | cold_prefill_2k_c1 | **ttft (ms)** | 6 | 9717.670 | 9732.099 | 9594.423–9806.824 | 0.73% | 9859.009 |
| packed_small | packed | cold_prefill_2k_c1 | tpot (ms) | 6 | 33.174 | 33.124 | 32.924–33.491 | 0.61% | 33.598 |
| packed_small | packed | cold_prefill_2k_c1 | end_to_end_latency (ms) | 6 | 10253.128 | 10269.066 | 10129.148–10341.605 | 0.70% | 10391.045 |
| packed_small | packed | cold_prefill_2k_c1 | aggregate_output_tokens_per_second (tokens/s) | 6 | 1.561 | 1.558 | 1.547–1.580 | 0.70% | — |
| packed_small | packed | decode_128_c1 | **tpot (ms)** | 6 | 31.632 | 31.632 | 31.592–31.674 | 0.10% | 31.701 |
| packed_small | packed | decode_128_c1 | ttft (ms) | 6 | 340.316 | 311.869 | 303.042–418.597 | 14.93% | 468.667 |
| packed_small | packed | decode_128_c1 | end_to_end_latency (ms) | 6 | 4399.833 | 4374.631 | 4354.347–4478.548 | 1.16% | 4534.940 |
| packed_small | packed | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 6 | 29.095 | 29.259 | 28.580–29.395 | 1.16% | — |
| packed_small | packed | interactive_stream_c1 | **ttft (ms)** | 6 | 305.271 | 290.993 | 282.031–374.517 | 11.56% | 427.551 |
| packed_small | packed | interactive_stream_c1 | tpot (ms) | 6 | 31.454 | 31.451 | 31.424–31.483 | 0.07% | 31.523 |
| packed_small | packed | interactive_stream_c1 | end_to_end_latency (ms) | 6 | 1324.232 | 1318.336 | 1288.338–1380.075 | 2.31% | 1436.591 |
| packed_small | packed | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 6 | 24.173 | 24.271 | 23.185–24.836 | 2.27% | — |

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
