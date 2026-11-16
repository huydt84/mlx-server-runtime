# MLX Air Benchmark Report

- Run: `baseline-f70afde-optimization`
- Status: `succeeded`
- Suite: `optimization`
- Focus: `None`
- Server mode: `self-launched`

## Results

Lower ms is better; higher tokens/s is better. Statistics are over trial means; p95 pools individual requests. **Bold** rows are each workload's primary metric.

| model | runtime | workload | metric | trials | mean | median | min–max | CV | request p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| dense | overlap | cache_pressure_c4 | **end_to_end_latency (ms)** | 4 | 42022.682 | 42125.589 | 40314.389–43525.161 | 3.98% | 51045.966 |
| dense | overlap | cache_pressure_c4 | aggregate_output_tokens_per_second (tokens/s) | 4 | 1.469 | 1.470 | 1.411–1.524 | 4.24% | — |
| dense | overlap | cold_prefill_2k_c1 | **ttft (ms)** | 4 | 9708.546 | 9705.852 | 9620.959–9801.520 | 0.82% | 9945.352 |
| dense | overlap | cold_prefill_2k_c1 | tpot (ms) | 4 | 39.079 | 38.897 | 38.605–39.917 | 1.51% | 40.185 |
| dense | overlap | cold_prefill_2k_c1 | end_to_end_latency (ms) | 4 | 10324.320 | 10328.707 | 10233.105–10406.760 | 0.72% | 10576.271 |
| dense | overlap | cold_prefill_2k_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 1.550 | 1.549 | 1.537–1.564 | 0.72% | — |
| dense | overlap | decode_128_c1 | **tpot (ms)** | 4 | 33.497 | 33.503 | 33.308–33.676 | 0.45% | 33.980 |
| dense | overlap | decode_128_c1 | ttft (ms) | 4 | 364.205 | 362.835 | 362.404–368.747 | 0.83% | 383.242 |
| dense | overlap | decode_128_c1 | end_to_end_latency (ms) | 4 | 3275.043 | 3274.535 | 3246.554–3304.547 | 0.99% | 4735.257 |
| dense | overlap | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 26.373 | 26.377 | 26.327–26.412 | 0.16% | — |
| dense | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 11.043 | 11.090 | 10.838–11.152 | 1.31% | — |
| dense | overlap | mixed_c8 | ttft (ms) | 4 | 7293.770 | 7233.589 | 7181.806–7526.097 | 2.15% | 17644.587 |
| dense | overlap | mixed_c8 | tpot (ms) | 4 | 507.208 | 506.768 | 501.593–513.705 | 1.14% | 651.552 |
| dense | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 23049.424 | 22947.872 | 22820.810–23481.141 | 1.32% | 31634.350 |
| dense | overlap | shared_prefix_warm_c16 | **ttft (ms)** | 4 | 7140.005 | 7132.607 | 6688.350–7606.458 | 5.48% | 12338.153 |
| dense | overlap | shared_prefix_warm_c16 | tpot (ms) | 4 | 304.382 | 298.499 | 292.347–328.182 | 5.45% | 431.185 |
| dense | overlap | shared_prefix_warm_c16 | end_to_end_latency (ms) | 4 | 16606.194 | 16416.971 | 15782.586–17808.247 | 5.34% | 18262.163 |
| dense | overlap | shared_prefix_warm_c16 | aggregate_output_tokens_per_second (tokens/s) | 4 | 29.937 | 30.231 | 27.872–31.415 | 5.15% | — |
| dense | serial | cache_pressure_c4 | **end_to_end_latency (ms)** | 4 | 40435.893 | 40185.016 | 39786.452–41587.090 | 2.09% | 48771.383 |
| dense | serial | cache_pressure_c4 | aggregate_output_tokens_per_second (tokens/s) | 4 | 1.517 | 1.524 | 1.480–1.539 | 1.83% | — |
| dense | serial | cold_prefill_2k_c1 | **ttft (ms)** | 4 | 9809.071 | 9804.950 | 9533.522–10092.863 | 3.15% | 10276.080 |
| dense | serial | cold_prefill_2k_c1 | tpot (ms) | 4 | 40.282 | 40.253 | 38.657–41.965 | 4.38% | 42.175 |
| dense | serial | cold_prefill_2k_c1 | end_to_end_latency (ms) | 4 | 10451.058 | 10447.005 | 10152.259–10757.962 | 3.30% | 10964.600 |
| dense | serial | cold_prefill_2k_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 1.532 | 1.533 | 1.487–1.576 | 3.30% | — |
| dense | serial | decode_128_c1 | **tpot (ms)** | 4 | 34.165 | 34.133 | 33.248–35.148 | 3.08% | 35.410 |
| dense | serial | decode_128_c1 | ttft (ms) | 4 | 376.391 | 378.166 | 363.706–385.528 | 2.82% | 401.023 |
| dense | serial | decode_128_c1 | end_to_end_latency (ms) | 4 | 3359.778 | 3350.036 | 3260.787–3478.251 | 3.15% | 4766.858 |
| dense | serial | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 25.726 | 25.687 | 25.012–26.519 | 3.08% | — |
| dense | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 10.992 | 11.087 | 10.560–11.233 | 2.70% | — |
| dense | serial | mixed_c8 | ttft (ms) | 4 | 7272.690 | 7200.758 | 7022.086–7667.156 | 3.80% | 17573.993 |
| dense | serial | mixed_c8 | tpot (ms) | 4 | 511.337 | 506.797 | 502.849–528.905 | 2.32% | 662.008 |
| dense | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 23156.446 | 22944.160 | 22642.805–24094.659 | 2.77% | 32158.310 |
| dense | serial | shared_prefix_warm_c16 | **ttft (ms)** | 4 | 6906.116 | 6916.722 | 6770.019–7021.001 | 1.56% | 12110.329 |
| dense | serial | shared_prefix_warm_c16 | tpot (ms) | 4 | 294.224 | 293.345 | 291.838–298.366 | 1.02% | 393.913 |
| dense | serial | shared_prefix_warm_c16 | end_to_end_latency (ms) | 4 | 16061.212 | 16002.730 | 15930.682–16308.705 | 1.06% | 16776.527 |
| dense | serial | shared_prefix_warm_c16 | aggregate_output_tokens_per_second (tokens/s) | 4 | 30.677 | 30.771 | 30.245–30.919 | 0.98% | — |
| moe | overlap | decode_128_c1 | **tpot (ms)** | 4 | 19.280 | 19.287 | 18.999–19.550 | 1.36% | 19.703 |
| moe | overlap | decode_128_c1 | ttft (ms) | 4 | 192.353 | 193.134 | 182.058–201.084 | 4.07% | 214.568 |
| moe | overlap | decode_128_c1 | end_to_end_latency (ms) | 4 | 2678.851 | 2678.919 | 2654.434–2703.131 | 0.82% | 2736.319 |
| moe | overlap | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 47.783 | 47.780 | 47.351–48.220 | 0.82% | — |
| moe | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 35.740 | 35.736 | 35.542–35.947 | 0.55% | — |
| moe | overlap | mixed_c8 | ttft (ms) | 4 | 2218.336 | 2216.032 | 2177.790–2263.491 | 1.59% | 5048.397 |
| moe | overlap | mixed_c8 | tpot (ms) | 4 | 155.796 | 155.515 | 155.106–157.049 | 0.59% | 199.489 |
| moe | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 7077.533 | 7079.363 | 7036.760–7114.646 | 0.51% | 9574.895 |
| moe | overlap | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 96.273 | 96.283 | 95.864–96.662 | 0.34% | — |
| moe | overlap | steady_short_c4 | ttft (ms) | 4 | 418.975 | 417.683 | 415.392–425.143 | 1.03% | 441.789 |
| moe | overlap | steady_short_c4 | tpot (ms) | 4 | 28.204 | 28.103 | 28.074–28.535 | 0.79% | 29.014 |
| moe | overlap | steady_short_c4 | end_to_end_latency (ms) | 4 | 1325.994 | 1326.736 | 1318.010–1332.495 | 0.46% | 1349.079 |
| moe | overlap | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 105.167 | 105.001 | 104.216–106.447 | 0.93% | — |
| moe | overlap | steady_short_c8 | ttft (ms) | 4 | 616.171 | 606.776 | 582.746–668.388 | 6.24% | 1027.960 |
| moe | overlap | steady_short_c8 | tpot (ms) | 4 | 57.002 | 57.667 | 54.389–58.286 | 3.11% | 67.476 |
| moe | overlap | steady_short_c8 | end_to_end_latency (ms) | 4 | 2413.913 | 2418.882 | 2384.184–2433.705 | 0.93% | 2859.732 |
| moe | serial | decode_128_c1 | **tpot (ms)** | 4 | 19.669 | 19.593 | 19.225–20.262 | 2.55% | 20.345 |
| moe | serial | decode_128_c1 | ttft (ms) | 4 | 186.542 | 186.185 | 177.472–196.326 | 4.23% | 202.229 |
| moe | serial | decode_128_c1 | end_to_end_latency (ms) | 4 | 2719.693 | 2715.635 | 2657.414–2790.087 | 2.33% | 2800.039 |
| moe | serial | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 47.082 | 47.143 | 45.875–48.166 | 2.32% | — |
| moe | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 35.733 | 35.827 | 35.257–36.019 | 0.99% | — |
| moe | serial | mixed_c8 | ttft (ms) | 4 | 2209.486 | 2209.306 | 2163.715–2255.618 | 1.72% | 5051.755 |
| moe | serial | mixed_c8 | tpot (ms) | 4 | 155.855 | 156.124 | 153.978–157.192 | 0.90% | 199.102 |
| moe | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 7072.780 | 7055.751 | 7020.637–7158.979 | 0.92% | 9674.911 |
| moe | serial | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 95.622 | 96.400 | 92.366–97.322 | 2.37% | — |
| moe | serial | steady_short_c4 | ttft (ms) | 4 | 423.703 | 420.161 | 402.844–451.645 | 5.08% | 486.922 |
| moe | serial | steady_short_c4 | tpot (ms) | 4 | 28.612 | 28.478 | 28.088–29.404 | 1.96% | 29.723 |
| moe | serial | steady_short_c4 | end_to_end_latency (ms) | 4 | 1337.906 | 1326.676 | 1312.715–1385.555 | 2.48% | 1394.109 |
| moe | serial | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 104.819 | 104.616 | 104.153–105.890 | 0.77% | — |
| moe | serial | steady_short_c8 | ttft (ms) | 4 | 545.730 | 553.147 | 509.378–567.245 | 5.06% | 1050.777 |
| moe | serial | steady_short_c8 | tpot (ms) | 4 | 59.375 | 59.051 | 58.510–60.887 | 1.77% | 70.291 |
| moe | serial | steady_short_c8 | end_to_end_latency (ms) | 4 | 2415.330 | 2418.839 | 2390.244–2433.397 | 0.79% | 2946.225 |
| primary | overlap | cache_pressure_c4 | **end_to_end_latency (ms)** | 4 | 21241.646 | 21401.750 | 20600.400–21562.687 | 2.05% | 25558.051 |
| primary | overlap | cache_pressure_c4 | aggregate_output_tokens_per_second (tokens/s) | 4 | 2.921 | 2.928 | 2.849–2.981 | 2.34% | — |
| primary | overlap | cold_prefill_2k_c1 | **ttft (ms)** | 4 | 4845.496 | 4856.378 | 4782.813–4886.415 | 0.91% | 4940.342 |
| primary | overlap | cold_prefill_2k_c1 | tpot (ms) | 4 | 20.200 | 20.271 | 19.729–20.527 | 1.67% | 20.868 |
| primary | overlap | cold_prefill_2k_c1 | end_to_end_latency (ms) | 4 | 5177.800 | 5187.906 | 5108.285–5227.102 | 0.96% | 5294.751 |
| primary | overlap | cold_prefill_2k_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 3.090 | 3.084 | 3.061–3.132 | 0.97% | — |
| primary | overlap | decode_128_c1 | **tpot (ms)** | 4 | 14.727 | 14.750 | 14.623–14.784 | 0.49% | 14.939 |
| primary | overlap | decode_128_c1 | ttft (ms) | 4 | 206.936 | 206.271 | 200.060–215.141 | 3.24% | 247.244 |
| primary | overlap | decode_128_c1 | end_to_end_latency (ms) | 4 | 2112.659 | 2112.986 | 2106.289–2118.376 | 0.24% | 2173.524 |
| primary | overlap | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 60.585 | 60.575 | 60.421–60.768 | 0.24% | — |
| primary | overlap | interactive_stream_c1 | **ttft (ms)** | 4 | 187.745 | 185.742 | 182.698–196.798 | 3.31% | 205.378 |
| primary | overlap | interactive_stream_c1 | tpot (ms) | 4 | 14.584 | 14.571 | 14.491–14.702 | 0.74% | 14.778 |
| primary | overlap | interactive_stream_c1 | end_to_end_latency (ms) | 4 | 682.751 | 679.527 | 676.408–695.540 | 1.27% | 701.707 |
| primary | overlap | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 46.869 | 47.085 | 46.001–47.303 | 1.25% | — |
| primary | overlap | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 21.252 | 21.449 | 20.480–21.632 | 2.53% | — |
| primary | overlap | mixed_c8 | ttft (ms) | 4 | 3658.972 | 3641.407 | 3600.810–3752.265 | 1.96% | 8662.646 |
| primary | overlap | mixed_c8 | tpot (ms) | 4 | 267.150 | 264.892 | 260.112–278.704 | 3.15% | 346.572 |
| primary | overlap | mixed_c8 | end_to_end_latency (ms) | 4 | 11972.019 | 11855.333 | 11754.011–12423.398 | 2.61% | 16128.114 |
| primary | overlap | shared_prefix_warm_c16 | **ttft (ms)** | 4 | 4125.081 | 4131.243 | 4070.438–4167.402 | 0.98% | 6954.388 |
| primary | overlap | shared_prefix_warm_c16 | tpot (ms) | 4 | 186.068 | 186.851 | 183.565–187.004 | 0.90% | 237.211 |
| primary | overlap | shared_prefix_warm_c16 | end_to_end_latency (ms) | 4 | 9923.860 | 9931.818 | 9841.035–9990.767 | 0.70% | 10293.926 |
| primary | overlap | shared_prefix_warm_c16 | aggregate_output_tokens_per_second (tokens/s) | 4 | 49.930 | 49.873 | 49.515–50.458 | 0.82% | — |
| primary | overlap | short_nonstream_c1 | **end_to_end_latency (ms)** | 4 | 684.407 | 684.495 | 676.082–692.557 | 1.13% | 700.336 |
| primary | overlap | short_nonstream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 46.754 | 46.745 | 46.198–47.325 | 1.13% | — |
| primary | overlap | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 78.550 | 78.355 | 77.355–80.133 | 1.52% | — |
| primary | overlap | steady_short_c4 | ttft (ms) | 4 | 627.317 | 628.310 | 619.598–633.051 | 0.92% | 679.246 |
| primary | overlap | steady_short_c4 | tpot (ms) | 4 | 31.143 | 31.097 | 30.820–31.556 | 1.11% | 32.850 |
| primary | overlap | steady_short_c4 | end_to_end_latency (ms) | 4 | 1627.905 | 1630.055 | 1597.223–1654.286 | 1.47% | 1701.950 |
| primary | overlap | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 81.667 | 82.710 | 76.788–84.463 | 4.38% | — |
| primary | overlap | steady_short_c8 | ttft (ms) | 4 | 946.260 | 933.684 | 903.884–1013.789 | 5.42% | 1436.456 |
| primary | overlap | steady_short_c8 | tpot (ms) | 4 | 69.186 | 67.341 | 66.855–75.206 | 5.82% | 82.792 |
| primary | overlap | steady_short_c8 | end_to_end_latency (ms) | 4 | 3123.934 | 3081.082 | 3014.537–3319.036 | 4.57% | 3724.523 |
| primary | serial | cache_pressure_c4 | **end_to_end_latency (ms)** | 4 | 21642.517 | 21663.493 | 20796.376–22446.705 | 3.35% | 25627.318 |
| primary | serial | cache_pressure_c4 | aggregate_output_tokens_per_second (tokens/s) | 4 | 2.844 | 2.829 | 2.770–2.946 | 2.85% | — |
| primary | serial | cold_prefill_2k_c1 | **ttft (ms)** | 4 | 4872.317 | 4869.029 | 4843.915–4907.295 | 0.55% | 4962.403 |
| primary | serial | cold_prefill_2k_c1 | tpot (ms) | 4 | 20.419 | 20.426 | 20.096–20.727 | 1.44% | 21.294 |
| primary | serial | cold_prefill_2k_c1 | end_to_end_latency (ms) | 4 | 5219.732 | 5218.158 | 5195.848–5246.765 | 0.51% | 5311.449 |
| primary | serial | cold_prefill_2k_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 3.065 | 3.066 | 3.049–3.079 | 0.51% | — |
| primary | serial | decode_128_c1 | **tpot (ms)** | 4 | 14.786 | 14.745 | 14.660–14.994 | 1.09% | 15.235 |
| primary | serial | decode_128_c1 | ttft (ms) | 4 | 204.263 | 205.406 | 196.159–210.079 | 3.26% | 250.450 |
| primary | serial | decode_128_c1 | end_to_end_latency (ms) | 4 | 2124.922 | 2127.738 | 2099.557–2144.654 | 1.04% | 2189.940 |
| primary | serial | decode_128_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 60.240 | 60.158 | 59.681–60.963 | 1.05% | — |
| primary | serial | interactive_stream_c1 | **ttft (ms)** | 4 | 184.731 | 184.774 | 180.606–188.772 | 2.14% | 200.444 |
| primary | serial | interactive_stream_c1 | tpot (ms) | 4 | 14.844 | 14.788 | 14.619–15.182 | 1.78% | 15.112 |
| primary | serial | interactive_stream_c1 | end_to_end_latency (ms) | 4 | 679.809 | 679.255 | 677.018–683.709 | 0.44% | 687.936 |
| primary | serial | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 47.067 | 47.105 | 46.798–47.260 | 0.44% | — |
| primary | serial | mixed_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 20.702 | 20.901 | 19.803–21.202 | 3.04% | — |
| primary | serial | mixed_c8 | ttft (ms) | 4 | 3734.796 | 3704.219 | 3593.607–3937.139 | 4.47% | 8745.118 |
| primary | serial | mixed_c8 | tpot (ms) | 4 | 274.799 | 271.996 | 269.245–285.958 | 2.75% | 356.282 |
| primary | serial | mixed_c8 | end_to_end_latency (ms) | 4 | 12281.850 | 12158.410 | 11979.670–12830.910 | 3.12% | 16590.391 |
| primary | serial | shared_prefix_warm_c16 | **ttft (ms)** | 4 | 4293.964 | 4302.737 | 4156.981–4413.400 | 2.48% | 7148.042 |
| primary | serial | shared_prefix_warm_c16 | tpot (ms) | 4 | 191.283 | 191.036 | 186.838–196.221 | 2.06% | 245.479 |
| primary | serial | shared_prefix_warm_c16 | end_to_end_latency (ms) | 4 | 10251.584 | 10205.146 | 10073.321–10522.722 | 2.05% | 10848.930 |
| primary | serial | shared_prefix_warm_c16 | aggregate_output_tokens_per_second (tokens/s) | 4 | 47.963 | 48.180 | 46.729–48.764 | 2.05% | — |
| primary | serial | short_nonstream_c1 | **end_to_end_latency (ms)** | 4 | 677.262 | 676.177 | 675.459–681.236 | 0.40% | 687.220 |
| primary | serial | short_nonstream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 47.243 | 47.318 | 46.967–47.369 | 0.40% | — |
| primary | serial | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 78.920 | 79.113 | 77.890–79.565 | 0.91% | — |
| primary | serial | steady_short_c4 | ttft (ms) | 4 | 620.256 | 615.801 | 611.169–638.253 | 1.97% | 675.488 |
| primary | serial | steady_short_c4 | tpot (ms) | 4 | 31.226 | 31.198 | 31.173–31.336 | 0.24% | 31.715 |
| primary | serial | steady_short_c4 | end_to_end_latency (ms) | 4 | 1620.317 | 1615.141 | 1608.377–1642.610 | 0.95% | 1652.065 |
| primary | serial | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 81.947 | 81.956 | 80.801–83.075 | 1.26% | — |
| primary | serial | steady_short_c8 | ttft (ms) | 4 | 667.899 | 670.727 | 624.871–705.272 | 4.94% | 1264.600 |
| primary | serial | steady_short_c8 | tpot (ms) | 4 | 77.333 | 77.471 | 76.507–77.884 | 0.76% | 90.145 |
| primary | serial | steady_short_c8 | end_to_end_latency (ms) | 4 | 3098.323 | 3097.504 | 3058.450–3139.835 | 1.20% | 3709.066 |
| runtime_control | overlap | interactive_stream_c1 | **ttft (ms)** | 4 | 240.618 | 239.854 | 234.735–248.028 | 2.29% | 272.560 |
| runtime_control | overlap | interactive_stream_c1 | tpot (ms) | 4 | 90.696 | 90.447 | 87.865–94.028 | 3.38% | 94.218 |
| runtime_control | overlap | interactive_stream_c1 | end_to_end_latency (ms) | 4 | 3085.414 | 3078.389 | 3004.076–3180.801 | 2.93% | 3210.864 |
| runtime_control | overlap | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 10.378 | 10.400 | 10.060–10.652 | 2.93% | — |
| runtime_control | overlap | short_nonstream_c1 | **end_to_end_latency (ms)** | 4 | 3049.030 | 3038.488 | 2989.017–3130.126 | 2.27% | 3130.427 |
| runtime_control | overlap | short_nonstream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 10.499 | 10.533 | 10.223–10.706 | 2.26% | — |
| runtime_control | overlap | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 35.447 | 35.608 | 34.136–36.438 | 3.26% | — |
| runtime_control | overlap | steady_short_c4 | ttft (ms) | 4 | 614.942 | 610.208 | 594.929–644.424 | 3.41% | 668.370 |
| runtime_control | overlap | steady_short_c4 | tpot (ms) | 4 | 95.613 | 95.493 | 92.595–98.873 | 3.37% | 100.545 |
| runtime_control | overlap | steady_short_c4 | end_to_end_latency (ms) | 4 | 3610.322 | 3594.426 | 3503.451–3748.985 | 3.34% | 3809.524 |
| runtime_control | overlap | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 51.122 | 51.047 | 50.170–52.221 | 2.12% | — |
| runtime_control | overlap | steady_short_c8 | ttft (ms) | 4 | 952.657 | 955.567 | 938.677–960.819 | 1.02% | 1841.761 |
| runtime_control | overlap | steady_short_c8 | tpot (ms) | 4 | 127.321 | 127.448 | 124.037–130.349 | 2.62% | 144.652 |
| runtime_control | overlap | steady_short_c8 | end_to_end_latency (ms) | 4 | 4928.148 | 4936.532 | 4817.119–5022.408 | 2.18% | 5899.688 |
| runtime_control | serial | interactive_stream_c1 | **ttft (ms)** | 4 | 246.307 | 245.088 | 238.850–256.203 | 2.95% | 266.600 |
| runtime_control | serial | interactive_stream_c1 | tpot (ms) | 4 | 89.295 | 89.116 | 86.375–92.571 | 3.40% | 92.868 |
| runtime_control | serial | interactive_stream_c1 | end_to_end_latency (ms) | 4 | 3047.592 | 3042.917 | 2949.514–3155.019 | 3.13% | 3177.666 |
| runtime_control | serial | interactive_stream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 10.508 | 10.519 | 10.142–10.849 | 3.12% | — |
| runtime_control | serial | short_nonstream_c1 | **end_to_end_latency (ms)** | 4 | 3013.463 | 3015.254 | 2964.343–3059.003 | 1.55% | 3076.034 |
| runtime_control | serial | short_nonstream_c1 | aggregate_output_tokens_per_second (tokens/s) | 4 | 10.621 | 10.614 | 10.461–10.795 | 1.55% | — |
| runtime_control | serial | steady_short_c4 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 36.643 | 36.611 | 36.240–37.109 | 0.97% | — |
| runtime_control | serial | steady_short_c4 | ttft (ms) | 4 | 592.412 | 591.081 | 580.655–606.832 | 2.10% | 646.862 |
| runtime_control | serial | steady_short_c4 | tpot (ms) | 4 | 92.411 | 92.597 | 90.533–93.915 | 1.55% | 94.457 |
| runtime_control | serial | steady_short_c4 | end_to_end_latency (ms) | 4 | 3489.877 | 3494.728 | 3443.895–3526.158 | 0.98% | 3589.336 |
| runtime_control | serial | steady_short_c8 | **aggregate_output_tokens_per_second (tokens/s)** | 4 | 52.566 | 52.531 | 52.401–52.800 | 0.38% | — |
| runtime_control | serial | steady_short_c8 | ttft (ms) | 4 | 816.157 | 817.792 | 800.013–829.031 | 1.78% | 1748.829 |
| runtime_control | serial | steady_short_c8 | tpot (ms) | 4 | 127.029 | 127.033 | 126.858–127.191 | 0.11% | 143.630 |
| runtime_control | serial | steady_short_c8 | end_to_end_latency (ms) | 4 | 4787.353 | 4787.643 | 4771.233–4802.894 | 0.34% | 5772.767 |

## Trials

| round | model | runtime | workload | trial | load mode | concurrency | succeeded | errors |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | primary | serial | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | primary | serial | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | primary | serial | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | primary | serial | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | primary | serial | cold_prefill_2k_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | primary | serial | cold_prefill_2k_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | primary | serial | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | primary | serial | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | primary | serial | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | primary | serial | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | primary | serial | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | primary | serial | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | primary | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | primary | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | primary | serial | shared_prefix_warm_c16 | 0 | burst | 16 | 16 | 0 |
| 1 | primary | serial | shared_prefix_warm_c16 | 1 | burst | 16 | 16 | 0 |
| 1 | primary | serial | cache_pressure_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | primary | serial | cache_pressure_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | primary | overlap | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | primary | overlap | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | primary | overlap | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | primary | overlap | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | primary | overlap | cold_prefill_2k_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | primary | overlap | cold_prefill_2k_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | primary | overlap | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | primary | overlap | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | primary | overlap | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | primary | overlap | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | primary | overlap | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | primary | overlap | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | primary | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | primary | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | primary | overlap | shared_prefix_warm_c16 | 0 | burst | 16 | 16 | 0 |
| 1 | primary | overlap | shared_prefix_warm_c16 | 1 | burst | 16 | 16 | 0 |
| 1 | primary | overlap | cache_pressure_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | primary | overlap | cache_pressure_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | dense | overlap | cold_prefill_2k_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | dense | overlap | cold_prefill_2k_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | dense | overlap | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | dense | overlap | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | dense | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | dense | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | dense | overlap | shared_prefix_warm_c16 | 0 | burst | 16 | 16 | 0 |
| 1 | dense | overlap | shared_prefix_warm_c16 | 1 | burst | 16 | 16 | 0 |
| 1 | dense | overlap | cache_pressure_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | dense | overlap | cache_pressure_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | dense | serial | cold_prefill_2k_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | dense | serial | cold_prefill_2k_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | dense | serial | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | dense | serial | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | dense | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | dense | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | dense | serial | shared_prefix_warm_c16 | 0 | burst | 16 | 16 | 0 |
| 1 | dense | serial | shared_prefix_warm_c16 | 1 | burst | 16 | 16 | 0 |
| 1 | dense | serial | cache_pressure_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | dense | serial | cache_pressure_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | moe | serial | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | moe | serial | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | moe | serial | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | moe | serial | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | moe | serial | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | moe | serial | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | moe | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | moe | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | moe | overlap | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 1 | moe | overlap | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 1 | moe | overlap | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | moe | overlap | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | moe | overlap | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | moe | overlap | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | moe | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | moe | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | runtime_control | overlap | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | overlap | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | overlap | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | overlap | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | overlap | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | runtime_control | overlap | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | runtime_control | overlap | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | runtime_control | overlap | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 1 | runtime_control | serial | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | serial | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | serial | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | serial | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 1 | runtime_control | serial | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 1 | runtime_control | serial | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 1 | runtime_control | serial | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 1 | runtime_control | serial | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | primary | overlap | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | primary | overlap | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | primary | overlap | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | primary | overlap | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | primary | overlap | cold_prefill_2k_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | primary | overlap | cold_prefill_2k_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | primary | overlap | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | primary | overlap | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | primary | overlap | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | primary | overlap | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | primary | overlap | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | primary | overlap | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | primary | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | primary | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | primary | overlap | shared_prefix_warm_c16 | 0 | burst | 16 | 16 | 0 |
| 2 | primary | overlap | shared_prefix_warm_c16 | 1 | burst | 16 | 16 | 0 |
| 2 | primary | overlap | cache_pressure_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | primary | overlap | cache_pressure_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | primary | serial | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | primary | serial | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | primary | serial | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | primary | serial | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | primary | serial | cold_prefill_2k_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | primary | serial | cold_prefill_2k_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | primary | serial | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | primary | serial | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | primary | serial | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | primary | serial | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | primary | serial | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | primary | serial | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | primary | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | primary | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | primary | serial | shared_prefix_warm_c16 | 0 | burst | 16 | 16 | 0 |
| 2 | primary | serial | shared_prefix_warm_c16 | 1 | burst | 16 | 16 | 0 |
| 2 | primary | serial | cache_pressure_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | primary | serial | cache_pressure_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | dense | serial | cold_prefill_2k_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | dense | serial | cold_prefill_2k_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | dense | serial | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | dense | serial | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | dense | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | dense | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | dense | serial | shared_prefix_warm_c16 | 0 | burst | 16 | 16 | 0 |
| 2 | dense | serial | shared_prefix_warm_c16 | 1 | burst | 16 | 16 | 0 |
| 2 | dense | serial | cache_pressure_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | dense | serial | cache_pressure_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | dense | overlap | cold_prefill_2k_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | dense | overlap | cold_prefill_2k_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | dense | overlap | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | dense | overlap | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | dense | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | dense | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | dense | overlap | shared_prefix_warm_c16 | 0 | burst | 16 | 16 | 0 |
| 2 | dense | overlap | shared_prefix_warm_c16 | 1 | burst | 16 | 16 | 0 |
| 2 | dense | overlap | cache_pressure_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | dense | overlap | cache_pressure_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | moe | overlap | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | moe | overlap | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | moe | overlap | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | moe | overlap | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | moe | overlap | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | moe | overlap | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | moe | overlap | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | moe | overlap | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | moe | serial | decode_128_c1 | 0 | sequential | 1 | 4 | 0 |
| 2 | moe | serial | decode_128_c1 | 1 | sequential | 1 | 4 | 0 |
| 2 | moe | serial | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | moe | serial | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | moe | serial | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | moe | serial | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | moe | serial | mixed_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | moe | serial | mixed_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | runtime_control | serial | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | serial | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | serial | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | serial | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | serial | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | runtime_control | serial | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | runtime_control | serial | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | runtime_control | serial | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
| 2 | runtime_control | overlap | interactive_stream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | overlap | interactive_stream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | overlap | short_nonstream_c1 | 0 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | overlap | short_nonstream_c1 | 1 | sequential | 1 | 6 | 0 |
| 2 | runtime_control | overlap | steady_short_c4 | 0 | closed-loop | 4 | 16 | 0 |
| 2 | runtime_control | overlap | steady_short_c4 | 1 | closed-loop | 4 | 16 | 0 |
| 2 | runtime_control | overlap | steady_short_c8 | 0 | closed-loop | 8 | 32 | 0 |
| 2 | runtime_control | overlap | steady_short_c8 | 1 | closed-loop | 8 | 32 | 0 |
