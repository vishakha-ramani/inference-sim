# meta-llama/llama-3.1-70b-instruct: varying `hardware` (H100, A100-SXM; baseline hardware=H100)

Fixed: tp=4, workload=chatbot, 32 req/s, 800 requests, seed 7.

- hardware=A100-SXM vs hardware=H100, Time to first token: +1.71 s (+3581%). Biggest impact: waiting for a batch slot, +1.69 s (99% of the change). Changed most: waiting for a batch slot, +16707%.
- hardware=A100-SXM vs hardware=H100, Time to first token, cause: direct +15.7 ms (1% of the change), knock-on +1.69 s (99%).
- hardware=A100-SXM vs hardware=H100, Inter-token latency: +16.5 ms (+76%). Biggest impact: weight reads, +9.0 ms (54% of the change). Changed most: compute, +337%.
- hardware=A100-SXM vs hardware=H100, Inter-token latency, cause: direct +13.8 ms (84% of the change), knock-on +2.7 ms (16%).
- hardware=A100-SXM vs hardware=H100, End-to-end latency: +5.89 s (+106%). Biggest impact: weight reads, +2.29 s (39% of the change). Changed most: compute, +332%.
- hardware=A100-SXM vs hardware=H100, End-to-end latency, cause: direct +3.52 s (60% of the change), knock-on +2.37 s (40%).
- hardware=A100-SXM vs hardware=H100, Paid GPU-time per request: +48.6 GPU-ms (+30%). Biggest impact: weight reads, +27.5 GPU-ms (57% of the change). Changed most: compute, +217%.
- hardware=A100-SXM vs hardware=H100, Paid GPU-time per request, cause: direct +87.6 GPU-ms (180% of the change), knock-on −39.0 GPU-ms (-80%). The two orders disagree (direct +95.9 GPU-ms vs +79.3 GPU-ms): cost and scheduling interact, so the split is an average.

GPU-time per request (busy, paid) and utilization:
- hardware=H100: busy 161 GPU-ms, paid 161 GPU-ms per request, GPU utilization 100%.
- hardware=A100-SXM: busy 210 GPU-ms, paid 210 GPU-ms per request, GPU utilization 100%.
