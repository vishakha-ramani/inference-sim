# meta-llama/llama-3.1-70b-instruct: varying `tp` (4, 8; baseline tp=4)

Fixed: hardware=H100, workload=chatbot, 64 users, 800 requests, seed 7.

- tp=8 vs tp=4, Time to first token: −14.8 ms (-24%). Biggest impact: weight reads, −7.1 ms (48% of the change). Changed most: compute, -51%.
- tp=8 vs tp=4, Time to first token, cause: direct −10.1 ms (68% of the change), knock-on −4.7 ms (32%).
- tp=8 vs tp=4, Inter-token latency: −8.4 ms (-42%). Biggest impact: weight reads, −7.0 ms (83% of the change). Changed most: kv-cache reads/writes, -50%.
- tp=8 vs tp=4, Inter-token latency, cause: direct −8.3 ms (100% of the change), knock-on −0.01 ms (0%).
- tp=8 vs tp=4, End-to-end latency: −4.35 s (-42%). Biggest impact: weight reads, −3.63 s (83% of the change). Changed most: kv-cache reads/writes, -50%.
- tp=8 vs tp=4, End-to-end latency, cause: direct −4.34 s (100% of the change), knock-on −6.9 ms (0%).
- tp=8 vs tp=4, Paid GPU-time per request: +112 GPU-ms (+16%). Biggest impact: fixed per-step overhead, +106 GPU-ms (95% of the change). Changed most: fixed per-step overhead, +100%.
- tp=8 vs tp=4, Paid GPU-time per request, cause: direct +111 GPU-ms (99% of the change), knock-on +0.90 GPU-ms (1%).

GPU-time per request (busy, paid) and utilization:
- tp=4: busy 703 GPU-ms, paid 703 GPU-ms per request, GPU utilization 100%.
- tp=8: busy 815 GPU-ms, paid 815 GPU-ms per request, GPU utilization 100%.
