# meta-llama/llama-3.1-70b-instruct: varying `tp` (4, 8; baseline tp=4)

Fixed: hardware=H100, workload=chatbot, 32 req/s, 800 requests, seed 7.

- tp=8 vs tp=4, Time to first token: −14.8 ms (-31%). Biggest impact: weight reads, −7.0 ms (47% of the change). Changed most: kv-cache reads/writes, -72%.
- tp=8 vs tp=4, Time to first token, cause: direct −8.7 ms (59% of the change), knock-on −6.1 ms (41%).
- tp=8 vs tp=4, Inter-token latency: −10.2 ms (-47%). Biggest impact: weight reads, −7.0 ms (68% of the change). Changed most: kv-cache reads/writes, -72%.
- tp=8 vs tp=4, Inter-token latency, cause: direct −8.5 ms (83% of the change), knock-on −1.8 ms (17%).
- tp=8 vs tp=4, End-to-end latency: −2.63 s (-47%). Biggest impact: weight reads, −1.78 s (68% of the change). Changed most: kv-cache reads/writes, -72%.
- tp=8 vs tp=4, End-to-end latency, cause: direct −2.17 s (82% of the change), knock-on −460 ms (18%).
- tp=8 vs tp=4, Paid GPU-time per request: +126 GPU-ms (+78%). Biggest impact: weight reads, +70.8 GPU-ms (56% of the change). Changed most: fixed per-step overhead, +207%.
- tp=8 vs tp=4, Paid GPU-time per request, cause: direct +35.0 GPU-ms (28% of the change), knock-on +91.3 GPU-ms (72%).

GPU-time per request (busy, paid) and utilization:
- tp=4: busy 161 GPU-ms, paid 161 GPU-ms per request, GPU utilization 100%.
- tp=8: busy 287 GPU-ms, paid 288 GPU-ms per request, GPU utilization 100%.
