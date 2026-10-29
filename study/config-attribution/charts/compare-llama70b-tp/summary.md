# meta-llama/llama-3.1-70b-instruct: varying `tp` (2, 4, 8; baseline tp=2)

Fixed: hardware=H100, workload=chatbot, 32 req/s, 800 requests, seed 7.

- tp=4 vs tp=2, Time to first token: −57.67 s (-100%). Biggest impact: waiting for a batch slot, −57.46 s (100% of the change). Changed most: waiting for a batch slot, -100%.
- tp=8 vs tp=2, Time to first token: −57.69 s (-100%). Biggest impact: waiting for a batch slot, −57.46 s (100% of the change). Changed most: waiting for a batch slot, -100%.
- tp=4 vs tp=2, Inter-token latency: −11.8 ms (-35%). Biggest impact: weight reads, −13.9 ms (118% of the change). Changed most: kv-cache reads/writes, +75%.
- tp=8 vs tp=2, Inter-token latency: −22.0 ms (-66%). Biggest impact: weight reads, −20.9 ms (95% of the change). Changed most: weight reads, -75%.
- tp=4 vs tp=2, End-to-end latency: −60.68 s (-92%). Biggest impact: waiting for a batch slot, −57.46 s (95% of the change). Changed most: waiting for a batch slot, -100%.
- tp=8 vs tp=2, End-to-end latency: −63.30 s (-96%). Biggest impact: waiting for a batch slot, −57.46 s (91% of the change). Changed most: waiting for a batch slot, -100%.
- tp=4 vs tp=2, Paid GPU-time per request: −227 GPU-ms (-58%). Biggest impact: weight reads, −217 GPU-ms (95% of the change). Changed most: weight reads, -67%.
- tp=8 vs tp=2, Paid GPU-time per request: −101 GPU-ms (-26%). Biggest impact: weight reads, −146 GPU-ms (145% of the change). Changed most: fixed per-step overhead, +132%.

GPU-time per request (busy, paid) and utilization:
- tp=2: busy 389 GPU-ms, paid 389 GPU-ms per request, GPU utilization 100%.
- tp=4: busy 161 GPU-ms, paid 161 GPU-ms per request, GPU utilization 100%.
- tp=8: busy 287 GPU-ms, paid 288 GPU-ms per request, GPU utilization 100%.
