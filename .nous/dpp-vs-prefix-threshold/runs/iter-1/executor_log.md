Validation passed. Here's a summary of what was found:

## Experiment Complete — Iteration 1 Results

**All 20 simulation runs completed successfully. Validation: `{"status": "pass"}`**

### Key Findings

**h-main (CONFIRMED):** DPP(V=100, SLO=50ms) is byte-identical to AlwaysDisaggregate at every tested rate (5–30 req/s). The constant term V·c_D/2 = 650,000μs in the LHS completely dominates the DPP threshold equation, making the Z-feedback and queue-depth terms irrelevant. Difference is exactly 0%, not "within 1%".

**h-control-negative (CONFIRMED):** NeverDisaggregate consistently has higher mean ITL than PrefixThreshold, with the gap growing monotonically: +1.2% at rate=5 → +2.7% at rate=10 → +7.4% at rate=20 → **+14.7% at rate=30**. Unexpected nuance: Never has *lower* TTFT at rates ≤20 (no ~6.6ms KV transfer overhead), with the crossover near rate=30.

**h-robustness (CONFIRMED):** At rate=50, Never TTFT=**6,293ms** (109x disaggregating). Disaggregating policies maintain TTFT<100ms for completed requests, but an unexpected finding emerged: they drop ~30% of requests (598–621 dropped_unservable, 668–751 preemptions) due to decode KV cache exhaustion. The prefill-decode imbalance ratio reaches 1.44 at saturation.

### 4 Principles Extracted
- **RP-1:** DPP(V=100) ≡ AlwaysDisaggregate; V must be ≤0.01 for adaptive behavior (with BLIS's c_D=13000μs)
- **RP-2:** Disaggregation ITL improvement grows 1%→15% from rate=5→30 req/s
- **RP-3:** Disaggregation adds ~6.6ms TTFT overhead that is only overcome by Never's queuing delay near rate=30
- **RP-4:** At saturation, Never queues catastrophically; disaggregating policies maintain low TTFT but drop ~30% of requests via KV exhaustion