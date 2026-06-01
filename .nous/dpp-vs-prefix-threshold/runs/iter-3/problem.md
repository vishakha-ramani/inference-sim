# Problem Framing — Iter-3: DPP vs PrefixThreshold in Multi-Instance (2P+2D) Topology

## Research Question

Does the ranking between DPP and PrefixThreshold invert in a 2P+2D topology compared to the 1P+1D topology studied in iterations 1-2?

Specifically: In a 2-prefill + 2-decode instance topology, does DPP(V≤10, SLO=20ms) achieve lower mean ITL than PrefixThreshold(N=16) across a load sweep, while simultaneously maintaining higher completion rate at near-saturation loads?

**Mechanism hypothesis:** In 1P+1D, all traffic funnels to a single decode instance regardless of decider. PT wins on ITL because full disaggregation offloads prefill compute. In 2P+2D, PT still funnels all decode work to 2 decode-only instances, but DPP keeps ~90% of requests local (Disaggregate=false), distributing them across the 2 decode instances which also handle their own prefill — avoiding KV transfer overhead (mean 6.6ms) and decode KV exhaustion under load. This makes DPP's "keep local" strategy superior to PT's "always offload" strategy when decode capacity is the bottleneck.

**Code evidence for mechanism:**
- `sim/disaggregation.go:129-141` — DPP Decide(): sums Q_D across all decode snapshots, Q_P across all prefill snapshots. In 2P+2D, Q_D is the aggregate of 2 decode instances.
- `sim/cluster/cluster.go:1889-1914` — executeDisaggregatedRouting: when Disaggregate=false, request goes to the pre-selected decode pod for BOTH prefill and decode. The 2 prefill-only instances sit idle for non-disaggregated requests.
- `sim/cluster/cluster.go:1166-1181` — buildPoolFilteredSnapshots: filters by pool role, so DPP sees queue depths from all 2 prefill and 2 decode instances.

## System Interface

- **Build:** `go build -o blis .` (validated, exit 0)
- **CLI flags:**
  - `--num-instances 4 --prefill-instances 2 --decode-instances 2` — 2P+2D topology (`cmd/root.go:984,1023-1024`)
  - `--pd-decider {never,prefix-threshold,dpp}` — disaggregation policy (`cmd/root.go:1014`)
  - `--pd-prefix-threshold N` — PT uncached-token threshold (`cmd/root.go:1015`)
  - `--dpp-v V` — DPP penalty weight (`cmd/root.go:1016`)
  - `--dpp-eta η` — DPP queue weight ratio (`cmd/root.go:1017`)
  - `--dpp-ttft-slo-d D` — DPP TTFT SLO in ms (`cmd/root.go:1018`)
  - `--rate λ` — Poisson arrival rate (`cmd/root.go:973`)
  - `--num-requests N` — total requests (`cmd/root.go:974`)
  - `--metrics-path <file>` — write cluster-aggregate JSON (`cmd/root.go:1007`)
  - `--seed 42` — deterministic RNG (`cmd/root.go:977`)
- **Output format:** `--metrics-path` writes cluster-aggregate JSON with fields: `itl_mean_ms`, `ttft_mean_ms`, `completed_requests`, `dropped_unservable`, `preemption_count`. PD metrics (disaggregation count) printed to stdout — grep for "Disaggregated Requests".
- **Code evidence:** CLI flags defined at `cmd/root.go:973-1025`. DPP factory at `sim/cluster/cluster.go:429-436` with hardcoded W_P=29900μs, c_D=13000μs.

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --num-instances 4 --prefill-instances 2 --decode-instances 2 \
  --pd-decider prefix-threshold --pd-prefix-threshold 16 \
  --rate 45 --num-requests 2000 \
  --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 \
  --seed 42 --metrics-path results/baseline_pt_2p2d_r45.json
```

## Baseline Validation

Command exits 0. Output file produced at `results/baseline_pt_2p2d_r45.json`.
Key metrics: ITL=31.84ms, TTFT=60.09ms, completed=1625/2000, dropped=375, preemptions=417, disaggregated=1609/2000.

## Experimental Conditions

### Condition 1: NeverDisaggregate rate sweep (control — tests that decode-local is optimal at low load)
Rates: 30, 35, 40, 45, 50, 55, 60 req/s. `--pd-decider never`. All other flags same as baseline.
Purpose: Establishes the "decode-local" ITL floor; confirms prefill instances are unused.

### Condition 2: PrefixThreshold(N=16) rate sweep (incumbent policy)
Rates: 30, 35, 40, 45, 50, 55, 60 req/s. `--pd-decider prefix-threshold --pd-prefix-threshold 16`. All other flags same as baseline.
Purpose: Captures PT's saturation cliff and ITL degradation in 2P+2D.

### Condition 3: DPP(V=5, SLO=20ms) rate sweep (primary challenger)
Rates: 30, 35, 40, 45, 50, 55, 60 req/s. `--pd-decider dpp --dpp-v 5 --dpp-eta 1.0 --dpp-ttft-slo-d 20.0`. All other flags same as baseline.
Purpose: Tests whether DPP achieves ITL close to Never while maintaining 100% completion where PT drops.

### Condition 4: DPP(V=10, SLO=20ms) rate sweep (secondary challenger — higher disagg fraction)
Rates: 30, 35, 40, 45, 50, 55, 60 req/s. `--pd-decider dpp --dpp-v 10 --dpp-eta 1.0 --dpp-ttft-slo-d 20.0`. All other flags same as baseline.
Purpose: Tests whether V=10 (22% disagg vs V=5's 12%) maintains the stability benefit while improving decode throughput at high load.

### Condition 5: DPP(V=5, SLO=22ms) rate sweep (phase-transition control)
Rates: 30, 35, 40, 45, 50 req/s. `--pd-decider dpp --dpp-v 5 --dpp-eta 1.0 --dpp-ttft-slo-d 22.0`. All other flags same as baseline.
Purpose: Tests DPP above the Z-feedback phase transition (RP-7). Should behave like AlwaysDisaggregate and match PT's drop pattern.

## Success Criteria

1. **ITL dominance (h-main):** DPP(V=5, SLO=20) achieves mean ITL lower than PT at every tested rate from 30-60 req/s.
2. **Completion advantage (h-main):** DPP(V=5, SLO=20) completes more requests than PT at rates where PT drops (≥35 req/s).
3. **Mechanism validation (h-control-negative):** DPP(V=5, SLO=22ms) — above the phase transition — produces completion rates and drop counts similar to PT (±10%), confirming the Z-suppression mechanism is responsible for the advantage.
4. **Scale effect (h-robustness):** The DPP advantage (lower ITL and higher completion than PT) persists across the full rate sweep 30-60, not just at a single operating point.

## Constraints

- All runs use seed=42 for determinism (INV-6).
- 2000 requests per condition for stable metrics (per campaign spec).
- Rate sweep covers 30-60 req/s (probes show 30=stable, 35=onset of PT drops, 60=heavy saturation).
- No code changes required — pure flag-variation experiment.

## Prior Knowledge

Active principles that constrain this design:
- **RP-5:** DPP partial disaggregation is strictly dominated for ITL in **1P+1D** topology. This experiment tests whether the same holds in 2P+2D.
- **RP-6:** DPP(V≤10, SLO=20) functions as stability controller at near-saturation. We expect this to hold in 2P+2D.
- **RP-7:** Z-feedback creates a sharp phase transition at SLO≈21ms. We use this to design the negative control (SLO=22ms crosses the boundary → full disaggregation → PT-like behavior).
- **RP-1:** DPP(V=100) ≡ AlwaysDisaggregate. At V=100, DPP should match or exceed PT's disagg fraction and drop rate.
