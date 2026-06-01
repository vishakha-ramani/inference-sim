# Problem Framing: DPP vs PrefixThreshold Baseline Characterization

## Research Question

Does the DriftPlusPenaltyDecider (DPP) — a queue-aware, TTFT-SLO-feedback controller — empirically outperform the PrefixThresholdDecider on mean ITL while keeping TTFT comparable, in a 1P+1D disaggregated topology?

**Iteration 1 goal**: Characterize the three static baselines (Never, Always, PrefixThreshold N=16) and the dynamic DPP controller across a load sweep to identify: (a) the load regime where disaggregation helps, (b) the ITL–TTFT tradeoff each policy makes, and (c) whether DPP's queue-state feedback provides any measurable advantage over PrefixThreshold's cache-state heuristic.

**Source files implementing the mechanism:**
- `sim/disaggregation.go:89-148` — DriftPlusPenaltyDecider struct and Decide() logic
- `sim/disaggregation.go:178-244` — PrefixThresholdDecider struct and Decide() logic
- `sim/cluster/cluster.go:394-436` — DPP factory with hardcoded W_P=29900μs, c_D=13000μs
- `sim/cluster/cluster.go:1887-1914` — Routing integration point (disaggregation decision)
- `sim/cluster/pd_metrics.go:11-62` — Disaggregation metrics collection

## System Interface

**Build command:**
```bash
go build -o blis .
```

**CLI flags relevant to experiment:**
| Flag | Default | Semantics | Source |
|------|---------|-----------|--------|
| `--pd-decider` | `"never"` | Disaggregation policy: never, always, prefix-threshold, dpp | `cmd/root.go:1026` |
| `--pd-prefix-threshold` | `16` | Non-cached token threshold for prefix-threshold decider | `cmd/root.go:1034` |
| `--dpp-v` | `100` | DPP penalty weight V (trades TTFT for lower ITL) | `cmd/root.go:1028` |
| `--dpp-eta` | `1.0` | DPP queue weight ratio η | `cmd/root.go:1029` |
| `--dpp-ttft-slo-d` | `50.0` | DPP TTFT SLO target d (ms); Z grows when TTFT exceeds this | `cmd/root.go:1030` |
| `--prefill-instances` | `0` | Number of dedicated prefill instances | `cmd/root.go:1023` |
| `--decode-instances` | `0` | Number of dedicated decode instances | `cmd/root.go:1024` |
| `--num-instances` | `1` | Total instances (must ≥ prefill + decode) | `cmd/root.go:996` |
| `--rate` | `1.0` | Poisson arrival rate (req/s) | `cmd/root.go:999` |
| `--num-requests` | `100` | Total requests to simulate | `cmd/root.go:1000` |
| `--prompt-tokens` | `512` | Mean prompt token count | `cmd/root.go:1003` |
| `--prefix-tokens` | `0` | Shared prefix length (cached tokens) | `cmd/root.go:1007` |
| `--output-tokens` | `128` | Output token count | `cmd/root.go:1008` |
| `--seed` | `42` | RNG seed for determinism | `cmd/root.go:1001` |
| `--metrics-path` | `""` | File to write MetricsOutput JSON | `cmd/root.go:2090` |

**Output format:** Simulation metrics are emitted to stdout as JSON after `=== Simulation Metrics ===` header (one block per instance + one "cluster" aggregate). PD metrics are emitted separately as `=== PD Metrics ===` with disaggregation count and throughput stats.

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b \
  --prefill-instances 1 --decode-instances 1 --num-instances 2 \
  --pd-decider never \
  --rate 30 --num-requests 2000 \
  --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 \
  --seed 42
```

## Baseline Validation

Ran the baseline command (NeverDisaggregate, rate=30, 2000 requests). Exit code 0. Output:
- `completed_requests`: 2000
- `ttft_mean_ms`: 64.85
- `itl_mean_ms`: 32.86
- All requests processed locally on the decode instance (no disaggregation)

Comparison at same parameters with PrefixThreshold: ITL=28.65ms (13% improvement), TTFT=64.79ms (comparable), Disag=1990/2000.

## Experimental Conditions

**Fixed topology:** 1 prefill instance + 1 decode instance (--num-instances 2).
**Fixed workload:** --prompt-tokens 512, --prefix-tokens 480, --output-tokens 128 (prefix-heavy: 32 uncached tokens per request).
**Fixed N=2000 requests, seed=42.**

### Condition 1: NeverDisaggregate (control)
All requests processed locally on decode instance. No use of prefill server.
```bash
./blis run --model qwen/qwen3-14b --prefill-instances 1 --decode-instances 1 --num-instances 2 --pd-decider never --rate {RATE} --num-requests 2000 --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 --seed 42 --metrics-path results/never_rate{RATE}.json
```

### Condition 2: AlwaysDisaggregate
All requests disaggregated: prefill on prefill instance, decode on decode instance, KV transfer in between.
```bash
./blis run --model qwen/qwen3-14b --prefill-instances 1 --decode-instances 1 --num-instances 2 --pd-decider always --rate {RATE} --num-requests 2000 --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 --seed 42 --metrics-path results/always_rate{RATE}.json
```

### Condition 3: PrefixThreshold (N=16)
Disaggregate when non-cached tokens > 16. With 32 uncached tokens per request, this disaggregates ~99% of requests.
```bash
./blis run --model qwen/qwen3-14b --prefill-instances 1 --decode-instances 1 --num-instances 2 --pd-decider prefix-threshold --pd-prefix-threshold 16 --rate {RATE} --num-requests 2000 --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 --seed 42 --metrics-path results/pt16_rate{RATE}.json
```

### Condition 4: DPP (V=100, SLO=50ms) — default parameters
Disaggregate via Lyapunov threshold: η·Q_D + V·c_D/2 > Q_P + Z·ΔT/W_P.
With default V=100, this makes LHS = Q_D + 650000, effectively always disaggregating.
```bash
./blis run --model qwen/qwen3-14b --prefill-instances 1 --decode-instances 1 --num-instances 2 --pd-decider dpp --dpp-v 100 --dpp-ttft-slo-d 50 --rate {RATE} --num-requests 2000 --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 --seed 42 --metrics-path results/dpp_v100_slo50_rate{RATE}.json
```

### Rate sweep values
{RATE} ∈ {5, 10, 20, 30, 50}

These span: low load (rate=5, no queueing), medium (rate=20–30, moderate queueing), and high (rate=50, capacity saturation where Never's TTFT explodes to 6293ms).

## Success Criteria

1. **Disaggregation helps ITL**: At rate≥20, PrefixThreshold and AlwaysDisaggregate achieve lower mean ITL than NeverDisaggregate (validated: 7-15% improvement observed).
2. **TTFT overhead characterized**: Disaggregation adds TTFT overhead from KV transfer (~10-15ms at low load); this overhead is bounded and predictable.
3. **Rate-dependent behavior identified**: The rate at which NeverDisaggregate's TTFT degrades catastrophically (>1000ms) is identified. Validated: rate=50 produces TTFT=6293ms for Never.
4. **DPP vs PrefixThreshold differentiation**: Determine whether DPP(V=100, SLO=50ms) produces measurably different results from PrefixThreshold(N=16) at any tested rate.

## Constraints

- Deterministic: fixed seed=42, same parameters → byte-identical output (INV-6)
- 2000 requests per condition for statistical stability
- No external servers needed (pure simulation via `blis run`)
- Total of 20 simulation runs (4 deciders × 5 rates)

## Prior Knowledge

This is the first iteration. No active principles from prior experiments.

**Key mechanism insight from probing:** DPP's default V=100 creates an LHS term (V·c_D/2 = 650000μs) that overwhelms all queue-based terms, making it behave identically to AlwaysDisaggregate in all sub-saturation regimes. The Z-feedback mechanism only activates when TTFT consistently exceeds the SLO target, which requires either very low V or very tight SLO relative to actual TTFT. This will be the focus of iter-2: finding the V regime where DPP's adaptive behavior actually engages.
