# Problem Framing: DPP V Sweep — Iter 2

## Research Question

Does DPP's Z-feedback mechanism produce a regime where partial disaggregation outperforms PrefixThreshold(N=16) on mean ITL, or is partial disaggregation a dominated strategy in the 1P+1D topology?

Iter-1 established that DPP(V=100) ≡ AlwaysDisaggregate ≡ PrefixThreshold in sub-saturation. This iteration explores the V/SLO parameter space where DPP makes *variable* decisions (disaggregation fraction between 0% and 100%) and tests whether that adaptive behavior produces better ITL than the static full-disaggregation baseline.

**Relevant source files:**
- `sim/disaggregation.go:127-141` — DPP Decide() equation: disaggregate iff `η·Q_D + V·c_D/2 > Q_P + Z·ΔT/W_P`
- `sim/disaggregation.go:146-148` — UpdateTTFT: `Z = max(0, Z + ttftUs - SLO)`
- `sim/cluster/cluster.go:429-436` — DPP factory: W_P=29900μs, c_D=13000μs hardcoded

## System Interface

- **Build:** `go build -o blis .` (validated, exit 0)
- **CLI flags (code evidence):**
  - `--pd-decider dpp` — `cmd/root.go:1026` selects DriftPlusPenaltyDecider
  - `--dpp-v V` — `cmd/root.go:1028` penalty weight (float64)
  - `--dpp-eta η` — `cmd/root.go:1029` queue weight ratio (float64, default 1.0)
  - `--dpp-ttft-slo-d D` — `cmd/root.go:1030` TTFT SLO in ms (float64, default 50.0)
  - `--pd-decider prefix-threshold` — `cmd/root.go:1026` PrefixThresholdDecider
  - `--pd-prefix-threshold N` — `cmd/root.go:1027` threshold (int, default 16)
  - `--metrics-path <file>` — `cmd/root.go:~980` writes cluster-aggregate JSON metrics to file
- **Output format:** JSON to stdout after `=== Simulation Metrics ===` header. `--metrics-path` writes cluster-aggregate metrics to a file.

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b \
  --prefill-instances 1 --decode-instances 1 --num-instances 2 \
  --pd-decider prefix-threshold --pd-prefix-threshold 16 \
  --rate 30 --num-requests 2000 \
  --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 \
  --seed 42 --metrics-path results/baseline_pt_r30.json
```

## Baseline Validation

Exit code 0. Cluster-aggregate metrics from `--metrics-path`:
- `itl_mean_ms`: 28.65
- `ttft_mean_ms`: 64.79
- `completed_requests`: 2000
- `dropped_unservable`: 0
- `preemption_count`: 0
- Disaggregated Requests: 1990/2000 (99.5%)

## Experimental Conditions

### Condition 1: DPP V sweep at rate=30, SLO=20ms (sub-saturation, active Z-feedback)

Sweep V ∈ {1.0, 3.0, 5.0, 10.0, 20.0, 50.0} with `--dpp-ttft-slo-d 20.0` at rate=30.

For each V value:
```bash
./blis run --model qwen/qwen3-14b \
  --prefill-instances 1 --decode-instances 1 --num-instances 2 \
  --pd-decider dpp --dpp-v <V> --dpp-ttft-slo-d 20.0 \
  --rate 30 --num-requests 2000 \
  --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 \
  --seed 42 --metrics-path results/dpp_v<V>_slo20_r30.json
```

Expected disaggregation fractions (from probes): V=1→3%, V=3→8%, V=5→10%, V=10→19%, V=20→35%, V=50→73%.

### Condition 2: DPP V sweep at rate=40, SLO=20ms (near-saturation)

Same V values as Condition 1, at rate=40 where PT begins dropping requests.

```bash
./blis run --model qwen/qwen3-14b \
  --prefill-instances 1 --decode-instances 1 --num-instances 2 \
  --pd-decider dpp --dpp-v <V> --dpp-ttft-slo-d 20.0 \
  --rate 40 --num-requests 2000 \
  --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 \
  --seed 42 --metrics-path results/dpp_v<V>_slo20_r40.json
```

### Condition 3: PrefixThreshold and NeverDisaggregate baselines at both rates

```bash
# PT at rate=40
./blis run --model qwen/qwen3-14b \
  --prefill-instances 1 --decode-instances 1 --num-instances 2 \
  --pd-decider prefix-threshold --pd-prefix-threshold 16 \
  --rate 40 --num-requests 2000 \
  --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 \
  --seed 42 --metrics-path results/baseline_pt_r40.json

# Never at rate=30
./blis run --model qwen/qwen3-14b \
  --prefill-instances 1 --decode-instances 1 --num-instances 2 \
  --pd-decider never \
  --rate 30 --num-requests 2000 \
  --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 \
  --seed 42 --metrics-path results/baseline_never_r30.json

# Never at rate=40
./blis run --model qwen/qwen3-14b \
  --prefill-instances 1 --decode-instances 1 --num-instances 2 \
  --pd-decider never \
  --rate 40 --num-requests 2000 \
  --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 \
  --seed 42 --metrics-path results/baseline_never_r40.json
```

### Condition 4: SLO sensitivity at V=5, rate=30

Sweep SLO ∈ {15, 20, 22, 25, 30} ms to characterize the transition from Z-dominated (always-local) to V-dominated (always-disaggregate).

```bash
./blis run --model qwen/qwen3-14b \
  --prefill-instances 1 --decode-instances 1 --num-instances 2 \
  --pd-decider dpp --dpp-v 5.0 --dpp-ttft-slo-d <SLO> \
  --rate 30 --num-requests 2000 \
  --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 \
  --seed 42 --metrics-path results/dpp_v5_slo<SLO>_r30.json
```

## Success Criteria

1. **ITL comparison (h-main):** If DPP at any V in {1-50} with SLO=20ms achieves mean ITL ≤ PT's 28.65ms at rate=30, DPP's adaptive behavior provides ITL benefit. Based on probes, we predict this will NOT occur.
2. **Completion rate (h-robustness):** At rate=40, DPP(V≤10, SLO=20ms) should complete significantly more requests (≥95%) than PT (~86%) by avoiding decode KV exhaustion.
3. **Transition sharpness (h-control-negative):** The SLO sweep should show a sharp transition from ~0% disaggregation (SLO≤20ms) to ~100% disaggregation (SLO≥25ms) at V=5, confirming Z-dominance creates a binary switch rather than smooth adaptation.

## Constraints

- All runs use seed=42 (INV-6 determinism, no multi-seed needed)
- 2000 requests per condition (sufficient for metric stability at rate≤50)
- Rate≤50 to avoid horizon truncation (INV-1: completed < num_requests)
- RP-1: DPP(V=100) produces byte-identical results to AlwaysDisaggregate — do not re-test this
- RP-3: KV transfer overhead of ~6.6ms mean TTFT penalty is a fixed cost of disaggregation

## Prior Knowledge

- **RP-1:** DPP(V=100, c_D=13000μs) is functionally equivalent to AlwaysDisaggregate. V must be ≤50 for adaptive behavior.
- **RP-2:** P/D disaggregation reduces ITL monotonically with load (1-15% improvement rate 5-30).
- **RP-3:** Disaggregation trades lower ITL for higher TTFT at low-moderate loads due to ~6.6ms KV transfer overhead.
- **RP-4:** At saturation (rate=50), disaggregating policies maintain low TTFT but drop ~30% of requests.
- **Probe finding (new):** DPP's Z-feedback creates a binary regime: SLO≥25ms → always disaggregates; SLO≤20ms → suppresses to <10% disaggregation. The intermediate V regime (1-50 with SLO=20) consistently produces higher ITL (33-36ms) than full disaggregation (28.65ms) at rate=30.
- **Probe finding (new):** η has zero observable effect because Z·ΔT/W_P dominates after initial disaggregation events, and Q_D remains too small relative to V·c_D/2.
