# Handoff: DPP vs PrefixThreshold — Iteration 2

## Goal

Run a DPP V sweep at two load points (rate=30 sub-saturation, rate=40 near-saturation) with SLO=20ms to characterize DPP's partial-disaggregation regime. Also sweep SLO at fixed V=5 to map the binary transition. Compare all results against PrefixThreshold(N=16) baseline. Test whether DPP's queue-aware feedback ever produces better ITL than the stateless cache-threshold approach.

## Key Discoveries

1. **DPP partial disaggregation regime identified:** V∈[1,50] with SLO=20ms at rate=30 produces disaggregation fractions from 3% (V=1) to 73% (V=50). The sweet spot for variable behavior is V∈[1-50].

2. **Partial disaggregation is ITL-dominated:** Every tested V in the active regime produces HIGHER mean ITL than PrefixThreshold at rate=30. V=1→33.69ms, V=5→36.16ms, V=10→35.87ms, V=50→30.34ms, vs PT=28.65ms. DPP converges to PT performance only as V→100 (always disaggregate).

3. **DPP's advantage is completion rate at saturation:** At rate=40, DPP(V≤10, SLO=20) completes all 2000 requests (vs PT's 1725) by suppressing disaggregation and avoiding decode KV exhaustion. But ITL is worse (38-40ms vs PT's 34ms) and TTFT is catastrophic (4800-6800ms vs PT's 326ms).

4. **Z-feedback creates a binary switch, not gradient:** At V=5, SLO≤20ms → disagg<15%. SLO≥25ms → disagg=100%. No stable intermediate. The transition is ~5ms wide (SLO 20→25ms).

5. **η has zero observable effect:** Sweeping η from 0.1 to 5.0 at V=5/SLO=20/rate=30 produces identical results. The Z·ΔT/W_P term dominates the RHS within ~10 requests, making the Q_D and Q_P terms negligible.

6. **DPP transfer time (ΔT) is computed from model config:** `cluster.go:400-420` computes dppTransferTimeUs from KV bytes per token and configured bandwidth. Fallback: 3406μs. Actual: ~6600μs mean observed in PD metrics.

7. **Rate=40 is the decode saturation cliff for disaggregating policies:** PT drops 275/2000 requests (14% dropped_unservable) with 256 preemptions. DPP(V≤10) avoids this by keeping requests local.

## System Interface

- **Build:** `go build -o blis .` (validated, exit 0)
- **Run baseline:** `./blis run --model qwen/qwen3-14b --prefill-instances 1 --decode-instances 1 --num-instances 2 --pd-decider prefix-threshold --pd-prefix-threshold 16 --rate 30 --num-requests 2000 --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 --seed 42 --metrics-path results/baseline_pt_r30.json`
- **Output format:** `--metrics-path <file>` writes cluster-aggregate JSON. Parse with `python3 -c "import json; d=json.load(open('<file>')); print(d['itl_mean_ms'])"`. PD metrics (disaggregation count) are printed to stdout only — grep for "Disaggregated Requests".
- **Baseline result:** ITL=28.65ms, TTFT=64.79ms, completed=2000, disagg=1990/2000 (PT, rate=30)

## Code Map

- `sim/disaggregation.go:129-141` — DPP Decide(): `lhs = η·Q_D + V·c_D/2; rhs = Q_P + Z·ΔT/W_P; disagg = lhs > rhs`. Check if disaggregation fraction varies between V values.
- `sim/disaggregation.go:146-148` — UpdateTTFT: `Z = max(0, Z + ttftUs - SLO)`. Check Z accumulation if disaggregation fraction is unexpected.
- `sim/cluster/cluster.go:429-436` — DPP factory: hardcoded W_P=29900μs, c_D=13000μs. These constants determine V's effective scale.
- `sim/cluster/cluster.go:400-420` — dppTransferTimeUs computation from model config + bandwidth.
- `sim/cluster/cluster.go:1255-1262` — Where UpdateTTFT is called: uses TransferCompleteTime - ArrivalTime as TTFT proxy.
- `sim/cluster/pd_metrics.go:64-141` — PD metrics collection. Look here if disaggregation count seems wrong.

## Code Targets

No code changes needed for iter-2 (pure flag-variation experiment).

## What I Tried That Didn't Work

1. **V∈{0.001, 0.003, 0.005, 0.008, 0.01} (from iter-1 suggestion)** — All produce disagg=1/2000 at SLO=20ms. Too small; Z dominates after the very first disaggregated request. V must be ≥0.1 for any variation.
2. **η sweep (0.1 to 5.0)** — Zero effect on results. Q_D is always negligible compared to V·c_D/2 in the LHS, and Z·ΔT/W_P dominates the RHS after a few requests. η is a dead parameter in this topology.
3. **Looking for V where DPP beats PT on ITL** — Swept V=1 through V=100 at rate=30/SLO=20. ITL monotonically approaches PT from above as V increases. No V exists where DPP is better; partial disaggregation is strictly ITL-dominated.
4. **Multi-flag shell expansion in for-loops** — `--pd-decider prefix-threshold --pd-prefix-threshold 16` doesn't expand correctly in a for-loop variable. Run each decider command separately.

## What I Excluded and Why

1. **η sweeps in the formal experiment** — Probes confirmed η has zero effect. Including it would add conditions without scientific value.
2. **V>50 with SLO=20ms** — At V=50, disagg=73% and ITL=30.34ms (already approaching PT's 28.65ms). At V=100, it matches PT exactly. Testing V>50 with tight SLO doesn't add information beyond confirming convergence to always-disaggregate.
3. **Rate≤20 conditions** — At low rates, all policies perform well and differences are <3ms. The interesting behavior is at rate=30 (moderate load) and rate=40 (near-saturation).
4. **Mixed-prefix workload YAML** — With prefix_tokens=480/prompt_tokens=512, PrefixThreshold disaggregates 99.5% regardless. Testing with different prefix ratios is a valid iter-3 direction but changes the workload, not the DPP parameters.
5. **DPP vs PT at rate=50** — Both policies have issues at rate=50 (PT drops 30%+, DPP's behavior depends on V). Rate=40 captures the saturation regime without horizon truncation complications.

## Evolution of Thinking

**Iter-1 assumption:** "V must be ~0.01 or below for Z-feedback to matter." 
**Correction:** V must be ≥0.1 (not 0.01) for any disaggregation variation. The iter-1 suggestion of V∈{0.001-0.01} was wrong because it confused the effective scale. V=0.01 gives V·c_D/2=65 which needs Z=570 to overcome — Z grows by 30000 per disaggregated request, so after 1 request Z≈30000 and Z·0.114=3420 >> 65.

**Key shift:** The research question has evolved from "find V where DPP beats PT" to "prove that no V exists where DPP partial disaggregation beats PT on ITL in this topology." The partial disaggregation regime is dominated — requests not disaggregated face the same ITL penalty as NeverDisaggregate, which is always worse than fully disaggregated processing.

**New insight:** DPP's value proposition is NOT better ITL but better completion rate at saturation. It's a stability controller that sacrifices ITL and TTFT to avoid decode KV exhaustion. This reframes the question for iter-3: is DPP a useful back-pressure mechanism when combined with a primary disaggregation strategy?

## Current Status

- **Validated:** V sweep commands work at both rates. Metrics extraction via `--metrics-path` confirmed. PD metrics extraction from stdout confirmed. All parameter combinations tested produce expected disaggregation fractions.
- **Uncertain:** Whether the Z-feedback has bang-bang oscillation (Z grows → disagg stops → Z decays slowly → disagg resumes) or truly reaches steady state. This would show up as periodic disaggregation patterns across the request stream, but we can't observe per-request decisions from stdout.
- **Suggested next (iter-3):** Test whether a hybrid approach (DPP as back-pressure on top of PT) can beat either policy alone. Alternatively, explore larger topologies (2P+2D) where decode KV pressure is lower and the ITL benefit of disaggregation is less saturating.

## Warnings & Constraints

1. **`--metrics-path` writes cluster-aggregate only** — PD metrics (disaggregation count, prefill/decode throughput) are printed to stdout. Capture stdout separately if you need disagg count.
2. **The V→ITL curve is NOT monotonically decreasing** — V=5 (ITL=36.16) is worse than V=3 (ITL=35.69) at rate=30. This is because higher V disaggregates more requests → more compete at decode KV → intermittent queue pressure. The minimum ITL is at V=100 (full disaggregation).
3. **Rate=40 completed_requests varies by decider** — PT completes 1725/2000, DPP(V≤10) completes 2000/2000. When comparing ITL, note that PT's ITL is computed over fewer (surviving) requests which may have lower average latency than the full population.
4. **SLO is in milliseconds on CLI but converted to microseconds internally** — `--dpp-ttft-slo-d 20.0` becomes 20000μs in the DPP equation. Don't confuse units.
5. **stderr warnings are normal** — "Using model defaults for TP", "Unable to detect TP from model" etc. Redirect 2>/dev/null for clean parsing.
6. **Disaggregation fraction for 500 requests (probes) vs 2000 requests (formal runs) will differ slightly** — Z accumulates over time, so longer runs may have slightly lower disagg fraction than short probes at the same V/SLO.
