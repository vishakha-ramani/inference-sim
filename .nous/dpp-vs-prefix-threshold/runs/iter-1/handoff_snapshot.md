# Handoff: DPP vs PrefixThreshold — Iteration 1

## Goal

Run a rate sweep (5, 10, 20, 30, 50 req/s) comparing 4 disaggregation policies (Never, Always, PrefixThreshold N=16, DPP V=100 SLO=50ms) on a 1P+1D topology with prefix-heavy workload. Collect ITL, TTFT, and disaggregation fraction at each point. This establishes the baseline for iter-2's DPP parameter sweep.

## Key Discoveries

1. **DPP(V=100) ≡ AlwaysDisaggregate** at all tested loads. The V·c_D/2 = 100×13000/2 = 650000 term in the LHS completely dominates, making DPP insensitive to queue depths or TTFT feedback. V must be ~0.01 or below for Z-feedback to matter.

2. **PrefixThreshold(N=16) disaggregates 99%+ of requests** when prompt=512, prefix=480 (32 uncached tokens > 16). It behaves nearly identically to Always at sub-saturation rates.

3. **The CLI flag is `--prompt-tokens` (not `--input-tokens`)**. Using `--input-tokens` causes "unknown flag" error.

4. **Rate=50 is the saturation cliff**: NeverDisaggregate TTFT jumps from 65ms (rate=30) to 6293ms (rate=50). Disaggregating policies remain at ~58ms.

5. **DPP's Z-feedback only engages when SLO < actual local TTFT (~30ms)**. With SLO=10ms, Z grows rapidly after the first disaggregated request, suppressing all subsequent disaggregation. With SLO≥25ms, Z stays near zero and DPP always disaggregates. The transition is sharp, not gradual.

6. **Hardcoded DPP parameters**: W_P=29900μs (prefill service time), c_D=13000μs (decode cost per token). These are set at `cluster.go:434-435` and affect the equation balance.

7. **At rate=50+, `completed_requests` may be < `num_requests`** due to simulation horizon. Use 2000 requests at rate≤50 to ensure all complete. At rate=100, only 864/2000 complete within the default 300s horizon.

## System Interface

- **Build:** `go build -o blis .` (validated, exit 0)
- **Run baseline:** `./blis run --model qwen/qwen3-14b --prefill-instances 1 --decode-instances 1 --num-instances 2 --pd-decider never --rate 30 --num-requests 2000 --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 --seed 42`
- **Output format:** Stdout JSON blocks after `=== Simulation Metrics ===` header (per-instance + cluster aggregate). PD metrics printed separately after `=== PD Metrics ===`. Use `--metrics-path <file>` for file output.
- **Baseline result:** ITL=32.86ms, TTFT=64.85ms, completed=2000 (NeverDisaggregate, rate=30)

## Code Map

- `sim/disaggregation.go:129-141` — DPP Decide() core equation. Check if DPP is varying decisions between requests.
- `sim/disaggregation.go:146-147` — UpdateTTFT: Z = max(0, Z + ttft - SLO). Check Z accumulation behavior.
- `sim/disaggregation.go:230-244` — PrefixThreshold Decide(): cachedBlocks from cacheQuery, then nonCached > threshold.
- `sim/cluster/cluster.go:429-436` — DPP factory: W_P=29900, c_D=13000 hardcoded. Check dppTransferTimeUs computation.
- `sim/cluster/cluster.go:1887-1914` — Where disaggregation decision is made during routing.
- `sim/cluster/pd_metrics.go:64-141` — CollectPDMetrics(): counts DisaggregatedCount. Check for per-policy stats.
- `cmd/root.go:1026-1034` — CLI flag definitions for all PD-related flags.
- `sim/metrics.go:155-160` — Where MetricsOutput JSON is printed to stdout.

## Code Targets

No code changes needed for iter-1 (pure flag-variation experiment).

## What I Tried That Didn't Work

1. **`--input-tokens` flag** — doesn't exist. The correct flag is `--prompt-tokens`.
2. **Omitting `--num-instances 2`** — with only prefill+decode instances and default num-instances=1, the topology validation may behave unexpectedly. Always specify `--num-instances 2` explicitly.
3. **Workload YAML with `aggregate_rate: 0`** — requires per-client `trace_rate` in lifecycle windows. Use `aggregate_rate: 30` (or desired rate) with `rate_fraction` instead.
4. **Trying to vary DPP behavior with V sweep** at default SLO=50ms — V has no effect because even V=0.001 gives V·c_D/2 = 6.5, and with SLO≥TTFT, Z≈0 so RHS≈Q_P≈0. DPP always disaggregates regardless of V when SLO is above actual TTFT.
5. **Multi-flag `--pd-decider dpp --dpp-v 100` in shell variable expansion** — fails when used in a for-loop variable. Execute each decider's command separately.

## What I Excluded and Why

1. **DPP parameter sweeps** — reserved for iter-2 once we know the baseline. The default V=100 is documented behavior; we need to characterize it, not optimize it yet.
2. **Mixed-prefix workload YAML** — created (`inputs/workload_mixed_prefix.yaml`) but not used in the experiment. With prefix_tokens=480, PrefixThreshold already disaggregates 99%+, so mixing groups doesn't differentiate policies in iter-1. Useful for iter-2 when testing cache-aware vs queue-aware tradeoffs.
3. **Bernoulli decider** — stochastic baseline, interesting but not part of the DPP-vs-PT core question.
4. **Multi-seed runs** — not needed for iter-1 (deterministic simulation, INV-6). Same seed always produces same results.
5. **Rate >50** — causes horizon truncation (completed < num_requests), complicating metric interpretation.

## Evolution of Thinking

**Initial assumption:** DPP's queue-feedback would create load-adaptive behavior different from PrefixThreshold at varying rates.

**Discovery:** DPP(V=100) is mathematically equivalent to AlwaysDisaggregate because V·c_D/2 = 650000 dominates all other terms. The parameter V was chosen in the paper's context (where c_D may have different semantics), but in BLIS's implementation, c_D=13000μs is a large constant that amplifies V dramatically.

**Revised understanding:** The iter-1 experiment will confirm this equivalence empirically. The interesting DPP behavior requires either (a) much smaller V (iter-2), or (b) tighter SLO that makes Z grow significantly (also iter-2). The key scientific question shifts from "does DPP beat PT?" to "at what V/SLO does DPP's adaptive behavior activate, and is that regime useful?"

## Current Status

- **Validated:** All 4 decider commands work correctly. Output format understood. Rate sweep parameters chosen (5-50). 2000 requests sufficient for all rates≤50.
- **Uncertain:** Whether PrefixThreshold vs Always show any meaningful difference (probing shows them within <0.5% at most rates). The cache hit rate's effect on PT's decisions needs more exploration.
- **Suggested next (iter-2):** Sweep V∈{0.001, 0.003, 0.005, 0.008, 0.01} with SLO∈{10, 15, 20, 25} at rate=30 to find the V/SLO regime where DPP partially disaggregates (neither 0% nor 100%). Then compare that regime's ITL/TTFT against PrefixThreshold.

## Warnings & Constraints

1. **`--metrics-path` writes only one file** — if both per-instance and cluster metrics are needed, parse stdout. The file contains the last-printed metrics block (cluster aggregate).
2. **DPP output at default V is byte-identical to Always** — do NOT assume DPP is broken if results match Always exactly. This is expected mathematical behavior.
3. **Prefix-tokens=480 with prompt-tokens=512 leaves only 32 uncached tokens** — this means PrefixThreshold(N=16) will disaggregate ~99% of requests. To test PT selectivity, you'd need prefix_tokens closer to prompt_tokens (e.g., prefix_tokens=500 → only 12 uncached → PT keeps local).
4. **stderr contains logrus warnings** (model defaults, TP detection). These are informational only. Redirect stderr to /dev/null for clean output parsing.
5. **The `=== Simulation Metrics ===` header appears multiple times** — once per instance + once for cluster. To get cluster-level metrics, parse the last occurrence or filter for `"instance_id": "cluster"`.
