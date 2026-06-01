# Handoff: DPP vs PrefixThreshold — Iteration 3

## Goal

Run a full rate sweep (30-60 req/s) in a 2P+2D topology comparing DPP(V=5, SLO=20ms), DPP(V=10, SLO=20ms), PrefixThreshold(N=16), and NeverDisaggregate. Test whether DPP's disaggregation-suppression strategy inverts the ITL ranking vs PT when decode capacity is distributed across multiple instances. Also run DPP(V=5, SLO=22ms) as negative control to confirm Z-suppression is the causal mechanism.

## Key Discoveries

1. **Ranking inversion in 2P+2D:** PrefixThreshold is the WORST performer in 2P+2D topology at all tested rates (30-60 req/s). PT gives ITL=24.8-32.7ms with drops starting at rate=35 (13/2000) escalating to 634/2000 at rate=55. NeverDisaggregate is best (ITL=18.2-27.0ms, zero drops), and DPP(V=5, SLO=20) is a close second (ITL=18.8-29.3ms, zero drops through rate=55).

2. **PT saturates decode pool:** In 2P+2D, PT disaggregates ~99% of requests (1980/2000 at rate=30). All decode work concentrates on 2 decode-only instances. At rate≥35, decode KV exhaustion causes preemptions and drops. Meanwhile, the 2 prefill instances process prefill fast but create decode bottleneck.

3. **DPP "keep local" is optimal in 2P+2D:** DPP(V=5, SLO=20) disaggregates only 10-13% of requests (203-258 out of 2000 across rates 30-60). The remaining ~90% execute locally on decode pods (both prefill+decode). Since uncached tokens are only 32 (512-480=32), local prefill is fast (~1ms). This avoids 6.6ms mean KV transfer AND decode KV exhaustion.

4. **NeverDisaggregate uses only decode pods:** With `--pd-decider never` in a 2P+2D topology, executeDisaggregatedRouting still fires but disaggregate=false sends ALL requests to decode pool only. The 2 prefill instances sit completely idle. Never's excellent ITL comes from distributing 30-60 req/s across 2 full-capability decode instances with zero overhead.

5. **DPP(V=10) boundary:** V=10 disaggregates ~22% at rate=45 (441/2000). Still completes 2000/2000. V=20 crosses the drop boundary (150 drops at rate=45). V=50+ converges to PT behavior.

6. **Phase transition confirmed in 2P+2D:** DPP(V=5, SLO=22ms) at rate=50 → 1402/2000 disaggregated, 598 dropped, ITL=34.84ms. Matches PT-like behavior, confirming RP-7 holds in the larger topology.

7. **Mixed topology (1P+1D+2PD) shows PT working well:** ITL=16.28ms at rate=45 with zero drops. PrefillDecode instances balance both pools. This is a potential iter-4 direction but out of scope here.

## System Interface

- **Build:** `go build -o blis .` (validated, exit 0)
- **Run baseline (PT, 2P+2D):** `./blis run --model qwen/qwen3-14b --num-instances 4 --prefill-instances 2 --decode-instances 2 --pd-decider prefix-threshold --pd-prefix-threshold 16 --rate 45 --num-requests 2000 --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 --seed 42 --metrics-path results/baseline_pt_2p2d_r45.json`
- **Output format:** `--metrics-path <file>` writes cluster-aggregate JSON. PD metrics (disaggregation count) printed to stdout — grep for "Disaggregated Requests". Dropped count in both JSON (`dropped_unservable`) and stdout (`Dropped Unservable:`).
- **Baseline result (PT, rate=45):** ITL=31.84ms, TTFT=60.09ms, completed=1625, dropped=375, disagg=1609/2000

## Code Map

- `sim/disaggregation.go:129-141` — DPP Decide(): `lhs = η·Q_D + V·c_D/2; rhs = Q_P + Z·ΔT/W_P`. In 2P+2D, Q_D sums 2 decode snapshots, Q_P sums 2 prefill snapshots. Check if disaggregation count differs from expected.
- `sim/disaggregation.go:146-148` — UpdateTTFT: `Z = max(0, Z + ttftUs - SLO)`. Check Z accumulation if disaggregation fraction is unexpected.
- `sim/cluster/cluster.go:429-436` — DPP factory: hardcoded W_P=29900μs, c_D=13000μs. These constants determine V's effective scale.
- `sim/cluster/cluster.go:1166-1181` — buildPoolFilteredSnapshots: filters by pool role + IsRoutable(). Relevant if instances unexpectedly appear in wrong pool.
- `sim/cluster/cluster.go:1889-1914` — executeDisaggregatedRouting: when Disaggregate=false, request goes to the pre-selected decode pod. This is why "keep local" works — decode pods handle both phases.
- `sim/cluster/cluster.go:1255-1262` — UpdateTTFT called with TransferCompleteTime - ArrivalTime. Check here if Z growth seems wrong.
- `sim/cluster/pool.go:115-132` — BuildPoolMembershipFromIndices: instances 0..prefill-1 are Prefill, next decode are Decode. In 2P+2D: instance_0,1=Prefill, instance_2,3=Decode.
- `sim/cluster/pd_metrics.go:64-141` — PD metrics collection. Look here if disaggregation count or dropped_at_decode_kv seems wrong.

## Code Targets

No code changes needed for iter-3 (pure flag-variation experiment).

## What I Tried That Didn't Work

1. **Short runs (500 requests) at high rates** — 500 requests at rate=150 complete before saturation builds. Need 2000 requests to observe steady-state decode KV exhaustion.
2. **Looking for PT's crossover point in 2P+2D** — Expected PT to eventually beat Never at very high rates (>80 req/s) due to prefill offloading. Instead, Never maintains 100% completion even at rate=150/2000req with ITL~38ms. The 2 decode instances handle the full load without needing prefill offloading. PT never catches up because its decode-only strategy is fundamentally bottlenecked.
3. **DPP(V=100, SLO=20ms) in 2P+2D** — Produces identical results to DPP(V=50) at rate=45 (both fully disaggregate, 1541/2000 completed). Confirms RP-1 holds in multi-instance.
4. **DPP SLO sweep (22-40ms) at rate=50** — All SLO≥22ms produce identical results (1402 disagg, 598 dropped). Once above the phase transition, the exact SLO value doesn't matter because full disaggregation emerges immediately.

## What I Excluded and Why

1. **AlwaysDisaggregate as separate condition** — DPP(V=100) ≡ AlwaysDisaggregate (RP-1), and PT already disaggregates 99%+. Including Always would be redundant with PT in this topology.
2. **Rate > 60 req/s** — At rate=60, PT drops 562/2000 (28%). Higher rates just increase drop fraction without new insight. The interesting regime is 30-60 where the transition happens.
3. **Mixed-prefix workload YAML** — With prefix_tokens=480/prompt_tokens=512, all requests have 32 uncached tokens. PT disaggregates whenever uncached > 16, so all requests disaggregate. A mixed workload with some prefix_tokens=0 would give PT the option NOT to disaggregate some requests, potentially improving its behavior. This is a valid iter-4 direction.
4. **Multi-seed runs** — All runs use seed=42 for determinism. Multi-seed would test noise sensitivity but is unnecessary given the 20%+ ITL differences observed.
5. **prefill-decode-instances topology** — The 1P+1D+2PD topology showed PT working well (ITL=16.28ms), suggesting the topology itself determines which policy wins. This is an iter-4 question: "which topology+policy combination is globally optimal?"

## Evolution of Thinking

**Iter-2 conclusion:** "DPP's value is NOT better ITL but better completion rate at saturation (stability controller)."

**Iter-3 correction:** DPP's value depends on topology. In 2P+2D, DPP achieves BOTH better ITL AND better completion than PT. The mechanism shifts from "DPP suppresses disaggregation → worse ITL but no drops" (1P+1D) to "DPP suppresses disaggregation → better ITL AND no drops" (2P+2D) because keeping requests local on 2 decode instances is strictly better than overloading 2 decode instances with remote KV from 2 prefill instances.

**Key reframe:** The question isn't "DPP vs PT" in isolation — it's "which policy matches which topology?" PT works well when prefill offloading genuinely relieves decode (single decode instance, or mixed PD instances). DPP works well when disaggregation creates more pressure than it relieves (multiple decode instances that can handle local prefill cheaply). The prefix-heavy workload (32 uncached tokens) makes local prefill nearly free, amplifying DPP's advantage.

**New insight:** NeverDisaggregate is actually the "oracle best" in 2P+2D with this workload because it never incurs KV transfer overhead and distributes load across 2 decode pods. DPP is the best *adaptive* policy because it approximates Never's behavior (10% disagg is close to 0%) while retaining the ability to shift to disaggregation if conditions change. PT is a bad match for dedicated-pool topology when prefill is cheap.

## Current Status

- **Validated:** All 5 condition commands work in 2P+2D (PT, Never, DPP V=5, DPP V=10, DPP V=5 SLO=22). Metrics extraction via `--metrics-path` confirmed. Disagg count from stdout confirmed. All probe results consistent.
- **Uncertain:** (1) Whether the V=10 boundary for 100% completion is exactly at rate=50 or higher — the probe at rate=45 shows 100% but rate=50+ wasn't tested for V=10. (2) Whether the ranking inversion holds with a mixed-prefix workload (some requests with prefix_tokens=0 that PT would NOT disaggregate). (3) Whether the advantage persists at 3P+3D or larger topologies.
- **Suggested next (iter-4):** (1) Test mixed-prefix workload (50% prefix_tokens=480, 50% prefix_tokens=0) where PT gets cache signal to make selective decisions. (2) Test prefill-decode-instances topology (1P+1D+2PD) where shared instances balance both pools. (3) Explore whether DPP with SLO tuned to ~55ms (above observed TTFT in 2P+2D at low queue) could achieve full disaggregation with back-pressure only at saturation — a "smart Always" mode.

## Warnings & Constraints

1. **NeverDisaggregate wastes prefill instances in P+D topology** — When `--pd-decider never` is used with `--prefill-instances 2 --decode-instances 2`, only the 2 decode instances are active. Prefill instances are completely idle. This means Never's "best ITL" is with 50% of the cluster unused — not a fair resource comparison.
2. **PT's "Disaggregated Requests" count includes requests that were later dropped** — At rate=45, "Disaggregated Requests: 1609" but only "completed_requests: 1625" (and 375 dropped). The disagg decision happens before the request might be dropped at decode KV allocation.
3. **stderr warnings are normal** — "Using model defaults for TP", "Unable to detect TP from model" etc. Redirect 2>/dev/null for clean parsing.
4. **SLO is in milliseconds on CLI but microseconds internally** — `--dpp-ttft-slo-d 20.0` → 20000μs in equation.
5. **Per-instance metrics in stdout show instance_0,1 (prefill) with 0 ITL** — Prefill instances report ITL=0 because they only do prefill, not decode tokens. Only instance_2,3 (decode) show real ITL values. The cluster aggregate correctly combines decode instances only.
6. **DPP disagg count increases slightly with rate** — At rate=30 it's 203, at rate=60 it's 258. This is because higher arrival rate produces higher Q_D which increases the LHS of the DPP inequality before Z catches up.
