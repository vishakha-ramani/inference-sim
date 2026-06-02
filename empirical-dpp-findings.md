# Empirical DPP Disaggregation Decider — Findings

**Branch:** `feat/empirical-dpp`  
**Campaign ID:** `edpp-realistic-workloads`  
**Tool:** [nous](https://github.com/vishakha-ramani/agentic-strategy-evolution) — hypothesis-driven experimentation framework  
**Simulator:** BLIS (this repo)  
**Date:** June 2026

---

## The Problem

When a request arrives at an LLM inference cluster with P/D disaggregation, the router
must decide: send this request to a dedicated **prefill server** (where its prompt is
processed, then the resulting state transferred over the network to a **decode server**),
or handle it entirely **locally** on one of the decode servers.

The production heuristic shipped in llm-d is **PrefixThresholdDecider**: disaggregate
when the number of uncached prompt tokens exceeds a fixed threshold N (default N=16).
This is cache-aware but **queue-blind** — it ignores how busy the prefill and decode
pools actually are.

This work asks: can a simple **queue-depth signal** do better?

---

## The Algorithm: EmpiricalDPPDecider

A Drift-Plus-Penalty inspired policy with no analytically-derived constants.
All system parameters are estimated from observations at runtime.

**Decision rule** (evaluated at each request arrival):

```
disaggregate  iff  V · η · Q_D  >  Q_P  +  Z · κ̂
```

| Symbol | Meaning | How it's set |
|--------|---------|-------------|
| `Q_D` | Aggregate decode queue depth | Live from RouterState.Snapshots |
| `Q_P` | Aggregate prefill queue depth | Live from RouterState.PrefillSnapshots |
| `V` | Adaptive penalty weight | Updated each epoch: `V += α·(ITL_obs − ITL_target)/ITL_target` |
| `η` | Queue weight ratio | Fixed; operator-specified (default 1.0) |
| `Z` | Virtual TTFT queue | `Z = max(0, Z + TTFT_obs − d)` per completed disaggregated request |
| `κ̂` | Empirical KV-transfer cost ratio | EWMA of `ΔT_obs / W_P_approx` |
| `d` | TTFT SLO target | Operator-specified (default 100 ms) |

**What the operator provides:** ITL target (ms) and TTFT SLO (ms) — goals, not system
parameters. No knowledge of model internals, hardware constants, or service times needed.

**How it adapts:**
- `V` rises when observed ITL exceeds the target → more disaggregation to offload prefill
- `V` falls when ITL is comfortable → less disaggregation, lower TTFT
- `Z` grows when disaggregated TTFT misses the SLO → suppresses future disaggregation
- `κ̂` tracks the observed ratio of KV transfer time to prefill time → makes Z dimensionally consistent

---

## What Was Implemented

### New file: `sim/disaggregation_edpp.go`

Contains:
- `TTFTUpdater` interface — called after each completed disaggregated request
- `ObservationUpdater` interface — extends TTFTUpdater with transfer and ITL callbacks
- `EmpiricalDPPConfig` struct — constructor parameters (all operator goals, no system constants)
- `EmpiricalDPPDecider` struct — the full adaptive decider implementation
- `NewEmpiricalDPPDecider(cfg)` — constructor with safe defaulting
- `Decide()`, `UpdateTTFT()`, `UpdateTransferObservation()`, `UpdateRequestCompletion()`
- `CurrentV()`, `CurrentKappa()`, `CurrentZ()` — state accessors for observability

### Modified files

| File | Change |
|------|--------|
| `sim/router_state.go` | Added `PrefillSnapshots []RoutingSnapshot` field so deciders can observe both pool queue depths |
| `sim/bundle.go` | Registered `"edpp"` as a valid disaggregation decider name |
| `sim/cluster/deployment.go` | Added `EDPP*` config fields (`EDPPEta`, `EDPPTTFTSloD`, `EDPPITLTargetMs`, `EDPPVInit`, `EDPPVMin`, `EDPPVMax`, `EDPPAlpha`, `EDPPEpochSize`) |
| `sim/cluster/cluster.go` | Added `"edpp"` factory case; populated `state.PrefillSnapshots` before each disaggregation decision; wired `TTFTUpdater` and `ObservationUpdater` callbacks at decode completion |
| `cmd/root.go` | Added 9 `--edpp-*` CLI flags (see [CLI flags](#cli-flags)) |

---

## CLI Flags

```
--pd-decider edpp           Select EmpiricalDPPDecider

--edpp-ttft-slo-d   100.0   TTFT SLO target d in ms (Z grows when exceeded)
--edpp-itl-target    30.0   ITL target in ms (V adapts toward this)
--edpp-eta            1.0   Queue weight ratio η
--edpp-v-init         1.0   Initial V
--edpp-v-min          0.05  Minimum V (prevents collapse to never-disaggregate)
--edpp-v-max         50.0   Maximum V (prevents runaway to always-disaggregate)
--edpp-alpha          0.1   V step size per epoch
--edpp-epoch-size      50   Completed requests per V update
```

---

## Workloads Used

Two production-mirror workloads from the
[inference-perf catalog](https://github.com/kubernetes-sigs/inference-perf):

### interactive-chat (`inference-perf-interactive-chat.yaml`)

Short, conversational requests with a moderate shared prefix.

| Parameter | Value |
|-----------|-------|
| Model | meta-llama/llama-3.3-70b-instruct |
| Prefix length | 5,000 tokens (shared system prompt) |
| Per-turn input | lognormal, median ~39 tokens |
| Output | gaussian, mean 300 tokens |
| Turns per session | ~4 |
| Think time between turns | 45 s |

**Key property:** 39-token median input means uncached tokens per request are small and
variable — PrefixThreshold(N=16) disaggregates ~89% unconditionally.

### code-generation (`inference-perf-code-generation.yaml`)

Long requests with a large repository-context prefix.

| Parameter | Value |
|-----------|-------|
| Model | meta-llama/llama-3.3-70b-instruct |
| Prefix length | 30,000 tokens (repo context) |
| Per-turn input | lognormal, median ~1,173 tokens |
| Output | lognormal, mean 425 tokens |
| Turns per session | ~15 |
| Think time between turns | 15 s |

**Key property:** 1,173-token median input means **every** request exceeds N=16 —
PrefixThreshold(N=16) always disaggregates 100%, regardless of load.

---

## Experimental Setup

**Topology:** 2 prefill instances + 2 decode instances  
(`--num-instances 4 --prefill-instances 2 --decode-instances 2`)

**Latency model:** `trained-physics` (default)

**Baseline decider:** `--pd-decider prefix-threshold --pd-prefix-threshold 16`  
(llm-d's shipped default)

**Seeds:** 42, 123, 456, 789, 1001 (5 seeds for statistical robustness in iter-2/3)

---

## Results

### Iter-1 — Baseline characterization

Ran NeverDisaggregate, PrefixThreshold(N=16), and EDPP(SLO=100ms, ITL_target=30ms)
across rate sweeps on both workloads.

**Interactive-chat — key finding:**
- PT(16) disaggregates ~89% of requests at every rate (unconditional)
- EDPP matches Never within 0.2% ITL at low load; V hits floor (0.05)
- At saturation (rate=100): EDPP TTFT = **49.6ms** vs PT = **97.7ms** vs Never = **253ms**
  — EDPP selectively disaggregates 64.6%, PT over-disaggregates, Never saturates

**Code-generation — key finding:**
- PT(16) disaggregates **100%** unconditionally, adding KV-transfer overhead at every rate
- PT TTFT is 25–46% worse than Never even at low load — a pure overhead tax
- EDPP matches Never at low load; at saturation EDPP's Z feedback engages

### Iter-2 — Statistical significance + V adaptation mechanism (interactive-chat)

Ran 5 seeds × {r50, r75, r100, r150} with PT vs EDPP. Added V_init ablation.

**h-main: CONFIRMED** — Zero overlap between EDPP and PT across all 5 seeds:
- r100: EDPP TTFT ∈ [49.1, 50.2] ms  vs  PT TTFT ∈ [74.8, 97.7] ms
- r150: EDPP TTFT ∈ [49, 53] ms  vs  PT TTFT ∈ [165, 250] ms

**h-ablation: CONFIRMED** — V_init=0.05 and V_init=50.0 produce byte-identical results.
**V adaptation is a no-op for interactive-chat.** The reason: prefill takes ~1 ms for
39-token inputs, so Q_P ≈ 0 always. The decision simplifies to `disaggregate iff Q_D > 0`.
Any V > 0 gives the same threshold. The adaptive V/Z/κ machinery runs correctly but
never influences the outcome.

**Effective policy for chat:** `disaggregate iff Q_D > 0` — load-proportional disaggregation
without any tuning.

**TTFT advantage ratio scales with rate:**

| Rate | EDPP disagg | PT/EDPP TTFT ratio |
|------|-------------|-------------------|
| r50 | 20.8% | 1.1× |
| r75 | 36.2% | 1.3× |
| r100 | 64.6% | **2.0×** |
| r150 | 82.6% | **3.0×** |

### Iter-3 — Z feedback active, V_init sensitivity (code-generation)

Ran 5 seeds × {r20, r30, r40, r50, r80} plus SLO sensitivity and V_init ablation.

**h-main: CONFIRMED** — PT is catastrophically worse at saturation:

| Rate | EDPP TTFT | PT TTFT | Never TTFT | EDPP disagg |
|------|-----------|---------|-----------|-------------|
| r20 (below sat) | ~72ms | ~95ms | ~72ms | 10% |
| r50 | ~120ms | ~900ms | ~350ms | 50% |
| r80 (heavy sat) | 99–147ms | **1,455–2,721ms** | 582ms | 30–54% |

PT is **worse than Never** at saturation — 100% disaggregation overloads the prefill pool
with 1,173-token requests, causing prefill queue buildup that Never avoids.

**h-ablation: PARTIALLY CONFIRMED** — V_init sensitivity is confirmed and seed-dependent.
V_init=50 always disaggregates *fewer* requests than V_init=1.0 (Z overcorrection):
it starts too aggressively, floods prefill, TTFT spikes, Z grows and suppresses
disaggregation for most of the run. Net: fewer total disaggregations despite higher V.

**Z feedback is active on code-gen** (unlike chat): expensive prefill builds Q_P,
Z accumulates, and the full `V·η·Q_D > Q_P + Z·κ̂` rule matters.

**SLO sensitivity (RP-7):** Relaxing SLO from 100ms to 200ms increases disaggregation
by 6–60% and reduces TTFT by 5–25% — confirms Z's SLO threshold is a meaningful
operator control for code-gen workloads.

---

## Summary of Principles (8 extracted)

| ID | Statement |
|----|-----------|
| RP-1 | PT(16) forces 100% disaggregation for code-gen (median 1173-token inputs), adding fixed KV-transfer overhead at every load level — 10–20× TTFT degradation vs EDPP at saturation |
| RP-2 | PT TTFT is worse than Never at ALL load levels for code-gen: 30–32% above Never below saturation, 4.7–20.9× at saturation |
| RP-3 | EDPP's V adaptation is a **no-op for short-input workloads** (chat): Q_P ≈ 0, decision reduces to `Q_D > 0`, any V > 0 gives identical results |
| RP-4 | EDPP beats PT by 2–3× on TTFT at saturation for chat via load-proportional selective disaggregation (64–83% vs PT's unconditional 89%) |
| RP-5 | EDPP disaggregation fraction scales monotonically with rate for code-gen: r20=9.8% → r80=50%; flattens near 50% at saturation due to Z feedback stabilization |
| RP-6 | EDPP matches Never TTFT within 2% below saturation for both workloads; PT adds 30–32% overhead even below saturation |
| RP-7 | Relaxing EDPP's TTFT SLO from 100ms to 200ms increases disaggregation 6–60% and reduces TTFT 5–25% for code-gen — Z's threshold is a meaningful tuning knob |
| RP-8 | V_init sensitivity for code-gen is seed-dependent: V_init=50 always disaggregates fewer requests than V_init=1.0 (Z overcorrection confirmed); optimal V_init is not simply "lowest" |

---

## How to Reproduce

### Prerequisites

```bash
git clone https://github.com/vishakha-ramani/inference-sim
cd inference-sim
git checkout feat/empirical-dpp
go build -o blis .
```

### Interactive-chat: EDPP vs PrefixThreshold

```bash
# EDPP — best configuration
./blis run --model meta-llama/llama-3.3-70b-instruct \
  --num-instances 4 --prefill-instances 2 --decode-instances 2 \
  --pd-decider edpp --edpp-ttft-slo-d 100 --edpp-itl-target 30 \
  --workload-spec inference-perf-interactive-chat.yaml \
  --aggregate-rate 100 --num-requests 2000 --seed 42
# Expected: mean_itl ≈ 23ms, mean_ttft ≈ 50ms, 0 dropped requests

# PrefixThreshold baseline
./blis run --model meta-llama/llama-3.3-70b-instruct \
  --num-instances 4 --prefill-instances 2 --decode-instances 2 \
  --pd-decider prefix-threshold --pd-prefix-threshold 16 \
  --workload-spec inference-perf-interactive-chat.yaml \
  --aggregate-rate 100 --num-requests 2000 --seed 42
# Expected: mean_ttft ≈ 98ms (2× worse than EDPP)
```

### Code-generation: EDPP vs PrefixThreshold

```bash
# EDPP
./blis run --model meta-llama/llama-3.3-70b-instruct \
  --num-instances 4 --prefill-instances 2 --decode-instances 2 \
  --pd-decider edpp --edpp-ttft-slo-d 100 --edpp-itl-target 30 \
  --workload-spec inference-perf-code-generation.yaml \
  --aggregate-rate 50 --num-requests 1000 --seed 42
# Expected: mean_ttft ≈ 120ms, ~50% disaggregation, 0 dropped

# PrefixThreshold baseline
./blis run --model meta-llama/llama-3.3-70b-instruct \
  --num-instances 4 --prefill-instances 2 --decode-instances 2 \
  --pd-decider prefix-threshold --pd-prefix-threshold 16 \
  --workload-spec inference-perf-code-generation.yaml \
  --aggregate-rate 50 --num-requests 1000 --seed 42
# Expected: mean_ttft ≈ 900ms (7.5× worse than EDPP), many drops
```

### Re-run the full nous campaign

```bash
cd /path/to/agentic-strategy-evolution
NOUS_ALLOW_AUTO_APPROVE=1 venv/bin/nous run examples/edpp-campaign.yaml --auto-approve
```

Full campaign artifacts are under `.nous/edpp-realistic-workloads/`.

---

## Open Questions

1. **V adaptation for heterogeneous workloads.** For chat, V never matters (Q_P ≈ 0).
   For code-gen, V_init sensitivity is seed-dependent. Is there a workload mix where
   V adaptation provides a stable, reproducible advantage?

2. **Combining queue-depth and cache signals.** EDPP ignores the KV cache state; PT
   ignores queue state. A hybrid that uses both signals (`disaggregate iff Q_D > Q_P AND
   uncached_tokens > threshold`) might dominate both on heterogeneous workloads.

3. **3P+3D and larger topologies.** All results are for 2P+2D. Does the advantage hold
   at larger scale? Does Z stabilization near 50% disaggregation persist?

4. **Non-Poisson arrivals.** All experiments use Poisson arrivals. Bursty arrivals (e.g.,
   Erlang/gamma with CV>1) may change Z accumulation dynamics.
