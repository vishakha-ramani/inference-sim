# DPP vs PrefixThreshold: Experiment Findings

**Branch:** `feat/dpp-decider`  
**Campaign ID:** `dpp-vs-prefix-threshold`  
**Tool:** [nous](https://github.com/vishakha-ramani/agentic-strategy-evolution) — hypothesis-driven experimentation framework  
**Simulator:** BLIS (this repo)  
**Date:** June 2026

---

## Background: What Is P/D Disaggregation?

When an LLM processes a request, it does two phases of work:

1. **Prefill** — reads your entire prompt and produces the first output token. This is compute-heavy and proportional to how many input tokens you send.
2. **Decode** — generates the remaining output tokens one at a time. Each step is relatively cheap but there are many of them.

**Prefill/Decode (P/D) disaggregation** splits these two phases onto *different* machines. Requests are routed to a dedicated *prefill server*, which processes the prompt; the resulting state (called a KV cache) is then transferred over the network to a dedicated *decode server*, which generates the output.

The benefit: prefill-heavy requests stop interfering with ongoing decode work, lowering inter-token latency (ITL). The cost: every disaggregated request pays a network transfer fee (~6.6 ms in our simulation).

**The routing decision** — whether to disaggregate a given request or handle it locally — is made by a *disaggregation decider*.

---

## The Two Policies Under Test

### Policy A: PrefixThresholdDecider (the llm-d heuristic)

```
if (uncached_prefill_tokens > N):
    disaggregate
else:
    keep local
```

Default threshold `N = 16` — matching the value shipped in llm-d's production configs. It is **cache-aware but queue-blind**: it looks at how much new prefill work this request brings, and routes it to the prefill server if that work exceeds the threshold. It has no knowledge of how busy the prefill or decode queues are, and no feedback from observed latencies.

### Policy B: DriftPlusPenaltyDecider (DPP)

A Lyapunov optimization / drift-plus-penalty algorithm. At each decision it evaluates:

```
disaggregate  if  η·Q_D + V·(c_D/2)  >  Q_P + Z·(ΔT/W_P)
keep local    otherwise
```

Where:
- `Q_P`, `Q_D` — number of requests currently queued in the prefill and decode pools
- `V` — a tunable penalty weight (larger V → more willing to disaggregate)
- `η` — a weight balancing the two queue depths (default 1.0)
- `Z` — a *virtual TTFT queue*, updated after each completed request as `Z = max(0, Z + observed_TTFT - SLO_target)`. When TTFT consistently exceeds the SLO, Z grows and suppresses future disaggregation.
- `ΔT` — KV transfer time, `W_P` — mean prefill service time (both hardcoded from model calibration)

DPP is **queue-aware and TTFT-SLO-aware** but **cache-blind**: it knows how congested each pool is and reacts to TTFT violations, but it does not look at per-request KV cache state.

---

## What Was Implemented

The DPP decider (`DriftPlusPenaltyDecider`) and `BernoulliDisaggregate` were implemented and merged into this branch (`feat/dpp-decider`, commit `d8e99394`).

### Files touched (this branch, relevant to disaggregation)

| File | What changed |
|------|-------------|
| `sim/disaggregation.go` | Added `BernoulliDisaggregate` (lines 73–87) and `DriftPlusPenaltyDecider` (lines 89–148). Implemented `Decide()` and `UpdateTTFT()`. Added `TTFTUpdater` interface. |
| `sim/router_state.go` | Added `PrefillSnapshots []RoutingSnapshot` field so deciders can observe both pool queue depths. |
| `sim/bundle.go` | Registered `"bernoulli"` and `"dpp"` as valid decider names. |
| `cmd/root.go` | Added CLI flags: `--pd-decider dpp`, `--dpp-v`, `--dpp-eta`, `--dpp-ttft-slo-d`. |
| `sim/cluster/cluster.go` | Wired DPP factory (hardcoded `W_P=29900 μs`, `c_D=13000 μs` for qwen3-14b). Calls `UpdateTTFT` after each completed disaggregated request. |

The `PrefixThresholdDecider` was already present on `main`.

---

## Experimental Setup

### Simulator

BLIS (`blis run`) — discrete-event simulation of LLM inference. No real GPU required. Poisson arrivals, trained-physics latency model (`--latency-model trained-physics`).

### Workload used (synthetic)

All three iterations used the same fixed workload:

```
Model:          qwen/qwen3-14b
Prompt tokens:  512  (--prompt-tokens 512)
Prefix tokens:  480  (--prefix-tokens 480)  → 32 uncached tokens per request
Output tokens:  128  (--output-tokens 128)
Arrivals:       Poisson  (--rate λ)
Requests:       2000 per run  (--num-requests 2000)
Seed:           42  (--seed 42)
```

This is a **prefix-heavy workload**: 480 of the 512 input tokens are a shared prefix already in the KV cache. Only 32 tokens are new and need actual prefill work. This makes the per-request prefill cheap (~1 ms) and gives `PrefixThresholdDecider` a meaningful cache signal: every request exceeds `N=16` and gets disaggregated.

**Note:** This is a simplified synthetic workload. The repo also contains `inference-perf-code-generation.yaml` and `inference-perf-interactive-chat.yaml` (mirrors of the kubernetes-sigs/inference-perf catalog) which represent more realistic production workloads. These were not used in this campaign — see [Open Questions](#open-questions).

### Tool: nous

[nous](https://github.com/vishakha-ramani/agentic-strategy-evolution) is an AI-driven experimentation loop. It runs three phases per iteration:
1. **Design** (Claude Opus) — explores the code, reads the existing results, and writes a hypothesis bundle with testable predictions.
2. **Execute+Analyze** (Claude Sonnet) — builds BLIS, runs all conditions, parses results, writes findings.
3. **Human gate** — pauses for review before proceeding to the next iteration.

All campaign artifacts live in `.nous/dpp-vs-prefix-threshold/`.

---

## What Was Run: Three Iterations

### Iteration 1 — Baseline characterization (1P+1D topology)

**Topology:** 1 prefill instance + 1 decode instance (`--num-instances 2 --prefill-instances 1 --decode-instances 1`)  
**Conditions:** NeverDisaggregate, AlwaysDisaggregate, PrefixThreshold (N=16), DPP (V=100, SLO=50ms)  
**Rate sweep:** 5, 10, 20, 30, 50 req/s → 20 total runs

**Results:**

| Rate (req/s) | Never ITL | PT / Always / DPP ITL | Never TTFT | PT TTFT |
|-------------|-----------|----------------------|------------|---------|
| 5 | 29.5ms | 29.2ms | 43ms | 50ms |
| 20 | 30.8ms | 28.5ms (+7.4% better) | 46ms | 52ms |
| 30 | 33.6ms | 28.65ms (+14.7% better) | 52ms | 65ms |
| 50 | — | — | **6,293ms** (Never saturates) | 57ms |

**Key finding:** DPP(V=100) is **byte-identical to AlwaysDisaggregate**. The constant term `V·c_D/2 = 100 × 13000/2 = 650,000 μs` in the DPP threshold equation is so large it drowns all queue and Z-feedback terms entirely. DPP at this V is not adaptive at all.

**Principles extracted:**
- RP-1: DPP(V=100) ≡ AlwaysDisaggregate in all sub-saturation regimes
- RP-2: Disaggregation reduces ITL by 1%–15% as load grows from 5→30 req/s
- RP-3: Disaggregation adds ~6.6ms KV-transfer overhead to TTFT, worth it only near queue saturation
- RP-4: At saturation (rate=50), Never's TTFT explodes 109× while disaggregating policies maintain <100ms

---

### Iteration 2 — DPP V sweep and Z-feedback characterization (1P+1D)

**Conditions:** DPP with V ∈ {1, 3, 5, 10, 20, 50}, SLO=20ms at rates 30 and 40 req/s. Also SLO sweep (15–30ms) at V=5 to characterize the Z-feedback phase transition.

**Results at rate=30 (sub-saturation, 1P+1D):**

| V | Disagg fraction | Mean ITL | vs PrefixThreshold (28.65ms) |
|---|----------------|----------|------------------------------|
| 1 | 3% | 33.7ms | **+17.5% worse** |
| 5 | 10% | 36.2ms | **+26.2% worse** (worst) |
| 10 | 19% | 34.8ms | **+21.4% worse** |
| 50 | 73% | 30.3ms | **+5.9% worse** |
| 100 (default) | 100% | 28.65ms | identical to PT |

**DPP cannot beat PrefixThreshold in 1P+1D.** Partial disaggregation is a dominated strategy here: it splits work across both instances inefficiently, paying KV-transfer overhead without fully offloading the prefill bottleneck.

**Z-feedback phase transition (V=5, rate=30):**

| SLO | Disagg fraction | ITL |
|-----|----------------|-----|
| 15ms | 7% | 34.9ms |
| 20ms | 10% | 36.2ms |
| **22ms** | **90.6%** | **29.2ms** |
| 25ms | 100% | 28.65ms |

Sharp binary jump between SLO=20ms and SLO=22ms. The critical threshold (~21ms) equals the **minimum achievable TTFT for a disaggregated request at empty queue** (prefill scheduling ~15ms + KV transfer ~6ms). Below this, Z grows from the very first request and permanently suppresses disaggregation. Above it, Z stays zero and DPP freely disaggregates. **Z-feedback is a bistable switch, not a proportional controller.**

**Principles extracted:**
- RP-5: DPP partial disaggregation is strictly dominated for ITL in 1P+1D (no V value beats PT)
- RP-6: DPP (V≤10, SLO=20ms) at near-saturation achieves 100% completion vs PT's 86% (by keeping traffic local, avoiding decode KV exhaustion) — but at TTFT cost of ~3000ms
- RP-7: Z-feedback creates a sharp phase transition anchored to minimum achievable disagg TTFT

---

### Iteration 3 — Topology flip to 2P+2D

**Insight driving this iteration:** In 1P+1D, keeping traffic local overloads the single decode instance. In 2P+2D, two full-capability decode instances can absorb local traffic efficiently — and every locally-handled request *saves* the 6.6ms KV-transfer overhead that PrefixThreshold pays on 99% of requests.

**Topology:** 2 prefill + 2 decode instances (`--num-instances 4 --prefill-instances 2 --decode-instances 2`)  
**Conditions:** Never, PT(N=16), DPP(V=5, SLO=20ms), DPP(V=10, SLO=20ms), DPP(V=5, SLO=22ms)  
**Rate sweep:** 30, 35, 40, 45, 50, 55, 60 req/s → 33 total runs

**Results:**

| Rate | Never | DPP (V=5) | DPP (V=10) | PT | PT drops |
|------|-------|-----------|------------|-----|---------|
| 30 req/s | 18.2ms | 18.8ms | 19.6ms | 24.8ms | 0 |
| 35 req/s | 19.8ms | 21.3ms | 22.4ms | 27.6ms | 13 (0.65%) |
| 40 req/s | 21.1ms | 22.8ms | 24.0ms | 29.4ms | 172 (8.6%) |
| 45 req/s | 23.0ms | 24.6ms | 26.9ms | 31.8ms | 375 (18.75%) |
| 55 req/s | 27.0ms | 29.3ms | 32.3ms | 32.7ms* | 634 (31.7%) |
| 60 req/s | 29.0ms | 31.7ms | 33.1ms | 30.1ms** | 562 (28.1%) |

\* At rate=55 PT's ITL would be substantially higher if we included dropped requests.  
\*\* At rate=60 PT's apparent win is **survivor bias** — only the 72% of requests that completed are counted.

**DPP(V=5, SLO=20ms) beats PrefixThreshold on ITL by 10–30% at rates 30–55 req/s, with zero request drops at all rates through rate=55.**

**Ranking in 2P+2D:** `Never > DPP(V=5) > DPP(V=10) >> PrefixThreshold`

Interestingly, `NeverDisaggregate` is the outright ITL winner in 2P+2D. With only 32 uncached tokens per request, prefill work is tiny — there is almost nothing worth offloading. Both decode instances run full capacity (both prefill and decode phases locally) with zero KV-transfer overhead. The two prefill-only instances sit **completely idle** under PT while the two decode instances absorb all PT's disaggregated work and saturate.

**Negative control (SLO=22ms):** DPP(V=5, SLO=22ms) disaggregates 89–98% of requests and behaves identically to PrefixThreshold — confirms that the SLO threshold (not V or η) is the causal mechanism for DPP's win.

**Principles extracted:**
- RP-8: DPP(V≤10, SLO=20ms) achieves lower ITL than PrefixThreshold in 2P+2D topology by 10–30% across sub-saturation to moderate-saturation loads
- RP-9: Policy ranking is **topology-dependent**: the same DPP that is dominated in 1P+1D dominates in 2P+2D

---

## The Core Insight

DPP with a tight TTFT SLO (below the minimum achievable disaggregated TTFT) acts as a **"keep local by default"** policy. The Z-feedback permanently accumulates from the very first request and shuts off disaggregation for ~90% of traffic. This matters in 2P+2D because:

1. Keeping a request local means the decode instance handles both prefill and decode — no 6.6ms KV transfer
2. With 32 uncached tokens, local prefill is fast (~1ms) — the overhead is negligible
3. Two decode instances together have more than enough capacity for 30–55 req/s
4. PrefixThreshold, blindly disaggregating 99% of traffic, ships KV state over the network for every request AND concentrates all decode work onto the 2 decode-only instances, which eventually saturate

**The "Lyapunov optimization" framing** — where DPP is supposed to balance queue depths and minimize a drift function — does not describe what is actually happening here. The Z-feedback is not proportional; it is a binary suppressor. DPP(V=5, SLO=20ms) is essentially `NeverDisaggregate` with a small escape valve (it still disaggregates ~10% of requests when Z is low at startup).

---

## Campaign Artifacts

All experiment inputs, commands, results, and analysis live under `.nous/dpp-vs-prefix-threshold/`:

```
.nous/dpp-vs-prefix-threshold/
├── state.json                      # Final state: DONE
├── ledger.json                     # Per-iteration accuracy and principles summary
├── principles.json                 # All 9 extracted principles (RP-1 through RP-9)
├── runs/
│   ├── iter-1/
│   │   ├── problem.md              # Baseline characterization framing
│   │   ├── bundle.yaml             # 3-arm hypothesis bundle
│   │   ├── experiment_plan.yaml    # 20-run matrix (4 deciders × 5 rates)
│   │   ├── findings.json           # Confirmed/refuted per arm
│   │   ├── executor_log.md         # Full run log with result tables
│   │   └── results/                # Per-arm JSON metrics files
│   ├── iter-2/
│   │   ├── problem.md              # V sweep + Z-feedback characterization
│   │   ├── bundle.yaml             # 3-arm bundle (DPP dominated, binary switch, saturation)
│   │   ├── executor_log.md         # 33-run log with V/SLO sweep tables
│   │   └── results/
│   └── iter-3/
│       ├── problem.md              # 2P+2D topology flip
│       ├── bundle.yaml             # 3-arm bundle (ranking inversion hypothesis)
│       ├── executor_log.md         # 33-run log with full rate sweep tables
│       └── results/
```

---

## How to Reproduce

```bash
# Prerequisites: Go 1.21+, on branch feat/dpp-decider
cd /path/to/inference-sim
go build -o blis .

# The winning DPP configuration (beats PrefixThreshold by 23% on ITL)
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --prefill-instances 2 --decode-instances 2 \
  --pd-decider dpp --dpp-v 5 --dpp-ttft-slo-d 20 \
  --rate 45 --num-requests 2000 \
  --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 \
  --seed 42
# Expected: ITL ≈ 24.6ms, TTFT ≈ 23ms, 0 dropped requests

# PrefixThreshold baseline for comparison
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --prefill-instances 2 --decode-instances 2 \
  --pd-decider prefix-threshold --pd-prefix-threshold 16 \
  --rate 45 --num-requests 2000 \
  --prompt-tokens 512 --prefix-tokens 480 --output-tokens 128 \
  --seed 42
# Expected: ITL ≈ 31.8ms, TTFT ≈ 60ms, 375 dropped requests (18.75%)
```

---

## Open Questions

1. **Realistic TTFT SLO.** The experiment used SLO=20ms, which is tighter than typical production targets (100–200ms). This value was chosen because it falls just below the minimum achievable disaggregated TTFT (~21ms), triggering permanent Z-suppression. With a realistic SLO (e.g., 100ms), Z would never accumulate and DPP would behave like AlwaysDisaggregate — losing its advantage entirely. The DPP win is *directly dependent* on setting SLO below the minimum disaggregated TTFT.

2. **Realistic workload.** This campaign used a synthetic, uniform workload (512 fixed prompt tokens, 480 prefix tokens). The repo contains two production-mirror workloads from the inference-perf catalog:
   - `inference-perf-interactive-chat.yaml` — short dynamic prompts (~5,000-token prefix, lognormal input lengths ~50 tokens/turn)
   - `inference-perf-code-generation.yaml` — massive prompts (~30,000-token prefix, representing repository-context code generation)
   
   Under these workloads, the uncached token distribution is heterogeneous. PrefixThresholdDecider's cache signal becomes more informative (some requests have lots of uncached tokens, some have few). DPP's behavior under mixed uncached lengths is untested.

3. **Queue-depth signal.** DPP's queue-depth terms (Q_P, Q_D) had negligible effect in all experiments. The constant term `V·c_D/2` and the Z-accumulation dominated every decision. A reformulation that makes the queue-depth signal primary — rather than a secondary correction — might produce a genuinely adaptive policy.

4. **3P+3D and larger topologies.** The 2P+2D result may generalize: as long as there are ≥2 decode instances and uncached prefill is small, "keep local" beats "always disaggregate." Testing at larger scale would confirm whether RP-8 and RP-9 hold generally.
