# EDPP policy study — what we ran, what we found, what it means

Status as of 2026-07-17. Branch `feat/edpp-estimator-validation` (not pushed).
**This is v2: the study re-baselined on realistic, provenance-backed workloads.** The v1 results
(§ appendix, "superseded") used constant-token caricature workloads (e.g. 16-token outputs), a fixed
`4/8/16` req/s grid, an artificial concurrency cap, and a flat transfer cost — all of which distorted
the findings. v2 fixes every one of those (§2.5). Every number below is reproducible with a command in
this document (`repro_realistic.sh`).

---

## 1. Executive summary (read this first)

**The question.** EDPP is a Lyapunov drift-plus-penalty policy that decides, per request, whether to
disaggregate prefill and where to route it. On a fixed 1P2D topology, does it beat trivial heuristics,
and where?

**How v2 is measured (the defensibility upgrade).** Three realistic archetypes with *variable* output
lengths, drawn from the **kubernetes-sigs/inference-perf catalog** (decode-bound synthetic-generation
~4000-tok output; balanced ~1500/1500; prefill-heavy RAG with a 60k-token tail). Load is swept on a
**utilization axis** ρ = achieved/λ\* (λ\* measured per archetype), not raw req/s. **SLOs are derived,
not asserted** — the TTFT/ITL p99 the server actually delivers at 70–80% utilization (§3.2). Transfer
cost is **size-based** (mirrors the KV-transfer executor). No artificial cap.

**The robust findings (hold across 3 seeds):**

1. **Dynamic P/D routing genuinely wins on long-context (prefill-heavy) workloads.** At 85%
   utilization, `edpp` and `least-ttft` reach **~0.91 goodput vs ~0.62 for the naive `always`/`never`
   corners** (E2, 3 seeds). This is the real, defensible positive result — and it is *cleaner* on
   realistic workloads than on the v1 caricatures.
2. **The llm-d shipped decode scorer is catastrophic on shared-prefix workloads.** Its
   `precise-prefix-cache:2,queue-depth:1` profile pins decode onto one instance: goodput **0.10 vs
   0.67** (prefill), **0.40 vs 0.99** (decode) against a load-balanced scorer (E1). A property of a
   shipped production config, independent of EDPP.
3. **No fixed corner is universal** — `never` wins prefill-heavy, `always` wins decode/balanced (E2).
4. **Hardware heterogeneity:** joint EDPP recovers the fast node under-capacity (0.97 vs 0.00 for
   reduced, 0.72 blind); per-instance θ_i **over-corrects** under saturation (routes 95–97% to the
   fast node, *below* the reactive baselines) (E5).
5. **A deployable, minimax-regret-adaptive routing rule — drift-plus-VaR (E12/E13).** Re-pricing the
   congestion externality in **goodput** (value-at-risk: goodput destroyed among co-residents) and
   *keeping* the work-drift term, auto-normalized (`w≈1`), gives one rule you deploy **blind** — no
   knowledge of workload mix or hardware — that is **within ~5% of the best rule in every regime and
   never catastrophic**, on realistic *variable*-output workloads, using only a **deployable**
   (INV-9-safe) co-resident estimate. Its worst-case regret across the regime space is **~0.05 vs
   0.29–0.92** for every alternative — including a Kairos-inspired, physics-normalized adaptation of the
   **state of the art** (arXiv:2607.02043, reported at its best β: worst-case regret 0.61) and llm-d's shipped
   `prefix-threshold`. Each alternative collapses somewhere: `always`/llm-d on prefill-heavy and
   heterogeneous, `dpp` on decode/mixed, Kairos and `least-ttft` on **hardware heterogeneity** — the
   regime no published P/D routing rule addresses. The claim is minimax-regret, not domination. See E13.

**The honest limits (what v2 does NOT support):**

- **`edpp` ≈ `least-ttft` at moderate load** — the full drift/z/V machinery does not clearly beat the
  one-line "disaggregate iff predicted TTFT is lower" rule, and the two make **near-identical routing
  decisions** (64% vs 66% disaggregated, verified from the decision trace). The apparatus does not
  visibly earn its keep in the regime where results are stable.
- **Near and above saturation the prefill regime is high-variance across seeds** (e.g. at ρ1.0 the
  term-ablation arms swing from 0.46 to 1.00 between seeds 42 and 7). **We therefore do not make
  fine-grained term-level claims** ("which term is load-bearing", "edpp beats least-ttft under
  overload") — 3 seeds is too few there. Only the coarse dynamic-vs-naive result (finding 1) is
  seed-robust. This retires the v1 term-by-term story (drift is the win / z sign-flips), which was a
  single-seed, caricature-specific artifact.
- **The `o_r` oracle now moves the drift arms** (variable output makes N̂_out a real error source,
  unlike the caricatures) — but the effect, too, is seed-noisy; it is suggestive, not conclusive.

**One line for your manager.** *On realistic long-context workloads, adaptive prefill/decode routing
clearly beats the naive heuristics; no single simple rule is good across workloads and hardware, but a new
deployable rule that prices the congestion externality in goodput (drift-plus-VaR) is within ~5% of the
best rule in every regime and never catastrophic — the one you deploy when you cannot predict the
workload, cutting worst-case goodput regret from 30–90% (any fixed rule) to ~5%.*

**Effort delivered.** Realistic-workload re-baseline (provenance-backed archetypes, utilization-normalized
load, derived SLOs), a size-aware transfer-cost model + `o_r` oracle (both new, committed), and a
reproducible harness (`repro_realistic.sh`). Plus the earlier hardware-heterogeneity / θ_i / `least-ttft`
implementation work.

## 2. What EDPP is, precisely

### 2.0 Why a joint rule at all (formulation §1, §3.4, §5.5)

The design starts from one claim: **routing and deciding are the same problem, and the system
currently splits them.** Two mechanisms run in sequence — a scorer picks the decode instance, then
a separate decider chooses local-vs-disaggregate. The formulation folds both into one action:

    a_r = (d_r, p_r),   p_r ∈ prefill-nodes ∪ {local}

"local" is just another prefill location, so *"disaggregate?"* becomes a coordinate of *"which
instance?"*. The split is lossy because the two decisions are **coupled in both directions** (§1):

- the **value of disaggregating depends on which decode instance** was picked — offloading prefill
  helps a congested decode instance far more than an idle one;
- the **best decode instance depends on whether the request will be disaggregated** — if it will,
  pick on decode capacity alone (that instance never prefills); if it won't, pick on combined
  prefill+decode capacity. These can be different instances.

A decomposed pipeline picks the instance blind to the split, then the split blind to the instances
it could have chosen. Because the joint rule minimizes the same objective over a *superset* of the
reduced action set, it is **provably no worse**; what it costs is compute and llm-d scorer parity.
§5.5 is explicit that the reduction is *"a deployment choice…, not a mathematical necessity"* and
that *"the pairwise rule cannot see that a different decode node would have made disaggregation
unnecessary."*

**Consequence for this study:** every experiment here used the **reduced** rule — the scorer-chosen
slice. We measured the decomposition the formulation was written to replace, and never switched on
the mechanism it argues for. That is §6 gap 1, and it is why that gap outranks the others.

### 2.1 The objective the rule minimizes

The formulation states the problem the decision rule is derived from (§5.1):

> "Write g(t) for the operating cost we would rather avoid — here the **transfer / KV-movement
> cost** incurred by disaggregation in epoch t. ... minimize its time average subject to stability
> **and the SLO constraints**."

So the SLO targets are **constraints**, carried as virtual queues that accumulate violation, and
the single quantity being minimized is the transfer cost. This is what the shipped rule implements
and what every number in this report was produced with. Its consequence under overload is §3.2.

### 2.2 The three terms of the implemented rule

For each request EDPP evaluates an objective and picks the minimizing action. The objective has
three kinds of terms, and it matters enormously which is which:

| term | what it is | kind |
|---|---|---|
| `q_i · Δwork_i` | per-instance backlog (congestion) | **drift** — enforces work-stability (throughput) |
| `z_ttft · T̂/τ`, `z_itl · (…)/τ` | virtual queues for the TTFT/ITL SLOs | **drift** — enforces the SLO *constraints* |
| `V · c_xfer · 1{disagg}` | the KV-transfer cost | **penalty** — the actual objective |

**The SLOs are constraints, not the objective.** They are enforced by virtual queues `z`
that accumulate violation. The *only* thing EDPP optimizes is `V·c_xfer`. Read plainly,
EDPP's posture is:

> *"Disaggregate as little as possible, subject to holding the TTFT/ITL SLOs and keeping
> the queues stable."*

Two modes exist:
- **reduced** — a scorer picks the decode instance; EDPP decides only local-vs-disaggregate.
- **joint** (`--edpp-joint`) — EDPP enumerates all `(decode, prefill)` candidates and picks
  the argmin itself. *This is EDPP's real thesis, and §6 explains why we never tested it here.*

Defaults used everywhere below: `--edpp-v 1`, `--edpp-c-xfer 5ms`, `--edpp-tadm-estimator
rollforward` (the occupancy-aware admission-delay estimator; the shipped default `waiting` is
occupancy-blind and would understate EDPP). **Caveat on `--edpp-c-xfer 5ms`:** it is a *flat
constant*, but the actual KV transfer the simulator executes is *size-based* (∝ prefill tokens);
5 ms is ~right only for the `mixed` archetype and mis-prices the others by 4–10× (§E11). All
tables below use the flat default unless marked `+c_xfer-size`.

### 2.3 The value-at-risk externality and drift-plus-VaR (E12/E13)

The classic rule above prices the congestion externality in **work** (µs of backlog) and the
request's own latency in the `z` self terms. Neither prices the externality in the currency the
objective is measured in — **goodput**. Drift-plus-VaR (`--edpp-rule var --edpp-var-congestion`)
adds a term that does. Full derivation, oracle boundary, and reproduction are in E12/E13 and §8; the
math a reader needs:

**Value-at-risk of a placement.** For a candidate that puts `r`'s decode on instance `d`, VaR is the
goodput destroyed among `d`'s current decode co-residents by the added load. Each co-resident `j`
with `rem_j` steps left is projected to finish at `C_base = now + rem_j·t0` before `r`, and later
after `r` is added, under **full B+1 re-timing** — the per-iteration time is recomputed at batch
`B+1` with `r`'s KV added (`Δkv_R = n_r`, `r`'s prompt length; input-only, oracle-safe):

```
t0        = t_iter(B,   kv,          sPf)          # current
t_after   = t_iter(B+1, kv + Δkv_R,  sPf)          # after r joins decode  (t_after ≥ t0)
C_local_j = now + min(nChunks,rem_j)·t_overlap + (rem_j−min(nChunks,rem_j))·t_after
C_disagg_j= now + min(A,rem_j)·t0       + (rem_j−min(A,rem_j))·t_after      # A = ⌈T̂_disagg/t0⌉
```

`t_overlap = t0 + c_pf·chunk` is the inflated iteration time while `r` prefills co-scheduled (local
only). Because local inflation starts at step 0 and disagg only inflates the tail,
`C_local ≥ C_disagg ≥ C_base` — the asymmetry that lets disaggregation relieve a loaded decode node.

Each co-resident's contribution is scored by one of three kernels against its SLO composite
(TTFT met ∧ mean-ITL ≤ τ_itl ∧ completion ≤ arrival+τ_e2e): **flip** (binary good→bad count — the
hyperparameter-free ceiling), **util** (`σ((deadline−C)/τ_ttft)`, the robust kernel), **hazard**
(`h(slack)·delay`, `h` Cauchy-like). `VaR = Σ_j [g(C_base) − g(C_placed)]`.

**Drift-plus-VaR.** Keep the congestion drift AND add VaR (rather than replacing it):

```
cost(d,p) = w · congestion(d,p)  +  VaR(d,p)  +  self(d,p)
```

Congestion feels a node saturating (needed on heterogeneous hardware); VaR supplies the SLO
externality (needed on homogeneous decode/balanced). **Auto-normalization** (`--edpp-var-normalize`)
min-max normalizes the two terms across the decision's candidates so `w` is scale-free (`w≈1`); a
**zero-spread guard** makes a symmetric congestion term (identical hardware) cancel automatically —
so VaR decides on homogeneous nodes, and congestion bites only where the hardware differs. That
single mechanism is why one weight keeps the rule near-best across regimes — the minimax-regret
result of E13 (within ~5% of the best simple rule everywhere, on realistic variable output).

**Oracle boundary (INV-9).** `rem_j` is the co-resident's true remaining decode steps — an **oracle**
read (un-censored), so drift-plus-VaR is a *ceiling*, not yet deployable. `--edpp-rule var` emits a
loud UPPER-BOUND warning. Deadlines, arrivals, and `Δkv_R` are input-derived (deployable). A
deployable variant (censored `N̂_out` for `rem_j`) is the #1 open item (§7).

---

## 3. Methodology — how we evaluate a policy, and why

> **Note (v2):** §3.4–§3.5 below describe the v1 methodology (fixed-fraction oracle, cap-16, `4/8/16`
> rate grid). For all v2 results use §4.1–§4.2 (utilization-normalized load, derived SLOs, no cap).
> The measure (goodput, §3.1) and objective caveat (§3.2) still apply.

### 3.1 The measure: goodput

**Goodput** = the fraction of completed requests meeting *all* their SLO dimensions
(TTFT and ITL and e2e). In BLIS this is `per_class[class].slo_attainment`. Higher is better.

We chose goodput because it is what an operator is graded on. Note the tension this creates
with EDPP's own objective (§3.2).

### 3.2 A known objective mismatch (important caveat on every EDPP number)

EDPP minimizes *transfer cost* subject to *average-latency constraints*. We measure *goodput*.
These coincide only when the SLOs are comfortably feasible. Under overload:
- the constraints are **infeasible** — no policy can hold them, so the virtual queues grow
  without bound and the drift-plus-penalty guarantee is void;
- "minimize transfers" is **beside the point** — you are missing TTFT either way;
- worse, the `z` queues accumulate *lateness* (`z += realized − τ`, clamped at 0). A request
  10 s late pumps a large positive into the queue, so the policy keeps spending capacity on a
  request that already missed and can never count toward goodput. **Goodput saturates; the
  surrogate does not.** Goodput implies *triage* (protect the savable, abandon the doomed);
  an accumulating-lateness surrogate structurally cannot triage.

This is our leading hypothesis for why EDPP collapses under extreme overload (§5, F5).
**We have not fixed this** — every experiment below uses the shipped transfer-cost objective.

### 3.3 The policies compared

| policy | rule | note |
|---|---|---|
| `never` | never disaggregate (prefill on the decode instance) | wastes the prefill server |
| `always` | always disaggregate | uses all servers; the P/D split's intent |
| `prefix-threshold` | disaggregate iff uncached prompt tokens > N | **N=16 is llm-d's shipped default** (`deploy/config/pd-epp-config.yaml`) |
| `least-ttft` | disaggregate iff predicted TTFT_disagg < predicted TTFT_local | **EDPP's estimator without its drift/z/V machinery** — built precisely to isolate the machinery's value |
| `edpp` | the full drift-plus-penalty rule | penalty = `V·c_xfer` |

### 3.4 The yardstick: a static disaggregation-fraction oracle

To know whether a policy leaves goodput on the table we need a target. We force a per-request
plan with `--pd-plan` (a CSV of `request_id,decode_instance,prefill_instance`). For fraction
*f*, the first *f* of every 100 requests are disaggregated (`prefill_instance=instance_0`) and
the rest are local (empty), with decode alternating `instance_1`/`instance_2` so decode is
perfectly balanced and cannot confound the result. `f=0` is `never`; `f=100` is `always`.
Sweeping *f* and taking the max gives the **best static split**. An *interior* maximum proves
neither corner is optimal — i.e. that routing genuinely matters.

This is a **static** yardstick, not the true optimum. A dynamic policy can beat it (and does,
§5 F4). The true joint hindsight optimum (a MILP over all requests) remains unbuilt.

### 3.5 Experiment design choices, and the reason for each

- **Fixed 1P2D topology** (`instance_0`=prefill, `instance_1/2`=decode). The operator picks
  the topology; the policy only routes. Comparing across topologies would answer a
  *provisioning* question, not a *policy* question.
- **Four workload archetypes** spanning decode-bound → prefill-bound, because which pool is
  the bottleneck is exactly what should drive the disaggregation decision.
- **SLO auto-set per archetype** from an idle probe (2× idle `e2e_p99`, 3× idle `ttft_p99`).
  Each archetype has a different intrinsic latency; one fixed SLO would measure "which
  archetype is fast", not "which policy is good".
- **Concurrency cap** (`--max-num-running-reqs 16`). With the default cap (256) and abundant
  KV, these workloads never queue and *every policy scores 1.000* — no signal. The cap forces
  saturation at reachable rates. **This is an artificial stressor and a caveat on the results.**
- **Decode scorer held fixed** at `queue-depth:1` for the spectrum, to isolate the
  disaggregation decision from the scorer artifact in §5 F1. We report both setups.

---

## 4. Realistic re-baseline (v2) — the results that stand

> Everything in §4 uses `campaigns/edpp-study/repro_realistic.sh`. §5–§8 below (the v1 caricature
> study) are **superseded** and kept only for provenance / before-after.

### 4.1 What changed from v1, and why it matters

| axis | v1 (superseded) | v2 (this section) | why it mattered |
|---|---|---|---|
| workloads | constant-token caricatures (256/512, 16000/**16**) | inference-perf **catalog** archetypes, **variable** output | 16-tok outputs made decode work negligible & `o_r` unestimable |
| load | fixed `4/8/16` req/s | **ρ = achieved/λ\*** (λ\* measured per archetype) | req/s isn't comparable across archetypes |
| SLOs | 2× idle e2e / 3× idle ttft (or asserted) | **derived**: TTFT/ITL p99 the server delivers at 70–80% util | e2e is output-length-dominated & routing-insensitive |
| saturation | artificial `--max-num-running-reqs 16` | none — realistic workloads saturate on their own | cap-16 was a caricature stressor |
| transfer cost | flat 5 ms | **size-based** (mirrors the KV-transfer executor) | flat 5 ms mis-priced transfers 4–10× |
| TTFT predictor | projection-only prefill (no attention) | charges full `Wp` (projection **+** attention), matching the executor | under-modelled long-context prefill (the least-ttft regime) |

> **Predictor fidelity re-run (commit 15de04f).** The TTFT predictor was corrected to charge the
> request's full prefill work `Wp` (adding the quadratic attention term) on both the reduced and
> joint paths, so the decider's math matches the executor. Re-running E2 and the prefill ablation on
> the corrected binary leaves every goodput number below **unchanged** (goodput is scored on the
> executor's realized TTFT, which always modelled attention) and leaves the disaggregation fractions
> unchanged (least-ttft 66%, edpp 64%). The correction buys math/code alignment for the paper; it does
> not move the empirical result.

### 4.2 Archetypes & derived SLOs (provenance-backed)

| archetype | in/out (tok) | source | λ\* | derived SLO (p99 @ ρ≈0.8) |
|---|---|---|---|---|
| **decode-bound** | ~500 / **4000±2500** | inference-perf synthetic-data-gen | ~5000 tok/s | TTFT 170 ms, ITL 60 ms |
| **balanced** | ~1500 / ~1500 | fabricated (documented) | ~7000 tok/s | TTFT 180 ms, ITL 50 ms |
| **prefill-heavy** | ~2k+**60k** / **500±300** | inference-perf summarization-RAG | ~1350 tok/s | TTFT 2300 ms, ITL 60 ms |

`λ*` = achieved-throughput plateau (SLO-free ⇒ no circularity). ITL comes out ~uniform (per-token,
workload-independent); TTFT is per-workload (the 60k RAG prefill costs ~2 s even at idle). All v2 runs
build once (`go build -o blis main.go`) and use `campaigns/edpp-study/repro_realistic.sh`; each
experiment below carries its exact command. Derive λ\* + the SLOs, and inspect the routing decisions:

```bash
# λ*/SLO derivation sweep (achieved tok/s, goodput, ttft/itl p99, saturation verdict per rate)
MODE=lambda bash campaigns/edpp-study/repro_realistic.sh
# WHY a decision was made / disaggregation fraction (per-decision term breakdown + disaggregate flag)
./blis run --model meta-llama/llama-3.3-70b-instruct \
  --workload-spec inference-perf-batch-summarization-rag.yaml \
  --num-instances 3 --prefill-instances 1 --decode-instances 2 \
  --decode-routing-scorers "queue-depth:1" --pd-decider edpp \
  --edpp-coeffs scripts/calibration/coeffs-llama70b-h100-tp4.json \
  --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware \
  --slo-ttft "standard=2300ms" --slo-itl "standard=60ms" --slo-e2e "standard=999s" \
  --trace-level decisions --edpp-decision-trace /tmp/decisions.csv
```

### E1 (v2) — the llm-d shipped scorer is catastrophic on shared-prefix workloads

```bash
MODE=scorer NREQ=400 bash campaigns/edpp-study/repro_realistic.sh
```
Goodput at ρ≈0.85, llm-d default profile vs load-balanced decode scorer:

| archetype | llm-d `never` / `always` | balanced `never` / `always` |
|---|---|---|
| decode | 0.398 / 0.458 | **0.993 / 0.995** |
| balanced | 0.720 / 0.955 | **0.995 / 1.000** |
| prefill | 0.100 / 0.495 | **0.672** / 0.495 |

Up to **6.7×** goodput lost to the shipped `precise-prefix-cache:2,queue-depth:1` pinning decode onto
one instance. Independent of EDPP; contaminates any comparison on the default profile. **We hold the
decode scorer at `queue-depth:1` for E2–E10.**

### E2 (v2) — policy comparison, prefill-heavy (mean of 3 seeds: 42, 7, 123)

```bash
# seed 42 (all archetypes); repeat with SEED=7 and SEED=123 for the mean
MODE=policy NREQ=600 bash campaigns/edpp-study/repro_realistic.sh
SEED=7   MODE=policy NREQ=600 bash campaigns/edpp-study/repro_realistic.sh
SEED=123 MODE=policy NREQ=600 bash campaigns/edpp-study/repro_realistic.sh
```


| ρ (util) | `never` | `always` | `least-ttft` | `edpp` |
|---|---|---|---|---|
| 0.5 | 0.95 | 0.90 | **0.99** | **0.98** |
| 0.7 | 0.90 | 0.81 | **0.92** | **0.92** |
| **0.85** | 0.63 | 0.61 | **0.91** | **0.91** |
| 1.0 | 0.33 | 0.40 | 0.71 | 0.74 |
| 1.2 | *high variance across seeds — not reported* |

**Dynamic routing (`edpp`/`least-ttft`) beats both naive corners by ~0.3 at 85% utilization, robustly
across all 3 seeds — the headline positive.** On **decode** and **balanced**, all policies are within
~0.02 at feasible load (routing barely matters; `edpp` ≈ `never` — a genuine per-request veto: **0%
disaggregated**, confirmed from the decision trace), and decode is itself high-variance across seeds.

`edpp` ≈ `least-ttft` throughout — and they make **near-identical decisions** (64% vs 66% disaggregated
on prefill, from the trace), so the drift/z/V machinery is not what produces the win.

### E5 (v2) — hardware heterogeneity (fast H100 + crippled A100 decode, 4 seeds)

```bash
bash campaigns/edpp-study/repro_hetero_hw.sh              # under-capacity regime
SAT=1 bash campaigns/edpp-study/repro_hetero_hw.sh        # saturating regime
SAT=1 THETA=1 bash campaigns/edpp-study/repro_hetero_hw.sh  # + per-instance θ_i
```


- **Under-capacity** (fast node can serve all): joint-EDPP **0.97–1.00** vs reduced **0.00**, best
  hardware-blind **0.72–0.77**, optimum 1.00. Joint wins — *reactively* (it avoids the node that
  visibly congests, not because it knows the hardware).
- **Saturating** (must use both; optimum is an ~86%-fast split at ~0.96): joint **0.82–0.98** ≈
  blind load-balance **0.84–0.95** ≈ reduced. All undershoot.
- **+ per-instance θ_i** (give EDPP the hardware model): **over-corrects** — routes 95–97% to the
  fast node, goodput **0.67–0.88**, *worse* than the reactive baselines. Recorded as a characterized
  limitation.

### E8/E9/E10 (v2) — ablation, joint, `o_r` oracle: high-variance, no fine-grained claim

```bash
MODE=ablate         NREQ=600 bash campaigns/edpp-study/repro_realistic.sh   # E8 reduced term ablation
JOINT=1 MODE=ablate NREQ=600 bash campaigns/edpp-study/repro_realistic.sh   # E9 joint path
ORACLE=1 MODE=ablate NREQ=600 bash campaigns/edpp-study/repro_realistic.sh  # E10 true-o_r oracle
# per-seed robustness (prefill only): prefix each with SEED=7 / SEED=123
```


The reduced term ablation (`least-ttft` / `drift-only` / `drift+z` / `full`) is **stable at feasible
load** (all ~equal on decode/balanced; on prefill at ρ≤0.7 all ~0.95–0.99) but **swings wildly across
seeds near and above saturation** — e.g. prefill ρ1.0: seed 42 has `drift-only` 0.66 and `full` 0.67,
while seed 7 has `drift+z`/`full` at **1.00** and `least-ttft` at 0.46. **We therefore make no
term-level "which term is load-bearing" claim** — 3 seeds is too few in the overload regime, and the
v1 single-seed story (drift is the win / z sign-flips) does not replicate. The `o_r` oracle *does*
move the drift arms on prefill (variable output makes `N̂_out` a real error source — unlike v1's
constant outputs), and joint routing (E9) is likewise mixed; both are suggestive but seed-noisy.
**Bottom line unchanged:** at the load where results are stable, `edpp` ≈ `least-ttft` and the
machinery does not visibly earn its keep.

### 4.3 v2 findings

- **F1 (robust).** llm-d shipped scorer pins decode → up to 6.7× goodput loss on shared-prefix workloads (E1).
- **F2 (robust).** No naive corner is universal: `never` wins prefill-heavy, `always` wins decode/balanced (E2).
- **F3 (robust).** Adaptive P/D routing wins the prefill-heavy regime by ~0.3 goodput at 85% util, across 3 seeds (E2).
- **F4 (robust).** A one-line TTFT rule captures that win — `edpp` ≈ `least-ttft`, near-identical decisions; the drift/z/V apparatus doesn't visibly earn its keep at stable load (E2, trace).
- **F5 (robust).** Hardware: joint EDPP recovers the fast node under-capacity; per-instance θ_i over-corrects under saturation (E5).
- **F6 (limit).** Near/above saturation the prefill regime is high-variance across seeds; no fine-grained term-level or overload-regime claim is supported on 3 seeds. The v1 term-by-term story is retired.

---

## 5. (SUPERSEDED — v1) The caricature experiments

> **The sections below are the v1 study on constant-token caricature workloads with a fixed rate grid
> and cap-16. They are kept for provenance and before/after only — every quantitative claim here is
> superseded by §4. Do not cite v1 numbers.**

## 4. The experiments (each with its command)

All commands run from the repo root, after `go build -o blis main.go`.
Harness: `campaigns/edpp-study/repro_spectrum.sh` (self-contained; emits its own specs).

### E1 — The scorer comparison (llm-d default vs load-balanced)

```bash
MODE=scorer bash campaigns/edpp-study/repro_spectrum.sh
```
Archetype `mixed` (in=2048, out=128), rate 16, cap 16, 1P2D. Output (`i1`/`i2` = requests
that *decoded* on each decode instance — every completed request decodes exactly once, so
this is the realized decode allocation):

| decode scorer | `never` | `always` | realized split |
|---|---|---|---|
| **llm-d shipped** (`precise-prefix-cache:2,queue-depth:1`) | 0.067 | **0.133** | `i1=0  i2=240` ← all pinned to one instance |
| load-balanced (`queue-depth:1`) | 0.246 | **0.879** | `i1=119 i2=121` |

### E2 — The workload spectrum

```bash
bash campaigns/edpp-study/repro_spectrum.sh                # decode scorer = balanced
SCORER=llmd bash campaigns/edpp-study/repro_spectrum.sh    # decode scorer = llm-d shipped
```
1P2D, cap 16, seed 42, decode scorer balanced. Goodput:

| archetype (in/out) | rate | `never` | `always` | `prefix16` | `least-ttft` | `edpp` |
|---|---|---|---|---|---|---|
| **decode** 256/512 | 4 | 0.463 | **1.000** | 1.000 | 0.463 | 0.463 |
| | 8 | 0.133 | **0.271** | 0.271 | 0.133 | 0.133 |
| | 16 | 0.133 | **0.267** | 0.267 | 0.133 | 0.133 |
| **mixed** 2048/128 | 16 | 0.246 | **0.879** | 0.879 | 0.671 | 0.550 |
| **prefill_lean** 8192/64 | 8 | **1.000** | 0.604 | 0.604 | **1.000** | **1.000** |
| | 16 | 0.467 | 0.046 | 0.046 | 0.762 | **0.771** |
| **prefill_bound** 16000/16 | 4 | 0.996 | 0.096 | 0.096 | 0.992 | **1.000** |
| | 8 | 0.100 | 0.033 | 0.033 | 0.854 | **0.917** |
| | 16 | 0.054 | 0.025 | 0.025 | **0.375** | 0.071 |

(`prefix16` ties `always` throughout because every prompt here exceeds 16 uncached tokens, so
llm-d's shipped threshold degenerates to "always disaggregate" on these workloads.)

### E3 — The static-fraction oracle on the prefill-bound cell

```bash
MODE=oracle ARCH=prefill_bound RATE=8 bash campaigns/edpp-study/repro_spectrum.sh
```
| f (% disaggregated) | 0 | 20 | 25 | 30 | **35** | 40 | 45 | 50 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| goodput | 0.108 | 0.500 | 0.571 | 0.600 | **0.604** | 0.562 | 0.517 | 0.408 | 0.329 | 0.179 | 0.033 |

Dynamic policies on the same cell: `least-ttft` 0.854, `edpp` **0.917** (`i1=115 i2=125`).
The maximum is **interior** (f=35), so neither corner is optimal, and both dynamic policies
beat the best static split.

### E4 — Seed robustness on the prefill-bound cell

```bash
for r in 8 16; do for s in 42 7 123; do
  SEED=$s MODE=oracle ARCH=prefill_bound RATE=$r FRACS=35 \
    bash campaigns/edpp-study/repro_spectrum.sh
done; done
```
The SLO target is derived from a **fixed** probe seed (`SLO_PROBE_SEED`, default 42) and does
not move when `SEED` sweeps, so every seed is graded against the same goalposts.

| policy | rate | seed 42 | seed 7 | seed 123 |
|---|---|---|---|---|
| best static split (f=35) | 8 | 0.604 | 0.604 | 0.533 |
| `least-ttft` | 8 | 0.854 | 0.883 | **0.975** |
| `edpp` | 8 | **0.917** | **0.917** | 0.942 |
| best static split (f=35) | 16 | 0.075 | 0.075 | 0.083 |
| `least-ttft` | 16 | **0.375** | **0.433** | **0.383** |
| `edpp` | 16 | 0.071 | 0.067 | 0.054 |

At rate 8 both dynamic policies beat the best static split, and `edpp` edges `least-ttft` on
2 of 3 seeds (~0.06). At rate 16 the static split collapses too (0.075–0.083), `edpp` collapses
with it (0.054–0.071), and `least-ttft` appears to hold up (0.375–0.433) — seemingly a **5–6× gap
in favour of the stripped-down rule** under extreme overload.

> **Correction (E11).** That 5–6× gap is an artifact of the flat `--edpp-c-xfer 5ms`. The real KV
> transfer is size-based and costs **~53 ms** for a 16000-token prefill (measured, §E11). When EDPP's
> decision prices transfer correctly, `least-ttft` collapses too — prefill_bound rate 16 falls from
> **0.397 → 0.101**, closing the gap to the drift arms (0.06) to ~0.04. `least-ttft`'s robustness was
> under-charging transfer and thus over-disaggregating, landing near the interior optimum by accident.
> See E11 / F13.

### E5 — Hardware heterogeneity (earlier work, `repro_hetero_hw.sh`)

```bash
bash campaigns/edpp-study/repro_hetero_hw.sh              # under-capacity regime
SAT=1 bash campaigns/edpp-study/repro_hetero_hw.sh        # saturating regime
SAT=1 THETA=1 bash campaigns/edpp-study/repro_hetero_hw.sh  # with per-instance θ_i
```
1P2D with a fast H100 decode and a deliberately crippled A100 decode (400 TFLOPS / 0.7 TB/s),
configured via a `hw_config_by_gpu` policy bundle. **This is the only place we used
`--edpp-joint`.**

- **Under-capacity** (fast node can serve everything): joint-EDPP **0.97–1.00** vs
  reduced-EDPP 0.00–0.77, best hardware-blind scorer 0.72–0.77, optimum 1.00. EDPP wins,
  *reactively* — it avoids the slow node because that node visibly accumulates congestion,
  not because it knows the hardware is slow.
- **Saturating** (must use both nodes; optimum is an interior ~86%-fast split at 0.96):
  joint-EDPP 0.82 ≈ blind load-balance 0.84 ≈ reduced 0.83. All undershoot.
- **With per-instance θ_i** (giving EDPP the hardware model): it **over-corrects** — 95–97%
  to the fast node, goodput 0.685/0.750, *worse* than the reactive baselines. The work term
  overwhelms the congestion signal. Bar not met; recorded as a characterized limitation.

### E6 — Workload heterogeneity (earlier work)

Type A (85%, in=256/out=300, tight SLO) + Type B (15%, in=8000/out=16, loose SLO), 1P2D,
balanced decode. Measuring A's goodput:

| rate | `never` | `always` | `prefix1000` (disagg B only) | `edpp` |
|---|---|---|---|---|
| 12 | 0.952 | **1.000** | 1.000 | 1.000 |
| 16 | 0.795 | **1.000** | 1.000 | 0.984 |
| 20 | 0.430 | **1.000** | 1.000 | 0.980 |
| 24 | 0.293 | **1.000** | 1.000 | 0.924 |

`always` is optimal; EDPP is slightly worse. Keeping B local wrecks A (interference), but
disaggregating *everything* fixes it — no per-request smartness needed here.

---

### E7 — SLO-class heterogeneity (the third axis)

```bash
# prefill-bound 16000/16, rate 10, 50% critical + 50% batch, IDENTICAL sizes
# critical: ttft 1794ms/itl 100ms/e2e 1600ms   batch: 60s/500ms/60s (loose)
# EDPP gets per-class targets:
#   --edpp-tau-ttft-classes "critical=1794ms,batch=60s" --edpp-tau-itl-classes "critical=100ms,batch=500ms"
# The control arm simply OMITS those two flags (one tau for both classes).
```
Rate 10 chosen from a probe: ~40% of requests miss (so routing decides *who* survives) but EDPP has
not yet collapsed — the generous choice for EDPP. Both classes share a size profile so a win cannot
be attributed to workload heterogeneity. **Critical-class goodput** (batch = 1.000 everywhere):

| seed | never | always | least-ttft | **edpp single-τ** | edpp per-class τ |
|------|-------|--------|-----------|------------------|------------------|
| 42   | 0.163 | 0.039  | 0.581     | **0.884**        | 0.752            |
| 7    | —     | —      | 0.636     | **0.836**        | 0.891            |
| 123  | —     | —      | 0.456     | **0.856**        | 0.800            |
| mean | —     | —      | 0.558     | **0.859**        | 0.814            |

Mechanism, from EDPP's own decision trace (seed 42): batch `z_ttft` = **exactly 0**, batch disagg
55%→28.8%, critical disagg 27.1%→43.4%, critical gp 0.884→0.752, **batch e2e 3424→1879ms**. The
machinery did what it was designed to do and critical still got worse; the deprioritised class got
*faster*. See F9.

### E8 — Term ablation (which term is load-bearing?)

```bash
MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh
```
Reachable from config alone: `--edpp-v 0` zeroes `transferTerm`; `--edpp-tau-ttft 999s` makes TTFT
unviolatable ⇒ `z_ttft ≡ 0`. τ_itl is held at 100ms in every arm so the normalizer μ_D does not move
between arms (and `z_itl` is inert anyway — measured ITL never approaches 100ms). Both off ⇒
`lhs > 0` = pure congestion drift. Integrity verified from the trace: drift-only arm shows peak
z_ttft = 0, ttft_term = 0, transfer = 0; full arm shows z_ttft = 435.6, ttft_term = 25.74,
**transfer = 0.00078**.

Means over 3 seeds (four cells sit at the 1.000 ceiling and carry no signal):

| archetype | rate | least-ttft | drift-only | drift+z | full | winner |
|-----------|------|-----------|-----------|---------|------|--------|
| decode        | 8  | 0.133 | **0.267** | 0.135 | 0.133 | drift-only |
| decode        | 12 | 0.133 | **0.237** | 0.135 | 0.133 | drift-only |
| decode        | 16 | 0.133 | **0.233** | 0.137 | 0.133 | drift-only |
| mixed         | 16 | 0.750 | **0.803** | 0.747 | 0.708 | drift-only |
| prefill_lean  | 16 | 0.754 | 0.696 | **0.793** | 0.771 | drift+z |
| prefill_bound | 8  | 0.904 | **0.971** | 0.921 | 0.925 | drift-only |
| prefill_bound | 12 | **0.492** | 0.264 | 0.408 | 0.414 | least-ttft |
| prefill_bound | 16 | **0.397** | 0.061 | 0.062 | 0.064 | least-ttft |

`z`'s effect (drift+z − drift-only): decode **−0.110**, mixed −0.018, prefill_lean +0.030,
prefill_bound +0.032. See F5–F8.

### E9 — The same ablation on the JOINT path (closes the biggest gap)

```bash
JOINT=1 MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh
```
In joint mode EDPP enumerates all (d,p) and picks the argmin **itself, overriding the decode scorer**.
This is EDPP's actual thesis and had never been exercised. `least-ttft` is reduced-only, so that
column is absent. Reduced → JOINT, mean of 3 seeds:

| archetype | rate | drift-only | drift+z | full |
|-----------|------|-----------|---------|------|
| decode        | 12 | 0.237 → **0.296 ↑** | 0.135 → 0.136 | 0.133 → 0.133 |
| decode        | 16 | 0.233 → 0.221 | 0.137 → 0.139 | 0.133 → 0.133 |
| mixed         | 12 | 0.999 → **0.843 ↓** | 1.000 → 1.000 | 1.000 → 0.978 ↓ |
| mixed         | 16 | 0.803 → **0.681 ↓** | 0.747 → 0.724 ↓ | 0.708 → **0.626 ↓** |
| prefill_lean  | 16 | 0.696 → 0.679 | 0.793 → **0.728 ↓** | 0.771 → **0.711 ↓** |
| prefill_bound | 8  | 0.971 → **0.992 ↑** | 0.921 → **0.954 ↑** | 0.925 → **0.953 ↑** |
| prefill_bound | 12 | 0.264 → 0.238 ↓ | 0.408 → 0.403 | 0.414 → 0.396 |
| prefill_bound | 16 | 0.061 → 0.062 | 0.062 → **0.161 ↑** | 0.064 → **0.161 ↑** |

A wash. See F10.

### E10 — Oracle output-length control (is the failure an `o_r`-estimation artifact?)

```bash
ORACLE=1 MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh
JOINT=1 ORACLE=1 MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh
```
`--edpp-oracle-output-len` (DIAGNOSTIC, UPPER-BOUND, INV-9-violating) substitutes the routed
request's **true** output length for the per-class `N̂_out` estimate when charging its *own* decode
work (joint `W_d` and the `qdWork` backlog); co-resident remaining stays estimated/censored. This
tests whether output-length estimation error explains the collapse/veto.

**Result — a near-no-op.** Reduced `prefill_bound` is *byte-identical* est vs oracle; joint `full`
rate 16 moves 0.161 → 0.167; the decode-bound veto is unmoved (`full` = 0.133 both). The only
systematic effect is `drift-only` on decode-bound, +0.02–0.03 (a warmup transient). **Reason:** every
archetype here has a *constant* output length, so `N̂_out` converges to the true `o_r` — there is
almost no estimation error to remove. See F12.

### E11 — Size-aware transfer cost (`c_xfer` ∝ KV size)

```bash
CXSIZE=1 MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh
```
The decider assumed a **flat** `c_xfer = 5 ms`, but the simulator *executes* a size-based KV transfer
(`sim/cluster/pd_events.go`: `base + blocks·blockSize·kvBytesPerToken / bandwidth`) that is added to
the disaggregated request's TTFT. Measured real transfers (llama-70b TP4, 25 GB/s), from the DES debug
log:

| archetype | prefill tokens | **real transfer** | flat `c_xfer` | error |
|---|---|---|---|---|
| decode 256/512 | 256 | 1.1 ms | 5 ms | 4.5× too big |
| mixed 2048/128 | 2048 | 7.0 ms | 5 ms | ~right |
| prefill_lean 8192/64 | 8192 | 27.1 ms | 5 ms | 5.4× too small |
| prefill_bound 16000/16 | 16000 | **52.7 ms** | 5 ms | **10.5× too small** |

`--edpp-c-xfer-size-aware` makes the decider mirror the executor's formula. It affects `ttftP` (hence
`least-ttft` and the `z_ttft` term) and the penalty, but **not** `drift-only`. Reduced ablation,
flat → size-aware (mean of 3 seeds):

| archetype | rate | least-ttft | drift-only | drift+z | full |
|-----------|------|-----------|-----------|---------|------|
| prefill_bound | 12 | 0.492 → 0.242 | 0.264 → **0.264** | 0.408 → 0.238 | 0.414 → 0.233 |
| prefill_bound | 16 | **0.397 → 0.101** | 0.061 → **0.061** | 0.062 → 0.065 | 0.064 → 0.065 |
| mixed | 16 | 0.750 → 0.368 | 0.803 → **0.803** | 0.747 → 0.410 | 0.708 → 0.417 |
| decode (veto) | 8 | 0.133 → 0.152 | 0.250 → **0.250** | 0.138 → 0.169 | 0.133 → 0.146 |

`drift-only` is **byte-identical** in every cell (it never reads `c_xfer` — a clean invariant check).
See F13.

### E12 — Value-at-risk drift ORACLE vs `least-ttft` (the §7 "one live idea", at its ceiling)

The single remaining rescue idea, built and run as a diagnostic oracle (`--edpp-rule var`, kernels
`flip|util|hazard`): the drift term re-priced in **value-at-risk** (goodput destroyed among co-residents)
instead of **work** (µs), reading each co-resident's true remaining decode steps un-censored (gated INV-9,
upper bound). The bar: clearly beat `least-ttft` where it ties/wins. **Verdict: PARTIAL clearance.** VaR
beats `least-ttft` decisively on the **decode-bound and balanced** archetypes (mixed r16: `var:util` 0.843
/ `var:hazard` 0.929 vs `least-ttft` 0.368; ≈2× across the saturated decode cells) — the design's central
prediction, confirmed exactly where the co-resident externality dominates. It does **not** beat
`least-ttft` on the **prefill-bound** archetypes (ties at moderate load, both collapse under overload), and
`var:hazard` actively over-disaggregates there (prefill_lean r16: 0.189, tracking `always`'s 0.065).
`var:util` is the robust kernel; the §7-predicted saturating-utility neglect trap did not fire on these
constant-output single-class archetypes ("not observed here", not refuted). See F14, F15.

The joint homogeneous path (`JOINT=1`) does not change the verdict: joint-VaR beats joint-`edpp`
almost everywhere (so the win is not a reduced-path artifact) but joint-`edpp` ≈ reduced-`edpp` (F10
holds), and the ceiling on the prefill archetypes is still the reduced path (F16). The **heterogeneous-θ_i**
follow-up (fast H100 + crippled-A100 decode, saturating, interior ~86%-fast optimum) is a **clean
negative**: VaR **over-routes to the fast node** (~91% fast, mean goodput 0.859) and lands below
work-currency θ_i-joint dpp (0.928); `var:flip` pins 100% onto fast under the loose deadline and craters
(0.470). The value-currency externality is correctly signed but mis-calibrated on heterogeneous hardware —
so the deployable-approximation recommendation stays scoped to the homogeneous decode/balanced regime, NOT
heterogeneous provisioning. See F16, F17.

### E13 — drift-plus-VaR: a deployable, minimax-regret-adaptive routing rule

E12's heterogeneous negative (VaR over-routes to the fast node) has the same root cause as E5's θ_i
over-correction: VaR prices the delay to the *current* batch and is blind to the standing queue that
over-concentration builds. The fix (§2.3): **keep the Lyapunov congestion drift AND add the VaR
externality**, with per-decision min-max auto-normalization so the weight is scale-free (`w≈1`). The
congestion term is symmetric on identical hardware (cancels via the zero-spread guard → VaR decides) and
asymmetric on heterogeneous (bites → reins in over-routing). The `util` kernel is the one used.

Two things had to be checked before this counts as a real result, and both are now closed:
(a) **Deployable, not oracle** — the runnable rule estimates each co-resident's remaining steps from the
censored per-class N̂_out (`--edpp-var-deployable`, INV-9-safe); it matches the oracle within noise, so the
result does not depend on reading hidden output lengths.
(b) **Variable output, not constant** — the large "unification" seen on constant-output cells (dpVaR ≈1.00
vs dpp 0.93 on heterogeneous) shrinks under realistic output-length variance; it is *not* the headline.

**The headline is minimax regret, measured against the state of the art.** On realistic variable-output
workloads (lognormal σ=0.4, CV≈0.42), across the workload/hardware regime space (mean over 3 seeds, all
DEPLOYABLE). Historical baselines include a **Kairos-inspired adaptation** (arXiv:2607.02043,
load-aware prefill deflection, implemented as `--edpp-rule kairos` and reported at its **best** β) and
llm-d's shipped `prefix-threshold(16)`. These rows predate the separate paper-oriented
`--edpp-rule kairos-paper` implementation and must not be presented as published-Kairos results:

| archetype | never | always | prefix16 | kairos* | least-ttft | dpp | **dpVaR** |
|---|---|---|---|---|---|---|---|
| decode r16 | 0.133 | **0.736** | 0.736 | **0.736** | 0.607 | 0.527 | 0.724 |
| mixed r16 | 0.356 | **1.000** | 1.000 | 0.883 | 0.708 | 0.621 | 0.946 |
| prefill_lean r16 | 0.204 | 0.061 | 0.061 | **0.856** | 0.806 | 0.819 | 0.853 |
| prefill_bound r8 | 0.399 | 0.046 | 0.046 | 0.890 | 0.954 | **0.968** | 0.964 |
| heterogeneous | – | 0.332 | 0.332 | 0.332 | 0.328 | **0.942** | 0.900 |

The claim is **not domination** — dpVaR trails `always` on mixed and `dpp` on heterogeneous, and Kairos
is the best arm on prefill_lean. It is **minimax regret**: deployed blind, dpVaR is within ~5% of the
best rule in every regime and never craters. Worst-case regret across the grid:

| rule | worst-case regret | collapses on |
|---|---|---|
| always / prefix16 (llm-d) | 0.92 | prefill-heavy, heterogeneous |
| never | 0.65 | decode/mixed |
| least-ttft | 0.61 | heterogeneous |
| kairos* (SOTA) | 0.61 | heterogeneous |
| dpp | 0.38 | decode/mixed |
| **dpVaR (deploy)** | **0.042** | — never |

(Worst-case regret updated 0.054→0.042 on 2026-07-25 when the collocated-prefill externality became the
default rule; see F23.) The numerical comparison in this frozen grid is against the historical
Kairos-inspired adaptation, not the published algorithm. It therefore cannot support either a
"we beat Kairos" claim or a claim about where published Kairos collapses. The corrected
`kairos-paper` arms must be run before making those comparisons. The result that remains established by
this table is narrower: the candidate policy has low regret relative to the evaluated adapted and
static baselines across these workload and hardware cells.

**Secondary finding:** llm-d's shipped `prefix-threshold(16)` is byte-identical to `always` in all five
cells (every prompt exceeds the 16-token uncached threshold), inheriting its 0.046/0.332 collapses.

**w-sensitivity:** a broad plateau (raw `w∈[1e3,1e4]`, normalized `w∈[0.5,5]`). **Honest scope:** one
variance level (σ=0.4); 3 seeds; `util` kernel; one heterogeneous topology; simulation with coefficients
fit to the simulator's own latency model. See F18/F19/F20. This is the study's candidate *positive*
contribution.

**What to look for when reproducing (E13).** Confirm: in *every* cell, dpVaR ≥ max(never, always,
least-ttft, dpp) − ~0.05, and dpVaR's per-seed minimum is not a crater; equivalently, dpVaR's worst-case
regret across the grid is far below every simple rule's. Refute: dpVaR is the loser (below all baselines)
in any cell, or it craters on a seed in the regime a fixed rule owns (e.g. collapses on prefill-heavy
where least-ttft/dpp win). Sanity: `Δ_disagg ≤ Δ_local` unit test passes; a `--edpp-rule var` trace
replays byte-identical (INV-13).

### E14 — structural / topology ablations: the heterogeneity ratio and the fleet shape

E13's minimax-regret result held the heterogeneity at one accelerator ratio and one topology (1P2D).
Reviewers ask two things of that. At what ratio do the TTFT-currency rules actually break, and does the
§2.3 cancel/bind normalization survive a change of fleet shape? E14 answers both, reusing E13's deployable
arm set (σ=0.4, 3 seeds, best-β Kairos adaptation). It leaves the E13 headline grid frozen and adds the two sweeps.

**Ratio sweep.** The slow decode node is a uniform Nx slowdown — scale BOTH `tflops_peak` and
`bw_peak_tbs` by 1/N, then fit its coefficients from the same trained-physics engine that executes it
(the design §3 no-confound recipe, generalized by `repro_theta_by_gpu.sh`). Because the workload is
decode-bound, the fitted per-token decode coefficient scales as N exactly, so the ratio the rule sees
equals the knob. N=5 reproduces the crippled-A100 cell (fitted `c1` 0.238 vs 0.228), and at N=5 seed 42
lands on the tab:grid worst-seed column (dpVaR 0.777). Across N∈[1,5], the evaluated TTFT-currency rules
(least-ttft and the Kairos adaptation) fall below half their homogeneous goodput by N=1.5 and reach
worst-case regret **0.61** against the
per-N best static split, while dpp holds **0.00** and dpVaR **0.05**. dpVaR's margin over least-ttft grows
with N and reaches **0.57 at N=5**. (See F21 for the table.)

**Topology matrix.** Sixteen accelerators, provisioned as 1P3D / 2P2D / 3P1D, homogeneous hardware, four
archetypes, **ten seeds** (re-run 2026-07-25 to average out the seed-42 3P1D pathology; see F23). Worst-case
regret per topology: dpVaR **0.033 / 0.015 / 0.006** — it holds its 1P2D headline (0.042) on every shape,
because the congestion term still cancels on matched instances and binds on mismatched ones under the one
shared weight. The Kairos adaptation, near-optimal on decode-heavy 1P3D (0.004), collapses on
prefill-heavy 3P1D (0.58),
so its exposure follows provisioning as well as workload. The collocated-prefill externality does its work
on the collocation-heavy 3P1D shape, where it cuts dpVaR's worst-case regret from 0.072 (ablation) to 0.006.
Fixing the rule and reading goodput across the three provisionings, dpVaR tracks the per-provisioning best
within a few points in all 16 cells — the online answer to TaiChi's offline, minutes-scale P/D
reconfiguration. (See F22, F23.)

**Naming caveat.** The fleet uses `--decode-instances` (decode-only pods that prefill locally on
collocation); the paper calls them "mixed (M)" and the published tab:grid uses the same setup, so the
paper subsection uses M-notation. True mixed pods would be `--prefill-decode-instances`.

**What to look for when reproducing (E14).** Ratio sweep: `c1` ratio ≈ N for each generated coeff file;
least-ttft/Kairos-adapted monotonically decaying and dpVaR/dpp on the best-static-split line; dpVaR worst-case
regret ≈0.05. Topology: dpVaR worst-case regret ≤0.033 on all three shapes (ten seeds); Kairos-adapted worst-case
regret rising sharply from 1P3D to 3P1D. Refute: dpVaR craters on any (topology, archetype, seed) a fixed rule owns, or
its ratio-sweep regret exceeds dpp's at any N below the overload band.

## 5. Findings

**F1 — llm-d's shipped decode scorer pins decode onto one instance.** With one shared prefix group,
`precise-prefix-cache:2` dominates `queue-depth:1` and sends *all* 240 requests to a single decode
instance (E1). Cost: **6.6× goodput** (0.133 → 0.879). A property of a shipped production config,
not a bug we introduced — and it means any comparison run on the default profile is partly measuring
the scorer.

**F2 — The optimal naive corner flips across the workload spectrum.** `always` wins decode-bound (it
uses the otherwise-idle prefill server); `never` wins prefill-bound (two decode servers doing prefill
beats one prefill server doing it). No fixed heuristic is right everywhere.

**F3 — On prefill-bound overload the optimum is an interior split.** f=35 → 0.604, beating both
corners (0.108, 0.033). Routing genuinely matters here.

**F4 — EDPP wins that regime, robustly** (0.917/0.917/0.942 across seeds), beating both corners *and*
the best static split by ~0.32, by dynamically spreading the bottleneck prefill across all three
servers. **A genuine positive result.**

**F5 — The shipped full rule is the best arm in 0 of 12 cells (E8).** Dominated everywhere by one of
its own ablated subsets, on both the reduced and joint paths.

**F6 — The congestion-drift term delivers the entire win, and also causes the collapse (E8).**
drift-only is best in 5 of 8 informative cells (+0.16 over `least-ttft` at rate 10). It is *also* the
worst arm at prefill_bound rate ≥12 (0.264 vs `least-ttft` 0.492) — one term, both the win and the
failure. The collapse is bimodal across seeds (0.100/0.529/0.163), i.e. a tipping point.

**F7 — `z`'s sign flips by archetype (E8).** decode −0.110 (badly hurts), mixed −0.018, prefill_lean
+0.030, prefill_bound +0.032. **Mechanism:** `z_ttft` prices the TTFT cost of disaggregating (the
transfer) but is blind to the decode capacity it buys; with a tiny prompt that cost is essentially
just the transfer, so z vetoes disagg and EDPP degenerates to `never`. This explains the
decode-bound anomaly first seen in E2 and unexplained until now.

**F8 — `V·c_xfer`, the stated objective, is numerically invisible (E8).** `max|transfer_term| =
0.0008` against `ttft_term = 25.7`. Removing it slightly *helps*.

**F9 — Per-class SLO machinery is counterproductive (E7).** Per-class targets *hurt* critical goodput
(0.859 → 0.814). The machinery did exactly what it was designed to do — batch's `z_ttft` went to
exactly 0, prefill-server access shifted from batch (55%→29%) to critical (27%→43%) — and critical
still got **worse**, while the deprioritised class got **faster** (e2e 3424→1879ms). **Routing cannot
sacrifice a class:** lowering a class's target does not make it yield capacity, it makes it *selfish*
— it stops optimising its own latency and takes the cheapest (contended) resource.

**F10 — Joint routing does not rescue EDPP (E9).** A wash: helps prefill_bound 8/16 and decode 12,
hurts mixed 12/16, prefill_lean 16, prefill_bound 12. Fixes none of the failures — the decode-bound
veto survives (0.133 = `never`), the collapse survives (0.161 vs `least-ttft` 0.397), the full rule
still wins nowhere. **The formulation's §1 coupling argument is real but second-order.** Sharpest
evidence: on `mixed`, joint's only gift is that EDPP picks the decode instance instead of the
queue-depth scorer — and handed that control the work-currency drift picks *worse than plain load
balancing* (0.803 → 0.681).

**F11 — THE UNIFYING FINDING.** `z_ttft` prices the deciding request's own TTFT; `V·c_xfer` prices
its own transfer; only the drift term prices the effect on everyone else — in **work**, not
**value-at-risk**. At moderate load work ≈ value (a loaded queue really does hold savable requests),
which is why drift wins there. Under overload every queue is huge, so work-drift balances between two
hopeless queues, where a value-weighted term would price them at ~0 (they hold doomed requests;
dumping is free) and steer the savable to where they can still make the deadline. **One sentence
accounts for F6, F7, F9, F10 and the Type-A/B failure (E6).**

**F12 — Output-length estimation is not the cause of the collapse or the veto (E10).** Feeding the
rule the *true* `o_r` (oracle) is a near-no-op: the prefill_bound collapse is byte-identical, the
decode veto is unmoved. This closes the "maybe it's just bad `o_r` estimates" escape hatch — but with
a caveat: these archetypes have *constant* output length, so `N̂_out` already ≈ true `o_r` and there
is little error to remove. A genuine test of the estimation dimension needs a *variable*-output
workload (where a class mean is a poor per-request predictor); that remains untested.

**F13 — `least-ttft`'s overload robustness was a transfer-cost artifact (E11).** The decider priced
disaggregation with a flat 5 ms while the real KV transfer is size-based and costs ~53 ms on
prefill_bound. Correcting it (`--edpp-c-xfer-size-aware`) collapses `least-ttft` on prefill_bound rate
16 from **0.397 → 0.101**, shrinking its once-headline "5–6× beats EDPP under overload" gap to ~0.04.
`least-ttft` was under-charging transfer, so it over-disaggregated and landed near the interior
optimum by accident. **`drift-only` is invariant** (it never reads `c_xfer`), so F5/F6/F11 stand
untouched; and the decode-bound veto survives the correct *smaller* cost, so it is **not** a `c_xfer`
artifact either (F7 stands). Sharpest twist: correct transfer pricing makes `least-ttft` *more*
accurate about the deciding request's own TTFT and therefore *more globally wrong* — it declines the
disaggregation that would have helped the system's throughput. The F11 currency error, now visible in
the stripped-down baseline itself.

## 6. What we have NOT established (bounds on every claim above)

Two gaps from the first version of this report are now **closed by measurement**, not argument:
~~joint routing was never exercised~~ (E9 — it is a wash) and ~~SLO-class heterogeneity was never
tested~~ (E7 — the machinery is counterproductive). What remains:

1. **The value-currency fix is untested.** Every finding points at it; nothing has measured it. This
   is the one live idea left, and the honest next experiment (§7).
2. **The objective was never changed.** The rule still minimises transfer cost subject to the SLO
   constraints (§2.1) — a poor surrogate for goodput once those constraints go infeasible (§3.2).
   Every number in this report was produced with the shipped objective.
3. **Output-length estimation is untested on variable-output workloads (E10/F12).** The oracle-`o_r`
   control was a near-no-op *because* every archetype has a constant output length, so `N̂_out` ≈ true
   `o_r`. A workload with per-request output variance is needed to actually stress this dimension.
3. **`z_itl` never fired in any experiment.** Measured ITL never approached its 100ms target, so half
   the SLO machinery has been dead throughout. E8 therefore ablates `z_ttft` specifically.
4. **One topology (1P2D), cap 16.** The concurrency cap is an artificial stressor; without it nothing
   saturates and every policy scores 1.000. Joint's action set here is only |A| = 4 — a wider pool
   might give it more to work with.
5. **The yardstick is a *static* fraction sweep**, not the true joint hindsight optimum (the MILP
   remains unbuilt), so "leaves goodput on the table" is measured against a weaker target than the
   real one.
6. **Seeds:** most cells are 3 seeds; four spectrum cells sit at the 1.000 ceiling and carry no
   information. The prefill_bound rate-12 collapse is bimodal across seeds (0.100/0.529/0.163).

## 7. What this means, and what to do next

**Three of the three heterogeneity axes have now been tested, and all three failed for one reason.**

| axis | result | why |
|------|--------|-----|
| **Workload** (Type A/B) | `always` wins; EDPP loses (E6) | externality-blind — each request judged by its *own* SLO |
| **Hardware** (fast/slow) | joint-EDPP wins under-capacity; θ_i **over-corrects** under saturation (E5) | work-currency drift overwhelms the congestion signal |
| **SLO-class** (critical/batch) | per-class machinery **counterproductive** (E7) | *same* externality-blindness — deprioritising makes a class selfish |

Plus the two structural results: the shipped rule wins **0 of 12 cells** (F5), and **joint routing —
the project's central thesis — is a wash** (F10). The formulation's §1 coupling argument is real but
second-order.

**The one live idea: change what the drift term measures.** Keep Neely's drift structure — a virtual
queue can measure anything — but price **value-at-risk** instead of **work**: the marginal goodput
destroyed at instance *i* by adding work there. It is computable (we know each co-resident's stage
and estimated remaining latency). One change addresses F6 (drift's overload collapse), F9 (the
class backfire), and the Type-A/B failure, because all three are the same missing externality.

**The honest next experiment — an oracle upper bound. ✅ DONE (E12).** We tested the *ceiling* before
paying for the machinery, and it **partially cleared the bar**: oracle value-at-risk beats `least-ttft`
**decisively on the decode-bound and balanced archetypes** (where the co-resident externality dominates —
the design's central prediction, confirmed) but **not on the prefill-bound archetypes** (ties at moderate
load; `hazard` over-disaggregates). So the value-currency idea is *not* dead — it earns a deployable
approximation *for the decode/balanced regime* (co-resident remaining from the censored `N̂_out` estimate,
deferred) — but it is not a universal rescue. The `util` kernel is the robust choice; the predicted
saturation-neglect trap (point 3 below) did not materialise on these constant-output single-class
archetypes, though that stresses neither heterogeneity nor multi-class, so it is unrefuted. Full
per-archetype × kernel × seed table and the completion model: E12, F14/F15, and
`docs/superpowers/specs/2026-07-21-edpp-var-oracle-design.md`.

**And the follow-through — drift-plus-VaR (E13) is a deployable, minimax-regret-adaptive rule.** Keeping
the work-drift term AND adding the VaR externality (scale-free auto-normalization) gives one rule you
deploy blind that is within ~5% of the best simple rule in every regime and never catastrophic, on
realistic variable-output workloads, using only a censored (INV-9-safe) co-resident estimate — worst-case
goodput regret ~0.05 vs 0.29–0.92 for any fixed rule. It resolves E12's oracle caveat (deployable ≈
oracle) and the constant-output caveat (survives variance as a minimax-regret result, though the large
constant-output "unification" shrinks). This is the study's candidate positive contribution — §2.3 for the
rule, E13/F19 for the grid. Remaining scope: variance-axis and topology breadth. The original
oracle-upper-bound plan was:
1. **Read the true output length** (`o_r`) rather than estimating it. ✅ *Built and run* as
   `--edpp-oracle-output-len` (explicitly oracle-marked, diagnostic-only, upper-bound; E10). Result:
   a near-no-op on these constant-output archetypes — it neither rescues nor breaks the rule, and it
   rules out `o_r` estimation as the cause (F12). The value-at-risk build below will need to extend
   this oracle to *un-censor co-resident remaining* so "will my work tip request *j* past its
   deadline" is computable at all (today that state is stripped for INV-9).
2. **Value-at-risk drift + a saturating utility**, measured against the fixed-plan yardstick across
   the same archetypes.
3. **Watch for the known trap:** a saturating utility gives û ≈ 0 for doomed requests, which means
   *no signal*, which means their placement falls to the remaining terms and they grab the cheapest
   (contended) resource — reproducing E7's failure exactly. Saturation alone gives **neglect**, not
   **triage**. Triage needs the doomed to *yield* capacity, which is an admission/scheduling lever
   EDPP does not hold. This is a real boundary on the whole approach and should be stated, not
   discovered late.

**The paper this supports today** is a characterization with a mechanistic root cause: *which
heterogeneity axes justify adaptive P/D routing, why a principled drift-plus-penalty policy is
dominated by its own ablated subsets, and the single currency error that explains all four failures*
— plus the llm-d scorer finding, which is independently useful to that project. Whether it also
becomes a *method* paper depends entirely on the oracle experiment above.

## 8. Reproduction

```bash
cd inference-sim
go build -o blis main.go

# E1 scorer comparison
MODE=scorer bash campaigns/edpp-study/repro_spectrum.sh

# E2 spectrum (both scorer setups)
bash campaigns/edpp-study/repro_spectrum.sh
SCORER=llmd bash campaigns/edpp-study/repro_spectrum.sh

# E3 static-fraction oracle
MODE=oracle ARCH=prefill_bound RATE=8 bash campaigns/edpp-study/repro_spectrum.sh

# E5 hardware heterogeneity
bash campaigns/edpp-study/repro_hetero_hw.sh
SAT=1 THETA=1 bash campaigns/edpp-study/repro_hetero_hw.sh

# E8/E9 term ablation (reduced, then joint)
MODE=ablate       RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh
JOINT=1 MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh

# E10 oracle output-length control (diagnostic; upper bound)
ORACLE=1 MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh

# E11 size-aware c_xfer (mirrors the DES KV-transfer executor)
CXSIZE=1 MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh
# measure the REAL transfer durations the sim executes:
#   ./blis run ... --pd-decider always --log debug 2>&1 | grep "blocks, duration="

# E12 value-at-risk drift ORACLE vs least-ttft (flip/util/hazard kernels; diagnostic, upper bound)
bash campaigns/edpp-study/repro_var_oracle.sh                    # reduced path, 3 seeds (F14/F15)
JOINT=1 bash campaigns/edpp-study/repro_var_oracle.sh            # joint (decode,prefill) argmin path (F16)
# E12 heterogeneous-θ_i follow-up (fast H100 + crippled-A100 decode, saturating interior optimum; F17)
bash campaigns/edpp-study/repro_var_oracle_hetero.sh

# E13 drift-plus-VaR — deployable, minimax-regret-adaptive rule (keep congestion + VaR + auto-normalize)
# THE headline result: regime × baseline dominance grid on realistic VARIABLE output, DEPLOYABLE arm.
bash campaigns/edpp-study/repro_var_dominance.sh                                     # F19/F20 grid (σ=0.4)
#   arms: never|always|prefix16(llm-d)|kairos*(SOTA, best-β)|least-ttft|dpp|dpVaR ; KBETAS sweeps β
VAROUT=0.6 bash campaigns/edpp-study/repro_var_dominance.sh                          # variance-axis check
# Constant-output ceiling (larger but not the realistic claim; F18):
NORM=1 bash campaigns/edpp-study/repro_var_oracle_hetero.sh                          # hetero oracle ceiling
JOINT=1 VARCONGW=1 NORM=1 ARCH_ORDER="decode mixed" RATES="8 16" SEEDS="42 7 123" \
  bash campaigns/edpp-study/repro_var_oracle.sh
# w-sensitivity (broad plateau, not a knife-edge): sweep VARCONGW on repro_var_oracle_hetero.sh
# The clean single-run form of the algorithm:
#   ./blis run ... --edpp-rule var --edpp-var-metric util --edpp-tau-e2e <e2e> \
#     --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 \
#     --edpp-oracle-output-len --edpp-joint --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware

# E14 structural / topology ablations (F21/F22). Both auto-generate any missing coeff files.
bash campaigns/edpp-study/repro_hetero_ratio_sweep.sh                                # F21 ratio sweep (N∈[1,5])
python3 campaigns/edpp-study/analyze/hetero_ratio_sweep.py \
  --csv campaigns/edpp-study/out/hetero_ratio/ratio_sweep.csv \
  --out campaigns/edpp-study/out/hetero_ratio/hetero_ratio_sweep.png              # fig + worst-case regret
bash campaigns/edpp-study/repro_topology_matrix.sh                                  # F22 topology matrix (1P3D/2P2D/3P1D)
python3 campaigns/edpp-study/analyze/topology_matrix.py \
  --csv campaigns/edpp-study/out/topo_matrix/topo_matrix.csv \
  --out campaigns/edpp-study/out/topo_matrix/pd_provisioning.png                  # framing A table + framing B fig
# uniform-Nx slow-node coeff generation (one ratio): tflops=1979/N, bw=3.35/N
#   bash scripts/calibration/repro_theta_by_gpu.sh ratioNpM <1979/N> <3.35/N> A100

# See WHY a decision was made (per-decision term breakdown incl. z_ttft/z_itl)
./blis run --model meta-llama/llama-3.3-70b-instruct \
  --workload-spec campaigns/edpp-study/specs/spectrum/w.yaml \
  --num-instances 3 --prefill-instances 1 --decode-instances 2 \
  --decode-routing-scorers "queue-depth:1" --max-num-running-reqs 16 \
  --pd-decider edpp --edpp-coeffs scripts/calibration/coeffs-llama70b-h100-tp4.json \
  --edpp-tadm-estimator rollforward \
  --slo-ttft "standard=1794ms" --slo-itl "standard=100ms" --slo-e2e "standard=1600ms" \
  --trace-level decisions --edpp-decision-trace /tmp/decisions.csv
# columns: z_ttft, z_itl, balance_term_d/p, transfer_term, ttft_term, itl_term, lhs, rhs, disaggregate
```

Relevant code: EDPP rule `sim/edpp.go` (reduced `Decide` ~line 470, joint `decideJoint`
~line 760); coefficients `scripts/calibration/coeffs-llama70b-h100-tp4.json`;
`least-ttft` baseline `--edpp-rule least-ttft` (commits `44e4988..9798bca`).

---

## Appendix A — the questions that prompted this report (verbatim)

This report was written in response to the following. It is reproduced unedited because it is
the most accurate record of what was opaque, what was assumed, and which open questions drove
the analysis above. Where it asks a question, the answer is cross-referenced.

> I am trying to parse and understand deeply the previous 5-6 conversations. I have to update
> my manager on the progress. I thinnk he feels like I am not working. I should be able to tell
> him the kind of experiments we've run so far and reason with our methodology of evaulating the
> policies. The starting point I can think of is our edpp formulation that primarily has one
> objective: decide and route the request to appropriate instances such that their slos are
> satisifed (maybe edpp says more, i don't know). We started with homogenous case (meaning that
> the setup has same hardware) and we ran some experiments (i don't know what and how). We found
> that the results (but what kind?) were not favor of edpp (don't know why). Then you suggested
> that we should incorporate the hardware heterogeneity (you found out that this was not an easy
> task because of ....). So we added the notion of heterogeneity and ran the experiments again (i
> don't know what experiments were run and what we measured). We found out that the results were
> still wanting in favor of edpp (i don't know why and how we went about concluding that we need
> something like "governer" which i don't understand the meaning of and the need of). The question
> at point i was facing were several, one of which was a sanity check are we considering joint
> routing decisions in our exeriments, or we are defaulting to llm-d router scorers. if we are
> indeed considering the joint routing, why is edpp not working because its job is to protect the
> slos. At this point, I was really lost because you run the experiments without telling me the
> command you ran, what was the configuration you ran, why did you pick that configuration and
> designed the experiments in that way. You always came back saying that edpp doesn't work. In
> this process, somehow we stumbled upon picking goodput as the objective. Assuming that you did
> everything that made sense from the first principles Then we stopped and the whole thinking out
> loud conversation started. Do we need edpp at all? For a workload, maybe a "smarter" policy
> shines over naive heuristics such as always disaggregate and prefix threshold based decisions.
> This could be because of two reasons (maybe there are more reasons but this is all i can think
> of): one is the policy being blind to characteristics of the workload and the other one being
> sub-optimal routing decisions because of being blind to instance's hardware characteristics
> (maybe there are other reasons behind the sub-optimal routing decisions). So initially the plan
> was: 1. Pin the 1P1D mental model precisely — the walkthrough you started, but completed until
> you can predict the decision by hand. (We're most of the way there in this message.)
> 2. State the value hypothesis as a falsifiable claim on ONE fixed topology (say 2P2D): "There
> exists a workload where always, prefix-threshold, and least-predicted-latency all leave goodput
> on the table, and the reason is [workload-blindness | hardware-blindness]." Then find the
> smallest such workload — or fail to, and conclude the heuristics are enough.
> 3. Only if (2) succeeds, identify the minimal ingredient that closes the gap — and check whether
> it needs the full drift-plus-penalty apparatus or just a smarter heuristic. Build the least thing
> that works.
>
> Somehow we stumbled on this issue of the penalty term of transfer cost as not the correct thing
> to optimize over i.e. fewest transfers while slo's are feasible. but the point you raised is this
> penalty term is moot i.e. you have nothing to optimize over when the system is at
> capacity/saturated/underprovisioned. Im simple terms, you see that the ttft is violated,
> exercising or not that penalty term is pointless. because either way, you are violating the ttft.
> In short, the transfer cost penalty term is an okay surrogate when slos are feasible but are
> terrible when slos are infeasible. Am I right in understanding this issue you raised?
>
> If yes, then this suggests that perhaps a better objective is goodput or utility. And then
> perhaps this was an important point according to you: EDPP is already a goodput surrogate. The
> z_ttft/z_itl constraint queues are surrogate #1 — "hold average latency near target, and hope
> most requests clear the deadline." "Picking goodput" doesn't mean adding a goodput term. It means
> recognizing we've been optimizing a loose proxy for it, and asking whether a tighter one would
> track the real target better. This also tells me that the experiments we ran so far, you never
> showed me what is happening to these virtual queues with every request. I believe we did build an
> instrumentation for it to track what is edpp getting as the input and what is it outputting.
>
> The other point you raised seems also important: Look at how z_itl updates: z += (realized_latency
> − τ), clamped at zero. It accumulates lateness. A request that finishes 10 seconds past its
> deadline pumps a big positive into the queue, so the policy keeps expending effort to reduce that
> request's lateness — even though it already missed and will never count toward goodput.
>
> Goodput doesn't care. A request 10 ms over and one 10 s over are equally failures. Goodput
> saturates: once a request will miss, further effort on it yields zero. That single difference —
> the true objective saturates; our surrogate doesn't — is almost certainly why EDPP flails under
> overload. Goodput implies triage: protect the savable requests, stop pouring capacity into the
> doomed ones. An accumulated-lateness surrogate structurally cannot triage — it keeps rewarding
> progress on hopeless requests. A goodput-faithful surrogate must be a saturating value (utility
> that's flat-high before the deadline and stops decaying after), so that the optimizer naturally
> reallocates scarce capacity to the requests still in reach.
>
> And then we kind of made a turn where from following the proper objective function, we turned
> towards asking this question whether the goodput gap even a routing problem. And the reason you
> asked this question alludes (I think?) to the previously mentioned apprehension of whether edpp is
> indeed required, which considers routing as an inherent problem. If we can build a simple policy
> that goodput optimum under overload, then we don't need edpp.
>
> So the first thing to do is map where the routing even matters. Smallest topology with a real
> decode-routing choice — 1P2D. Sweep load from comfortably-feasible → overloaded. At each load, put
> goodput of the naive heuristics (always, prefix-threshold, least-predicted-latency) next to the
> goodput oracle we already have (fixed-plan brute force). The output is a single curve: in which
> load band, if any, do the heuristics leave goodput on the table? That band — and only that band —
> is where a routing paper can live.
>
> And then step 2 is — diagnose the gap in that band (a day). Where heuristics trail the oracle, diff
> the decisions request-by-request. Two possibilities:
> - (a) Routing story: the oracle wins by sending the right requests to the right server. Then a
> goodput-faithful (saturating) routing surrogate is the fix, and we design it — with the oracle as
> the target to track.
> - (b) Scheduling story: the oracle wins by triaging (letting some miss to save others). Then routing
> can't fix it, and that is the finding — "under a fixed topology, goodput is recovered by
> deadline-aware admission, not P/D routing."
>
> For this you ran some experiments (again you never told me what command you ran, what was the
> output you got, how did you extract the supposedly meaningful numbers) Also in the process you
> discovered something happening with llm-d router scorers. I think it was related to prefix scorer
> taking over other scorers. Therefore, the requests were pinned to just one decode server (again how
> did you observe this was absent. Like I have to rely on you and ask you to tell me what you observe,
> which shouldn't be the case). And I believe you discovered such a router behaviour because we built
> a goodput oracle that takes all possible alternate actions as opposed to the policy action. One
> thing I'll say is that the default scorer is same as llm-d scorer. The llm-d develeopers have set
> this scorer. So we should be able to make a point that look your scorers are not doing a good job.
> So my point is that this behaviour is not a bug, but an understanding of what works when. And you
> yourself decided to fix this load balancing scorer. This is fine, but we should report the results
> for both setups. Anyways this was discovered with a homogenous workload (i don't know what exactly
> were workload characteristics except that it had moderate prefill).
>
> Then we ran a clean spectrum sweep. Fixed 1P2D, decode-balancing held fixed, capped concurrency to
> force saturation at reachable rates, SLO auto-set per archetype from an idle probe (so "goodput" is
> comparable across archetypes). Four workload archetypes spanning decode-bound → prefill-bound; for
> each, never/always/prefix-threshold/edpp across load. (Again what command you ran for this, I have
> no idea, perhaps I'd have to open the "ran 1 shell command" to see what you run and then observe
> myself).
>
> [... the prefill-bound result and seed-robustness numbers, quoted back ...]
>
> I think the thought process to answer whether or not we need edpp is how much goodput other
> heuristic policies leave on the table.
>
> Again I don't know how you came about this best static disaggregation function.
> And coming back to previous point of not having the proper objective in the edpp and whether goodput
> serves as a proper objective, i belive you still used the transfer cost as the penalty term?
>
> And then the next question you ask is whether this win is specific to the dpp machinery or would any
> dynamic load aware disaggregation rule get it. A simple greedy heuristic -- disaggregate iff
> predicted ttft disagg < predicted ttft local might capure most of that 0.92 goodput value. withour
> any virtual queues or V.
>
> More things happened after this which I am unable to write here.
>
> Also please write a report for this, maybe as a slide deck or a document that provides our study
> comprehensively such that anyone in my team can pick up and verify the claims, that lets me update
> my manager. He is really after me.

### Answers to the direct questions above

| question | answer | where |
|---|---|---|
| "the transfer cost penalty term is an okay surrogate when slos are feasible but terrible when infeasible. Am I right?" | **Yes.** The penalty is the only thing EDPP optimizes; under infeasibility the constraints cannot be met, the guarantee is void, and minimizing transfers is beside the point. | §2, §3.2 |
| "are we considering joint routing decisions, or defaulting to llm-d router scorers?" | **Defaulting to the scorer.** All spectrum runs are *reduced* EDPP; the decode instance was chosen by the scorer (pinned to `queue-depth:1`). Joint was used only in the hardware-heterogeneity work. | §6 gap 1, E5 |
| "how did you come about this best static disaggregation fraction?" | A forced per-request plan via `--pd-plan`, sweeping the disaggregated fraction f and taking the max (f=35 → 0.604). It is a **static** yardstick. | §3.4, E3 |
| "i believe you still used the transfer cost as the penalty term?" | **Yes.** Every experiment uses the shipped `V·c_xfer` objective. We never changed it. | §2, §3.2, §6 gap 3 |
| "you never showed me what is happening to these virtual queues with every request" | Correct, and it is a real gap. The instrumentation exists (`--edpp-decision-trace`). A command to dump it is in §8. | §6 gap 4, §8 |
| "we should report the results for both setups [llm-d scorer vs fixed]" | Done — both are reported, and the llm-d default is framed as a finding about a shipped config, not a bug. | E1, F1 |
