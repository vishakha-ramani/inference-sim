# Implementation audit against the new policy contract

## Scope

This audit originally mapped `infocom/CAUSAL-SLO-EXTERNALITY-POLICY-CONTRACT.md`
to the simulator before the new policy path was added. The findings below preserve
that reuse boundary; the disposition at the end records what the dedicated
implementation now does. It does not reinterpret completed campaigns as evaluations
of the new controller.

The current code already contains most of the necessary prediction machinery, but no
existing flag combination implements the contract exactly. Reusing the old joint
objective without a dedicated path would silently retain terms and queue semantics
that the new formulation does not claim.

## Mapping

| Contract item | Current implementation | Status |
|---|---|---|
| Enumerate all \(D(P+1)\) actions | `decideJoint` enumerates every local and `(decode,prefill)` candidate | Reusable |
| Candidate-specific decode and prefill physics | `coeffsFor`, `apForInstance`, and joint candidate helpers use the candidate locations | Reusable |
| Exact causal overlap | `varJointCandidateBreakdown` supports exact marginal prefill overlap | Reusable when enabled |
| Deployable remaining-output estimate | `reqNHatOut` and censored co-resident estimates avoid future output lengths | Reusable |
| Causal SLO externality | The corrected joint VaR breakdown contains decode, collocated-prefill, and prefill-pool components | Reusable after value-function consistency is fixed |
| Arriving-request SLO value | `jointSelfGood` exists behind `VarGoodputObjective` | Reusable after value-function consistency is fixed |
| Per-instance work split | Candidate scoring charges local `Wp+Wd` to the decoder and remote `Wp`/`Wd` to their selected instances | Reusable |
| Per-instance capacity state | `qByInstance` records waiting work by selected instance | Semantics do not match the contract |
| Fixed-weight net-good plus capacity score | Existing flag combinations also retain TTFT/ITL deficit terms or per-decision normalization | New scorer required |
| Dedicated policy identity | Current behavior is composed from `Rule`, `Joint`, and several `Var*` flags | New configuration path required |
| Full candidate trace | Current trace records only the three VaR components and their total | Must add own-good, capacity, and final-score fields |
| Deterministic tie breaking | Candidate ordering and tolerance are deterministic | Reusable |

## Contract violations to repair

### The util kernel does not implement the declared phase-aware SLO value

`goodSelf` evaluates the arriving request with a TTFT utility multiplied by an E2E
utility, while `gDecodeUtil` evaluates an already-decoding co-resident with only its
E2E utility. Neither util path contains the mean-ITL factor that the binary composite
kernel uses. Consequently the arriving credit and decode-side causal charge do not
use the contract's composite value. The hazard kernel is a weighted delay rather than
a potential difference and likewise cannot support the proposed accounting theorem.

The new policy needs a dedicated bounded composite value that is called by both the
arriving-request and decode-resident projections. Prefill residents retain the
explicitly declared TTFT-only phase value until their assigned decoder state is added
to snapshots. Existing policy behavior should remain available as historical
ablations rather than being changed in place.

### The existing queues contain waiting work, not conserved capacity work

`OnRoute` adds predicted work to `qByInstance`, but `OnAdmit` removes an entire phase's
work as soon as the request enters a running batch. The queue therefore predicts
admission waiting and is useful to admission estimators, but it does not implement

\[
Q_i(t+1)=[Q_i(t)-B_i(t)]^++\Delta W_i(a_t).
\]

The new controller needs a separate virtual workload queue that drains by elapsed
service budget. The existing admission queue should remain unchanged because other
policies and estimators consume it.

### Work booking is not fully location-specific

Joint candidate scoring uses the selected instance's coefficient vector, but
`OnRoute` books committed work with the global coefficient vector. It also returns
early when the prompt has no uncached tokens, which omits the request's decode work.
Thus the scored and committed work can disagree on heterogeneous hardware and for
fully cached requests. The new workload queue must recompute both phase demands from
the committed locations and must always book decode work.

### The existing `V` parameter does not weight projected net good

In the current joint code, `V` scales the standalone transfer penalty. That penalty is
removed when `VarGoodputObjective` is enabled, so `V` does not implement the
net-good-versus-capacity trade-off in the new formula. The dedicated policy must give
`V` one meaning: the fixed multiplier on causal SLO externality minus arriving-request
SLO value.

### The old joint scorer carries undeclared terms

Even with joint routing, corrected VaR, congestion, and arriving good enabled, the
existing `jointCandidateCost` adds TTFT and ITL deficit terms. The normalized branch
also min--max normalizes congestion and the goodput term per decision. Neither
behavior belongs to the initial fixed-weight contract. A dedicated scorer should
reuse the prediction helpers but assemble only the declared three components.

## Safe reuse boundary

The implementation should reuse topology enumeration, cache queries, latency-law
coefficients, admission-delay prediction, output-length censoring, causal completion
projections, and deterministic ordering. It should not reuse the old final scalar
score, the admission-waiting queue as a capacity dual, or the current util/hazard
kernel as the policy's common SLO value.

The historical CLI flags and completed campaign artifacts should remain behaviorally
unchanged. The new controller needs a separate flag and trace schema so that a result
cannot be mistaken for a rerun of the earlier policy.

## Implementation order

1. Add and test a common composite SLO-value function.
2. Add per-instance virtual workload queues with elapsed-time drain and committed,
   location-specific work booking.
3. Add a dedicated joint score with externality, own-good, and capacity components.
4. Extend candidate tracing to expose each component and the exact sum.
5. Add a deterministic sacrifice-decoder regression in which externality alone
   prefers an already damaged node but the complete score does not overload it.
6. Run the complete unit suite before any policy experiment.

## Implementation disposition (2026-07-30)

The dedicated implementation is now wired behind
`--edpp-joint-slo-externality` and
`--edpp-decomposed-slo-externality`, with named no-externality and no-capacity
ablations. It enumerates the complete joint action set (or the scorer-fixed decode
slice), uses one phase-aware composite value, applies the corrected overlap
`max(remote_lead, decode_admission)`, and assembles only

\[
V(\widehat{\mathcal E}-\hat g_r)
+\sum_i (Q_i/S_i)(\Delta W_i/S_i).
\]

The implementation maintains a separate conserved workload queue that drains by
elapsed nominal service. Committed work is recomputed from the chosen locations'
cache states and coefficient vectors, and decode work is retained for fully cached
requests. The admission-waiting queues remain separate and keep their event-exact
drain semantics.

Candidate traces now include causal SLO externality, arriving-request value,
projected net-good cost, decode and prefill capacity terms, the final score, and
chosen-score regret. Focused tests cover all three SLO dimensions, the fixed realized
TTFT factor for decode residents, TTFT-only phase value for prefill residents,
overlap-aware remote TTFT, elapsed-service queue drain, local and remote work
placement, heterogeneous coefficients, score decomposition, decomposition control,
both named ablations, and the sacrifice-decoder regression.

The initial controller is deliberately restricted to disjoint dedicated prefill and
decode pools. Shared-role instances would require a declared multi-resource service
model rather than allowing one role's nominal drain rate to overwrite another's.
No new experiment result or end-to-end policy-regret claim follows from this
implementation status alone.

## Post-implementation validation checkpoint (2026-07-30)

The first pilot exposed a cluster-wiring defect rather than a workload result. The
dedicated policies forced deployable resident-state interpretation inside the
scorer, but the cluster enabled `RunningDecode` and `RunningPrefill` snapshots only
for the older `var` rule. Consequently every resident externality was zero in the
first run. The cluster now enables censored admission detail for every dedicated
resident-externality policy, including dynamically added instances, and a regression
test covers all four joint/decomposed policy identities. After this correction the
full Go suite passes. The public joint least-TTFT comparator was also corrected to
use the overlap-aware remote decode join and is covered by a focused test.

The corrected pilot on `1p3m:synth:medium` completed all ten arms without drops,
timeouts, or unfinished requests. At the development seed, the full controller
reached goodput 0.464, compared with 0.430 without resident externality, 0.435
without capacity prices, and 0.414 for the pilot's 50-percent static joint plan.
Its trace contained a nonzero mean chosen externality of 0.0152. This establishes
that all declared terms are live; it does not establish generalization.

Development-only calibration over the declared grid selected the universal value
`V=4` before the fresh seeds were examined. A broad fresh evaluation was stopped
after 357 of 720 artifacts because the focal conditions had already answered the
research question and completing the matrix was not necessary for the redesign.
All four fresh seeds were complete for the three conditions below:

| condition | full | no externality | no capacity | least TTFT joint | Kairos | selected conditional static |
|---|---:|---:|---:|---:|---:|---:|
| `1p3m:shared:medium` | 0.224 | 0.174 | 0.797 | 0.068 | 0.757 | 0.838 (`phi=1`) |
| `1p3m:shared:near_high` | 0.061 | 0.047 | 0.119 | 0.025 | 0.182 | 0.168 (`phi=1`) |
| `1p3m:rag:near_high` | 0.618 | 0.602 | 0.490 | 0.861 | 0.334 | 0.651 (`phi=0.5`) |

These are held-out four-seed means, but they are a partial evaluation rather than a
completed campaign. They support a negative design conclusion: the present
fixed-scale capacity term is not suitable as the final controller. On shared-prefix
medium load, projected net good favors remote placement in 87.5 percent of decisions,
whereas the capacity term favors local placement in 89.1 percent, and the two terms
conflict in 92.3 percent. Removing capacity raises the actual remote fraction from
56.0 to 87.5 percent and raises goodput from 0.224 to 0.797. Near high load the same
direction persists. On RAG near high, projected net good favors remote placement in
96.1 percent of decisions, while capacity favors local placement in 77.6 percent;
the mean absolute capacity gap is 4.05 against 0.185 for the already-`V`-weighted
net-good gap. Capacity helps relative to removing it in that condition, but it
dominates the welfare term and still trails direct least-TTFT routing by 0.243.

## Capacity-model implication

The preserved paper already contains a more physical capacity account than the
new controller uses. It defines the occupancy of a remote action as dedicated
prefill time plus decode time and the occupancy of a local action as collocated time,
including the baseline saved when prefill shares an iteration with decode. In
contrast, the current capacity queue books marginal `Wp` and `Wd`, converts them to
service through a nominal `mu`, and divides by a fixed `TauRef` scale. That indirect
conversion omits workload-specific baseline occupancy and can distort the relative
price of one prefill instance and several decode instances.

A theory-first successor should therefore define the constrained resources in
occupancy time itself. If `Delta t` is elapsed wall time, each instance can maintain

\[
Q_i^+ = [Q_i-\Delta t]^+ + \Delta T_i(a),
\]

where a remote action books `t^P` on its prefill instance and `t^D` on its decode
instance, while a local action books `t^coll` on its decode instance. This is the
online counterpart of the paper's fleet-capacity inequalities and drains at one unit
of occupancy per unit of wall time, eliminating the nominal-`mu` conversion. Placing
the new demand outside the positive part is causal: idle service before the arrival
cannot serve that request, and it matches the original contract's queue update. The
causal SLO externality remains the resident-welfare term. The RAG result also shows
that projected own good does not automatically replace a continuous own-latency
signal; any such signal must be introduced through an explicit objective or
constraint rather than as an undeclared penalty.

The paper's per-decision min--max normalization explains why its older empirical
rule avoided raw scale domination, but it is adaptive reweighting and the paper
itself states that it is not a fixed-weight drift transformation. Restoring it may
be a useful diagnostic ablation, but it should not be the final mechanism if the
goal is a standard primal-dual guarantee. No end-to-end policy-regret claim follows
from the partial evaluation or from the proposed occupancy queue.

## Occupancy-capacity successor implementation (2026-07-30)

The physical capacity successor is now available as an opt-in refinement of either
dedicated controller through `--edpp-slo-externality-occupancy-capacity`. The
evaluated marginal-work controller remains the default so its existing artifacts
retain their original meaning.

The new mode fixes the reference decode width (B) to
`--max-num-running-reqs` and books the paper's per-request demands directly. A
remote action books

\[
t^P_{r,p}=n_c(r)\alpha^P_p+W^P_{r,p},\qquad
t^D_{r,d}=\hat o_r\alpha^D_d/B+W^D_{r,d},
\]

on the selected prefill and decode instances. A local action books

\[
t^{\mathrm{coll}}_{r,d}=t^D_{r,d}+W^P_{r,d},
\]

so collocation shares the prefill-iteration baselines instead of charging them
twice. Cache state, coefficients, and the censored output estimate remain
candidate-specific. Each queue is stored in microseconds of occupancy, drains by
one microsecond per microsecond of elapsed wall time, and enters the score as the
physical-seconds cross term (Q_i\Delta T_i). Neither nominal `mu` nor `TauRef`
appears in this mode.

Focused regressions cover configuration guards, local/remote placement, the
collocated baseline saving, elapsed wall-time drain, equality of scored and
committed demand, and propagation of the scheduler's reference width through the
cluster. This is an implementation checkpoint only; no experiment result or
performance claim has yet been produced for the occupancy-capacity mode.

## Bounded occupancy-capacity diagnostic (2026-07-30)

The pre-registered development diagnostic ran exactly 48 simulations: conditions
`1p3m:shared:medium`, `1p3m:shared:near_high`, and `1p3m:rag:near_high`; seeds 42
and 123; and eight arms comprising occupancy capacity at `V` in `{0.25,1,4}`, the
fixed-scale controller at its frozen `V=4`, the no-capacity ablation, joint
least-TTFT, Kairos, and the previously selected conditional-static plan. All runs
passed the hard request-conservation and completion gates. These remain development
seeds and are not evidence from a fresh evaluation.

One global value, `V=4`, dominated the smaller occupancy values in every tested
condition. Its results were:

| condition | occupancy goodput / remote | fixed-scale | no capacity | least TTFT | Kairos | static |
|---|---:|---:|---:|---:|---:|---:|
| shared medium | 0.195 / 0.527 | 0.189 / 0.564 | 0.877 / 0.874 | 0.065 / 0.000 | 0.836 / 1.000 | 0.889 / 1.000 |
| shared near-high | 0.081 / 0.430 | 0.062 / 0.500 | 0.086 / 0.842 | 0.022 / 0.000 | 0.158 / 1.000 | 0.159 / 1.000 |
| RAG near-high | 0.883 / 0.711 | 0.591 / 0.271 | 0.496 / 0.680 | 0.847 / 0.256 | 0.249 / 0.901 | 0.591 / 0.500 |

The new candidate trace records the raw queue and demand behind every capacity
cross term. Across the selected runs it checked 75,088 inter-decision queue
transitions and 112,668 candidate cross terms. The maximum queue-update error was
zero, the maximum floating-point cross-term error was below `9e-16`, every initial
queue was zero, and every chosen snapshot had zero score regret. The negative
result is therefore not an accounting or argmin implementation defect.

The occupancy account reduces shared-medium capacity-versus-net-good conflict only
from 92.6% to 88.9%. More importantly, it moves remote routing away from the
all-remote static plan: 56.4% to 52.7% at medium load and 50.0% to 43.0% near high.
Goodput gains over the fixed-scale controller are only 0.006 and 0.019. The bounded
decision gate is consequently **STOP**, despite the large RAG improvement; no fresh
seeds or broader campaign should follow from this result.

The trace explains the remaining conflict. At shared medium, the chosen occupancy
bookings load the single prefill instance at about 0.96 occupancy-seconds per wall
second while each of the three decoders is near 0.73. A remote candidate consumes
about 41.6 ms of total per-request occupancy on average versus 25.0 ms for a local
candidate, so the capacity term continues to favor local placement even though the
static all-remote plan has much higher SLO goodput. On the observed `V=4` snapshots,
the median welfare weight needed to overturn a conflicting shared-medium capacity
comparison is 3.81, the 90th percentile is 5.35, and an offline rescore at `V=8`
would point remote on 86.6% of snapshots. That rescore is not a policy evaluation:
changing `V` changes later queues, cache state, and residents. It identifies a scale
boundary for a possible next *small* diagnostic, not permission to extend the
campaign.

## Final occupancy-capacity decision at V=8 (2026-07-30)

The preregistered final diagnostic ran exactly the six authorized development runs:
the same three focal conditions at seeds 42 and 123, with occupancy capacity fixed
to `V=8`. All requests reached valid terminal states. Queue transitions were exact,
all capacity cross terms agreed within floating-point tolerance, and every chosen
candidate was the traced score argmin.

| condition | goodput | remote fraction | gate |
|---|---:|---:|---|
| `1p3m:shared:medium` | 0.277 | 0.571 | fail: goodput < 0.53 and remote < 0.75 |
| `1p3m:shared:near_high` | 0.047 | 0.507 | fail: below the `V=4` goodput of 0.081 |
| `1p3m:rag:near_high` | 0.880 | 0.779 | pass: goodput >= 0.83 |

The conservation gate passed, but the three performance requirements did not all
pass. In particular, replaying `V=8` changed the induced queues, caches, and resident
sets enough that the shared-medium remote fraction reached only 57.1%, not the 86.6%
suggested by the offline rescore of `V=4` snapshots. Capacity-versus-welfare conflict
also remained 89.0% on shared medium.

The preregistered decision is therefore **STOP**. Do not expand this controller to
fresh seeds and do not continue tuning its capacity scale. Subsequent paper work
should use the reduced causal-externality/no-capacity policy and must not claim a
Lyapunov or `O(V), O(1/V)` capacity trade-off for it. The complete decision artifacts
are in `out/slo_externality_occupancy_v8_decision/`.

## Public-workload and realistic-heterogeneity closeout (2026-07-30)

After the stop decision, the user authorized one final bounded robustness check to
determine whether the conclusion was an artifact of the original workload set or
homogeneous hardware. This was not another tuning round: occupancy remained fixed at
`V=8`, the no-capacity controller used the same causal SLO externality, and Kairos
remained fixed at `beta=0.5`. The study added three request-shape distributions from
the public Kubernetes inference-perf catalog and compared homogeneous H100 1P2D with
a realistic H100-prefill, H100/A100-decode 1P2D fleet.

At the user's direction, the translations treat requests independently and omit
turn count, think time, and context accumulation. They therefore test public token
and SLO regimes rather than reproducing the full conversational benchmarks. The
executable upstream YAMLs were used as the source of truth. This matters because
deep research's YAML specifies lognormal dynamic prompt length and normal per-turn
output where `config.json` labels aggregate input exponential and output bimodal;
reasoning's YAML likewise specifies lognormal rather than exponential output. The
deep-research input maximum was reduced from 150K to 121K so its 2K prefix and 4K
maximum output fit the evaluated 128K model context.

The upstream YAMLs also use staged closed-loop concurrency and name a Gemma-3-1B
server. The simulator retained neither: it used capacity-normalized Poisson arrivals
and the calibrated Llama-3.3-70B H100/A100 model. The result therefore concerns the
catalog's public request shapes under the paper's evaluated system, not a reproduction
of the complete upstream benchmark.

Twenty-seven saturated fixed-plan runs independently normalized each workload and
fleet. Two deep-research points that concentrated all decodes on one H100 shed 13
and 19 requests because of transient KV concentration; they remained visible in the
grid but were ineligible to define capacity. The selected zero-drop capacity points
set the evaluation rates to 85% of measured capacity. All 72 policy runs then
completed with zero drops, timeouts, length caps, or unfinished requests.

| fleet/workload | occupancy V=8 | no capacity | best public baseline | static joint yardstick |
|---|---:|---:|---:|---:|
| H100 / interactive | 0.980 | 0.980 | 0.980 | 0.945 |
| H100 / reasoning | 1.000 | 1.000 | 1.000 | 1.000 |
| H100 / deep research | 0.853 | 0.872 | 0.872 | 0.869 |
| H100+A100 / interactive | 0.940 | 0.938 | 0.900 | 0.863 |
| H100+A100 / reasoning | 0.975 | 0.978 | 0.975 | 0.972 |
| H100+A100 / deep research | 0.775 | 0.803 | 0.806 | 0.812 |

The registered occupancy-minus-no-capacity deltas were `0.000`, `0.000`, and
`-0.019` on homogeneous H100, then `+0.002`, `-0.003`, and `-0.028` on the
realistic heterogeneous fleet. Heterogeneity shifted these deltas by no more than
0.009 on any workload, so it did not reveal a hidden hardware-specific advantage.
Occupancy remained within 0.05 of the best public baseline everywhere, but it
improved over no capacity by the required 0.02 on zero of three workloads on both
fleets. Both preregistered revival gates therefore failed.

The decision traces make the workload dependence intelligible. Interactive chat's
moderate prompt and tight one-second TTFT target caused the net-good term to favor
remote placement on about 91% of requests; occupancy still routed 91% remotely, so
its mostly local-favoring capacity comparisons were too small to change the result.
Reasoning has small prompts and very long output, so the score favored local service
on 90--99% of requests, as expected for a decode-heavy workload. Deep research has
very long prompts, and the net-good term favored remote placement on 86% of H100
requests and 93% of heterogeneous requests. Here the occupancy term favored local
on 37% and 36%, reduced the remote fraction from the no-capacity controller's
81%/92% to 64%/71%, and lowered goodput. Its mean absolute capacity gap was also
larger than the weighted net-good gap on both fleets. Thus the capacity term is not
merely inactive; it over-corrects in the prefill-heavy regime where remote prefill
is structurally useful.

Across the evaluation, every occupancy queue transition was exact, every candidate
capacity cross term matched its raw queue-times-demand reconstruction, and every
chosen action was the traced minimum-score candidate. The final decision remains
**CLOSE**: workload realism and realistic H100/A100 heterogeneity do not rescue the
occupancy-capacity controller. The defensible surviving direction is the myopic
causal-SLO-externality/no-capacity policy, without a Lyapunov or end-to-end policy
regret claim. The protocol, translated workloads, runner, and complete results are
in `PUBLIC-WORKLOAD-HETEROGENEITY-CLOSEOUT-PROTOCOL.md`,
`workloads/public-closeout/`, `run_public_workload_heterogeneity_closeout.py`, and
`out/public_workload_heterogeneity_closeout/`.

## Held-out confirmation of the no-capacity policy (2026-07-30)

After the capacity-controller closeout, a separate protocol froze the surviving
joint causal-SLO-externality/no-capacity policy and compared it with joint
least-TTFT, Kairos at `beta=0.5`, and the development-selected static joint
yardstick. The four seeds `262147`, `524309`, `1048583`, and `2097169` had not been
used on these public-workload conditions. The resulting 96 runs cover the same
three public request shapes on homogeneous H100 and realistic H100/A100 fleets.
There was no tuning, every run reached a valid terminal state, and no run dropped a
request.

| fleet/workload | no capacity | least TTFT | Kairos | static yardstick |
|---|---:|---:|---:|---:|
| H100 / interactive | 0.979 | 0.932 | 0.976 | 0.901 |
| H100 / reasoning | 1.000 | 1.000 | 1.000 | 1.000 |
| H100 / deep research | 0.859 | 0.839 | 0.852 | 0.847 |
| H100+A100 / interactive | 0.915 | 0.867 | 0.662 | 0.728 |
| H100+A100 / reasoning | 0.998 | 1.000 | 0.981 | 0.989 |
| H100+A100 / deep research | 0.798 | 0.786 | 0.788 | 0.797 |

Among deployable policies, maximum regret to the best mean in each of the six cells
was `0.0016` for no capacity, `0.0483` for least-TTFT, and `0.2533` for Kairos.
Across all 24 paired cell-seed observations, no capacity exceeded least-TTFT by
`0.0212` goodput (95% interval `[0.0119, 0.0305]`) and Kairos by `0.0488`
(`[0.0106, 0.0870]`). The policy was not a universal per-cell winner: it trailed
least-TTFT by `0.0016` on heterogeneous reasoning, with an interval spanning zero.
The chosen-candidate trace check was exact.

This held-out result supports the bounded empirical claim that myopic causal SLO
externality without capacity prices is robust across these public request shapes and
the evaluated realistic hardware asymmetry. It does not restore a capacity theorem,
Lyapunov trade-off, universal-win claim, or end-to-end policy-regret result. The
frozen protocol and artifacts are in
`PUBLIC-NO-CAPACITY-FRESH-CONFIRMATION-PROTOCOL.md` and
`out/public_workload_heterogeneity_closeout/`.

## Post-confirmation routing-value revision (2026-07-30)

The held-out confirmation above used the then-declared smooth
`TTFT x mean-ITL x E2E` routing value. Before the next experiment campaign, the
routing value was narrowed to the smooth `TTFT x E2E` product used by the
ground-up formulation. Mean ITL remains unchanged in reported composite goodput
and in policies, such as Kairos, whose contract uses it as a safety constraint.

This was a policy change, not a relabeling. The 96-run confirmation remains valid
only for the earlier three-factor routing value and is not reinterpreted as a
result for the revised policy. The warning that the two-factor policy lacked
confirmation is superseded by the separately calibrated and held-out campaigns
below.

## Load and calibrated-static benchmark for the final policy (2026-07-30)

The final smooth `TTFT x E2E` causal-externality policy was evaluated at 60%, 80%,
and 95% of a separately measured fixed-plan capacity envelope. The benchmark
covered interactive, reasoning, and deep-research request shapes on homogeneous
H100 and realistic H100/A100 fleets. At every fleet/workload/load condition, the
development phase selected a goodput-tuned static plan independently of the plan
that maximized capacity. The held-out comparison then froze both plans and compared
the final policy with joint least-TTFT, Kairos at `beta=0.5`, the goodput-tuned
static plan, and the capacity-selected static plan.

The 360 held-out runs span 18 conditions and four seeds per policy. There were no
hard-invalid runs, drops, timeouts, or length caps, and all 72 focal traces chose
the exact score argmin. Among deployable policies, the final policy had worst
regret `0.0125` to the best deployable mean in each condition, compared with
`0.0525` for joint least-TTFT and `0.2292` for Kairos. Its equal-condition mean
goodput was `0.9327`, versus `0.9177` and `0.9018` respectively.

These results confirm the revised two-factor policy across load. The two frozen
static plans remain descriptive condition-tuned yardsticks: neither is a
request-selection oracle or an end-to-end upper bound. The full capacity choices,
calibrated plans, per-condition results, and validity checks are in
`out/public_load_static_benchmark_v1/PUBLIC-LOAD-STATIC-BENCHMARK.md`.

## Externality and joint-decision ablations (2026-07-30)

A fresh-seed ablation held the workloads, fleets, loads, candidate actions, and
estimators fixed while comparing four arms: full joint causal externality,
joint own-request value only, joint resident-externality only, and full
decode-first routing. Across 288 runs in the same 18 public conditions, all runs
were valid and all constrained choices, score identities, candidate counts, and
disabled score components were exact.

| arm | equal-condition mean goodput | worst-condition goodput | mean remote fraction |
|---|---:|---:|---:|
| full joint | 0.9318 | 0.7766 | 0.4662 |
| joint own-only | 0.9174 | 0.7703 | 0.3048 |
| joint resident-only | 0.9253 | 0.7781 | 0.8520 |
| decode-first full | 0.8895 | 0.6433 | 0.5173 |

Adding the resident externality to the own-request term improved paired goodput by
`0.0145` with a 95% interval of `[0.0099, 0.0191]`. Adding arriving-request value
to the resident-only arm improved it by `0.0066 [0.0026, 0.0105]`. Joint action
selection improved over decode-first by `0.0423 [0.0257, 0.0589]`. Thus both score
components contribute, and the larger structural effect comes from selecting the
decoder and local/remote prefill action jointly. Details are in
`out/public_externality_decomposition_ablation_v1/PUBLIC-EXTERNALITY-DECOMPOSITION-ABLATION.md`.

## Joint request-level counterfactual diagnostic (2026-07-30)

The final policy was also checked with exact one-request forced-action replays.
For each sampled decision, the captured plan for every other request was held fixed
while all alternative joint placements were replayed. This is a local hindsight
diagnostic, not a global oracle and not a deployable comparator.

Across 144 sampled decisions from the 18 public conditions, requiring 432
alternative-action runs, the online choice agreed with a best forced action 94.4%
of the time. The eight disagreements were all decoder-selection errors: there were
no cases in which a chosen local decision should have been remote or a chosen
remote decision should have been local. Mean goodput regret per sampled decision
was `0.00029`, or 0.056 equivalent good requests. All replay gates matched, all
online choices were exact score argmins, and no deviation run dropped, timed out,
or length-capped a request. Full results are in
`out/public_joint_counterfactual_v1/PUBLIC-JOINT-COUNTERFACTUAL.md`.

## Mixed-workload and burst robustness (2026-07-30)

The final policy next faced four nonstationary profiles on both fleets:
sequential workload shifts, concurrent capacity-balanced Poisson traffic,
overdispersed gamma traffic with arrival `CV=3`, and a short `1.60x` load spike.
Condition-tuned static plans were selected in 96 development runs and frozen before
128 held-out confirmation runs. The confirmation had zero invalid runs, drops,
timeouts, or length caps, and all 32 focal traces chose exact score argmins.

Under the registered request-weighted goodput metric, the final policy's worst
regret was `0.0026`, versus `0.0769` for joint least-TTFT and `0.1940` for Kairos.
Its paired mean advantage was `0.0165 [0.0053, 0.0277]` over least-TTFT,
`0.0638 [0.0425, 0.0851]` over Kairos, and `0.0791 [0.0545, 0.1038]` over the
tuned static yardstick.

The result has an important weighting caveat. Capacity-balanced mixtures contain
many more interactive requests, so request-weighted goodput is dominated by that
class. In the preregistered descriptive check that gives interactive, reasoning,
and deep research equal weight within each condition, the ranking reverses:
least-TTFT has worst regret `0.0155`, Kairos `0.0491`, and the final policy
`0.0509`. The request-weighted metric remains the registered primary outcome, but
the final paper must disclose this fairness trade-off prominently and must not
claim uniform class-level dominance. The complete report is in
`out/public_mixed_burst_benchmark_v1/PUBLIC-MIXED-BURST-BENCHMARK.md`.

## Final-policy topology and provisioning sweep (2026-07-30)

The last completed robustness study provisioned four homogeneous H100 instances as
`1P3D`, `2P2D`, and `3P1D`. Each topology was independently capacity-normalized;
the held-out evaluation ran at 90% of its workload-specific capacity. The study
used 135 capacity runs and 90 development runs to freeze condition-tuned static
fractions, followed by 144 confirmation runs across nine topology/workload
conditions. No confirmation run was invalid or had drops, timeouts, or length
caps, and all 36 focal traces chose exact score argmins.

| deployable policy | worst regret across nine conditions | equal-condition mean goodput |
|---|---:|---:|
| causal externality | 0.0108 | 0.9120 |
| Kairos | 0.0172 | 0.9120 |
| joint least-TTFT | 0.0592 | 0.8991 |

The final policy exceeded least-TTFT by a paired mean of
`0.0129 [0.0070, 0.0188]`. Its differences from Kairos
(`-0.0000 [-0.0035, 0.0034]`) and the tuned static yardstick
(`-0.0003 [-0.0030, 0.0024]`) were indistinguishable from zero. It therefore
remains minimax-robust across the evaluated provisionings, but the data support
parity rather than a universal-win claim against Kairos or tuned static routing.
The complete capacity, calibration, and confirmation results are in
`out/public_final_topology_sweep_v1/PUBLIC-FINAL-TOPOLOGY-SWEEP.md`.

## Current implementation-audit conclusion (2026-07-30)

The final implemented policy is now the smooth `TTFT x E2E` arriving-request value
minus smooth resident causal externality, with joint decoder and local/remote
prefill selection and no capacity queue or capacity penalty. ITL is deliberately
absent from this routing score; it remains part of the hard reported composite
goodput and of comparator policies whose contracts include it.

The revised policy is no longer awaiting evaluation. It has load-normalized,
held-out support across public request shapes, realistic H100/A100 asymmetry,
nonstationary mixtures, bursts, and three homogeneous topology splits, plus direct
component, decomposition, and request-level diagnostics. The evidence supports a
bounded empirical robustness claim. It does not support a Lyapunov theorem,
`O(V), O(1/V)` trade-off, global-oracle gap, universal per-condition victory, or
uniform per-class superiority.

## Workload-tuned llm-d prefix-threshold extension (2026-07-31)

The primary load benchmark initially omitted llm-d's decomposed prefix-threshold
policy. A frozen append-only protocol added it using the exact implemented
decision order: select the decoder with
`precise-prefix-cache:2,queue-depth:1`, compare the decoder's uncached prompt
tokens with a threshold, and, only when disaggregating, select a prefill instance
with the same scorer profile. The decoder is not reconsidered. Cache scoring and
the threshold query share a 50 ms delayed cache snapshot.

Thirteen thresholds from 0 through 131,072 were evaluated on seeds 42 and 123.
One threshold was selected per workload by equal-run mean goodput across both
fleets and all three loads. All 468 development runs were valid. The frozen
thresholds were 1,024 for interactive and 16 for reasoning and deep research.
The latter two selections follow the preregistered tie break: all smaller tested
thresholds tied on reasoning, while thresholds 0 through 8,192 tied on deep
research.

The extension then ran only the new policy on the original four held-out traces,
for 72 matched runs. This reuse enables paired comparison with the stored outputs
but is not a new fresh-seed confirmation. Every added run had exact terminal
accounting with zero drops, timeouts, or length caps.

| deployable policy | worst shortfall | equal-condition mean goodput |
|---|---:|---:|
| causal externality | 0.0125 | 0.9327 |
| joint least-TTFT | 0.0525 | 0.9177 |
| workload-tuned llm-d threshold | 0.1892 | 0.8927 |
| Kairos | 0.2292 | 0.9018 |

Causal externality exceeded the tuned threshold policy by paired mean goodput
`0.0399 [0.0215, 0.0583]`. The threshold policy routed 98.9% of interactive
requests and every observed reasoning and deep-research request remotely. Its
worst result was heterogeneous interactive traffic at 80% capacity: goodput
`0.728` versus `0.918` for causal externality. This exposes a limitation of the
decomposed pool scorer rather than an untuned threshold: it contains prefix and
queue state but no accelerator-speed term, and the threshold cannot revisit its
decoder choice. The protocol, runner, calibration grid, and results are in
`PUBLIC-LLMD-PREFIX-THRESHOLD-EXTENSION-PROTOCOL.md`,
`run_public_llmd_prefix_threshold_extension.py`, and
`out/public_llmd_prefix_threshold_extension_v1/`.

## Unified six-policy confirmation (2026-07-31)

The retrospective extension above is now archival. After freezing all capacities,
rates, static plans, llm-d thresholds, and policy settings, one unified confirmation
ran every policy on the same four previously unused seeds. The Kairos arm used the
paper-oriented implementation: `alpha=1.3`, `beta=1.0`, the request TTFT gate,
strictest-resident TBT protection, exact queued prefill tokens, and executable
discrete chunk search. It did not use the admission-aware and transfer-aware
extensions in the historical `kairos` compatibility mode.

The unified campaign contains 432 runs across 18 workload/fleet/load cells. All
runs completed with zero hard-invalid outcomes, drops, timeouts, or length caps.
All 72 focal decision traces chose the exact recorded score argmin.

| deployable policy | worst shortfall | equal-condition mean goodput |
|---|---:|---:|
| causal externality | 0.0058 | 0.9331 |
| joint least-TTFT | 0.0425 | 0.9201 |
| Kairos, paper mode | 0.0625 | 0.9180 |
| workload-tuned llm-d threshold | 0.2125 | 0.8966 |

Causal externality exceeded joint least-TTFT by paired mean goodput
`0.0131 [0.0093, 0.0168]`, paper-mode Kairos by
`0.0151 [0.0100, 0.0202]`, and the workload-tuned llm-d threshold by
`0.0365 [0.0224, 0.0506]`. Its difference from the condition-tuned static plan
was `0.0007 [-0.0015, 0.0029]`, which supports aggregate parity rather than
dominance over that condition-informed yardstick.

This block supersedes the earlier 360-run primary benchmark and retrospective
72-run llm-d addition for the paper's headline comparison. The complete report is
`out/public_llmd_prefix_threshold_extension_v1/PUBLIC-LLMD-PREFIX-THRESHOLD-UNIFIED-CONFIRMATION.md`.

## Paper-mode Kairos secondary extension (2026-07-31)

The mixed/bursty and topology studies originally retained the historical
admission-aware Kairos adaptation. A frozen matched extension replaced only
those rows with the same paper-oriented Kairos mode used in the unified primary
benchmark. Workloads, rates, SLOs, seeds, request counts, static plans, and all
other policy outputs remained fixed.

All 68 new runs passed exact terminal accounting with zero drops, timeouts, or
length caps: 32 runs across eight mixed/bursty cells and 36 runs across nine
topology cells.

For mixed and bursty traffic, the request-weighted worst shortfalls are `0.0104`
for causal externality, `0.0769` for joint least-TTFT, and `0.1124` for
paper-mode Kairos. Causal externality exceeds paper-mode Kairos by paired mean
goodput `0.0225 [0.0088, 0.0363]`. Under equal class weighting, paper-mode
Kairos ranks first with worst shortfall `0.0238`, joint least-TTFT reaches
`0.0295`, and causal externality remains third at `0.0509`.

Across the topology sweep, worst shortfall is `0.0016` for causal externality,
`0.0406` for paper-mode Kairos, and `0.0483` for joint least-TTFT. Causal
externality exceeds paper-mode Kairos by `0.0130 [0.0077, 0.0183]` on average
and remains statistically tied with the condition-tuned static plan.

These corrected rows supersede the adapted-Kairos comparisons in the two
earlier audit sections. The archived original artifacts remain intact. Updated
matched reports are
`out/public_mixed_burst_benchmark_v1/PUBLIC-MIXED-BURST-PAPER-KAIROS.md` and
`out/public_final_topology_sweep_v1/PUBLIC-FINAL-TOPOLOGY-PAPER-KAIROS.md`.

### Topology Kairos reporting correction (2026-07-31)

The topology Kairos comparison above is withdrawn. The implementation selected
the first prefill snapshot and forced execution there through
`PrefillPodHint`. This is immaterial in the single-prefill primary and
mixed/bursty experiments, but it makes the `2P2D` and `3P1D` results an
arbitrary first-instance policy. More fundamentally, the published Kairos
policy defines one prefill path and does not specify selection among multiple
prefill instances. The paper therefore reports no Kairos aggregate in the
topology table. Kairos remains a comparator only in single-prefill studies.
