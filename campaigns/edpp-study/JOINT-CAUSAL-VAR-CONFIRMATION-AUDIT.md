# Audit of the capacity-normalized joint-routing confirmation

## Scope and provenance

This audit concerns `out/joint_causal_var_confirmation_v2`. The earlier `v1`
directory is superseded: its capacity selector could choose the largest observed
completion rate even when that calibration arm dropped requests. The selector was
then changed to choose only among zero-drop calibration arms, and the campaign was
rerun as `v2`. The corrected run contains 144 paired policy runs over four frozen
seeds, three workloads, two loads, and the 1P3D, 2P2D, and 3P1D topologies.

The experiment compares two policies that use the same causal marginal resident-loss
estimator. The decomposed policy accepts the ordinary router's decode destination and
then chooses local prefill or one of the remote prefill instances. The joint policy
minimizes the same resident-loss score over all D(P+1) decode/prefill actions. This is
therefore an ablation of joint action selection, not a comparison against Kairos,
least predicted TTFT, a production prefix threshold, or a static joint plan.

All 144 runs conserved requests and drained their queues. Three runs, all RAG at 95%
load with seed 33554393, nevertheless dropped nine requests in total: the 1P3D
decomposed run dropped three, while the 2P2D decomposed and joint runs dropped one and
five. Thus 141 runs satisfy the campaign's stricter zero-drop validity condition.

## End-to-end result

The table reports the paired change in the fraction of requests satisfying the
composite SLO. Values are percentage points, and the interval is the four-seed paired
95% interval emitted by the campaign.

| Topology | Workload | Load | Joint minus decomposed | 95% interval | W/T/L |
|---|---|---:|---:|---:|---:|
| 1P3D | Synthetic | 80% | -0.22 | [-0.91, +0.48] | 0/3/1 |
| 1P3D | Synthetic | 95% | -0.06 | [-0.26, +0.14] | 0/3/1 |
| 1P3D | RAG | 80% | +5.92 | [-2.21, +14.05] | 4/0/0 |
| 1P3D | RAG | 95% | +11.11 | [+8.26, +13.95] | 4/0/0 |
| 1P3D | Shared-prefix | 80% | -2.84 | [-50.89, +45.20] | 3/0/1 |
| 1P3D | Shared-prefix | 95% | +12.80 | [+10.41, +15.19] | 4/0/0 |
| 2P2D | Synthetic | 80% | 0.00 | [0.00, 0.00] | 0/4/0 |
| 2P2D | Synthetic | 95% | 0.00 | [0.00, 0.00] | 0/4/0 |
| 2P2D | RAG | 80% | +1.05 | [-8.54, +10.64] | 2/0/2 |
| 2P2D | RAG | 95% | +3.52 | [-6.51, +13.55] | 3/0/1 |
| 2P2D | Shared-prefix | 80% | -5.29 | [-49.47, +38.89] | 3/0/1 |
| 2P2D | Shared-prefix | 95% | +10.64 | [+8.70, +12.58] | 4/0/0 |
| 3P1D | All workloads | Both | 0.00 | [0.00, 0.00] | 0/24/0 |

Three near-high-load cells have a resolved positive effect: 1P3D RAG and both
multi-decode shared-prefix topologies. The 3P1D equality is a structural control,
because one decode instance makes joint and decomposed decode selection identical.
The synthetic result is also nearly an identity: 98.7% of synthetic decisions have
zero resident loss for every feasible action, so the two policies follow the same tie
path. The two RAG cells on 2P2D remain unresolved, while the shared-prefix medium-load
means conceal a severe seed-specific failure described below.

## What the decisions show

On 1P3D RAG, the joint policy overrides the ordinary router's decode destination on
62.0% of medium-load decisions and 62.1% of near-high decisions. The decomposed
router incurs positive score-space regret on 61.9% and 63.7% of those requests,
respectively. Remote prefill has lower predicted resident loss than the best local
action on 51.0% and 51.4% of the joint policy's snapshots, while local prefill is
lower on 44.6% and 45.3%. This is consistent with RAG's long, mostly uncached prompts:
both prefill placement and decoder selection can change which residents overlap the
new request.

The 2P2D RAG topology exposes the same decision mechanism but not the same stable
end-to-end benefit. Joint selection overrides the ordinary decoder on 47.1% of
medium-load decisions and 49.9% of near-high decisions. Remote prefill has the lower
resident-loss score on 87.1% and 86.0% of snapshots, yet the paired goodput intervals
cross zero. A lower predicted externality is therefore not, by itself, evidence that
the arriving request or the system as a whole receives greater SLO value.

For shared-prefix traffic, every selected action's measured loss comes from decode
residents; the collocated-prefill and prefill-pool components are zero in these
traces. Across all three topologies, remote prefill has strictly lower resident loss
on 61.9% of joint-policy snapshots and ties the best local action on the remaining
38.1%; a local action is never strictly lower. This is expected for highly cached,
short-prefill requests when there are no prefill residents to protect: remote prefill
mainly postpones the request's arrival at decode. It also shows why this campaign
does not establish a general prefill-allocation result.

## Saturation failure of a resident-loss-only objective

At medium shared-prefix load, seed 67108859 collapses under joint selection on both
multi-decode topologies. Goodput falls from 0.784 to 0.304 on 1P3D and from 0.695 to
0.226 on 2P2D, even though the other three paired seeds improve by 8.95--14.53 and
6.90--9.78 percentage points. In the failing runs, the joint policy initially spreads
decode work, but after roughly the first thousand requests it concentrates most new
requests on one decoder. By the second half of each run, 88--94% of decisions are
local and the selected action's resident-loss score is zero for essentially every
request, even though alternative actions often have positive scores.

This behavior follows from the current objective rather than from joint enumeration
itself. The policy minimizes only the SLO value destroyed among resident requests. If
one decoder's residents are already outside the useful part of the SLO kernel, adding
another request can have zero measured marginal damage there. The policy then treats
that overloaded decoder as the cheapest destination, sacrifices the arriving
request's own SLO value, and has no queue or capacity price that pushes work back to
the other decoders. The failure is the concrete reason not to present causal resident
loss alone as the paper's policy.

## What “regret” means in this campaign

The reported router-induced regret is

\[
  \min_{a:\,d(a)=d_{\mathrm{router}}} \widehat L_{\mathrm{resident}}(a)
  - \min_{a\in\mathcal A_t} \widehat L_{\mathrm{resident}}(a).
\]

It measures the resident-loss score paid by fixing the decoder before making the P/D
decision. It is not conditional static-disaggregation regret, goodput regret against
another online policy, or an end-to-end regret theorem. The confirmation campaign did
not evaluate a condition-tuned static joint plan, and it contains no public-policy
baseline. Those comparisons cannot be inferred from this result.

## Consequence for the new policy and paper

The experiment supports joint action selection as an important design requirement,
but it does not support a paper whose contribution is “minimize causal VaR over joint
actions.” A principled next policy should choose a joint action by maximizing marginal
system SLO value, or equivalently by minimizing

\[
  \widehat L_{\mathrm{resident},t}(a)
  - \widehat G_{\mathrm{new},t}(a)
  + Q^D_{d(a),t}\,w^D_t(a)
  + \mathbf 1\{a\text{ is remote}\}Q^P_{p(a),t}\,w^P_t(a).
\]

Here the first term is the causal resident externality already implemented, the
second is the predicted SLO value earned by the arriving request, and the queue terms
are dual prices for explicit decode and prefill capacity constraints. These additions
are not empirical patches: together they form the one-step drift-plus-penalty or
primal-dual objective for marginal welfare under resource constraints. They also
address the observed sacrifice-node failure directly, because a destination with zero
remaining resident value is no longer free when it gives the new request little value
or accumulates scarce-resource work.

Before this formulation can carry a paper, a fresh campaign must compare ordinary
routing plus the same causal P/D selector, joint routing without the causal externality,
and the full marginal-welfare joint policy. The external comparisons should use public
baselines and a condition-tuned static joint plan; the latter is the correct offline
yardstick for conditional static-disaggregation regret. The present confirmation is
best retained as the motivating ablation and failure analysis for that development.
