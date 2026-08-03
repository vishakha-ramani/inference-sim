# Public load/static-plan benchmark protocol

## Frozen question

This campaign evaluates the revised joint causal-SLO-externality policy after
mean ITL was removed from its routing value. The policy now maximizes projected
smooth `TTFT x E2E` value for the arrival minus smooth causal value removed from
residents. Reported goodput remains the hard conjunction of TTFT, mean ITL, and
E2E.

The experiment asks whether one deployable policy remains competitive across
public request shapes, hardware, and load after separating three different
objects:

1. the fixed joint plan that maximizes stable completion capacity;
2. the fixed joint plan that maximizes calibration goodput at a known load;
3. a deployable online policy that does not know the workload/load condition.

The first two are static planning yardsticks, not request-selection or global
oracles. The best-tested-policy reference used for minimax ranking is likewise
not an oracle.

## Conditions

The six base conditions are the existing independent-request translations of
the public inference-perf `interactive`, `reasoning`, and `deep_research`
request shapes on:

- homogeneous H100 `1P2D`; and
- realistic heterogeneous H100-prefill/H100+A100-decode `1P2D`.

Workload distributions, SLO targets, model, coefficients, transfer model,
scheduler limits, and request counts remain those declared
in `PUBLIC-WORKLOAD-HETEROGENEITY-CLOSEOUT-PROTOCOL.md`. The simulator request
timeout is disabled for this campaign so it cannot silently tighten the
reasoning workload's declared 802-second E2E target. Timeout, drop, length-cap,
and terminal-state accounting are still reported and gated.

Development seeds are `42`, `123`, and `2024` for capacity measurement and
`42`, `123` for static-goodput calibration. The confirmation seeds are
`2000000011`, `2000000033`, `2000000063`, and `2000000087`. They have not appeared in the
existing campaign sources or results and must not be inspected before all
capacity/rate/static selections are written.

The causal-externality and joint least-TTFT arms use the validated scheduler
rollout from `main.tex`: local TTFT is final local-prefill-step completion plus
token post-processing; remote TTFT is the maximum of remote-prefill-plus-transfer
and decode admission, followed by the first decode iteration and token
post-processing. Their scheduler snapshots are live (`0` refresh interval) and
the scheduler and preemption policies are FCFS, matching the validated estimator.
Kairos retains its published estimator.

## Stage A: fixed-plan capacity envelope

A fixed joint plan is parameterized by:

- `phi`: fraction of requests prefilling remotely; and
- `psi`: fraction of decodes sent to the second decoder, which is the A100 on
  the heterogeneous fleet.

The coarse grid is:

- `phi in {0, 0.2, 0.4, 0.6, 0.8, 1}`;
- homogeneous `psi=0.5`;
- heterogeneous `psi in {0, 0.2, 0.4, 0.6, 0.8, 1}`.

Every capacity point uses the frozen saturating offered rate, loose SLOs, and
all three capacity seeds. A plan is capacity-eligible only if every seed is
hard-valid and has zero drops. Its capacity estimate is mean central completion
RPS across seeds.

After the coarse sweep, each workload/fleet refines the coarse winner at the
Cartesian product of the winner and its clipped `+/-0.1` neighbors in `phi` and
`psi`; homogeneous `psi` remains 0.5. Previously evaluated points are reused.
The final capacity-selected plan maximizes mean completion RPS across the union
of coarse and refinement points, breaking exact ties by lower seed standard
deviation, then lower `phi`, then lower `psi`.

The fleet ceiling is

```text
C_w = max_(phi,psi) mean_seed C_w(phi,psi).
```

This is a measured envelope over the tested deterministic fixed-plan family,
not a proof of globally optimal scheduling capacity.

## Stage B: condition-tuned static-goodput plans

For every workload/fleet, evaluation rates are frozen to:

```text
lambda_low  = 0.60 C_w
lambda_mid  = 0.80 C_w
lambda_high = 0.95 C_w.
```

At each rate, the same coarse joint-plan grid is evaluated on static-calibration
seeds `42` and `123` under the actual SLO targets. A plan is eligible only if
both runs are hard-valid and zero-drop. Its score is mean composite goodput over
the two seeds.

Each condition/load then refines the coarse winner using clipped `+/-0.1`
neighbors exactly as in Stage A. The best calibrated static plan is selected
from the union of coarse and refinement points by higher mean goodput, lower
seed standard deviation, lower `phi`, then lower `psi`.

This selection is called the **condition-tuned static plan**. It knows the
workload, fleet, and offered rate, but its deterministic request assignment sees
no live queue, cache, or resident state and does not optimize request identities.

## Stage C: held-out policy comparison

After `capacity_selection.json` and `static_goodput_selection.json` are frozen,
run every workload/fleet/load/seed combination for:

1. joint causal SLO externality without capacity prices;
2. joint least projected TTFT;
3. paper Kairos with `alpha=1.3`, request-TTFT gating, resident-ITL protection,
   and discrete chunk candidates;
4. the workload-tuned llm-d prefix-threshold policy, using
   `precise-prefix-cache:2,queue-depth:1` for both pools and frozen thresholds
   `1024` (interactive), `16` (reasoning), and `16` (deep research);
5. the condition-tuned static plan from Stage B; and
6. the capacity-selected static plan from Stage A, as a descriptive control.

The first four policies are deployable and enter the minimax ranking. The two
static plans are excluded from that ranking. With six arms, Stage C contains
`3 x 2 x 3 x 4 x 6 = 432` runs.

## Registered outputs and validity

For every run report injected, completed, dropped, timed out, length capped,
unfinished, goodput, per-dimension attainment, remote fraction, and decoder
share. For every condition/load/policy report all four held-out seed values and
their mean.

Rank the four deployable policies by maximum regret to the best deployable
mean in each of the 18 workload/fleet/load cells, breaking ties by higher
equal-cell-weighted mean goodput. Report paired held-out-seed deltas from the
causal-externality policy to every comparator with 95% intervals. Report the
static-plan gap separately and never call it policy regret or an oracle gap.

All runs must satisfy:

```text
completed + dropped = injected
still_queued = 0
still_running = 0
timed_out = 0
length_capped = 0.
```

Chosen focal-policy actions must equal the minimum traced candidate score.
Any incomplete grid, invalid run, selection made from confirmation outcomes, or
candidate-trace mismatch invalidates the corresponding stage.

## Scope

This remains a discrete-event-simulator study of independent public token/SLO
shapes. It does not reproduce upstream multi-turn or staged-concurrency
benchmarks. It can support a held-out empirical comparison and a static-plan
gap, but not a capacity-feasibility theorem, Lyapunov trade-off, global-oracle
gap, or end-to-end policy-regret claim.
