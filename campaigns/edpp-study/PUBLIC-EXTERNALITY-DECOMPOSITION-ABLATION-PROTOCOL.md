# Public externality/decomposition ablation protocol

Status: frozen before any ablation confirmation run.

## Question

At the public benchmark's already frozen rates, which part of the final routing
score produces goodput, and does joint action selection improve on fixing the
decode instance first?

The routing value is the smooth TTFT x E2E value. ITL is not a routing-objective
term; reported goodput remains the hard TTFT/mean-ITL/E2E conjunction.

## Frozen conditions

- Workloads: interactive, reasoning, and deep research.
- Fleets: homogeneous H100 1P2D and realistic heterogeneous H100/A100 1P2D.
- Loads: 0.60C, 0.80C, and 0.95C.
- Capacities and exact offered rates: copied without modification from
  `out/public_load_static_benchmark_v1/capacity_selection.json`.
- Requests per run: 300 for interactive; 160 for reasoning and deep research.
- Generic request timeout: disabled, so it cannot supersede a declared SLO.
- V: 8. With no capacity term, V is a common positive multiplier and does not
  alter the argmin.

## Frozen arms

All arms use identical estimators, candidate physics, caches, SLOs, and inputs.

1. `joint_full`: joint argmin of resident externality minus arriving-request
   projected good; no capacity term.
2. `joint_own_only`: joint argmin with resident externality removed; no capacity
   term. This maximizes only the arriving request's projected good.
3. `joint_resident_only`: joint argmin with arriving-request projected good
   removed; no capacity term. This minimizes only resident externality.
4. `decode_first_full`: the existing queue-depth scorer fixes the decode
   instance, then the full score chooses local versus remote prefill; no capacity
   term.

The decomposition control is deliberately decode-first. A "split-first" control
is not included because minimizing the same score over placement and then decode
is mathematically the joint argmin; making it differ would require introducing
and tuning another first-stage heuristic.

## Confirmation seeds

The four confirmation seeds are new to this campaign and were selected before
running any ablation arm:

`67108879, 134217757, 268435459, 536870923`

Development seed 42 may be used only for implementation smoke tests.

## Size and analysis

The confirmation contains:

`3 workloads x 2 fleets x 3 loads x 4 policies x 4 seeds = 288 runs`.

Primary output is per-request goodput, summarized by equal-cell means and paired
seed differences from `joint_full`. Report separately:

- the effect of adding resident externality: full minus own-only;
- the effect of adding arriving-request value: full minus resident-only;
- the effect of joint selection: joint-full minus decode-first-full.

These are component and structural ablations, not oracle comparisons.

## Validity gates

- Exactly 288 confirmation runs and 18 condition cells.
- Terminal accounting exact in every run.
- Zero timeout and length-cap outcomes.
- Every candidate row satisfies
  `score = 8 * (resident_externality - own_good)` because capacity is disabled.
- Capacity terms are exactly zero in every arm.
- Externality is exactly zero in `joint_own_only`.
- Own good is exactly zero in `joint_resident_only`.
- Joint arms choose a global candidate argmin.
- The decode-first arm retains the scorer-selected decode and chooses the argmin
  within that decoder's local/remote candidates.

No arm, seed, or workload may be removed after inspection.
