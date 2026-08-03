# Public-workload and heterogeneity closeout protocol

## Question and scope

This is a final bounded development study, not a new policy-tuning campaign. It
asks whether the occupancy-capacity controller's negative decision changes on
three additional public inference-perf request shapes or on a realistic
heterogeneous decode pool. The implementation remains frozen at `V=8`, and the
study does not support an end-to-end regret or Lyapunov trade-off claim.

The study uses the executable `inference-perf.yaml` files from the upstream
`kubernetes-sigs/inference-perf` workload catalog, inspected on 2026-07-30. At
the user's direction, all requests are independent: turn count, think time, and
context accumulation are excluded. The local translations are under
`campaigns/edpp-study/workloads/public-closeout/`.

The upstream execution YAMLs use staged closed-loop concurrency and name
`google/gemma-3-1b-it` as the server model. This simulator study instead uses
open-loop Poisson arrivals whose rates are normalized to measured capacity and
the calibrated Llama-3.3-70B H100/A100 model. Consequently these are public
request-shape translations, not claims to reproduce the complete upstream
benchmark or its server.

| local workload | upstream request shape retained | declared deviation |
|---|---|---|
| interactive chat | 1,000-token prefix, lognormal 4K input, normal 300-token output, 50 prefix groups | conversation dynamics and staged concurrency omitted |
| reasoning | 250-token prefix, lognormal 1K input, lognormal 8K output, 10 prefix groups | staged concurrency omitted; executable YAML is used instead of config.json's exponential label |
| deep research | 2K prefix, lognormal 45K input, normal 300-token output, 5 prefix groups | conversation dynamics and staged concurrency omitted; input maximum capped from 150K to 121K for the 128K model context |

The deep-research executable YAML and `config.json` are not identical: the YAML
uses lognormal dynamic prompt length and normal per-turn output, while the JSON
calls aggregate input exponential and output bimodal. The executable YAML is the
source of truth here. Likewise, the reasoning YAML specifies lognormal output
where the JSON specifies exponential output.

## Fleet and load normalization

Every run uses a 1P2D topology. The homogeneous fleet has H100 prefill and two
H100 decode instances. The heterogeneous fleet uses the existing realistic
bundle: H100 prefill, one H100 decode instance, and one A100 decode instance
with `coeffs-llama70b-a100real-tp4.json`. The obsolete fabricated/crippled A100
is excluded.

Capacity is measured separately for every workload and fleet using saturated,
loose-SLO fixed joint plans. The remote fraction is swept over `{0, 0.5, 1}`.
Homogeneous decode routing uses an even split; heterogeneous routing tests both
quarantining the A100 and a `0.4` A100 share, approximately proportional to the
measured H100/A100 service rates. Evaluation load is fixed at 85% of the best
observed fixed-plan completion rate for that workload/fleet.
Fixed-plan points that shed requests remain in the diagnostic grid but are
ineligible to define capacity. Policy-run shedding is retained as a measured
failure: composite goodput counts a shed request as not good.

## Frozen arms and samples

The evaluation uses seeds 42 and 123 with 12 parallel workers. It contains six
fixed arms and no tuning grid:

1. occupancy-capacity causal SLO externality, `V=8`;
2. the same occupancy controller without the resident externality term;
3. causal SLO externality without a capacity term, `V=8`;
4. joint least-projected-TTFT;
5. Kairos with its fixed `beta=0.5` setting;
6. the best capacity-probe fixed joint plan, reported as an offline yardstick.

The public catalog supplies TTFT and ITL targets but no end-to-end target. The
study uses the catalog TTFT/ITL values and declares
`E2E = TTFT + mean_output_tokens * ITL`: 16 s for interactive chat, 802 s for
reasoning, and 40 s for deep research. These are per-request simulator targets,
not claims that BLIS reproduces the catalog's percentile-level scoring rule.

## Registered interpretation

The occupancy controller is revived on a fleet only if it improves goodput over
the no-capacity causal-externality controller by at least 0.02 on two of the
three workloads, loses by no more than 0.05 on the third, and is within 0.05 of
the better public baseline on at least two workloads. A pass only on the
heterogeneous fleet narrows the claim to heterogeneous hardware; it does not
reverse the homogeneous failures. If neither fleet passes, occupancy-controller
development closes and only the myopic causal-externality result remains.

Candidate traces will report the actual remote fraction and, for each request,
whether the best remote action's score is lower than, higher than, or tied with
the best local action. The net-good and capacity terms are reported separately,
including their conflict fraction and absolute magnitudes. These are snapshot
decision diagnostics, not end-to-end policy regret.
