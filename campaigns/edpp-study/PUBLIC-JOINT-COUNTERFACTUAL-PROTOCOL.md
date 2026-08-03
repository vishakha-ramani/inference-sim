# Public joint request-counterfactual protocol

Status: frozen before running the diagnostic seed.

## Question

For requests routed by the final joint causal-externality policy, would changing
that request alone to another joint action have improved total trace goodput?

This is an exact one-request forced-action diagnostic. It is not a global oracle:
all other request actions remain fixed to the policy's original realized plan.

## Frozen conditions

- Policy: joint causal resident externality minus arriving-request value, no
  capacity term, V=8.
- Routing value: smooth TTFT x E2E. ITL remains only in reported hard goodput.
- Workloads: interactive, reasoning, and deep research.
- Fleets: homogeneous H100 1P2D and realistic H100/A100 1P2D.
- Loads: 0.60C, 0.80C, and 0.95C, copied unchanged from
  `out/public_load_static_benchmark_v1/capacity_selection.json`.
- Diagnostic seed: `1073741827`, not used by an earlier campaign.
- Generic request timeout: disabled.

The 1P2D action set has four actions:

1. local prefill and decode on D0;
2. remote prefill P0 to decode D0;
3. local prefill and decode on D1;
4. remote prefill P0 to decode D1.

## Plan capture and replay gate

For each of the 18 conditions:

1. Run the final online policy and capture every candidate and chosen action.
2. Convert the chosen actions into a total fixed plan.
3. Replay the unchanged fixed plan.
4. Require exact equality of overall, TTFT, ITL, and E2E goodput plus terminal
   accounting. A failed replay gate invalidates that condition.

The chosen action is captured from the joint candidate trace, not the outcome
trace, because local outcome records do not always preserve an explicit joint
decode override.

## Request sampling

Sample exactly eight requests per condition after plan capture, for 144 sampled
decisions total.

Sampling is deterministic and stratified by the policy's chosen joint action:

- rank requests by SHA-256 of `condition|request_id`;
- take up to two requests from each chosen-action stratum;
- fill any unoccupied slots from the remaining requests in hash order.

This prevents a high-frequency action from completely hiding rarer decisions.
The sample is frozen before any deviation result is read.

## Forced alternatives

For every sampled request, rerun the complete trace three times, once for each
alternative joint action. Only that request's plan row changes. Thus the
diagnostic contains 432 deviation runs, in addition to 18 online baselines and
18 unchanged-plan replay gates.

The hindsight-best forced action is the action with maximum final trace goodput.
The policy agrees when its original action ties for that maximum. Positive
one-request regret is:

`max_action goodput(action) - baseline goodput`.

For tied improving actions, classify the error using the best action with the
fewest changed dimensions (decoder and local/remote placement), then lexical
action order. This produces conservative decoder-only, placement-only, or both
labels.

## Reported metrics

- agreement with the best forced action;
- positive-regret fraction and mean/total goodput regret;
- equivalent number of good requests recovered;
- local decisions that should have been remote and the reverse;
- decoder-only, placement-only, and joint decoder+placement errors;
- goodput lost by each error class;
- results by fleet, workload, load, and original chosen action.

## Validity gates

- Exactly 18 policy baselines, 18 unchanged-plan replays, 144 sampled requests,
  and 432 deviation runs.
- Every captured plan is total and has four candidate actions per request.
- Every online policy choice is the exact joint-score argmin.
- Every unchanged-plan replay matches its online baseline exactly.
- Terminal accounting is exact in every baseline, replay, and deviation.
- No timeout or length-cap outcomes.

No result is used to retune the policy.
