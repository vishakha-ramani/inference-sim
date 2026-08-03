# Public no-capacity fresh confirmation protocol

## Frozen question

After the occupancy-capacity controller failed its registered development gates,
this confirmation compares the surviving causal-SLO-externality/no-capacity policy
against public baselines. It is not another capacity-controller experiment and does
not tune any policy.

## Conditions and samples

The six conditions are the three translated inference-perf request shapes
(`interactive`, `reasoning`, and `deep_research`) on the homogeneous H100 1P2D fleet
and the realistic H100-prefill/H100+A100-decode 1P2D fleet. Workloads, SLOs, request
counts, capacity-selected static plans, and rates at 85% of development-measured
capacity are frozen from `PUBLIC-WORKLOAD-HETEROGENEITY-CLOSEOUT-PROTOCOL.md`.

The held-out seeds are `262147`, `524309`, `1048583`, and `2097169`. None was used
in the public-workload development closeout.

## Frozen arms

1. joint causal SLO externality without capacity (`V=8`; multiplication by positive
   `V` does not change this capacity-free argmin);
2. joint least projected TTFT;
3. Kairos with `beta=0.5`;
4. the development capacity-selected static joint plan, as an offline yardstick.

The closed occupancy controller and its ablations are not rerun.

## Registered analysis

For every cell and policy, report mean goodput, remote fraction, all four seed values,
and drops or unfinished requests. Among the three deployable policies, rank policies
by maximum regret to the best deployable mean in each of the six cells, breaking ties
by higher overall mean goodput. Report paired per-seed goodput deltas from the
no-capacity policy to each comparator with 95% intervals. The static plan is excluded
from the deployable minimax ranking.

This study is descriptive and has no new tuning or pass threshold. It can support a
myopic causal-externality result, but not a Lyapunov, policy-regret, or universal-win
claim.
