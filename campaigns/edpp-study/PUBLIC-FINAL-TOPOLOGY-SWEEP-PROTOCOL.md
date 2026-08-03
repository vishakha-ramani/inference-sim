# Final-policy public topology sweep protocol

> Reporting amendment (2026-07-31): Kairos is excluded from the topology
> comparison. Its published policy defines a single prefill path and does not
> specify routing among multiple prefill instances. The archived Kairos runs in
> this campaign are therefore not paper results and must not be reported.

## Frozen question

The existing topology results in `infocom/main.tex` evaluate the rejected
congestion-normalized controller.  This campaign replaces that evidence for the
final joint causal-SLO-externality policy, whose routing value is smooth
`TTFT x E2E` arriving-request value minus smooth resident externality, with no
capacity price.  Reported goodput remains the hard conjunction of TTFT, mean
ITL, and E2E.

The question is whether that one online policy remains competitive when a fixed
four-instance homogeneous H100 fleet is provisioned as:

- `1P3D`: one dedicated prefill and three decode-capable instances;
- `2P2D`: two dedicated prefill and two decode-capable instances; or
- `3P1D`: three dedicated prefill and one decode-capable instance.

This is a provisioning robustness study.  Capacity and goodput are compared
within each topology; raw request rates are never compared across differently
provisioned fleets.

## Workloads and SLOs

Use the completed independent-request public translations without modification:

| workload | TTFT | mean ITL | E2E |
|---|---:|---:|---:|
| interactive chat | 1 s | 50 ms | 16 s |
| reasoning | 2 s | 100 ms | 802 s |
| deep research | 10 s | 100 ms | 40 s |

The model, H100 coefficients, transfer model, queue scorer, scheduler limit,
and request counts are inherited from the public load/static benchmark.  The
generic simulator request timeout is disabled so it cannot tighten the 802 s
reasoning target.

## Stage A: topology-specific capacity envelopes

For every topology/workload, test deterministic fixed joint plans with remote
prefill share:

```text
phi in {0, 0.25, 0.50, 0.75, 1.0}.
```

Remote prefills and decodes are round-robined evenly over their eligible pools.
Capacity seeds are `42`, `123`, and `2024`.  Every point uses the workload's
frozen saturating offered rate and loose SLOs.  A point is capacity-eligible only
if all three runs are hard-valid and zero-drop.  Select the greatest mean central
completion rate, breaking ties by lower seed standard deviation and lower
`phi`.  This measured five-plan envelope is not a proof of global capacity.

## Stage B: topology-specific static calibration

For every topology/workload, evaluate the same five shares at:

```text
lambda = 0.90 C_topology,workload
```

using development seeds `42` and `123` and the actual SLOs.  Select the
hard-valid, zero-drop plan with highest mean composite goodput, breaking ties by
lower seed standard deviation and lower `phi`.  Freeze the selections before
confirmation.  This is a coarse condition-tuned static plan, not an oracle or
upper bound.

## Stage C: held-out comparison

Confirmation seeds are `2000000181`, `2000000193`, `2000000207`, and
`2000000227`.  As of protocol freeze, none appeared in campaign sources or
artifacts.  Do not inspect these runs before `capacity_selection.json` and
`static_selection.json` are written.

Compare five arms:

1. joint causal SLO externality without capacity prices (`V=8`);
2. joint least projected TTFT;
3. the workload-tuned llm-d prefix-threshold policy using
   `precise-prefix-cache:2,queue-depth:1` on both pools;
4. the frozen topology/workload-tuned static plan; and
5. the topology/workload capacity-selected static plan.

There are `3 topologies x 3 workloads x 4 seeds x 5 arms = 180` confirmation
runs.  Rank only the three deployable policies by worst regret to the best
deployable mean across the nine topology/workload cells.  Also report worst
regret separately within each topology.  Static gaps are descriptive and must
not be called oracle or policy regret.

Kairos is not an arm because its published policy does not define routing among
multiple prefill instances. The causal and least-TTFT arms use the validated
FCFS scheduler-step TTFT rollout with live arrival-time scheduler snapshots.

## Validity

Every confirmation run must satisfy:

```text
completed + dropped = injected
still_queued = 0
still_running = 0
timed_out = 0
length_capped = 0
```

Every focal-policy candidate trace must select an exact score argmin.  An
incomplete grid, missing seed, invalid terminal state, confirmation-informed
selection, or argmin mismatch invalidates the corresponding result.

This remains a discrete-event simulation of one model and homogeneous H100
instances.  It does not establish an optimal provisioning or global-oracle gap.
