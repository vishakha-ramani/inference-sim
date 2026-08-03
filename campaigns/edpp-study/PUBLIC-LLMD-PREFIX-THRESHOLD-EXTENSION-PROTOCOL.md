# Workload-tuned llm-d prefix-threshold extension protocol

Status: frozen before execution on 2026-07-31.

## Question

Does the final causal-externality policy remain competitive after adding a
strong, workload-tuned version of llm-d's decomposed prefix-threshold policy to
the completed public load/static benchmark?

This is an append-only comparator extension. Historical policy outputs are not
rerun or changed. The extension reuses their deterministic request traces for
paired comparison and is not described as a new fresh-seed confirmation.

## Policy

The policy reproduces llm-d's decode-first P/D sequence:

1. Score the decode pool and select one decode instance.
2. Query the selected decoder's prefix-cache match and compute
   `uncached = input_tokens - cached_blocks * block_size`.
3. Disaggregate exactly when `uncached > threshold`.
4. If disaggregated, independently score the prefill pool and select one
   prefill instance. The decoder selected in step 1 is not reconsidered.

Both pools use llm-d's shipped P/D scorer profile:

`precise-prefix-cache:2,queue-depth:1`

The precise-cache and threshold queries use the same cache snapshot with the
production-style 50 ms signal delay. This is decomposed routing, not joint
routing. The shipped threshold 16 is present in the calibration grid, but the
reported comparator is workload-tuned and must not be labeled the unmodified
shipped llm-d configuration unless 16 wins for that workload.

## Frozen conditions

Reuse the completed benchmark's:

- three public request shapes: interactive, reasoning, and deep research;
- homogeneous H100 and realistic H100/A100 `1P2D` fleets;
- rates at 60%, 80%, and 95% of the already frozen measured capacity;
- actual TTFT, mean-ITL, and E2E targets;
- request counts, model, latency coefficients, transfer model, and cache model.

No capacity or static-plan stage is repeated.

## Threshold calibration

Candidate thresholds, in uncached tokens, are frozen as:

`0, 16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072`

Use development seeds 42 and 123. Select exactly one threshold for each
workload by maximizing equal-run mean hard composite goodput across both fleets,
all three loads, and both seeds (12 runs per threshold). A point is eligible only
if all 12 runs terminate exactly with zero drops, timeouts, and length caps.

Tie breaking is frozen as:

1. higher mean goodput;
2. lower standard deviation across the 12 runs;
3. threshold 16, if it is among the remaining exact ties;
4. smaller threshold.

The held-out output files and historical policy means must not be read by the
calibration or selection code.

## Matched held-out extension

After writing `threshold_selection.json`, run only the frozen tuned policy on
the original benchmark's four trace seeds:

`4194319, 8388617, 16777259, 33554467`

This produces exactly:

`3 workloads x 2 fleets x 3 loads x 4 seeds = 72 runs`.

These seeds are reused to obtain paired comparisons with stored policy outputs.
They are not represented as previously unseen or fresh. Thresholds and all
policy settings remain frozen before these runs.

## Outcomes and gates

Report:

- mean goodput and remote fraction in every one of the 18 conditions;
- equal-condition mean goodput;
- worst-condition shortfall to the best tested deployable-policy mean after
  adding this policy;
- paired goodput difference between causal externality and this comparator;
- all selected thresholds and every calibration point.

Every run must satisfy exact terminal accounting with zero drops, timeouts,
length caps, queued requests, and running requests. Failure invalidates the
affected comparison; it does not authorize retuning.

The tuned threshold is a condition-informed baseline over a restricted policy
family, not an oracle or upper bound.

## Superseding unified confirmation (frozen 2026-07-31)

To eliminate the retrospective-addition caveat from the primary paper result,
run one unified confirmation after the thresholds above have been frozen. Use
four seeds not used by any earlier campaign:

`1000000007, 1000000009, 1000000033, 1000000087`

For every one of the 18 frozen conditions, run all six policies on each seed:

1. causal externality;
2. joint least-TTFT;
3. paper-mode Kairos with `alpha=1.3`, `beta=1.0`, the request TTFT gate,
   strictest-resident TBT protection, and discrete chunk search;
4. workload-tuned llm-d prefix threshold;
5. frozen goodput-tuned static plan;
6. frozen capacity-selected static plan.

This produces `18 x 6 x 4 = 432` runs. Capacity points, rates, static plans,
thresholds, scorers, and every policy setting remain unchanged. Analyze only
after all runs finish and pass the original terminal-accounting gates. This
unified block supersedes the retrospective 72-run addition for the paper's
primary comparison; the earlier block remains an archival consistency check.
