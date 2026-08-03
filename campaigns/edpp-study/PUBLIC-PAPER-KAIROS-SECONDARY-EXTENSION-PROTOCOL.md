# Paper-Kairos secondary extension protocol

> Reporting amendment (2026-07-31): the topology portion is withdrawn. Kairos
> does not specify multi-prefill-instance selection, so only the single-prefill
> mixed/bursty extension remains reportable.

Status: frozen before execution on 2026-07-31.

## Question

Do the mixed-workload/burst and topology conclusions change when their
historical admission-aware Kairos adaptation is replaced by the paper-oriented
Kairos implementation used in the unified primary benchmark?

## Frozen replacement

Run only `kairos-paper` with `alpha=1.3` and `beta=1.0`. Paper mode enforces
the request TTFT gate, protects the strictest resident TBT target, uses exact
queued prefill tokens, and searches executable discrete chunk sizes. It omits
the admission-delay and transfer additions of the historical adaptation.

Every workload, fleet, topology, offered rate, request trace, SLO, request
count, and seed remains frozen to the completed studies. Existing causal
externality, joint least-TTFT, and condition-tuned static outputs are reused
without modification.

## Runs

- Mixed workload and burst: 2 fleets x 4 profiles x 4 seeds = 32 runs.
- Topology: 3 provisionings x 3 workloads x 4 seeds = 36 runs.
- Total new runs: 68.

The combined matched analyses replace the adapted-Kairos rows but preserve the
archived original CSVs and reports. The replacement is a corrected baseline
extension, not a newly seeded rerun of the other policies.

## Validity gates

All 68 new runs must finish with exact terminal accounting, zero drops, zero
timeouts, and zero output-length caps. Each combined analysis must contain the
original frozen seed set exactly once for every policy and cell. No condition
may be removed or retuned after inspection.
