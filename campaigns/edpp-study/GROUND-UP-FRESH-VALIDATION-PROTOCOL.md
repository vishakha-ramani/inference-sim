# Ground-up VaR fresh validation protocol

This protocol was frozen after the opened-development-seed comparison selected
the rebuilt reduced policy and before any fresh-seed result was generated. The
comparator section was revised after the user clarified that joint routing
must not be part of a ground-up policy-building campaign. The candidate and
its parameters were not changed.

## Candidate

`var_prefill_nostability_exactvar_pathwork`:

- reduced local-versus-remote decision only;
- normal decode and prefill routing;
- deployable utility VaR;
- exact marginal prefill overlap for `VaR(disagg)`;
- path-specific observable prefill work;
- `lambda_p = 0`;
- no arriving-request self-good term;
- no TTFT/decode-join overlap flag.

The decision is:

```text
disaggregate iff VaR(local) - VaR(disagg) > 0
```

## Fresh seeds

`16381, 32771, 65537, 131071`

These seed identifiers did not occur in the campaign artifacts or scripts
before this protocol was written.

## Conditions

The existing nine frozen minimax conditions:

- synthetic, RAG, and shared-prefix workloads;
- 60%, 80%, and 95% of each workload's measured static fleet ceiling;
- the same 1P2D topology, request counts, targets, and normal routers.

## Comparator panel

- selected exact reduced VaR candidate;
- legacy reduced VaR-only;
- corrected reduced VaR+prefill (`lambda_p=0.25`);
- always disaggregate;
- never disaggregate;
- the best calibrated static fraction for the condition, reported separately
  as a system-level yardstick.

No parameter is selected on the fresh seeds.

## Primary analysis

For each policy and condition, average goodput over the four fresh seeds.
For the deployable ground-up arms, define the reference as the best tested arm
mean in that condition. Report each arm's maximum regret across the nine
conditions. Keep the condition-tuned static fraction outside that ranking.

This is regret to the best tested ground-up arm, not oracle regret. The
condition-tuned static fraction is a separate system-level yardstick, and
placement-pinned one-request replay is the request-selection diagnostic.
