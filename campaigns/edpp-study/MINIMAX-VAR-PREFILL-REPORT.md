# Minimax-regret evaluation of `var-prefill`

Date: 2026-07-29

## Question

Does one globally calibrated simplified policy,

```text
VaR(local) - VaR(disagg) > lambda_p * prefill-queue stability
```

have the smallest worst-condition goodput regret among the evaluated policies?

The answer on this panel is **no across all policies, but yes within the
routing-preserving reduced-policy comparison set**. Original joint `dpvar`
ranks first; simplified `var-prefill` ranks second. Their observed difference
is not resolved by four held-out seeds.

## Frozen setup

- Topology: 1 prefill pod and 2 decode pods.
- Decode routing: existing queue-depth router unless a joint policy overrides
  it.
- Prefill routing: existing prefill router unless a joint policy supplies a
  pod.
- Workloads: synthetic batch, mixed RAG, and shared prefix.
- Loads: 60%, 80%, and 95% of each workload's stable static fleet ceiling.
- Static fractions: `0.0, 0.1, ..., 1.0`, produced exactly by a
  routing-preserving Bresenham plan.
- Calibration seed: `42`.
- Held-out seeds: `7, 123, 2024, 9001`.
- Lambda grid: `0.25, 0.5, 1.0, 2.0, 4.0`.
- Primary metric: maximum, over the nine workload/load conditions, of goodput
  regret to the best held-out policy mean in that condition.
- Secondary metric: maximum regret to the frozen condition-specific static
  yardstick.

The campaign ran 537 simulations: 33 capacity probes, 99 static calibration
runs, 45 lambda calibration runs, and 360 held-out policy runs. All held-out
runs conserved requests and had zero unservable drops, timeouts, unfinished
requests, and length-capped requests.

## Fleet ceilings and evaluated rates

| workload | stable ceiling (req/s) | ceiling phi | low | medium | near-high |
|---|---:|---:|---:|---:|---:|
| synthetic | 1.492 | 0.5 | 0.895 | 1.194 | 1.418 |
| RAG | 3.761 | 1.0 | 2.257 | 3.009 | 3.573 |
| shared-prefix | 80.420 | 1.0 | 48.252 | 64.336 | 76.399 |

## Frozen static yardsticks

| condition | best calibration phi |
|---|---:|
| synthetic low | 0.1 |
| synthetic medium | 0.1 |
| synthetic near-high | 0.4 |
| RAG low | 0.6 |
| RAG medium | 0.6 |
| RAG near-high | 0.8 |
| shared-prefix low | 1.0 |
| shared-prefix medium | 1.0 |
| shared-prefix near-high | 1.0 |

The best universal static fraction by calibration minimax regret was
`phi=1.0`, with calibration worst regret `0.195`.

Lambda calibration selected `lambda_p=0.25`. Its calibration worst regret was
`0.085`.

## Held-out minimax result

| rank | policy | worst regret | worst condition | worst regret to static yardstick | mean phi distance |
|---:|---|---:|---|---:|---:|
| 1 | original joint `dpvar` | 0.148 | RAG near-high | 0.077 | 0.386 |
| 2 | reduced `var_prefill` | 0.168 | RAG near-high | 0.083 | 0.218 |
| 3 | Kairos | 0.224 | RAG medium | 0.052 | 0.354 |
| 4 | always disaggregate | 0.289 | RAG medium | 0.117 | 0.378 |
| 5 | universal static `phi=1.0` | 0.289 | RAG medium | 0.117 | 0.378 |
| 6 | joint DPP | 0.543 | shared-prefix low | 0.543 | 0.319 |
| 7 | reduced least-TTFT | 0.697 | shared-prefix medium | 0.697 | 0.565 |
| 8 | never disaggregate | 0.697 | shared-prefix medium | 0.697 | 0.622 |
| 9 | joint least-TTFT | 0.703 | shared-prefix medium | 0.703 | 0.570 |

Both top policies attain their worst regret in RAG near-high. The paired
held-out goodput difference (`dpvar - var_prefill`) there is `0.020`, with a
95% t interval of `[-0.012, 0.052]`. The observed ordering therefore favors
joint `dpvar`, but four seeds do not statistically resolve the top-two gap.

Within the routing-preserving comparison set, the minimax order is:

1. `var_prefill`
2. always disaggregate / universal `phi=1.0`
3. reduced least-TTFT / never disaggregate

## What the fraction diagnostic says

`var_prefill` has the smallest mean distance to the calibration oracle
fraction (`0.218`) among the evaluated deployable policies. This supports the
mechanism claim that it adapts its disaggregation share across workloads:

- almost zero for synthetic;
- roughly 0.62--0.68 for RAG;
- approximately 0.48, 0.90, and 0.48 for shared-prefix low, medium, and
  near-high, respectively.

It does **not** support using fraction distance as the primary outcome.
At RAG near-high, joint least-TTFT obtains the best held-out mean goodput while
using a fraction near `0.20`, far from the static oracle `0.8`. Joint routing
changes decode and prefill placement as well as the fraction, so its goodput
cannot be explained by fraction matching alone.

The clearest simplified-policy failure is shared-prefix near-high:
`var_prefill` realizes about `phi=0.475` and goodput `0.019`, whereas the frozen
static `phi=1.0` yardstick obtains `0.088`. The prefill-stability charge appears
to discourage offload precisely when this workload needs nearly complete
disaggregation.

## Conclusion and next experiment

The broad claim that simplified `var-prefill` has the least worst-case regret
is not supported. A narrower claim is supported: it is the best
routing-preserving reduced P/D policy in this panel, and it substantially
improves worst regret over any one static fraction.

The next clean experiment should keep every parameter frozen and add held-out
seeds to resolve the `0.020` top-two gap. In parallel, decision traces for
shared-prefix near-high should test why the prefill-stability term suppresses
offload. Any change to the rule or lambda grid should be calibrated in a new
campaign and evaluated on new held-out seeds, rather than tuned against these
results.

## Confirmation and subsequent ground-up diagnosis

An independent eight-seed confirmation preserved the original ranking.
`dpvar` had worst regret `0.161`; `var_prefill` had `0.179`. The paired
RAG-near-high gap was `0.018`, with a 95% t interval of
`[-0.012, 0.049]`. Combining the original and confirmation seeds gave a gap of
`0.019`, interval `[-0.002, 0.040]`.

The request-level follow-up changed the interpretation. A placement-pinned
one-request replay showed that 19/20 sampled long RAG requests kept local by
`var_prefill` improved total goodput when disaggregated, while all 20 sampled
shared-medium stability vetoes improved goodput when sent remote.

This motivated a ground-up ladder:

1. co-resident VaR only;
2. add prefill-queue stability;
3. add the arriving request's predicted composite-good.

On the eight confirmation/development seeds, removing the stability term
improved all three shared-prefix conditions on every seed. The expanded-panel
worst regret was `0.165` for VaR-only, `0.161` for joint `dpvar`, and `0.179`
for `var_prefill`. The `dpvar - VaR-only` RAG-near-high gap was only `0.0048`,
with interval `[-0.0178, 0.0273]`.

Adding arriving-request value helped the RAG-near-high seed-13 failure, but
hurt shared-prefix because the TTFT estimator predicted remote slower for
every shared request. Paired one-request outcomes instead found remote TTFT
about `2.75 ms` faster at the median, versus the model's invariant prediction
that it was `35.15 ms` slower.

The current policy candidate is therefore VaR-only.

A subsequent phase-timing replay confirmed that the old remote TTFT formula
serialized remote prefill/transfer and decode-queue waiting even though they
overlap. An explicit overlap-aware ablation repaired that causal error. On a
fresh seed with 80 placement-pinned request pairs, it reduced TTFT-difference
MAE from 133.25 ms to 118.60 ms, but route-order accuracy improved only from
17/80 to 20/80. The remaining admission-state error is too large for the
arriving-request value to drive decisions.

The no-retuning policy check agreed: overlap alone left VaR-only exactly
unchanged in all nine cells, while overlap plus arriving-request value incurred
0.133 worst regret within the ablation, driven by shared-prefix low. The
arriving-request term remains rejected. The complete mechanism narrative and
next protocol are in `GROUND-UP-POLICY-DEVELOPMENT.md`.
