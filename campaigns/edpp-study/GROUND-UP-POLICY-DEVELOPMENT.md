# Ground-up development of the P/D policy

Date: 2026-07-30

## Current conclusion

The current reduced-policy candidate is **corrected causal VaR-only**:

```text
J_local  = VaR(local)
J_remote = VaR(remote)

disaggregate iff J_local - J_remote > 0
```

Here, VaR is the predicted SLO value destroyed among requests already sharing
the affected decode or prefill resource. The candidate:

- charges only the arriving request's marginal prefill work during chunks that
  actually overlap each co-resident;
- uses the local and remote paths' own observable prefix-cache state;
- has no prefill-stability coefficient (`lambda_p = 0`);
- has no arriving-request self-good or TTFT term.

The evidence does not say that stability or arriving-request value can never
help. It says a term should be added only after a measured failure identifies
the missing mechanism. The old stability term is rejected because it lowers
fresh goodput without preventing a demonstrated prefill-pool failure. The
arriving-request self-good term is rejected because the current TTFT ordering
is not reliable enough to drive it.

This is deliberately a reduced policy. Decode and prefill placement remain
owned by the configured routing scorers; the policy decides only whether the
request's prefill stays local or runs in the prefill pool.

## What the paper should ask

The clean paper story is **better per-request disaggregation decisions while
routing remains with the existing scorer**. Joint routing is a different,
larger action space: it can improve goodput by choosing a better instance even
if its P/D decision is worse. It should be considered only as a separately
identified system extension after the reduced decision rule is established.

The policy does not need to beat every arm in every workload. The primary
system claim should be about **worst-condition regret** over a declared
workload and load panel, with all policies in that ranking sharing the stated
scope.

The best static disaggregation fraction is useful as a fleet-capacity
yardstick. It answers, “How much P/D separation can this fleet use under this
condition?” It does not identify which individual requests should be
disaggregated. A Bresenham schedule is request-agnostic, so agreement with its
request labels has no policy meaning.

The request-level question is:

> Given the request and the system state at its arrival, did disaggregating
> this request preserve more total SLO value than keeping it local?

That question allows the same request shape to be local in one state and
remote in another.

## Evidence ladder

### 1. Starting observation: aggregate performance hid bad decisions

The original reduced `var_prefill` policy combined VaR with a prefill-queue
stability term. Its aggregate goodput looked plausible, but that did not tell
us whether it chose the right requests. The first ground-up step was therefore
not another policy ranking; it was to replay individual decisions while
holding normal-router placements fixed.

### 2. Which-request counterfactual

For seed 13, the diagnostic replayed the original normal-router placements,
pinned every P/D decision, and flipped one sampled request. All baseline
replays reproduced original goodput exactly. This isolates a one-request P/D
deviation while holding placement decisions fixed.

| condition and sampled decision | flip helps total goodput | interpretation |
|---|---:|---|
| RAG near-high: large request kept local although predicted remote TTFT was lower | 19/20 | the policy often kept the wrong high-impact request local |
| shared medium: local due to stability veto | 20/20 | the blanket stability charge rejected useful remote prefills |
| shared near-high: remote or stability-veto groups | 0/40 | composite goodput was already in a saturated dead zone |

This is local hindsight regret, not a global request-label oracle. A flip can
change later queue evolution, and zero delta can mean that the discrete SLO
metric is insensitive away from a deadline.

### 3. VaR-only versus `+stability`

The seed-13 ladder suggested that the prefill-stability term was harmful. The
same comparison was then run on all eight previously opened confirmation
seeds, without tuning:

| condition | mean delta: VaR-only minus `+stability` | 95% t interval | W/T/L |
|---|---:|---:|---:|
| synthetic low | 0.000 | [0.000, 0.000] | 0/8/0 |
| synthetic medium | 0.000 | [0.000, 0.000] | 0/8/0 |
| synthetic near-high | 0.000 | [0.000, 0.000] | 0/8/0 |
| RAG low | -0.006 | [-0.018, 0.005] | 3/0/5 |
| RAG medium | +0.008 | [-0.003, 0.018] | 6/0/2 |
| RAG near-high | +0.014 | [-0.021, 0.049] | 6/0/2 |
| shared-prefix low | **+0.078** | **[+0.075, +0.081]** | 8/0/0 |
| shared-prefix medium | **+0.057** | **[+0.033, +0.082]** | 8/0/0 |
| shared-prefix near-high | **+0.032** | **[+0.029, +0.035]** | 8/0/0 |

One VaR-only RAG near-high run dropped one unservable request. The request is
conserved and counted as a zero-good failure; it is not excluded.

These seeds are now development data for the new variant. They cannot serve as
fresh held-out evidence for a final VaR-only claim.

### 4. Adding the arriving request's own value

The next dimensionless objective was:

```text
VaR(local) - VaR(disagg)
  + good_self(disagg) - good_self(local)
  > lambda_p * prefill_queue_stability
```

It adds no new coefficient because VaR and `good_self` are both measured in
predicted good-request value.

At frozen seed 13 it improved RAG near-high from 0.464 to 0.473. The self term
correctly singled out a long-prompt population: requests for which it favored
remote prefill had median uncached prompt length about 48.9k tokens.

But it reduced shared-prefix goodput:

| condition | `var_prefill` | `+self` | delta |
|---|---:|---:|---:|
| shared low | 0.904 | 0.832 | -0.072 |
| shared medium | 0.674 | 0.623 | -0.051 |
| shared near-high | 0.038 | 0.033 | -0.005 |

The trace explains the failure: the self term favored local execution for
every shared-prefix request because the TTFT estimator ranked remote as slower
for every request.

## TTFT diagnosis and causal overlap repair

The one-request replays provide paired local and remote TTFT for the same
sampled request. In every shared-prefix stratum, the model predicted:

```text
TTFT(remote) - TTFT(local) = +35.152 ms
```

The realized paired median was approximately:

```text
TTFT(remote) - TTFT(local) = -2.755 ms
```

At shared medium, the predicted route ordering was wrong for all 40 sampled
requests across the remote and stability-veto groups. At RAG near-high:

- for the long requests predicted to favor remote, the TTFT ordering was
  correct for 13/20;
- for the requests predicted to favor local but sent remote by VaR, remote was
  actually faster for 15/20, so the ordering was correct for only 5/20.

Admission traces locate much of the shared-prefix bias:

- prefill admission is close: realized median 15.563 ms versus predicted
  16.618 ms;
- remote decode admission is overpredicted by roughly 10--12 ms at the
  median;
- local admission is underpredicted by roughly 6--10 ms at the median.

The paired phase timestamps confirmed a structural error. The old remote TTFT
formula added a decode-admission estimate measured at routing time *after* the
remote prefill and transfer interval:

```text
t_adm_prefill(now) + remote_prefill_work + transfer
  + t_adm_decode(now) + first_decode
```

For representative shared-medium request 312, local execution waited 33.969 ms
for admission and then 27.782 ms for its first token, totaling 61.751 ms.
Remote prefill plus transfer consumed 17.449 ms while the decode queue drained,
then the decode sub-request waited the remaining 16.520 ms. Both paths reached
the same absolute decode-admission instant; remote emitted its first token at
58.997 ms. The old formula incorrectly serialized the 17.449 ms lead and the
full decode wait.

The explicit `--edpp-ttft-overlap-aware` ablation therefore uses:

```text
remote_lead =
    t_adm_prefill(now) + remote_prefill_work + transfer

TTFT(remote) =
    max(remote_lead, t_adm_decode(now)) + first_decode
```

This is a causal repair, not a fitted offset.

### Fresh paired validation

The repair was evaluated at fresh seed 29 on 80 time-stratified requests. Every
replay pinned the exact baseline decode and prefill placements and flipped only
the sampled request:

| condition | serial order | overlap order | serial MAE | overlap MAE |
|---|---:|---:|---:|---:|
| shared low | 1/20 | 1/20 | 36.85 ms | 15.99 ms |
| shared medium | 3/20 | 3/20 | 34.31 ms | 14.31 ms |
| shared near-high | 1/20 | 5/20 | 36.46 ms | 10.90 ms |
| RAG near-high | 12/20 | 11/20 | 425.37 ms | 433.19 ms |
| **overall** | **17/80** | **20/80** | **133.25 ms** | **118.60 ms** |

Overlap removes a real source of bias and lowers overall absolute error, but it
does not make request-level ordering trustworthy. Realized remote was faster
for 65/80 requests; the repaired estimator predicted remote for only 21/80.
Shared admission traces explain the residual: normal 50 ms telemetry snapshots
underestimate the current local iteration/admission wait by roughly 12--16 ms
in representative cases. That state error is larger than the typical 2.75 ms
remote advantage.

### No-retuning policy check

At frozen seed 13, VaR-only with the overlap repair was exactly identical to
VaR-only in all nine cells: no decision, fraction, or goodput changed. The
repair did not move a co-resident VaR comparison across its decision boundary.

Re-testing the deferred arriving-request value with the repaired formula still
failed:

| condition | VaR-only | overlap + `good_self` | delta |
|---|---:|---:|---:|
| RAG low | 0.964 | 0.947 | -0.017 |
| RAG medium | 0.820 | 0.818 | -0.002 |
| RAG near-high | 0.484 | 0.480 | -0.004 |
| shared low | 0.987 | 0.854 | -0.133 |
| shared medium | 0.681 | 0.665 | -0.016 |
| shared near-high | 0.070 | 0.070 | 0.000 |

Its worst regret within this ablation was 0.133, versus 0.004 for VaR-only.
The correct ground-up decision is therefore to reject `good_self` for now,
not to fit another correction to these development outcomes.

## Auditing how VaR was calculated

The next audit found a concrete error in `VaR(remote)`. The legacy prefill-pool
model charged every surviving prefill occupant the arriving request's complete
serial remote-prefill duration:

```text
number_of_chunks * baseline_iteration_time + full_prefill_work
```

This is not the marginal delay caused by the arriving request. Baseline
iteration time is paid even without that request, and an occupant with one
remaining chunk cannot overlap a many-chunk arriving request in full.

The corrected marginal work over `k` overlapping chunks is:

```text
processed = min(uncached_tokens, k * chunk_tokens)
cached_prefix = total_prompt_tokens - uncached_tokens

marginal_prefill_work =
    C_pf * processed
    + C_attn * processed * (cached_prefix + processed / 2)
```

This handles prefix-cache hits, a partial final chunk, and each occupant's
actual overlap horizon. It excludes baseline iteration time. The same causal
overlap rule is used for local prefill interference.

A second invariant uses each path's own observable cache state when computing
uncached tokens, prefill work, chunk count, and queue-work booking. In the
evaluated one-prefill-node topology the remote cache location is known. This
path-specific repair did not change the seed-13 decisions beyond the exact
overlap repair, but it removes a real modeling inconsistency and is retained.

On eight opened development seeds, corrected VaR improved over legacy VaR-only
most clearly at RAG near-high:

| condition | legacy VaR | corrected VaR | paired delta | 95% t interval |
|---|---:|---:|---:|---:|
| RAG low | 0.948 | 0.953 | +0.005 | [-0.006, +0.015] |
| RAG medium | 0.749 | 0.764 | +0.015 | [-0.022, +0.052] |
| RAG near-high | 0.425 | 0.479 | **+0.054** | **[+0.027, +0.081]** |

Adding the old stability term back on top of the corrected calculation reduced
RAG near-high by 0.050 and shared low by 0.078 on those same development
seeds. This rules out the explanation that stability was useful but merely fed
the wrong cache or prefill-work operand.

## Fresh ground-up validation

The corrected policy was frozen and evaluated on four seed identifiers not
previously present in the campaign artifacts. The inherited joint policy was
not part of this comparison. The deployable ladder contains only successive
ground-up variants and the two endpoint decisions:

| rank | deployable ground-up arm | worst regret | worst condition |
|---:|---|---:|---|
| 1 | corrected causal VaR | **0.047** | shared near-high |
| 2 | legacy VaR-only | 0.065 | RAG near-high |
| 3 | corrected VaR + stability | 0.082 | shared low |
| 4 | always disaggregate | 0.139 | RAG medium |
| 5 | never disaggregate | 0.703 | shared medium |

For each condition, regret is relative to the best mean among these five
tested deployable arms. It is not oracle regret.

The estimator repair left synthetic and shared-prefix decisions unchanged but
improved mean RAG goodput over legacy VaR-only:

| condition | corrected | legacy | paired delta | W/T/L |
|---|---:|---:|---:|---:|
| RAG low | 0.952 | 0.940 | +0.012 | 3/0/1 |
| RAG medium | 0.713 | 0.690 | +0.023 | 3/0/1 |
| RAG near-high | 0.390 | 0.325 | +0.065 | 3/0/1 |

Four seeds are too few for the RAG intervals to exclude zero, so the
appropriate claim is replication of the direction and a lower observed
worst-case regret, not a definitive per-condition effect.

The corrected stability ablation was worse than corrected VaR at shared low
by 0.077 (95% interval [-0.090, -0.064]) and at shared near-high by 0.028
([-0.042, -0.014]). It had no fresh condition with a resolved benefit.

The best calibrated static fraction is reported separately. It is
condition-tuned and request-agnostic, so it is a system-level yardstick rather
than a member of the deployable ranking or a request-selection oracle.
Corrected VaR exceeded it on all four RAG-low seeds and had higher RAG means at
all three loads. The static fraction remained better on shared-prefix,
especially near-high:

```text
corrected VaR:       goodput 0.045, realized fraction 0.890
best static fraction: goodput 0.092, fixed fraction 1.000
```

All 216 runs passed the hard-validity gate. Two RAG near-high runs dropped one
unservable request; each drop remains a zero-good outcome in the denominator.

## Next mechanism question

The policy now has a specific residual failure to explain: at shared-prefix
near-high it keeps roughly 11% of requests local even though the all-remote
endpoint is better. The next step is not to restore a generic stability
penalty. It is to inspect the local decisions made by corrected VaR:

1. Separate strict negative-VaR decisions from exact or near-zero ties.
2. Replay groups of those local decisions as remote while pinning all routing.
   One-request flips can be zero in this saturated condition even when a group
   change crosses an SLO threshold.
3. Determine whether the missing signal is prefill feasibility, decode
   interference not represented in VaR, or tie-breaking when both predicted
   externalities are zero.
4. Add the smallest observable term or guard that targets the identified
   failure, then repeat the ablation and use new seeds.

Before reconsidering `good_self`, observable admission-state prediction under
the normal 50 ms snapshot interval should pass a declared paired route-order
accuracy threshold.

This sequence is itself the paper's policy-development evidence: each term has
a first-principles hypothesis, a falsifiable diagnostic, and an observed
consequence.
