# Complete decisive-campaign experiment summary

## 1. Question and answer

The experiment asked:

> Does the current deployable drift-plus-VaR policy earn its complexity when
> requests have observable routing-time differences?

The registered answer is **no**. dpVaR has meaningful wins, especially on RAG,
but fails both robustness criteria:

- dpVaR worst-case regret: **0.212**.
- Best universal static-policy worst-case regret: **0.210**.
- Required dpVaR worst-case regret: at most 0.100 and at least 0.050 better
  than every universal static policy.

This is a policy result. It does not test or invalidate the corrected analytical
capacity derivation in the paper.

## 2. Simulator and common configuration

The study used BLIS, a deterministic discrete-event simulator. It did not run
requests on physical GPUs.

| setting | value |
|---|---|
| model | `meta-llama/llama-3.3-70b-instruct` |
| latency coefficients | `coeffs-llama70b-h100-tp4.json` |
| policy fleet | one prefill-only instance plus two mixed instances (`1P2M`) |
| SLO-reference fleet | three mixed instances (`3M`) |
| mixed-pool decode routing | `queue-depth:1` |
| maximum concurrent requests per instance | 256 |
| KV block size | 16 tokens, simulator default |
| PD transfer bandwidth | 25 GB/s, simulator default |
| cache-signal delay | 50 ms, simulator default |
| arrival process | open-loop Poisson for synthetic and RAG |
| calibration seed | 42 |
| held-out policy seeds | 7, 123, 2024, 9001 |
| independent capacity/SLO seeds | 101, 211, 307, 401, 503 |

On the `1P2M` fleet, `instance_0` performs dedicated prefill. `instance_1` and
`instance_2` can perform a complete local request or decode a request prefetched
on `instance_0`. All three instances use the same frozen H100-TP4 coefficient
file.

All EDPP-family policies used:

- the roll-forward admission-delay estimator,
- size-aware KV-transfer cost,
- class-specific TTFT and ITL targets,
- no hidden output-length oracle.

`least_ttft_joint`, `dpp_joint`, and dpVaR enumerated the full set of local and
disaggregated decode/prefill candidates. dpVaR additionally used the deployable
censored output-length estimator,
normalized congestion plus the `util` VaR kernel, congestion weight 1, the
goodput objective, and class-specific E2E targets.

## 3. Workloads

### 3.1 Synthetic data generation

This is a local BLIS mirror of inference-perf's batch synthetic-data-generation
workload.

| property | configuration |
|---|---|
| class | `batch` |
| arrival | Poisson |
| shared prefix | 2,000 tokens, one prefix group |
| variable suffix | lognormal, `mu=5.259`, `sigma=1.3824`, range 50-15,000 |
| total prompt observed in probe | 2,050-17,000 tokens |
| output | Gaussian, mean 4,000, SD 2,500, range 500-8,000 |
| probe diversity | 302 distinct prompt lengths in 500 requests |
| probe cache-hit rate | 0.2664 |

Inference-perf describes the output distribution as uniform. BLIS has no native
uniform distribution, so the mirror uses a bounded Gaussian with the catalog's
stated mean and spread.

### 3.2 RAG summarization

This is a two-client approximation of inference-perf's bimodal batch
summarization/RAG workload.

| client | share | class | shared prefix | suffix distribution | output |
|---|---:|---|---:|---|---|
| vector QA | 0.78 | `standard` | 500 | lognormal, `mu=7.527`, `sigma=0.3853`, 1,000-8,000 | Gaussian 500 +/- 300, 50-2,000 |
| document read | 0.22 | `batch` | 500 | lognormal, `mu=10.949`, `sigma=0.3246`, 8,000-80,000 | Gaussian 500 +/- 300, 50-2,000 |

Both clients use Poisson arrivals and the same prefix group. The probe observed
total prompts from 1,500 to 80,500 tokens, 687 distinct lengths in 800 requests,
and cache-hit rate 0.0114.

BLIS has no native bimodal distribution. The two-client split approximates the
catalog's short-vector-QA and long-document modes. The per-mode spreads are
estimated from the catalog description rather than copied from a native
bimodal specification.

### 3.3 Shared prefix

This uses BLIS's native inference-perf-format example:

| property | value |
|---|---:|
| unique system prompts | 9 |
| users per prompt | 5 |
| shared prefix | 100 tokens |
| question | 447 tokens |
| total prompt | 547 tokens |
| output | 248 tokens |
| class | `standard` |
| probe cache-hit rate | 0.0561 |

The stationary comparisons replace the source's two stages with one rate.
The separate staged diagnostic restores two consecutive stages at the measured
low and high rates. This workload varies prefix identity and load, not prompt
length, so it is excluded from the registered variable-prompt significance
criterion.

## 4. Capacity measurement and policy rates

The campaign first measured a throughput envelope for deterministic fixed
shares. For `phi` in `{0, .2, .4, .6, .8, 1}`, `phi` is the fraction of requests
whose prefill runs on the dedicated prefill instance. The remainder runs prefill
and decode locally on a mixed instance.

Fixed plans use Bresenham interleaving, not random sampling, and alternate
decodes evenly across the two mixed instances. Capacity is the central
completion rate between the 10th and 90th percentiles of completion time. This
removes fill and drain transients.

| workload | saturation offer | requests/probe | phi=0 | phi=1 | best grid point | 3M reference |
|---|---:|---:|---:|---:|---:|---:|
| synthetic | 4 req/s | 1,200 | 1.480 | 1.482 | 1.495 at phi=.2 | 2.042 |
| RAG | 8 req/s | 1,200 | 3.224 | 3.746 | 4.672 at phi=.8 | 4.707 |
| shared-prefix | 160 req/s | 6,000 | 72.483 | 80.355 | 80.355 at phi=1 | 105.610 |

This is a discrete fixed-share envelope, not a proof of the global fleet
capacity. The RAG interior overload probes dropped 9-74 unservable requests;
drops were allowed only during capacity probing. All policy comparisons had
zero drops.

Rates were derived from the lower of the `phi=0` and `phi=1` endpoints:

- low: 60% of the lower endpoint;
- medium: 85% of the lower endpoint;
- high: the larger of 95% of the lower endpoint and 70% of the best fixed-share
  point.

RAG high is an exception: it is 90% of the lower endpoint. The general high-rate
formula produced decode-KV drops for `least_ttft_joint`; the lower rate passed a
two-seed zero-drop preflight. This adjustment was made before the final
evaluation and applied to every policy.

| workload | low | medium | high | requests per policy/seed |
|---|---:|---:|---:|---:|
| synthetic | 0.888 req/s | 1.258 req/s | 1.406 req/s | 800 |
| RAG | 1.934 req/s | 2.740 req/s | 2.902 req/s | 1,000 |
| shared-prefix | 43.490 req/s | 61.610 req/s | 68.859 req/s | 4,000 |

The staged shared-prefix run used 6,000 requests, a 60-second low stage at
43.490 req/s followed by a 60-second high stage at 68.859 req/s.

## 5. SLO derivation

SLOs were measured, not chosen by inspecting policy performance.

1. Measure the `3M` reference fleet's capacity independently on seeds 101, 211,
   307, 401, and 503.
2. Run that fleet at 60%, 70%, and 80% of its mean measured capacity.
3. For each seed, workload, class, and latency dimension, compute p90.
4. Set the base SLO to the mean of the five seed-level p90 values at 70%.
5. Use the 60% and 80% values only as diagnostics.

| workload | 60% target run | 70% target run | 80% target run |
|---|---:|---:|---:|
| synthetic | 1.225 req/s | 1.429 req/s | 1.633 req/s |
| RAG | 2.824 req/s | 3.295 req/s | 3.766 req/s |
| shared-prefix | 63.366 req/s | 73.927 req/s | 84.488 req/s |

Each target run used 1,200 synthetic or 6,000 shared-prefix requests. RAG used
1,500 requests at 60% and 80%, and 3,000 at 70%. The 70% RAG count was doubled
after the initial standard-class TTFT estimate missed the registered reliability
gate; this happened before policy evaluation.

### 5.1 Final targets and reliability

The table reports the target, its t-based 95% interval across five seed-level
p90s, the coefficient of variation, and the total contributing request count.

| workload/class | dimension | target (ms) | 95% CI (ms) | CV | requests |
|---|---|---:|---:|---:|---:|
| synthetic/batch | TTFT | 78.904 | [76.784, 81.024] | .022 | 6,000 |
| synthetic/batch | ITL | 33.011 | [31.654, 34.368] | .033 | 6,000 |
| synthetic/batch | E2E | 214,979.959 | [206,622.972, 223,336.947] | .031 | 6,000 |
| RAG/standard | TTFT | 1,437.324 | [1,230.708, 1,643.940] | .116 | 11,663 |
| RAG/standard | ITL | 54.143 | [49.891, 58.395] | .063 | 11,663 |
| RAG/standard | E2E | 37,445.945 | [35,695.791, 39,196.100] | .038 | 11,663 |
| RAG/batch | TTFT | 3,158.586 | [2,826.581, 3,490.591] | .085 | 3,337 |
| RAG/batch | ITL | 58.262 | [54.210, 62.313] | .056 | 3,337 |
| RAG/batch | E2E | 42,580.983 | [39,566.662, 45,595.304] | .057 | 3,337 |
| shared/standard | TTFT | 61.086 | [60.952, 61.219] | .002 | 30,000 |
| shared/standard | ITL | 23.942 | [23.621, 24.262] | .011 | 30,000 |
| shared/standard | E2E | 5,967.822 | [5,888.322, 6,047.321] | .011 | 30,000 |

A target was accepted only when:

- at least 500 class-specific requests contributed,
- CV was at most 0.15, and
- the relative 95% CI half-width was at most 0.20.

All twelve target-defining values passed. The values were passed to BLIS rounded
to three decimals; offline sensitivity rescoring uses exactly the same rounded
values.

The 60% and 80% RAG TTFT diagnostics did not always pass the reliability gates.
They are not used as SLOs, but show that RAG tail TTFT is highly load-sensitive.

## 6. Metric

The performance value called `goodput` in the campaign CSV is SLO attainment:

```text
goodput_fraction =
    requests meeting TTFT AND ITL AND E2E targets
    ------------------------------------------------
                     injected requests
```

A request is good only if all three class-specific targets are met. Drops,
timeouts, and unfinished requests remain in the denominator and cannot be good.
The final experiments had none.

The reported value is a fraction from 0 to 1, not requests per second. Since
every policy in a condition receives the same offered rate, comparing fractions
also compares SLO-good requests per second within that condition.

## 7. Policies

| policy | behavior |
|---|---|
| `always` | Every request prefills on `instance_0`, transfers KV, and decodes on a mixed instance. |
| `never` | Every request prefills and decodes locally on a mixed instance; the dedicated prefill instance is idle. |
| `least_ttft_joint` | Enumerates every local and disaggregated decode/prefill pair and minimizes the arriving request's predicted TTFT. It ignores Lyapunov deficits and co-resident externality. |
| `dpp_joint` | Joint drift-plus-penalty policy using congestion/work drift, TTFT/ITL virtual deficits, and transfer penalty. |
| `kairos` | Load-aware prefill deflection. It places prefill on a decode node only when a TBT-safe chunk schedule beats the dedicated prefill path. |
| `dpvar` | Joint deployable drift-plus-VaR: normalized congestion plus predicted loss of co-resident SLO utility, minus the arriving request's predicted goodput. |
| `universal_phi` | One deterministic fixed disaggregation share used for every workload and rate. |
| `universal_threshold` | Disaggregate when uncached prompt tokens on the selected decode instance exceed one universal threshold. |
| `tuned_phi` | Fixed share selected separately for each workload/rate on calibration seed 42. An offline condition-aware yardstick. |
| `tuned_threshold` | Prefix threshold selected separately for each workload/rate on seed 42. An offline condition-aware yardstick. |

The threshold rule is:

```text
uncached_tokens = prompt_tokens - cached_blocks * 16
disaggregate if uncached_tokens > threshold
```

## 8. Parameter calibration

On seed 42, the campaign ran:

- fixed share `phi`: 0, .2, .4, .6, .8, 1;
- prefix threshold: 0, 512, 2,048, 8,192, 32,768, 100,000 tokens;
- Kairos `beta`: .25, .5, 1.

There were 15 candidates for each of 9 conditions, or 135 calibration runs.

For condition-tuned parameters, the highest seed-42 goodput won, with the smaller
parameter breaking ties. For universal parameters, the selection maximized the
worst seed-42 goodput across all nine conditions, then the mean, then preferred
the smaller parameter.

Universal parameters used in evaluation:

| parameter | selected value |
|---|---:|
| fixed share | 1.0 |
| prefix threshold | 0 tokens |
| Kairos beta | 0.5 |

The universal fixed share therefore has the same P/D-placement semantics as
`always`. The fixed-share plan assigns
decode instances by deterministic alternation, while `always` uses queue-depth
decode routing, so their realized performance need not be identical. Threshold
zero disaggregates every request with any uncached prompt tokens and retains
queue-depth decode routing, making it nearly identical to `always`. Their
separate inclusion verifies the calibration result and both static
implementations.

Condition-specific calibration results:

| condition | tuned phi | tuned threshold | condition-best Kairos beta |
|---|---:|---:|---:|
| synthetic low | 0.0 | 0 | .25 |
| synthetic medium | 0.6 | 2,048 | .25 |
| synthetic high | 0.8 | 0 | .25 |
| RAG low | 0.6 | 32,768 | 1.0 |
| RAG medium | 0.6 | 8,192 | 1.0 |
| RAG high | 0.6 | 8,192 | 1.0 |
| shared low | 1.0 | 0 | .25 |
| shared medium | 1.0 | 0 | .25 |
| shared high | 1.0 | 0 | .25 |

Only the tuned share and threshold were evaluated as condition-aware
yardsticks. Kairos was evaluated with the universal beta .5.

## 9. Evaluation procedure

Each of the ten policies ran at every workload/rate condition on seed 42 and the
four held-out seeds:

```text
3 workloads * 3 rates * 10 policies * 5 seeds = 450 runs
```

Seed 42 was retained as a reproducibility check but excluded from held-out means
and confidence intervals. Reported confidence intervals use paired differences
across seeds 7, 123, 2024, and 9001 with a t critical value for three degrees of
freedom.

The per-condition reference is the highest held-out mean among all ten policies,
including the condition-tuned yardsticks. Policy regret is reference minus
held-out mean. Worst regret is the maximum over the nine conditions.

## 10. Held-out performance

Every cell is mean goodput fraction over seeds 7, 123, 2024, and 9001.

| policy | syn-L | syn-M | syn-H | RAG-L | RAG-M | RAG-H | shared-L | shared-M | shared-H | worst regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `always` | .764 | .189 | .158 | .908 | .723 | .674 | .990 | .986 | .330 | .210 |
| `never` | .772 | .191 | .158 | .933 | .567 | .451 | .873 | .048 | .030 | .942 |
| `least_ttft_joint` | .660 | .211 | .165 | .983 | .687 | .596 | .226 | .133 | .121 | .857 |
| `dpp_joint` | .406 | .158 | .132 | .844 | .548 | .481 | .608 | .062 | .036 | .928 |
| `kairos` | .764 | .189 | .158 | .959 | .784 | .712 | .990 | .986 | .330 | .173 |
| `dpvar` | .675 | .252 | .207 | .992 | .907 | .837 | .973 | .778 | .365 | .212 |
| `universal_phi` | .780 | .187 | .155 | .911 | .727 | .670 | .990 | .990 | .334 | .214 |
| `universal_threshold` | .764 | .189 | .158 | .908 | .723 | .674 | .990 | .986 | .330 | .210 |
| `tuned_phi` | .808 | .188 | .155 | .984 | .870 | .792 | .990 | .990 | .334 | .092 |
| `tuned_threshold` | .764 | .194 | .158 | .987 | .916 | .885 | .990 | .986 | .330 | .058 |

The low absolute synthetic goodput at medium and high rate is not a throughput
failure: the synthetic job has very long outputs and a tight conjunction of
TTFT, ITL, and E2E targets. All requests still completed.

### 10.1 dpVaR against the best universal static policy

| condition | static comparator | paired difference | 95% CI |
|---|---|---:|---:|
| synthetic low | `universal_phi` | -.104 | [-.306, .098] |
| synthetic medium | `never` | +.061 | [.011, .111] |
| synthetic high | `never` | +.049 | [-.017, .116] |
| RAG low | `never` | +.059 | [.005, .113] |
| RAG medium | `universal_phi` | +.180 | [.139, .222] |
| RAG high | `always` | +.163 | [.129, .197] |

dpVaR has significant wins on synthetic medium, RAG low, RAG medium, and RAG
high. It nevertheless fails the robust-regret criteria because:

- it trails the static control at synthetic low; and
- at shared medium, its held-out mean is .778 versus .990 for
  `universal_phi`.

The shared-medium loss is seed-sensitive. dpVaR gets .283 goodput and realizes
phi=.617 on seed 2024, while the other held-out seeds realize about phi=.91 and
get .935-.951 goodput.

## 11. Staged-load diagnostic

Eight deployable policies ran on the two-stage shared-prefix workload; the two
condition-tuned yardsticks were excluded. Held-out means were:

| policy | staged goodput |
|---|---:|
| dpVaR | .628 |
| always | .578 |
| universal fixed share | .527 |

The paired dpVaR-minus-`always` difference is +.050 with 95% CI
[-.120, .220]. It is inconclusive.

## 12. SLO sensitivity

Static policies were rescored from their request traces at .8x and 1.2x every
target. dpVaR was rerun because its decisions depend on the SLO values.

| target multiplier | registered verdict | dpVaR worst regret |
|---:|---|---:|
| .8 | pass | .041 |
| 1.0 | fail | .212 |
| 1.2 | fail | .242 |

The conclusion is target-sensitive. The base targets are statistically stable
at the registered 70% reference point, but changing their strictness changes the
policy conclusion.

## 13. Registered decision rule

The method claim passed only if all three conditions held:

1. dpVaR worst regret was at least .05 below every universal static policy.
2. dpVaR regret was at most .10 in every condition.
3. At least one variable-prompt condition had a paired dpVaR advantage above
   .02 whose 95% CI excluded zero.

Results: criterion 1 failed, criterion 2 failed, and criterion 3 passed.

## 14. Run counts and validity gates

| phase | runs | purpose |
|---|---:|---|
| workload probe | 3 | confirm prompt variability and prefix reuse |
| capacity | 33 | six fixed shares plus five 3M reference seeds per workload |
| target derivation | 45 | three reference loads, five seeds, three workloads |
| calibration | 135 | share, threshold, and Kairos parameter selection |
| evaluation | 450 | ten policies, nine conditions, five seeds |
| sensitivity | 72 | dpVaR at two target multipliers, nine conditions, four seeds |
| staged load | 40 | eight policies and five seeds |
| **total** | **778** | |

Every target, calibration, evaluation, sensitivity, and staged run required:

- request conservation;
- zero unservable drops;
- zero timed-out requests;
- zero requests left queued or running;
- zero length-capped requests;
- realized arrival rate within 10% of the requested rate; and
- fixed-share realization within one request of the requested share.

The staged run is exempt from the single-rate check because it intentionally has
two rates. Capacity probes may drop requests because they intentionally overload
the system; no other phase may do so.

All non-capacity runs passed.

## 15. Important limitations

1. This is simulation evidence using simulator-fitted coefficients, not a
   physical multi-GPU deployment.
2. It covers one model, one homogeneous `1P2M` topology, and one transfer setup.
3. There are four held-out policy seeds. Several intervals, especially synthetic
   low and staged load, remain wide.
4. The SLOs are derived from a `3M` reference fleet rather than an external
   production contract. They are reliable at the selected reference point, but
   the policy verdict is sensitive to target scaling.
5. Synthetic output and RAG bimodality require documented approximations because
   BLIS lacks native uniform and bimodal distributions.
6. RAG high was reduced after a zero-drop safety preflight on seeds 7 and 9001.
   The adjustment used validity, not goodput, but those seeds later appear in the
   held-out set and this must remain disclosed.
7. The capacity number is the best point on a six-value fixed-share grid, not a
   global optimization over all possible stateful policies.
8. The shared-medium seed divergence identifies instability but does not yet
   establish its causal mechanism.

## 16. Reproduction files

- Protocol: `campaigns/edpp-study/DECISIVE-PROTOCOL.md`
- Runner: `campaigns/edpp-study/run_decisive_campaign.py`
- Capacity: `campaigns/edpp-study/out/decisive/capacity.json`
- SLO targets: `campaigns/edpp-study/out/decisive/targets.json`
- SLO reliability: `campaigns/edpp-study/out/decisive/target_reliability.json`
- Selected parameters: `campaigns/edpp-study/out/decisive/selection.json`
- Raw evaluation index: `campaigns/edpp-study/out/decisive/evaluate.csv`
- Evaluation summary: `campaigns/edpp-study/out/decisive/evaluation_summary.csv`
- Decision: `campaigns/edpp-study/out/decisive/decision.json`
- Human-readable decision: `campaigns/edpp-study/out/decisive/DECISION.md`
