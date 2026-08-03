# Decisive EDPP campaign report

## Executive decision

**The current dpVaR method claim fails the registered test.**

The campaign asked whether dpVaR earns its complexity when requests have
observable routing-time differences. It used public inference-perf workload
specifications, prefix reuse, variable prompt lengths, a concurrency cap of 256,
strong static controls, independent SLO seeds, and four held-out evaluation
seeds. The decision required all three registered criteria to pass:

| criterion | result |
|---|---|
| dpVaR worst-case regret is at least 0.05 below every universal static policy | fail |
| dpVaR worst-case regret is at most 0.10 | fail |
| dpVaR has a significant win above 0.02 on a variable-prompt workload | pass |

dpVaR's worst-case regret is **0.212**. The best universal static policy's
worst-case regret is **0.210**. The current manuscript therefore cannot claim
that dpVaR is robustly better than simple deployable placement policies.

This result does **not** invalidate the corrected capacity formulation. It tests
the routing rule, not the capacity derivation.

## Experimental design

- Policy topology: one dedicated prefill instance and two mixed instances.
- Reference topology: three mixed instances, used only to derive SLOs.
- Model: `meta-llama/llama-3.3-70b-instruct`.
- Concurrency cap: 256.
- Workloads: inference-perf synthetic generation, inference-perf RAG, and the
  inference-perf shared-prefix staged workload.
- Calibration seed: 42.
- Held-out policy seeds: 7, 123, 2024, and 9001.
- Independent capacity/SLO seeds: 101, 211, 307, 401, and 503.
- Deployable controls: `always`, `never`, `least_ttft_joint`, `dpp_joint`,
  `kairos`, one universal fixed share, and one universal prefix threshold.
- Offline yardsticks: a condition-tuned fixed share and prefix threshold.

The workload probe confirmed variable prompts and prefix reuse:

| workload | prompt range | distinct prompt lengths | cache-hit rate |
|---|---:|---:|---:|
| synthetic | 2,050-17,000 | 302 / 500 | 0.266 |
| RAG | 1,500-80,500 | 687 / 800 | 0.011 |
| shared-prefix | 547 | 1 / 500 | 0.056 |

The shared-prefix workload varies prefix identity and load stage rather than
prompt length. The registered variable-prompt significance test therefore uses
only synthetic and RAG conditions.

## Reliable SLOs

Each target is the mean of five seed-level p90 measurements at 70% of the
independently measured three-mixed-instance capacity. Targets are class-specific
for the bimodal RAG workload.

| workload | class | TTFT (ms) | ITL (ms) | E2E (ms) |
|---|---|---:|---:|---:|
| synthetic | batch | 78.904 | 33.011 | 214,979.959 |
| RAG | standard | 1,437.324 | 54.143 | 37,445.945 |
| RAG | batch | 3,158.586 | 58.262 | 42,580.983 |
| shared-prefix | standard | 61.086 | 23.942 | 5,967.822 |

Every target-defining row passed all registered gates:

- at least 500 class-specific requests,
- seed-level coefficient of variation no greater than 0.15, and
- relative 95% confidence-interval half-width no greater than 0.20.

The RAG target run was increased from 1,500 to 3,000 requests per seed after its
standard-class TTFT initially missed the reliability gate. The final estimate
uses 11,663 standard-class requests, with CV 0.116 and relative interval
half-width 0.144. This change occurred before policy evaluation and is recorded
in the protocol.

The neighboring 60% and 80% RAG TTFT diagnostics remain unstable. They do not
define the target, but they show that tail TTFT grows sharply near saturation.

## Primary results

The table reports held-out means. The reference is the best mean among all
evaluated policies, including the offline yardsticks.

| condition | reference | dpVaR | best universal static | dpVaR regret |
|---|---:|---:|---:|---:|
| synthetic low | 0.808 | 0.675 | 0.780 | 0.133 |
| synthetic medium | 0.252 | 0.252 | 0.191 | 0.000 |
| synthetic high | 0.208 | 0.208 | 0.158 | 0.000 |
| RAG low | 0.992 | 0.992 | 0.933 | 0.000 |
| RAG medium | 0.916 | 0.908 | 0.727 | 0.008 |
| RAG high | 0.885 | 0.838 | 0.675 | 0.047 |
| shared-prefix low | 0.990 | 0.973 | 0.990 | 0.017 |
| shared-prefix medium | 0.990 | 0.779 | 0.990 | 0.212 |
| shared-prefix high | 0.365 | 0.365 | 0.334 | 0.000 |

dpVaR has statistically significant paired wins over the best universal static
policy on synthetic medium, RAG low, RAG medium, and RAG high. The largest are:

- RAG medium: +0.180, 95% CI [0.139, 0.222].
- RAG high: +0.163, 95% CI [0.129, 0.197].

Those wins are real, but they do not satisfy a robustness claim. At
shared-prefix medium load, one held-out seed drops to 0.283 goodput and selects a
0.617 disaggregation share; the other held-out seeds select about 0.91 and reach
0.935-0.951 goodput. This is evidence of policy instability, not yet a causal
explanation of it.

## Staged-load diagnostic

Excluding calibration seed 42, dpVaR averages 0.628 goodput and `always` averages
0.578. The paired difference is +0.050 with 95% CI [-0.120, 0.220]. The interval
is too wide to support either a win or a loss under staged load.

## SLO sensitivity

| target multiplier | verdict | dpVaR worst regret |
|---:|---|---:|
| 0.8 | pass | 0.041 |
| 1.0 | fail | 0.212 |
| 1.2 | fail | 0.242 |

Static policies are rescored from request-level traces. dpVaR is rerun because
its decisions depend on the targets. Offline rescoring now uses the exact
three-decimal target values passed to BLIS; at factor 1.0 it matches the executed
static-policy goodput exactly.

The conclusion is SLO-sensitive. A claim that dpVaR is target-robust would be
unsupported even if only the looser 0.8x targets were reported.

## Integrity audit

- 33 capacity runs completed; overload drops are allowed only in these probes.
- 45 target runs passed the validity gates.
- 135 calibration runs passed the validity gates.
- 450 final policy runs passed the validity gates.
- 72 sensitivity runs passed the validity gates.
- 40 staged-load runs passed the validity gates.
- Every non-capacity run has request conservation, zero unservable drops, zero
  timeouts, zero unfinished requests, zero length caps, and acceptable realized
  arrival rate.
- Calibration seed 42 is excluded from confidence intervals.
- The campaign runner passes Python compilation, Ruff, and `git diff --check`.

## Paper consequence

The manuscript should not retain its current headline max-regret or robust
dominance claims. Shortening the existing argument to nine pages would preserve
a claim that the stronger experiment rejects.

Two directions remain defensible:

1. Design a new routing rule that addresses the shared-prefix instability,
   freeze it before evaluation, and repeat this protocol with new held-out seeds.
2. Reframe the paper around capacity and the empirical limits of snapshot-based
   adaptive placement. That requires a direct failure-mechanism experiment; the
   seed-level share divergence above is suggestive but insufficient by itself.

The first is a new method paper. The second is a characterization paper. The
current dpVaR method paper is not submission-ready.

## Reproduction artifacts

- Complete experiment summary: `campaigns/edpp-study/EXPERIMENT-SUMMARY.md`
- Protocol: `campaigns/edpp-study/DECISIVE-PROTOCOL.md`
- Runner: `campaigns/edpp-study/run_decisive_campaign.py`
- Decision: `campaigns/edpp-study/out/decisive/DECISION.md`
- SLO report: `campaigns/edpp-study/out/decisive/SLO-REPORT.md`
- Sensitivity report: `campaigns/edpp-study/out/decisive/SENSITIVITY.md`
- Machine-readable result: `campaigns/edpp-study/out/decisive/decision.json`
