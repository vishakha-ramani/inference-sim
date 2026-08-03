# Decisive EDPP campaign

## Question

Does the current deployable drift-plus-VaR rule earn its complexity when requests
carry observable routing-time differences?

The earlier headline experiment cannot answer this question. It used a
concurrency cap of 16, constant prompt lengths, no prefix reuse, and omitted
strong static placement controls. This campaign changes the workload and the
controls without changing the rule.

## Workloads

The campaign uses three workloads already distributed with BLIS.

| name | source | routing-time signal |
|---|---|---|
| `synth` | `inference-perf-batch-synthetic-data-generation.yaml` | heavy-tailed prompt length and a 2,000-token shared prefix |
| `rag` | `inference-perf-batch-summarization-rag.yaml` | bimodal prompt length and a 500-token shared prefix |
| `shared` | `examples/inference-perf-shared-prefix.yaml` | nine prefix groups and a low-to-high load transition |

The first two workloads retain their catalog-derived token distributions. The
third retains its prefix groups and token sizes, but shortens each load stage so
that repeated policy comparisons are practical.

## Fixed experimental choices

- Model: `meta-llama/llama-3.3-70b-instruct`
- Coefficients: `scripts/calibration/coeffs-llama70b-h100-tp4.json`
- Policy topology: one dedicated prefill instance and two mixed instances
- Concurrency cap: 256
- Decode routing for decomposed policies: `queue-depth:1`
- Cache signal delay: the simulator default, 50 ms
- Calibration seed: 42
- Held-out seeds: 7, 123, 2024, and 9001
- Arrival process: the source workload's process
- Policy-run floor: 800 synthetic, 1,000 RAG, and 4,000 shared-prefix requests

The rule is frozen before the campaign. No coefficient, weight, estimator, or
normalization setting may be changed after a held-out result is read.

## Capacity and operating points

For each workload, a saturated fixed-share sweep estimates service capacity.
The estimate is the central completion rate between the 10th and 90th
percentiles of completion time. This excludes fill and drain transients, which
distort `responses_per_sec` on workloads with long outputs. These intentional
overload probes may reject arrivals after a finite KV path saturates; rejected
requests are excluded from the completion-rate estimate and the probe is never
reported as a goodput result. Every operating-point run below retains the
zero-drop gate.

The overload sweep uses 1,200 requests for each long-output or long-context
workload and 6,000 for the compact shared-prefix workload. These counts leave
hundreds of completions in each central throughput interval while avoiding an
unnecessary quadratic event-queue cost from a longer overloaded trace.

The policy comparison uses three offered rates:

1. `low`: 60% of the lower endpoint capacity, so both `always` and `never` are
   live competitors.
2. `medium`: 85% of the lower endpoint capacity.
3. `high`: the larger of 95% of the lower endpoint capacity and 70% of the best
   capacity found by the fixed-share family. This keeps the three rates ordered
   when the fixed-share capacity curve is nearly flat.

RAG is the exception: its `high` point is 90% of the lower endpoint capacity.
The general formula caused decode-KV rejections in two held-out
`least_ttft_joint` runs; a two-seed safety probe at the revised point had zero
rejections. This adjustment was made solely to satisfy the registered
zero-drop gate and applies to every policy.

The shared-prefix stress test places its low and high rates in consecutive load
stages.

## SLO targets

Targets come from a separate three-mixed-instance fleet that makes no P/D
placement decision. Capacity and target derivation use seeds 101, 211, 307, 401,
and 503, which are disjoint from policy calibration and evaluation.

The reference runs at 60%, 70%, and 80% of its measured capacity. The
target-defining 70% runs use 1,200 synthetic, 3,000 RAG, or 6,000 shared-prefix
requests per seed. The neighboring RAG diagnostics use 1,500 requests per seed.
The base target is the mean of the five per-seed p90 values at 70% utilization.
The 60% and 80% runs show how much the target depends on the chosen operating
point.
For every target the campaign reports the seed standard deviation, 95%
confidence interval, and sample count. A target is accepted only when:

- at least 500 class-specific requests contribute,
- the coefficient of variation across seed p90 values is at most 0.15, and
- the 95% confidence-interval half-width is at most 20% of the target.

Targets are class-specific. In the RAG workload, short vector-QA requests are
`standard` and long document reads are `batch`; combining them under one target
would make the bimodal workload's SLO uninterpretable. Synthetic generation
remains `batch`, and the shared-prefix workload remains `standard`.

The main result uses these targets. A sensitivity pass evaluates the decisive
arms at 0.8 and 1.2 times each target. Static policies are rescored from their
request-level traces; `dpvar` is rerun because its placement decision depends
on the target. A conclusion that changes across this band is reported as
target-sensitive, not as a routing result.

## Policies

The campaign reports the following separately.

Deployable policies:

- `always`
- `never`
- `least_ttft_joint`
- `dpp_joint`
- `kairos`, with beta selected on seed 42
- `dpvar`, the current paper rule
- `universal_phi`, one fixed disaggregation share selected on seed 42 across
  every calibration condition
- `universal_threshold`, one cache-aware uncached-token threshold selected on
  seed 42 across every calibration condition

Offline yardsticks:

- `tuned_phi`, the best fixed share for each workload and rate on seed 42
- `tuned_threshold`, the best uncached-token threshold for each workload and
  rate on seed 42

The offline yardsticks are not described as deployable policies. They measure
how much goodput remains available to a simple input-conditioned assignment.

## Validity gates

Every SLO-reference, calibration, and policy-evaluation run must satisfy all of
the following:

- request conservation
- zero unservable drops
- zero timed-out requests
- zero requests still queued or running
- zero length-capped requests
- achieved arrival rate within 10% of the configured rate
- realized fixed share within one request of the requested share

The probe also verifies nonconstant prompt lengths and nonzero prefix reuse
before the main campaign begins.

## Decision rule

The method-paper claim passes only if all three conditions hold on held-out
seeds.

1. `dpvar` has worst-case regret at least 0.05 lower than every universal static
   baseline.
2. `dpvar` has regret at most 0.10 in every condition.
3. In at least one workload with variable prompts, the paired goodput advantage
   over the best universal static baseline exceeds 0.02 and its 95% confidence
   interval excludes zero.

Failure does not invalidate the capacity model. It means the current adaptive
routing rule does not support the manuscript's method claim. The defensible
paper direction would then be a characterization of why snapshot-based routing
mis-selects placements, provided that result survives the catalog workloads.
