# EDPP working plan

Last updated 2026-07-29 (goodput ladder and realized-share diagnostic added). Supersedes the ordering in `TODO.md`, which predates the
capacity-ceiling work.

The ordering principle Vishakha set: **establish the methodology first, and do the
policy evaluation last.** Every number in the current Policy Evaluation section
rests on a superseded protocol, and re-running it before the protocol is settled
would waste the runs.

## Notation

One symbol per phenomenon. **phi** is the share of arrivals that disaggregate,
meaning their prefill runs on the dedicated prefill instance and their decode on a
mixed instance. The remaining `1 - phi` are served whole on a mixed instance.
`phi*` is the share that maximizes capacity. The paper writes it `\pdshare` and
renders it as phi, and the harnesses take it as `--phi`.

The harnesses called it `f` until 2026-07-29. That collided with the pool-split
fractions `f_i` in the capacity derivation and read as a different quantity from
the paper's, so it was swept. Log files written before the rename label the static
arm `plan@phi*`, and `analyze/goodput_ladder.py` accepts both spellings.

**theta is a different quantity and is retired.** It was the fraction of a mixed
instance's TIME given to prefill, which is what the superseded `pd_best` bound
used. The two move in opposite directions as prompts lengthen, so keeping both
names would make the old numbers look like a contradiction rather than a different
parameterization.

| cell | theta* (retired) | phi* |
|---|---|---|
| decode | 0.00 | 1.00 |
| mixed | 0.36 | 0.485 |
| prefill_lean | 0.703 | 0.387 |
| prefill_bound | 0.920 | 0.337 |

Both rows for `decode` describe the same policy, `always`. theta = 0 means mixed
instances do no prefill and phi = 1 means every request splits.

Two fairness constraints apply to every comparison from here on.

- `never@3M` is not a baseline. The operator fixes the disaggregated topology and
  the paper asks how to route within it. `never@3M` survives only as the fleet the
  SLO targets are derived on, where it grades no policy.
- `never@1P2D` is not a baseline either. It leaves the dedicated prefill instance
  completely idle, measured at a mean batch occupancy of zero, so it competes with
  two instances where every other policy uses three.

---

## Done

**1. Measure the capacity ceiling of a 1P2D fleet, and locate the optimal split.**

`pd_best` in `gamma_cap.py` was never a ceiling. It gives each mixed instance a
time share for prefill and counts the per-iteration baseline once for each stage,
which under-prices every mixture that collocates. The implementable family is
parameterized by `f`, the share of arrivals that disaggregate, and its capacity is

    C(f) = min( 1/(f*t_P) , |M| / (f*t_D + (1-f)*t_coll) )

Built `mix_cap.py` (the model), `make_pd_plan.py` (plan generator), and
`repro_mix_ceiling.sh` (the sweep). `--pd-plan` forces a per-request placement, so
a fixed-`f` plan realizes one point of the curve. Endpoints reproduce
`--pd-decider always` and `never` to 0.02 percent, which is the harness self-check.

Results, H100 coefficients, seed 42, gated on zero shed / zero left over /
genuinely saturated:

| | prefill_lean (8192/64) | prefill_bound (16000/16) |
|---|---|---|
| measured ceiling | **19.75** at f = 0.39 | **11.64** at f = 0.34 |
| predicted phi\* | 0.387 | 0.337 |
| `always` | 8.2765 (0.42 of ceiling) | 4.1454 (0.36) |
| `never@1P2D` | 13.5430 (0.69) | 7.9062 (0.68) |
| dpVaR (from the ladder) | >= 17.687 (>= 0.895) | 9.382 (0.81) |

Identity, verified analytically and in measurement to four decimals on all
four cells: **phi\* = C_always / C_mix(phi\*)**, where `C_always` is the capacity
of the `always` policy and `C_mix(phi\*)` is the peak of the curve. It holds by
derivation wherever the dedicated prefill instance binds under full separation,
and trivially on `decode`, where phi\* sits at the boundary.

Model accuracy splits by which instance binds. Prefill-bound: within about one
percent over ten points. Collocation-bound: five to ten percent under over
thirteen points, always on the low side.

**2. Paper correctness pass on everything except Policy Evaluation.**

Committed at `infocom` `b036b5f` and PUSHED to Overleaf (verified in sync). Replaced
the capacity subsections, connected the Slater condition to the capacity region,
fixed 14 broken `\paren*` delimiter calls, and withdrew two unsupported intro
claims. `main.pdf` builds clean at 24 pages with no undefined references.

---

**3. Goodput ladder below `always`'s ceiling.** DONE. `repro_goodput_ladder.sh`
plus `analyze/goodput_ladder.py`. Three arms, no `never` in any form. Rates as
fractions of the measured ceilings, with three points per cell below `always`'s
own capacity. 60 runs, all gated clean.

Two results, pointing opposite ways.

*The comparison is now fair, and the rule wins where it counts.* At lean rate 4
all three arms serve 4.03 req/s and `always` scores 0.493 against dpVaR's 0.804,
a 1.6x goodput advantage at matched throughput and at 48 percent of `always`'s own
capacity. Replicated on prefill_bound at rate 2. `always` fails purely on
first-token time. Its ITL attainment is exactly 1.000 at every rate on both cells,
because it never collocates, so mixed instances only decode and batches stay small.

*A fixed share beats the rule on all ten rate points*, by 0.063 to 0.117 against
seed spreads of 0.007 to 0.083.

| cell | rate | always | dpvar | plan@phi* |
|---|---|---|---|---|
| lean | 4 | 0.493 | 0.804 | **0.867** |
| lean | 6 | 0.253 | 0.726 | **0.803** |
| lean | 8 | 0.033 | 0.609 | **0.707** |
| bound | 2 | 0.508 | 0.801 | **0.876** |
| bound | 3 | 0.230 | 0.708 | **0.807** |
| bound | 4 | 0.031 | 0.646 | **0.718** |

**4. Why the fixed share wins.** DONE. `repro_realized_share.sh` reads
`Disaggregated Requests` off stdout. Controls pass exactly, `always` at 1.0000 and
the plan at phi\*.

**dpVaR systematically under-disaggregates.** Realized share against phi\*:

| cell | rate | dpvar share | phi* |
|---|---|---|---|
| lean | 4 | **0.0850** | 0.39 |
| lean | 6 | 0.2189 | 0.39 |
| lean | 8 | 0.3533 | 0.39 |
| bound | 2 | **0.0367** | 0.34 |
| bound | 3 | 0.0900 | 0.34 |
| bound | 4 | 0.2083 | 0.34 |

It splits 8.5 percent of requests where the optimum is 39, leaving the dedicated
prefill instance almost idle. The share rises with load and converges toward phi\*
only from below, and only as congestion forces it.

Mechanism, consistent with what `infocom/main.tex` Model Validation already
documents. Disaggregating adds a KV transfer to the first-token path, and for an
8192 or 16000 token prompt that cost is large and certain. The queueing it avoids
is neither, because the roll-forward reads the current batch and cannot see the
standing queue that collocating everything is about to build. The rule pays a
visible cost to avoid an invisible one and declines. Snapshot blindness showing up
as a routing bias rather than an estimator error.

## Next

**5. f-sweep for GOODPUT.** The question that decides whether item 4 breaks the
paper or sharpens it. phi\* was derived from the capacity model and knows nothing
about the SLO targets. Sweep the share against goodput at fixed sub-ceiling rates
and ask two things.

- Does the goodput-optimal share differ from the throughput-optimal phi\*?
- Does it move with load, and with workload?

If it moves, no fixed share is deployable and the rule's premise holds. It then
needs to find the right share, and item 4 says how it currently fails to. If one
share works everywhere, the paper needs a different argument.

**6. Style pass on Problem Formulation.** Real violations of
`infocom/writing-preferences.md`, all pre-existing, all in the theory derivation:

- 5 em-dashes joining clauses (banned outright), in the value-at-risk derivation
- 8 uses of "charge" or "price" as the accounting verb (§I contribution 3, §IV)
- about 4 antithesis "rather than" frames

Deliberately not bulk-fixed. The preferences say recast rather than
search-replace, and never restyle a sentence whose logic is still being corrected.
These sit in the derivation, so careless recasting risks meaning. Do it as one
careful reviewable pass.

**7. Then, and only then, Policy Evaluation.** Everything in it rests on the
concurrency-16 protocol, and its heterogeneous regime used the fabricated A100
bandwidth. Re-run against the settled protocol.

---

## Open decisions for Vishakha

- **Overleaf.** `infocom` is in sync with `overleaf/main` at `b036b5f`. Nothing
  unpushed. Fetch before any further edit, because a co-author can commit there.
- **`writing-preferences.md` is untracked** while `infocom/CLAUDE.md` instructs
  everyone to follow it. A co-author working on Overleaf does not have it.
- **SLO targets are derived on `never@3M`** (`main.tex`, Policy Evaluation setup).
  It grades no policy, and it is the last place an aggregated fleet appears.
  Alternatives both cost something. Deriving on the 1P2D fleet reintroduces
  self-grading. Asserting targets reintroduces the "asserted not derived" critique.

## Known defects, not blocking

- `gamma_cap.py:105` still loads `coeffs-llama70b-a100crippled-tp4.json` for its
  hetero row, so every heterogeneous capacity it prints is void. The realistic
  file exists at `coeffs-llama70b-a100real-tp4.json` (alpha_D 25.56 ms, 1.54x the
  H100, against the crippled 69.31 ms).
- `sim.CalculateMean` divides by 1000, folding a ticks-to-milliseconds conversion
  into a function whose name says only "mean". Correct for the latency series it
  serves and a trap for anything else. `meanInt` in `sim/metrics.go` exists
  because of it.
- The TraceV2 export's `status` field contradicts the metrics. A run reporting
  `completed_requests: 1000, still_queued: 0, still_running: 0` exported 387 rows
  marked `incomplete` with zero timestamps. Worth an issue.
- Deep overload plus a large request count triggers `dropped_unservable`, which
  silently changes what `responses_per_sec` measures. `repro_mix_ceiling.sh` gates
  on it. Older harnesses do not, though the runs on record are clean.
- Achieved batch is far below the 256 concurrency limit on prefill-heavy cells
  (mixed instances 3 to 24, dedicated prefill instance 1.10). The 1.10 confirms
  the paper's baseline-sharing claim. Substituting the measured mean into the
  decode term overshoots the residual roughly threefold, because the mean runs
  over every iteration including the prefill-chunk steps already accounted for
  separately. Closing this needs the batch conditioned on decode steps, a further
  change to `sim/`.

## Standing caveat

Everything measured in the ceiling work is seed 42. Rates and optima are stable
across offered load and across two run lengths. No error bars anywhere.
