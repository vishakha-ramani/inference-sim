# Public mixed-workload and burst benchmark protocol

## Frozen question

This campaign tests whether the final joint causal-SLO-externality policy adapts
when request shapes change within one trace, coexist concurrently, or arrive in
bursts.  The routing objective is the frozen smooth `TTFT x E2E` value minus
resident causal externality, without a capacity term.  Reported goodput remains
the hard conjunction of TTFT, mean ITL, and E2E.

The public request-shape translations, model, fleet definitions, transfer model,
and per-class SLOs are inherited from the completed public load/static benchmark.
The three classes are renamed so their different targets coexist in one trace:

| semantic workload (BLIS SLO class) | TTFT | mean ITL | E2E |
|---|---:|---:|---:|
| interactive (`critical`) | 1 s | 50 ms | 16 s |
| reasoning (`batch`) | 2 s | 100 ms | 802 s |
| deep research (`standard`) | 10 s | 100 ms | 40 s |

The simulator request timeout is disabled.  This prevents its generic 300-second
default from silently tightening the reasoning E2E target.

## Capacity normalization and profiles

Rates use the frozen per-workload capacity estimates in
`out/public_load_static_benchmark_ttft_rollout_v2/capacity_selection.json`.  They are not
formed by adding raw request rates.  If `C_i` is the measured capacity of class
`i`, a concurrent class assigned normalized capacity share `s_i` is offered at
`rho s_i C_i`.

Four profiles are frozen for both the homogeneous H100 `1P2D` fleet and the
realistic H100-prefill/H100+A100-decode `1P2D` fleet:

1. `sequential_shift`: interactive, then reasoning, then deep research.  Every
   phase is offered at `0.80 C_i` and is sized for 100 expected requests before
   per-client rounding.  There is no drain gap between phases.
2. `concurrent_poisson`: all three classes coexist with equal one-third capacity
   shares at total normalized load `rho=0.80`.
3. `concurrent_gamma_cv3`: the same mean rates and class shares as the concurrent
   Poisson trace, but every class uses Gamma inter-arrivals with CV 3.
4. `short_spike`: a Poisson mixture carries normalized load 0.60 for the whole
   trace and an additional normalized load 1.00 during the central 20%.  Thus
   the mean normalized load is 0.80 and the central peak is 1.60.  Base and
   spike traffic both use equal one-third class capacity shares.

The steady concurrent traces are sized for 720 expected requests before
per-client rounding.  The spike trace is sized for 3,000 so the rare reasoning
cohort has nonzero traffic in both its full-trace base window and shorter burst
window without changing the source tenant/prefix populations.  Absolute
per-window rates and lifecycle windows are encoded in the native workload
schema.  Window generation preserves sampled arrival CV while rescaling to the
registered mean rate.  A 3,600-second post-arrival drain horizon is provided;
no requests are generated in that drain interval.

## Static calibration

Development seeds are `42` and `123`.  At each fleet/profile condition, test a
bounded fixed joint-plan grid:

- `phi in {0, 0.5, 1}` for the remote-prefill share;
- homogeneous fleet: `psi=0.5` for the second-decoder share;
- heterogeneous fleet: `psi in {0, 0.5, 1}`.

Select the hard-valid, zero-drop plan with greatest mean composite goodput,
breaking ties by lower seed standard deviation, lower `phi`, then lower `psi`.
Freeze `static_selection.json` before confirmation.  This is a condition-tuned
static plan, not an oracle or upper bound.

## Held-out comparison

Confirmation seeds are `2000000099`, `2000000123`, `2000000141`, and
`2000000159`.  As of protocol freeze, none appeared in campaign sources or
artifacts.  Do not inspect confirmation results until static selection is
written.

Compare:

1. joint causal SLO externality without capacity prices (`V=8`);
2. joint least projected TTFT;
3. paper Kairos (`alpha=1.3`);
4. a class-aware llm-d prefix-threshold policy using
   `precise-prefix-cache:2,queue-depth:1` on both pools and frozen thresholds
   `critical=1024,batch=16,standard=16`; and
5. the frozen condition-tuned static plan.

There are `4 profiles x 2 fleets x 4 seeds x 5 policies = 160` confirmation
runs.  Only the first four deployable policies enter minimax ranking.  Report
overall and per-class goodput, remote fraction, all held-out values, paired
deltas, and pre/during/post attainment for the spike profile.

The causal and least-TTFT arms use the validated FCFS scheduler-step TTFT
rollout with live arrival-time scheduler snapshots. Kairos retains its own
published estimator.

## Validity

Every run must satisfy:

```text
completed + dropped = injected
still_queued = 0
still_running = 0
timed_out = 0
length_capped = 0
```

The generated class counts must all be nonzero.  Every focal-policy candidate
trace must select an exact score argmin.  An incomplete grid, missing seed,
confirmation-informed static selection, invalid terminal accounting, or trace
argmin mismatch invalidates the corresponding result.

This is a discrete-event simulation of independent public request shapes.  It
does not reproduce upstream multi-turn semantics and does not estimate a global
oracle gap.
