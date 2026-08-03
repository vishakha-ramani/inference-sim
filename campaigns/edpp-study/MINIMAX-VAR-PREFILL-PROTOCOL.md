# Minimax var-prefill campaign protocol

Date: 2026-07-29

## Claim

The primary claim is not that `var-prefill` wins every workload. It is that one
globally calibrated, deployable policy has the smallest worst-condition
goodput regret among the evaluated deployable policies.

For condition `c = (workload, normalized load)`, the calibration static
yardstick is:

```text
phi*(c) = argmax_phi G_static(phi, c), phi in {0.0, 0.1, ..., 1.0}
G_static*(c) = G_static(phi*(c), c)
```

The calibration regret and selection rule are:

```text
R(lambda, c) = G_static*(c) - G_var-prefill(lambda, c)
lambda* = argmin_lambda max_c R(lambda, c)
```

Ties use mean regret, then the smaller parameter.

Held-out reporting uses both:

1. regret to the frozen condition-specific static yardstick; and
2. regret to the best held-out mean among all frozen evaluated policies.

The realized fraction distance `abs(phi_policy - phi*(c))` is a mechanism
diagnostic, not the primary outcome.

## Isolation

- Calibration seed: `42`.
- Held-out seeds: `7, 123, 2024, 9001`.
- Held-out outcomes do not select rates, static fractions, lambda, or any
  comparator parameter.
- SLO targets are reused from `out/decisive/targets.json`. They were derived
  from the non-PD 3M reference fleet, so the PD first-token correction does not
  alter their derivation.

## Workloads and topology

Workloads:

- synthetic batch;
- mixed RAG;
- shared prefix.

Every policy run uses the same 1P2D topology, queue-depth decode routing, and
normal prefill routing. The routing-preserving static plan controls only P/D:

```text
python3 make_pd_plan.py --preserve-routing
```

An empty `decode_instance` preserves queue-depth routing. `prefill_instance=auto`
means disaggregate and preserve normal prefill-pool routing.

## Fleet ceiling and load levels

For each workload, seed 42 runs every static fraction at a deliberately
saturating offered rate. Central completion rate (10%-90% completion window)
defines the capacity estimate. The highest-capacity fraction defines the
workload's static fleet ceiling. Overload probes may drop requests, but a
fraction is eligible to define the ceiling only when it completes every
injected request with zero drops.

Policy conditions use:

```text
low       = 0.60 * static ceiling
medium    = 0.80 * static ceiling
near-high = 0.95 * static ceiling
```

The 0.95 point avoids making finite-run fill/drain noise the definition of
policy failure while remaining a near-ceiling test.

## Grids

```text
phi      = {0.0, 0.1, ..., 1.0}
lambda_p = {0.25, 0.5, 1.0, 2.0, 4.0}
```

Static capacity and condition selection use only routing-preserving plans.

## Frozen held-out policy panel

- `always`
- `never`
- reduced `least-ttft`
- joint `least-ttft`
- joint DPP
- Kairos with the previously calibrated global beta `0.5`
- original deployable joint dpVaR
- one universal routing-preserving static fraction, selected by calibration
  minimax regret
- simplified reduced `var-prefill` with the newly selected global lambda
- condition-specific routing-preserving static fraction (offline yardstick)

## Validity

Every non-capacity run requires:

- request conservation;
- zero timeouts;
- zero still-running or still-queued requests;
- zero length-capped requests;
- offered-rate error at most 10%;
- realized fixed share within one request of its plan, after allowing a
  request dropped before P/D commitment to explain one missing planned action.

A static fraction is eligible to define either the condition oracle or the
universal static comparator only if it additionally completes every injected
request with zero unservable drops.

For an evaluated adaptive policy, an unservable drop is a measurable policy
failure rather than a reason to discard the run. Goodput uses all injected
arrivals as its denominator, so each dropped request contributes zero goodput.
Drop fraction is reported explicitly. This prevents a policy that overloads a
bad route from escaping the regret comparison through an invalid-run filter.

## Reporting

For each condition and policy:

- held-out mean and 95% t interval;
- TTFT, ITL, and E2E attainment;
- unservable drop fraction;
- realized disaggregation fraction;
- regret to the frozen static yardstick;
- regret to the best frozen policy.

For each deployable policy:

- worst-condition regret;
- condition attaining that worst regret;
- maximum and mean fraction distance to `phi*(c)`.

## Frozen confirmation extension

Preregistered: 2026-07-30, after the four-seed result and before any
confirmation run.

The first panel left the paired `dpvar - var_prefill` difference unresolved.
Eight new seeds are therefore added without changing workloads, rates, SLOs,
static fractions, policy parameters, lambda, or analysis:

```text
13, 29, 61, 251, 509, 1021, 4093, 8191
```

They are disjoint from the calibration seed, the original held-out seeds, and
the target-derivation seeds. All ten frozen arms are repeated in all nine
conditions. Confirmation-only results are primary for replication; the
combined twelve-seed analysis is a secondary precision estimate. Neither
result may select a new lambda or alter the rule.

The mechanism diagnosis uses decision traces only for the already identified
shared-prefix near-high failure. It reports the observed VaR benefit,
prefill-stability charge, decision margin, skip reasons, and their relationship
to the realized fraction. It does not select a new parameter.
