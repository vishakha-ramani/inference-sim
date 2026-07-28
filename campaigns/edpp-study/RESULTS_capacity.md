# Fleet capacity, Γ^cap, and three defects in the v2.1 protocol

Built 2026-07-28. Supersedes the original `gamma_cap.py`, which reported
Γ^cap = 1.0000 in every regime. Companion harness:
`repro_batchcap_probe.sh`. Reproduce the tables with:

```bash
go build -o blis .
python3 campaigns/edpp-study/gamma_cap.py --batch-cap 16 --tau-itl-ms 67 --packing packed
./campaigns/edpp-study/repro_batchcap_probe.sh          # ~25 min
```

## 1. What fleet capacity is

A fluid feasibility question over assignments, not a simulation. Choose
fractions `x_i` (which instance prefills a request) and `y_i` (which decodes
it); instance *i*'s utilisation is

```
rho_i = lambda * [ x_i * t_P(theta_i) + y_i * t_D(theta_i, B_i) ]
```

and fleet capacity is the largest λ for which some (x, y) holds
`max_i rho_i <= 1`. It is policy-independent because it quantifies over *all*
assignments: any real router induces some long-run empirical assignment and
therefore cannot beat the min-max. Topology enters only as a restriction on the
support of x and y, so "what does 1P2D cost us versus 3 mixed instances" is the
same problem with and without that restriction.

Three points are reported: `coll` (collocation forced, x = y — the `never`
corner), `disag` (roles separated per 1P2D — the `always` corner), and `free`
(unrestricted — the fleet bound).

## 2. Why the old Γ^cap was vacuous

The previous script built work from `CPf, CAttn, C0, C1` only. Those are all
**marginal** coefficients; it omitted **α ≈ 16.61 ms**, the per-iteration
intercept and the largest term in the trained physics model. It then divided by
a lumped `|I| = 3`, pooling one prefill-only instance with two decode-only
instances and letting prefill work borrow capacity the topology forbids. (The
sharper per-pool form was already computed in that script and sent to stderr:
on prefill_lean, lumped ρ = 0.422 versus per-pool ρ_pf = 0.859.)

α is charged per **iteration**, not per request, so it amortises over whatever
shares that iteration — and the two stages differ:

**Decode.** Serving λ_d req/s of *o* output tokens at batch *B* needs
`lambda_d * o / B` iterations/s, so `t_D = o * (alpha / B + dbar)`. This *falls*
with B toward the marginal-only asymptote `o * dbar`. Decode therefore has no
finite work-conservation ceiling in B; capacity is set by whatever caps the
batch. Marginal-only work is exactly the B → ∞ limit, which is why the old
script's decode numbers were an unreachable asymptote rather than a bound.

**Prefill.** A request of a_p tokens needs `ceil(a_p / S)` iterations at chunk
budget S = 2048. When a_p ≥ S those iterations are full and α **cannot** be
amortised. On prefill_lean (8192 in → 4 full chunk-iterations):

| | marginal only | + 4·α_P | measured (`always` plateau) |
|---|---|---|---|
| instance-time / request | 53.7 ms | 120.2 ms | — |
| 1 P-instance capacity | 18.6 req/s | **8.32 req/s** | **8.25 req/s** |

The missing α *is* the entire discrepancy. Amortisation is workload-dependent,
not a constant: 256-token prompts pack ~8 per iteration, so α divides by ~8
there. The script reports `packed` and `unpacked` bounds for this reason.

**Collocated.** One iteration carries both stages, so α is paid once, not
twice: `t_coll = o * alpha / B + o * dbar + Wp_marginal`. This is why `coll` can
exceed `disag` even though it forgoes specialisation.

## 3. The decode "76× gap" is the batch cap

`cmd/root.go:1279` defaults `--max-num-running-reqs` to **256**. Every v2.1
harness overrides it to **16** — `repro_knee_sweep.sh:29,44`,
`repro_grid_v2.sh:61,73`, `repro_decomposition.sh:29,30`,
`repro_ratio_sweep_v2.sh:69`, `repro_topology_v2.sh:63`,
`repro_policy_curves.sh:62,63`, `repro_never_fair.sh:59,60`. The rationale is
documented at `repro_spectrum.sh:19` ("caps concurrency so the system actually
saturates at reachable" rates).

Measured plateau in deep overload (decode cell 256/512, 1P2D, `always`, seed
42), against the model:

| batch cap | predicted | measured | error | ITL mean |
|---|---|---|---|---|
| 16 | 3.657 | **3.5925** | −1.8% | 17.20 ms |
| 64 | 13.50 | **13.09** | −3.0% | 18.84 ms |
| 256 | 40.38 | **36.48** | −9.7% | 25.55 ms |

Predicted exceeds measured in all three, so the model is a proper upper bound.
The error grows with B because of **size-biased batch composition**: in deep
overload the batch fills with long-output requests, so the mean context exceeds
the `L = a_p + o/2` assumption and δ̄ is larger than modelled.

Two consequences of the cap beyond capacity:

- **α is 97.2% of iteration time at B = 16** (68.6% at B = 256). The fleet is a
  fixed-cost machine, and marginal work — which is what the congestion term and
  the W\*_i normalisation are built on — is ~3% of actual instance time. This is
  the likely reason the congestion-weight ablation was inert.
- **ITL is structurally pinned** to a 16.61–17.09 ms band at B = 16, and the
  sweep confirms it: itl_mean moves only 17.00 → 17.20 ms across rates 2 → 6,
  p99 17.27 → 17.30. That is why `z_itl` was identically zero and ITL never
  missed anywhere. It is a configuration artifact, not a property of the
  workload. At cap 256 ITL does move with load (17.31 → 18.99 ms, rates 4 → 14).

## 4. The knee criterion measures drain, not saturation

`responses_per_sec = CompletedRequests / SimEndedTime` (`sim/metrics.go:130`,
with `vllmRuntime = SimEndedTime` at line 73) — a **fixed** request count over
the **full** run including the post-arrival drain, divided by the *nominal*
rate. Two contaminants: Poisson variation in when the Nth arrival lands (±4% at
N = 600) and the drain tail (lognormal outputs to 4096 tokens ⇒ ~70 s).

Decode cell at cap 16, `always`:

| offered | achieved | ratio | duration |
|---|---|---|---|
| 2.0 | 1.9939 | 0.997 | 300.9 s |
| 3.0 | 2.7357 | **0.912** | 328.9 s |
| 3.5 | 3.3340 | **0.953** | 314.9 s |
| 4.0 | 3.4950 | 0.874 | 343.4 s |
| 6.0 | 3.5202 | 0.587 | 511.3 s |

The ratio is **non-monotone**, and rate 3.0 ran *longer* (328.9 s, 900 requests)
than rate 3.5 (314.9 s, 1050 requests) — fewer requests, lower load, longer
run. Worse, cap 16 and cap 64 give *identical* achieved throughput at rate 3.0
(2.7357 vs 2.7359), proving rate 3.0 is unsaturated at both caps while the
criterion reads it as failing the 0.95 threshold.

So every knee in `specs/grid_v2/cells.txt` was set by an unreliable criterion.
Decode's 3.0 happens to land at ρ ≈ 0.82 of the true 3.65 — roughly right by
luck. **Replacement**: read the plateau in deep overload (offer ~3× capacity;
achieved *is* capacity), or detect backlog growth. Do not use a fixed-N
achieved/offered ratio near the knee.

## 5. The corrected table

`--batch-cap 16 --tau-itl-ms 67 --packing packed` (the v2.1 configuration):

```
regime           B  bind  cap_coll  cap_disag  cap_free  knee  r_coll  r_disag  r_free  Gam^cap
decode          16   cap      3.64       3.65      5.43   3.0    0.82     0.82    0.55   1.0000
mixed           16   cap     12.55      13.65     17.05  12.0    0.96     0.88    0.70   1.0000
prefill_lean    16   cap     10.23       8.32     14.12  12.0    1.17     1.44    0.85   0.6934
prefill_bound   16   cap      7.90       4.15     11.11   8.0    1.01     1.93    0.72   0.5186
hetero          16   cap     14.44      14.77     29.55   6.0    0.42     0.41    0.20   1.0000
```

`r_* = knee / cap_*`. **r > 1 means that assignment cannot serve the knee rate
at all**, so any policy pinned to it must fail there.

This now *predicts* the measured policy behaviour rather than merely
accompanying it:

- **prefill_lean**: both static corners are infeasible at rate 12
  (r_disag = 1.44, r_coll = 1.17) while the free assignment sits at 0.85. That
  is exactly why `always` saturates at its own 8.25 req/s, `never` breaks at
  12–14, and the adaptive arms hold 1.000 through the knee.
- **prefill_bound**: r_disag = 1.93 — `always` is at half the required capacity.
- **decode**: both corners at 0.82, and the free assignment only 0.55. Nothing
  is under pressure, which is why `always ≡ never ≡ optimal` up to 3.5. The cell
  is a weak testbed *by construction*, and now we can say so analytically.
- **hetero**: r_free = 0.20. The rate-6 cell runs at a fifth of fleet capacity,
  confirming it is an easy test.
- **Γ^cap is now a real negative certificate**: 0.693 on prefill_lean and 0.519
  on prefill_bound. No routing rule can exceed those at the stated rates.

**Routing loss is near zero on prefill_lean.** cap_free = 14.12 req/s against a
measured best-policy goodput rate of 14.27 (lt-joint) / 14.22 (dpvar) / 14.21
(kairos) — the adaptive rules extract essentially the whole fleet bound (the
slight overshoot is within the model's ~2–10% error). `always` gets
8.32/14.12 = 59% and `never` 10.23/14.12 = 72%. This **corrects** an earlier
claim in this campaign that the best router leaves ~30% of capacity unclaimed;
that figure came from the α-blind computation.

## 6. The crippled A100 has zero capacity, not low capacity

`coeffs-llama70b-a100crippled-tp4.json` has α_D = **69.31 ms**, which exceeds
the derived ITL target of 67 ms. So `batch_itl` returns 0: that instance cannot
meet the ITL target **at any batch size**, and `cap_coll` for the hetero cell
collapses to exactly the H100's 14.44 req/s. The "slow" instance is not slow,
it is infeasible — every request placed there misses ITL by construction. That
matches the traces (every slow-instance placement misses ITL regardless of
policy) and it is the right semantics for `μ_nom` clamping to `edppMinMu`, but
it should be stated as a structural fact rather than presented as a hardware
speed difference.

## 7. Open items

- **Batch cap 16 vs 256 is undecided.** Cap 16 suppresses batching, pins ITL,
  makes decode fixed-cost-dominated, and lowers every knee, so it plausibly
  shapes the "decode cell is a weak testbed" verdict, the never@3M dominance
  result, and every knee rate. Cap 256 restores ITL as a live dimension but
  moves decode capacity to ~40 req/s, multiplying run cost.
- Knee rates in `cells.txt` need regenerating under a saturation criterion that
  is not drain-contaminated.
- `free_capacity` grids θ at 401 points per hardware type and supports at most
  two types; exact enough for the coefficients' precision but not general.
- The `free` point charges α once per stage, so it understates the true
  unrestricted bound wherever a collocated iteration could share it. Reported
  as `max(free, coll)` with a `*coll>free` flag; no regime triggers it here.
