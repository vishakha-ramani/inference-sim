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

`--batch-cap 256 --packing packed` (the v3 configuration; the v2.1 runs used
`--batch-cap 16`, whose figures are in section 3):

```
regime          B  bind    coll   disag  pd_best    agg3    free  knee   r_pd  G@knee  G@1.25   G@1.5
decode        256   cap   39.13   40.38    40.38   58.70   56.33   3.0   0.07  1.0000  0.9047  0.8142
mixed         256   cap   57.60   34.00    58.38   86.41   58.39  12.0   0.21  1.0000  0.8500  0.7309
prefill_lean  256   cap   15.02    8.32    20.01   22.53   20.03  12.0   0.60  1.0000  0.8249  0.6973
prefill_bound 256   cap    8.41    4.15    11.78   12.62   11.79   8.0   0.68  1.0000  0.8075  0.6754
hetero        256   cap  181.18  230.69   230.69  324.76  252.00   6.0   0.03  1.0000  1.0000  0.9781
```

`pd_best` is the tightest bound respecting 1P2D: prefill may run on any
instance, decode only on the two mixed ones. **No policy on this topology can
beat it.** `agg3` is the aggregated 3-mixed reference fleet — the shape
`never@3M` runs and the shape the v3 SLO targets are derived on. `free` prices a
re-provisioned fleet of the same size.

**The topology costs 7–32% of capacity.** `pd_best / agg3` is 0.69 on decode,
0.68 on mixed, 0.89 on prefill_lean, 0.93 on prefill_bound, 0.71 on hetero.
That quantifies analytically what was previously only an empirical observation
(aggregated 3M dominating 1P2M on homogeneous cells): dedicating one of three
instances to prefill-only costs real decode capacity, and it costs most where
decode is the bottleneck.

**Γ^cap is vacuous at every v2.1 knee rate, and that is the theorem working
correctly.** Those rates sit at `r_pd` = 0.03–0.68 of capacity, so they are
servable and no ceiling should bite. The ceiling carries information only above
capacity, which is why `G@1.25` and `G@1.5` are reported: at 1.5× capacity no
policy of any kind can exceed 0.68–0.81 good on the four homogeneous cells.

> **Superseded intermediate result — do not quote.** An earlier version of this
> table reported Γ^cap = 0.6934 (prefill_lean) and 0.5186 (prefill_bound) as
> negative certificates. Those came from constraining prefill to the single P
> instance, which is the `always` corner's limit, not a bound over all policies.
> Under the theorem's actual constraints the ceiling is 1.0 at those rates.

`r_* = knee / cap_*`. **r > 1 means that assignment cannot serve the knee rate
at all**, so any policy pinned to it must fail there.

This now *predicts* the measured policy behaviour rather than merely
accompanying it:

- **prefill_lean**: `always` needs 1.44x its own prefill-pool capacity at rate
  12 and `never` 1.17x its own, while a topology-respecting assignment sits at
  0.60. That is exactly why `always` saturates at 8.25 req/s, `never` breaks at
  12–14, and the adaptive arms hold 1.000 through the knee — the headroom
  exists, but only for a policy willing to use both roles.
- **prefill_bound**: `always` is at 1.93x its capacity — less than half of what
  it needs.
- **decode**: `pd_best` is 40.38 req/s at cap 256 against a knee of 3.0, so
  nothing is under pressure at all. The cell is a weak testbed *by
  construction*, and now we can say so analytically rather than by observation.
- **hetero**: r_pd = 0.03. The rate-6 cell runs at three percent of capacity.

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
