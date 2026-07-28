#!/usr/bin/env python3
"""Fleet capacity and the analytical goodput ceiling Gamma^cap, per regime.

Rewritten 2026-07-28. The previous version built work from the MARGINAL
coefficients only (CPf, CAttn, C0, C1) and divided by a lumped |I| = 3. Both
were wrong, and together they made every capacity figure 2x to 36x too high --
which also made Gamma^cap = 1.0000 uninformative, since you could not tell
whether the rate was genuinely below capacity or the bound was simply too
loose to bite:

  1. It dropped alpha (~16.61 ms), the per-iteration intercept and the LARGEST
     term in the trained physics model. Marginal-only work is the
     perfect-amortization limit (batch -> infinity, prefill chunks perfectly
     packed), which for prefill is loose by a factor of ~2.2 on the prefill
     cells and for the batch caps we actually run is loose by ~36x on decode.
  2. It pooled 1 prefill-only instance with 2 decode-only instances, letting
     prefill work borrow decode capacity that the 1P2D topology forbids.

WHAT CAPACITY MEANS HERE. Fleet capacity is a fluid feasibility question over
assignments, not a simulation. Pick fractions x_i (which instance prefills a
request) and y_i (which decodes it); instance i's utilisation is

    rho_i = lambda * [ x_i * t_P(theta_i) + y_i * t_D(theta_i, B_i) ]

and capacity is the largest lambda for which some (x, y) holds max_i rho_i <= 1.
It is policy-independent because it quantifies over ALL assignments: any real
router induces some long-run empirical assignment and therefore cannot beat the
min-max. Topology enters only as a restriction on the support of x and y, so
"what does 1P2D cost us versus 3 mixed instances" is the same LP with and
without that restriction. We report five points:

    coll    -- collocation forced (x_i = y_i), the `never` corner
    disag   -- roles separated, the `always` corner
    pd_best -- tightest bound respecting 1P2D (prefill on any instance, decode
               only on the two mixed ones). NO policy on this topology beats it.
    agg3    -- the aggregated 3-mixed reference fleet, which is what never@3M
               runs and what the v3 SLO targets are derived on
    free    -- every instance may take either role: a RE-PROVISIONED fleet of
               the same size, reported to price the topology choice

THE WORK MODEL. Per iteration, T = alpha + B * dbar + CPf * s (+ attention),
where B is the decode batch, dbar = C0 + C1 * L the marginal decode cost at
context L, and s the new prefill tokens in that iteration. alpha is charged per
ITERATION, not per request, so it amortises over whatever shares the iteration.
That makes the two stages behave differently and is the crux of the rewrite:

  DECODE. Serving lambda_d req/s of o output tokens at batch B needs
  lambda_d * o / B iterations/s, so per-request instance time is
      t_D = o * (alpha / B + dbar).
  This FALLS with B, to the marginal-only asymptote o * dbar as B -> infinity.
  So decode has no finite work-conservation ceiling in B; capacity is set by
  whatever caps the batch. Every v2.1 harness passes
  --max-num-running-reqs 16 (the simulator default is 256, cmd/root.go:1279),
  at which alpha is 97% of iteration time.

  PREFILL. A request of a_p tokens needs ceil(a_p / S) iterations at chunk
  budget S (--max-num-scheduled-tokens, default 2048). When a_p >= S those
  iterations are full and alpha CANNOT be amortised; when a_p < S several
  requests pack into one iteration and it can. We report both bounds:
      packed:   n_it = a_p / S      (optimistic; perfect packing)
      unpacked: n_it = ceil(a_p / S) (pessimistic; one request per iteration)
  and t_P = n_it * alpha_P + CPf * a_p + CAttn * a_p * (a_p / 2).

  COLLOCATED. One iteration carries both stages, so alpha is shared. Holding
  the workload ratio s / B = a_p / o gives
      t_coll = o * alpha / B + o * dbar + Wp_marginal,
  i.e. alpha is paid once per iteration rather than once per stage.

VALIDATION (campaigns/edpp-study/repro_batchcap_probe.sh, decode cell 256/512,
1P2D, seed 42, measured plateau in deep overload):
    cap  16: predicted 3.657 req/s, measured 3.5925  (-1.8%)
    cap  64: predicted 13.50 req/s, measured 13.09   (-3.0%)
and on prefill_lean the prefill-pool bound is 8.32 req/s against the measured
`always` plateau of 8.25 req/s (-0.9%).

GAMMA^CAP. Given lambda above capacity, the best any policy can do is serve the
CHEAPEST requests. Gamma^cap is the largest fraction gamma such that the gamma
cheapest requests fit BOTH constraints of the paper's capacity-ceiling theorem:
decode work fits in the |M| mixed instances (a prefill-only instance cannot lend
time to a decode backlog) and total work fits in the whole fleet, each stage
priced at the cheapest instance that may run it. m(gamma) is the lower partial
mean, taken in increasing order of total cost. This is an upper bound no routing
rule can exceed, so Gamma^cap < 1 is a negative certificate.

IT IS VACUOUS BELOW CAPACITY BY CONSTRUCTION, and every v2.1 knee rate is below
capacity, so G@knee is 1.0 in all five regimes. That is not a defect -- it is
the theorem correctly reporting that those operating points are servable. To
show the ceiling has content we also evaluate it at 1.25x and 1.5x pd_best.
NOTE a superseded intermediate result: constraining prefill to the single P
instance gave 0.693 / 0.519 on the prefill cells, but that is the `always`
corner's limit, not a bound over all policies, and must not be quoted as one.

CAVEAT ON THE ROLE-SEPARATED BOUNDS. Charging alpha separately to t_P and t_D
overcounts for any instance that ends up collocating, since a shared iteration
pays alpha once. So pd_best and free understate the truth wherever coll exceeds
them; pd_best takes the max of the three and flags the case.
"""
import argparse
import json
import math

import numpy as np

COEFF_PATH = "scripts/calibration/coeffs-llama70b-h100-tp4.json"
COEFF_PATH_SLOW = "scripts/calibration/coeffs-llama70b-a100crippled-tp4.json"

# name, input tokens, mean output, v2.1 knee rate, topology kind.
# Knees are from specs/grid_v2/cells.txt. They are printed for comparison only
# and are NOT trusted: repro_batchcap_probe.sh showed the achieved/offered >=
# 0.95 criterion that produced them is dominated by the post-arrival drain and
# by Poisson variation in the arrival window, and is non-monotone in rate near
# the knee. Compare against the `cap` columns, not against the knee.
REGIMES = [
    # name,            a_p,    E[o], knee, hetero
    ("decode",          256,    512,  3.0, False),
    ("mixed",          2048,    128, 12.0, False),
    ("prefill_lean",   8192,     64, 12.0, False),
    ("prefill_bound", 16000,     16,  8.0, False),
    ("hetero",          256,     64,  6.0, True),
]

SIGMA = 0.4          # output lognormal sigma, matches the grid's outdist()
N_MC = 2_000_000
SEED = 20260727


def load_coeffs(path):
    j = json.load(open(path))
    return dict(
        alphaP=j["prefill"]["alpha_p_us"],
        CPf=j["prefill"]["c_pf_us_per_token"],
        CAttn=j["prefill"]["c_attn_us_per_unit"],
        alphaD=j["decode"]["alpha_us"],
        C0=j["decode"]["c0_us_per_req"],
        C1=j["decode"]["c1_us_per_token"],
    )


def sample_outputs(mean_o, n, rng):
    """Match the grid's outdist(): lognormal with E[out] = mean_o, floored/capped."""
    if SIGMA == 0:
        return np.full(n, float(mean_o))
    mu = math.log(mean_o) - SIGMA * SIGMA / 2.0
    hi = int(mean_o * 8) + 16
    return np.clip(rng.lognormal(mu, SIGMA, size=n), 4, hi)


def t_prefill(c, a_p, chunk, packed):
    """Per-request prefill instance time, microseconds. Scalar (a_p is constant)."""
    n_it = a_p / chunk if packed else math.ceil(a_p / chunk)
    marginal = c["CPf"] * a_p + c["CAttn"] * a_p * (a_p / 2.0)
    return n_it * c["alphaP"] + marginal


def t_decode(c, a_p, o, batch):
    """Per-request decode instance time, microseconds. Vector over o.

    dbar is evaluated at the mean context over the request's decode lifetime,
    a_p + (o - 1) / 2, so the o**2 term carries the output-length variance
    exactly rather than through E[o]**2.
    """
    dbar = c["C0"] + c["C1"] * (a_p + (o - 1.0) / 2.0)
    return np.where(o > 0, o * (c["alphaD"] / batch + dbar), 0.0)


def t_collocated(c, a_p, o, batch, chunk, packed):
    """Per-request instance time when one iteration carries both stages.

    alpha is paid once per iteration rather than once per stage, so this is
    strictly cheaper than t_prefill + t_decode.
    """
    dbar = c["C0"] + c["C1"] * (a_p + (o - 1.0) / 2.0)
    marginal_p = c["CPf"] * a_p + c["CAttn"] * a_p * (a_p / 2.0)
    # Prefill chunks that cannot share an iteration with anything still pay
    # their own alpha; the shared part is folded into o * alpha / batch.
    extra_it = max(0.0, (a_p / chunk if packed else math.ceil(a_p / chunk)) - 1.0)
    return np.where(o > 0,
                    o * (c["alphaD"] / batch + dbar) + marginal_p + extra_it * c["alphaP"],
                    0.0)


def batch_itl(c, tau_itl_us, a_p, mean_o, cap):
    """Largest batch meeting the ITL target, capped by --max-num-running-reqs.

    Returns (effective_batch, binder) so the caller can say which constraint
    actually set it.
    """
    dbar = c["C0"] + c["C1"] * (a_p + mean_o / 2.0)
    if tau_itl_us is None:
        return cap, "cap"
    b_itl = (tau_itl_us - c["alphaD"]) / dbar
    if b_itl <= 0:
        return 0.0, "itl-infeasible"
    return (cap, "cap") if cap <= b_itl else (b_itl, "itl")


def pd_capacity(tp_list, td_list):
    """Min-max capacity respecting the 1P2D topology, single request class.

    Instance 0 is prefill-only, so it can absorb prefill share but no decode.
    Instances 1..n are mixed: `never` collocates whole requests on them, so they
    take either role. Give mixed instance i a fraction theta_i of its budget to
    prefill; capacity is the largest lambda with both stages covered.

    This is the tightest bound that no policy on this topology can cross, and it
    is the right comparison for the grid. It charges alpha once per stage, so it
    is conservative for any instance that actually collocates -- a shared
    iteration pays alpha once. Where cap_coll exceeds it, cap_coll is the truth.
    """
    grid = np.linspace(0.0, 1.0, 401)
    mixed = list(zip(tp_list[1:], td_list[1:]))
    best = 0.0
    for th in grid:
        # Same theta on every mixed instance: exact when they are identical, and
        # the mixed pool is homogeneous in every regime we run.
        pref = 1.0 / tp_list[0] + sum(th / p for p, _ in mixed)
        dec = sum((1.0 - th) / d for _, d in mixed)
        best = max(best, min(pref, dec))
    return best


def free_capacity(tp_list, td_list):
    """Min-max capacity with x, y unrestricted, single request class.

    Every instance may take either role, so this is the capacity of a
    RE-PROVISIONED fleet of the same size -- the aggregated shape, not 1P2D. It
    is reported to price the topology choice, not as a bound on 1P2D policies.

    Give instance i a fraction theta_i of its unit budget to prefill. Then the
    prefill share it can absorb is theta_i / tp_i and the decode share
    (1 - theta_i) / td_i, both per unit lambda. Feasibility at rate lambda
    needs sum_i theta_i / tp_i >= lambda and sum_i (1 - theta_i) / td_i >=
    lambda. Maximising the achievable lambda over theta is a 1-D LP; with a
    handful of instances a grid over theta per DISTINCT hardware type is exact
    to the grid resolution, which is all the precision the coefficients carry.
    """
    types = sorted(set(zip(tp_list, td_list)))
    counts = [sum(1 for p, d in zip(tp_list, td_list) if (p, d) == t) for t in types]
    grid = np.linspace(0.0, 1.0, 401)
    best = 0.0
    if len(types) == 1:
        (tp, td), k = types[0], counts[0]
        for th in grid:
            best = max(best, min(k * th / tp, k * (1 - th) / td))
        return best
    if len(types) == 2:
        (p0, d0), (p1, d1) = types
        k0, k1 = counts
        for t0 in grid:
            pref0, dec0 = k0 * t0 / p0, k0 * (1 - t0) / d0
            for t1 in grid:
                best = max(best, min(pref0 + k1 * t1 / p1, dec0 + k1 * (1 - t1) / d1))
        return best
    raise NotImplementedError("free_capacity supports up to 2 hardware types")


def gamma_cap(costs_by_pool, lam):
    """Largest fraction gamma of requests servable, ordered cheapest-first.

    costs_by_pool: list of (per-request cost array in seconds, pool size).
    Requests are sorted by TOTAL cost so every pool sees the same request set,
    which is what a router actually has to choose. Returns (gamma, binder).
    """
    total = sum(c for c, _ in costs_by_pool)
    order = np.argsort(total)
    n = len(total)
    gamma, binder = 1.0, "none"
    for cost, pool in costs_by_pool:
        pm = np.cumsum(cost[order]) / n          # m(k/n)
        if lam * pm[-1] <= pool:
            continue
        k = int(np.count_nonzero(lam * pm <= pool))
        if k / n < gamma:
            gamma, binder = k / n, f"pool={pool}"
    return gamma, binder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-cap", type=int, default=16,
                    help="--max-num-running-reqs used by the harnesses (default 16; "
                         "the simulator default is 256)")
    ap.add_argument("--chunk", type=int, default=2048,
                    help="--max-num-scheduled-tokens (simulator default 2048)")
    ap.add_argument("--packing", choices=["packed", "unpacked", "both"], default="both",
                    help="whether prefill chunks from different requests share an iteration")
    ap.add_argument("--tau-itl-ms", type=float, default=None,
                    help="if set, also cap the batch by the ITL target")
    args = ap.parse_args()

    c = load_coeffs(COEFF_PATH)
    try:
        c_slow = load_coeffs(COEFF_PATH_SLOW)
    except FileNotFoundError:
        c_slow = None
    rng = np.random.default_rng(SEED)
    tau = args.tau_itl_ms * 1e3 if args.tau_itl_ms else None
    packings = ["packed", "unpacked"] if args.packing == "both" else [args.packing]

    print(f"# batch cap {args.batch_cap}, chunk budget {args.chunk}, "
          f"tau_itl {'unset' if tau is None else f'{args.tau_itl_ms} ms'}")
    print(f"# alpha_D {c['alphaD']/1e3:.2f} ms, alpha_P {c['alphaP']/1e3:.2f} ms\n")

    hdr = (f"{'regime':<14} {'pack':<9} {'B':>5} {'bind':>5} "
           f"{'coll':>7} {'disag':>7} {'pd_best':>8} {'agg3':>7} {'free':>7} "
           f"{'knee':>5} {'r_pd':>6} {'G@knee':>7} {'G@1.25':>7} {'G@1.5':>7}")
    print(hdr)
    print("-" * len(hdr))

    for name, a_p, mean_o, knee, hetero in REGIMES:
        o = sample_outputs(mean_o, N_MC, rng)
        for packing in packings:
            packed = packing == "packed"
            # Homogeneous H100 fleet unless the regime is heterogeneous, in
            # which case instance 0 (the prefill role in 1P2D) stays H100 and
            # one decode instance is the crippled A100.
            fleet = [c, c, c] if not (hetero and c_slow) else [c, c, c_slow]

            batches, binders = zip(*(batch_itl(ci, tau, a_p, mean_o, args.batch_cap)
                                     for ci in fleet))
            tp = [t_prefill(ci, a_p, args.chunk, packed) / 1e6 for ci in fleet]
            td = [(t_decode(ci, a_p, o, b).mean() / 1e6 if b > 0 else math.inf)
                  for ci, b in zip(fleet, batches)]

            # coll: `never`. Collocation happens on the decode instances of the
            # 1P2D topology (the P instance takes no whole requests), so |M| = 2.
            tc = [t_collocated(ci, a_p, o, b, args.chunk, packed).mean() / 1e6
                  if b > 0 else math.inf for ci, b in zip(fleet, batches)]
            cap_coll = sum(1.0 / t for t in tc[1:])

            # agg3: the aggregated reference fleet. All three instances take
            # whole requests, which is the shape never@3M runs and the shape the
            # v3 SLO targets are derived on.
            cap_agg3 = sum(1.0 / t for t in tc)

            # disagg: `always` on 1P2D. Prefill pool = instance 0, decode = 1,2.
            cap_disag = min(1.0 / tp[0], sum(1.0 / t for t in td[1:]))

            # pd: the tightest bound respecting 1P2D (prefill anywhere, decode
            # only on the two mixed instances). free: re-provisioned fleet.
            cap_pd = pd_capacity(tp, td)
            cap_free = free_capacity(tp, td)

            # Two constraints, matching the paper's capacity-ceiling theorem:
            # decode work fits in the |M| mixed instances (a prefill-only
            # instance cannot lend time to a decode backlog), and total work
            # fits in the whole fleet. Each stage is priced at the cheapest
            # instance that may run it.
            td_arr = np.minimum.reduce(
                [t_decode(ci, a_p, o, b) / 1e6 if b > 0 else np.full(N_MC, math.inf)
                 for ci, b in zip(fleet[1:], batches[1:])])
            tot_arr = td_arr + min(tp)
            pools = [(td_arr, float(len(fleet) - 1)), (tot_arr, float(len(fleet)))]
            gam, _ = gamma_cap(pools, knee)
            # The ceiling is vacuous below capacity by construction, so also
            # report it above capacity, where it has content.
            gam125, _ = gamma_cap(pools, 1.25 * cap_pd)
            gam150, _ = gamma_cap(pools, 1.50 * cap_pd)

            # cap_pd charges alpha once per stage, so collocation can beat it.
            pd_best = max(cap_pd, cap_coll, cap_disag)
            flag = " *coll>pd" if cap_coll > cap_pd + 1e-9 else ""
            print(f"{name:<14} {packing:<9} {batches[1]:>5.0f} {binders[1]:>5} "
                  f"{cap_coll:>7.2f} {cap_disag:>7.2f} {pd_best:>8.2f} "
                  f"{cap_agg3:>7.2f} {cap_free:>7.2f} "
                  f"{knee:>5.1f} {knee/pd_best:>6.2f} "
                  f"{gam:>7.4f} {gam125:>7.4f} {gam150:>7.4f}{flag}")

    print("\n# All capacities in req/s. coll = never on 1P2D (2 mixed instances);")
    print("#   disag = always on 1P2D; pd_best = tightest bound respecting 1P2D, so")
    print("#   NO policy on this topology can beat it; agg3 = the aggregated 3-mixed")
    print("#   reference fleet that never@3M runs; free = re-provisioned fleet, shown")
    print("#   to price the topology choice. r_* = knee / cap_*; r > 1 means that")
    print("#   assignment cannot serve the knee rate at all.")
    print("# G@x is Gamma^cap: the largest good fraction ANY policy can achieve.")
    print("#   G@knee is at the v2.1 knee rate; G@1.25 and G@1.5 are at 1.25x and")
    print("#   1.5x pd_best. It is vacuous (1.0) below capacity BY CONSTRUCTION, so")
    print("#   the overload columns are where it carries information.")
    print("# pd_best and free charge alpha once per stage; a collocated iteration")
    print("#   pays it once for both, so they understate the truth wherever coll")
    print("#   exceeds them (flagged *coll>pd). pd_best takes the max of the three.")


if __name__ == "__main__":
    main()
