#!/usr/bin/env python3
"""Fleet capacity and the analytical goodput ceiling Gamma^cap, per regime.

Rewritten 2026-07-28. The previous version built work from the MARGINAL
coefficients only (CPf, CAttn, C0, C1) and divided by a lumped |I| = 3. Both
were wrong, and together they pinned Gamma^cap at 1.0000 in every regime:

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
without that restriction. We report three points:

    coll   -- collocation forced (x_i = y_i), the `never` corner
    disagg -- roles separated per the 1P2D topology, the `always` corner
    free   -- x and y unrestricted; the policy-independent fleet bound

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
cheapest requests fit: lambda * m_P(gamma) <= |P| and lambda * m_D(gamma) <=
|D|, where m(gamma) is the lower partial mean. Requests are ordered by cost,
which here means by output length. This is an upper bound on attainable goodput
that no routing rule can exceed, so Gamma^cap < 1 is a negative certificate.

CAVEAT ON `free`. Charging alpha separately to t_P and t_D overcounts for any
instance that ends up collocating, since a shared iteration pays alpha once.
So `free` as computed is a bound on the role-separated assignment; the true
unrestricted fleet capacity is >= max(free, coll). We report both and take the
max, and flag when coll wins.
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


def free_capacity(tp_list, td_list):
    """Min-max capacity with x, y unrestricted, single request class.

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
           f"{'cap_coll':>9} {'cap_disag':>10} {'cap_free':>9} "
           f"{'knee':>5} {'r_coll':>7} {'r_disag':>8} {'r_free':>7} {'Gam^cap':>8}")
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

            # disagg: `always` on 1P2D. Prefill pool = instance 0, decode = 1,2.
            cap_disag = min(1.0 / tp[0], sum(1.0 / t for t in td[1:]))

            # free: roles unrestricted across all 3 instances.
            cap_free = free_capacity(tp, td)

            gam, _ = gamma_cap(
                [(np.full(N_MC, tp[0]), 1.0),
                 (t_decode(c, a_p, o, batches[1]) / 1e6, 2.0)], knee)

            flag = " *coll>free" if cap_coll > cap_free + 1e-9 else ""
            print(f"{name:<14} {packing:<9} {batches[1]:>5.0f} {binders[1]:>5} "
                  f"{cap_coll:>9.2f} {cap_disag:>10.2f} {cap_free:>9.2f} "
                  f"{knee:>5.1f} {knee/cap_coll:>7.2f} {knee/cap_disag:>8.2f} "
                  f"{knee/cap_free:>7.2f} {gam:>8.4f}{flag}")

    print("\n# cap_* are req/s; r_* = knee / cap_*, the utilisation the v2.1 knee rate")
    print("#   implies against each assignment. r_* > 1 means that assignment CANNOT")
    print("#   serve the knee rate at all, so any policy pinned to it must fail there.")
    print("# Gam^cap uses the 1P2D pools at the knee rate; < 1 is a negative")
    print("#   certificate no routing rule can beat.")
    print("# The role-separated capacities charge alpha once per stage; a collocated")
    print("#   iteration pays it once for both, so cap_free understates the true")
    print("#   unrestricted bound wherever cap_coll exceeds it (flagged *coll>free).")


if __name__ == "__main__":
    main()
