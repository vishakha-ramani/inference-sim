#!/usr/bin/env python3
"""Fluid capacity of the IMPLEMENTABLE mixed plans on a 1P2D fleet.

Why this exists alongside gamma_cap.py
--------------------------------------
gamma_cap.py's `pd_best` column gives each mixed instance a time share theta for
prefill and (1 - theta) for decode, and prices the two shares at t_P and t_D
respectively. That charges the per-iteration intercept alpha ONCE PER STAGE. It
is the right accounting for a request whose prefill and decode land on different
instances, and the wrong accounting for a request that does both on the same
instance, because one collocated iteration carries both stages and pays alpha
once. gamma_cap.py guards the two pure corners by taking a max over `coll` and
`disag`, but the guard does not reach the interior, so every mixture strictly
between the corners is under-priced.

This script prices the interior correctly, which means `pd_best` is NOT an upper
bound on 1P2D policies and this value is.

The family of plans
-------------------
On a 1P2D fleet, instance 0 may only prefill and instances 1..2 (the mixed pool)
may do either. A per-request plan therefore has exactly three options, and one of
them is dominated:

  (a) prefill on instance 0, decode on a mixed instance  -> disaggregated
  (b) prefill and decode on the same mixed instance      -> collocated
  (c) prefill on mixed i, decode on mixed j != i         -> dominated by (b)

Option (c) is dominated because it costs t_P + t_D of mixed-pool time where (b)
costs t_coll, and t_coll is smaller by one alpha_P. Since the two mixed
instances are identical, concentrating that work loses nothing.

So a plan is described by one number: f, the fraction of requests served by (a).
The remaining 1 - f are served by (b). Decode is split evenly across the mixed
pool by symmetry.

The capacity
------------
Instance 0 absorbs f of the arrivals, each costing t_P:

    rho_0 = lambda * f * t_P

Each mixed instance absorbs half the decode-only work and half the whole-request
work:

    rho_mixed = lambda * [ (f/2) * t_D + ((1-f)/2) * t_coll ]

Stability needs both below one, so

    C(f) = min( 1 / (f * t_P) ,  2 / (f * t_D + (1-f) * t_coll) )

and the ceiling is max over f. The first term falls as 1/f; the second rises,
because t_D < t_coll. They cross at f*.

Two endpoints are structural, and the harness uses them as a self-check:

    C(f -> 0) = 2 / t_coll          = `coll`  in gamma_cap = never on 1P2D
    C(f  = 1) = min(1/t_P, 2/t_D)   = `disag` in gamma_cap = always on 1P2D

so a plan at f = 0 must reproduce --pd-decider never and a plan at f = 1 must
reproduce --pd-decider always. If the simulator disagrees, the harness is wrong
before any ceiling is measured.

Usage
-----
    python3 campaigns/edpp-study/mix_cap.py                  # run from repo root
    python3 campaigns/edpp-study/mix_cap.py --grid 40001
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "campaigns/edpp-study")
import gamma_cap as g  # noqa: E402  (path insert must precede the import)

CHUNK = 2048
BATCH_CAP = 256


def cell_times(coeffs, a_p, mean_o, rng):
    """Per-request occupancy of ONE instance, in seconds, for the three roles.

    Averaged over a Monte-Carlo sample of the lognormal output distribution
    rather than evaluated at the mean output length, because the decode cost is
    quadratic in output length and the two differ by a couple of percent.
    """
    o = g.sample_outputs(mean_o, g.N_MC, rng)
    batch, _ = g.batch_itl(coeffs, None, a_p, mean_o, BATCH_CAP)
    t_p = g.t_prefill(coeffs, a_p, CHUNK, True) / 1e6
    t_d = g.t_decode(coeffs, a_p, o, batch).mean() / 1e6
    t_c = g.t_collocated(coeffs, a_p, o, batch, CHUNK, True).mean() / 1e6
    return t_p, t_d, t_c


def mix_capacity(t_p, t_d, t_c, grid):
    """C(f) over a grid, plus the maximiser. f = 0 excluded (1/f diverges)."""
    f = np.linspace(0.0, 1.0, grid)[1:]
    prefill_side = 1.0 / (f * t_p)
    mixed_side = 2.0 / (f * t_d + (1.0 - f) * t_c)
    cap = np.minimum(prefill_side, mixed_side)
    i = int(np.argmax(cap))
    return f, cap, float(f[i]), float(cap[i])


def cap_at(f, t_p, t_d, t_c):
    """C(f) at a single f. f = 0 is the collocated corner (no prefill instance use)."""
    if f <= 0.0:
        return 2.0 / t_c
    return min(1.0 / (f * t_p), 2.0 / (f * t_d + (1.0 - f) * t_c))


def emit_grid(cell, f_values, overload, coeffs, rng):
    """Print `f C_pred offered` rows for the harness to consume.

    The offered rate is a fixed modest multiple of the PREDICTED capacity at that
    f, not a multiple of any policy's ceiling. That keeps every point saturated by
    the same margin, which matters because the backlog — not the rate — is what
    triggers `dropped_unservable`. Driving every f at one large rate sheds
    requests at the f values whose capacity is lowest, and a run with drops is not
    a capacity measurement.

    The harness still verifies saturation from the output rather than trusting
    this, because if the prediction were too low the offered rate could fall below
    the true capacity and the run would measure the arrival rate instead.
    """
    for name, a_p, mean_o, _knee, hetero in g.REGIMES:
        if name != cell:
            continue
        t_p, t_d, t_c = cell_times(coeffs, a_p, mean_o, rng)
        for f in f_values:
            c = cap_at(f, t_p, t_d, t_c)
            print(f"{f:.4f} {c:.4f} {overload * c:.4f}")
        return
    raise SystemExit(f"unknown cell {cell!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=20001,
                    help="resolution of the f grid (default 20001)")
    ap.add_argument("--emit-grid", metavar="CELL",
                    help="print 'f C_pred offered' rows for one cell instead of the table")
    ap.add_argument("--f-values", default="0,0.10,0.20,0.28,0.34,0.39,0.45,0.55,0.70,0.85,1.0",
                    help="comma-separated f values for --emit-grid")
    ap.add_argument("--overload", type=float, default=1.15,
                    help="offered rate as a multiple of C_pred(f) for --emit-grid")
    args = ap.parse_args()

    coeffs = g.load_coeffs(g.COEFF_PATH)
    rng = np.random.default_rng(g.SEED)

    if args.emit_grid:
        emit_grid(args.emit_grid, [float(x) for x in args.f_values.split(",")],
                  args.overload, coeffs, rng)
        return

    print(f"# fleet 1P2D, batch cap {BATCH_CAP}, chunk {CHUNK}, packed prefill")
    print(f"# alpha_D {coeffs['alphaD']/1e3:.2f} ms, alpha_P {coeffs['alphaP']/1e3:.2f} ms")
    print("# f = fraction of requests disaggregated (prefill on instance 0);")
    print("#     the rest are served whole on a mixed instance.\n")

    hdr = (f"{'cell':<14} {'t_P':>7} {'t_D':>7} {'t_coll':>7} | "
           f"{'never':>7} {'always':>7} {'pd_best':>8} {'agg3':>7} | "
           f"{'f*':>6} {'C_mix':>7} {'/agg3':>6} {'/pd_best':>9}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for name, a_p, mean_o, _knee, hetero in g.REGIMES:
        if hetero:
            # The heterogeneous regime in gamma_cap.py still loads the crippled
            # A100 coefficients, so its capacities are not usable. Skipped here
            # rather than reported wrong.
            continue
        t_p, t_d, t_c = cell_times(coeffs, a_p, mean_o, rng)
        never = 2.0 / t_c
        always = min(1.0 / t_p, 2.0 / t_d)
        agg3 = 3.0 / t_c
        pd_best = g.pd_capacity([t_p] * 3, [t_d] * 3)
        _f, _cap, f_star, c_mix = mix_capacity(t_p, t_d, t_c, args.grid)

        print(f"{name:<14} {t_p*1e3:>7.2f} {t_d*1e3:>7.2f} {t_c*1e3:>7.2f} | "
              f"{never:>7.2f} {always:>7.2f} {pd_best:>8.2f} {agg3:>7.2f} | "
              f"{f_star:>6.3f} {c_mix:>7.2f} {c_mix/agg3:>6.3f} {c_mix/pd_best:>9.3f}")
        rows.append((name, f_star, c_mix, never, always, pd_best, agg3))

    print()
    print("# All capacities in req/s for the whole 3-instance fleet.")
    print("# C_mix > pd_best wherever f* is interior: pd_best charges alpha once per")
    print("#   stage on the mixed instances, so it under-prices every mixture that")
    print("#   collocates. C_mix is the tightest fluid bound on 1P2D policies.")
    print("# C_mix / agg3 is the honest cost of disaggregation: what the best")
    print("#   implementable 1P2D plan gives up against three mixed instances.")
    print()
    print("# Probe rates for the overload sweep (must exceed C(f) at every f,")
    print("# so they are multiples of C_mix, not of any policy's own ceiling):")
    for name, _f_star, c_mix, *_ in rows:
        print(f"#   {name:<14} 1.5x -> {1.5*c_mix:6.1f}   2.5x -> {2.5*c_mix:6.1f}")


if __name__ == "__main__":
    main()
