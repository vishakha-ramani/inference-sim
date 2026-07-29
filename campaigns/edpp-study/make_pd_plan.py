#!/usr/bin/env python3
"""Emit a total --pd-plan CSV realising a fixed mixing share phi on 1P2D.

What phi means
--------------
phi is the share of requests that are DISAGGREGATED: their prefill runs on
instance 0 (the prefill-only instance) and their decode on a mixed instance. The
remaining 1 - phi are served whole on a mixed instance, which the plan expresses
with the prefill_instance value `local` (sim/fixed_plan_decider.go: "" or
"local" means prefill locally on the decode instance, and disaggregation does not
fire at all).

So phi = 1 reproduces --pd-decider always and phi = 0 reproduces --pd-decider never.
The harness runs both endpoints as a self-check before trusting any interior
point.

Why the assignment is deterministic rather than sampled
-------------------------------------------------------
We are measuring a capacity ceiling, which is a property of the long-run
assignment, so any extra variance is noise we do not want. Bresenham-style
interleaving puts the disaggregated requests as evenly through the stream as the
ratio allows, which means every prefix of the trace already sits at share phi.
Sampling phi from an RNG would give the same mean with added burstiness and would
make the result depend on a second seed.

Decode instances alternate within each class separately, so both mixed instances
receive an even share of the decode-only work AND an even share of the
whole-request work. Splitting one shared counter would correlate the two.

Plans must be total: sim/fixed_plan_decider.go panics on a request that is
absent, by design (R1, no silent fallback). So --num-requests must not exceed the
row count. The harness passes the same N it puts in the workload spec.

Usage
-----
    python3 campaigns/edpp-study/make_pd_plan.py --n 9600 --phi 0.387 > plan.csv
    python3 campaigns/edpp-study/make_pd_plan.py --n 100 --phi 0 --decode-instances 2
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True,
                    help="number of requests; must be >= the run's --num-requests")
    ap.add_argument("--phi", type=float, required=True,
                    help="share of requests disaggregated (prefill on the prefill instance)")
    ap.add_argument("--prefill-instance", default="instance_0",
                    help="the prefill-only instance (default instance_0)")
    ap.add_argument("--decode-instances", type=int, default=2,
                    help="size of the mixed/decode pool; ids follow the prefill instance")
    ap.add_argument("--id-prefix", default="request_",
                    help="request id prefix (sim/workload/generator.go uses request_<i>)")
    args = ap.parse_args()

    if not 0.0 <= args.phi <= 1.0:
        raise SystemExit(f"--phi must be in [0, 1], got {args.phi}")
    if args.decode_instances < 1:
        raise SystemExit("--decode-instances must be at least 1")

    # Mixed pool ids sit immediately after the prefill instance: 1P2D gives
    # instance_0 the prefill role and instance_1, instance_2 the mixed roles.
    mixed = [f"instance_{i}" for i in range(1, 1 + args.decode_instances)]

    print("request_id,decode_instance,prefill_instance")

    # Bresenham interleave: emit a disaggregated request whenever the running
    # error crosses one. Over n rows this places exactly round(phi*n) of them, as
    # evenly spread as the ratio permits.
    err = 0.0
    n_disagg = 0
    n_local = 0
    for i in range(args.n):
        err += args.phi
        if err >= 1.0 - 1e-12:
            err -= 1.0
            decode = mixed[n_disagg % len(mixed)]
            n_disagg += 1
            prefill = args.prefill_instance
        else:
            decode = mixed[n_local % len(mixed)]
            n_local += 1
            prefill = "local"
        print(f"{args.id_prefix}{i},{decode},{prefill}")


if __name__ == "__main__":
    main()
