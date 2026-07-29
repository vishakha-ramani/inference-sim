#!/usr/bin/env python3
"""Aggregate repro_goodput_ladder.sh output into a per-rate table.

Reads the harness log on stdin or from a path and reports, for each cell and
rate, the seed mean and the seed spread of every arm's goodput. The spread
matters because the gaps we care about are a few points wide, and a gap inside
the spread is not a result.

It also recomputes the saturation flag against the REALIZED arrival rate rather
than the nominal one. The harness compares achieved throughput against the rate
requested in the workload spec, and a Poisson trace does not deliver exactly that
rate, so every arm on a slow trace gets flagged together. A row where all three
arms report the same achieved throughput is a property of the trace.

Usage:
    python3 campaigns/edpp-study/analyze/goodput_ladder.py \
        campaigns/edpp-study/out/goodput_ladder_run.log
"""

import sys
from collections import defaultdict

# Logs written before the f -> phi rename use "plan@f*". Accept both so older
# logs stay readable.
ARMS = ["always", "dpvar", "plan@phi*"]
ARM_ALIASES = {"plan@f*": "plan@phi*"}
CEIL = {"prefill_lean": 19.75, "prefill_bound": 11.64}
ALWAYS_CAP = {"prefill_lean": 8.28, "prefill_bound": 4.15}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    text = open(path).read() if path else sys.stdin.read()

    # cell -> rate -> arm -> list of (achieved, goodput, ttft, itl, e2e, p99)
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ln in text.split("\n"):
        p = ln.split()
        if len(p) < 9 or not p[0].startswith("prefill_"):
            continue
        arm = ARM_ALIASES.get(p[1], p[1])
        if arm not in ARMS:
            continue
        cell, rate = p[0], float(p[2])
        try:
            vals = tuple(float(x) for x in p[3:9])
        except ValueError:
            continue
        rows[cell][rate][arm].append(vals)

    for cell in sorted(rows):
        cap, ceil = ALWAYS_CAP[cell], CEIL[cell]
        print(f"\n=== {cell} "
              f"(measured ceiling {ceil}, always caps at {cap}) ===")
        hdr = (f"{'rate':>6} {'x_ceil':>7} {'vs always cap':>14} | "
               + " | ".join(f"{a:>17}" for a in ARMS))
        print(hdr)
        print(f"{'':>6} {'':>7} {'':>14} | "
              + " | ".join(f"{'goodput':>9}{'spread':>8}" for _ in ARMS))
        print("-" * len(hdr))
        for rate in sorted(rows[cell]):
            per = rows[cell][rate]
            cells = []
            for a in ARMS:
                v = [x[1] for x in per.get(a, [])]
                if not v:
                    cells.append(f"{'--':>9}{'':>8}")
                    continue
                mean = sum(v) / len(v)
                spread = max(v) - min(v)
                cells.append(f"{mean:>9.3f}{spread:>8.3f}")
            note = "under" if rate < cap else ("at" if rate < 1.05 * cap else "over")
            print(f"{rate:>6.1f} {rate/ceil:>7.2f} {note:>14} | " + " | ".join(cells))

        # by-dimension detail at the rates where every arm is under its capacity
        under = [r for r in sorted(rows[cell]) if r < cap]
        if under:
            print(f"\n  by dimension at the rates below always's ceiling "
                  f"({', '.join(str(r) for r in under)}):")
            print(f"    {'rate':>5} {'arm':<9} {'achieved':>9} {'goodput':>8}"
                  f" {'ttft':>7} {'itl':>7} {'e2e':>7} {'p99 ms':>9}")
            for rate in under:
                for a in ARMS:
                    v = rows[cell][rate].get(a, [])
                    if not v:
                        continue
                    n = len(v)
                    m = [sum(x[i] for x in v) / n for i in range(6)]
                    print(f"    {rate:>5.1f} {a:<9} {m[0]:>9.3f} {m[1]:>8.3f}"
                          f" {m[2]:>7.3f} {m[3]:>7.3f} {m[4]:>7.3f} {m[5]:>9.1f}")

    print("\nNOTE: rows where all three arms report the same achieved throughput")
    print("differ from the nominal rate because of the arrival trace, not because")
    print("a policy saturated.")


if __name__ == "__main__":
    main()
