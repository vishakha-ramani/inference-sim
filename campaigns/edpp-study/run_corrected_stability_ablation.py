#!/usr/bin/env python3
"""Test the old prefill-stability term after correcting VaR and path work.

This is not a lambda sweep. It compares the exact-prefill VaR ground floor to
the already selected lambda_p=0.25, with the same exact VaR and path-specific
work accounting in both arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_decisive_campaign as base
import run_minimax_var_prefill as minimax


SEED = 13
LAMBDA = 0.25
ARMS = (
    ("var_prefill_nostability_exactvar_pathwork", 0.0),
    ("var_prefill_exactvar_pathwork", LAMBDA),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=minimax.DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    targets = minimax.targets()

    runs = [
        base.Run(
            "corrected_stability_ablation",
            workload,
            label,
            rate,
            SEED,
            policy,
            parameter,
            "1p2m",
            requests,
            targets[workload.name],
        )
        for workload, label, rate, requests in minimax.conditions(out)
        for policy, parameter in ARMS
    ]
    rows = base.run_many(runs, out, args.jobs, args.force)
    minimax.require_hard_valid(rows, "corrected stability ablation")
    base.write_rows(out / "corrected_stability_ablation_seed13.csv", rows)

    comparisons: list[dict[str, Any]] = []
    for workload, label, _, _ in minimax.conditions(out):
        members = {
            row["policy"]: row
            for row in rows
            if row["workload"] == workload.name and row["rate_label"] == label
        }
        no_stability = members[
            "var_prefill_nostability_exactvar_pathwork"
        ]
        stability = members["var_prefill_exactvar_pathwork"]
        comparisons.append(
            {
                "condition": f"{workload.name}:{label}",
                "no_stability_goodput": float(no_stability["goodput"]),
                "stability_goodput": float(stability["goodput"]),
                "stability_delta": float(stability["goodput"])
                - float(no_stability["goodput"]),
                "no_stability_phi": float(no_stability["realized_phi"]),
                "stability_phi": float(stability["realized_phi"]),
            }
        )
    result = {
        "seed": SEED,
        "lambda": LAMBDA,
        "tuned_here": False,
        "comparisons": comparisons,
    }
    (out / "corrected_stability_ablation_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
