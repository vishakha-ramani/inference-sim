#!/usr/bin/env python3
"""Development-seed confirmation of the VaR-only ground-floor policy.

Runs lambda_p=0 on the eight already-opened confirmation seeds, then compares
it with the frozen lambda_p=0.25 var-prefill arm from confirmation.csv.  These
seeds are development evidence after this analysis, not fresh held-out data.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import run_decisive_campaign as base
import run_minimax_var_prefill as minimax


POLICY = "var_prefill_nostability"


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
            "var_only_confirmation",
            workload,
            label,
            rate,
            seed,
            POLICY,
            0,
            "1p2m",
            requests,
            targets[workload.name],
        )
        for workload, label, rate, requests in minimax.conditions(out)
        for seed in minimax.CONFIRMATION_SEEDS
    ]
    var_rows = base.run_many(runs, out, args.jobs, args.force)
    minimax.require_hard_valid(var_rows, "var-only confirmation")
    base.write_rows(out / "var_only_confirmation.csv", var_rows)

    confirmation = minimax.read_rows(out / "confirmation.csv")
    if not confirmation:
        raise SystemExit("confirmation.csv is missing")
    old_rows = [
        row
        for row in confirmation
        if row["policy"] == "var_prefill"
        and int(row["seed"]) in minimax.CONFIRMATION_SEEDS
    ]

    per_condition: list[dict[str, Any]] = []
    for workload, label, _, _ in minimax.conditions(out):
        old = {
            int(row["seed"]): float(row["goodput"])
            for row in old_rows
            if row["workload"] == workload.name
            and row["rate_label"] == label
        }
        new = {
            int(row["seed"]): float(row["goodput"])
            for row in var_rows
            if row["workload"] == workload.name
            and row["rate_label"] == label
        }
        seeds = sorted(set(old) & set(new))
        deltas = [new[seed] - old[seed] for seed in seeds]
        mean, ci_low, ci_high = base.ci95(deltas)
        per_condition.append(
            {
                "condition": f"{workload.name}:{label}",
                "var_prefill_mean": statistics.fmean(old[seed] for seed in seeds),
                "var_only_mean": statistics.fmean(new[seed] for seed in seeds),
                "paired_delta": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wins": sum(delta > 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "losses": sum(delta < 0 for delta in deltas),
            }
        )

    # Recompute the development-seed minimax ranking after adding VaR-only.
    combined: list[dict[str, Any]] = [*confirmation, *var_rows]
    policies = sorted({row["policy"] for row in combined})
    condition_keys = [
        f"{workload.name}:{label}"
        for workload, label, _, _ in minimax.conditions(out)
    ]
    means: dict[str, dict[str, float]] = {}
    for policy in policies:
        means[policy] = {}
        for workload, label, _, _ in minimax.conditions(out):
            values = [
                float(row["goodput"])
                for row in combined
                if row["policy"] == policy
                and row["workload"] == workload.name
                and row["rate_label"] == label
            ]
            if values:
                means[policy][f"{workload.name}:{label}"] = statistics.fmean(
                    values
                )
    complete = {
        policy: values
        for policy, values in means.items()
        if set(values) == set(condition_keys)
    }
    best_by_condition = {
        condition: max(values[condition] for values in complete.values())
        for condition in condition_keys
    }
    ranking = sorted(
        (
            {
                "policy": policy,
                "worst_regret": max(
                    best_by_condition[condition] - values[condition]
                    for condition in condition_keys
                ),
                "worst_condition": max(
                    condition_keys,
                    key=lambda condition: (
                        best_by_condition[condition] - values[condition]
                    ),
                ),
            }
            for policy, values in complete.items()
        ),
        key=lambda row: (row["worst_regret"], row["policy"]),
    )

    result = {
        "seeds": list(minimax.CONFIRMATION_SEEDS),
        "status": (
            "development evidence; these seeds are no longer held out for "
            "future policy variants"
        ),
        "per_condition": per_condition,
        "minimax_ranking_with_var_only": ranking,
    }
    (out / "var_only_confirmation_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    lines = [
        "# VaR-only confirmation on development seeds",
        "",
        "Eight previously opened confirmation seeds; no tuning. These are now",
        "development evidence, so a future final policy needs fresh seeds.",
        "",
        "| condition | +stability mean | VaR-only mean | paired delta | 95% t interval | W/T/L |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in per_condition:
        lines.append(
            f"| `{row['condition']}` | {row['var_prefill_mean']:.3f} | "
            f"{row['var_only_mean']:.3f} | {row['paired_delta']:+.3f} | "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend(
        [
            "",
            "The minimax table is recomputed over the original confirmation arms",
            "plus VaR-only; the per-condition reference is the best policy mean",
            "in this expanded development panel.",
            "",
            "| rank | policy | worst regret | worst condition |",
            "|---:|---|---:|---|",
        ]
    )
    for index, row in enumerate(ranking, 1):
        lines.append(
            f"| {index} | `{row['policy']}` | "
            f"{row['worst_regret']:.3f} | `{row['worst_condition']}` |"
        )
    lines.append("")
    (out / "VAR-ONLY-CONFIRMATION.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
