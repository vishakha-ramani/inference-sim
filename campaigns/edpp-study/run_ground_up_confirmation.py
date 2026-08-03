#!/usr/bin/env python3
"""Development-seed confirmation of the ground-up VaR candidates.

Runs, without retuning:

1. exact marginal prefill VaR + path-specific work, lambda_p=0;
2. the same corrected model with the previously selected lambda_p=0.25.

The original confirmation panel and VaR-only confirmation are reused only for
the condition-wise best-policy envelope. These seeds are development evidence.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import run_decisive_campaign as base
import run_minimax_var_prefill as minimax


ARMS = (
    ("var_prefill_nostability_exactvar_pathwork", 0.0),
    ("var_prefill_exactvar_pathwork", 0.25),
)


def policy_means(
    rows: list[dict[str, Any]],
    condition_keys: list[str],
    out: Path,
) -> dict[str, dict[str, float]]:
    means: dict[str, dict[str, float]] = {}
    for policy in sorted({str(row["policy"]) for row in rows}):
        values_by_condition: dict[str, float] = {}
        for workload, label, _, _ in minimax.conditions(out):
            values = [
                float(row["goodput"])
                for row in rows
                if row["policy"] == policy
                and row["workload"] == workload.name
                and row["rate_label"] == label
            ]
            if values:
                values_by_condition[f"{workload.name}:{label}"] = (
                    statistics.fmean(values)
                )
        if set(values_by_condition) == set(condition_keys):
            means[policy] = values_by_condition
    return means


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
            "ground_up_confirmation",
            workload,
            label,
            rate,
            seed,
            policy,
            parameter,
            "1p2m",
            requests,
            targets[workload.name],
        )
        for workload, label, rate, requests in minimax.conditions(out)
        for seed in minimax.CONFIRMATION_SEEDS
        for policy, parameter in ARMS
    ]
    new_rows = base.run_many(runs, out, args.jobs, args.force)
    minimax.require_hard_valid(new_rows, "ground-up confirmation")
    base.write_rows(out / "ground_up_confirmation.csv", new_rows)

    old_var_rows = minimax.read_rows(out / "var_only_confirmation.csv")
    if not old_var_rows:
        raise SystemExit("var_only_confirmation.csv is missing")
    old_panel = minimax.read_rows(out / "confirmation.csv")
    if not old_panel:
        raise SystemExit("confirmation.csv is missing")

    per_condition: list[dict[str, Any]] = []
    for workload, label, _, _ in minimax.conditions(out):
        by_policy = {
            policy: {
                int(row["seed"]): float(row["goodput"])
                for row in [*old_var_rows, *new_rows]
                if row["policy"] == policy
                and row["workload"] == workload.name
                and row["rate_label"] == label
            }
            for policy in (
                "var_prefill_nostability",
                "var_prefill_nostability_exactvar_pathwork",
                "var_prefill_exactvar_pathwork",
            )
        }
        seeds = sorted(
            set.intersection(*(set(values) for values in by_policy.values()))
        )
        exact_delta = [
            by_policy["var_prefill_nostability_exactvar_pathwork"][seed]
            - by_policy["var_prefill_nostability"][seed]
            for seed in seeds
        ]
        stability_delta = [
            by_policy["var_prefill_exactvar_pathwork"][seed]
            - by_policy["var_prefill_nostability_exactvar_pathwork"][seed]
            for seed in seeds
        ]
        exact_mean, exact_low, exact_high = base.ci95(exact_delta)
        stability_mean, stability_low, stability_high = base.ci95(
            stability_delta
        )
        per_condition.append(
            {
                "condition": f"{workload.name}:{label}",
                "legacy_var_only_mean": statistics.fmean(
                    by_policy["var_prefill_nostability"][seed]
                    for seed in seeds
                ),
                "exact_var_mean": statistics.fmean(
                    by_policy[
                        "var_prefill_nostability_exactvar_pathwork"
                    ][seed]
                    for seed in seeds
                ),
                "corrected_stability_mean": statistics.fmean(
                    by_policy["var_prefill_exactvar_pathwork"][seed]
                    for seed in seeds
                ),
                "exact_minus_legacy": exact_mean,
                "exact_ci_low": exact_low,
                "exact_ci_high": exact_high,
                "stability_minus_exact": stability_mean,
                "stability_ci_low": stability_low,
                "stability_ci_high": stability_high,
                "exact_wins_ties_losses": [
                    sum(delta > 0 for delta in exact_delta),
                    sum(delta == 0 for delta in exact_delta),
                    sum(delta < 0 for delta in exact_delta),
                ],
                "stability_wins_ties_losses": [
                    sum(delta > 0 for delta in stability_delta),
                    sum(delta == 0 for delta in stability_delta),
                    sum(delta < 0 for delta in stability_delta),
                ],
            }
        )

    combined: list[dict[str, Any]] = [*old_panel, *old_var_rows, *new_rows]
    condition_keys = [
        f"{workload.name}:{label}"
        for workload, label, _, _ in minimax.conditions(out)
    ]
    means = policy_means(combined, condition_keys, out)
    best_by_condition = {
        condition: max(values[condition] for values in means.values())
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
            for policy, values in means.items()
        ),
        key=lambda row: (row["worst_regret"], row["policy"]),
    )
    result = {
        "seeds": list(minimax.CONFIRMATION_SEEDS),
        "status": (
            "development evidence; exact VaR and corrected-stability variants "
            "were introduced after these seeds were opened"
        ),
        "arms": [
            {"policy": policy, "lambda": parameter}
            for policy, parameter in ARMS
        ],
        "per_condition": per_condition,
        "best_observed_policy_mean_by_condition": best_by_condition,
        "minimax_ranking": ranking,
    }
    (out / "ground_up_confirmation_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    lines = [
        "# Ground-up VaR confirmation",
        "",
        "Eight opened development seeds; no retuning. A final selected policy",
        "still requires fresh held-out seeds.",
        "",
        "| condition | legacy VaR-only | exact VaR | corrected +stability | exact−legacy (95% CI) | stability−exact (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in per_condition:
        lines.append(
            f"| `{row['condition']}` | {row['legacy_var_only_mean']:.3f} | "
            f"{row['exact_var_mean']:.3f} | "
            f"{row['corrected_stability_mean']:.3f} | "
            f"{row['exact_minus_legacy']:+.3f} "
            f"([{row['exact_ci_low']:+.3f}, {row['exact_ci_high']:+.3f}]) | "
            f"{row['stability_minus_exact']:+.3f} "
            f"([{row['stability_ci_low']:+.3f}, "
            f"{row['stability_ci_high']:+.3f}]) |"
        )
    lines.extend(
        [
            "",
            "Worst regret is to the best observed policy mean in each condition",
            "over the expanded development panel.",
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
    (out / "GROUND-UP-CONFIRMATION.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
