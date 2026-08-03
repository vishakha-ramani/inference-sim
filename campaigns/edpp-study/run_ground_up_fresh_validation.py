#!/usr/bin/env python3
"""Fresh-seed validation of the ground-up reduced VaR policy ladder.

This campaign compares successive ingredients in the rebuilt P/D rule, plus
endpoint and condition-tuned static-fraction yardsticks. Joint routing is
outside its scope.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import run_decisive_campaign as base
import run_minimax_var_prefill as minimax


FRESH_SEEDS = (16381, 32771, 65537, 131071)
CANDIDATE = "var_prefill_nostability_exactvar_pathwork"
LEGACY_VAR = "var_prefill_nostability"
CORRECTED_STABILITY = "var_prefill_exactvar_pathwork"
STATIC_YARDSTICK = "routing_phi"
DEPLOYABLE_ARMS = (
    (CANDIDATE, 0.0),
    (LEGACY_VAR, 0.0),
    (CORRECTED_STABILITY, 0.25),
    ("always", None),
    ("never", None),
)
DISPLAY = {
    CANDIDATE: "corrected VaR",
    LEGACY_VAR: "legacy VaR-only",
    CORRECTED_STABILITY: "corrected VaR + stability",
    "always": "always disaggregate",
    "never": "never disaggregate",
    STATIC_YARDSTICK: "best calibrated static fraction",
}


def paired_comparison(
    rows: list[dict[str, Any]],
    means: dict[str, dict[str, float]],
    out: Path,
    left_policy: str,
    right_policy: str,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for workload, label, _, _ in minimax.conditions(out):
        condition = f"{workload.name}:{label}"
        left = {
            int(row["seed"]): float(row["goodput"])
            for row in rows
            if row["policy"] == left_policy
            and row["workload"] == workload.name
            and row["rate_label"] == label
        }
        right = {
            int(row["seed"]): float(row["goodput"])
            for row in rows
            if row["policy"] == right_policy
            and row["workload"] == workload.name
            and row["rate_label"] == label
        }
        deltas = [left[seed] - right[seed] for seed in FRESH_SEEDS]
        mean, low, high = base.ci95(deltas)
        comparisons.append(
            {
                "condition": condition,
                "left_policy": left_policy,
                "left_mean": means[left_policy][condition],
                "right_policy": right_policy,
                "right_mean": means[right_policy][condition],
                "paired_delta": mean,
                "ci_low": low,
                "ci_high": high,
                "wins": sum(delta > 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "losses": sum(delta < 0 for delta in deltas),
            }
        )
    return comparisons


def append_comparison_table(
    lines: list[str],
    title: str,
    comparisons: list[dict[str, Any]],
) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| condition | left | right | paired left−right | 95% t interval | W/T/L |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| `{row['condition']}` | {row['left_mean']:.3f} | "
            f"{row['right_mean']:.3f} | {row['paired_delta']:+.3f} | "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=minimax.DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    targets = minimax.targets()
    static = minimax.static_selection(out)

    runs: list[base.Run] = []
    for workload, label, rate, requests in minimax.conditions(out):
        condition = f"{workload.name}:{label}"
        arms = [
            *DEPLOYABLE_ARMS,
            (
                STATIC_YARDSTICK,
                float(static["condition"][condition]["phi"]),
            ),
        ]
        for seed in FRESH_SEEDS:
            for policy, parameter in arms:
                runs.append(
                    base.Run(
                        "ground_up_fresh_validation",
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
                )

    rows = base.run_many(runs, out, args.jobs, args.force)
    minimax.require_hard_valid(rows, "ground-up fresh validation")
    base.write_rows(out / "ground_up_fresh_validation.csv", rows)

    conditions = [
        f"{workload.name}:{label}"
        for workload, label, _, _ in minimax.conditions(out)
    ]
    policies = tuple(policy for policy, _ in DEPLOYABLE_ARMS) + (
        STATIC_YARDSTICK,
    )
    means: dict[str, dict[str, float]] = {}
    realized_phi: dict[str, dict[str, float]] = {}
    for policy in policies:
        means[policy] = {}
        realized_phi[policy] = {}
        for workload, label, _, _ in minimax.conditions(out):
            condition = f"{workload.name}:{label}"
            members = [
                row
                for row in rows
                if row["policy"] == policy
                and row["workload"] == workload.name
                and row["rate_label"] == label
            ]
            if len(members) != len(FRESH_SEEDS):
                raise RuntimeError(
                    f"{policy}/{condition}: expected {len(FRESH_SEEDS)} "
                    f"seeds, got {len(members)}"
                )
            means[policy][condition] = statistics.fmean(
                float(row["goodput"]) for row in members
            )
            realized_phi[policy][condition] = statistics.fmean(
                float(row["realized_phi"]) for row in members
            )

    deployable_policies = tuple(policy for policy, _ in DEPLOYABLE_ARMS)
    best_deployable = {
        condition: max(
            means[policy][condition] for policy in deployable_policies
        )
        for condition in conditions
    }
    ranking = sorted(
        (
            {
                "policy": policy,
                "worst_regret": max(
                    best_deployable[condition] - means[policy][condition]
                    for condition in conditions
                ),
                "worst_condition": max(
                    conditions,
                    key=lambda condition: (
                        best_deployable[condition] - means[policy][condition]
                    ),
                ),
            }
            for policy in deployable_policies
        ),
        key=lambda row: (row["worst_regret"], row["policy"]),
    )

    exact_vs_legacy = paired_comparison(
        rows, means, out, CANDIDATE, LEGACY_VAR
    )
    stability_vs_exact = paired_comparison(
        rows, means, out, CORRECTED_STABILITY, CANDIDATE
    )
    exact_vs_static = paired_comparison(
        rows, means, out, CANDIDATE, STATIC_YARDSTICK
    )
    strict_invalid_runs = [
        {
            "condition": f"{row['workload']}:{row['rate_label']}",
            "policy": row["policy"],
            "seed": int(row["seed"]),
            "dropped": int(row["dropped"]),
        }
        for row in rows
        if not row["valid"]
    ]

    result = {
        "seeds": list(FRESH_SEEDS),
        "status": (
            "candidate frozen before these seeds; comparator panel revised "
            "without retuning after joint routing was removed from scope"
        ),
        "reference": (
            "best tested deployable ground-up arm mean per condition; "
            "not an oracle"
        ),
        "policy_means": means,
        "realized_phi_means": realized_phi,
        "best_deployable_by_condition": best_deployable,
        "ground_up_minimax_ranking": ranking,
        "exact_vs_legacy_var_only": exact_vs_legacy,
        "corrected_stability_vs_exact": stability_vs_exact,
        "exact_vs_best_calibrated_static_fraction": exact_vs_static,
        "hard_valid_runs": len(rows),
        "strict_invalid_runs": strict_invalid_runs,
    }
    (out / "ground_up_fresh_validation_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    lines = [
        "# Ground-up VaR fresh validation",
        "",
        "Joint routing is intentionally outside this campaign. The candidate",
        "was frozen before these four seeds. The",
        "comparator panel was revised without retuning after that scope was",
        "clarified.",
        "",
        "## Deployable ground-up ladder",
        "",
        "| rank | policy | worst regret | worst condition |",
        "|---:|---|---:|---|",
    ]
    for index, row in enumerate(ranking, 1):
        lines.append(
            f"| {index} | {DISPLAY[row['policy']]} | "
            f"{row['worst_regret']:.3f} | `{row['worst_condition']}` |"
        )
    lines.extend(
        [
            "",
            "Regret is to the best tested deployable arm in this ground-up",
            "ladder in each condition. It is not oracle regret.",
            "",
        ]
    )
    append_comparison_table(
        lines,
        "Estimator repair: corrected VaR versus legacy VaR-only",
        exact_vs_legacy,
    )
    append_comparison_table(
        lines,
        "Term ablation: corrected VaR + stability versus corrected VaR",
        stability_vs_exact,
    )
    append_comparison_table(
        lines,
        "System yardstick: corrected VaR versus best calibrated static fraction",
        exact_vs_static,
    )
    lines.extend(
        [
            "The static fraction knows the workload/load condition and uses one",
            "request-agnostic Bresenham share. It is neither a deployable adaptive",
            "policy nor a request-selection oracle.",
            "",
            f"All {len(rows)} runs passed the hard-validity gate. "
            f"{len(strict_invalid_runs)} runs had counted unservable drops and",
            "therefore failed the stricter zero-drop `valid` flag; those drops",
            "remain in the goodput denominator.",
            "",
        ]
    )
    (out / "GROUND-UP-FRESH-VALIDATION.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
