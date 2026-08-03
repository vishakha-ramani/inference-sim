#!/usr/bin/env python3
"""Fresh public-baseline panel for the reduced causal-VaR policy.

The corrected causal-VaR, fixed-corner, and condition-static runs are reused
from the frozen ground-up campaign.  This script runs only the missing
reader-facing baselines on the same fresh seeds and reports a common
worst-condition-regret comparison.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import run_decisive_campaign as base
import run_ground_up_fresh_validation as ground
import run_minimax_var_prefill as minimax


PHASE = "public_baseline_fresh_validation"
CANDIDATE = ground.CANDIDATE
STATIC_YARDSTICK = ground.STATIC_YARDSTICK
NEW_ARMS = (
    ("kairos", 0.5),
    ("least_ttft", None),
    # The llm-d endpoint-picker profile ships a 16-token uncached-prefix
    # threshold.  Keep its externally meaningful value rather than selecting
    # a new threshold on these conditions.
    ("threshold", 16),
)
DEPLOYABLE = (
    CANDIDATE,
    "kairos",
    "least_ttft",
    "threshold",
    "always",
    "never",
)
DISPLAY = {
    CANDIDATE: "causal VaR",
    "kairos": "Kairos",
    "least_ttft": "least predicted TTFT",
    "threshold": "prefix threshold (16)",
    "always": "always remote",
    "never": "always local",
    STATIC_YARDSTICK: "condition-tuned static share",
}


def policy_means(
    rows: list[dict[str, Any]], policies: tuple[str, ...], out: Path
) -> dict[str, dict[str, float]]:
    means: dict[str, dict[str, float]] = {policy: {} for policy in policies}
    for policy in policies:
        for workload, label, _, _ in minimax.conditions(out):
            condition = f"{workload.name}:{label}"
            members = [
                row
                for row in rows
                if row["policy"] == policy
                and row["workload"] == workload.name
                and row["rate_label"] == label
            ]
            if len(members) != len(ground.FRESH_SEEDS):
                raise RuntimeError(
                    f"{policy}/{condition}: expected {len(ground.FRESH_SEEDS)} "
                    f"fresh seeds, got {len(members)}"
                )
            means[policy][condition] = statistics.fmean(
                float(row["goodput"]) for row in members
            )
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
            PHASE,
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
        for seed in ground.FRESH_SEEDS
        for policy, parameter in NEW_ARMS
    ]
    new_rows = base.run_many(runs, out, args.jobs, args.force)
    minimax.require_hard_valid(new_rows, "public baseline fresh validation")
    base.write_rows(out / f"{PHASE}.csv", new_rows)

    frozen_rows = minimax.read_rows(out / "ground_up_fresh_validation.csv")
    if not frozen_rows:
        raise SystemExit("ground_up_fresh_validation.csv is missing")
    reusable = [
        row
        for row in frozen_rows
        if row["policy"] in {CANDIDATE, "always", "never", STATIC_YARDSTICK}
    ]
    rows: list[dict[str, Any]] = [*reusable, *new_rows]
    policies = (*DEPLOYABLE, STATIC_YARDSTICK)
    means = policy_means(rows, policies, out)
    conditions = list(means[CANDIDATE])

    deployable_best = {
        condition: max(means[policy][condition] for policy in DEPLOYABLE)
        for condition in conditions
    }
    ranking = []
    for policy in DEPLOYABLE:
        regrets = {
            condition: deployable_best[condition] - means[policy][condition]
            for condition in conditions
        }
        worst_condition = max(regrets, key=regrets.get)
        ranking.append(
            {
                "policy": policy,
                "display": DISPLAY[policy],
                "worst_regret": regrets[worst_condition],
                "worst_condition": worst_condition,
            }
        )
    ranking.sort(key=lambda row: (row["worst_regret"], row["policy"]))

    pairwise = []
    for policy in policies:
        if policy == CANDIDATE:
            continue
        records = []
        for workload, label, _, _ in minimax.conditions(out):
            condition = f"{workload.name}:{label}"
            candidate_by_seed = {
                int(row["seed"]): float(row["goodput"])
                for row in rows
                if row["policy"] == CANDIDATE
                and row["workload"] == workload.name
                and row["rate_label"] == label
            }
            baseline_by_seed = {
                int(row["seed"]): float(row["goodput"])
                for row in rows
                if row["policy"] == policy
                and row["workload"] == workload.name
                and row["rate_label"] == label
            }
            deltas = [
                candidate_by_seed[seed] - baseline_by_seed[seed]
                for seed in ground.FRESH_SEEDS
            ]
            mean, low, high = base.ci95(deltas)
            records.append(
                {
                    "condition": condition,
                    "causal_mean": means[CANDIDATE][condition],
                    "baseline_mean": means[policy][condition],
                    "paired_delta": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "wins": sum(delta > 0 for delta in deltas),
                    "ties": sum(delta == 0 for delta in deltas),
                    "losses": sum(delta < 0 for delta in deltas),
                }
            )
        pairwise.append(
            {
                "policy": policy,
                "display": DISPLAY[policy],
                "conditions": records,
            }
        )

    static_shortfalls = {
        condition: max(
            means[STATIC_YARDSTICK][condition] - means[CANDIDATE][condition],
            0.0,
        )
        for condition in conditions
    }
    result = {
        "status": "same frozen fresh seeds and conditions as ground-up validation",
        "seeds": list(ground.FRESH_SEEDS),
        "deployable_policies": list(DEPLOYABLE),
        "policy_means": means,
        "best_deployable_by_condition": deployable_best,
        "minimax_ranking": ranking,
        "causal_var_pairwise": pairwise,
        "static_yardstick_worst_shortfall": max(static_shortfalls.values()),
        "static_yardstick_worst_condition": max(
            static_shortfalls, key=static_shortfalls.get
        ),
    }
    (out / f"{PHASE}_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    lines = [
        "# Fresh public-baseline validation",
        "",
        "All policies use the same four frozen fresh seeds and nine conditions.",
        "The condition-tuned static share is an offline yardstick and is not in",
        "the deployable regret ranking.",
        "",
        "| rank | policy | worst regret | worst condition |",
        "|---:|---|---:|---|",
    ]
    for index, row in enumerate(ranking, 1):
        lines.append(
            f"| {index} | {row['display']} | {row['worst_regret']:.3f} | "
            f"`{row['worst_condition']}` |"
        )
    lines.extend(
        [
            "",
            "## Mean goodput by condition",
            "",
            "| condition | "
            + " | ".join(DISPLAY[p] for p in policies)
            + " |",
            "|---|" + "---:|" * len(policies),
        ]
    )
    for condition in conditions:
        lines.append(
            f"| `{condition}` | "
            + " | ".join(f"{means[p][condition]:.3f}" for p in policies)
            + " |"
        )
    lines.extend(
        [
            "",
            f"Causal VaR's largest one-sided shortfall to the condition-tuned "
            f"static yardstick is {max(static_shortfalls.values()):.3f} on "
            f"`{max(static_shortfalls, key=static_shortfalls.get)}`.",
            "",
        ]
    )
    (out / "PUBLIC-BASELINE-FRESH-VALIDATION.md").write_text(
        "\n".join(lines)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
