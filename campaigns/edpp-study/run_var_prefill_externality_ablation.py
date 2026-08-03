#!/usr/bin/env python3
"""No-retuning ablation for the VaR prefill-externality calculation.

At the frozen development seed, compare the current VaR-only ground floor with:

1. exact marginal prefill overlap only; and
2. exact marginal prefill overlap plus causal decode-queue overlap.

The experiment changes no fitted coefficient, threshold, or lambda.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_decisive_campaign as base
import run_minimax_var_prefill as minimax


SEED = 13
POLICIES = (
    "var_prefill_nostability",
    "var_prefill_nostability_exactvar",
    "var_prefill_nostability_exactvar_pathwork",
    "var_prefill_nostability_exactvar_overlap",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=minimax.DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    target_map = minimax.targets()

    runs = [
        base.Run(
            "var_prefill_externality_ablation",
            workload,
            label,
            rate,
            SEED,
            policy,
            0,
            "1p2m",
            requests,
            target_map[workload.name],
        )
        for workload, label, rate, requests in minimax.conditions(out)
        for policy in POLICIES
    ]
    rows = base.run_many(runs, out, args.jobs, args.force)
    minimax.require_hard_valid(rows, "VaR prefill-externality ablation")
    base.write_rows(out / "var_prefill_externality_ablation_seed13.csv", rows)

    comparisons: list[dict[str, Any]] = []
    for workload, label, _, _ in minimax.conditions(out):
        members = {
            row["policy"]: row
            for row in rows
            if row["workload"] == workload.name and row["rate_label"] == label
        }
        baseline = members["var_prefill_nostability"]
        exact = members["var_prefill_nostability_exactvar"]
        exact_pathwork = members["var_prefill_nostability_exactvar_pathwork"]
        exact_overlap = members["var_prefill_nostability_exactvar_overlap"]
        comparisons.append(
            {
                "condition": f"{workload.name}:{label}",
                "baseline_goodput": float(baseline["goodput"]),
                "exact_goodput": float(exact["goodput"]),
                "exact_pathwork_goodput": float(exact_pathwork["goodput"]),
                "exact_overlap_goodput": float(exact_overlap["goodput"]),
                "exact_delta": float(exact["goodput"])
                - float(baseline["goodput"]),
                "overlap_interaction_delta": float(exact_overlap["goodput"])
                - float(exact["goodput"]),
                "pathwork_delta": float(exact_pathwork["goodput"])
                - float(exact["goodput"]),
                "baseline_phi": float(baseline["realized_phi"]),
                "exact_phi": float(exact["realized_phi"]),
                "exact_pathwork_phi": float(exact_pathwork["realized_phi"]),
                "exact_overlap_phi": float(exact_overlap["realized_phi"]),
            }
        )

    best = {
        row["condition"]: max(
            row["baseline_goodput"],
            row["exact_goodput"],
            row["exact_pathwork_goodput"],
            row["exact_overlap_goodput"],
        )
        for row in comparisons
    }
    field_by_policy = {
        "var_prefill_nostability": "baseline_goodput",
        "var_prefill_nostability_exactvar": "exact_goodput",
        "var_prefill_nostability_exactvar_pathwork": "exact_pathwork_goodput",
        "var_prefill_nostability_exactvar_overlap": "exact_overlap_goodput",
    }
    ranking = sorted(
        (
            {
                "policy": policy,
                "worst_regret": max(
                    best[row["condition"]] - row[field] for row in comparisons
                ),
                "worst_condition": max(
                    comparisons,
                    key=lambda row: best[row["condition"]] - row[field],
                )["condition"],
            }
            for policy, field in field_by_policy.items()
        ),
        key=lambda row: (row["worst_regret"], row["policy"]),
    )
    result = {
        "seed": SEED,
        "tuned": False,
        "policies": list(POLICIES),
        "comparisons": comparisons,
        "within_ablation_minimax": ranking,
    }
    (out / "var_prefill_externality_ablation_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    lines = [
        "# VaR prefill-externality ablation",
        "",
        f"Frozen development seed `{SEED}`; `lambda_p=0`; no parameter tuning.",
        "",
        "| condition | baseline GP | exact-prefill GP | exact+path-work GP | exact+TTFT-overlap GP | exact delta | path-work delta | overlap interaction | phi: base / exact / path / overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| `{row['condition']}` | {row['baseline_goodput']:.3f} | "
            f"{row['exact_goodput']:.3f} | "
            f"{row['exact_pathwork_goodput']:.3f} | "
            f"{row['exact_overlap_goodput']:.3f} | "
            f"{row['exact_delta']:+.3f} | "
            f"{row['pathwork_delta']:+.3f} | "
            f"{row['overlap_interaction_delta']:+.3f} | "
            f"{row['baseline_phi']:.3f} / {row['exact_phi']:.3f} / "
            f"{row['exact_pathwork_phi']:.3f} / "
            f"{row['exact_overlap_phi']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`exact-prefill` changes only the co-resident delay used by the",
            "prefill-pool part of `VaR(disagg)`: it removes baseline iteration",
            "time and charges only R's causal prefill work over shared chunks.",
            "`path-work` additionally uses the remote pool's own cache state for",
            "remote TTFT/work and Q booking. The final arm instead adds the",
            "separately motivated TTFT/decode-join overlap.",
            "",
            "| rank within ablation | policy | worst regret | worst condition |",
            "|---:|---|---:|---|",
        ]
    )
    for index, row in enumerate(ranking, 1):
        lines.append(
            f"| {index} | `{row['policy']}` | "
            f"{row['worst_regret']:.3f} | `{row['worst_condition']}` |"
        )
    lines.append("")
    (out / "VAR-PREFILL-EXTERNALITY-ABLATION.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
