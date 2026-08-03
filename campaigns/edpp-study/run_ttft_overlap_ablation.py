#!/usr/bin/env python3
"""No-retuning policy check for the TTFT overlap correction.

At the frozen development seed, compare:

1. the current VaR-only ground floor;
2. VaR-only with only the causal TTFT-overlap correction; and
3. the corrected estimator plus the previously deferred arriving-request
   composite-good term.

No lambda or estimator constant is selected here.
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
    "var_prefill_nostability_overlap",
    "var_prefill_self_nostability_overlap",
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
            "ttft_overlap_ablation",
            workload,
            label,
            rate,
            SEED,
            policy,
            0,
            "1p2m",
            requests,
            targets[workload.name],
        )
        for workload, label, rate, requests in minimax.conditions(out)
        for policy in POLICIES
    ]
    rows = base.run_many(runs, out, args.jobs, args.force)
    minimax.require_hard_valid(rows, "TTFT overlap ablation")
    base.write_rows(out / "ttft_overlap_ablation_seed13.csv", rows)

    comparisons: list[dict[str, Any]] = []
    for workload, label, _, _ in minimax.conditions(out):
        members = {
            row["policy"]: row
            for row in rows
            if row["workload"] == workload.name
            and row["rate_label"] == label
        }
        current = members["var_prefill_nostability"]
        overlap = members["var_prefill_nostability_overlap"]
        self_overlap = members["var_prefill_self_nostability_overlap"]
        comparisons.append(
            {
                "condition": f"{workload.name}:{label}",
                "current_goodput": float(current["goodput"]),
                "overlap_goodput": float(overlap["goodput"]),
                "self_overlap_goodput": float(self_overlap["goodput"]),
                "overlap_delta": float(overlap["goodput"])
                - float(current["goodput"]),
                "self_delta": float(self_overlap["goodput"])
                - float(overlap["goodput"]),
                "current_phi": float(current["realized_phi"]),
                "overlap_phi": float(overlap["realized_phi"]),
                "self_overlap_phi": float(self_overlap["realized_phi"]),
            }
        )

    best = {
        row["condition"]: max(
            row["current_goodput"],
            row["overlap_goodput"],
            row["self_overlap_goodput"],
        )
        for row in comparisons
    }
    field_by_policy = {
        "var_prefill_nostability": "current_goodput",
        "var_prefill_nostability_overlap": "overlap_goodput",
        "var_prefill_self_nostability_overlap": "self_overlap_goodput",
    }
    ranking = sorted(
        (
            {
                "policy": policy,
                "worst_regret": max(
                    best[row["condition"]] - row[field]
                    for row in comparisons
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
    (out / "ttft_overlap_ablation_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    lines = [
        "# TTFT overlap policy ablation",
        "",
        f"Frozen development seed `{SEED}`; `lambda_p=0`; no parameter tuning.",
        "",
        "| condition | VaR-only GP | +overlap GP | +overlap+self GP | overlap delta | self delta | phi: base / overlap / +self |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| `{row['condition']}` | {row['current_goodput']:.3f} | "
            f"{row['overlap_goodput']:.3f} | "
            f"{row['self_overlap_goodput']:.3f} | "
            f"{row['overlap_delta']:+.3f} | {row['self_delta']:+.3f} | "
            f"{row['current_phi']:.3f} / {row['overlap_phi']:.3f} / "
            f"{row['self_overlap_phi']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The overlap-only arm tests whether the corrected timing changes",
            "co-resident VaR decisions. The `+self` arm is the previously",
            "deferred arriving-request value; it is accepted only if the paired",
            "ordering validation and this no-retuning policy check support it.",
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
    (out / "TTFT-OVERLAP-ABLATION.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
