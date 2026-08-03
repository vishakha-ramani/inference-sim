#!/usr/bin/env python3
"""First ground-up var-prefill policy ablation.

No parameter is tuned here.  At frozen seed 13 and frozen lambda=0.25, compare:

  1. co-resident VaR + prefill stability (var_prefill);
  2. the same rule + arriving-request composite-good (var_prefill_self).

The second rule remains reduced/routing-preserving.  This script is a mechanism
diagnostic, not a held-out performance claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import run_decisive_campaign as base
import run_minimax_var_prefill as minimax
import trace_request_selection as selection


SEED = 13
POLICIES = (
    "var_prefill_nostability",
    "var_prefill",
    "var_prefill_self",
)
TRACE_CELLS = (
    ("shared", "medium"),
    ("shared", "near_high"),
    ("rag", "near_high"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_trace(
    out: Path,
    workload_name: str,
    rate_label: str,
    *,
    force: bool,
) -> Path:
    condition = {
        (workload.name, label): (workload, rate, requests)
        for workload, label, rate, requests in minimax.conditions(out)
    }[(workload_name, rate_label)]
    workload, rate, requests = condition
    targets = minimax.targets()[workload_name]
    weight = minimax.lambda_selection(out)
    root = out / "policy_ladder_trace"
    tag = f"{workload_name}_{rate_label}_var_prefill_self_s{SEED}_n{requests}"
    spec_path = root / "specs" / f"{tag}.yaml"
    metrics_path = root / "metrics" / f"{tag}.json"
    stdout_path = root / "stdout" / f"{tag}.txt"
    trace_path = root / "decisions" / f"{tag}.csv"
    for path in (spec_path, metrics_path, stdout_path, trace_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    base.make_spec(workload, rate, SEED, requests, spec_path, None)
    if (
        not force
        and metrics_path.exists()
        and stdout_path.exists()
        and trace_path.exists()
    ):
        return trace_path

    command = [
        str(base.ROOT / "blis"),
        "run",
        "--model",
        base.MODEL,
        "--workload-spec",
        str(spec_path),
        "--num-requests",
        str(requests),
        *base.topology_args("1p2m"),
        *base.slo_args(targets),
        *base.policy_args("var_prefill_self", targets, weight, None),
        "--seed",
        str(SEED),
        "--trace-level",
        "decisions",
        "--edpp-decision-trace",
        str(trace_path),
        "--metrics-path",
        str(metrics_path),
    ]
    proc = subprocess.run(command, cwd=base.ROOT, capture_output=True, text=True)
    stdout_path.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"trace run failed for {tag}:\n"
            f"{proc.stderr[-4000:]}\n{proc.stdout[-4000:]}"
        )
    return trace_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=minimax.DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    target_map = minimax.targets()
    weight = minimax.lambda_selection(out)

    runs = [
        base.Run(
            "policy_ladder",
            workload,
            label,
            rate,
            SEED,
            policy,
            weight,
            "1p2m",
            requests,
            target_map[workload.name],
        )
        for workload, label, rate, requests in minimax.conditions(out)
        for policy in POLICIES
    ]
    rows = base.run_many(runs, out, args.jobs, args.force)
    minimax.require_hard_valid(rows, "policy ladder")
    base.write_rows(out / "policy_ladder_seed13.csv", rows)

    comparisons: list[dict[str, Any]] = []
    for workload, label, _, _ in minimax.conditions(out):
        members = {
            row["policy"]: row
            for row in rows
            if row["workload"] == workload.name
            and row["rate_label"] == label
        }
        old = members["var_prefill"]
        new = members["var_prefill_self"]
        var_only = members["var_prefill_nostability"]
        comparisons.append(
            {
                "workload": workload.name,
                "rate_label": label,
                "var_only_goodput": float(var_only["goodput"]),
                "var_prefill_goodput": float(old["goodput"]),
                "with_self_goodput": float(new["goodput"]),
                "stability_delta": float(old["goodput"])
                - float(var_only["goodput"]),
                "goodput_delta": float(new["goodput"])
                - float(old["goodput"]),
                "var_only_phi": float(var_only["realized_phi"]),
                "var_prefill_phi": float(old["realized_phi"]),
                "with_self_phi": float(new["realized_phi"]),
                "phi_delta": float(new["realized_phi"])
                - float(old["realized_phi"]),
                "with_self_dropped": int(new["dropped"]),
            }
        )
    write_csv(out / "policy_ladder_comparison.csv", comparisons)

    trace_summary: list[dict[str, Any]] = []
    for workload_name, rate_label in TRACE_CELLS:
        trace_path = run_trace(
            out, workload_name, rate_label, force=args.force
        )
        trace_rows = read_csv(trace_path)
        evaluated = [row for row in trace_rows if not row["skip_reason"]]
        self_diffs = [
            float(row["self_good_disagg"]) - float(row["self_good_local"])
            for row in evaluated
        ]
        trace_summary.append(
            {
                "workload": workload_name,
                "rate_label": rate_label,
                "requests": len(trace_rows),
                "remote_fraction": sum(
                    row["disaggregate"].lower() == "true"
                    for row in trace_rows
                )
                / len(trace_rows),
                "self_favors_remote_fraction": sum(
                    value > 0 for value in self_diffs
                )
                / len(self_diffs),
                "self_favors_local_fraction": sum(
                    value < 0 for value in self_diffs
                )
                / len(self_diffs),
                "self_tie_fraction": sum(value == 0 for value in self_diffs)
                / len(self_diffs),
                "mean_self_good_difference": sum(self_diffs)
                / len(self_diffs),
            }
        )
    write_csv(out / "policy_ladder_trace_summary.csv", trace_summary)

    result = {
        "seed": SEED,
        "lambda": weight,
        "tuned": False,
        "formula": (
            "VaR_local - VaR_disagg + good_self_disagg - "
            "good_self_local > lambda * prefill_stability"
        ),
        "comparisons": comparisons,
        "trace_summary": trace_summary,
    }
    (out / "policy_ladder_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    lines = [
        "# Ground-up policy ladder: arriving-request value",
        "",
        f"Frozen seed `{SEED}` and frozen `lambda_p={weight}`; no tuning.",
        "",
        "Both policies retain normal decode and prefill routing. The only change",
        "is the dimensionless arriving-request value difference:",
        "",
        "```text",
        "VaR(local) - VaR(disagg)",
        "  + good_self(disagg) - good_self(local)",
        "  > lambda_p * prefill_queue_stability",
        "```",
        "",
        "| condition | VaR-only GP | +stability GP | +self GP | stability delta | self delta | phi: VaR / +stability / +self |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| `{row['workload']}:{row['rate_label']}` | "
            f"{row['var_only_goodput']:.3f} | "
            f"{row['var_prefill_goodput']:.3f} | "
            f"{row['with_self_goodput']:.3f} | "
            f"{row['stability_delta']:+.3f} | "
            f"{row['goodput_delta']:+.3f} | "
            f"{row['var_only_phi']:.3f} / "
            f"{row['var_prefill_phi']:.3f} / "
            f"{row['with_self_phi']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This is a structural seed-13 diagnostic. A promising change must be",
            "calibrated afresh and evaluated on disjoint seeds before it supports",
            "a minimax-regret claim.",
            "",
        ]
    )
    (out / "POLICY-LADDER.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
