#!/usr/bin/env python3
"""Trace request-level decision changes from the exact prefill VaR repair.

This is a mechanism diagnostic at the frozen development seed. It compares the
same generated requests under the current VaR-only trajectory and the exact
marginal-prefill trajectory. Because trajectories diverge after a changed
decision, the resulting labels are descriptive; placement-pinned one-request
counterfactuals provide the causal follow-up.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

import run_decisive_campaign as base
import run_minimax_var_prefill as minimax


SEED = 13
CELLS = ("low", "medium", "near_high")
POLICIES = (
    "var_prefill_nostability",
    "var_prefill_nostability_exactvar",
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


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = q * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def dist(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": percentile(values, 0.5),
        "p10": percentile(values, 0.1),
        "p90": percentile(values, 0.9),
    }


def trace_run(
    out: Path, rate_label: str, policy: str, *, force: bool
) -> dict[str, Path]:
    workload = base.WORKLOADS["rag"]
    condition = {
        label: (rate, requests)
        for candidate, label, rate, requests in minimax.conditions(out)
        if candidate.name == "rag"
    }
    rate, requests = condition[rate_label]
    targets = minimax.targets()["rag"]
    root = out / "var_prefill_externality_trace"
    tag = f"rag_{rate_label}_{policy}_s{SEED}_n{requests}"
    paths = {
        "spec": root / "specs" / f"{tag}.yaml",
        "metrics": root / "metrics" / f"{tag}.json",
        "stdout": root / "stdout" / f"{tag}.txt",
        "decisions": root / "decisions" / f"{tag}.csv",
        "outcomes": root / "outcomes" / f"{tag}.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    base.make_spec(workload, rate, SEED, requests, paths["spec"], None)
    if (
        not force
        and paths["metrics"].exists()
        and paths["stdout"].exists()
        and paths["decisions"].exists()
        and paths["outcomes"].exists()
    ):
        return paths

    command = [
        str(base.ROOT / "blis"),
        "run",
        "--model",
        base.MODEL,
        "--workload-spec",
        str(paths["spec"]),
        "--num-requests",
        str(requests),
        *base.topology_args("1p2m"),
        *base.slo_args(targets),
        *base.policy_args(policy, targets, 0, None),
        "--seed",
        str(SEED),
        "--trace-level",
        "decisions",
        "--edpp-decision-trace",
        str(paths["decisions"]),
        "--pd-outcome-trace",
        str(paths["outcomes"]),
        "--metrics-path",
        str(paths["metrics"]),
    ]
    proc = subprocess.run(command, cwd=base.ROOT, capture_output=True, text=True)
    paths["stdout"].write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"trace run failed ({tag}):\n{proc.stderr[-4000:]}\n"
            f"{proc.stdout[-4000:]}"
        )
    return paths


def parse_decisions(path: Path) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        parsed[row["request_id"]] = {
            "request_id": row["request_id"],
            "clock": int(row["clock"]),
            "class": row["class"],
            "ap": int(row["ap"]),
            "disaggregate": row["disaggregate"].lower() == "true",
            "qp": float(row["qp"]),
            "qd": float(row["qd"]),
            "t_adm_p": float(row["t_adm_p"]),
            "t_adm_d": float(row["t_adm_d"]),
            "ttft_p": float(row["ttft_p"]),
            "ttft_d": float(row["ttft_d"]),
            "var_local_decode": float(row["var_local_decode"]),
            "var_local_colloc_prefill": float(
                row["var_local_colloc_prefill"]
            ),
            "var_disagg_decode": float(row["var_disagg_decode"]),
            "var_disagg_colloc_prefill": float(
                row["var_disagg_colloc_prefill"]
            ),
            "var_disagg_prefill_pool": float(
                row["var_disagg_prefill_pool"]
            ),
            "lhs": float(row["lhs"]),
        }
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=minimax.DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    root = out / "var_prefill_externality_trace"

    all_changes: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "seed": SEED,
        "tuned": False,
        "caveat": (
            "Decision changes are observed on diverging trajectories. "
            "They are candidates for placement-pinned counterfactuals, not "
            "causal request labels."
        ),
        "cells": {},
    }
    for rate_label in CELLS:
        paths = {
            policy: trace_run(out, rate_label, policy, force=args.force)
            for policy in POLICIES
        }
        decisions = {
            policy: parse_decisions(policy_paths["decisions"])
            for policy, policy_paths in paths.items()
        }
        baseline = decisions["var_prefill_nostability"]
        exact = decisions["var_prefill_nostability_exactvar"]
        if set(baseline) != set(exact):
            raise RuntimeError(f"rag:{rate_label}: request IDs differ")
        total = len(baseline)
        changes: list[dict[str, Any]] = []
        for request_id in sorted(
            baseline, key=lambda item: (baseline[item]["clock"], item)
        ):
            old = baseline[request_id]
            new = exact[request_id]
            if old["disaggregate"] == new["disaggregate"]:
                continue
            direction = (
                "local_to_remote"
                if new["disaggregate"]
                else "remote_to_local"
            )
            row = {
                "workload": "rag",
                "rate_label": rate_label,
                "request_id": request_id,
                "direction": direction,
                "baseline_action": (
                    "remote" if old["disaggregate"] else "local"
                ),
                "exact_action": (
                    "remote" if new["disaggregate"] else "local"
                ),
                "baseline_clock": old["clock"],
                "exact_clock": new["clock"],
                "baseline_class": old["class"],
                "exact_class": new["class"],
                "baseline_ap": old["ap"],
                "exact_ap": new["ap"],
            }
            for prefix, source in (("baseline", old), ("exact", new)):
                for field in (
                    "qp",
                    "qd",
                    "t_adm_p",
                    "t_adm_d",
                    "ttft_p",
                    "ttft_d",
                    "var_local_decode",
                    "var_local_colloc_prefill",
                    "var_disagg_decode",
                    "var_disagg_colloc_prefill",
                    "var_disagg_prefill_pool",
                    "lhs",
                ):
                    row[f"{prefix}_{field}"] = source[field]
            changes.append(row)
            all_changes.append(row)

        by_direction: dict[str, Any] = {}
        for direction in ("local_to_remote", "remote_to_local"):
            members = [
                row for row in changes if row["direction"] == direction
            ]
            by_direction[direction] = {
                "count": len(members),
                "share_of_all_requests": len(members) / total if total else 0,
                "baseline_ap": dist(
                    [float(row["baseline_ap"]) for row in members]
                ),
                "baseline_qp": dist(
                    [float(row["baseline_qp"]) for row in members]
                ),
                "exact_qp": dist(
                    [float(row["exact_qp"]) for row in members]
                ),
                "baseline_prefill_pool_var": dist(
                    [
                        float(row["baseline_var_disagg_prefill_pool"])
                        for row in members
                    ]
                ),
                "exact_prefill_pool_var": dist(
                    [
                        float(row["exact_var_disagg_prefill_pool"])
                        for row in members
                    ]
                ),
                "baseline_lhs": dist(
                    [float(row["baseline_lhs"]) for row in members]
                ),
                "exact_lhs": dist(
                    [float(row["exact_lhs"]) for row in members]
                ),
            }
        result["cells"][f"rag:{rate_label}"] = {
            "requests": total,
            "changed": len(changes),
            "changed_share": len(changes) / total if total else 0,
            "by_direction": by_direction,
            "baseline_goodput": float(
                json.loads(paths["var_prefill_nostability"]["metrics"].read_text())[
                    "slo_attainment"
                ]
            ),
            "exact_goodput": float(
                json.loads(
                    paths["var_prefill_nostability_exactvar"][
                        "metrics"
                    ].read_text()
                )["slo_attainment"]
            ),
            "paths": {
                policy: {
                    name: str(path.relative_to(base.ROOT))
                    for name, path in policy_paths.items()
                }
                for policy, policy_paths in paths.items()
            },
        }

    write_csv(root / "changed_decisions.csv", all_changes)
    (root / "trace_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
