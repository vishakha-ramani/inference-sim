#!/usr/bin/env python3
"""Capacity-normalized confirmation of joint corrected causal VaR."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import run_decisive_campaign as base
import run_minimax_var_prefill as minimax

ANALYZE = Path(__file__).resolve().parent / "analyze"
sys.path.insert(0, str(ANALYZE))
import joint_routing_var_regret as regret  # noqa: E402

SEEDS = (8388593, 16777213, 33554393, 67108859)
TOPOLOGIES = {"1p3m": (1, 3), "2p2m": (2, 2), "3p1m": (3, 1)}
PHIS = (0.0, 0.25, 0.5, 0.75, 1.0)
LOADS = {"medium": 0.80, "near_high": 0.95}
POLICIES = ("joint_causal_var", "decomposed_causal_var")


def hard_valid(row: dict[str, Any]) -> bool:
    return (
        int(row["completed"]) + int(row["dropped"]) == int(row["injected"])
        and int(row["still_queued"]) == 0
        and int(row["still_running"]) == 0
        and int(row["timed_out"]) == 0
        and int(row["length_capped"]) == 0
    )


def write_inventory(path: Path, prefill: int, decode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance_id", "capability"])
        for index in range(prefill):
            writer.writerow([f"instance_{index}", "prefill"])
        for index in range(prefill, prefill + decode):
            writer.writerow([f"instance_{index}", "mixed"])


def capacity_runs(targets: dict[str, Any]) -> list[base.Run]:
    runs = []
    for topology in TOPOLOGIES:
        for workload in base.WORKLOADS.values():
            for phi in PHIS:
                requests = base.CAPACITY_REQUESTS[workload.name]
                if topology == "3p1m" and workload.name == "shared":
                    requests = 2000
                if topology == "3p1m" and workload.name == "rag":
                    requests = 500
                runs.append(base.Run(
                    "joint_routing_capacity", workload, "saturation",
                    workload.saturation_rate, base.CALIBRATION_SEED,
                    "routing_phi", phi, topology,
                    requests, targets[workload.name],
                ))
    return runs


def select_capacity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for topology in TOPOLOGIES:
        selected[topology] = {}
        for workload in base.WORKLOADS:
            members = [
                row for row in rows
                if row["topology"] == topology and row["workload"] == workload
            ]
            if len(members) != len(PHIS):
                raise RuntimeError(f"{topology}/{workload}: incomplete capacity sweep")
            if not all(hard_valid(row) for row in members):
                raise RuntimeError(f"{topology}/{workload}: hard-invalid capacity run")
            eligible = [row for row in members if int(row["dropped"]) == 0]
            if not eligible:
                raise RuntimeError(
                    f"{topology}/{workload}: no zero-drop capacity arm"
                )
            best = max(
                eligible, key=lambda row: float(row["central_completion_rps"])
            )
            ceiling = float(best["central_completion_rps"])
            if ceiling <= 0:
                raise RuntimeError(f"{topology}/{workload}: non-positive capacity")
            selected[topology][workload] = {
                "ceiling_rps": ceiling,
                "selected_phi": float(best["parameter"]),
                "rates": {name: fraction * ceiling for name, fraction in LOADS.items()},
                "sweep": [
                    {
                        "phi": float(row["parameter"]),
                        "central_completion_rps": float(row["central_completion_rps"]),
                        "dropped": int(row["dropped"]),
                    }
                    for row in sorted(members, key=lambda row: float(row["parameter"]))
                ],
            }
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=base.CAMPAIGN / "out" / "joint_causal_var_confirmation_v2",
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    targets = minimax.targets()

    cap_rows = base.run_many(capacity_runs(targets), out, args.jobs, args.force)
    base.write_rows(out / "capacity.csv", cap_rows)
    selected = select_capacity(cap_rows)
    (out / "capacity_selection.json").write_text(json.dumps(selected, indent=2) + "\n")

    runs: list[base.Run] = []
    for topology in TOPOLOGIES:
        for workload_name, workload in base.WORKLOADS.items():
            for label, rate in selected[topology][workload_name]["rates"].items():
                if workload_name == "shared":
                    requests = base.POLICY_MIN_REQUESTS[workload_name]
                else:
                    requests = max(
                        base.POLICY_MIN_REQUESTS[workload_name],
                        int(math.ceil(rate * base.ARRIVAL_SECONDS)),
                    )
                for seed in SEEDS:
                    for policy in POLICIES:
                        runs.append(base.Run(
                            "joint_routing_confirmation", workload, label, rate,
                            seed, policy, None, topology, requests,
                            targets[workload_name],
                        ))
    rows = base.run_many(runs, out, args.jobs, args.force)
    if not all(hard_valid(row) for row in rows):
        bad = [
            f"{row['topology']}/{row['workload']}/{row['rate_label']}/{row['policy']}/s{row['seed']}"
            for row in rows if not hard_valid(row)
        ]
        raise RuntimeError(f"hard-invalid confirmation runs: {', '.join(bad)}")
    base.write_rows(out / "confirmation.csv", rows)

    inventories: dict[str, Path] = {}
    for topology, (prefill, decode) in TOPOLOGIES.items():
        path = out / "inventories" / f"{topology}.csv"
        write_inventory(path, prefill, decode)
        inventories[topology] = path

    by_key = {
        (row["topology"], row["workload"], row["rate_label"], int(row["seed"]), row["policy"]): row
        for row in rows
    }
    comparisons = []
    for topology in TOPOLOGIES:
        for workload in base.WORKLOADS:
            for label in LOADS:
                deltas, regrets, regret_fractions = [], [], []
                for seed in SEEDS:
                    joint = by_key[(topology, workload, label, seed, "joint_causal_var")]
                    decomposed = by_key[(topology, workload, label, seed, "decomposed_causal_var")]
                    deltas.append(float(joint["goodput"]) - float(decomposed["goodput"]))
                    run = next(
                        item for item in runs
                        if item.topology == topology and item.workload.name == workload
                        and item.rate_label == label and item.seed == seed
                        and item.policy == "decomposed_causal_var"
                    )
                    trace_path = (
                        out / run.phase / "candidate_traces" / f"{run.tag()}.csv"
                    )
                    report = regret.analyze(inventories[topology], trace_path)
                    regrets.append(float(report["mean_router_induced_var_regret"]))
                    regret_fractions.append(float(report["positive_router_regret_fraction"]))
                uplift, low, high = base.ci95(deltas)
                comparisons.append({
                    "topology": topology, "workload": workload, "load": label,
                    "rate": selected[topology][workload]["rates"][label],
                    "joint_goodput": statistics.fmean(
                        float(by_key[(topology, workload, label, seed, "joint_causal_var")]["goodput"])
                        for seed in SEEDS
                    ),
                    "decomposed_goodput": statistics.fmean(
                        float(by_key[(topology, workload, label, seed, "decomposed_causal_var")]["goodput"])
                        for seed in SEEDS
                    ),
                    "paired_joint_uplift": uplift,
                    "paired_ci_low": low, "paired_ci_high": high,
                    "wins": sum(value > 0 for value in deltas),
                    "ties": sum(value == 0 for value in deltas),
                    "losses": sum(value < 0 for value in deltas),
                    "mean_router_induced_var_regret": statistics.fmean(regrets),
                    "positive_router_regret_fraction": statistics.fmean(regret_fractions),
                })

    result = {
        "status": "capacity-normalized confirmation; policy and seeds frozen before execution",
        "seeds": list(SEEDS),
        "loads": LOADS,
        "capacity": selected,
        "comparisons": comparisons,
        "hard_valid_runs": len(rows),
        "strict_zero_drop_failures": [
            {
                "topology": row["topology"], "workload": row["workload"],
                "load": row["rate_label"], "policy": row["policy"],
                "seed": int(row["seed"]), "dropped": int(row["dropped"]),
            }
            for row in rows if int(row["dropped"]) > 0
        ],
    }
    (out / "confirmation_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
