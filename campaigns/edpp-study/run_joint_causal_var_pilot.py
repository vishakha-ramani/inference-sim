#!/usr/bin/env python3
"""Paired pilot for corrected causal-VaR joint versus decomposed routing."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import run_decisive_campaign as base
import run_minimax_var_prefill as minimax

ANALYZE = Path(__file__).resolve().parent / "analyze"
sys.path.insert(0, str(ANALYZE))
import joint_routing_var_regret as regret  # noqa: E402

SEEDS = (524287, 1048573, 2097143, 4194301)
TOPOLOGIES = {
    "1p3m": (1, 3),
    "2p2m": (2, 2),
    "3p1m": (3, 1),
}
POLICIES = ("joint_causal_var", "decomposed_causal_var")
CONDITIONS = {
    "synth": (("medium", 1.19372, 500), ("near_high", 1.41754, 500)),
    "rag": (("medium", 3.00917, 500), ("near_high", 3.57339, 550)),
    "shared": (("medium", 64.3361, 2000), ("near_high", 76.3991, 2000)),
}


def write_inventory(path: Path, prefill: int, decode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance_id", "capability"])
        for index in range(prefill):
            writer.writerow([f"instance_{index}", "prefill"])
        for index in range(prefill, prefill + decode):
            writer.writerow([f"instance_{index}", "mixed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=base.CAMPAIGN / "out" / "joint_causal_var_v2",
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    targets = minimax.targets()

    runs: list[base.Run] = []
    for topology in TOPOLOGIES:
        for workload_name, conditions in CONDITIONS.items():
            workload = base.WORKLOADS[workload_name]
            for label, rate, requests in conditions:
                for seed in SEEDS:
                    for policy in POLICIES:
                        runs.append(base.Run(
                            "joint_routing_pilot", workload, label, rate, seed,
                            policy, None, topology, requests,
                            targets[workload_name],
                        ))

    rows = base.run_many(runs, out, args.jobs, args.force)
    base.write_rows(out / "joint_routing_pilot.csv", rows)

    inventories: dict[str, Path] = {}
    for topology, (prefill, decode) in TOPOLOGIES.items():
        path = out / "joint_routing_pilot" / "inventories" / f"{topology}.csv"
        write_inventory(path, prefill, decode)
        inventories[topology] = path

    by_key = {
        (row["topology"], row["workload"], row["rate_label"], int(row["seed"]), row["policy"]): row
        for row in rows
    }
    comparisons = []
    mechanism = []
    for topology in TOPOLOGIES:
        for workload_name, conditions in CONDITIONS.items():
            for label, _, _ in conditions:
                deltas = []
                regrets = []
                reversals = []
                for seed in SEEDS:
                    joint = by_key[(topology, workload_name, label, seed, "joint_causal_var")]
                    decomposed = by_key[(topology, workload_name, label, seed, "decomposed_causal_var")]
                    deltas.append(float(joint["goodput"]) - float(decomposed["goodput"]))
                    tag_run = next(
                        run for run in runs
                        if run.topology == topology and run.workload.name == workload_name
                        and run.rate_label == label and run.seed == seed
                        and run.policy == "decomposed_causal_var"
                    )
                    trace_path = (
                        out / "joint_routing_pilot" / "candidate_traces"
                        / f"{tag_run.tag()}.csv"
                    )
                    report = regret.analyze(inventories[topology], trace_path)
                    regrets.append(float(report["mean_router_induced_var_regret"]))
                    reversals.append(float(report["positive_router_regret_fraction"]))
                uplift, ci_low, ci_high = base.ci95(deltas)
                comparisons.append({
                    "topology": topology,
                    "workload": workload_name,
                    "load": label,
                    "joint_goodput": statistics.fmean(
                        float(by_key[(topology, workload_name, label, seed, "joint_causal_var")]["goodput"])
                        for seed in SEEDS
                    ),
                    "decomposed_goodput": statistics.fmean(
                        float(by_key[(topology, workload_name, label, seed, "decomposed_causal_var")]["goodput"])
                        for seed in SEEDS
                    ),
                    "paired_joint_uplift": uplift,
                    "paired_ci_low": ci_low,
                    "paired_ci_high": ci_high,
                    "wins": sum(delta > 0 for delta in deltas),
                    "ties": sum(delta == 0 for delta in deltas),
                    "losses": sum(delta < 0 for delta in deltas),
                    "mean_router_induced_var_regret": statistics.fmean(regrets),
                    "positive_router_regret_fraction": statistics.fmean(reversals),
                })
                mechanism.extend(regrets)

    result = {
        "status": "four-seed mechanism pilot; rates reused from the frozen 1P2D campaign and are not topology-normalized",
        "seeds": list(SEEDS),
        "policies": list(POLICIES),
        "comparisons": comparisons,
        "all_runs_hard_valid": all(
            int(row["completed"]) + int(row["dropped"]) == int(row["injected"])
            and int(row["still_queued"]) == 0
            and int(row["still_running"]) == 0
            and int(row["timed_out"]) == 0
            for row in rows
        ),
    }
    (out / "joint_routing_pilot_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
