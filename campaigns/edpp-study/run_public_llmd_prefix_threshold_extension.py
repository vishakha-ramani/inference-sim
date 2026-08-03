#!/usr/bin/env python3
"""Append a workload-tuned llm-d prefix-threshold comparator.

Protocol: PUBLIC-LLMD-PREFIX-THRESHOLD-EXTENSION-PROTOCOL.md.
Historical policy runs are read only during post-confirmation analysis. The
calibration and threshold-selection stages cannot read held-out artifacts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import run_decisive_campaign as base
import run_public_load_static_benchmark as benchmark
import run_public_workload_heterogeneity_closeout as public
import run_slo_externality_joint_campaign as diagnostic


DEFAULT_OUT = base.CAMPAIGN / "out" / "public_llmd_prefix_threshold_extension_v1"
SOURCE_OUT = base.CAMPAIGN / "out" / "public_load_static_benchmark_v1"
PROTOCOL = base.CAMPAIGN / "PUBLIC-LLMD-PREFIX-THRESHOLD-EXTENSION-PROTOCOL.md"
POLICY = "llmd_prefix_threshold_workload_tuned"
PAPER_KAIROS = "kairos_paper_alpha_1p3"
SCORERS = "precise-prefix-cache:2,queue-depth:1"
CACHE_SIGNAL_DELAY_US = 50_000
THRESHOLDS = (0, 16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
DEVELOPMENT_SEEDS = (42, 123)
CONFIRMATION_SEEDS = benchmark.CONFIRMATION_SEEDS
UNIFIED_SEEDS = (1000000007, 1000000009, 1000000033, 1000000087)
FOCAL = benchmark.FOCAL
RETROSPECTIVE_DEPLOYABLE = (*benchmark.DEPLOYABLE, POLICY)
UNIFIED_EXISTING_POLICIES = (
    FOCAL,
    "least_ttft_joint",
    PAPER_KAIROS,
    "static_joint_yardstick",
    "capacity_static",
)
UNIFIED_DEPLOYABLE = (FOCAL, "least_ttft_joint", PAPER_KAIROS, POLICY)


@dataclass(frozen=True)
class Run:
    phase: str
    config: public.WorkloadConfig
    rate_label: str
    rate: float
    seed: int
    hardware: str
    requests: int
    threshold: int

    def tag(self) -> str:
        return (
            f"{self.hardware}_{self.config.workload.name}_{self.rate_label}_"
            f"{self.rate:.7g}_{POLICY}_t{self.threshold}_"
            f"s{self.seed}_n{self.requests}"
        )


def topology_args(hardware: str) -> list[str]:
    args = [
        "--num-instances", "3",
        "--prefill-instances", "1",
        "--decode-instances", "2",
        "--prefill-routing-scorers", SCORERS,
        "--decode-routing-scorers", SCORERS,
        "--cache-signal-delay", str(CACHE_SIGNAL_DELAY_US),
        "--max-num-running-reqs", "256",
    ]
    if hardware == "h100_a100_realistic":
        args += ["--policy-config", str(public.HETERO_BUNDLE)]
    elif hardware != "h100_homogeneous":
        raise ValueError(f"unknown hardware {hardware}")
    return args


def execute_run(run: Run, out: Path, force: bool) -> dict[str, Any]:
    run_dir = out / run.phase
    spec_path = run_dir / "specs" / f"{run.tag()}.yaml"
    metrics_path = run_dir / "metrics" / f"{run.tag()}.json"
    stdout_path = run_dir / "stdout" / f"{run.tag()}.txt"
    public.make_spec(run, spec_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and metrics_path.exists() and stdout_path.exists():
        row = base.result_row(
            phase=run.phase,
            workload=run.config.workload.name,
            rate_label=run.rate_label,
            rate=run.rate,
            seed=run.seed,
            policy=POLICY,
            parameter=run.threshold,
            topology=f"1p2d:{run.hardware}",
            metrics_path=metrics_path,
            stdout_path=stdout_path,
        )
    else:
        command = [
            str(base.ROOT / "blis"), "run",
            "--model", base.MODEL,
            "--workload-spec", str(spec_path),
            "--num-requests", str(run.requests),
            *topology_args(run.hardware),
            *base.slo_args(run.config.targets),
            "--pd-decider", "prefix-threshold",
            "--pd-prefix-threshold", str(run.threshold),
            "--timeout", "-1",
            "--seed", str(run.seed),
            "--metrics-path", str(metrics_path),
        ]
        proc = subprocess.run(command, cwd=base.ROOT, capture_output=True, text=True)
        stdout_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                f"run failed ({run.tag()}):\n{proc.stderr[-5000:]}\n{proc.stdout[-5000:]}"
            )
        row = base.result_row(
            phase=run.phase,
            workload=run.config.workload.name,
            rate_label=run.rate_label,
            rate=run.rate,
            seed=run.seed,
            policy=POLICY,
            parameter=run.threshold,
            topology=f"1p2d:{run.hardware}",
            metrics_path=metrics_path,
            stdout_path=stdout_path,
        )

    row["hardware"] = run.hardware
    row["threshold"] = run.threshold
    row["zero_drop"] = int(row["dropped"]) == 0
    row["hard_valid"] = public.hard_valid(row)
    return row


def run_many(runs: Iterable[Run], out: Path, jobs: int, force: bool) -> list[dict[str, Any]]:
    runs = list(runs)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(execute_run, run, out, force): run for run in runs}
        for index, future in enumerate(as_completed(futures), 1):
            run = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/{len(runs)}] {run.tag()} goodput={float(row['goodput']):.3f} "
                f"remote={float(row['realized_phi']):.3f} valid={row['hard_valid']}",
                flush=True,
            )
    return sorted(
        rows,
        key=lambda row: (
            str(row["workload"]), str(row["hardware"]), str(row["rate_label"]),
            int(row["threshold"]), int(row["seed"]),
        ),
    )


def require_valid(rows: list[dict[str, Any]], label: str) -> None:
    bad = [
        f"{row['hardware']}/{row['workload']}/{row['rate_label']}/t{row['threshold']}/s{row['seed']}"
        for row in rows
        if not bool(row["hard_valid"]) or int(row["dropped"]) != 0
    ]
    if bad:
        raise RuntimeError(f"{label} has invalid runs: {', '.join(bad)}")


def load_capacity() -> dict[str, Any]:
    path = SOURCE_OUT / "capacity_selection.json"
    if not path.exists():
        raise SystemExit(f"missing frozen benchmark capacity artifact: {path}")
    return json.loads(path.read_text())


def make_runs(phase: str, seeds: tuple[int, ...], thresholds: dict[str, tuple[int, ...]]) -> list[Run]:
    capacity = load_capacity()
    runs: list[Run] = []
    for hardware in public.HARDWARES:
        for workload, config in public.WORKLOADS.items():
            for rate_label, _ in benchmark.LOADS:
                rate = float(capacity[hardware][workload]["evaluation_rates"][rate_label])
                for threshold in thresholds[workload]:
                    for seed in seeds:
                        runs.append(Run(
                            phase, config, rate_label, rate, seed, hardware,
                            config.evaluation_requests, threshold,
                        ))
    return runs


def select_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # This function deliberately accepts calibration rows only. It has no path or
    # argument through which historical held-out artifacts can be read.
    expected = 2 * len(public.HARDWARES) * len(benchmark.LOADS)
    result: dict[str, Any] = {
        "status": "frozen before matched held-out extension",
        "selection_metric": "equal-run mean hard composite goodput",
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "scorers_both_pools": SCORERS,
        "cache_signal_delay_us": CACHE_SIGNAL_DELAY_US,
        "workloads": {},
    }
    for workload in public.WORKLOADS:
        grid = []
        for threshold in THRESHOLDS:
            members = [
                row for row in rows
                if row["workload"] == workload and int(row["threshold"]) == threshold
            ]
            if len(members) != expected:
                raise RuntimeError(
                    f"incomplete calibration {workload}/t{threshold}: {len(members)}, want {expected}"
                )
            seeds = {int(row["seed"]) for row in members}
            if seeds != set(DEVELOPMENT_SEEDS):
                raise RuntimeError(f"wrong calibration seeds {workload}/t{threshold}: {seeds}")
            values = [float(row["goodput"]) for row in members]
            eligible = all(bool(row["hard_valid"]) and int(row["dropped"]) == 0 for row in members)
            grid.append({
                "threshold": threshold,
                "mean_goodput": statistics.fmean(values),
                "stdev_goodput": statistics.stdev(values),
                "minimum_goodput": min(values),
                "eligible": eligible,
            })
        eligible = [item for item in grid if item["eligible"]]
        if not eligible:
            raise RuntimeError(f"no eligible threshold for {workload}")
        winner = max(
            eligible,
            key=lambda item: (
                float(item["mean_goodput"]),
                -float(item["stdev_goodput"]),
                int(item["threshold"] == 16),
                -int(item["threshold"]),
            ),
        )
        result["workloads"][workload] = {
            "selected_threshold": int(winner["threshold"]),
            "calibration_mean_goodput": float(winner["mean_goodput"]),
            "calibration_stdev_goodput": float(winner["stdev_goodput"]),
            "grid": grid,
        }
    return result


def run_calibration(out: Path, jobs: int, force: bool) -> None:
    runs = make_runs(
        "llmd_threshold_calibration",
        DEVELOPMENT_SEEDS,
        {workload: THRESHOLDS for workload in public.WORKLOADS},
    )
    if len(runs) != 468:
        raise RuntimeError(f"calibration has {len(runs)} runs, want 468")
    rows = run_many(runs, out, jobs, force)
    require_valid(rows, "threshold calibration")
    public.write_rows(out / "calibration.csv", rows)
    selection = select_thresholds(rows)
    (out / "threshold_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2), flush=True)


def load_selection(out: Path) -> dict[str, Any]:
    path = out / "threshold_selection.json"
    if not path.exists():
        raise SystemExit("threshold selection missing; run --stage calibrate")
    return json.loads(path.read_text())


def confirmation_runs(out: Path) -> list[Run]:
    selection = load_selection(out)
    thresholds = {
        workload: (int(selection["workloads"][workload]["selected_threshold"]),)
        for workload in public.WORKLOADS
    }
    return make_runs("llmd_threshold_matched_confirmation", CONFIRMATION_SEEDS, thresholds)


def analyze_confirmation(out: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selection = load_selection(out)
    historical_path = SOURCE_OUT / "confirmation_result.json"
    if not historical_path.exists():
        raise SystemExit(f"missing historical result: {historical_path}")
    historical = json.loads(historical_path.read_text())

    means: dict[str, float] = {}
    remote: dict[str, float] = {}
    per_seed: dict[str, list[dict[str, Any]]] = {}
    for hardware in public.HARDWARES:
        for workload in public.WORKLOADS:
            for rate_label, _ in benchmark.LOADS:
                cell = f"{hardware}:{workload}:{rate_label}"
                members = [
                    row for row in rows
                    if row["hardware"] == hardware
                    and row["workload"] == workload
                    and row["rate_label"] == rate_label
                ]
                members.sort(key=lambda row: int(row["seed"]))
                if len(members) != len(CONFIRMATION_SEEDS):
                    raise RuntimeError(f"incomplete confirmation {cell}: {len(members)}")
                if tuple(int(row["seed"]) for row in members) != tuple(sorted(CONFIRMATION_SEEDS)):
                    raise RuntimeError(f"wrong confirmation seeds for {cell}")
                means[cell] = statistics.fmean(float(row["goodput"]) for row in members)
                remote[cell] = statistics.fmean(float(row["realized_phi"]) for row in members)
                per_seed[cell] = [
                    {
                        "seed": int(row["seed"]),
                        "goodput": float(row["goodput"]),
                        "remote_fraction": float(row["realized_phi"]),
                    }
                    for row in members
                ]

    merged_means = {
        cell: {**policy_means, POLICY: means[cell]}
        for cell, policy_means in historical["policy_means"].items()
    }
    best = {
        cell: max(merged_means[cell][policy] for policy in RETROSPECTIVE_DEPLOYABLE)
        for cell in merged_means
    }
    ranking = []
    for policy in RETROSPECTIVE_DEPLOYABLE:
        regrets = {cell: best[cell] - merged_means[cell][policy] for cell in merged_means}
        ranking.append({
            "policy": policy,
            "worst_regret": max(regrets.values()),
            "worst_cell": max(regrets, key=regrets.get),
            "mean_goodput": statistics.fmean(merged_means[cell][policy] for cell in merged_means),
            "regret_by_cell": regrets,
        })
    ranking.sort(key=lambda item: (item["worst_regret"], -item["mean_goodput"], item["policy"]))

    deltas = []
    by_cell = []
    for cell in merged_means:
        left = {int(item["seed"]): float(item["goodput"]) for item in historical["per_seed"][cell][FOCAL]}
        right = {int(item["seed"]): float(item["goodput"]) for item in per_seed[cell]}
        cell_deltas = [left[seed] - right[seed] for seed in CONFIRMATION_SEEDS]
        mean, low, high = base.ci95(cell_deltas)
        by_cell.append({"cell": cell, "mean_delta": mean, "ci_low": low, "ci_high": high})
        deltas.extend(cell_deltas)
    mean, low, high = base.ci95(deltas)

    return {
        "status": "retrospective paired comparator extension; not fresh-seed confirmation",
        "historical_result": str(historical_path),
        "confirmation_seeds_reused": list(CONFIRMATION_SEEDS),
        "threshold_selection": selection,
        "run_count": len(rows),
        "cell_count": len(means),
        "policy_means": means,
        "remote_fractions": remote,
        "per_seed": per_seed,
        "merged_policy_means": merged_means,
        "merged_deployable_minimax_ranking": ranking,
        "focal_minus_llmd": {
            "all_cells": {"mean_delta": mean, "ci_low": low, "ci_high": high},
            "by_cell": by_cell,
        },
        "hard_invalid_runs": sum(not bool(row["hard_valid"]) for row in rows),
        "runs_with_drops": sum(int(row["dropped"]) > 0 for row in rows),
        "runs_with_timeouts": sum(int(row["timed_out"]) > 0 for row in rows),
        "runs_with_length_caps": sum(int(row["length_capped"]) > 0 for row in rows),
    }


def write_report(out: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Workload-tuned llm-d prefix-threshold extension", "",
        "This is a paired retrospective addition to the completed public benchmark,",
        "not a new fresh-seed confirmation. Historical policy runs were not rerun.", "",
        "## Frozen workload thresholds", "",
        "| workload | selected threshold | calibration mean goodput | calibration stdev |",
        "|---|---:|---:|---:|",
    ]
    for workload, item in result["threshold_selection"]["workloads"].items():
        lines.append(
            f"| {workload.replace('_', ' ')} | {item['selected_threshold']} | "
            f"{item['calibration_mean_goodput']:.4f} | {item['calibration_stdev_goodput']:.4f} |"
        )
    lines += [
        "", "## Matched held-out result", "",
        "| cell | mean goodput | remote fraction |",
        "|---|---:|---:|",
    ]
    for cell, value in result["policy_means"].items():
        lines.append(f"| `{cell}` | {value:.3f} | {result['remote_fractions'][cell]:.3f} |")
    lines += [
        "", "## Merged deployable minimax ranking", "",
        "| rank | policy | worst shortfall | worst cell | equal-cell mean goodput |",
        "|---:|---|---:|---|---:|",
    ]
    for index, item in enumerate(result["merged_deployable_minimax_ranking"], 1):
        lines.append(
            f"| {index} | `{item['policy']}` | {item['worst_regret']:.4f} | "
            f"`{item['worst_cell']}` | {item['mean_goodput']:.4f} |"
        )
    delta = result["focal_minus_llmd"]["all_cells"]
    lines += [
        "", "## Paired comparison", "",
        "Positive means causal externality has higher goodput than the tuned llm-d-style policy.", "",
        f"Mean paired difference: `{delta['mean_delta']:+.4f}` "
        f"with 95% interval `[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]`.", "",
        "## Validity", "",
        f"- Runs: {result['run_count']} across {result['cell_count']} cells.",
        f"- Hard-invalid runs: {result['hard_invalid_runs']}.",
        f"- Runs with drops: {result['runs_with_drops']}.",
        f"- Runs with timeouts: {result['runs_with_timeouts']}.",
        f"- Runs with length caps: {result['runs_with_length_caps']}.", "",
        "The tuned threshold is a restricted, workload-informed baseline, not an oracle.", "",
    ]
    (out / "PUBLIC-LLMD-PREFIX-THRESHOLD-EXTENSION.md").write_text("\n".join(lines))


def run_confirmation(out: Path, jobs: int, force: bool) -> None:
    runs = confirmation_runs(out)
    if len(runs) != 72:
        raise RuntimeError(f"confirmation has {len(runs)} runs, want 72")
    rows = run_many(runs, out, jobs, force)
    require_valid(rows, "matched held-out extension")
    public.write_rows(out / "confirmation.csv", rows)
    result = analyze_confirmation(out, rows)
    (out / "confirmation_result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(out, result)
    print(json.dumps(result, indent=2), flush=True)


def analyze_unified(
    out: Path,
    old_runs: list[public.Run],
    old_rows: list[dict[str, Any]],
    llmd_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = old_rows + llmd_rows
    policies = (*UNIFIED_EXISTING_POLICIES, POLICY)
    cells = [
        f"{hardware}:{workload}:{rate_label}"
        for hardware in public.HARDWARES
        for workload in public.WORKLOADS
        for rate_label, _ in benchmark.LOADS
    ]
    means: dict[str, dict[str, float]] = {}
    remote: dict[str, dict[str, float]] = {}
    per_seed: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    for cell in cells:
        hardware, workload, rate_label = cell.split(":")
        means[cell] = {}
        remote[cell] = {}
        per_seed[cell] = {}
        for policy in policies:
            members = [
                row for row in rows
                if row["hardware"] == hardware
                and row["workload"] == workload
                and row["rate_label"] == rate_label
                and row["policy"] == policy
            ]
            members.sort(key=lambda row: int(row["seed"]))
            if len(members) != len(UNIFIED_SEEDS):
                raise RuntimeError(f"incomplete unified cell {cell}/{policy}: {len(members)}")
            if tuple(int(row["seed"]) for row in members) != tuple(sorted(UNIFIED_SEEDS)):
                raise RuntimeError(f"wrong unified seeds for {cell}/{policy}")
            means[cell][policy] = statistics.fmean(float(row["goodput"]) for row in members)
            remote[cell][policy] = statistics.fmean(float(row["realized_phi"]) for row in members)
            per_seed[cell][policy] = [
                {
                    "seed": int(row["seed"]),
                    "goodput": float(row["goodput"]),
                    "remote_fraction": float(row["realized_phi"]),
                }
                for row in members
            ]

    best = {cell: max(means[cell][policy] for policy in UNIFIED_DEPLOYABLE) for cell in cells}
    ranking = []
    for policy in UNIFIED_DEPLOYABLE:
        regrets = {cell: best[cell] - means[cell][policy] for cell in cells}
        ranking.append({
            "policy": policy,
            "worst_regret": max(regrets.values()),
            "worst_cell": max(regrets, key=regrets.get),
            "mean_goodput": statistics.fmean(means[cell][policy] for cell in cells),
            "regret_by_cell": regrets,
        })
    ranking.sort(key=lambda item: (item["worst_regret"], -item["mean_goodput"], item["policy"]))

    paired: dict[str, Any] = {}
    for comparator in policies:
        if comparator == FOCAL:
            continue
        deltas = []
        by_cell = []
        for cell in cells:
            left = {int(item["seed"]): float(item["goodput"]) for item in per_seed[cell][FOCAL]}
            right = {int(item["seed"]): float(item["goodput"]) for item in per_seed[cell][comparator]}
            cell_deltas = [left[seed] - right[seed] for seed in UNIFIED_SEEDS]
            mean, low, high = base.ci95(cell_deltas)
            by_cell.append({"cell": cell, "mean_delta": mean, "ci_low": low, "ci_high": high})
            deltas.extend(cell_deltas)
        mean, low, high = base.ci95(deltas)
        paired[comparator] = {
            "all_cells": {"mean_delta": mean, "ci_low": low, "ci_high": high},
            "by_cell": by_cell,
        }

    trace_runs = 0
    trace_exact = True
    for run in old_runs:
        if run.policy != FOCAL:
            continue
        summary = diagnostic.analyze_candidate_trace(public.trace_path(out, run), public.V)
        trace_runs += 1
        trace_exact = trace_exact and summary["positive_chosen_snapshot_score_regret_fraction"] == 0

    return {
        "status": "unified six-policy confirmation on previously unused seeds",
        "unified_seeds": list(UNIFIED_SEEDS),
        "threshold_selection": load_selection(out),
        "run_count": len(rows),
        "cell_count": len(cells),
        "policy_means": means,
        "remote_fractions": remote,
        "per_seed": per_seed,
        "deployable_minimax_ranking": ranking,
        "focal_paired_deltas": paired,
        "hard_invalid_runs": sum(not bool(row["hard_valid"]) for row in rows),
        "runs_with_drops": sum(int(row["dropped"]) > 0 for row in rows),
        "runs_with_timeouts": sum(int(row["timed_out"]) > 0 for row in rows),
        "runs_with_length_caps": sum(int(row["length_capped"]) > 0 for row in rows),
        "focal_candidate_trace_runs": trace_runs,
        "chosen_argmin_trace_exact": trace_exact,
    }


def write_unified_report(out: Path, result: dict[str, Any]) -> None:
    labels = {
        FOCAL: "causal externality",
        "least_ttft_joint": "joint least-TTFT",
        PAPER_KAIROS: "Kairos (paper, alpha=1.3)",
        POLICY: "workload-tuned llm-d threshold",
        "static_joint_yardstick": "goodput-tuned static",
        "capacity_static": "capacity-selected static",
    }
    lines = [
        "# Unified public benchmark with workload-tuned llm-d threshold", "",
        "All six policies use the same four previously unused request-trace seeds.",
        "All capacities, rates, plans, thresholds, and policy settings were frozen first.", "",
        "## Frozen thresholds", "",
        "| workload | threshold |", "|---|---:|",
    ]
    for workload, item in result["threshold_selection"]["workloads"].items():
        lines.append(f"| {workload.replace('_', ' ')} | {item['selected_threshold']} |")
    lines += [
        "", "## Mean goodput", "",
        "| cell | causal externality | least-TTFT | Kairos | llm-d threshold | goodput static | capacity static |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell, means in result["policy_means"].items():
        lines.append(
            f"| `{cell}` | {means[FOCAL]:.3f} | {means['least_ttft_joint']:.3f} | "
            f"{means[PAPER_KAIROS]:.3f} | {means[POLICY]:.3f} | "
            f"{means['static_joint_yardstick']:.3f} | {means['capacity_static']:.3f} |"
        )
    lines += [
        "", "## Deployable minimax ranking", "",
        "| rank | policy | worst shortfall | worst cell | equal-cell mean goodput |",
        "|---:|---|---:|---|---:|",
    ]
    for index, item in enumerate(result["deployable_minimax_ranking"], 1):
        lines.append(
            f"| {index} | {labels[item['policy']]} | {item['worst_regret']:.4f} | "
            f"`{item['worst_cell']}` | {item['mean_goodput']:.4f} |"
        )
    lines += ["", "## Focal paired differences", "", "| comparator | mean | 95% interval |", "|---|---:|---:|"]
    for comparator, item in result["focal_paired_deltas"].items():
        value = item["all_cells"]
        lines.append(
            f"| {labels[comparator]} | {value['mean_delta']:+.4f} | "
            f"[{value['ci_low']:+.4f}, {value['ci_high']:+.4f}] |"
        )
    lines += [
        "", "## Validity", "",
        f"- Runs: {result['run_count']} across {result['cell_count']} cells.",
        f"- Hard-invalid runs: {result['hard_invalid_runs']}.",
        f"- Runs with drops: {result['runs_with_drops']}.",
        f"- Runs with timeouts: {result['runs_with_timeouts']}.",
        f"- Runs with length caps: {result['runs_with_length_caps']}.",
        f"- Focal candidate traces: {result['focal_candidate_trace_runs']}; "
        + ("all exact argmins." if result["chosen_argmin_trace_exact"] else "ARGMIN FAILURE."), "",
    ]
    (out / "PUBLIC-LLMD-PREFIX-THRESHOLD-UNIFIED-CONFIRMATION.md").write_text("\n".join(lines))


def run_unified(out: Path, jobs: int, force: bool) -> None:
    capacity = json.loads((SOURCE_OUT / "capacity_selection.json").read_text())
    static = json.loads((SOURCE_OUT / "static_goodput_selection.json").read_text())
    old_seed_block = benchmark.CONFIRMATION_SEEDS
    old_policy_block = benchmark.CONFIRM_POLICIES
    benchmark.CONFIRMATION_SEEDS = UNIFIED_SEEDS
    benchmark.CONFIRM_POLICIES = UNIFIED_EXISTING_POLICIES
    try:
        old_runs = [
            replace(run, phase="public_load_static_unified_confirmation")
            for run in benchmark.confirmation_runs(capacity, static)
        ]
    finally:
        benchmark.CONFIRMATION_SEEDS = old_seed_block
        benchmark.CONFIRM_POLICIES = old_policy_block
    if len(old_runs) != 360:
        raise RuntimeError(f"unified historical-policy block has {len(old_runs)} runs, want 360")
    old_rows = public.run_many(old_runs, out, jobs, force)
    public.require_hard_valid(old_rows, "unified historical-policy block")
    public.write_rows(out / "unified_existing_policies.csv", old_rows)

    selection = load_selection(out)
    thresholds = {
        workload: (int(selection["workloads"][workload]["selected_threshold"]),)
        for workload in public.WORKLOADS
    }
    llmd_runs = make_runs("llmd_threshold_unified_confirmation", UNIFIED_SEEDS, thresholds)
    if len(llmd_runs) != 72:
        raise RuntimeError(f"unified llm-d block has {len(llmd_runs)} runs, want 72")
    llmd_rows = run_many(llmd_runs, out, jobs, force)
    require_valid(llmd_rows, "unified llm-d block")
    public.write_rows(out / "unified_llmd.csv", llmd_rows)

    result = analyze_unified(out, old_runs, old_rows, llmd_rows)
    (out / "unified_confirmation_result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_unified_report(out, result)
    print(json.dumps(result, indent=2), flush=True)


def smoke(out: Path, force: bool) -> None:
    capacity = load_capacity()
    config = public.WORKLOADS["interactive"]
    rate = float(capacity["h100_homogeneous"]["interactive"]["evaluation_rates"]["low_0p60"])
    runs = [
        Run("llmd_threshold_smoke", config, "low_0p60", rate, 314159, "h100_homogeneous", 24, threshold)
        for threshold in (16, 4096)
    ]
    rows = run_many(runs, out, jobs=2, force=force)
    require_valid(rows, "smoke")
    public.write_rows(out / "smoke.csv", rows)
    print("smoke passed: exact terminal accounting for both thresholds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stage", choices=("smoke", "calibrate", "confirm", "unified", "all"), default="all"
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not PROTOCOL.exists():
        raise SystemExit(f"missing frozen protocol {PROTOCOL}")
    if not public.HETERO_BUNDLE.exists():
        raise SystemExit(f"missing hardware bundle {public.HETERO_BUNDLE}")
    if args.stage == "smoke":
        smoke(out, args.force)
        return
    if args.stage in {"calibrate", "all"}:
        run_calibration(out, args.jobs, args.force)
    if args.stage in {"confirm", "all"}:
        run_confirmation(out, args.jobs, args.force)
    if args.stage in {"unified", "all"}:
        run_unified(out, args.jobs, args.force)


if __name__ == "__main__":
    main()
