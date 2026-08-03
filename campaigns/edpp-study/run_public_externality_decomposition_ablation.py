#!/usr/bin/env python3
"""Frozen public-workload externality and decode-decomposition ablation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import run_decisive_campaign as base
import run_public_load_static_benchmark as benchmark
import run_public_workload_heterogeneity_closeout as public


DEFAULT_OUT = (
    base.CAMPAIGN / "out" / "public_externality_decomposition_ablation_ttft_rollout_v2"
)
CAPACITY_SOURCE = (
    base.CAMPAIGN
    / "out"
    / "public_load_static_benchmark_ttft_rollout_v2"
    / "capacity_selection.json"
)
PROTOCOL = base.CAMPAIGN / "PUBLIC-EXTERNALITY-DECOMPOSITION-ABLATION-PROTOCOL.md"

V = 8.0
SEEDS = (67108879, 134217757, 268435459, 536870923)
POLICIES = (
    "joint_full",
    "joint_own_only",
    "joint_resident_only",
    "decode_first_full",
)
FOCAL = "joint_full"
DISPLAY = {
    "joint_full": "joint full",
    "joint_own_only": "joint own-only",
    "joint_resident_only": "joint resident-only",
    "decode_first_full": "decode-first full",
}


def load_capacity() -> dict[str, Any]:
    if not CAPACITY_SOURCE.exists():
        raise SystemExit(f"missing frozen capacity source: {CAPACITY_SOURCE}")
    return json.loads(CAPACITY_SOURCE.read_text())


def policy_args(run: public.Run) -> list[str]:
    targets = run.config.targets
    if run.policy == "joint_full":
        return base.policy_args(
            "joint_slo_externality_no_capacity", targets, V, None
        )
    if run.policy == "joint_own_only":
        return base.policy_args(
            "joint_slo_externality_no_capacity", targets, V, None
        ) + ["--edpp-slo-externality-no-externality"]
    if run.policy == "joint_resident_only":
        return base.policy_args(
            "joint_slo_externality_no_capacity", targets, V, None
        ) + ["--edpp-slo-externality-no-own-good"]
    if run.policy == "decode_first_full":
        return base.policy_args(
            "decomposed_slo_externality", targets, V, None
        ) + ["--edpp-slo-externality-no-capacity"]
    raise ValueError(f"unknown policy {run.policy}")


def execute_run(run: public.Run, out: Path, force: bool) -> dict[str, Any]:
    run_dir = out / run.phase
    spec_path = run_dir / "specs" / f"{run.tag()}.yaml"
    metrics_path = run_dir / "metrics" / f"{run.tag()}.json"
    stdout_path = run_dir / "stdout" / f"{run.tag()}.txt"
    candidate_path = public.trace_path(out, run)
    public.make_spec(run, spec_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        not force
        and metrics_path.exists()
        and stdout_path.exists()
        and candidate_path.exists()
    ):
        row = base.result_row(
            phase=run.phase,
            workload=run.config.workload.name,
            rate_label=run.rate_label,
            rate=run.rate,
            seed=run.seed,
            policy=run.policy,
            parameter=None,
            topology=f"1p2d:{run.hardware}",
            metrics_path=metrics_path,
            stdout_path=stdout_path,
        )
    else:
        command = [
            str(base.ROOT / "blis"),
            "run",
            "--model",
            base.MODEL,
            "--workload-spec",
            str(spec_path),
            "--num-requests",
            str(run.requests),
            *public.topology_args(run.hardware),
            *base.slo_args(run.config.targets),
            *policy_args(run),
            "--timeout",
            "-1",
            "--seed",
            str(run.seed),
            "--edpp-joint-candidate-trace",
            str(candidate_path),
            "--metrics-path",
            str(metrics_path),
        ]
        proc = subprocess.run(
            command, cwd=base.ROOT, capture_output=True, text=True
        )
        stdout_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                f"run failed ({run.tag()}):\n"
                f"{proc.stderr[-5000:]}\n{proc.stdout[-5000:]}"
            )
        row = base.result_row(
            phase=run.phase,
            workload=run.config.workload.name,
            rate_label=run.rate_label,
            rate=run.rate,
            seed=run.seed,
            policy=run.policy,
            parameter=None,
            topology=f"1p2d:{run.hardware}",
            metrics_path=metrics_path,
            stdout_path=stdout_path,
        )

    row["hardware"] = run.hardware
    row["hard_valid"] = public.hard_valid(row)
    return row


def run_many(
    runs: Iterable[public.Run], out: Path, jobs: int, force: bool
) -> list[dict[str, Any]]:
    planned = list(runs)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(execute_run, run, out, force): run for run in planned
        }
        for index, future in enumerate(as_completed(futures), 1):
            run = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/{len(planned)}] {run.tag()} "
                f"goodput={row['goodput']:.3f} "
                f"remote={row['realized_phi']:.3f} valid={row['hard_valid']}",
                flush=True,
            )
    return sorted(
        rows,
        key=lambda row: (
            row["hardware"],
            row["workload"],
            row["rate_label"],
            row["policy"],
            int(row["seed"]),
        ),
    )


def make_runs(capacity: dict[str, Any], *, smoke: bool) -> list[public.Run]:
    if smoke:
        hardware = "h100_homogeneous"
        name = "interactive"
        label = benchmark.LOADS[0][0]
        rate = float(capacity[hardware][name]["evaluation_rates"][label])
        return [
            public.Run(
                "public_externality_decomposition_smoke",
                public.WORKLOADS[name],
                label,
                rate,
                42,
                policy,
                hardware,
                40,
            )
            for policy in POLICIES
        ]

    runs = []
    for hardware in public.HARDWARES:
        for name, config in public.WORKLOADS.items():
            for label, _ in benchmark.LOADS:
                rate = float(capacity[hardware][name]["evaluation_rates"][label])
                for seed in SEEDS:
                    for policy in POLICIES:
                        runs.append(
                            public.Run(
                                "public_externality_decomposition_confirmation",
                                config,
                                label,
                                rate,
                                seed,
                                policy,
                                hardware,
                                config.evaluation_requests,
                            )
                        )
    if len(runs) != 288:
        raise RuntimeError(f"confirmation has {len(runs)} runs, want 288")
    return runs


def read_trace(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        raise RuntimeError(f"missing candidate trace: {path}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["request_id"]].append(row)
    return grouped


def validate_trace(path: Path, policy: str) -> dict[str, Any]:
    grouped = read_trace(path)
    max_score_error = 0.0
    max_capacity = 0.0
    max_forbidden_component = 0.0
    argmin_violations = 0
    candidate_count_violations = 0
    for request_id, rows in grouped.items():
        if len(rows) != 4:
            candidate_count_violations += 1
        chosen = [row for row in rows if row["chosen"] == "true"]
        if len(chosen) != 1:
            raise RuntimeError(
                f"{path}: request {request_id} has {len(chosen)} chosen rows"
            )
        selected = chosen[0]
        for row in rows:
            externality = float(row["slo_externality"])
            own_good = float(row["own_good"])
            capacity = float(row["capacity_total"])
            score = float(row["score"])
            expected = V * (externality - own_good)
            max_score_error = max(max_score_error, abs(score - expected))
            max_capacity = max(max_capacity, abs(capacity))
            if policy == "joint_own_only":
                max_forbidden_component = max(
                    max_forbidden_component, abs(externality)
                )
            elif policy == "joint_resident_only":
                max_forbidden_component = max(
                    max_forbidden_component, abs(own_good)
                )

        eligible = rows
        if policy == "decode_first_full":
            eligible = [row for row in rows if row["router_decode"] == "true"]
            if not eligible or selected["router_decode"] != "true":
                argmin_violations += 1
                continue
        best = min(float(row["score"]) for row in eligible)
        selected_score = float(selected["score"])
        if selected_score > best + 1e-10 * max(1.0, abs(best)):
            argmin_violations += 1

    return {
        "requests": len(grouped),
        "max_abs_score_identity_error": max_score_error,
        "max_abs_capacity_term": max_capacity,
        "max_abs_forbidden_component": max_forbidden_component,
        "argmin_violations": argmin_violations,
        "candidate_count_violations": candidate_count_violations,
    }


def paired_summary(
    per_seed: dict[str, dict[str, dict[int, float]]], comparator: str
) -> dict[str, Any]:
    all_deltas: list[float] = []
    by_cell = []
    for cell, policies in per_seed.items():
        deltas = [
            policies[FOCAL][seed] - policies[comparator][seed] for seed in SEEDS
        ]
        mean, low, high = base.ci95(deltas)
        by_cell.append(
            {
                "cell": cell,
                "mean_delta": mean,
                "ci_low": low,
                "ci_high": high,
                "wins": sum(value > 0 for value in deltas),
                "ties": sum(value == 0 for value in deltas),
                "losses": sum(value < 0 for value in deltas),
            }
        )
        all_deltas.extend(deltas)
    mean, low, high = base.ci95(all_deltas)
    return {
        "all_cells": {"mean_delta": mean, "ci_low": low, "ci_high": high},
        "by_cell": by_cell,
    }


def analyze(
    out: Path, runs: list[public.Run], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    cells = [
        (hardware, name, label)
        for hardware in public.HARDWARES
        for name in public.WORKLOADS
        for label, _ in benchmark.LOADS
    ]
    means: dict[str, dict[str, float]] = {}
    remotes: dict[str, dict[str, float]] = {}
    dimensions: dict[str, dict[str, dict[str, float]]] = {}
    per_seed: dict[str, dict[str, dict[int, float]]] = {}
    for hardware, name, label in cells:
        cell = f"{hardware}:{name}:{label}"
        means[cell], remotes[cell], dimensions[cell], per_seed[cell] = {}, {}, {}, {}
        for policy in POLICIES:
            members = [
                row
                for row in rows
                if row["hardware"] == hardware
                and row["workload"] == name
                and row["rate_label"] == label
                and row["policy"] == policy
            ]
            if len(members) != len(SEEDS):
                raise RuntimeError(f"incomplete cell {cell}/{policy}: {len(members)}")
            seed_values = {int(row["seed"]): float(row["goodput"]) for row in members}
            if set(seed_values) != set(SEEDS):
                raise RuntimeError(f"wrong seeds for {cell}/{policy}")
            per_seed[cell][policy] = seed_values
            means[cell][policy] = statistics.fmean(seed_values.values())
            remotes[cell][policy] = statistics.fmean(
                float(row["realized_phi"]) for row in members
            )
            dimensions[cell][policy] = {
                dimension: statistics.fmean(
                    float(row[f"good_{dimension}"]) for row in members
                )
                for dimension in ("ttft", "itl", "e2e")
            }

    trace_summaries = []
    for run in runs:
        summary = validate_trace(public.trace_path(out, run), run.policy)
        summary.update(
            {
                "hardware": run.hardware,
                "workload": run.config.workload.name,
                "rate_label": run.rate_label,
                "seed": run.seed,
                "policy": run.policy,
            }
        )
        trace_summaries.append(summary)

    aggregate = {
        policy: {
            "equal_cell_mean_goodput": statistics.fmean(
                means[cell][policy] for cell in means
            ),
            "worst_cell_goodput": min(means[cell][policy] for cell in means),
            "equal_cell_mean_remote_fraction": statistics.fmean(
                remotes[cell][policy] for cell in remotes
            ),
        }
        for policy in POLICIES
    }
    paired = {
        policy: paired_summary(per_seed, policy)
        for policy in POLICIES
        if policy != FOCAL
    }
    return {
        "status": "frozen fresh-seed externality/decomposition ablation",
        "routing_value": "smooth TTFT x E2E; reported goodput remains TTFT x ITL x E2E",
        "capacity_source": str(CAPACITY_SOURCE),
        "seeds": list(SEEDS),
        "run_count": len(runs),
        "cell_count": len(cells),
        "aggregate": aggregate,
        "policy_means": means,
        "remote_fractions": remotes,
        "dimension_attainment": dimensions,
        "paired_full_minus_comparator": paired,
        "hard_invalid_runs": sum(not bool(row["hard_valid"]) for row in rows),
        "runs_with_drops": sum(int(row["dropped"]) > 0 for row in rows),
        "runs_with_timeouts": sum(int(row["timed_out"]) > 0 for row in rows),
        "runs_with_length_caps": sum(
            int(row["length_capped"]) > 0 for row in rows
        ),
        "trace_run_count": len(trace_summaries),
        "trace_validation": {
            "max_abs_score_identity_error": max(
                item["max_abs_score_identity_error"] for item in trace_summaries
            ),
            "max_abs_capacity_term": max(
                item["max_abs_capacity_term"] for item in trace_summaries
            ),
            "max_abs_forbidden_component": max(
                item["max_abs_forbidden_component"] for item in trace_summaries
            ),
            "argmin_violations": sum(
                item["argmin_violations"] for item in trace_summaries
            ),
            "candidate_count_violations": sum(
                item["candidate_count_violations"] for item in trace_summaries
            ),
        },
    }


def write_report(out: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Public externality/decomposition ablation",
        "",
        "Routing uses smooth TTFT x E2E; reported goodput remains the hard",
        "TTFT/mean-ITL/E2E conjunction.",
        "",
        "## Aggregate result",
        "",
        "| arm | equal-cell mean goodput | worst-cell goodput | mean remote fraction |",
        "|---|---:|---:|---:|",
    ]
    for policy in POLICIES:
        item = result["aggregate"][policy]
        lines.append(
            f"| {DISPLAY[policy]} | {item['equal_cell_mean_goodput']:.4f} | "
            f"{item['worst_cell_goodput']:.4f} | "
            f"{item['equal_cell_mean_remote_fraction']:.4f} |"
        )
    lines += [
        "",
        "## Paired component effects",
        "",
        "Positive means the full joint policy has higher goodput.",
        "",
        "| comparison | mean delta | 95% CI |",
        "|---|---:|---:|",
    ]
    labels = {
        "joint_own_only": "add resident externality",
        "joint_resident_only": "add arriving-request value",
        "decode_first_full": "joint vs decode-first",
    }
    for policy in ("joint_own_only", "joint_resident_only", "decode_first_full"):
        item = result["paired_full_minus_comparator"][policy]["all_cells"]
        lines.append(
            f"| {labels[policy]} | {item['mean_delta']:+.4f} | "
            f"[{item['ci_low']:+.4f}, {item['ci_high']:+.4f}] |"
        )
    lines += [
        "",
        "## Per-cell mean goodput",
        "",
        "| cell | full | own-only | resident-only | decode-first |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell, values in result["policy_means"].items():
        lines.append(
            f"| `{cell}` | {values['joint_full']:.3f} | "
            f"{values['joint_own_only']:.3f} | "
            f"{values['joint_resident_only']:.3f} | "
            f"{values['decode_first_full']:.3f} |"
        )
    trace = result["trace_validation"]
    lines += [
        "",
        "## Validity",
        "",
        f"- Runs: {result['run_count']} across {result['cell_count']} cells.",
        f"- Hard-invalid runs: {result['hard_invalid_runs']}.",
        f"- Runs with drops: {result['runs_with_drops']}.",
        f"- Runs with timeouts: {result['runs_with_timeouts']}.",
        f"- Runs with length caps: {result['runs_with_length_caps']}.",
        f"- Candidate trace runs: {result['trace_run_count']}.",
        f"- Score-identity max error: {trace['max_abs_score_identity_error']:.3g}.",
        f"- Max capacity term: {trace['max_abs_capacity_term']:.3g}.",
        f"- Max forbidden ablation component: {trace['max_abs_forbidden_component']:.3g}.",
        f"- Constrained argmin violations: {trace['argmin_violations']}.",
        f"- Candidate-count violations: {trace['candidate_count_violations']}.",
        "",
        "These comparisons are component/structure ablations, not oracle gaps.",
    ]
    (out / "PUBLIC-EXTERNALITY-DECOMPOSITION-ABLATION.md").write_text(
        "\n".join(lines) + "\n"
    )


def run_stage(stage: str, out: Path, jobs: int, force: bool) -> None:
    if not PROTOCOL.exists():
        raise SystemExit(f"missing frozen protocol: {PROTOCOL}")
    capacity = load_capacity()
    smoke = stage == "smoke"
    runs = make_runs(capacity, smoke=smoke)
    rows = run_many(runs, out, jobs, force)
    public.require_hard_valid(rows, stage)
    if smoke:
        validations = [
            validate_trace(public.trace_path(out, run), run.policy) for run in runs
        ]
        if any(
            item["max_abs_score_identity_error"] > 1e-9
            or item["max_abs_capacity_term"] > 1e-12
            or item["max_abs_forbidden_component"] > 1e-12
            or item["argmin_violations"]
            or item["candidate_count_violations"]
            for item in validations
        ):
            raise RuntimeError(f"smoke trace validation failed: {validations}")
        public.write_rows(out / "smoke.csv", rows)
        print(json.dumps(validations, indent=2), flush=True)
        return

    public.write_rows(out / "confirmation.csv", rows)
    result = analyze(out, runs, rows)
    trace = result["trace_validation"]
    if (
        result["run_count"] != 288
        or result["hard_invalid_runs"]
        or result["runs_with_timeouts"]
        or result["runs_with_length_caps"]
        or trace["max_abs_score_identity_error"] > 1e-9
        or trace["max_abs_capacity_term"] > 1e-12
        or trace["max_abs_forbidden_component"] > 1e-12
        or trace["argmin_violations"]
        or trace["candidate_count_violations"]
    ):
        raise RuntimeError(f"confirmation validity gate failed: {result}")
    (out / "ablation_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_report(out, result)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "confirm", "all"), default="all")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = ("smoke", "confirm") if args.stage == "all" else (args.stage,)
    for stage in stages:
        run_stage(stage, args.out, args.jobs, args.force)


if __name__ == "__main__":
    main()
