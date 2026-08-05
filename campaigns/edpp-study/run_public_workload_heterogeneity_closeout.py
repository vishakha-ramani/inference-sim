#!/usr/bin/env python3
"""Bounded public-workload and realistic-heterogeneity closeout.

The protocol is frozen in PUBLIC-WORKLOAD-HETEROGENEITY-CLOSEOUT-PROTOCOL.md.
This runner is resumable and deliberately does not alter historical campaigns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

import run_decisive_campaign as base
import run_slo_externality_joint_campaign as diagnostic


DEFAULT_OUT = base.CAMPAIGN / "out" / "public_workload_heterogeneity_closeout"
WORKLOAD_DIR = base.CAMPAIGN / "workloads" / "public-closeout"
HETERO_BUNDLE = base.CAMPAIGN / "inputs" / "hetero-realistic-1p2d.yaml"
LLMD_PREFIX_POLICY = "llmd_prefix_threshold_workload_tuned"
LLMD_SCORERS = "precise-prefix-cache:2,queue-depth:1"
LLMD_THRESHOLDS = {"interactive": 1024, "reasoning": 16, "deep_research": 16}
DEV_SEEDS = (42, 123)
FRESH_SEEDS = (262147, 524309, 1048583, 2097169)
V = 8.0
REQUEST_TIMEOUT_SECS = 300
PHIS = (0.0, 0.5, 1.0)
HARDWARES = ("h100_homogeneous", "h100_a100_realistic")
CAPACITY_SHARE_BY_HARDWARE = {
    "h100_homogeneous": (0.5,),
    "h100_a100_realistic": (0.0, 0.4),
}


@dataclass(frozen=True)
class WorkloadConfig:
    workload: base.Workload
    capacity_requests: int
    evaluation_requests: int
    targets: dict[str, dict[str, float]]


WORKLOADS = {
    "interactive": WorkloadConfig(
        base.Workload(
            "interactive",
            WORKLOAD_DIR / "interactive-chat-single-turn.yaml",
            ("standard",),
            40.0,
        ),
        240,
        300,
        {
            "standard": {
                "ttft_ms": 1_000.0,
                "itl_ms": 50.0,
                "e2e_ms": 16_000.0,
            }
        },
    ),
    "reasoning": WorkloadConfig(
        base.Workload(
            "reasoning",
            WORKLOAD_DIR / "reasoning-single-turn.yaml",
            ("standard",),
            4.0,
        ),
        120,
        160,
        {
            "standard": {
                "ttft_ms": 2_000.0,
                "itl_ms": 100.0,
                "e2e_ms": 802_000.0,
            }
        },
    ),
    "deep_research": WorkloadConfig(
        base.Workload(
            "deep_research",
            WORKLOAD_DIR / "deep-research-single-turn.yaml",
            ("standard",),
            8.0,
        ),
        120,
        160,
        {
            "standard": {
                "ttft_ms": 10_000.0,
                "itl_ms": 100.0,
                "e2e_ms": 40_000.0,
            }
        },
    ),
}

CONTROLLERS = (
    "occupancy_v8",
    "occupancy_no_externality_v8",
    "causal_externality_no_capacity_v8",
)
POLICIES = (
    *CONTROLLERS,
    "least_ttft_joint",
    "kairos_beta_0p5",
    "static_joint_yardstick",
)
FRESH_POLICIES = (
    "causal_externality_no_capacity_v8",
    "least_ttft_joint",
    "kairos_beta_0p5",
    "static_joint_yardstick",
)
DISPLAY = {
    "occupancy_v8": "occupancy V=8",
    "occupancy_no_externality_v8": "occupancy, no externality",
    "causal_externality_no_capacity_v8": "causal externality, no capacity",
    "least_ttft_joint": "least projected TTFT (joint)",
    "kairos_beta_0p5": "Kairos (beta=0.5)",
    "static_joint_yardstick": "capacity-selected static joint plan",
}


@dataclass(frozen=True)
class Run:
    phase: str
    config: WorkloadConfig
    rate_label: str
    rate: float
    seed: int
    policy: str
    hardware: str
    requests: int
    phi: float | None = None
    psi: float | None = None

    def tag(self) -> str:
        phi = "none" if self.phi is None else str(self.phi).replace(".", "p")
        psi = "none" if self.psi is None else str(self.psi).replace(".", "p")
        return (
            f"{self.hardware}_{self.config.workload.name}_{self.rate_label}_"
            f"{self.rate:.6g}_{self.policy}_phi{phi}_psi{psi}_"
            f"s{self.seed}_n{self.requests}"
        )


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def make_spec(run: Run, path: Path) -> None:
    data = yaml.safe_load(run.config.workload.source.read_text())
    data["seed"] = run.seed
    data["aggregate_rate"] = run.rate
    data["num_requests"] = run.requests
    write_yaml(path, data)


def make_plan(path: Path, requests: int, phi: float, psi: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(base.CAMPAIGN / "make_pd_plan.py"),
        "--n",
        str(requests),
        "--phi",
        str(phi),
        "--prefill-instances",
        "1",
        "--decode-instances",
        "2",
        "--psi",
        str(psi),
    ]
    proc = subprocess.run(
        command, cwd=base.ROOT, check=True, capture_output=True, text=True
    )
    path.write_text(proc.stdout)


def topology_args(hardware: str, policy: str | None = None) -> list[str]:
    args = [
        "--num-instances",
        "3",
        "--prefill-instances",
        "1",
        "--decode-instances",
        "2",
        "--max-num-running-reqs",
        "256",
    ]
    if policy == LLMD_PREFIX_POLICY:
        args += [
            "--prefill-routing-scorers", LLMD_SCORERS,
            "--decode-routing-scorers", LLMD_SCORERS,
            "--cache-signal-delay", "50000",
        ]
    else:
        args += ["--decode-routing-scorers", "queue-depth:1"]
    if hardware == "h100_a100_realistic":
        args += ["--policy-config", str(HETERO_BUNDLE)]
    elif hardware != "h100_homogeneous":
        raise ValueError(f"unknown hardware {hardware}")
    return args


def loose_targets(config: WorkloadConfig) -> dict[str, dict[str, float]]:
    return {
        slo_class: {
            "ttft_ms": 999_000.0,
            "itl_ms": 999_000.0,
            "e2e_ms": 999_000_000.0,
        }
        for slo_class in config.workload.slo_classes
    }


def policy_args(
    run: Run, targets: dict[str, dict[str, float]], plan_path: Path | None
) -> list[str]:
    if run.policy in {"capacity_static", "static_joint_yardstick"}:
        if plan_path is None:
            raise ValueError(f"{run.policy} needs a plan")
        return ["--pd-plan", str(plan_path)]
    if run.policy == "occupancy_v8":
        return base.policy_args(
            "joint_slo_externality_occupancy", targets, V, None
        )
    if run.policy == "occupancy_no_externality_v8":
        return base.policy_args(
            "joint_slo_externality_occupancy", targets, V, None
        ) + ["--edpp-slo-externality-no-externality"]
    if run.policy == "causal_externality_no_capacity_v8":
        return base.policy_args(
            "joint_slo_externality_no_capacity", targets, V, None
        )
    if run.policy == "least_ttft_joint":
        return base.policy_args("least_ttft_joint", targets, None, None)
    if run.policy == "kairos_beta_0p5":
        return base.policy_args("kairos", targets, 0.5, None)
    if run.policy == "kairos_paper_alpha_1p3":
        return base.policy_args("kairos-paper", targets, None, None)
    if run.policy == LLMD_PREFIX_POLICY:
        return [
            "--pd-decider", "prefix-threshold",
            "--pd-prefix-threshold", str(LLMD_THRESHOLDS[run.config.workload.name]),
        ]
    raise ValueError(f"unknown policy {run.policy}")


def trace_path(out: Path, run: Run) -> Path:
    return out / run.phase / "candidate_traces" / f"{run.tag()}.csv"


def hard_valid(row: dict[str, Any]) -> bool:
    return (
        int(row["completed"]) + int(row["dropped"]) == int(row["injected"])
        and int(row["still_queued"]) == 0
        and int(row["still_running"]) == 0
        and int(row["timed_out"]) == 0
        and int(row["length_capped"]) == 0
    )


def execute_run(run: Run, out: Path, force: bool) -> dict[str, Any]:
    run_dir = out / run.phase
    spec_path = run_dir / "specs" / f"{run.tag()}.yaml"
    metrics_path = run_dir / "metrics" / f"{run.tag()}.json"
    stdout_path = run_dir / "stdout" / f"{run.tag()}.txt"
    make_spec(run, spec_path)

    plan_path: Path | None = None
    if run.policy in {"capacity_static", "static_joint_yardstick"}:
        if run.phi is None or run.psi is None:
            raise ValueError(f"{run.policy} needs phi and psi")
        plan_path = run_dir / "plans" / f"{run.tag()}.csv"
        make_plan(plan_path, run.requests, run.phi, run.psi)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    targets = (
        loose_targets(run.config)
        if run.phase == "public_closeout_capacity"
        else run.config.targets
    )
    if not force and metrics_path.exists() and stdout_path.exists():
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
            *topology_args(run.hardware, run.policy),
            *base.slo_args(targets),
            *policy_args(run, targets, plan_path),
            "--timeout",
            str(REQUEST_TIMEOUT_SECS),
            "--seed",
            str(run.seed),
            "--metrics-path",
            str(metrics_path),
        ]
        if run.policy in CONTROLLERS:
            candidate_path = trace_path(out, run)
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            command += ["--edpp-joint-candidate-trace", str(candidate_path)]
        proc = subprocess.run(command, cwd=base.ROOT, capture_output=True, text=True)
        stdout_path.write_text(
            proc.stdout + "\n--- STDERR ---\n" + proc.stderr
        )
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
    row["phi"] = "" if run.phi is None else run.phi
    row["psi_a100"] = "" if run.psi is None else run.psi
    row["zero_drop"] = int(row["dropped"]) == 0
    row["hard_valid"] = hard_valid(row)
    return row


def run_many(
    runs: Iterable[Run], out: Path, jobs: int, force: bool
) -> list[dict[str, Any]]:
    runs = list(runs)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(execute_run, run, out, force): run for run in runs}
        for index, future in enumerate(as_completed(futures), 1):
            run = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/{len(runs)}] {run.tag()} "
                f"goodput={row['goodput']:.3f} "
                f"completion={row['central_completion_rps']:.3f} "
                f"valid={row['hard_valid']}",
                flush=True,
            )
    return sorted(
        rows,
        key=lambda row: (
            row["hardware"],
            row["workload"],
            row["policy"],
            float(row["phi"]) if row["phi"] != "" else -1.0,
            float(row["psi_a100"]) if row["psi_a100"] != "" else -1.0,
            int(row["seed"]),
        ),
    )


def require_hard_valid(rows: list[dict[str, Any]], label: str) -> None:
    bad = [
        f"{row['hardware']}/{row['workload']}/{row['policy']}/s{row['seed']}"
        for row in rows
        if not row["hard_valid"]
    ]
    if bad:
        raise RuntimeError(f"{label} produced hard-invalid runs: {', '.join(bad)}")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capacity(out: Path, jobs: int, force: bool) -> None:
    runs = [
        Run(
            "public_closeout_capacity",
            config,
            "saturated",
            config.workload.saturation_rate,
            DEV_SEEDS[0],
            "capacity_static",
            hardware,
            config.capacity_requests,
            phi,
            psi,
        )
        for config in WORKLOADS.values()
        for hardware in HARDWARES
        for phi in PHIS
        for psi in CAPACITY_SHARE_BY_HARDWARE[hardware]
    ]
    rows = run_many(runs, out, jobs, force)
    require_hard_valid(rows, "capacity normalization")
    write_rows(out / "capacity.csv", rows)

    selection: dict[str, Any] = {}
    for hardware in HARDWARES:
        selection[hardware] = {}
        for name, config in WORKLOADS.items():
            members = [
                row
                for row in rows
                if row["hardware"] == hardware and row["workload"] == name
            ]
            eligible = [row for row in members if row["zero_drop"]]
            if not eligible:
                raise RuntimeError(
                    f"no zero-drop capacity point for {hardware}/{name}"
                )
            best = max(
                eligible,
                key=lambda row: (
                    float(row["central_completion_rps"]),
                    -float(row["phi"]),
                    -float(row["psi_a100"]),
                ),
            )
            observed = float(best["central_completion_rps"])
            offered = config.workload.saturation_rate
            if observed <= 0:
                raise RuntimeError(f"zero measured capacity for {hardware}/{name}")
            if offered < 1.05 * observed:
                raise RuntimeError(
                    f"capacity probe not saturated for {hardware}/{name}: "
                    f"offered {offered:.3f}, observed {observed:.3f}"
                )
            selection[hardware][name] = {
                "offered_saturation_rate": offered,
                "best_completion_rps": observed,
                "evaluation_rate": 0.85 * observed,
                "selected_phi": float(best["phi"]),
                "selected_psi_a100": float(best["psi_a100"]),
                "grid": [
                    {
                        "phi": float(row["phi"]),
                        "psi_a100": float(row["psi_a100"]),
                        "completion_rps": float(row["central_completion_rps"]),
                        "zero_drop": bool(row["zero_drop"]),
                        "dropped": int(row["dropped"]),
                    }
                    for row in members
                ],
            }
    (out / "capacity_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(json.dumps(selection, indent=2))


def load_capacity(out: Path) -> dict[str, Any]:
    path = out / "capacity_selection.json"
    if not path.exists():
        raise SystemExit("capacity_selection.json missing; run --stage capacity")
    return json.loads(path.read_text())


def pool_decisions(items: list[dict[str, Any]]) -> dict[str, Any]:
    pooled = diagnostic.pooled_decision_summary(items)
    total = sum(int(item["requests"]) for item in items)

    def weighted(field: str) -> float:
        return sum(
            float(item[field]) * int(item["requests"]) for item in items
        ) / total

    pooled["best_remote_vs_local_fraction"] = {
        direction: sum(
            float(item["best_remote_vs_local_fraction"][direction])
            * int(item["requests"])
            for item in items
        )
        / total
        for direction in ("remote", "local", "tie")
    }
    pooled["mean_absolute_remote_minus_local_score"] = weighted(
        "mean_absolute_remote_minus_local_score"
    )
    return pooled


def evaluate(out: Path, jobs: int, force: bool) -> None:
    capacity_selection = load_capacity(out)
    runs: list[Run] = []
    for hardware in HARDWARES:
        for name, config in WORKLOADS.items():
            selected = capacity_selection[hardware][name]
            for seed in DEV_SEEDS:
                for policy in POLICIES:
                    phi = None
                    psi = None
                    if policy == "static_joint_yardstick":
                        phi = float(selected["selected_phi"])
                        psi = float(selected["selected_psi_a100"])
                    runs.append(
                        Run(
                            "public_closeout_evaluation",
                            config,
                            "stress_0p85_capacity",
                            float(selected["evaluation_rate"]),
                            seed,
                            policy,
                            hardware,
                            config.evaluation_requests,
                            phi,
                            psi,
                        )
                    )

    rows = run_many(runs, out, jobs, force)
    require_hard_valid(rows, "closeout evaluation")
    write_rows(out / "evaluation.csv", rows)

    means: dict[str, dict[str, float]] = {}
    drops: dict[str, dict[str, float]] = {}
    for hardware in HARDWARES:
        means[hardware] = {}
        drops[hardware] = {}
        for name in WORKLOADS:
            key = f"{hardware}:{name}"
            means[hardware][name] = {}
            drops[hardware][name] = {}
            for policy in POLICIES:
                members = [
                    row
                    for row in rows
                    if row["hardware"] == hardware
                    and row["workload"] == name
                    and row["policy"] == policy
                ]
                if len(members) != len(DEV_SEEDS):
                    raise RuntimeError(
                        f"incomplete evaluation for {key}/{policy}: {len(members)}"
                    )
                means[hardware][name][policy] = statistics.fmean(
                    float(row["goodput"]) for row in members
                )
                drops[hardware][name][policy] = statistics.fmean(
                    int(row["dropped"]) / int(row["injected"])
                    for row in members
                )

    summaries: dict[str, dict[str, dict[str, Any]]] = {
        hardware: {name: {} for name in WORKLOADS} for hardware in HARDWARES
    }
    for hardware in HARDWARES:
        for name in WORKLOADS:
            for policy in CONTROLLERS:
                items = []
                for run in runs:
                    if (
                        run.hardware == hardware
                        and run.config.workload.name == name
                        and run.policy == policy
                    ):
                        items.append(
                            diagnostic.analyze_candidate_trace(
                                trace_path(out, run),
                                V,
                                occupancy_capacity=policy
                                in {
                                    "occupancy_v8",
                                    "occupancy_no_externality_v8",
                                },
                            )
                        )
                summaries[hardware][name][policy] = pool_decisions(items)

    gates: dict[str, Any] = {}
    for hardware in HARDWARES:
        deltas = {
            name: means[hardware][name]["occupancy_v8"]
            - means[hardware][name]["causal_externality_no_capacity_v8"]
            for name in WORKLOADS
        }
        public_margins = {
            name: means[hardware][name]["occupancy_v8"]
            - max(
                means[hardware][name]["least_ttft_joint"],
                means[hardware][name]["kairos_beta_0p5"],
            )
            for name in WORKLOADS
        }
        gains = sum(delta >= 0.02 for delta in deltas.values())
        no_material_regression = min(deltas.values()) >= -0.05
        public_competitive = sum(
            margin >= -0.05 for margin in public_margins.values()
        )
        gates[hardware] = {
            "occupancy_minus_no_capacity": deltas,
            "occupancy_minus_best_public_baseline": public_margins,
            "gains_at_least_0p02": gains,
            "no_loss_worse_than_0p05": no_material_regression,
            "public_competitive_within_0p05": public_competitive,
            "revival_gate_passes": (
                gains >= 2
                and no_material_regression
                and public_competitive >= 2
            ),
        }

    shifts = {
        name: gates["h100_a100_realistic"]["occupancy_minus_no_capacity"][name]
        - gates["h100_homogeneous"]["occupancy_minus_no_capacity"][name]
        for name in WORKLOADS
    }
    if gates["h100_homogeneous"]["revival_gate_passes"]:
        decision = "REVIVE: occupancy controller passes on homogeneous hardware"
    elif gates["h100_a100_realistic"]["revival_gate_passes"]:
        decision = "NARROW: occupancy controller is supported only on the heterogeneous fleet"
    else:
        decision = "CLOSE: occupancy-controller development remains stopped"

    trace_exact = True
    for hardware in HARDWARES:
        for name in WORKLOADS:
            for policy in ("occupancy_v8", "occupancy_no_externality_v8"):
                summary = summaries[hardware][name][policy]
                conservation = summary["occupancy_queue_conservation"]
                trace_exact = trace_exact and (
                    conservation["transition_violations"] == 0
                    and conservation["capacity_cross_term_violations"] == 0
                    and summary[
                        "positive_chosen_snapshot_score_regret_fraction"
                    ]
                    == 0
                )

    result = {
        "status": decision,
        "scope": (
            "development-seed closeout on independent request marginals; "
            "decision diagnostics are not end-to-end policy regret"
        ),
        "seeds": list(DEV_SEEDS),
        "v": V,
        "run_count": len(runs),
        "policy_means": means,
        "drop_fractions": drops,
        "decision_summaries": summaries,
        "revival_gates": gates,
        "heterogeneity_delta_shift": shifts,
        "trace_and_occupancy_conservation_exact": trace_exact,
    }
    (out / "closeout_result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(out, capacity_selection, result)
    print(json.dumps(result, indent=2))


def write_report(
    out: Path, capacity_selection: dict[str, Any], result: dict[str, Any]
) -> None:
    means = result["policy_means"]
    summaries = result["decision_summaries"]
    lines = [
        "# Public-workload and heterogeneity closeout",
        "",
        "This is a bounded development-seed closeout using independent request",
        "marginals from three upstream inference-perf YAMLs. Multi-turn behavior",
        "is intentionally outside scope, and the decision diagnostics below are",
        "not end-to-end policy regret.",
        "",
        "## Capacity normalization",
        "",
        "| fleet | workload | best completion rps | evaluation rps | phi | A100 decode share |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for hardware in HARDWARES:
        for name in WORKLOADS:
            cell = capacity_selection[hardware][name]
            lines.append(
                f"| `{hardware}` | `{name}` | "
                f"{cell['best_completion_rps']:.3f} | "
                f"{cell['evaluation_rate']:.3f} | "
                f"{cell['selected_phi']:.2f} | "
                f"{cell['selected_psi_a100']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Mean goodput",
            "",
            "| fleet/workload | "
            + " | ".join(DISPLAY[policy] for policy in POLICIES)
            + " |",
            "|---|" + "---:|" * len(POLICIES),
        ]
    )
    for hardware in HARDWARES:
        for name in WORKLOADS:
            lines.append(
                f"| `{hardware}:{name}` | "
                + " | ".join(
                    f"{means[hardware][name][policy]:.3f}"
                    for policy in POLICIES
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## What the controllers decided",
            "",
            "Remote/local/tie is the sign of the best-remote score minus the",
            "best-local score. A negative gap favors remote. Net-good and",
            "capacity columns show how often each term separately favored remote.",
            "",
            "| fleet/workload | controller | actual remote | total R/L/T | net-good R/L/T | capacity R/L/T | conflicts | mean abs(net) | mean abs(capacity) |",
            "|---|---|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for hardware in HARDWARES:
        for name in WORKLOADS:
            for policy in CONTROLLERS:
                cell = summaries[hardware][name][policy]
                total = cell["best_remote_vs_local_fraction"]
                net = cell["net_good_term_direction_fraction"]
                cap = cell["capacity_term_direction_fraction"]
                lines.append(
                    f"| `{hardware}:{name}` | `{DISPLAY[policy]}` | "
                    f"{cell['actual_remote_fraction']:.3f} | "
                    f"{total['remote']:.3f}/{total['local']:.3f}/{total['tie']:.3f} | "
                    f"{net['remote']:.3f}/{net['local']:.3f}/{net['tie']:.3f} | "
                    f"{cap['remote']:.3f}/{cap['local']:.3f}/{cap['tie']:.3f} | "
                    f"{cell['term_conflict_fraction']:.3f} | "
                    f"{cell['mean_absolute_remote_minus_local_net_good_term']:.6g} | "
                    f"{cell['mean_absolute_remote_minus_local_capacity_term']:.6g} |"
                )

    lines.extend(["", "## Registered decision", ""])
    for hardware in HARDWARES:
        gate = result["revival_gates"][hardware]
        lines.append(
            f"- `{hardware}`: "
            f"{'PASS' if gate['revival_gate_passes'] else 'FAIL'}; "
            f"occupancy gains of at least 0.02 on "
            f"{gate['gains_at_least_0p02']}/3 workloads, "
            f"public-baseline competitive on "
            f"{gate['public_competitive_within_0p05']}/3."
        )
    lines.extend(
        [
            "",
            f"Final decision: **{result['status']}**.",
            "",
            "Occupancy queue conservation and chosen-candidate trace checks: "
            + (
                "exact."
                if result["trace_and_occupancy_conservation_exact"]
                else "FAILED."
            ),
            "",
        ]
    )
    (out / "PUBLIC-WORKLOAD-HETEROGENEITY-CLOSEOUT.md").write_text(
        "\n".join(lines)
    )


def fresh_confirmation(out: Path, jobs: int, force: bool) -> None:
    capacity_selection = load_capacity(out)
    runs: list[Run] = []
    for hardware in HARDWARES:
        for name, config in WORKLOADS.items():
            selected = capacity_selection[hardware][name]
            for seed in FRESH_SEEDS:
                for policy in FRESH_POLICIES:
                    phi = None
                    psi = None
                    if policy == "static_joint_yardstick":
                        phi = float(selected["selected_phi"])
                        psi = float(selected["selected_psi_a100"])
                    runs.append(
                        Run(
                            "public_no_capacity_fresh_confirmation",
                            config,
                            "stress_0p85_capacity",
                            float(selected["evaluation_rate"]),
                            seed,
                            policy,
                            hardware,
                            config.evaluation_requests,
                            phi,
                            psi,
                        )
                    )
    if len(runs) != 96:
        raise RuntimeError(f"fresh confirmation has {len(runs)} runs, want 96")

    rows = run_many(runs, out, jobs, force)
    require_hard_valid(rows, "public no-capacity fresh confirmation")
    write_rows(out / "fresh_confirmation.csv", rows)

    means: dict[str, dict[str, dict[str, float]]] = {
        hardware: {name: {} for name in WORKLOADS} for hardware in HARDWARES
    }
    remote: dict[str, dict[str, dict[str, float]]] = {
        hardware: {name: {} for name in WORKLOADS} for hardware in HARDWARES
    }
    per_seed: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    for hardware in HARDWARES:
        for name in WORKLOADS:
            cell = f"{hardware}:{name}"
            per_seed[cell] = {}
            for policy in FRESH_POLICIES:
                members = [
                    row
                    for row in rows
                    if row["hardware"] == hardware
                    and row["workload"] == name
                    and row["policy"] == policy
                ]
                if len(members) != len(FRESH_SEEDS):
                    raise RuntimeError(
                        f"incomplete fresh confirmation for {cell}/{policy}"
                    )
                members.sort(key=lambda item: int(item["seed"]))
                means[hardware][name][policy] = statistics.fmean(
                    float(row["goodput"]) for row in members
                )
                remote[hardware][name][policy] = statistics.fmean(
                    float(row["realized_phi"]) for row in members
                )
                per_seed[cell][policy] = [
                    {
                        "seed": int(row["seed"]),
                        "goodput": float(row["goodput"]),
                        "remote_fraction": float(row["realized_phi"]),
                        "dropped": int(row["dropped"]),
                    }
                    for row in members
                ]

    deployable = (
        "causal_externality_no_capacity_v8",
        "least_ttft_joint",
        "kairos_beta_0p5",
    )
    cells = [(hardware, name) for hardware in HARDWARES for name in WORKLOADS]
    best = {
        f"{hardware}:{name}": max(
            means[hardware][name][policy] for policy in deployable
        )
        for hardware, name in cells
    }
    ranking = []
    for policy in deployable:
        regrets = {
            f"{hardware}:{name}": best[f"{hardware}:{name}"]
            - means[hardware][name][policy]
            for hardware, name in cells
        }
        ranking.append(
            {
                "policy": policy,
                "worst_regret": max(regrets.values()),
                "worst_cell": max(regrets, key=regrets.get),
                "mean_goodput": statistics.fmean(
                    means[hardware][name][policy] for hardware, name in cells
                ),
            }
        )
    ranking.sort(
        key=lambda item: (
            item["worst_regret"],
            -item["mean_goodput"],
            item["policy"],
        )
    )

    paired: dict[str, Any] = {}
    focal = "causal_externality_no_capacity_v8"
    for comparator in FRESH_POLICIES:
        if comparator == focal:
            continue
        cell_rows = []
        all_deltas = []
        for hardware, name in cells:
            left = {
                item["seed"]: item["goodput"]
                for item in per_seed[f"{hardware}:{name}"][focal]
            }
            right = {
                item["seed"]: item["goodput"]
                for item in per_seed[f"{hardware}:{name}"][comparator]
            }
            deltas = [left[seed] - right[seed] for seed in FRESH_SEEDS]
            mean, low, high = base.ci95(deltas)
            cell_rows.append(
                {
                    "cell": f"{hardware}:{name}",
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
        paired[comparator] = {
            "all_cells": {"mean_delta": mean, "ci_low": low, "ci_high": high},
            "by_cell": cell_rows,
        }

    trace_exact = True
    for run in runs:
        if run.policy != focal:
            continue
        summary = diagnostic.analyze_candidate_trace(trace_path(out, run), V)
        trace_exact = trace_exact and (
            summary["positive_chosen_snapshot_score_regret_fraction"] == 0
        )

    result = {
        "status": "held-out public-workload confirmation; no tuning",
        "fresh_seeds": list(FRESH_SEEDS),
        "run_count": len(runs),
        "policy_means": means,
        "remote_fractions": remote,
        "per_seed": per_seed,
        "deployable_minimax_ranking": ranking,
        "no_capacity_paired_deltas": paired,
        "hard_invalid_runs": sum(not row["hard_valid"] for row in rows),
        "runs_with_drops": sum(int(row["dropped"]) > 0 for row in rows),
        "chosen_argmin_trace_exact": trace_exact,
    }
    (out / "fresh_confirmation_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_fresh_confirmation_report(out, result)
    print(json.dumps(result, indent=2))


def write_fresh_confirmation_report(out: Path, result: dict[str, Any]) -> None:
    means = result["policy_means"]
    remote = result["remote_fractions"]
    lines = [
        "# Public no-capacity fresh confirmation",
        "",
        "Four held-out seeds; all policies and rates frozen before these runs.",
        "",
        "| fleet/workload | no capacity | least TTFT | Kairos | static yardstick | no-cap remote |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for hardware in HARDWARES:
        for name in WORKLOADS:
            lines.append(
                f"| `{hardware}:{name}` | "
                f"{means[hardware][name]['causal_externality_no_capacity_v8']:.3f} | "
                f"{means[hardware][name]['least_ttft_joint']:.3f} | "
                f"{means[hardware][name]['kairos_beta_0p5']:.3f} | "
                f"{means[hardware][name]['static_joint_yardstick']:.3f} | "
                f"{remote[hardware][name]['causal_externality_no_capacity_v8']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Deployable minimax ranking",
            "",
            "| rank | policy | worst regret | worst cell | mean goodput |",
            "|---:|---|---:|---|---:|",
        ]
    )
    for index, item in enumerate(result["deployable_minimax_ranking"], 1):
        lines.append(
            f"| {index} | `{item['policy']}` | {item['worst_regret']:.3f} | "
            f"`{item['worst_cell']}` | {item['mean_goodput']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Hard-invalid runs: {result['hard_invalid_runs']}; runs with drops: "
            f"{result['runs_with_drops']}.",
            "",
            "Chosen-candidate trace check: "
            + ("exact." if result["chosen_argmin_trace_exact"] else "FAILED."),
            "",
        ]
    )
    (out / "PUBLIC-NO-CAPACITY-FRESH-CONFIRMATION.md").write_text(
        "\n".join(lines)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("capacity", "evaluate", "fresh-confirm", "all"),
        default="all",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not HETERO_BUNDLE.exists():
        raise SystemExit(f"missing realistic heterogeneity bundle {HETERO_BUNDLE}")
    if args.stage in {"capacity", "all"}:
        capacity(out, args.jobs, args.force)
    if args.stage in {"evaluate", "all"}:
        evaluate(out, args.jobs, args.force)
    if args.stage == "fresh-confirm":
        fresh_confirmation(out, args.jobs, args.force)


if __name__ == "__main__":
    main()
