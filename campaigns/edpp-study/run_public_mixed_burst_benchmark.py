#!/usr/bin/env python3
"""Mixed-workload shifts and burst benchmark.

Protocol: PUBLIC-MIXED-BURST-BENCHMARK-PROTOCOL.md.
The runner is resumable and never instantiates held-out seeds during static
calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

import run_decisive_campaign as base
import run_public_load_static_benchmark as benchmark
import run_public_workload_heterogeneity_closeout as prior
import run_slo_externality_joint_campaign as diagnostic


DEFAULT_OUT = base.CAMPAIGN / "out" / "public_mixed_burst_benchmark_ttft_rollout_v2"
PROTOCOL = base.CAMPAIGN / "PUBLIC-MIXED-BURST-BENCHMARK-PROTOCOL.md"
CAPACITY_PATH = (
    base.CAMPAIGN
    / "out"
    / "public_load_static_benchmark_ttft_rollout_v2"
    / "capacity_selection.json"
)

STATIC_SEEDS = (42, 123)
CONFIRMATION_SEEDS = (2000000099, 2000000123, 2000000141, 2000000159)
PROFILES = (
    "sequential_shift",
    "concurrent_poisson",
    "concurrent_gamma_cv3",
    "short_spike",
)
CLASSES = ("interactive", "reasoning", "deep_research")
SLO_CLASS = {
    "interactive": "critical",
    "reasoning": "batch",
    "deep_research": "standard",
}
TARGETS = {
    "critical": {"ttft_ms": 1_000.0, "itl_ms": 50.0, "e2e_ms": 16_000.0},
    "batch": {"ttft_ms": 2_000.0, "itl_ms": 100.0, "e2e_ms": 802_000.0},
    "standard": {"ttft_ms": 10_000.0, "itl_ms": 100.0, "e2e_ms": 40_000.0},
}
V = 8.0
FOCAL = benchmark.FOCAL
ADAPTED_KAIROS = "kairos_beta_0p5"
PAPER_KAIROS = "kairos_paper_alpha_1p3"
KAIROS_POLICY = PAPER_KAIROS
LLMD_CLASS_PREFIX = "llmd_prefix_threshold_class_tuned"
LLMD_CLASS_THRESHOLDS = "critical=1024,batch=16,standard=16"
DEPLOYABLE = (FOCAL, "least_ttft_joint", KAIROS_POLICY, LLMD_CLASS_PREFIX)
POLICIES = (*DEPLOYABLE, "static_joint_yardstick")
DISPLAY = {
    FOCAL: "causal externality",
    "least_ttft_joint": "least TTFT",
    ADAPTED_KAIROS: "Kairos adaptation",
    PAPER_KAIROS: "Kairos (paper, alpha=1.3)",
    LLMD_CLASS_PREFIX: "class-aware llm-d threshold",
    "static_joint_yardstick": "goodput-tuned static",
}
PLAN_ROWS = 5_000
DRAIN_US = 3_600_000_000


@dataclass(frozen=True)
class Run:
    phase: str
    profile: str
    hardware: str
    seed: int
    policy: str
    phi: float | None = None
    psi: float | None = None
    smoke: bool = False

    def tag(self) -> str:
        phi = "none" if self.phi is None else str(self.phi).replace(".", "p")
        psi = "none" if self.psi is None else str(self.psi).replace(".", "p")
        suffix = "_smoke" if self.smoke else ""
        return (
            f"{self.hardware}_{self.profile}_{self.policy}_phi{phi}_psi{psi}_"
            f"s{self.seed}{suffix}"
        )


def load_capacities() -> dict[str, Any]:
    if not CAPACITY_PATH.exists():
        raise SystemExit(f"missing frozen capacity artifact: {CAPACITY_PATH}")
    return json.loads(CAPACITY_PATH.read_text())


def plan_grid(hardware: str) -> list[tuple[float, float]]:
    psis = (0.5,) if hardware == "h100_homogeneous" else (0.0, 0.5, 1.0)
    return [(phi, psi) for phi in (0.0, 0.5, 1.0) for psi in psis]


def source_cohort(name: str) -> dict[str, Any]:
    source = prior.WORKLOADS[name].workload.source
    data = yaml.safe_load(source.read_text())
    cohort = dict(data["cohorts"][0])
    cohort["slo_class"] = SLO_CLASS[name]
    cohort["rate_fraction"] = 1.0
    return cohort


def window_cohort(
    name: str,
    label: str,
    start_us: int,
    duration_us: int,
    rate: float,
    arrival: str,
) -> dict[str, Any]:
    cohort = source_cohort(name)
    cohort["id"] = f"{label}-{name}"
    cohort["arrival"] = {"process": arrival}
    if arrival == "gamma":
        cohort["arrival"]["cv"] = 3.0
    cohort["spike"] = {
        "start_time_us": start_us,
        "duration_us": duration_us,
        "trace_rate": rate,
    }
    return cohort


def profile_spec(
    profile: str,
    hardware: str,
    seed: int,
    smoke: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capacity = load_capacities()[hardware]
    caps = {name: float(capacity[name]["capacity_rps"]) for name in CLASSES}
    cohorts: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"normalized_mean_load": 0.8}

    if profile == "sequential_shift":
        expected_per_class = 60 if smoke else 100
        cursor = 0
        phases = []
        for name in CLASSES:
            rate = 0.8 * caps[name]
            duration_us = max(1, round(expected_per_class / rate * 1e6))
            cohorts.append(
                window_cohort(name, "phase", cursor, duration_us, rate, "poisson")
            )
            phases.append(
                {
                    "class": name,
                    "start_us": cursor,
                    "end_us": cursor + duration_us,
                    "rate_rps": rate,
                }
            )
            cursor += duration_us
        horizon_us = cursor
        metadata["phases"] = phases
    elif profile in {
        "concurrent_poisson",
        "concurrent_gamma_cv3",
        "short_spike",
    }:
        expected_total = 3_000 if profile == "short_spike" else 720
        mean_rates = {name: 0.8 * caps[name] / 3.0 for name in CLASSES}
        horizon_us = max(
            1, round(expected_total / sum(mean_rates.values()) * 1e6)
        )
        if profile != "short_spike":
            arrival = "gamma" if profile == "concurrent_gamma_cv3" else "poisson"
            for name in CLASSES:
                cohorts.append(
                    window_cohort(
                        name, "mix", 0, horizon_us, mean_rates[name], arrival
                    )
                )
        else:
            burst_start = round(0.4 * horizon_us)
            burst_duration = round(0.2 * horizon_us)
            for name in CLASSES:
                cohorts.append(
                    window_cohort(
                        name,
                        "base",
                        0,
                        horizon_us,
                        0.6 * caps[name] / 3.0,
                        "poisson",
                    )
                )
                cohorts.append(
                    window_cohort(
                        name,
                        "burst",
                        burst_start,
                        burst_duration,
                        1.0 * caps[name] / 3.0,
                        "poisson",
                    )
                )
            metadata["burst_start_us"] = burst_start
            metadata["burst_end_us"] = burst_start + burst_duration
            metadata["normalized_base_load"] = 0.6
            metadata["normalized_peak_load"] = 1.6
    else:
        raise ValueError(f"unknown profile {profile}")

    metadata["arrival_horizon_us"] = horizon_us
    metadata["horizon_us"] = horizon_us + DRAIN_US
    metadata["class_capacity_rps"] = caps
    spec = {
        "version": "2",
        "seed": seed,
        "category": "language",
        "aggregate_rate": 0,
        "horizon": horizon_us + DRAIN_US,
        "num_requests": 0,
        "cohorts": cohorts,
    }
    return spec, metadata


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def make_plan(path: Path, phi: float, psi: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(base.CAMPAIGN / "make_pd_plan.py"),
        "--n",
        str(PLAN_ROWS),
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


def policy_args(run: Run, plan_path: Path | None) -> list[str]:
    if run.policy == "static_joint_yardstick":
        if plan_path is None:
            raise ValueError("static plan path missing")
        return ["--pd-plan", str(plan_path)]
    if run.policy == FOCAL:
        return base.policy_args(
            "joint_slo_externality_no_capacity", TARGETS, V, None
        )
    if run.policy == "least_ttft_joint":
        return base.policy_args("least_ttft_joint", TARGETS, None, None)
    if run.policy == ADAPTED_KAIROS:
        return base.policy_args("kairos", TARGETS, 0.5, None)
    if run.policy == PAPER_KAIROS:
        return base.policy_args("kairos-paper", TARGETS, None, None)
    if run.policy == LLMD_CLASS_PREFIX:
        return [
            "--pd-decider", "prefix-threshold",
            "--pd-prefix-threshold", "16",
            "--pd-prefix-threshold-classes", LLMD_CLASS_THRESHOLDS,
        ]
    raise ValueError(f"unknown policy {run.policy}")


def trace_path(out: Path, run: Run) -> Path:
    return out / run.phase / "candidate_traces" / f"{run.tag()}.csv"


def is_good(request: dict[str, Any]) -> bool:
    target = TARGETS[str(request["slo_class"])]
    return (
        float(request["ttft_ms"]) <= target["ttft_ms"]
        and float(request["itl_ms"]) <= target["itl_ms"]
        and float(request["e2e_ms"]) <= target["e2e_ms"]
    )


def segment_attainment(
    metrics: dict[str, Any], profile: str, metadata: dict[str, Any]
) -> dict[str, float]:
    if profile != "short_spike":
        return {}
    start = float(metadata["burst_start_us"]) / 1e6
    end = float(metadata["burst_end_us"]) / 1e6
    groups: dict[str, list[dict[str, Any]]] = {"pre": [], "during": [], "post": []}
    for request in metrics.get("requests", []):
        arrival = float(request["arrived_at"])
        label = "pre" if arrival < start else "during" if arrival < end else "post"
        groups[label].append(request)
    result: dict[str, float] = {}
    for label, requests in groups.items():
        result[f"segment_{label}_count"] = len(requests)
        result[f"segment_{label}_goodput"] = (
            sum(is_good(request) for request in requests) / len(requests)
            if requests
            else 0.0
        )
    return result


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
    spec, metadata = profile_spec(run.profile, run.hardware, run.seed, run.smoke)
    write_yaml(spec_path, spec)

    plan_path: Path | None = None
    if run.policy == "static_joint_yardstick":
        if run.phi is None or run.psi is None:
            raise ValueError("static run missing phi/psi")
        plan_path = run_dir / "plans" / f"{run.tag()}.csv"
        make_plan(plan_path, run.phi, run.psi)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if force or not (metrics_path.exists() and stdout_path.exists()):
        command = [
            str(base.ROOT / "blis"),
            "run",
            "--model",
            base.MODEL,
            "--workload-spec",
            str(spec_path),
            "--num-requests",
            "0",
            *prior.topology_args(
                run.hardware,
                prior.LLMD_PREFIX_POLICY if run.policy == LLMD_CLASS_PREFIX else None,
            ),
            *base.slo_args(TARGETS),
            *policy_args(run, plan_path),
            "--timeout",
            "-1",
            "--seed",
            str(run.seed),
            "--metrics-path",
            str(metrics_path),
        ]
        if run.policy == FOCAL:
            candidate_path = trace_path(out, run)
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            command += ["--edpp-joint-candidate-trace", str(candidate_path)]
        proc = subprocess.run(command, cwd=base.ROOT, capture_output=True, text=True)
        stdout_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                f"run failed ({run.tag()}):\n{proc.stderr[-5000:]}\n{proc.stdout[-5000:]}"
            )

    row = base.result_row(
        phase=run.phase,
        workload=run.profile,
        rate_label="normalized_mean_0p80",
        rate=0.8,
        seed=run.seed,
        policy=run.policy,
        parameter=None,
        topology=f"1p2d:{run.hardware}",
        metrics_path=metrics_path,
        stdout_path=stdout_path,
    )
    metrics = json.loads(metrics_path.read_text())
    row["profile"] = run.profile
    row["hardware"] = run.hardware
    row["phi"] = "" if run.phi is None else run.phi
    row["psi_a100"] = "" if run.psi is None else run.psi
    row["horizon_us"] = metadata["horizon_us"]
    row["hard_valid"] = hard_valid(row)
    for name in CLASSES:
        item = metrics.get("per_class", {}).get(SLO_CLASS[name], {})
        row[f"{name}_count"] = int(item.get("count", 0))
        row[f"{name}_goodput"] = float(item.get("slo_attainment", 0.0))
        for dimension in ("ttft", "itl", "e2e"):
            row[f"{name}_good_{dimension}"] = float(
                item.get("slo_attainment_by_dim", {}).get(dimension, 0.0)
            )
    row.update(segment_attainment(metrics, run.profile, metadata))
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
                f"n={row['injected']} goodput={float(row['goodput']):.3f} "
                f"valid={row['hard_valid']}",
                flush=True,
            )
    return sorted(
        rows,
        key=lambda row: (
            row["hardware"],
            row["profile"],
            row["policy"],
            float(row["phi"]) if row["phi"] != "" else -1.0,
            float(row["psi_a100"]) if row["psi_a100"] != "" else -1.0,
            int(row["seed"]),
        ),
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require_valid(rows: list[dict[str, Any]], label: str) -> None:
    bad = [
        f"{row['hardware']}/{row['profile']}/{row['policy']}/s{row['seed']}"
        for row in rows
        if not row["hard_valid"]
        or any(int(row[f"{name}_count"]) == 0 for name in CLASSES)
    ]
    if bad:
        raise RuntimeError(f"{label} invalid runs: {', '.join(bad)}")


def static_runs() -> list[Run]:
    return [
        Run("static_calibration", profile, hardware, seed, "static_joint_yardstick", phi, psi)
        for hardware in prior.HARDWARES
        for profile in PROFILES
        for phi, psi in plan_grid(hardware)
        for seed in STATIC_SEEDS
    ]


def select_static(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selection: dict[str, Any] = {}
    for hardware in prior.HARDWARES:
        selection[hardware] = {}
        for profile in PROFILES:
            candidates = []
            for phi, psi in plan_grid(hardware):
                members = [
                    row
                    for row in rows
                    if row["hardware"] == hardware
                    and row["profile"] == profile
                    and float(row["phi"]) == phi
                    and float(row["psi_a100"]) == psi
                ]
                if len(members) != len(STATIC_SEEDS) or {
                    int(row["seed"]) for row in members
                } != set(STATIC_SEEDS):
                    raise RuntimeError(
                        f"incomplete static point {hardware}/{profile}/{phi}/{psi}"
                    )
                values = [float(row["goodput"]) for row in members]
                candidates.append(
                    {
                        "phi": phi,
                        "psi_a100": psi,
                        "mean_goodput": statistics.fmean(values),
                        "stdev_goodput": statistics.stdev(values),
                        "seed_values": {
                            str(int(row["seed"])): float(row["goodput"])
                            for row in members
                        },
                        "eligible": all(
                            row["hard_valid"] and int(row["dropped"]) == 0
                            for row in members
                        ),
                    }
                )
            eligible = [item for item in candidates if item["eligible"]]
            if not eligible:
                raise RuntimeError(f"no eligible static plan for {hardware}/{profile}")
            best = max(
                eligible,
                key=lambda item: (
                    item["mean_goodput"],
                    -item["stdev_goodput"],
                    -item["phi"],
                    -item["psi_a100"],
                ),
            )
            selection[hardware][profile] = {
                "selected_phi": best["phi"],
                "selected_psi_a100": best["psi_a100"],
                "calibration_mean_goodput": best["mean_goodput"],
                "calibration_stdev_goodput": best["stdev_goodput"],
                "calibration_seeds": list(STATIC_SEEDS),
                "grid": candidates,
            }
    return selection


def run_static(out: Path, jobs: int, force: bool) -> None:
    runs = static_runs()
    if len(runs) != 96:
        raise RuntimeError(f"static calibration has {len(runs)} runs, want 96")
    rows = run_many(runs, out, jobs, force)
    require_valid(rows, "static calibration")
    write_rows(out / "static_calibration.csv", rows)
    selection = select_static(rows)
    (out / "static_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2), flush=True)


def load_static(out: Path) -> dict[str, Any]:
    path = out / "static_selection.json"
    if not path.exists():
        raise SystemExit("static_selection.json missing; run --stage static first")
    return json.loads(path.read_text())


def confirmation_runs(static: dict[str, Any]) -> list[Run]:
    runs: list[Run] = []
    for hardware in prior.HARDWARES:
        for profile in PROFILES:
            chosen = static[hardware][profile]
            for seed in CONFIRMATION_SEEDS:
                for policy in POLICIES:
                    phi = chosen["selected_phi"] if policy == "static_joint_yardstick" else None
                    psi = chosen["selected_psi_a100"] if policy == "static_joint_yardstick" else None
                    runs.append(
                        Run("confirmation", profile, hardware, seed, policy, phi, psi)
                    )
    return runs


def ci95(values: list[float]) -> tuple[float, float, float]:
    return base.ci95(values)


def analyze_confirmation(
    out: Path, runs: list[Run], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    cells = [(hardware, profile) for hardware in prior.HARDWARES for profile in PROFILES]
    means: dict[str, dict[str, float]] = {}
    remote_means: dict[str, dict[str, float]] = {}
    per_seed: dict[str, dict[str, list[dict[str, Any]]]] = {}
    class_means: dict[str, dict[str, dict[str, float]]] = {}
    spike_segment_means: dict[str, dict[str, dict[str, float]]] = {}
    for hardware, profile in cells:
        cell = f"{hardware}:{profile}"
        means[cell] = {}
        remote_means[cell] = {}
        per_seed[cell] = {}
        class_means[cell] = {}
        if profile == "short_spike":
            spike_segment_means[cell] = {}
        for policy in POLICIES:
            members = [
                row
                for row in rows
                if row["hardware"] == hardware
                and row["profile"] == profile
                and row["policy"] == policy
            ]
            members.sort(key=lambda row: int(row["seed"]))
            if len(members) != len(CONFIRMATION_SEEDS):
                raise RuntimeError(f"incomplete confirmation {cell}/{policy}")
            means[cell][policy] = statistics.fmean(float(row["goodput"]) for row in members)
            remote_means[cell][policy] = statistics.fmean(
                float(row["realized_phi"]) for row in members
            )
            class_means[cell][policy] = {
                name: statistics.fmean(float(row[f"{name}_goodput"]) for row in members)
                for name in CLASSES
            }
            if profile == "short_spike":
                spike_segment_means[cell][policy] = {
                    label: statistics.fmean(
                        float(row[f"segment_{label}_goodput"]) for row in members
                    )
                    for label in ("pre", "during", "post")
                }
            per_seed[cell][policy] = [
                {
                    "seed": int(row["seed"]),
                    "goodput": float(row["goodput"]),
                    "remote_fraction": float(row["realized_phi"]),
                    "class_goodput": {
                        name: float(row[f"{name}_goodput"]) for name in CLASSES
                    },
                    "class_counts": {name: int(row[f"{name}_count"]) for name in CLASSES},
                    "spike_segments": {
                        label: float(row.get(f"segment_{label}_goodput", 0.0))
                        for label in ("pre", "during", "post")
                    }
                    if profile == "short_spike"
                    else {},
                }
                for row in members
            ]

    best = {cell: max(means[cell][policy] for policy in DEPLOYABLE) for cell in means}
    ranking = []
    for policy in DEPLOYABLE:
        regrets = {cell: best[cell] - means[cell][policy] for cell in means}
        ranking.append(
            {
                "policy": policy,
                "worst_regret": max(regrets.values()),
                "worst_cell": max(regrets, key=regrets.get),
                "mean_goodput": statistics.fmean(means[cell][policy] for cell in means),
                "regret_by_cell": regrets,
            }
        )
    ranking.sort(key=lambda item: (item["worst_regret"], -item["mean_goodput"], item["policy"]))

    # Descriptive robustness check only: the preregistered primary metric above
    # weights every request equally.  Capacity-balanced mixtures contain many
    # more interactive than reasoning requests, so also expose an equal-class
    # macro average without changing the primary ranking post hoc.
    macro_class_means = {
        cell: {
            policy: statistics.fmean(class_means[cell][policy].values())
            for policy in POLICIES
        }
        for cell in class_means
    }
    best_macro = {
        cell: max(macro_class_means[cell][policy] for policy in DEPLOYABLE)
        for cell in macro_class_means
    }
    macro_ranking = []
    for policy in DEPLOYABLE:
        regrets = {
            cell: best_macro[cell] - macro_class_means[cell][policy]
            for cell in macro_class_means
        }
        macro_ranking.append(
            {
                "policy": policy,
                "worst_regret": max(regrets.values()),
                "worst_cell": max(regrets, key=regrets.get),
                "mean_macro_goodput": statistics.fmean(
                    macro_class_means[cell][policy] for cell in macro_class_means
                ),
                "regret_by_cell": regrets,
            }
        )
    macro_ranking.sort(
        key=lambda item: (
            item["worst_regret"],
            -item["mean_macro_goodput"],
            item["policy"],
        )
    )

    paired: dict[str, Any] = {}
    for comparator in POLICIES:
        if comparator == FOCAL:
            continue
        deltas = []
        for cell in means:
            left = {item["seed"]: item["goodput"] for item in per_seed[cell][FOCAL]}
            right = {item["seed"]: item["goodput"] for item in per_seed[cell][comparator]}
            deltas.extend(left[seed] - right[seed] for seed in CONFIRMATION_SEEDS)
        mean, low, high = ci95(deltas)
        paired[comparator] = {"mean_delta": mean, "ci_low": low, "ci_high": high}

    trace_exact = True
    trace_runs = 0
    for run in runs:
        if run.policy != FOCAL:
            continue
        summary = diagnostic.analyze_candidate_trace(trace_path(out, run), V)
        trace_runs += 1
        trace_exact = trace_exact and (
            summary["positive_chosen_snapshot_score_regret_fraction"] == 0
        )

    return {
        "status": "held-out mixed-workload/burst confirmation; no confirmation tuning",
        "routing_value": "smooth TTFT x E2E; reported goodput is hard TTFT x ITL x E2E",
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "run_count": len(runs),
        "cell_count": len(cells),
        "policy_means": means,
        "remote_means": remote_means,
        "class_means": class_means,
        "spike_segment_means": spike_segment_means,
        "per_seed": per_seed,
        "deployable_minimax_ranking": ranking,
        "descriptive_equal_class_means": macro_class_means,
        "descriptive_equal_class_minimax_ranking": macro_ranking,
        "focal_paired_deltas": paired,
        "hard_invalid_runs": sum(not row["hard_valid"] for row in rows),
        "runs_with_drops": sum(int(row["dropped"]) > 0 for row in rows),
        "runs_with_timeouts": sum(int(row["timed_out"]) > 0 for row in rows),
        "runs_with_length_caps": sum(int(row["length_capped"]) > 0 for row in rows),
        "focal_candidate_trace_runs": trace_runs,
        "chosen_argmin_trace_exact": trace_exact,
    }


def write_report(
    out: Path,
    static: dict[str, Any],
    result: dict[str, Any],
    filename: str = "PUBLIC-MIXED-BURST-BENCHMARK.md",
) -> None:
    lines = [
        "# Public mixed-workload and burst benchmark",
        "",
        "The routing value is smooth TTFT x E2E; reported goodput remains the hard",
        "TTFT/mean-ITL/E2E conjunction.",
        "",
        f"Kairos arm: `{KAIROS_POLICY}`.",
        "",
        "## Frozen static plans",
        "",
        "| fleet/profile | phi | psi | calibration goodput |",
        "|---|---:|---:|---:|",
    ]
    for hardware in prior.HARDWARES:
        for profile in PROFILES:
            item = static[hardware][profile]
            lines.append(
                f"| `{hardware}:{profile}` | {item['selected_phi']:.2f} | "
                f"{item['selected_psi_a100']:.2f} | {item['calibration_mean_goodput']:.3f} |"
            )
    lines += [
        "",
        "## Held-out mean goodput",
        "",
        "| cell | causal externality | least TTFT | Kairos | class-aware llm-d | tuned static | focal remote |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell, means in result["policy_means"].items():
        lines.append(
            f"| `{cell}` | {means[FOCAL]:.3f} | {means['least_ttft_joint']:.3f} | "
            f"{means[KAIROS_POLICY]:.3f} | {means[LLMD_CLASS_PREFIX]:.3f} | "
            f"{means['static_joint_yardstick']:.3f} | "
            f"{result['remote_means'][cell][FOCAL]:.3f} |"
        )
    lines += [
        "",
        "## Held-out mean goodput by class",
        "",
        "| cell | policy | interactive | reasoning | deep research | remote fraction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell, policies in result["class_means"].items():
        for policy, values in policies.items():
            lines.append(
                f"| `{cell}` | `{policy}` | {values['interactive']:.3f} | "
                f"{values['reasoning']:.3f} | {values['deep_research']:.3f} | "
                f"{result['remote_means'][cell][policy]:.3f} |"
            )
    lines += [
        "",
        "## Short-spike segment goodput",
        "",
        "| cell | policy | before spike | during 1.60 load | after spike |",
        "|---|---|---:|---:|---:|",
    ]
    for cell, policies in result["spike_segment_means"].items():
        for policy, values in policies.items():
            lines.append(
                f"| `{cell}` | `{policy}` | {values['pre']:.3f} | "
                f"{values['during']:.3f} | {values['post']:.3f} |"
            )
    lines += [
        "",
        "## Deployable minimax ranking",
        "",
        "| rank | policy | worst regret | worst cell | equal-cell mean |",
        "|---:|---|---:|---|---:|",
    ]
    for rank, item in enumerate(result["deployable_minimax_ranking"], 1):
        lines.append(
            f"| {rank} | `{item['policy']}` | {item['worst_regret']:.4f} | "
            f"`{item['worst_cell']}` | {item['mean_goodput']:.4f} |"
        )
    lines += [
        "",
        "## Descriptive equal-class check",
        "",
        "The registered ranking above weights every request equally. Capacity-balanced",
        "mixtures contain many more interactive requests, so the following non-primary",
        "check gives interactive, reasoning, and deep research equal weight within a cell.",
        "",
        "| rank | policy | worst regret | worst cell | equal-cell macro mean |",
        "|---:|---|---:|---|---:|",
    ]
    for rank, item in enumerate(result["descriptive_equal_class_minimax_ranking"], 1):
        lines.append(
            f"| {rank} | `{item['policy']}` | {item['worst_regret']:.4f} | "
            f"`{item['worst_cell']}` | {item['mean_macro_goodput']:.4f} |"
        )
    lines += [
        "",
        "## Focal paired deltas",
        "",
        "| comparator | mean delta | 95% interval |",
        "|---|---:|---:|",
    ]
    for policy, item in result["focal_paired_deltas"].items():
        lines.append(
            f"| `{policy}` | {item['mean_delta']:+.4f} | "
            f"[{item['ci_low']:+.4f}, {item['ci_high']:+.4f}] |"
        )
    lines += [
        "",
        "## Validity",
        "",
        f"- Runs: {result['run_count']} across {result['cell_count']} cells.",
        f"- Hard-invalid runs: {result['hard_invalid_runs']}.",
        f"- Runs with drops: {result['runs_with_drops']}.",
        f"- Runs with timeouts: {result['runs_with_timeouts']}.",
        f"- Runs with length caps: {result['runs_with_length_caps']}.",
        f"- Focal candidate traces: {result['focal_candidate_trace_runs']}; "
        + ("all chosen actions are exact argmins." if result["chosen_argmin_trace_exact"] else "ARGMIN CHECK FAILED."),
        "",
        "The static comparator is a condition-tuned yardstick, not an oracle.",
        "",
    ]
    (out / filename).write_text("\n".join(lines))


def run_confirmation(out: Path, jobs: int, force: bool) -> None:
    static = load_static(out)
    runs = confirmation_runs(static)
    if len(runs) != 160:
        raise RuntimeError(f"confirmation has {len(runs)} runs, want 160")
    rows = run_many(runs, out, jobs, force)
    require_valid(rows, "confirmation")
    write_rows(out / "confirmation.csv", rows)
    result = analyze_confirmation(out, runs, rows)
    (out / "confirmation_result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(out, static, result)
    print(json.dumps(result, indent=2), flush=True)


def smoke(out: Path, force: bool) -> None:
    runs = [
        Run("smoke", "sequential_shift", "h100_homogeneous", 314159, FOCAL, smoke=True),
        Run("smoke", "short_spike", "h100_a100_realistic", 314159, "static_joint_yardstick", 0.5, 0.5, True),
    ]
    rows = run_many(runs, out, jobs=2, force=force)
    require_valid(rows, "smoke")
    summary = diagnostic.analyze_candidate_trace(trace_path(out, runs[0]), V)
    if summary["positive_chosen_snapshot_score_regret_fraction"] != 0:
        raise RuntimeError("smoke focal candidate trace is not exact")
    write_rows(out / "smoke.csv", rows)
    print("smoke passed: mixed classes, lifecycle windows, terminal accounting, and focal argmin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage", choices=("smoke", "static", "confirm", "all"), default="all")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not PROTOCOL.exists():
        raise SystemExit(f"missing frozen protocol: {PROTOCOL}")
    if not prior.HETERO_BUNDLE.exists():
        raise SystemExit(f"missing hardware bundle: {prior.HETERO_BUNDLE}")
    if args.stage == "smoke":
        smoke(out, args.force)
        return
    if args.stage in {"static", "all"}:
        run_static(out, args.jobs, args.force)
    if args.stage in {"confirm", "all"}:
        run_confirmation(out, args.jobs, args.force)


if __name__ == "__main__":
    main()
