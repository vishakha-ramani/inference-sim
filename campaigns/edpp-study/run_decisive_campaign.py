#!/usr/bin/env python3
"""Run the pre-registered decisive EDPP experiment.

The script is resumable. Each run writes a metrics JSON, stdout log, and one row
to a phase CSV. Existing valid metrics are reused unless --force is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "campaigns" / "edpp-study"
DEFAULT_OUT = CAMPAIGN / "out" / "decisive"
MODEL = "meta-llama/llama-3.3-70b-instruct"
COEFFS = ROOT / "scripts" / "calibration" / "coeffs-llama70b-h100-tp4.json"
CALIBRATION_SEED = 42
HELD_OUT_SEEDS = (7, 123, 2024, 9001)
ALL_SEEDS = (CALIBRATION_SEED,) + HELD_OUT_SEEDS
TARGET_SEEDS = (101, 211, 307, 401, 503)
PHIS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
THRESHOLDS = (0, 512, 2048, 8192, 32768, 100000)
KAIROS_BETAS = (0.25, 0.5, 1.0)
SLO_FACTORS = (0.8, 1.2)
ARRIVAL_SECONDS = 300
CAPACITY_REQUESTS = {
    "synth": 1200,
    "rag": 1200,
    "shared": 6000,
}
TARGET_REQUESTS = {
    "synth": 1200,
    "rag": 1500,
    "shared": 6000,
}
POLICY_MIN_REQUESTS = {
    "synth": 800,
    "rag": 1000,
    "shared": 4000,
}


@dataclass(frozen=True)
class Workload:
    name: str
    source: Path
    slo_classes: tuple[str, ...]
    saturation_rate: float


WORKLOADS = {
    "synth": Workload(
        "synth",
        ROOT / "inference-perf-batch-synthetic-data-generation.yaml",
        ("batch",),
        4.0,
    ),
    "rag": Workload(
        "rag",
        ROOT / "inference-perf-batch-summarization-rag.yaml",
        ("standard", "batch"),
        8.0,
    ),
    "shared": Workload(
        "shared",
        ROOT / "examples" / "inference-perf-shared-prefix.yaml",
        ("standard",),
        160.0,
    ),
}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    rank = q * (len(values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(values), 1.96
    )
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def make_spec(
    workload: Workload,
    rate: float,
    seed: int,
    requests: int,
    path: Path,
    stage_rates: tuple[float, float] | None = None,
) -> None:
    data = yaml.safe_load(workload.source.read_text())
    data["seed"] = seed
    if workload.name == "shared":
        stages = data["inference_perf"]["stages"]
        if stage_rates is None:
            duration = max(60, int(math.ceil(requests / max(rate, 0.01))))
            data["inference_perf"]["stages"] = [{"rate": rate, "duration": duration}]
        else:
            duration = max(
                60,
                int(math.ceil(requests / max(sum(stage_rates), 0.01))),
            )
            stages[0]["rate"], stages[1]["rate"] = stage_rates
            stages[0]["duration"] = duration
            stages[1]["duration"] = duration
    else:
        data["aggregate_rate"] = rate
        data["num_requests"] = requests
        if workload.name == "rag":
            data["clients"][0]["slo_class"] = "standard"
            data["clients"][1]["slo_class"] = "batch"
    write_yaml(path, data)


def make_plan(
    path: Path,
    requests: int,
    phi: float,
    preserve_routing: bool = False,
    prefill_instances: int = 1,
    decode_instances: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(CAMPAIGN / "make_pd_plan.py"),
        "--n",
        str(requests),
        "--phi",
        str(phi),
        "--prefill-instances",
        str(prefill_instances),
        "--decode-instances",
        str(decode_instances),
    ]
    if preserve_routing:
        command.append("--preserve-routing")
    proc = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    path.write_text(proc.stdout)


def topology_args(topology: str) -> list[str]:
    if topology == "3m":
        return [
            "--num-instances",
            "3",
            "--routing-scorers",
            "queue-depth:1",
            "--max-num-running-reqs",
            "256",
        ]
    if topology == "1p2m":
        return [
            "--num-instances",
            "3",
            "--prefill-instances",
            "1",
            "--decode-instances",
            "2",
            "--decode-routing-scorers",
            "queue-depth:1",
            "--max-num-running-reqs",
            "256",
        ]
    generalized = {
        "1p3m": (1, 3),
        "2p2m": (2, 2),
        "3p1m": (3, 1),
    }
    if topology in generalized:
        prefill, decode = generalized[topology]
        return [
            "--num-instances",
            str(prefill + decode),
            "--prefill-instances",
            str(prefill),
            "--decode-instances",
            str(decode),
            "--decode-routing-scorers",
            "queue-depth:1",
            "--max-num-running-reqs",
            "256",
        ]
    raise ValueError(f"unknown topology {topology}")


def class_target_string(targets: dict[str, dict[str, float]], dimension: str) -> str:
    return ",".join(
        f"{slo_class}={cli_target_value(values[dimension]):.3f}ms"
        for slo_class, values in sorted(targets.items())
    )


def cli_target_value(value: float) -> float:
    """Return the exact millisecond value sent to BLIS."""
    return float(f"{value:.3f}")


def slo_args(targets: dict[str, dict[str, float]]) -> list[str]:
    return [
        "--slo-ttft",
        class_target_string(targets, "ttft_ms"),
        "--slo-itl",
        class_target_string(targets, "itl_ms"),
        "--slo-e2e",
        class_target_string(targets, "e2e_ms"),
    ]


def scale_targets(
    targets: dict[str, dict[str, float]], factor: float
) -> dict[str, dict[str, float]]:
    return {
        slo_class: {
            dimension: value * factor for dimension, value in dimensions.items()
        }
        for slo_class, dimensions in targets.items()
    }


def edpp_common(targets: dict[str, dict[str, float]]) -> list[str]:
    first = targets[sorted(targets)[0]]
    args = [
        "--pd-decider",
        "edpp",
        # The validated scheduler rollout consumes an arrival-time FCFS
        # snapshot. Do not let the generic 50 ms observability default supply
        # an all-zero initial snapshot or stale running/waiting membership.
        "--snapshot-refresh-interval",
        "0",
        "--scheduler",
        "fcfs",
        "--preemption-policy",
        "fcfs",
        "--edpp-coeffs",
        str(COEFFS),
        "--edpp-tadm-estimator",
        "rollforward",
        "--edpp-c-xfer-size-aware",
        "--edpp-tau-ttft",
        f"{cli_target_value(first['ttft_ms']):.3f}ms",
        "--edpp-tau-itl",
        f"{cli_target_value(first['itl_ms']):.3f}ms",
    ]
    if len(targets) > 1:
        args += [
            "--edpp-tau-ttft-classes",
            class_target_string(targets, "ttft_ms"),
            "--edpp-tau-itl-classes",
            class_target_string(targets, "itl_ms"),
        ]
    return args


def policy_args(
    policy: str,
    targets: dict[str, dict[str, float]],
    parameter: float | int | None,
    plan_path: Path | None,
) -> list[str]:
    if policy == "aggregate":
        return []
    if policy == "always":
        return ["--pd-decider", "always"]
    if policy == "never":
        return ["--pd-decider", "never"]
    if policy in {
        "fixed_phi",
        "universal_phi",
        "tuned_phi",
        "routing_phi",
        "universal_routing_phi",
        "oracle_routing_phi",
        "conditional_static_joint",
    }:
        if plan_path is None:
            raise ValueError(f"{policy} requires a plan")
        return ["--pd-plan", str(plan_path)]
    if policy in {"threshold", "universal_threshold", "tuned_threshold"}:
        return [
            "--pd-decider",
            "prefix-threshold",
            "--pd-prefix-threshold",
            str(int(parameter)),
        ]
    common = edpp_common(targets)
    if policy == "least_ttft_joint":
        return common + [
            "--edpp-rule",
            "least-ttft",
            "--edpp-joint",
            "--edpp-ttft-overlap-aware",
        ]
    if policy == "joint_causal_var":
        first = targets[sorted(targets)[0]]
        e2e_classes = class_target_string(targets, "e2e_ms")
        return common + [
            "--edpp-rule",
            "var",
            "--edpp-var-metric",
            "util",
            "--edpp-var-deployable",
            "--edpp-var-exact-prefill-overlap",
            "--edpp-joint-causal-var",
            "--edpp-tau-e2e",
            f"{cli_target_value(first['e2e_ms']):.3f}ms",
        ] + (["--edpp-tau-e2e-classes", e2e_classes] if len(targets) > 1 else [])
    if policy == "decomposed_causal_var":
        first = targets[sorted(targets)[0]]
        e2e_classes = class_target_string(targets, "e2e_ms")
        return common + [
            "--edpp-rule",
            "var",
            "--edpp-var-metric",
            "util",
            "--edpp-var-deployable",
            "--edpp-var-exact-prefill-overlap",
            "--edpp-decomposed-causal-var",
            "--edpp-tau-e2e",
            f"{cli_target_value(first['e2e_ms']):.3f}ms",
        ] + (["--edpp-tau-e2e-classes", e2e_classes] if len(targets) > 1 else [])
    if policy in {
        "joint_slo_externality",
        "joint_slo_externality_occupancy",
        "decomposed_slo_externality",
        "joint_slo_externality_no_externality",
        "joint_slo_externality_no_capacity",
    }:
        first = targets[sorted(targets)[0]]
        e2e_classes = class_target_string(targets, "e2e_ms")
        mode = {
            "joint_slo_externality": ["--edpp-joint-slo-externality"],
            "joint_slo_externality_occupancy": [
                "--edpp-joint-slo-externality",
                "--edpp-slo-externality-occupancy-capacity",
            ],
            "decomposed_slo_externality": ["--edpp-decomposed-slo-externality"],
            "joint_slo_externality_no_externality": [
                "--edpp-joint-slo-externality",
                "--edpp-slo-externality-no-externality",
            ],
            "joint_slo_externality_no_capacity": [
                "--edpp-joint-slo-externality",
                "--edpp-slo-externality-no-capacity",
            ],
        }[policy]
        return common + [
            "--edpp-v",
            str(1.0 if parameter is None else parameter),
            *mode,
            "--edpp-tau-e2e",
            f"{cli_target_value(first['e2e_ms']):.3f}ms",
        ] + (["--edpp-tau-e2e-classes", e2e_classes] if len(targets) > 1 else [])
    if policy == "least_ttft":
        return common + ["--edpp-rule", "least-ttft"]
    if policy == "dpp_joint":
        return common + ["--edpp-joint"]
    if policy == "kairos":
        return common + ["--edpp-rule", "kairos", "--kairos-beta", str(parameter)]
    if policy == "kairos-paper":
        return common + [
            "--edpp-rule", "kairos-paper",
            "--kairos-alpha", "1.3",
            "--kairos-beta", "1.0",
        ]
    if policy == "dpvar":
        first = targets[sorted(targets)[0]]
        e2e_classes = class_target_string(targets, "e2e_ms")
        return common + [
            "--edpp-rule",
            "var",
            "--edpp-var-metric",
            "util",
            "--edpp-joint",
            "--edpp-var-congestion",
            "--edpp-var-normalize",
            "--edpp-var-congestion-weight",
            "1",
            "--edpp-var-deployable",
            "--edpp-var-goodput",
            "--edpp-tau-e2e",
            f"{cli_target_value(first['e2e_ms']):.3f}ms",
        ] + (["--edpp-tau-e2e-classes", e2e_classes] if len(targets) > 1 else [])
    if policy in {
        "var_prefill",
        "var_prefill_self",
        "var_prefill_nostability",
        "var_prefill_nostability_overlap",
        "var_prefill_nostability_exactvar",
        "var_prefill_nostability_exactvar_overlap",
        "var_prefill_nostability_exactvar_pathwork",
        "var_prefill_exactvar_pathwork",
        "var_prefill_self_nostability_overlap",
    }:
        first = targets[sorted(targets)[0]]
        e2e_classes = class_target_string(targets, "e2e_ms")
        prefill_weight = 0 if "nostability" in policy else parameter
        args = common + [
            "--edpp-rule",
            "var-prefill",
            "--edpp-var-metric",
            "util",
            "--edpp-var-prefill-weight",
            str(prefill_weight),
            "--edpp-var-deployable",
            "--edpp-tau-e2e",
            f"{cli_target_value(first['e2e_ms']):.3f}ms",
        ] + (["--edpp-tau-e2e-classes", e2e_classes] if len(targets) > 1 else [])
        if "self" in policy:
            args.append("--edpp-var-goodput")
        if "overlap" in policy:
            args.append("--edpp-ttft-overlap-aware")
        if "exactvar" in policy:
            args.append("--edpp-var-exact-prefill-overlap")
        if "pathwork" in policy:
            args.append("--edpp-path-specific-prefill-work")
        return args
    raise ValueError(f"unknown policy {policy}")


def parse_stdout(text: str) -> tuple[int | None, float | None]:
    disagg = re.search(r"Disaggregated Requests:\s*(\d+)", text)
    cache = re.search(r"Cache Hit Rate:\s*([0-9.]+)", text)
    return (
        int(disagg.group(1)) if disagg else None,
        float(cache.group(1)) if cache else None,
    )


def central_completion_rate(metrics: dict[str, Any]) -> float:
    completion = sorted(
        float(r["arrived_at"]) + float(r["e2e_ms"]) / 1000.0
        for r in metrics.get("requests", [])
    )
    if len(completion) < 20:
        return 0.0
    lo = max(0, int(0.10 * len(completion)))
    hi = min(len(completion) - 1, int(0.90 * len(completion)))
    elapsed = completion[hi] - completion[lo]
    return (hi - lo) / elapsed if elapsed > 0 else 0.0


def result_row(
    *,
    phase: str,
    workload: str,
    rate_label: str,
    rate: float,
    seed: int,
    policy: str,
    parameter: float | int | None,
    topology: str,
    metrics_path: Path,
    stdout_path: Path,
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text())
    stdout = stdout_path.read_text()
    disagg, cache_hit = parse_stdout(stdout)
    if disagg is None and policy != "aggregate":
        # BLIS emits the PD Metrics block only after at least one request is
        # disaggregated. Every non-aggregate campaign arm installs a P/D
        # decider, so an absent counter means an observed count of zero rather
        # than missing data.
        disagg = 0
    requests = metrics.get("requests", [])
    arrivals = sorted(float(r["arrived_at"]) for r in requests)
    actual_rate = 0.0
    if len(arrivals) > 1 and arrivals[-1] > arrivals[0]:
        actual_rate = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
    classes = metrics.get("per_class", {})
    class_count = sum(int(value.get("count", 0)) for value in classes.values())
    dims = {}
    for dimension in ("ttft", "itl", "e2e"):
        numerator = sum(
            int(value.get("count", 0))
            * float(value.get("slo_attainment_by_dim", {}).get(dimension, 0.0))
            for value in classes.values()
        )
        dims[dimension] = numerator / class_count if class_count else 0.0
    injected = int(metrics.get("injected_requests", 0))
    completed = int(metrics.get("completed_requests", 0))
    still_queued = int(metrics.get("still_queued", 0))
    still_running = int(metrics.get("still_running", 0))
    dropped = int(metrics.get("dropped_unservable", 0))
    timed_out = int(metrics.get("timed_out_requests", 0))
    capped = int(metrics.get("length_capped_requests", 0))
    conserved = completed + still_queued + still_running + dropped + timed_out == injected
    share_valid = True
    if phase != "capacity" and policy in {
        "fixed_phi",
        "universal_phi",
        "tuned_phi",
        "routing_phi",
        "universal_routing_phi",
        "oracle_routing_phi",
        "conditional_static_joint",
    }:
        expected_disaggregated = round(float(parameter) * injected)
        share_valid = disagg is not None and abs(disagg - expected_disaggregated) <= 1
    valid = (
        conserved
        and (dropped == 0 or phase == "capacity")
        and timed_out == 0
        and still_queued == 0
        and still_running == 0
        and capped == 0
        and share_valid
        and (rate_label == "staged" or abs(actual_rate - rate) <= 0.10 * rate)
    )
    return {
        "phase": phase,
        "workload": workload,
        "rate_label": rate_label,
        "offered_rate": rate,
        "actual_rate": actual_rate,
        "seed": seed,
        "policy": policy,
        "parameter": "" if parameter is None else parameter,
        "topology": topology,
        "injected": injected,
        "completed": completed,
        "dropped": dropped,
        "still_queued": still_queued,
        "still_running": still_running,
        "timed_out": timed_out,
        "length_capped": capped,
        "goodput": float(metrics.get("slo_attainment", 0.0)),
        "good_ttft": float(dims.get("ttft", 0.0)),
        "good_itl": float(dims.get("itl", 0.0)),
        "good_e2e": float(dims.get("e2e", 0.0)),
        "ttft_p99_ms": float(metrics.get("ttft_p99_ms", 0.0)),
        "itl_mean_ms": float(metrics.get("itl_mean_ms", 0.0)),
        "e2e_p99_ms": float(metrics.get("e2e_p99_ms", 0.0)),
        "central_completion_rps": central_completion_rate(metrics),
        "disaggregated": "" if disagg is None else disagg,
        "realized_phi": "" if disagg is None or injected == 0 else disagg / injected,
        "cache_hit_rate": "" if cache_hit is None else cache_hit,
        "valid": valid,
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "stdout_path": str(stdout_path.relative_to(ROOT)),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class Run:
    phase: str
    workload: Workload
    rate_label: str
    rate: float
    seed: int
    policy: str
    parameter: float | int | None
    topology: str
    requests: int
    targets: dict[str, dict[str, float]]
    stage_rates: tuple[float, float] | None = None

    def tag(self) -> str:
        parameter = "none" if self.parameter is None else str(self.parameter).replace(".", "p")
        topology = (
            f"{self.topology}_"
            if self.phase.startswith(("joint_routing_", "slo_externality_"))
            else ""
        )
        return (
            f"{topology}{self.workload.name}_{self.rate_label}_{self.rate:g}_"
            f"{self.policy}_{parameter}_s{self.seed}_n{self.requests}"
        )


def execute_run(run: Run, out: Path, force: bool) -> dict[str, Any]:
    run_dir = out / run.phase
    spec_path = run_dir / "specs" / f"{run.tag()}.yaml"
    metrics_path = run_dir / "metrics" / f"{run.tag()}.json"
    stdout_path = run_dir / "stdout" / f"{run.tag()}.txt"
    plan_path: Path | None = None
    make_spec(
        run.workload,
        run.rate,
        run.seed,
        run.requests,
        spec_path,
        run.stage_rates,
    )
    if run.policy in {
        "fixed_phi",
        "universal_phi",
        "tuned_phi",
        "routing_phi",
        "universal_routing_phi",
        "oracle_routing_phi",
        "conditional_static_joint",
    }:
        plan_path = run_dir / "plans" / f"{run.tag()}.csv"
        topology_counts = {
            "1p2m": (1, 2),
            "1p3m": (1, 3),
            "2p2m": (2, 2),
            "3p1m": (3, 1),
        }
        prefill_instances, decode_instances = topology_counts[run.topology]
        make_plan(
            plan_path,
            run.requests,
            float(run.parameter),
            preserve_routing=run.policy
            in {"routing_phi", "universal_routing_phi", "oracle_routing_phi"},
            prefill_instances=prefill_instances,
            decode_instances=decode_instances,
        )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and metrics_path.exists() and stdout_path.exists():
        return result_row(
            phase=run.phase,
            workload=run.workload.name,
            rate_label=run.rate_label,
            rate=run.rate,
            seed=run.seed,
            policy=run.policy,
            parameter=run.parameter,
            topology=run.topology,
            metrics_path=metrics_path,
            stdout_path=stdout_path,
        )

    command = [
        str(ROOT / "blis"),
        "run",
        "--model",
        MODEL,
        "--workload-spec",
        str(spec_path),
        "--num-requests",
        str(run.requests),
        *topology_args(run.topology),
        *slo_args(run.targets),
        *policy_args(run.policy, run.targets, run.parameter, plan_path),
        "--seed",
        str(run.seed),
        "--metrics-path",
        str(metrics_path),
    ]
    if run.policy in {
        "joint_causal_var",
        "decomposed_causal_var",
        "joint_slo_externality",
        "joint_slo_externality_occupancy",
        "decomposed_slo_externality",
        "joint_slo_externality_no_externality",
        "joint_slo_externality_no_capacity",
    }:
        candidate_path = run_dir / "candidate_traces" / f"{run.tag()}.csv"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        command += ["--edpp-joint-candidate-trace", str(candidate_path)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    stdout_path.write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"run failed ({run.tag()}):\n{proc.stderr[-4000:]}\n{proc.stdout[-4000:]}"
        )
    return result_row(
        phase=run.phase,
        workload=run.workload.name,
        rate_label=run.rate_label,
        rate=run.rate,
        seed=run.seed,
        policy=run.policy,
        parameter=run.parameter,
        topology=run.topology,
        metrics_path=metrics_path,
        stdout_path=stdout_path,
    )


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
                f"[{index}/{len(runs)}] {run.tag()} "
                f"goodput={row['goodput']:.3f} valid={row['valid']}",
                flush=True,
            )
    return sorted(
        rows,
        key=lambda row: (
            row["workload"],
            row["rate_label"],
            row["policy"],
            str(row["parameter"]),
            int(row["seed"]),
        ),
    )


def require_valid(rows: list[dict[str, Any]], phase: str) -> None:
    invalid = [
        f"{row['workload']}/{row['rate_label']}/{row['policy']}/s{row['seed']}"
        for row in rows
        if not row["valid"]
    ]
    if invalid:
        raise RuntimeError(f"{phase} produced invalid runs: {', '.join(invalid)}")


def loose_targets(workload: Workload) -> dict[str, dict[str, float]]:
    return {
        slo_class: {
            "ttft_ms": 999000.0,
            "itl_ms": 999000.0,
            "e2e_ms": 999000.0,
        }
        for slo_class in workload.slo_classes
    }


def phase_probe(out: Path, jobs: int, force: bool) -> None:
    runs = [
        Run(
            "probe",
            workload,
            "source",
            workload.saturation_rate,
            CALIBRATION_SEED,
            "always",
            None,
            "1p2m",
            500 if workload.name != "rag" else 800,
            loose_targets(workload),
        )
        for workload in WORKLOADS.values()
    ]
    rows = run_many(runs, out, jobs, force)
    require_valid(rows, "probe")
    write_rows(out / "probe.csv", rows)
    report = []
    for row in rows:
        metrics = json.loads((ROOT / row["metrics_path"]).read_text())
        prompts = [int(r["num_prefill_tokens"]) for r in metrics["requests"]]
        report.append(
            {
                "workload": row["workload"],
                "requests": len(prompts),
                "prompt_min": min(prompts),
                "prompt_median": statistics.median(prompts),
                "prompt_max": max(prompts),
                "distinct_prompt_lengths": len(set(prompts)),
                "cache_hit_rate": row["cache_hit_rate"],
                "valid": row["valid"],
            }
        )
    (out / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    failed = []
    for item in report:
        print(item)
        if float(item["cache_hit_rate"] or 0.0) <= 0.0:
            failed.append(f"{item['workload']}: no prefix reuse")
        if (
            item["workload"] in {"synth", "rag"}
            and int(item["distinct_prompt_lengths"]) <= 1
        ):
            failed.append(f"{item['workload']}: constant prompt length")
    if failed:
        raise RuntimeError("workload probe failed: " + ", ".join(failed))


def phase_capacity(out: Path, jobs: int, force: bool) -> None:
    runs: list[Run] = []
    for workload in WORKLOADS.values():
        requests = CAPACITY_REQUESTS[workload.name]
        for phi in PHIS:
            runs.append(
                Run(
                    "capacity",
                    workload,
                    "saturated",
                    workload.saturation_rate,
                    CALIBRATION_SEED,
                    "fixed_phi",
                    phi,
                    "1p2m",
                    requests,
                    loose_targets(workload),
                )
            )
        for seed in TARGET_SEEDS:
            runs.append(
                Run(
                    "capacity",
                    workload,
                    "saturated",
                    workload.saturation_rate,
                    seed,
                    "aggregate",
                    None,
                    "3m",
                    requests,
                    loose_targets(workload),
                )
            )
    rows = run_many(runs, out, jobs, force)
    require_valid(rows, "capacity")
    write_rows(out / "capacity.csv", rows)
    capacity: dict[str, Any] = {}
    for workload in WORKLOADS:
        selected = [row for row in rows if row["workload"] == workload]
        family = [row for row in selected if row["topology"] == "1p2m"]
        aggregate = [row for row in selected if row["topology"] == "3m"]
        best = max(family, key=lambda row: float(row["central_completion_rps"]))
        endpoint = {
            float(row["parameter"]): float(row["central_completion_rps"])
            for row in family
            if float(row["parameter"]) in (0.0, 1.0)
        }
        low_endpoint = min(endpoint.values())
        fastest_observed = max(
            float(row["central_completion_rps"]) for row in selected
        )
        saturation_rate = WORKLOADS[workload].saturation_rate
        if saturation_rate < 1.05 * fastest_observed:
            raise RuntimeError(
                f"{workload} capacity probe was not saturated: offered "
                f"{saturation_rate:.3f} rps, observed "
                f"{fastest_observed:.3f} rps"
            )
        high_rate = max(
            0.95 * low_endpoint,
            0.70 * float(best["central_completion_rps"]),
        )
        if workload == "rag":
            high_rate = 0.90 * low_endpoint
        capacity[workload] = {
            "best_1p2m_rps": float(best["central_completion_rps"]),
            "best_phi": float(best["parameter"]),
            "phi0_rps": endpoint[0.0],
            "phi1_rps": endpoint[1.0],
            "reference_3m_rps": statistics.fmean(
                float(row["central_completion_rps"]) for row in aggregate
            ),
            "reference_3m_seed_rps": {
                str(row["seed"]): float(row["central_completion_rps"])
                for row in aggregate
            },
            "saturated_fixed_share": {
                str(row["parameter"]): {
                    "completion_rps": float(row["central_completion_rps"]),
                    "completed": int(row["completed"]),
                    "dropped": int(row["dropped"]),
                }
                for row in family
            },
            "saturated_reference_3m": {
                str(row["seed"]): {
                    "completion_rps": float(row["central_completion_rps"]),
                    "completed": int(row["completed"]),
                    "dropped": int(row["dropped"]),
                }
                for row in aggregate
            },
            "rates": {
                "low": 0.60 * low_endpoint,
                "medium": 0.85 * low_endpoint,
                "high": high_rate,
            },
        }
    (out / "capacity.json").write_text(json.dumps(capacity, indent=2) + "\n")
    print(json.dumps(capacity, indent=2))


def load_capacity(out: Path) -> dict[str, Any]:
    path = out / "capacity.json"
    if not path.exists():
        raise SystemExit("capacity.json is missing; run the capacity phase first")
    return json.loads(path.read_text())


def phase_targets(out: Path, jobs: int, force: bool) -> None:
    capacity = load_capacity(out)
    runs = []
    for workload in WORKLOADS.values():
        for utilization in (0.60, 0.70, 0.80):
            rate = utilization * capacity[workload.name]["reference_3m_rps"]
            requests = TARGET_REQUESTS[workload.name]
            if workload.name == "rag" and utilization == 0.70:
                requests = 3000
            for seed in TARGET_SEEDS:
                runs.append(
                    Run(
                        "targets",
                        workload,
                        f"u{int(utilization * 100)}",
                        rate,
                        seed,
                        "aggregate",
                        None,
                        "3m",
                        requests,
                        loose_targets(workload),
                    )
                )
    rows = run_many(runs, out, jobs, force)
    require_valid(rows, "targets")
    write_rows(out / "targets.csv", rows)
    targets: dict[str, dict[str, dict[str, float]]] = {}
    reliability: dict[str, Any] = {}
    for workload in WORKLOADS.values():
        targets[workload.name] = {}
        reliability[workload.name] = {}
        for slo_class in workload.slo_classes:
            reliability[workload.name][slo_class] = {}
            for utilization in (60, 70, 80):
                seed_quantiles: dict[str, dict[str, float]] = {}
                total_count = 0
                for row in rows:
                    if (
                        row["workload"] != workload.name
                        or row["rate_label"] != f"u{utilization}"
                    ):
                        continue
                    metrics = json.loads((ROOT / row["metrics_path"]).read_text())
                    class_requests = [
                        request
                        for request in metrics["requests"]
                        if request.get("slo_class") == slo_class
                    ]
                    total_count += len(class_requests)
                    seed_quantiles[str(row["seed"])] = {
                        "ttft_ms": percentile(
                            [float(request["ttft_ms"]) for request in class_requests], 0.90
                        ),
                        "itl_ms": percentile(
                            [float(request["itl_ms"]) for request in class_requests], 0.90
                        ),
                        "e2e_ms": percentile(
                            [float(request["e2e_ms"]) for request in class_requests], 0.90
                        ),
                    }
                dimensions = {}
                for dimension in ("ttft_ms", "itl_ms", "e2e_ms"):
                    values = [
                        seed_quantiles[str(seed)][dimension] for seed in TARGET_SEEDS
                    ]
                    mean, ci_low, ci_high = ci95(values)
                    cv = statistics.stdev(values) / mean if mean and len(values) > 1 else 0.0
                    relative_half_width = (ci_high - ci_low) / (2 * mean) if mean else math.inf
                    dimensions[dimension] = {
                        "target": mean,
                        "seed_values": {
                            str(seed): seed_quantiles[str(seed)][dimension]
                            for seed in TARGET_SEEDS
                        },
                        "standard_deviation": statistics.stdev(values),
                        "cv": cv,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "relative_ci_half_width": relative_half_width,
                        "reliable": (
                            total_count >= 500
                            and cv <= 0.15
                            and relative_half_width <= 0.20
                        ),
                    }
                reliability[workload.name][slo_class][f"u{utilization}"] = {
                    "request_count": total_count,
                    "dimensions": dimensions,
                }
                if utilization == 70:
                    targets[workload.name][slo_class] = {
                        dimension: dimensions[dimension]["target"]
                        for dimension in ("ttft_ms", "itl_ms", "e2e_ms")
                    }
    (out / "targets.json").write_text(json.dumps(targets, indent=2) + "\n")
    (out / "target_reliability.json").write_text(
        json.dumps(reliability, indent=2) + "\n"
    )
    report_lines = [
        "# SLO derivation report",
        "",
        "Targets are the mean of per-seed p90 latency at 70% of the independently",
        "measured three-mixed-instance reference capacity. Seeds 101, 211, 307,",
        "401, and 503 are disjoint from policy calibration and evaluation.",
        "",
        "| workload | class | load | dimension | p90 mean (ms) | 95% CI (ms) | CV | requests | stable |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for workload, classes in reliability.items():
        for slo_class, utilizations in classes.items():
            for utilization, record in utilizations.items():
                for dimension, stats in record["dimensions"].items():
                    report_lines.append(
                        f"| `{workload}` | `{slo_class}` | "
                        f"{int(utilization[1:])}% | `{dimension[:-3]}` | "
                        f"{stats['target']:.2f} | "
                        f"[{stats['ci_low']:.2f}, {stats['ci_high']:.2f}] | "
                        f"{stats['cv']:.3f} | {record['request_count']} | "
                        f"{'yes' if stats['reliable'] else 'no'} |"
                    )
    report_lines.extend(
        [
            "",
            "Only the 70% rows define the registered targets and are hard-gated.",
            "The 60% and 80% rows diagnose sensitivity to the reference load.",
            "",
        ]
    )
    (out / "SLO-REPORT.md").write_text("\n".join(report_lines))
    unreliable = [
        f"{workload}/{slo_class}/{utilization}/{dimension}"
        for workload, classes in reliability.items()
        for slo_class, utilizations in classes.items()
        for utilization, record in utilizations.items()
        for dimension, stats in record["dimensions"].items()
        if utilization == "u70" and not stats["reliable"]
    ]
    if unreliable:
        raise RuntimeError(
            "unreliable SLO derivation; increase target sample size before evaluation: "
            + ", ".join(unreliable)
        )
    print(json.dumps(targets, indent=2))


def load_targets(out: Path) -> dict[str, dict[str, dict[str, float]]]:
    path = out / "targets.json"
    if not path.exists():
        raise SystemExit("targets.json is missing; run the targets phase first")
    return json.loads(path.read_text())


def base_conditions(out: Path) -> list[tuple[Workload, str, float, int]]:
    capacity = load_capacity(out)
    result = []
    for workload in WORKLOADS.values():
        for label, rate in capacity[workload.name]["rates"].items():
            if workload.name == "shared":
                requests = POLICY_MIN_REQUESTS[workload.name]
            else:
                requests = max(
                    POLICY_MIN_REQUESTS[workload.name],
                    int(rate * ARRIVAL_SECONDS),
                )
            result.append((workload, label, rate, requests))
    return result


def phase_calibrate(out: Path, jobs: int, force: bool) -> None:
    targets = load_targets(out)
    runs: list[Run] = []
    for workload, label, rate, requests in base_conditions(out):
        for phi in PHIS:
            runs.append(
                Run(
                    "calibrate",
                    workload,
                    label,
                    rate,
                    CALIBRATION_SEED,
                    "fixed_phi",
                    phi,
                    "1p2m",
                    requests,
                    targets[workload.name],
                )
            )
        for threshold in THRESHOLDS:
            runs.append(
                Run(
                    "calibrate",
                    workload,
                    label,
                    rate,
                    CALIBRATION_SEED,
                    "threshold",
                    threshold,
                    "1p2m",
                    requests,
                    targets[workload.name],
                )
            )
        for beta in KAIROS_BETAS:
            runs.append(
                Run(
                    "calibrate",
                    workload,
                    label,
                    rate,
                    CALIBRATION_SEED,
                    "kairos",
                    beta,
                    "1p2m",
                    requests,
                    targets[workload.name],
                )
            )
    rows = run_many(runs, out, jobs, force)
    require_valid(rows, "calibrate")
    write_rows(out / "calibrate.csv", rows)

    selection: dict[str, Any] = {"condition": {}, "universal": {}}
    for workload, label, _, _ in base_conditions(out):
        condition_rows = [
            row
            for row in rows
            if row["workload"] == workload.name and row["rate_label"] == label
        ]
        condition_key = f"{workload.name}:{label}"
        selection["condition"][condition_key] = {}
        for policy in ("fixed_phi", "threshold", "kairos"):
            candidates = [row for row in condition_rows if row["policy"] == policy]
            best = max(candidates, key=lambda row: (float(row["goodput"]), -float(row["parameter"])))
            selection["condition"][condition_key][policy] = float(best["parameter"])

    for policy, parameters in (
        ("fixed_phi", PHIS),
        ("threshold", THRESHOLDS),
        ("kairos", KAIROS_BETAS),
    ):
        worst_by_parameter = {}
        mean_by_parameter = {}
        for parameter in parameters:
            values = [
                float(row["goodput"])
                for row in rows
                if row["policy"] == policy and float(row["parameter"]) == float(parameter)
            ]
            worst_by_parameter[parameter] = min(values)
            mean_by_parameter[parameter] = statistics.fmean(values)
        best_parameter = max(
            parameters,
            key=lambda parameter: (
                worst_by_parameter[parameter],
                mean_by_parameter[parameter],
                -float(parameter),
            ),
        )
        selection["universal"][policy] = float(best_parameter)
    (out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


def load_selection(out: Path) -> dict[str, Any]:
    path = out / "selection.json"
    if not path.exists():
        raise SystemExit("selection.json is missing; run the calibrate phase first")
    return json.loads(path.read_text())


def phase_evaluate(out: Path, jobs: int, force: bool) -> None:
    targets = load_targets(out)
    selection = load_selection(out)
    runs: list[Run] = []
    for workload, label, rate, requests in base_conditions(out):
        condition = selection["condition"][f"{workload.name}:{label}"]
        policies = [
            ("always", None),
            ("never", None),
            ("least_ttft_joint", None),
            ("dpp_joint", None),
            ("dpvar", None),
            ("kairos", selection["universal"]["kairos"]),
            ("universal_phi", selection["universal"]["fixed_phi"]),
            ("universal_threshold", selection["universal"]["threshold"]),
            ("tuned_phi", condition["fixed_phi"]),
            ("tuned_threshold", condition["threshold"]),
        ]
        for seed in ALL_SEEDS:
            for policy, parameter in policies:
                runs.append(
                    Run(
                        "evaluate",
                        workload,
                        label,
                        rate,
                        seed,
                        policy,
                        parameter,
                        "1p2m",
                        requests,
                        targets[workload.name],
                    )
                )
    rows = run_many(runs, out, jobs, force)
    require_valid(rows, "evaluate")
    write_rows(out / "evaluate.csv", rows)


def phase_sensitivity(out: Path, jobs: int, force: bool) -> None:
    targets = load_targets(out)
    runs = [
        Run(
            "sensitivity",
            workload,
            f"{label}_slo{int(factor * 100)}",
            rate,
            seed,
            "dpvar",
            factor,
            "1p2m",
            requests,
            scale_targets(targets[workload.name], factor),
        )
        for workload, label, rate, requests in base_conditions(out)
        for factor in SLO_FACTORS
        for seed in HELD_OUT_SEEDS
    ]
    rows = run_many(runs, out, jobs, force)
    require_valid(rows, "sensitivity")
    write_rows(out / "sensitivity.csv", rows)


def phase_staged(out: Path, jobs: int, force: bool) -> None:
    capacity = load_capacity(out)
    targets = load_targets(out)
    selection = load_selection(out)
    workload = WORKLOADS["shared"]
    low = capacity["shared"]["rates"]["low"]
    high = capacity["shared"]["rates"]["high"]
    requests = 6000
    policies = [
        ("always", None),
        ("never", None),
        ("least_ttft_joint", None),
        ("dpp_joint", None),
        ("dpvar", None),
        ("kairos", selection["universal"]["kairos"]),
        ("universal_phi", selection["universal"]["fixed_phi"]),
        ("universal_threshold", selection["universal"]["threshold"]),
    ]
    runs = [
        Run(
            "staged",
            workload,
            "staged",
            (low + high) / 2.0,
            seed,
            policy,
            parameter,
            "1p2m",
            requests,
            targets["shared"],
            (low, high),
        )
        for seed in ALL_SEEDS
        for policy, parameter in policies
    ]
    rows = run_many(runs, out, jobs, force)
    require_valid(rows, "staged")
    write_rows(out / "staged.csv", rows)


def aggregate_evaluation(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["workload"], row["rate_label"], row["policy"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (workload, label, policy), members in sorted(groups.items()):
        all_values = [float(row["goodput"]) for row in members]
        held_values = [
            float(row["goodput"])
            for row in members
            if int(row["seed"]) in HELD_OUT_SEEDS
        ]
        mean, lo, hi = ci95(held_values)
        summary.append(
            {
                "workload": workload,
                "rate_label": label,
                "policy": policy,
                "all_seed_mean": statistics.fmean(all_values),
                "heldout_mean": mean,
                "heldout_ci_low": lo,
                "heldout_ci_high": hi,
                "heldout_min": min(held_values),
                "valid": all(row["valid"].lower() == "true" for row in members),
            }
        )
    return summary


def offline_goodput(
    row: dict[str, str], targets: dict[str, dict[str, float]]
) -> float:
    metrics = json.loads((ROOT / row["metrics_path"]).read_text())
    requests = metrics.get("requests", [])
    injected = int(metrics.get("injected_requests", 0))
    good = 0
    for request in requests:
        slo_class = request.get("slo_class") or "standard"
        class_targets = targets.get(slo_class)
        if class_targets is None:
            raise RuntimeError(f"no SLO target for class {slo_class!r}")
        if all(
            float(request[dimension]) <= cli_target_value(class_targets[dimension])
            for dimension in ("ttft_ms", "itl_ms", "e2e_ms")
        ):
            good += 1
    return good / injected if injected else 0.0


def analyze_sensitivity(
    out: Path,
    evaluation_rows: list[dict[str, str]],
) -> dict[str, Any]:
    sensitivity_rows = load_rows(out / "sensitivity.csv")
    if not sensitivity_rows:
        raise SystemExit("sensitivity.csv is missing or empty")
    targets = load_targets(out)
    static = ("always", "never", "universal_phi", "universal_threshold")
    summaries: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    verdicts: dict[str, bool] = {}

    for factor in (0.8, 1.0, 1.2):
        factor_key = f"{factor:.1f}"
        condition_means: dict[tuple[str, str], dict[str, float]] = {}
        condition_seed_values: dict[
            tuple[str, str], dict[str, dict[int, float]]
        ] = {}
        for workload, label, _, _ in base_conditions(out):
            key = (workload.name, label)
            condition_means[key] = {}
            condition_seed_values[key] = {}
            scaled = scale_targets(targets[workload.name], factor)
            for policy in static:
                members = [
                    row
                    for row in evaluation_rows
                    if row["workload"] == workload.name
                    and row["rate_label"] == label
                    and row["policy"] == policy
                    and int(row["seed"]) in HELD_OUT_SEEDS
                ]
                values = {
                    int(row["seed"]): offline_goodput(row, scaled) for row in members
                }
                if factor == 1.0:
                    mismatches = [
                        int(row["seed"])
                        for row in members
                        if abs(
                            values[int(row["seed"])] - float(row["goodput"])
                        )
                        > 1e-12
                    ]
                    if mismatches:
                        raise RuntimeError(
                            "factor-1 offline rescore does not reproduce executed "
                            f"{workload.name}/{label}/{policy} seeds {mismatches}"
                        )
                condition_seed_values[key][policy] = values
                condition_means[key][policy] = statistics.fmean(values.values())

            if factor == 1.0:
                dp_members = [
                    row
                    for row in evaluation_rows
                    if row["workload"] == workload.name
                    and row["rate_label"] == label
                    and row["policy"] == "dpvar"
                    and int(row["seed"]) in HELD_OUT_SEEDS
                ]
            else:
                dp_members = [
                    row
                    for row in sensitivity_rows
                    if row["workload"] == workload.name
                    and row["rate_label"] == f"{label}_slo{int(factor * 100)}"
                    and row["policy"] == "dpvar"
                ]
            dp_values = {
                int(row["seed"]): float(row["goodput"]) for row in dp_members
            }
            condition_seed_values[key]["dpvar"] = dp_values
            condition_means[key]["dpvar"] = statistics.fmean(dp_values.values())

            for policy, mean in condition_means[key].items():
                summaries.append(
                    {
                        "factor": factor,
                        "workload": workload.name,
                        "rate_label": label,
                        "policy": policy,
                        "heldout_mean": mean,
                    }
                )

        references = {
            key: max(means.values()) for key, means in condition_means.items()
        }
        worst_regret = {
            policy: max(
                references[key] - means[policy]
                for key, means in condition_means.items()
            )
            for policy in (*static, "dpvar")
        }
        best_static_worst = min(worst_regret[policy] for policy in static)
        condition_paired = {}
        for key, means in condition_means.items():
            workload, label = key
            if workload not in {"synth", "rag"}:
                continue
            best_static = max(static, key=lambda policy: means[policy])
            differences = [
                condition_seed_values[key]["dpvar"][seed]
                - condition_seed_values[key][best_static][seed]
                for seed in HELD_OUT_SEEDS
            ]
            mean, low, high = ci95(differences)
            condition_paired[f"{workload}:{label}"] = {
                "static_policy": best_static,
                "mean_difference": mean,
                "ci_low": low,
                "ci_high": high,
            }
        criteria = {
            "dpvar_worst_regret_beats_universal_static_by_0.05": (
                worst_regret["dpvar"] <= best_static_worst - 0.05
            ),
            "dpvar_worst_regret_at_most_0.10": worst_regret["dpvar"] <= 0.10,
            "significant_variable_prompt_win": any(
                value["mean_difference"] > 0.02 and value["ci_low"] > 0.0
                for value in condition_paired.values()
            ),
        }
        verdicts[factor_key] = all(criteria.values())
        paired[factor_key] = {
            "passed": verdicts[factor_key],
            "criteria": criteria,
            "worst_regret": worst_regret,
            "conditions": condition_paired,
        }

    write_rows(out / "sensitivity_summary.csv", summaries)
    result = {
        "target_sensitive": len(set(verdicts.values())) > 1,
        "verdicts": verdicts,
        "factors": paired,
    }
    (out / "sensitivity.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# SLO sensitivity",
        "",
        "| target multiplier | dpVaR-vs-static verdict | dpVaR worst regret |",
        "|---:|---|---:|",
    ]
    for factor, record in paired.items():
        lines.append(
            f"| {factor} | {'pass' if record['passed'] else 'fail'} | "
            f"{record['worst_regret']['dpvar']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"**Target-sensitive conclusion:** "
            f"{'yes' if result['target_sensitive'] else 'no'}",
            "",
            "Static policies are rescored from their request-level traces. dpVaR is",
            "rerun because its placement decision depends on the SLO targets.",
            "",
        ]
    )
    (out / "SENSITIVITY.md").write_text("\n".join(lines))
    return result


def phase_analyze(out: Path) -> None:
    rows = load_rows(out / "evaluate.csv")
    if not rows:
        raise SystemExit("evaluate.csv is missing or empty")
    summary = aggregate_evaluation(rows)
    references: dict[tuple[str, str], float] = {}
    for row in summary:
        key = (row["workload"], row["rate_label"])
        references[key] = max(references.get(key, 0.0), row["heldout_mean"])
    for row in summary:
        row["regret"] = references[(row["workload"], row["rate_label"])] - row["heldout_mean"]
    write_rows(out / "evaluation_summary.csv", summary)

    worst_regret = {}
    for policy in sorted({row["policy"] for row in summary}):
        worst_regret[policy] = max(
            row["regret"] for row in summary if row["policy"] == policy
        )
    static = ("always", "never", "universal_phi", "universal_threshold")
    best_static_worst = min(worst_regret[policy] for policy in static)
    dpvar_worst = worst_regret["dpvar"]

    paired: dict[str, Any] = {}
    for workload in ("synth", "rag"):
        for label in ("low", "medium", "high"):
            condition = [
                row
                for row in rows
                if row["workload"] == workload
                and row["rate_label"] == label
                and int(row["seed"]) in HELD_OUT_SEEDS
            ]
            means = {}
            for policy in static:
                values = [float(row["goodput"]) for row in condition if row["policy"] == policy]
                means[policy] = statistics.fmean(values)
            best_static = max(means, key=means.get)
            differences = []
            for seed in HELD_OUT_SEEDS:
                dp = next(
                    float(row["goodput"])
                    for row in condition
                    if row["policy"] == "dpvar" and int(row["seed"]) == seed
                )
                st = next(
                    float(row["goodput"])
                    for row in condition
                    if row["policy"] == best_static and int(row["seed"]) == seed
                )
                differences.append(dp - st)
            mean, lo, hi = ci95(differences)
            paired[f"{workload}:{label}"] = {
                "static_policy": best_static,
                "mean_difference": mean,
                "ci_low": lo,
                "ci_high": hi,
            }

    criterion1 = dpvar_worst <= best_static_worst - 0.05
    criterion2 = dpvar_worst <= 0.10
    criterion3 = any(
        value["mean_difference"] > 0.02 and value["ci_low"] > 0.0
        for value in paired.values()
    )
    decision = {
        "passed": criterion1 and criterion2 and criterion3,
        "criteria": {
            "dpvar_worst_regret_beats_universal_static_by_0.05": criterion1,
            "dpvar_worst_regret_at_most_0.10": criterion2,
            "significant_variable_prompt_win": criterion3,
        },
        "dpvar_worst_regret": dpvar_worst,
        "best_universal_static_worst_regret": best_static_worst,
        "worst_regret": worst_regret,
        "paired_dpvar_minus_best_static": paired,
        "all_runs_valid": all(row["valid"].lower() == "true" for row in rows),
    }
    staged_rows = load_rows(out / "staged.csv")
    if staged_rows:
        staged_summary = aggregate_evaluation(staged_rows)
        write_rows(out / "staged_summary.csv", staged_summary)
        staged_means = {
            row["policy"]: row["heldout_mean"] for row in staged_summary
        }
        dpvar_by_seed = {
            int(row["seed"]): float(row["goodput"])
            for row in staged_rows
            if row["policy"] == "dpvar" and int(row["seed"]) in HELD_OUT_SEEDS
        }
        always_by_seed = {
            int(row["seed"]): float(row["goodput"])
            for row in staged_rows
            if row["policy"] == "always" and int(row["seed"]) in HELD_OUT_SEEDS
        }
        staged_difference = [
            dpvar_by_seed[seed] - always_by_seed[seed] for seed in HELD_OUT_SEEDS
        ]
        staged_mean, staged_low, staged_high = ci95(staged_difference)
        decision["staged_diagnostic"] = {
            "heldout_means": staged_means,
            "dpvar_minus_always": {
                "mean_difference": staged_mean,
                "ci_low": staged_low,
                "ci_high": staged_high,
            },
            "all_runs_valid": all(
                row["valid"].lower() == "true" for row in staged_rows
            ),
        }
    decision["slo_sensitivity"] = analyze_sensitivity(out, rows)
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    lines = [
        "# Decisive campaign result",
        "",
        f"**Decision: {'PASS' if decision['passed'] else 'FAIL'}**",
        "",
        "## Registered criteria",
        "",
    ]
    for name, value in decision["criteria"].items():
        lines.append(f"- `{name}`: **{'pass' if value else 'fail'}**")
    lines.extend(
        [
            "",
            "## Worst-case regret",
            "",
            "| policy | regret |",
            "|---|---:|",
        ]
    )
    for policy, regret in sorted(worst_regret.items(), key=lambda item: item[1]):
        lines.append(f"| `{policy}` | {regret:.3f} |")
    lines.extend(
        [
            "",
            "## Paired comparison against the best universal static policy",
            "",
            "| condition | static policy | dpVaR difference | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for condition, value in paired.items():
        lines.append(
            f"| `{condition}` | `{value['static_policy']}` | "
            f"{value['mean_difference']:.3f} | "
            f"[{value['ci_low']:.3f}, {value['ci_high']:.3f}] |"
        )
    if "staged_diagnostic" in decision:
        staged = decision["staged_diagnostic"]
        difference = staged["dpvar_minus_always"]
        lines.extend(
            [
                "",
                "## Staged-load diagnostic",
                "",
                "| policy | held-out mean goodput |",
                "|---|---:|",
                f"| `dpvar` | {staged['heldout_means']['dpvar']:.3f} |",
                f"| `always` | {staged['heldout_means']['always']:.3f} |",
                f"| `universal_phi` | "
                f"{staged['heldout_means']['universal_phi']:.3f} |",
                "",
                f"Paired dpVaR-minus-`always`: {difference['mean_difference']:.3f}, "
                f"95% CI [{difference['ci_low']:.3f}, "
                f"{difference['ci_high']:.3f}]. This diagnostic is inconclusive.",
            ]
        )
    lines.extend(
        [
            "",
            "The calibration seed is excluded from confidence intervals. Per-condition",
            "tuned shares and thresholds are offline yardsticks, not deployable baselines.",
            "",
        ]
    )
    (out / "DECISION.md").write_text("\n".join(lines))
    print(json.dumps(decision, indent=2))


def build_binary() -> None:
    env = dict(**__import__("os").environ)
    env.setdefault("GOCACHE", "/tmp/edpp-go-build")
    subprocess.run(
        ["go", "build", "-o", "blis", "main.go"],
        cwd=ROOT,
        check=True,
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "probe",
            "capacity",
            "targets",
            "calibrate",
            "evaluate",
            "sensitivity",
            "staged",
            "analyze",
            "all",
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_build and args.phase != "analyze":
        build_binary()

    phases = (
        ("probe", phase_probe),
        ("capacity", phase_capacity),
        ("targets", phase_targets),
        ("calibrate", phase_calibrate),
        ("evaluate", phase_evaluate),
        ("sensitivity", phase_sensitivity),
        ("staged", phase_staged),
    )
    if args.phase == "all":
        for _, function in phases:
            function(out, args.jobs, args.force)
        phase_analyze(out)
    elif args.phase == "analyze":
        phase_analyze(out)
    else:
        function = dict(phases)[args.phase]
        function(out, args.jobs, args.force)


if __name__ == "__main__":
    main()
