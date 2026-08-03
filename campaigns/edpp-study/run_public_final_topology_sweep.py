#!/usr/bin/env python3
"""Topology/provisioning sweep for the final causal-externality policy."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import run_decisive_campaign as base
import run_public_load_static_benchmark as benchmark
import run_public_workload_heterogeneity_closeout as prior
import run_slo_externality_joint_campaign as diagnostic


DEFAULT_OUT = base.CAMPAIGN / "out" / "public_final_topology_sweep_ttft_rollout_v2"
PROTOCOL = base.CAMPAIGN / "PUBLIC-FINAL-TOPOLOGY-SWEEP-PROTOCOL.md"
TOPOLOGIES = {
    "1p3d": (1, 3),
    "2p2d": (2, 2),
    "3p1d": (3, 1),
}
PHIS = (0.0, 0.25, 0.5, 0.75, 1.0)
CAPACITY_SEEDS = (42, 123, 2024)
STATIC_SEEDS = (42, 123)
CONFIRMATION_SEEDS = (2000000181, 2000000193, 2000000207, 2000000227)
LOAD_FACTOR = 0.90
V = 8.0
FOCAL = benchmark.FOCAL
ADAPTED_KAIROS = "kairos_beta_0p5"
PAPER_KAIROS = "kairos_paper_alpha_1p3"
LLMD_PREFIX = prior.LLMD_PREFIX_POLICY
DEPLOYABLE = (FOCAL, "least_ttft_joint", LLMD_PREFIX)
POLICIES = (*DEPLOYABLE, "static_joint_yardstick", "capacity_static")


@dataclass(frozen=True)
class Run:
    phase: str
    topology: str
    workload: str
    rate: float
    seed: int
    policy: str
    requests: int
    phi: float | None = None

    def tag(self) -> str:
        phi = "none" if self.phi is None else str(self.phi).replace(".", "p")
        return (
            f"{self.topology}_{self.workload}_{self.rate:.7g}_{self.policy}_"
            f"phi{phi}_s{self.seed}_n{self.requests}"
        )


def config(name: str) -> prior.WorkloadConfig:
    return prior.WORKLOADS[name]


def loose_targets(name: str) -> dict[str, dict[str, float]]:
    return prior.loose_targets(config(name))


def trace_path(out: Path, run: Run) -> Path:
    return out / run.phase / "candidate_traces" / f"{run.tag()}.csv"


def plan_path(out: Path, run: Run) -> Path:
    return out / run.phase / "plans" / f"{run.tag()}.csv"


def hard_valid(row: dict[str, Any]) -> bool:
    return (
        int(row["completed"]) + int(row["dropped"]) == int(row["injected"])
        and int(row["still_queued"]) == 0
        and int(row["still_running"]) == 0
        and int(row["timed_out"]) == 0
        and int(row["length_capped"]) == 0
    )


def policy_args(run: Run, targets: dict[str, dict[str, float]], plan: Path | None) -> list[str]:
    if run.policy in {"capacity_static", "static_joint_yardstick"}:
        if plan is None:
            raise ValueError(f"{run.policy} needs a plan")
        return ["--pd-plan", str(plan)]
    if run.policy == FOCAL:
        return base.policy_args("joint_slo_externality_no_capacity", targets, V, None)
    if run.policy == "least_ttft_joint":
        return base.policy_args("least_ttft_joint", targets, None, None)
    if run.policy == ADAPTED_KAIROS:
        return base.policy_args("kairos", targets, 0.5, None)
    if run.policy == PAPER_KAIROS:
        return base.policy_args("kairos-paper", targets, None, None)
    if run.policy == LLMD_PREFIX:
        return [
            "--pd-decider", "prefix-threshold",
            "--pd-prefix-threshold", str(prior.LLMD_THRESHOLDS[run.workload]),
        ]
    raise ValueError(f"unknown policy {run.policy}")


def topology_args(run: Run) -> list[str]:
    args = base.topology_args(run.topology.replace("d", "m"))
    if run.policy != LLMD_PREFIX:
        return args
    scorer_index = args.index("--decode-routing-scorers") + 1
    args[scorer_index] = prior.LLMD_SCORERS
    args += [
        "--prefill-routing-scorers", prior.LLMD_SCORERS,
        "--cache-signal-delay", "50000",
    ]
    return args


def execute_run(run: Run, out: Path, force: bool) -> dict[str, Any]:
    run_dir = out / run.phase
    spec = run_dir / "specs" / f"{run.tag()}.yaml"
    metrics = run_dir / "metrics" / f"{run.tag()}.json"
    stdout = run_dir / "stdout" / f"{run.tag()}.txt"
    cfg = config(run.workload)
    base.make_spec(cfg.workload, run.rate, run.seed, run.requests, spec, None)

    plan: Path | None = None
    if run.policy in {"capacity_static", "static_joint_yardstick"}:
        if run.phi is None:
            raise ValueError("static run missing phi")
        plan = plan_path(out, run)
        prefill, decode = TOPOLOGIES[run.topology]
        base.make_plan(
            plan,
            run.requests,
            run.phi,
            prefill_instances=prefill,
            decode_instances=decode,
        )

    metrics.parent.mkdir(parents=True, exist_ok=True)
    stdout.parent.mkdir(parents=True, exist_ok=True)
    targets = loose_targets(run.workload) if run.phase == "capacity" else cfg.targets
    if force or not (metrics.exists() and stdout.exists()):
        command = [
            str(base.ROOT / "blis"),
            "run",
            "--model",
            base.MODEL,
            "--workload-spec",
            str(spec),
            "--num-requests",
            str(run.requests),
            *topology_args(run),
            *base.slo_args(targets),
            *policy_args(run, targets, plan),
            "--timeout",
            "-1",
            "--seed",
            str(run.seed),
            "--metrics-path",
            str(metrics),
        ]
        if run.policy == FOCAL:
            candidate = trace_path(out, run)
            candidate.parent.mkdir(parents=True, exist_ok=True)
            command += ["--edpp-joint-candidate-trace", str(candidate)]
        proc = subprocess.run(command, cwd=base.ROOT, capture_output=True, text=True)
        stdout.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                f"run failed ({run.tag()}):\n{proc.stderr[-5000:]}\n{proc.stdout[-5000:]}"
            )

    row = base.result_row(
        phase=run.phase,
        workload=run.workload,
        rate_label="saturated" if run.phase == "capacity" else "stress_0p90_capacity",
        rate=run.rate,
        seed=run.seed,
        policy=run.policy,
        parameter=run.phi,
        topology=run.topology,
        metrics_path=metrics,
        stdout_path=stdout,
    )
    row["topology"] = run.topology
    row["phi"] = "" if run.phi is None else run.phi
    row["hard_valid"] = hard_valid(row)
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
                f"completion={float(row['central_completion_rps']):.3f} valid={row['hard_valid']}",
                flush=True,
            )
    return sorted(
        rows,
        key=lambda row: (
            row["topology"], row["workload"], row["policy"],
            float(row["phi"]) if row["phi"] != "" else -1.0,
            int(row["seed"]),
        ),
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_terminal(rows: list[dict[str, Any]], label: str) -> None:
    bad = [
        f"{row['topology']}/{row['workload']}/{row['policy']}/s{row['seed']}"
        for row in rows if not row["hard_valid"]
    ]
    if bad:
        raise RuntimeError(f"{label} hard-invalid runs: {', '.join(bad)}")


def choose(
    rows: list[dict[str, Any]], seeds: tuple[int, ...], field: str
) -> dict[str, Any]:
    items = []
    for phi in PHIS:
        members = [row for row in rows if float(row["phi"]) == phi]
        if len(members) != len(seeds) or {int(row["seed"]) for row in members} != set(seeds):
            raise RuntimeError(f"incomplete phi={phi}: got {len(members)}")
        values = [float(row[field]) for row in members]
        items.append(
            {
                "phi": phi,
                "mean": statistics.fmean(values),
                "stdev": statistics.stdev(values),
                "values": {str(int(row["seed"])): float(row[field]) for row in members},
                "eligible": all(row["hard_valid"] and int(row["dropped"]) == 0 for row in members),
            }
        )
    eligible = [item for item in items if item["eligible"]]
    if not eligible:
        raise RuntimeError("no eligible fixed plan")
    best = max(eligible, key=lambda item: (item["mean"], -item["stdev"], -item["phi"]))
    return {"best": best, "grid": items}


def capacity_runs() -> list[Run]:
    return [
        Run("capacity", topology, name, cfg.workload.saturation_rate, seed,
            "capacity_static", cfg.capacity_requests, phi)
        for topology in TOPOLOGIES
        for name, cfg in prior.WORKLOADS.items()
        for phi in PHIS
        for seed in CAPACITY_SEEDS
    ]


def run_capacity(out: Path, jobs: int, force: bool) -> None:
    runs = capacity_runs()
    if len(runs) != 135:
        raise RuntimeError(f"capacity has {len(runs)} runs, want 135")
    rows = run_many(runs, out, jobs, force)
    require_terminal(rows, "capacity")
    write_rows(out / "capacity.csv", rows)
    result: dict[str, Any] = {}
    for topology in TOPOLOGIES:
        result[topology] = {}
        for name, cfg in prior.WORKLOADS.items():
            members = [row for row in rows if row["topology"] == topology and row["workload"] == name]
            selected = choose(members, CAPACITY_SEEDS, "central_completion_rps")
            best = selected["best"]
            if cfg.workload.saturation_rate < 1.05 * float(best["mean"]):
                raise RuntimeError(f"unsaturated probe {topology}/{name}")
            result[topology][name] = {
                "capacity_rps": best["mean"],
                "capacity_stdev": best["stdev"],
                "selected_phi": best["phi"],
                "evaluation_rate": LOAD_FACTOR * best["mean"],
                "capacity_seeds": list(CAPACITY_SEEDS),
                "grid": selected["grid"],
            }
    (out / "capacity_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def load_json(out: Path, name: str) -> dict[str, Any]:
    path = out / name
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text())


def static_runs(capacity: dict[str, Any]) -> list[Run]:
    return [
        Run("static_calibration", topology, name,
            float(capacity[topology][name]["evaluation_rate"]), seed,
            "static_joint_yardstick", cfg.evaluation_requests, phi)
        for topology in TOPOLOGIES
        for name, cfg in prior.WORKLOADS.items()
        for phi in PHIS
        for seed in STATIC_SEEDS
    ]


def run_static(out: Path, jobs: int, force: bool) -> None:
    capacity = load_json(out, "capacity_selection.json")
    runs = static_runs(capacity)
    if len(runs) != 90:
        raise RuntimeError(f"static calibration has {len(runs)} runs, want 90")
    rows = run_many(runs, out, jobs, force)
    require_terminal(rows, "static calibration")
    write_rows(out / "static_calibration.csv", rows)
    result: dict[str, Any] = {}
    for topology in TOPOLOGIES:
        result[topology] = {}
        for name in prior.WORKLOADS:
            members = [row for row in rows if row["topology"] == topology and row["workload"] == name]
            selected = choose(members, STATIC_SEEDS, "goodput")
            best = selected["best"]
            result[topology][name] = {
                "selected_phi": best["phi"],
                "calibration_mean_goodput": best["mean"],
                "calibration_stdev_goodput": best["stdev"],
                "calibration_seeds": list(STATIC_SEEDS),
                "grid": selected["grid"],
            }
    (out / "static_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def confirmation_runs(capacity: dict[str, Any], static: dict[str, Any]) -> list[Run]:
    runs = []
    for topology in TOPOLOGIES:
        for name, cfg in prior.WORKLOADS.items():
            rate = float(capacity[topology][name]["evaluation_rate"])
            for seed in CONFIRMATION_SEEDS:
                for policy in POLICIES:
                    phi = None
                    if policy == "static_joint_yardstick":
                        phi = static[topology][name]["selected_phi"]
                    elif policy == "capacity_static":
                        phi = capacity[topology][name]["selected_phi"]
                    runs.append(Run("confirmation", topology, name, rate, seed, policy, cfg.evaluation_requests, phi))
    return runs


def analyze(out: Path, runs: list[Run], rows: list[dict[str, Any]]) -> dict[str, Any]:
    means: dict[str, dict[str, float]] = {}
    remote: dict[str, dict[str, float]] = {}
    per_seed: dict[str, dict[str, list[float]]] = {}
    for topology in TOPOLOGIES:
        for name in prior.WORKLOADS:
            cell = f"{topology}:{name}"
            means[cell], remote[cell], per_seed[cell] = {}, {}, {}
            for policy in POLICIES:
                members = [
                    row for row in rows
                    if row["topology"] == topology and row["workload"] == name and row["policy"] == policy
                ]
                members.sort(key=lambda row: int(row["seed"]))
                if len(members) != len(CONFIRMATION_SEEDS):
                    raise RuntimeError(f"incomplete confirmation {cell}/{policy}")
                values = [float(row["goodput"]) for row in members]
                means[cell][policy] = statistics.fmean(values)
                remote[cell][policy] = statistics.fmean(float(row["realized_phi"]) for row in members)
                per_seed[cell][policy] = values

    best = {cell: max(means[cell][policy] for policy in DEPLOYABLE) for cell in means}
    ranking = []
    for policy in DEPLOYABLE:
        regrets = {cell: best[cell] - means[cell][policy] for cell in means}
        ranking.append({
            "policy": policy,
            "worst_regret": max(regrets.values()),
            "worst_cell": max(regrets, key=regrets.get),
            "mean_goodput": statistics.fmean(means[cell][policy] for cell in means),
            "regret_by_cell": regrets,
        })
    ranking.sort(key=lambda item: (item["worst_regret"], -item["mean_goodput"], item["policy"]))

    by_topology: dict[str, list[dict[str, Any]]] = {}
    for topology in TOPOLOGIES:
        cells = [cell for cell in means if cell.startswith(f"{topology}:")]
        items = []
        for policy in DEPLOYABLE:
            regrets = {cell: best[cell] - means[cell][policy] for cell in cells}
            items.append({"policy": policy, "worst_regret": max(regrets.values()), "worst_cell": max(regrets, key=regrets.get)})
        by_topology[topology] = sorted(items, key=lambda item: (item["worst_regret"], item["policy"]))

    paired: dict[str, Any] = {}
    for comparator in POLICIES:
        if comparator == FOCAL:
            continue
        deltas = [
            left - right
            for cell in means
            for left, right in zip(per_seed[cell][FOCAL], per_seed[cell][comparator])
        ]
        mean, low, high = base.ci95(deltas)
        paired[comparator] = {"mean_delta": mean, "ci_low": low, "ci_high": high}

    trace_exact = True
    trace_runs = 0
    for run in runs:
        if run.policy == FOCAL:
            summary = diagnostic.analyze_candidate_trace(trace_path(out, run), V)
            trace_runs += 1
            trace_exact = trace_exact and summary["positive_chosen_snapshot_score_regret_fraction"] == 0
    return {
        "status": "held-out final-policy topology sweep; no confirmation tuning",
        "run_count": len(runs),
        "cell_count": len(means),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "policy_means": means,
        "remote_means": remote,
        "per_seed_goodput": per_seed,
        "deployable_minimax_ranking": ranking,
        "deployable_ranking_by_topology": by_topology,
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
    capacity: dict[str, Any],
    static: dict[str, Any],
    result: dict[str, Any],
    filename: str = "PUBLIC-FINAL-TOPOLOGY-SWEEP.md",
) -> None:
    lines = [
        "# Final-policy public topology sweep", "",
        "The routing value is smooth TTFT x E2E; reported goodput remains the hard TTFT/mean-ITL/E2E conjunction.", "",
        "Kairos is excluded because its published policy does not define multi-prefill routing.", "",
        "## Capacity and frozen static plans", "",
        "| topology/workload | capacity rps | capacity phi | rate (0.90C) | tuned phi | calibration goodput |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for topology in TOPOLOGIES:
        for name in prior.WORKLOADS:
            cap, tuned = capacity[topology][name], static[topology][name]
            lines.append(
                f"| `{topology}:{name}` | {cap['capacity_rps']:.4f} | {cap['selected_phi']:.2f} | "
                f"{cap['evaluation_rate']:.4f} | {tuned['selected_phi']:.2f} | {tuned['calibration_mean_goodput']:.3f} |"
            )
    lines += ["", "## Held-out mean goodput", "",
              "| cell | causal externality | least TTFT | llm-d threshold | tuned static | capacity static | focal remote |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for cell, means in result["policy_means"].items():
        lines.append(
            f"| `{cell}` | {means[FOCAL]:.3f} | {means['least_ttft_joint']:.3f} | "
            f"{means[LLMD_PREFIX]:.3f} | {means['static_joint_yardstick']:.3f} | "
            f"{means['capacity_static']:.3f} | "
            f"{result['remote_means'][cell][FOCAL]:.3f} |"
        )
    lines += ["", "## Deployable minimax ranking", "",
              "| rank | policy | worst regret | worst cell | equal-cell mean |",
              "|---:|---|---:|---|---:|"]
    for rank, item in enumerate(result["deployable_minimax_ranking"], 1):
        lines.append(
            f"| {rank} | `{item['policy']}` | {item['worst_regret']:.4f} | "
            f"`{item['worst_cell']}` | {item['mean_goodput']:.4f} |"
        )
    lines += ["", "## Worst regret within each topology", "",
              "| topology | causal externality | least TTFT | llm-d threshold |",
              "|---|---:|---:|---:|"]
    for topology, items in result["deployable_ranking_by_topology"].items():
        values = {item["policy"]: item["worst_regret"] for item in items}
        lines.append(
            f"| `{topology}` | {values[FOCAL]:.4f} | {values['least_ttft_joint']:.4f} | "
            f"{values[LLMD_PREFIX]:.4f} |"
        )
    lines += ["", "## Focal paired deltas", "",
              "| comparator | mean delta | 95% interval |",
              "|---|---:|---:|"]
    for policy, item in result["focal_paired_deltas"].items():
        lines.append(
            f"| `{policy}` | {item['mean_delta']:+.4f} | "
            f"[{item['ci_low']:+.4f}, {item['ci_high']:+.4f}] |"
        )
    lines += ["", "## Validity", "",
              f"- Runs: {result['run_count']} across {result['cell_count']} cells.",
              f"- Hard-invalid runs: {result['hard_invalid_runs']}.",
              f"- Runs with drops: {result['runs_with_drops']}.",
              f"- Runs with timeouts: {result['runs_with_timeouts']}.",
              f"- Runs with length caps: {result['runs_with_length_caps']}.",
              f"- Focal candidate traces: {result['focal_candidate_trace_runs']}; "
              + ("all chosen actions are exact argmins." if result["chosen_argmin_trace_exact"] else "ARGMIN CHECK FAILED."),
              "", "The static comparator is a coarse condition-tuned yardstick, not an oracle.", ""]
    (out / filename).write_text("\n".join(lines))


def run_confirmation(out: Path, jobs: int, force: bool) -> None:
    capacity = load_json(out, "capacity_selection.json")
    static = load_json(out, "static_selection.json")
    runs = confirmation_runs(capacity, static)
    if len(runs) != 180:
        raise RuntimeError(f"confirmation has {len(runs)} runs, want 180")
    rows = run_many(runs, out, jobs, force)
    require_terminal(rows, "confirmation")
    write_rows(out / "confirmation.csv", rows)
    result = analyze(out, runs, rows)
    (out / "confirmation_result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(out, capacity, static, result)
    print(json.dumps(result, indent=2), flush=True)


def smoke(out: Path, force: bool) -> None:
    runs = [
        Run(
            "smoke", topology, "interactive", 2.0, 314159, policy, 24,
            0.5 if policy in {"static_joint_yardstick", "capacity_static"} else None,
        )
        for topology in TOPOLOGIES
        for policy in POLICIES
    ]
    rows = run_many(runs, out, jobs=len(runs), force=force)
    require_terminal(rows, "smoke")
    for run in runs:
        if run.policy == FOCAL:
            summary = diagnostic.analyze_candidate_trace(trace_path(out, run), V)
            if summary["positive_chosen_snapshot_score_regret_fraction"] != 0:
                raise RuntimeError(f"smoke argmin mismatch: {run.topology}")
    write_rows(out / "smoke.csv", rows)
    print("smoke passed: all policies and provisionings, terminal accounting, and exact argmins")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage", choices=("smoke", "capacity", "static", "confirm", "all"), default="all")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not PROTOCOL.exists():
        raise SystemExit(f"missing frozen protocol: {PROTOCOL}")
    if args.stage == "smoke":
        smoke(out, args.force)
        return
    if args.stage in {"capacity", "all"}:
        run_capacity(out, args.jobs, args.force)
    if args.stage in {"static", "all"}:
        run_static(out, args.jobs, args.force)
    if args.stage in {"confirm", "all"}:
        run_confirmation(out, args.jobs, args.force)


if __name__ == "__main__":
    main()
