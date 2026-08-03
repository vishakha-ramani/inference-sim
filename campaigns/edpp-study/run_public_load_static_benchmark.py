#!/usr/bin/env python3
"""Capacity/load/static-plan benchmark for the revised causal-SLO policy.

Protocol: PUBLIC-LOAD-STATIC-BENCHMARK-PROTOCOL.md.

The runner is resumable. Development stages never instantiate a confirmation
seed, and confirmation requires frozen capacity/static selection JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import run_decisive_campaign as base
import run_public_workload_heterogeneity_closeout as prior
import run_slo_externality_joint_campaign as diagnostic


DEFAULT_OUT = (
    base.CAMPAIGN / "out" / "public_load_static_benchmark_ttft_rollout_v2"
)
PROTOCOL = base.CAMPAIGN / "PUBLIC-LOAD-STATIC-BENCHMARK-PROTOCOL.md"

CAPACITY_SEEDS = (42, 123, 2024)
STATIC_SEEDS = (42, 123)
CONFIRMATION_SEEDS = (2000000011, 2000000033, 2000000063, 2000000087)

COARSE_SHARES = tuple(index / 5 for index in range(6))
LOADS = (
    ("low_0p60", 0.60),
    ("medium_0p80", 0.80),
    ("high_0p95", 0.95),
)

FOCAL = "causal_externality_no_capacity_v8"
PAPER_KAIROS = "kairos_paper_alpha_1p3"
LLMD_PREFIX = prior.LLMD_PREFIX_POLICY
DEPLOYABLE = (FOCAL, "least_ttft_joint", PAPER_KAIROS, LLMD_PREFIX)
CONFIRM_POLICIES = (
    *DEPLOYABLE,
    "static_joint_yardstick",
    "capacity_static",
)
DISPLAY = {
    FOCAL: "causal externality",
    "least_ttft_joint": "least TTFT",
    PAPER_KAIROS: "Kairos (paper, alpha=1.3)",
    LLMD_PREFIX: "workload-tuned llm-d threshold",
    "static_joint_yardstick": "goodput-tuned static",
    "capacity_static": "capacity-selected static",
}

# The reasoning target is 802 seconds. A generic 300-second request timeout
# would silently become a stricter outcome criterion at higher loads.
prior.REQUEST_TIMEOUT_SECS = -1


def plan_grid(hardware: str) -> list[tuple[float, float]]:
    psis = (0.5,) if hardware == "h100_homogeneous" else COARSE_SHARES
    return [(phi, psi) for phi in COARSE_SHARES for psi in psis]


def neighbors(value: float) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                round(max(0.0, min(1.0, value + delta)), 10)
                for delta in (-0.1, 0.0, 0.1)
            }
        )
    )


def refinement_grid(
    hardware: str, phi: float, psi: float
) -> list[tuple[float, float]]:
    phis = neighbors(phi)
    psis = (0.5,) if hardware == "h100_homogeneous" else neighbors(psi)
    coarse = set(plan_grid(hardware))
    return sorted(
        (candidate_phi, candidate_psi)
        for candidate_phi in phis
        for candidate_psi in psis
        if (candidate_phi, candidate_psi) not in coarse
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing required stage artifact: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def stdev_or_zero(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def grouped_plan_stats(
    rows: Iterable[dict[str, Any]],
    expected_seeds: tuple[int, ...],
    value_field: str,
    include_load: bool,
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key_parts = [
            str(row["hardware"]),
            str(row["workload"]),
        ]
        if include_load:
            key_parts.append(str(row["rate_label"]))
        key_parts.extend(
            [
                f"{float(row['phi']):.10f}",
                f"{float(row['psi_a100']):.10f}",
            ]
        )
        groups[tuple(key_parts)].append(row)

    by_cell: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    wanted = set(expected_seeds)
    for key, members in groups.items():
        seeds = {int(row["seed"]) for row in members}
        if seeds != wanted or len(members) != len(expected_seeds):
            raise RuntimeError(
                f"incomplete fixed-plan point {key}: seeds={sorted(seeds)}, "
                f"want={sorted(wanted)}"
            )
        values = [float(row[value_field]) for row in members]
        item = {
            "hardware": key[0],
            "workload": key[1],
            "rate_label": key[2] if include_load else None,
            "phi": float(key[-2]),
            "psi_a100": float(key[-1]),
            "mean": statistics.fmean(values),
            "stdev": stdev_or_zero(values),
            "values": {
                str(int(row["seed"])): float(row[value_field])
                for row in sorted(members, key=lambda value: int(value["seed"]))
            },
            "eligible": all(
                as_bool(row["hard_valid"]) and int(row["dropped"]) == 0
                for row in members
            ),
            "drops": sum(int(row["dropped"]) for row in members),
        }
        cell = key[:3] if include_load else key[:2]
        by_cell[cell].append(item)
    return by_cell


def choose_plan(items: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in items if item["eligible"]]
    if not eligible:
        raise RuntimeError("fixed-plan cell has no hard-valid zero-drop point")
    return max(
        eligible,
        key=lambda item: (
            float(item["mean"]),
            -float(item["stdev"]),
            -float(item["phi"]),
            -float(item["psi_a100"]),
        ),
    )


def capacity_runs(
    grids: dict[tuple[str, str], list[tuple[float, float]]]
) -> list[prior.Run]:
    runs: list[prior.Run] = []
    for hardware in prior.HARDWARES:
        for name, config in prior.WORKLOADS.items():
            for phi, psi in grids[(hardware, name)]:
                for seed in CAPACITY_SEEDS:
                    runs.append(
                        prior.Run(
                            "public_closeout_capacity",
                            config,
                            "saturated",
                            config.workload.saturation_rate,
                            seed,
                            "capacity_static",
                            hardware,
                            config.capacity_requests,
                            phi,
                            psi,
                        )
                    )
    return runs


def capacity_selection(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    grouped = grouped_plan_stats(
        rows, CAPACITY_SEEDS, "central_completion_rps", include_load=False
    )
    result: dict[str, Any] = {}
    winners: dict[tuple[str, str], dict[str, Any]] = {}
    for hardware in prior.HARDWARES:
        result[hardware] = {}
        for name, config in prior.WORKLOADS.items():
            cell = (hardware, name)
            items = sorted(
                grouped[cell],
                key=lambda item: (item["phi"], item["psi_a100"]),
            )
            best = choose_plan(items)
            winners[cell] = best
            offered = config.workload.saturation_rate
            if offered < 1.05 * float(best["mean"]):
                raise RuntimeError(
                    f"capacity probe not saturated for {hardware}/{name}: "
                    f"offered={offered}, observed={best['mean']}"
                )
            result[hardware][name] = {
                "offered_saturation_rate": offered,
                "capacity_rps": best["mean"],
                "capacity_stdev": best["stdev"],
                "selected_phi": best["phi"],
                "selected_psi_a100": best["psi_a100"],
                "capacity_seeds": list(CAPACITY_SEEDS),
                "evaluation_rates": {
                    label: factor * float(best["mean"])
                    for label, factor in LOADS
                },
                "grid": items,
            }
    return result, winners


def run_capacity(out: Path, jobs: int, force: bool) -> None:
    coarse_grids = {
        (hardware, name): plan_grid(hardware)
        for hardware in prior.HARDWARES
        for name in prior.WORKLOADS
    }
    coarse = prior.run_many(capacity_runs(coarse_grids), out, jobs, force)
    prior.require_hard_valid(coarse, "coarse capacity sweep")
    prior.write_rows(out / "capacity_coarse.csv", coarse)

    _, coarse_winners = capacity_selection(coarse)
    refine_grids = {
        cell: refinement_grid(
            cell[0], float(winner["phi"]), float(winner["psi_a100"])
        )
        for cell, winner in coarse_winners.items()
    }
    refine_runs = capacity_runs(refine_grids)
    refined = prior.run_many(refine_runs, out, jobs, force) if refine_runs else []
    prior.require_hard_valid(refined, "refined capacity sweep")
    prior.write_rows(out / "capacity_refine.csv", refined)

    all_rows = coarse + refined
    prior.write_rows(out / "capacity.csv", all_rows)
    selection, _ = capacity_selection(all_rows)
    (out / "capacity_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(json.dumps(selection, indent=2), flush=True)


def load_capacity(out: Path) -> dict[str, Any]:
    path = out / "capacity_selection.json"
    if not path.exists():
        raise SystemExit("capacity selection missing; run --stage capacity")
    return json.loads(path.read_text())


def static_runs(
    capacity: dict[str, Any],
    grids: dict[tuple[str, str, str], list[tuple[float, float]]],
) -> list[prior.Run]:
    runs: list[prior.Run] = []
    for hardware in prior.HARDWARES:
        for name, config in prior.WORKLOADS.items():
            for label, _ in LOADS:
                rate = float(capacity[hardware][name]["evaluation_rates"][label])
                for phi, psi in grids[(hardware, name, label)]:
                    for seed in STATIC_SEEDS:
                        runs.append(
                            prior.Run(
                                "public_static_calibration",
                                config,
                                label,
                                rate,
                                seed,
                                "capacity_static",
                                hardware,
                                config.evaluation_requests,
                                phi,
                                psi,
                            )
                        )
    return runs


def static_selection(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]]:
    grouped = grouped_plan_stats(
        rows, STATIC_SEEDS, "goodput", include_load=True
    )
    result: dict[str, Any] = {}
    winners: dict[tuple[str, str, str], dict[str, Any]] = {}
    for hardware in prior.HARDWARES:
        result[hardware] = {}
        for name in prior.WORKLOADS:
            result[hardware][name] = {}
            for label, _ in LOADS:
                cell = (hardware, name, label)
                items = sorted(
                    grouped[cell],
                    key=lambda item: (item["phi"], item["psi_a100"]),
                )
                best = choose_plan(items)
                winners[cell] = best
                result[hardware][name][label] = {
                    "selected_phi": best["phi"],
                    "selected_psi_a100": best["psi_a100"],
                    "calibration_mean_goodput": best["mean"],
                    "calibration_stdev": best["stdev"],
                    "calibration_seeds": list(STATIC_SEEDS),
                    "grid": items,
                }
    return result, winners


def run_static(out: Path, jobs: int, force: bool) -> None:
    capacity = load_capacity(out)
    coarse_grids = {
        (hardware, name, label): plan_grid(hardware)
        for hardware in prior.HARDWARES
        for name in prior.WORKLOADS
        for label, _ in LOADS
    }
    coarse = prior.run_many(
        static_runs(capacity, coarse_grids), out, jobs, force
    )
    prior.require_hard_valid(coarse, "coarse static-goodput calibration")
    prior.write_rows(out / "static_goodput_coarse.csv", coarse)

    _, coarse_winners = static_selection(coarse)
    refine_grids = {
        cell: refinement_grid(
            cell[0], float(winner["phi"]), float(winner["psi_a100"])
        )
        for cell, winner in coarse_winners.items()
    }
    refine_runs = static_runs(capacity, refine_grids)
    refined = prior.run_many(refine_runs, out, jobs, force) if refine_runs else []
    prior.require_hard_valid(refined, "refined static-goodput calibration")
    prior.write_rows(out / "static_goodput_refine.csv", refined)

    all_rows = coarse + refined
    prior.write_rows(out / "static_goodput_calibration.csv", all_rows)
    selection, _ = static_selection(all_rows)
    (out / "static_goodput_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(json.dumps(selection, indent=2), flush=True)


def load_static(out: Path) -> dict[str, Any]:
    path = out / "static_goodput_selection.json"
    if not path.exists():
        raise SystemExit("static selection missing; run --stage static")
    return json.loads(path.read_text())


def confirmation_runs(
    capacity: dict[str, Any], static: dict[str, Any]
) -> list[prior.Run]:
    runs: list[prior.Run] = []
    for hardware in prior.HARDWARES:
        for name, config in prior.WORKLOADS.items():
            cap = capacity[hardware][name]
            for label, _ in LOADS:
                rate = float(cap["evaluation_rates"][label])
                tuned = static[hardware][name][label]
                for seed in CONFIRMATION_SEEDS:
                    for policy in CONFIRM_POLICIES:
                        phi = None
                        psi = None
                        if policy == "static_joint_yardstick":
                            phi = float(tuned["selected_phi"])
                            psi = float(tuned["selected_psi_a100"])
                        elif policy == "capacity_static":
                            phi = float(cap["selected_phi"])
                            psi = float(cap["selected_psi_a100"])
                        runs.append(
                            prior.Run(
                                "public_load_static_confirmation",
                                config,
                                label,
                                rate,
                                seed,
                                policy,
                                hardware,
                                config.evaluation_requests,
                                phi,
                                psi,
                            )
                        )
    return runs


def confirmation_analysis(
    out: Path,
    runs: list[prior.Run],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cells = [
        (hardware, name, label)
        for hardware in prior.HARDWARES
        for name in prior.WORKLOADS
        for label, _ in LOADS
    ]
    per_seed: dict[str, dict[str, list[dict[str, Any]]]] = {}
    means: dict[str, dict[str, float]] = {}
    remote: dict[str, dict[str, float]] = {}
    dimensions: dict[str, dict[str, dict[str, float]]] = {}
    for hardware, name, label in cells:
        cell = f"{hardware}:{name}:{label}"
        per_seed[cell] = {}
        means[cell] = {}
        remote[cell] = {}
        dimensions[cell] = {}
        for policy in CONFIRM_POLICIES:
            members = [
                row
                for row in rows
                if row["hardware"] == hardware
                and row["workload"] == name
                and row["rate_label"] == label
                and row["policy"] == policy
            ]
            members.sort(key=lambda row: int(row["seed"]))
            if len(members) != len(CONFIRMATION_SEEDS):
                raise RuntimeError(
                    f"incomplete confirmation for {cell}/{policy}: "
                    f"{len(members)}"
                )
            means[cell][policy] = statistics.fmean(
                float(row["goodput"]) for row in members
            )
            remote[cell][policy] = statistics.fmean(
                float(row["realized_phi"]) for row in members
            )
            dimensions[cell][policy] = {
                dimension: statistics.fmean(
                    float(row[f"good_{dimension}"]) for row in members
                )
                for dimension in ("ttft", "itl", "e2e")
            }
            per_seed[cell][policy] = [
                {
                    "seed": int(row["seed"]),
                    "goodput": float(row["goodput"]),
                    "good_ttft": float(row["good_ttft"]),
                    "good_itl": float(row["good_itl"]),
                    "good_e2e": float(row["good_e2e"]),
                    "remote_fraction": float(row["realized_phi"]),
                    "dropped": int(row["dropped"]),
                    "timed_out": int(row["timed_out"]),
                    "length_capped": int(row["length_capped"]),
                    "still_queued": int(row["still_queued"]),
                    "still_running": int(row["still_running"]),
                }
                for row in members
            ]

    best_deployable = {
        cell: max(means[cell][policy] for policy in DEPLOYABLE)
        for cell in means
    }
    ranking = []
    for policy in DEPLOYABLE:
        regrets = {
            cell: best_deployable[cell] - means[cell][policy]
            for cell in means
        }
        ranking.append(
            {
                "policy": policy,
                "worst_regret": max(regrets.values()),
                "worst_cell": max(regrets, key=regrets.get),
                "mean_goodput": statistics.fmean(
                    means[cell][policy] for cell in means
                ),
                "regret_by_cell": regrets,
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
    for comparator in CONFIRM_POLICIES:
        if comparator == FOCAL:
            continue
        all_deltas: list[float] = []
        by_cell = []
        for cell in means:
            left = {
                item["seed"]: item["goodput"]
                for item in per_seed[cell][FOCAL]
            }
            right = {
                item["seed"]: item["goodput"]
                for item in per_seed[cell][comparator]
            }
            deltas = [
                left[seed] - right[seed] for seed in CONFIRMATION_SEEDS
            ]
            mean, low, high = base.ci95(deltas)
            by_cell.append(
                {
                    "cell": cell,
                    "mean_delta": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "wins": sum(delta > 0 for delta in deltas),
                    "ties": sum(delta == 0 for delta in deltas),
                    "losses": sum(delta < 0 for delta in deltas),
                }
            )
            all_deltas.extend(deltas)
        mean, low, high = base.ci95(all_deltas)
        paired[comparator] = {
            "all_cells": {
                "mean_delta": mean,
                "ci_low": low,
                "ci_high": high,
            },
            "by_cell": by_cell,
        }

    trace_exact = True
    trace_runs = 0
    for run in runs:
        if run.policy != FOCAL:
            continue
        summary = diagnostic.analyze_candidate_trace(
            prior.trace_path(out, run), prior.V
        )
        trace_runs += 1
        trace_exact = trace_exact and (
            summary["positive_chosen_snapshot_score_regret_fraction"] == 0
        )

    return {
        "status": "held-out load/static-plan confirmation; no confirmation tuning",
        "routing_value": "smooth TTFT x E2E; reported goodput remains TTFT x ITL x E2E",
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "run_count": len(runs),
        "cell_count": len(cells),
        "policy_means": means,
        "remote_fractions": remote,
        "dimension_attainment": dimensions,
        "per_seed": per_seed,
        "deployable_minimax_ranking": ranking,
        "focal_paired_deltas": paired,
        "hard_invalid_runs": sum(not as_bool(row["hard_valid"]) for row in rows),
        "runs_with_drops": sum(int(row["dropped"]) > 0 for row in rows),
        "runs_with_timeouts": sum(int(row["timed_out"]) > 0 for row in rows),
        "runs_with_length_caps": sum(
            int(row["length_capped"]) > 0 for row in rows
        ),
        "focal_candidate_trace_runs": trace_runs,
        "chosen_argmin_trace_exact": trace_exact,
    }


def write_report(
    out: Path,
    capacity: dict[str, Any],
    static: dict[str, Any],
    result: dict[str, Any],
) -> None:
    lines = [
        "# Public load/static-plan benchmark",
        "",
        "The routing value is smooth TTFT x E2E; reported goodput remains",
        "the hard TTFT/mean-ITL/E2E conjunction.",
        "",
        "## Frozen capacity and static plans",
        "",
        "| fleet/workload | capacity rps | cap phi | cap psi | load | rate | goodput phi | goodput psi | calibration goodput |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for hardware in prior.HARDWARES:
        for name in prior.WORKLOADS:
            cap = capacity[hardware][name]
            for label, _ in LOADS:
                tuned = static[hardware][name][label]
                lines.append(
                    f"| `{hardware}:{name}` | {cap['capacity_rps']:.4f} | "
                    f"{cap['selected_phi']:.2f} | "
                    f"{cap['selected_psi_a100']:.2f} | `{label}` | "
                    f"{cap['evaluation_rates'][label]:.4f} | "
                    f"{tuned['selected_phi']:.2f} | "
                    f"{tuned['selected_psi_a100']:.2f} | "
                    f"{tuned['calibration_mean_goodput']:.3f} |"
                )

    lines.extend(
        [
            "",
            "## Held-out mean goodput",
            "",
            "| cell | causal externality | least TTFT | Kairos | llm-d threshold | goodput static | capacity static | focal remote |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell, means in result["policy_means"].items():
        lines.append(
            f"| `{cell}` | {means[FOCAL]:.3f} | "
            f"{means['least_ttft_joint']:.3f} | "
            f"{means[PAPER_KAIROS]:.3f} | "
            f"{means[LLMD_PREFIX]:.3f} | "
            f"{means['static_joint_yardstick']:.3f} | "
            f"{means['capacity_static']:.3f} | "
            f"{result['remote_fractions'][cell][FOCAL]:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Deployable minimax ranking",
            "",
            "| rank | policy | worst regret | worst cell | equal-cell mean goodput |",
            "|---:|---|---:|---|---:|",
        ]
    )
    for index, item in enumerate(result["deployable_minimax_ranking"], 1):
        lines.append(
            f"| {index} | `{item['policy']}` | "
            f"{item['worst_regret']:.4f} | `{item['worst_cell']}` | "
            f"{item['mean_goodput']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Validity",
            "",
            f"- Runs: {result['run_count']} across {result['cell_count']} cells.",
            f"- Hard-invalid runs: {result['hard_invalid_runs']}.",
            f"- Runs with drops: {result['runs_with_drops']}.",
            f"- Runs with timeouts: {result['runs_with_timeouts']}.",
            f"- Runs with length caps: {result['runs_with_length_caps']}.",
            f"- Focal candidate traces: {result['focal_candidate_trace_runs']}; "
            + (
                "all chosen actions are exact argmins."
                if result["chosen_argmin_trace_exact"]
                else "ARGMIN CHECK FAILED."
            ),
            "",
            "Static-plan differences are descriptive static-plan gaps, not oracle",
            "or end-to-end policy-regret estimates.",
            "",
        ]
    )
    (out / "PUBLIC-LOAD-STATIC-BENCHMARK.md").write_text("\n".join(lines))


def run_confirmation(out: Path, jobs: int, force: bool) -> None:
    capacity = load_capacity(out)
    static = load_static(out)
    runs = confirmation_runs(capacity, static)
    if len(runs) != 432:
        raise RuntimeError(f"confirmation has {len(runs)} runs, want 432")
    rows = prior.run_many(runs, out, jobs, force)
    prior.require_hard_valid(rows, "held-out load/static confirmation")
    prior.write_rows(out / "confirmation.csv", rows)
    result = confirmation_analysis(out, runs, rows)
    (out / "confirmation_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_report(out, capacity, static, result)
    print(json.dumps(result, indent=2), flush=True)


def smoke(out: Path, force: bool) -> None:
    config = prior.WORKLOADS["interactive"]
    runs = [
        prior.Run(
            "public_load_static_smoke",
            config,
            "smoke",
            2.0,
            31415,
            policy,
            "h100_homogeneous",
            24,
            0.5 if policy in {"capacity_static", "static_joint_yardstick"} else None,
            0.5 if policy in {"capacity_static", "static_joint_yardstick"} else None,
        )
        for policy in CONFIRM_POLICIES
    ]
    rows = prior.run_many(runs, out, jobs=len(runs), force=force)
    prior.require_hard_valid(rows, "benchmark smoke")
    focal_run = next(run for run in runs if run.policy == FOCAL)
    summary = diagnostic.analyze_candidate_trace(
        prior.trace_path(out, focal_run), prior.V
    )
    if summary["positive_chosen_snapshot_score_regret_fraction"] != 0:
        raise RuntimeError("smoke candidate trace is not exact")
    prior.write_rows(out / "smoke.csv", rows)
    print("smoke passed: all six policies, terminal accounting, and focal argmin trace exact")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("smoke", "capacity", "static", "confirm", "all"),
        default="all",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not PROTOCOL.exists():
        raise SystemExit(f"missing frozen protocol {PROTOCOL}")
    if not prior.HETERO_BUNDLE.exists():
        raise SystemExit(f"missing hardware bundle {prior.HETERO_BUNDLE}")

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
