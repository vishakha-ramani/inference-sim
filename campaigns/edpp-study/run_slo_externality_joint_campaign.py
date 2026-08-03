#!/usr/bin/env python3
"""Fresh theory-first evaluation of constrained causal-SLO-externality routing.

The campaign has three explicit stages:

* pilot verifies the new executable, traces, and every comparison arm on one cell;
* calibrate selects one global V and a conditional static joint mapping using
  development seeds only;
* evaluate freezes those choices and runs disjoint fresh seeds.

The conditional static comparator is a deterministic distribution over the full
joint action set: its condition-specific remote share is fixed, local decodes are
spread uniformly across decode instances, and remote actions are spread uniformly
across decode/prefill pairs. It sees no live queues, residents, or cache state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import run_decisive_campaign as base
import run_minimax_var_prefill as minimax


DEFAULT_OUT = base.CAMPAIGN / "out" / "slo_externality_joint_v1"
CALIBRATION_SELECTION_SOURCE = DEFAULT_OUT / "calibration_selection.json"
CAPACITY_SOURCE = (
    base.CAMPAIGN
    / "out"
    / "joint_causal_var_confirmation_v2"
    / "capacity_selection.json"
)
TOPOLOGIES = ("1p3m", "2p2m", "3p1m")
WORKLOADS = ("synth", "rag", "shared")
LOADS = ("medium", "near_high")
DEV_SEEDS = (42, 123)
FRESH_SEEDS = (262147, 524309, 1048583, 2097169)
V_GRID = (0.25, 1.0, 4.0)
PHI_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

NEW_POLICIES = (
    "joint_slo_externality",
    "decomposed_slo_externality",
    "joint_slo_externality_no_externality",
    "joint_slo_externality_no_capacity",
)
PUBLIC_ARMS: tuple[tuple[str, float | int | None], ...] = (
    ("kairos", 0.5),
    ("least_ttft_joint", None),
    ("threshold", 16),
    ("always", None),
    ("never", None),
)
STATIC_POLICY = "conditional_static_joint"
OCCUPANCY_POLICY = "joint_slo_externality_occupancy"
OCCUPANCY_DIAGNOSTIC_KEYS = (
    "1p3m:shared:medium",
    "1p3m:shared:near_high",
    "1p3m:rag:near_high",
)
OCCUPANCY_V8 = 8.0
OCCUPANCY_V8_SOURCE = (
    base.CAMPAIGN
    / "out"
    / "slo_externality_occupancy_diagnostic_v1"
    / "occupancy_diagnostic_result.json"
)
DISPLAY = {
    "joint_slo_externality": "constrained joint controller",
    OCCUPANCY_POLICY: "occupancy-capacity controller",
    "decomposed_slo_externality": "decomposed controller",
    "joint_slo_externality_no_externality": "without causal externality",
    "joint_slo_externality_no_capacity": "without capacity prices",
    "kairos": "Kairos",
    "least_ttft_joint": "least projected TTFT (joint)",
    "threshold": "prefix threshold (16)",
    "always": "always remote",
    "never": "always local",
    STATIC_POLICY: "conditional static joint plan",
}


@dataclass(frozen=True)
class Condition:
    topology: str
    workload: base.Workload
    load: str
    rate: float
    requests: int

    @property
    def key(self) -> str:
        return f"{self.topology}:{self.workload.name}:{self.load}"


def load_capacity() -> dict[str, Any]:
    if not CAPACITY_SOURCE.exists():
        raise SystemExit(f"missing frozen capacity source {CAPACITY_SOURCE}")
    return json.loads(CAPACITY_SOURCE.read_text())


def conditions() -> list[Condition]:
    capacity = load_capacity()
    result: list[Condition] = []
    for topology in TOPOLOGIES:
        for workload_name in WORKLOADS:
            workload = base.WORKLOADS[workload_name]
            for load in LOADS:
                rate = float(capacity[topology][workload_name]["rates"][load])
                requests = (
                    base.POLICY_MIN_REQUESTS[workload_name]
                    if workload_name == "shared"
                    else max(
                        base.POLICY_MIN_REQUESTS[workload_name],
                        int(math.ceil(rate * base.ARRIVAL_SECONDS)),
                    )
                )
                result.append(Condition(topology, workload, load, rate, requests))
    return result


def hard_valid(row: dict[str, Any]) -> bool:
    return (
        int(row["completed"]) + int(row["dropped"]) == int(row["injected"])
        and int(row["still_queued"]) == 0
        and int(row["still_running"]) == 0
        and int(row["timed_out"]) == 0
        and int(row["length_capped"]) == 0
    )


def require_hard_valid(rows: Iterable[dict[str, Any]], label: str) -> None:
    bad = [
        f"{row['topology']}/{row['workload']}/{row['rate_label']}/"
        f"{row['policy']}/{row['parameter']}/s{row['seed']}"
        for row in rows
        if not hard_valid(row)
    ]
    if bad:
        raise RuntimeError(f"{label} produced hard-invalid runs: {', '.join(bad)}")


def run_for(
    phase: str,
    condition: Condition,
    seed: int,
    policy: str,
    parameter: float | int | None,
    requests: int | None = None,
) -> base.Run:
    targets = minimax.targets()[condition.workload.name]
    return base.Run(
        phase,
        condition.workload,
        condition.load,
        condition.rate,
        seed,
        policy,
        parameter,
        condition.topology,
        condition.requests if requests is None else requests,
        targets,
    )


def trace_path(out: Path, run: base.Run) -> Path:
    return out / run.phase / "candidate_traces" / f"{run.tag()}.csv"


def direction(value: float, tolerance: float = 1e-12) -> str:
    if value < -tolerance:
        return "remote"
    if value > tolerance:
        return "local"
    return "tie"


def mean_or_zero(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def direction_fractions(counts: dict[str, int], total: int) -> dict[str, float]:
    return {
        label: counts.get(label, 0) / total if total else 0.0
        for label in ("remote", "local", "tie")
    }


def analyze_candidate_trace(
    path: Path, v: float, *, occupancy_capacity: bool = False
) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing candidate trace {path}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            grouped[row["request_id"]].append(row)

    actual_remote = 0
    total_direction = defaultdict(int)
    net_direction = defaultdict(int)
    capacity_direction = defaultdict(int)
    conflicts = 0
    chosen_externality: list[float] = []
    chosen_own_good: list[float] = []
    chosen_capacity: list[float] = []
    chosen_score_regret: list[float] = []
    score_gaps: list[float] = []
    net_gaps: list[float] = []
    capacity_gaps: list[float] = []
    decode_counts: dict[str, int] = defaultdict(int)
    prefill_counts: dict[str, int] = defaultdict(int)
    previous_clock: int | None = None
    previous_queues: dict[str, float] = {}
    previous_booking: dict[str, float] = {}
    initial_queue_max = 0.0
    queue_transition_errors: list[float] = []
    queue_transition_violations = 0
    capacity_cross_term_errors: list[float] = []
    capacity_cross_term_violations = 0

    for request_id, rows in grouped.items():
        chosen = [row for row in rows if row["chosen"] == "true"]
        if len(chosen) != 1:
            raise RuntimeError(f"{path}: request {request_id} has {len(chosen)} chosen rows")
        selected = chosen[0]

        if occupancy_capacity:
            clock = int(rows[0]["clock"])
            current_queues: dict[str, float] = {}
            for row in rows:
                decode_id = row["decode_instance"]
                current_queues[decode_id] = float(row["capacity_queue_decode"])
                prefill_id = row["prefill_instance"]
                if prefill_id:
                    current_queues[prefill_id] = float(
                        row["capacity_queue_prefill"]
                    )

                queue_decode = float(row["capacity_queue_decode"])
                queue_prefill = float(row["capacity_queue_prefill"])
                demand_decode = float(row["capacity_demand_decode"])
                demand_prefill = float(row["capacity_demand_prefill"])
                expected_cross_term = (
                    queue_decode * demand_decode
                    + queue_prefill * demand_prefill
                ) / 1_000_000**2
                cross_error = abs(
                    float(row["capacity_total"]) - expected_cross_term
                )
                capacity_cross_term_errors.append(cross_error)
                if cross_error > 1e-10 * max(1.0, abs(expected_cross_term)):
                    capacity_cross_term_violations += 1

            if previous_clock is None:
                initial_queue_max = max(current_queues.values(), default=0.0)
            else:
                elapsed = float(clock - previous_clock)
                for instance in current_queues.keys() | previous_queues.keys():
                    expected_queue = max(
                        previous_queues.get(instance, 0.0)
                        + previous_booking.get(instance, 0.0)
                        - elapsed,
                        0.0,
                    )
                    queue_error = abs(
                        current_queues.get(instance, 0.0) - expected_queue
                    )
                    queue_transition_errors.append(queue_error)
                    if queue_error > 1e-8 * max(1.0, abs(expected_queue)):
                        queue_transition_violations += 1

            previous_clock = clock
            previous_queues = current_queues
            previous_booking = {
                selected["decode_instance"]: float(
                    selected["capacity_demand_decode"]
                )
            }
            if selected["prefill_instance"]:
                previous_booking[selected["prefill_instance"]] = float(
                    selected["capacity_demand_prefill"]
                )

        is_remote = selected["local"] != "true"
        actual_remote += int(is_remote)
        decode_counts[selected["decode_instance"]] += 1
        if is_remote:
            prefill_counts[selected["prefill_instance"]] += 1
        chosen_externality.append(float(selected["slo_externality"]))
        chosen_own_good.append(float(selected["own_good"]))
        chosen_capacity.append(float(selected["capacity_total"]))
        chosen_score_regret.append(float(selected["chosen_score_regret"]))

        local = min(
            (row for row in rows if row["local"] == "true"),
            key=lambda row: float(row["score"]),
        )
        remote_rows = [row for row in rows if row["local"] != "true"]
        if not remote_rows:
            continue
        remote = min(remote_rows, key=lambda row: float(row["score"]))
        net_gap = v * (
            float(remote["net_good_cost"]) - float(local["net_good_cost"])
        )
        capacity_gap = float(remote["capacity_total"]) - float(local["capacity_total"])
        score_gap = float(remote["score"]) - float(local["score"])
        assert abs(score_gap - (net_gap + capacity_gap)) <= 1e-8 * max(1, abs(score_gap))
        net_gaps.append(net_gap)
        capacity_gaps.append(capacity_gap)
        score_gaps.append(score_gap)
        total_direction[direction(score_gap)] += 1
        net_direction[direction(net_gap)] += 1
        capacity_direction[direction(capacity_gap)] += 1
        if direction(net_gap) != "tie" and direction(capacity_gap) != "tie" and direction(net_gap) != direction(capacity_gap):
            conflicts += 1

    count = len(grouped)
    return {
        "requests": count,
        "actual_remote_fraction": actual_remote / count if count else 0.0,
        "best_remote_vs_local": dict(total_direction),
        "best_remote_vs_local_fraction": direction_fractions(
            total_direction, len(score_gaps)
        ),
        "net_good_term_direction": dict(net_direction),
        "net_good_term_direction_fraction": direction_fractions(
            net_direction, len(net_gaps)
        ),
        "capacity_term_direction": dict(capacity_direction),
        "capacity_term_direction_fraction": direction_fractions(
            capacity_direction, len(capacity_gaps)
        ),
        "term_conflict_fraction": conflicts / len(score_gaps) if score_gaps else 0.0,
        "mean_remote_minus_local_score": mean_or_zero(score_gaps),
        "mean_absolute_remote_minus_local_score": mean_or_zero(
            [abs(value) for value in score_gaps]
        ),
        "mean_remote_minus_local_net_good_term": mean_or_zero(net_gaps),
        "mean_absolute_remote_minus_local_net_good_term": mean_or_zero(
            [abs(value) for value in net_gaps]
        ),
        "mean_remote_minus_local_capacity_term": mean_or_zero(capacity_gaps),
        "mean_absolute_remote_minus_local_capacity_term": mean_or_zero(
            [abs(value) for value in capacity_gaps]
        ),
        "mean_chosen_externality": mean_or_zero(chosen_externality),
        "mean_chosen_own_good": mean_or_zero(chosen_own_good),
        "mean_chosen_capacity": mean_or_zero(chosen_capacity),
        "mean_chosen_snapshot_score_regret": mean_or_zero(chosen_score_regret),
        "positive_chosen_snapshot_score_regret_fraction": (
            sum(value > 1e-12 for value in chosen_score_regret) / count if count else 0.0
        ),
        "decode_action_counts": dict(sorted(decode_counts.items())),
        "prefill_action_counts": dict(sorted(prefill_counts.items())),
        "occupancy_queue_conservation": {
            "enabled": occupancy_capacity,
            "checked_transitions": len(queue_transition_errors),
            "initial_queue_max_us": initial_queue_max,
            "max_abs_transition_error_us": max(
                queue_transition_errors, default=0.0
            ),
            "transition_violations": queue_transition_violations,
            "checked_capacity_cross_terms": len(capacity_cross_term_errors),
            "max_abs_capacity_cross_term_error": max(
                capacity_cross_term_errors, default=0.0
            ),
            "capacity_cross_term_violations": capacity_cross_term_violations,
        },
    }


def run_pilot(out: Path, jobs: int, force: bool) -> None:
    condition = next(
        item
        for item in conditions()
        if item.topology == "1p3m" and item.workload.name == "synth" and item.load == "medium"
    )
    arms = [
        *( (policy, 1.0) for policy in NEW_POLICIES ),
        *PUBLIC_ARMS,
        (STATIC_POLICY, 0.5),
    ]
    runs = [
        run_for(
            "slo_externality_pilot",
            condition,
            DEV_SEEDS[0],
            policy,
            parameter,
            requests=min(condition.requests, 800),
        )
        for policy, parameter in arms
    ]
    rows = base.run_many(runs, out, jobs, force)
    require_hard_valid(rows, "pilot")
    base.write_rows(out / "pilot.csv", rows)
    decisions = {
        run.policy: analyze_candidate_trace(trace_path(out, run), 1.0)
        for run in runs
        if run.policy in NEW_POLICIES
    }
    result = {"condition": condition.key, "rows": rows, "decisions": decisions}
    (out / "pilot_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def calibrate(out: Path, jobs: int, force: bool) -> None:
    runs: list[base.Run] = []
    for condition in conditions():
        for seed in DEV_SEEDS:
            for v in V_GRID:
                runs.append(
                    run_for(
                        "slo_externality_calibration",
                        condition,
                        seed,
                        "joint_slo_externality",
                        v,
                    )
                )
            for phi in PHI_GRID:
                runs.append(
                    run_for(
                        "slo_externality_static_calibration",
                        condition,
                        seed,
                        STATIC_POLICY,
                        phi,
                    )
                )
    rows = base.run_many(runs, out, jobs, force)
    require_hard_valid(rows, "calibration")
    base.write_rows(out / "calibration.csv", rows)

    adaptive_rows = [row for row in rows if row["policy"] == "joint_slo_externality"]
    by_v: dict[float, dict[str, float]] = {v: {} for v in V_GRID}
    for v in V_GRID:
        for condition in conditions():
            members = [
                row for row in adaptive_rows
                if row["topology"] == condition.topology
                and row["workload"] == condition.workload.name
                and row["rate_label"] == condition.load
                and float(row["parameter"]) == v
            ]
            if len(members) != len(DEV_SEEDS):
                raise RuntimeError(f"incomplete V calibration for {v}/{condition.key}")
            by_v[v][condition.key] = statistics.fmean(float(row["goodput"]) for row in members)
    best_by_condition = {
        condition.key: max(by_v[v][condition.key] for v in V_GRID)
        for condition in conditions()
    }
    v_ranking = []
    for v in V_GRID:
        regrets = {
            key: best_by_condition[key] - by_v[v][key]
            for key in best_by_condition
        }
        v_ranking.append({
            "v": v,
            "worst_regret": max(regrets.values()),
            "mean_goodput": statistics.fmean(by_v[v].values()),
            "worst_condition": max(regrets, key=regrets.get),
        })
    v_ranking.sort(key=lambda row: (row["worst_regret"], -row["mean_goodput"], row["v"]))
    selected_v = float(v_ranking[0]["v"])

    static_rows = [row for row in rows if row["policy"] == STATIC_POLICY]
    static_selection: dict[str, Any] = {}
    for condition in conditions():
        candidates = []
        for phi in PHI_GRID:
            members = [
                row for row in static_rows
                if row["topology"] == condition.topology
                and row["workload"] == condition.workload.name
                and row["rate_label"] == condition.load
                and float(row["parameter"]) == phi
            ]
            if len(members) != len(DEV_SEEDS):
                raise RuntimeError(f"incomplete static calibration for {phi}/{condition.key}")
            candidates.append({
                "phi": phi,
                "mean_goodput": statistics.fmean(float(row["goodput"]) for row in members),
                "zero_drop": all(int(row["dropped"]) == 0 for row in members),
            })
        eligible = [item for item in candidates if item["zero_drop"]]
        if not eligible:
            raise RuntimeError(f"no zero-drop static joint plan for {condition.key}")
        selected = max(eligible, key=lambda item: (item["mean_goodput"], -item["phi"]))
        static_selection[condition.key] = {"selected_phi": selected["phi"], "grid": candidates}

    result = {
        "status": "development-only selection; frozen before fresh seeds",
        "development_seeds": list(DEV_SEEDS),
        "v_grid": list(V_GRID),
        "v_ranking": v_ranking,
        "selected_v": selected_v,
        "conditional_static_joint": static_selection,
    }
    (out / "calibration_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def diagnostic_conditions() -> list[Condition]:
    selected = [item for item in conditions() if item.key in OCCUPANCY_DIAGNOSTIC_KEYS]
    if tuple(item.key for item in selected) != OCCUPANCY_DIAGNOSTIC_KEYS:
        by_key = {item.key: item for item in selected}
        selected = [by_key[key] for key in OCCUPANCY_DIAGNOSTIC_KEYS]
    if tuple(item.key for item in selected) != OCCUPANCY_DIAGNOSTIC_KEYS:
        raise RuntimeError("could not resolve the three occupancy diagnostic conditions")
    return selected


def diagnostic_arm_key(policy: str, parameter: float | int | None) -> str:
    if policy == OCCUPANCY_POLICY:
        return f"occupancy_V={float(parameter):g}"
    return policy


def pooled_decision_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    requests = sum(int(item["requests"]) for item in items)
    if not requests:
        raise RuntimeError("cannot pool empty decision summaries")

    def weighted(field: str) -> float:
        return sum(
            float(item[field]) * int(item["requests"]) for item in items
        ) / requests

    conservation = [item["occupancy_queue_conservation"] for item in items]
    return {
        "requests": requests,
        "actual_remote_fraction": weighted("actual_remote_fraction"),
        "term_conflict_fraction": weighted("term_conflict_fraction"),
        "net_good_term_direction_fraction": {
            label: sum(
                float(item["net_good_term_direction_fraction"][label])
                * int(item["requests"])
                for item in items
            )
            / requests
            for label in ("remote", "local", "tie")
        },
        "capacity_term_direction_fraction": {
            label: sum(
                float(item["capacity_term_direction_fraction"][label])
                * int(item["requests"])
                for item in items
            )
            / requests
            for label in ("remote", "local", "tie")
        },
        "mean_absolute_remote_minus_local_net_good_term": weighted(
            "mean_absolute_remote_minus_local_net_good_term"
        ),
        "mean_absolute_remote_minus_local_capacity_term": weighted(
            "mean_absolute_remote_minus_local_capacity_term"
        ),
        "mean_chosen_snapshot_score_regret": weighted(
            "mean_chosen_snapshot_score_regret"
        ),
        "positive_chosen_snapshot_score_regret_fraction": weighted(
            "positive_chosen_snapshot_score_regret_fraction"
        ),
        "occupancy_queue_conservation": {
            "checked_transitions": sum(
                int(item["checked_transitions"]) for item in conservation
            ),
            "initial_queue_max_us": max(
                (float(item["initial_queue_max_us"]) for item in conservation),
                default=0.0,
            ),
            "max_abs_transition_error_us": max(
                (
                    float(item["max_abs_transition_error_us"])
                    for item in conservation
                ),
                default=0.0,
            ),
            "transition_violations": sum(
                int(item["transition_violations"]) for item in conservation
            ),
            "checked_capacity_cross_terms": sum(
                int(item["checked_capacity_cross_terms"])
                for item in conservation
            ),
            "max_abs_capacity_cross_term_error": max(
                (
                    float(item["max_abs_capacity_cross_term_error"])
                    for item in conservation
                ),
                default=0.0,
            ),
            "capacity_cross_term_violations": sum(
                int(item["capacity_cross_term_violations"])
                for item in conservation
            ),
        },
    }


def occupancy_diagnostic(out: Path, jobs: int, force: bool) -> None:
    if not CALIBRATION_SELECTION_SOURCE.exists():
        raise SystemExit(
            f"missing frozen development selection {CALIBRATION_SELECTION_SOURCE}"
        )
    selection = json.loads(CALIBRATION_SELECTION_SOURCE.read_text())
    fixed_v = float(selection["selected_v"])

    runs: list[base.Run] = []
    for condition in diagnostic_conditions():
        static_phi = float(
            selection["conditional_static_joint"][condition.key]["selected_phi"]
        )
        arms = [
            *((OCCUPANCY_POLICY, v) for v in V_GRID),
            ("joint_slo_externality", fixed_v),
            ("joint_slo_externality_no_capacity", fixed_v),
            ("least_ttft_joint", None),
            ("kairos", 0.5),
            (STATIC_POLICY, static_phi),
        ]
        for seed in DEV_SEEDS:
            for policy, parameter in arms:
                runs.append(
                    run_for(
                        "slo_externality_occupancy_diagnostic",
                        condition,
                        seed,
                        policy,
                        parameter,
                    )
                )
    if len(runs) != 48:
        raise RuntimeError(f"occupancy diagnostic has {len(runs)} runs, want 48")

    rows = base.run_many(runs, out, jobs, force)
    require_hard_valid(rows, "occupancy diagnostic")
    base.write_rows(out / "occupancy_diagnostic.csv", rows)

    summaries: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    traced_policies = {
        OCCUPANCY_POLICY,
        "joint_slo_externality",
        "joint_slo_externality_no_capacity",
    }
    for run in runs:
        if run.policy not in traced_policies:
            continue
        summary = analyze_candidate_trace(
            trace_path(out, run),
            float(run.parameter),
            occupancy_capacity=run.policy == OCCUPANCY_POLICY,
        )
        summary["seed"] = run.seed
        summaries[run.topology + ":" + run.workload.name + ":" + run.rate_label][
            diagnostic_arm_key(run.policy, run.parameter)
        ].append(summary)

    pooled = {
        condition: {
            arm: pooled_decision_summary(items) for arm, items in arms.items()
        }
        for condition, arms in summaries.items()
    }

    means: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for condition in diagnostic_conditions():
        condition_rows = [
            row
            for row in rows
            if row["topology"] == condition.topology
            and row["workload"] == condition.workload.name
            and row["rate_label"] == condition.load
        ]
        grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in condition_rows:
            parameter = None if row["parameter"] == "" else float(row["parameter"])
            grouped_rows[diagnostic_arm_key(row["policy"], parameter)].append(row)
        for arm, members in grouped_rows.items():
            if len(members) != len(DEV_SEEDS):
                raise RuntimeError(f"incomplete diagnostic arm {condition.key}/{arm}")
            means[condition.key][arm] = {
                "goodput": statistics.fmean(float(row["goodput"]) for row in members),
                "remote_fraction": statistics.fmean(
                    float(row["realized_phi"]) for row in members
                ),
            }

    best_occupancy = {
        condition.key: max(
            means[condition.key][diagnostic_arm_key(OCCUPANCY_POLICY, v)][
                "goodput"
            ]
            for v in V_GRID
        )
        for condition in diagnostic_conditions()
    }
    v_ranking = []
    for v in V_GRID:
        arm = diagnostic_arm_key(OCCUPANCY_POLICY, v)
        regrets = {
            condition.key: best_occupancy[condition.key]
            - means[condition.key][arm]["goodput"]
            for condition in diagnostic_conditions()
        }
        v_ranking.append(
            {
                "v": v,
                "worst_regret": max(regrets.values()),
                "mean_goodput": statistics.fmean(
                    means[condition.key][arm]["goodput"]
                    for condition in diagnostic_conditions()
                ),
                "worst_condition": max(regrets, key=regrets.get),
            }
        )
    v_ranking.sort(
        key=lambda item: (item["worst_regret"], -item["mean_goodput"], item["v"])
    )
    selected_v = float(v_ranking[0]["v"])
    selected_arm = diagnostic_arm_key(OCCUPANCY_POLICY, selected_v)

    shared_checks: dict[str, Any] = {}
    for condition_key in OCCUPANCY_DIAGNOSTIC_KEYS[:2]:
        static_phi = float(
            selection["conditional_static_joint"][condition_key]["selected_phi"]
        )
        occupancy = means[condition_key][selected_arm]
        fixed = means[condition_key]["joint_slo_externality"]
        shared_checks[condition_key] = {
            "goodput_gain_over_fixed": occupancy["goodput"] - fixed["goodput"],
            "occupancy_remote_fraction": occupancy["remote_fraction"],
            "fixed_remote_fraction": fixed["remote_fraction"],
            "static_remote_fraction": static_phi,
            "moves_remote_fraction_toward_static": abs(
                occupancy["remote_fraction"] - static_phi
            )
            < abs(fixed["remote_fraction"] - static_phi),
        }

    shared_medium = OCCUPANCY_DIAGNOSTIC_KEYS[0]
    conflict_reduction = (
        pooled[shared_medium]["joint_slo_externality"]["term_conflict_fraction"]
        - pooled[shared_medium][selected_arm]["term_conflict_fraction"]
    )
    rag_key = OCCUPANCY_DIAGNOSTIC_KEYS[2]
    rag_goodput_delta = (
        means[rag_key][selected_arm]["goodput"]
        - means[rag_key]["joint_slo_externality"]["goodput"]
    )
    conservation = [
        pooled[condition.key][selected_arm]["occupancy_queue_conservation"]
        for condition in diagnostic_conditions()
    ]
    trace_valid = (
        all(item["transition_violations"] == 0 for item in conservation)
        and all(item["capacity_cross_term_violations"] == 0 for item in conservation)
        and all(item["initial_queue_max_us"] == 0 for item in conservation)
        and all(
            pooled[condition.key][selected_arm][
                "positive_chosen_snapshot_score_regret_fraction"
            ]
            == 0
            for condition in diagnostic_conditions()
        )
    )
    gate = {
        "selected_global_occupancy_v": selected_v,
        "shared_medium_conflict_reduction": conflict_reduction,
        "capacity_conflict_reduced": conflict_reduction > 0,
        "shared_checks": shared_checks,
        "rag_near_high_goodput_delta_vs_fixed": rag_goodput_delta,
        "rag_not_materially_degraded_0p05": rag_goodput_delta >= -0.05,
        "trace_and_queue_conservation_valid": trace_valid,
    }
    gate["passes_bounded_diagnostic"] = (
        gate["capacity_conflict_reduced"]
        and all(
            item["goodput_gain_over_fixed"] > 0
            and item["moves_remote_fraction_toward_static"]
            for item in shared_checks.values()
        )
        and gate["rag_not_materially_degraded_0p05"]
        and trace_valid
    )

    result = {
        "status": "development-only bounded diagnostic; no fresh seeds",
        "conditions": list(OCCUPANCY_DIAGNOSTIC_KEYS),
        "seeds": list(DEV_SEEDS),
        "run_count": len(runs),
        "fixed_controller_v": fixed_v,
        "occupancy_v_grid": list(V_GRID),
        "v_ranking": v_ranking,
        "means": means,
        "pooled_decision_diagnostics": pooled,
        "decision_gate": gate,
    }
    (out / "occupancy_diagnostic_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    arm_order = [
        *(diagnostic_arm_key(OCCUPANCY_POLICY, v) for v in V_GRID),
        "joint_slo_externality",
        "joint_slo_externality_no_capacity",
        "least_ttft_joint",
        "kairos",
        STATIC_POLICY,
    ]
    lines = [
        "# Occupancy-capacity bounded diagnostic",
        "",
        "Development seeds only; 48 runs over the three pre-registered conditions.",
        "",
        "| condition | arm | goodput | remote fraction |",
        "|---|---|---:|---:|",
    ]
    for condition in diagnostic_conditions():
        for arm in arm_order:
            cell = means[condition.key][arm]
            lines.append(
                f"| `{condition.key}` | `{arm}` | {cell['goodput']:.3f} | "
                f"{cell['remote_fraction']:.3f} |"
            )
    lines.extend(
        [
            "",
            f"Selected one global occupancy value: `V={selected_v:g}`.",
            f"Bounded decision gate: **{'PASS' if gate['passes_bounded_diagnostic'] else 'STOP'}**.",
            "",
        ]
    )
    (out / "OCCUPANCY-DIAGNOSTIC.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


def occupancy_v8_decision(out: Path, jobs: int, force: bool) -> None:
    """Run only the preregistered three-condition, two-seed V=8 decision."""
    if not OCCUPANCY_V8_SOURCE.exists():
        raise SystemExit(f"missing V=4 diagnostic source {OCCUPANCY_V8_SOURCE}")
    source = json.loads(OCCUPANCY_V8_SOURCE.read_text())

    runs = [
        run_for(
            "slo_externality_occupancy_v8_decision",
            condition,
            seed,
            OCCUPANCY_POLICY,
            OCCUPANCY_V8,
        )
        for condition in diagnostic_conditions()
        for seed in DEV_SEEDS
    ]
    if len(runs) != 6:
        raise RuntimeError(f"V=8 decision has {len(runs)} runs, want 6")

    rows = base.run_many(runs, out, jobs, force)
    require_hard_valid(rows, "occupancy V=8 decision")
    base.write_rows(out / "occupancy_v8_decision.csv", rows)

    summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        summary = analyze_candidate_trace(
            trace_path(out, run), OCCUPANCY_V8, occupancy_capacity=True
        )
        summary["seed"] = run.seed
        summaries[f"{run.topology}:{run.workload.name}:{run.rate_label}"].append(
            summary
        )
    pooled = {
        condition: pooled_decision_summary(items)
        for condition, items in summaries.items()
    }

    means: dict[str, dict[str, float]] = {}
    per_seed: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    for condition in diagnostic_conditions():
        members = [
            row
            for row in rows
            if row["topology"] == condition.topology
            and row["workload"] == condition.workload.name
            and row["rate_label"] == condition.load
        ]
        if len(members) != len(DEV_SEEDS):
            raise RuntimeError(f"incomplete V=8 decision for {condition.key}")
        means[condition.key] = {
            "goodput": statistics.fmean(float(row["goodput"]) for row in members),
            "remote_fraction": statistics.fmean(
                float(row["realized_phi"]) for row in members
            ),
        }
        per_seed[condition.key] = [
            {
                "seed": int(row["seed"]),
                "goodput": float(row["goodput"]),
                "remote_fraction": float(row["realized_phi"]),
            }
            for row in sorted(members, key=lambda item: int(item["seed"]))
        ]

    shared_medium, shared_near_high, rag_near_high = OCCUPANCY_DIAGNOSTIC_KEYS
    v4_near_high = float(
        source["means"][shared_near_high]["occupancy_V=4"]["goodput"]
    )
    conservation = [
        pooled[key]["occupancy_queue_conservation"]
        for key in OCCUPANCY_DIAGNOSTIC_KEYS
    ]
    trace_valid = (
        all(item["transition_violations"] == 0 for item in conservation)
        and all(item["capacity_cross_term_violations"] == 0 for item in conservation)
        and all(item["initial_queue_max_us"] == 0 for item in conservation)
        and all(
            pooled[key]["positive_chosen_snapshot_score_regret_fraction"] == 0
            for key in OCCUPANCY_DIAGNOSTIC_KEYS
        )
    )
    gates = {
        "shared_medium_goodput_at_least_0p53": (
            means[shared_medium]["goodput"] >= 0.53
        ),
        "shared_medium_remote_fraction_at_least_0p75": (
            means[shared_medium]["remote_fraction"] >= 0.75
        ),
        "shared_near_high_improves_over_v4": (
            means[shared_near_high]["goodput"] > v4_near_high
        ),
        "rag_near_high_goodput_at_least_0p83": (
            means[rag_near_high]["goodput"] >= 0.83
        ),
        "trace_and_queue_conservation_exact": trace_valid,
    }
    passes = all(gates.values())
    result = {
        "status": "PASS: freeze V=8 for fresh confirmation"
        if passes
        else "STOP: abandon capacity-controller development",
        "development_only": True,
        "v": OCCUPANCY_V8,
        "conditions": list(OCCUPANCY_DIAGNOSTIC_KEYS),
        "seeds": list(DEV_SEEDS),
        "run_count": len(runs),
        "v4_shared_near_high_goodput": v4_near_high,
        "means": means,
        "per_seed": per_seed,
        "pooled_decision_diagnostics": pooled,
        "gates": gates,
        "passes_all_gates": passes,
    }
    (out / "occupancy_v8_decision_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    lines = [
        "# Occupancy-capacity V=8 decision",
        "",
        "Development-only preregistered decision: three conditions by seeds 42/123.",
        "",
        "| condition | goodput | remote fraction |",
        "|---|---:|---:|",
    ]
    for key in OCCUPANCY_DIAGNOSTIC_KEYS:
        lines.append(
            f"| `{key}` | {means[key]['goodput']:.3f} | "
            f"{means[key]['remote_fraction']:.3f} |"
        )
    lines.extend(["", "## Preregistered gates", ""])
    for name, passed in gates.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            f"Final decision: **{'PASS' if passes else 'STOP'}**.",
            "",
        ]
    )
    (out / "OCCUPANCY-V8-DECISION.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


def paired(rows: list[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
    result = []
    for condition in conditions():
        left_by_seed = {
            int(row["seed"]): float(row["goodput"])
            for row in rows
            if row["topology"] == condition.topology
            and row["workload"] == condition.workload.name
            and row["rate_label"] == condition.load
            and row["policy"] == left
        }
        right_by_seed = {
            int(row["seed"]): float(row["goodput"])
            for row in rows
            if row["topology"] == condition.topology
            and row["workload"] == condition.workload.name
            and row["rate_label"] == condition.load
            and row["policy"] == right
        }
        deltas = [left_by_seed[seed] - right_by_seed[seed] for seed in FRESH_SEEDS]
        mean, low, high = base.ci95(deltas)
        result.append({
            "condition": condition.key,
            "paired_delta": mean,
            "ci_low": low,
            "ci_high": high,
            "wins": sum(value > 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
            "losses": sum(value < 0 for value in deltas),
        })
    return result


def evaluate(out: Path, jobs: int, force: bool) -> None:
    selection_path = out / "calibration_selection.json"
    if not selection_path.exists():
        raise SystemExit("calibration_selection.json missing; run --stage calibrate first")
    selection = json.loads(selection_path.read_text())
    selected_v = float(selection["selected_v"])

    runs: list[base.Run] = []
    for condition in conditions():
        static_phi = float(selection["conditional_static_joint"][condition.key]["selected_phi"])
        arms = [
            *( (policy, selected_v) for policy in NEW_POLICIES ),
            *PUBLIC_ARMS,
            (STATIC_POLICY, static_phi),
        ]
        for seed in FRESH_SEEDS:
            for policy, parameter in arms:
                runs.append(
                    run_for(
                        "slo_externality_fresh_evaluation",
                        condition,
                        seed,
                        policy,
                        parameter,
                    )
                )
    rows = base.run_many(runs, out, jobs, force)
    require_hard_valid(rows, "fresh evaluation")
    base.write_rows(out / "fresh_evaluation.csv", rows)

    policies = (*NEW_POLICIES, *(policy for policy, _ in PUBLIC_ARMS), STATIC_POLICY)
    means: dict[str, dict[str, float]] = {policy: {} for policy in policies}
    for policy in policies:
        for condition in conditions():
            members = [
                row for row in rows
                if row["topology"] == condition.topology
                and row["workload"] == condition.workload.name
                and row["rate_label"] == condition.load
                and row["policy"] == policy
            ]
            if len(members) != len(FRESH_SEEDS):
                raise RuntimeError(f"incomplete fresh evaluation for {policy}/{condition.key}")
            means[policy][condition.key] = statistics.fmean(float(row["goodput"]) for row in members)

    deployable = (*NEW_POLICIES, *(policy for policy, _ in PUBLIC_ARMS))
    best_deployable = {
        condition.key: max(means[policy][condition.key] for policy in deployable)
        for condition in conditions()
    }
    minimax_ranking = []
    for policy in deployable:
        regrets = {
            condition.key: best_deployable[condition.key] - means[policy][condition.key]
            for condition in conditions()
        }
        minimax_ranking.append({
            "policy": policy,
            "display": DISPLAY[policy],
            "worst_regret": max(regrets.values()),
            "worst_condition": max(regrets, key=regrets.get),
        })
    minimax_ranking.sort(key=lambda item: (item["worst_regret"], item["policy"]))

    decision_summaries: dict[str, dict[str, list[dict[str, Any]]]] = {
        policy: defaultdict(list) for policy in NEW_POLICIES
    }
    run_by_tag = {run.tag(): run for run in runs}
    for run in run_by_tag.values():
        if run.policy not in NEW_POLICIES:
            continue
        summary = analyze_candidate_trace(trace_path(out, run), selected_v)
        summary["seed"] = run.seed
        key = f"{run.topology}:{run.workload.name}:{run.rate_label}"
        decision_summaries[run.policy][key].append(summary)

    pairwise = {
        policy: paired(rows, "joint_slo_externality", policy)
        for policy in policies
        if policy != "joint_slo_externality"
    }
    result = {
        "status": (
            "fresh seeds; V and conditional static joint mapping frozen from development seeds; "
            "decision comparisons are snapshot diagnostics, not policy regret"
        ),
        "fresh_seeds": list(FRESH_SEEDS),
        "selected_v": selected_v,
        "policy_means": means,
        "best_deployable_by_condition": best_deployable,
        "minimax_ranking": minimax_ranking,
        "full_controller_pairwise": pairwise,
        "decision_summaries": decision_summaries,
        "strict_zero_drop_failures": [
            {
                "condition": f"{row['topology']}:{row['workload']}:{row['rate_label']}",
                "policy": row["policy"],
                "seed": int(row["seed"]),
                "dropped": int(row["dropped"]),
            }
            for row in rows
            if int(row["dropped"]) > 0
        ],
    }
    (out / "fresh_evaluation_result.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Fresh constrained joint-controller evaluation",
        "",
        "The controller and all tuning choices were frozen before the fresh seeds.",
        "The conditional static joint plan is an offline yardstick and is excluded",
        "from the deployable minimax ranking. Decision-score regret is evaluated on",
        "the controller's snapshots and is not end-to-end policy regret.",
        "",
        "| rank | policy | worst regret | worst condition |",
        "|---:|---|---:|---|",
    ]
    for index, row in enumerate(minimax_ranking, 1):
        lines.append(
            f"| {index} | {row['display']} | {row['worst_regret']:.3f} | "
            f"`{row['worst_condition']}` |"
        )
    lines.extend([
        "",
        "## Mean goodput by condition",
        "",
        "| condition | " + " | ".join(DISPLAY[policy] for policy in policies) + " |",
        "|---|" + "---:|" * len(policies),
    ])
    for condition in conditions():
        lines.append(
            f"| `{condition.key}` | "
            + " | ".join(f"{means[policy][condition.key]:.3f}" for policy in policies)
            + " |"
        )
    lines.append("")
    (out / "SLO-EXTERNALITY-FRESH-RESULT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("pilot", "calibrate", "occupancy", "occupancy-v8", "evaluate", "all"),
        default="pilot",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "pilot":
        run_pilot(out, args.jobs, args.force)
    elif args.stage == "calibrate":
        calibrate(out, args.jobs, args.force)
    elif args.stage == "occupancy":
        occupancy_diagnostic(out, args.jobs, args.force)
    elif args.stage == "occupancy-v8":
        occupancy_v8_decision(out, args.jobs, args.force)
    elif args.stage == "evaluate":
        evaluate(out, args.jobs, args.force)
    else:
        run_pilot(out, args.jobs, args.force)
        calibrate(out, args.jobs, args.force)
        evaluate(out, args.jobs, args.force)


if __name__ == "__main__":
    main()
