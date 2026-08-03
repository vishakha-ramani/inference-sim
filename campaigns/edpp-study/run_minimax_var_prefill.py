#!/usr/bin/env python3
"""Run the preregistered minimax-regret var-prefill campaign.

This companion imports the established decisive-campaign workload, validity,
and result machinery, while using routing-preserving static P/D plans.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import run_decisive_campaign as base


DEFAULT_OUT = base.CAMPAIGN / "out" / "minimax_var_prefill_v1"
TARGETS_SOURCE = base.CAMPAIGN / "out" / "decisive" / "targets.json"

PHIS = tuple(round(i / 10, 1) for i in range(11))
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)
LOAD_FRACTIONS = {"low": 0.60, "medium": 0.80, "near_high": 0.95}
CONFIRMATION_SEEDS = (13, 29, 61, 251, 509, 1021, 4093, 8191)


def targets() -> dict[str, dict[str, dict[str, float]]]:
    if not TARGETS_SOURCE.exists():
        raise SystemExit(
            f"missing {TARGETS_SOURCE}; the decisive campaign targets are required"
        )
    return json.loads(TARGETS_SOURCE.read_text())


def conditions(
    out: Path,
) -> list[tuple[base.Workload, str, float, int]]:
    path = out / "capacity_selection.json"
    if not path.exists():
        raise SystemExit("capacity_selection.json missing; run capacity first")
    selected = json.loads(path.read_text())
    result = []
    for workload in base.WORKLOADS.values():
        for label, rate in selected[workload.name]["rates"].items():
            requests = (
                base.POLICY_MIN_REQUESTS[workload.name]
                if workload.name == "shared"
                else max(
                    base.POLICY_MIN_REQUESTS[workload.name],
                    int(math.ceil(rate * base.ARRIVAL_SECONDS)),
                )
            )
            result.append((workload, label, float(rate), requests))
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def hard_valid(row: dict[str, Any]) -> bool:
    """Validity excluding unservable drops, which are a measurable policy failure.

    Goodput uses injected arrivals as its denominator, so a dropped request already
    contributes zero. Static oracle arms have the stronger zero-drop eligibility
    requirement below; adaptive policies are allowed to fail this way without
    invalidating the rest of the campaign.
    """

    injected = int(row["injected"])
    conserved = (
        int(row["completed"])
        + int(row["still_queued"])
        + int(row["still_running"])
        + int(row["dropped"])
        + int(row["timed_out"])
        == injected
    )
    rate = float(row["offered_rate"])
    rate_valid = (
        row["rate_label"] == "staged"
        or (rate > 0 and abs(float(row["actual_rate"]) - rate) <= 0.10 * rate)
    )
    share_valid = True
    if row["phase"] != "capacity" and row["policy"] in {
        "fixed_phi",
        "universal_phi",
        "tuned_phi",
        "routing_phi",
        "universal_routing_phi",
        "oracle_routing_phi",
    }:
        expected = round(float(row["parameter"]) * injected)
        share_valid = (
            row["disaggregated"] != ""
            # A request dropped as unservable before its fixed P/D action is
            # committed cannot increment the disaggregation counter. Preserve
            # exact-plan validation for routed requests while allowing that
            # explicitly counted failure to explain the missing action.
            and abs(int(row["disaggregated"]) - expected)
            <= int(row["dropped"]) + 1
        )
    return (
        conserved
        and int(row["timed_out"]) == 0
        and int(row["still_queued"]) == 0
        and int(row["still_running"]) == 0
        and int(row["length_capped"]) == 0
        and share_valid
        and rate_valid
    )


def require_hard_valid(rows: list[dict[str, Any]], phase: str) -> None:
    invalid = [
        f"{row['workload']}/{row['rate_label']}/{row['policy']}/"
        f"{row['parameter']}/s{row['seed']}"
        for row in rows
        if not hard_valid(row)
    ]
    if invalid:
        raise RuntimeError(
            f"{phase} produced hard-invalid runs: {', '.join(invalid)}"
        )


def static_eligible(row: dict[str, Any]) -> bool:
    return (
        hard_valid(row)
        and int(row["dropped"]) == 0
        and int(row["completed"]) == int(row["injected"])
    )


def run_capacity(out: Path, jobs: int, force: bool) -> None:
    runs = [
        base.Run(
            "capacity",
            workload,
            "saturated",
            workload.saturation_rate,
            base.CALIBRATION_SEED,
            "routing_phi",
            phi,
            "1p2m",
            base.CAPACITY_REQUESTS[workload.name],
            base.loose_targets(workload),
        )
        for workload in base.WORKLOADS.values()
        for phi in PHIS
    ]
    rows = base.run_many(runs, out, jobs, force)
    base.require_valid(rows, "capacity")
    base.write_rows(out / "capacity.csv", rows)

    selection: dict[str, Any] = {}
    for workload in base.WORKLOADS:
        family = [row for row in rows if row["workload"] == workload]
        stable = [
            row
            for row in family
            if int(row["dropped"]) == 0
            and int(row["completed"]) == int(row["injected"])
        ]
        if not stable:
            raise RuntimeError(
                f"{workload}: no zero-drop, fully-completed static fraction "
                "was observed in the saturation sweep"
            )
        best = max(
            stable,
            key=lambda row: (
                float(row["central_completion_rps"]),
                -float(row["parameter"]),
            ),
        )
        ceiling = float(best["central_completion_rps"])
        fastest = max(float(row["central_completion_rps"]) for row in family)
        offered = base.WORKLOADS[workload].saturation_rate
        if offered < 1.05 * fastest:
            raise RuntimeError(
                f"{workload}: saturation offer {offered:.3f} is less than "
                f"1.05 x measured capacity {fastest:.3f}"
            )
        selection[workload] = {
            "ceiling_rps": ceiling,
            "ceiling_phi": float(best["parameter"]),
            "rates": {
                label: fraction * ceiling
                for label, fraction in LOAD_FRACTIONS.items()
            },
            "grid": {
                str(row["parameter"]): {
                    "central_completion_rps": float(row["central_completion_rps"]),
                    "completed": int(row["completed"]),
                    "dropped": int(row["dropped"]),
                    "stable": row in stable,
                }
                for row in family
            },
        }
    (out / "capacity_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(json.dumps(selection, indent=2))


def run_static_calibration(out: Path, jobs: int, force: bool) -> None:
    target_map = targets()
    runs = [
        base.Run(
            "static_calibration",
            workload,
            label,
            rate,
            base.CALIBRATION_SEED,
            "routing_phi",
            phi,
            "1p2m",
            requests,
            target_map[workload.name],
        )
        for workload, label, rate, requests in conditions(out)
        for phi in PHIS
    ]
    rows = base.run_many(runs, out, jobs, force)
    require_hard_valid(rows, "static calibration")
    base.write_rows(out / "static_calibration.csv", rows)

    selection: dict[str, Any] = {"condition": {}}
    for workload, label, _, _ in conditions(out):
        key = f"{workload.name}:{label}"
        candidates = [
            row
            for row in rows
            if row["workload"] == workload.name and row["rate_label"] == label
            and static_eligible(row)
        ]
        if not candidates:
            raise RuntimeError(f"{key}: no stable static fraction")
        best = max(
            candidates,
            key=lambda row: (float(row["goodput"]), -float(row["parameter"])),
        )
        selection["condition"][key] = {
            "phi": float(best["parameter"]),
            "goodput": float(best["goodput"]),
        }

    regrets: dict[float, list[float]] = {}
    for phi in PHIS:
        values = []
        deployable_everywhere = True
        for workload, label, _, _ in conditions(out):
            key = f"{workload.name}:{label}"
            candidate = next(
                row
                for row in rows
                if row["workload"] == workload.name
                and row["rate_label"] == label
                and float(row["parameter"]) == phi
            )
            if not static_eligible(candidate):
                deployable_everywhere = False
                break
            values.append(
                selection["condition"][key]["goodput"]
                - float(candidate["goodput"])
            )
        if deployable_everywhere:
            regrets[phi] = values
    if not regrets:
        raise RuntimeError(
            "no static fraction completed every calibration condition without drops"
        )
    universal_phi = min(
        regrets,
        key=lambda phi: (
            max(regrets[phi]),
            statistics.fmean(regrets[phi]),
            phi,
        ),
    )
    selection["universal"] = {
        "phi": universal_phi,
        "worst_regret": max(regrets[universal_phi]),
        "mean_regret": statistics.fmean(regrets[universal_phi]),
    }
    (out / "static_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(json.dumps(selection, indent=2))


def static_selection(out: Path) -> dict[str, Any]:
    path = out / "static_selection.json"
    if not path.exists():
        raise SystemExit("static_selection.json missing; run static-calibration first")
    return json.loads(path.read_text())


def run_lambda_calibration(out: Path, jobs: int, force: bool) -> None:
    target_map = targets()
    runs = [
        base.Run(
            "lambda_calibration",
            workload,
            label,
            rate,
            base.CALIBRATION_SEED,
            "var_prefill",
            weight,
            "1p2m",
            requests,
            target_map[workload.name],
        )
        for workload, label, rate, requests in conditions(out)
        for weight in LAMBDAS
    ]
    rows = base.run_many(runs, out, jobs, force)
    require_hard_valid(rows, "lambda calibration")
    base.write_rows(out / "lambda_calibration.csv", rows)

    oracle = static_selection(out)["condition"]
    records = {}
    for weight in LAMBDAS:
        by_condition = {}
        for workload, label, _, _ in conditions(out):
            key = f"{workload.name}:{label}"
            row = next(
                row
                for row in rows
                if row["workload"] == workload.name
                and row["rate_label"] == label
                and float(row["parameter"]) == weight
            )
            regret = oracle[key]["goodput"] - float(row["goodput"])
            by_condition[key] = {
                "goodput": float(row["goodput"]),
                "regret": regret,
                "realized_phi": float(row["realized_phi"]),
                "dropped": int(row["dropped"]),
            }
        records[str(weight)] = {
            "worst_regret": max(v["regret"] for v in by_condition.values()),
            "mean_regret": statistics.fmean(
                v["regret"] for v in by_condition.values()
            ),
            "conditions": by_condition,
        }
    selected = min(
        LAMBDAS,
        key=lambda weight: (
            records[str(weight)]["worst_regret"],
            records[str(weight)]["mean_regret"],
            weight,
        ),
    )
    result = {"selected_lambda": selected, "grid": records}
    (out / "lambda_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def lambda_selection(out: Path) -> float:
    path = out / "lambda_selection.json"
    if not path.exists():
        raise SystemExit("lambda_selection.json missing; run lambda-calibration first")
    return float(json.loads(path.read_text())["selected_lambda"])


def run_policy_panel(
    out: Path,
    jobs: int,
    force: bool,
    *,
    phase: str,
    seeds: tuple[int, ...],
    output_name: str,
) -> None:
    target_map = targets()
    static = static_selection(out)
    selected_lambda = lambda_selection(out)
    universal_phi = float(static["universal"]["phi"])
    runs: list[base.Run] = []
    for workload, label, rate, requests in conditions(out):
        condition_phi = float(
            static["condition"][f"{workload.name}:{label}"]["phi"]
        )
        policies = [
            ("always", None),
            ("never", None),
            ("least_ttft", None),
            ("least_ttft_joint", None),
            ("dpp_joint", None),
            ("kairos", 0.5),
            ("dpvar", None),
            ("universal_routing_phi", universal_phi),
            ("var_prefill", selected_lambda),
            ("oracle_routing_phi", condition_phi),
        ]
        for seed in seeds:
            for policy, parameter in policies:
                runs.append(
                    base.Run(
                        phase,
                        workload,
                        label,
                        rate,
                        seed,
                        policy,
                        parameter,
                        "1p2m",
                        requests,
                        target_map[workload.name],
                    )
                )
    rows = base.run_many(runs, out, jobs, force)
    require_hard_valid(rows, phase)
    base.write_rows(out / output_name, rows)


def run_evaluation(out: Path, jobs: int, force: bool) -> None:
    run_policy_panel(
        out,
        jobs,
        force,
        phase="evaluation",
        seeds=base.HELD_OUT_SEEDS,
        output_name="evaluation.csv",
    )


def run_confirmation(out: Path, jobs: int, force: bool) -> None:
    run_policy_panel(
        out,
        jobs,
        force,
        phase="confirmation",
        seeds=CONFIRMATION_SEEDS,
        output_name="confirmation.csv",
    )


def analyze(
    out: Path,
    *,
    row_files: tuple[str, ...] = ("evaluation.csv",),
    summary_name: str = "evaluation_summary.csv",
    result_name: str = "minimax_result.json",
    report_name: str = "MINIMAX-RESULT.md",
    title: str = "Minimax var-prefill result",
    seed_description: str = "four held-out seeds",
) -> None:
    rows = [
        row
        for name in row_files
        for row in read_rows(out / name)
    ]
    if not rows:
        raise SystemExit(
            f"{', '.join(row_files)} missing; run the corresponding panel first"
        )
    static = static_selection(out)
    policies = sorted({row["policy"] for row in rows})
    summaries = []
    condition_means: dict[str, dict[str, float]] = {}
    for workload, label, _, _ in conditions(out):
        key = f"{workload.name}:{label}"
        condition_means[key] = {}
        for policy in policies:
            members = [
                row
                for row in rows
                if row["workload"] == workload.name
                and row["rate_label"] == label
                and row["policy"] == policy
            ]
            values = [float(row["goodput"]) for row in members]
            mean, low, high = base.ci95(values)
            phis = [
                float(row["realized_phi"])
                for row in members
                if row["realized_phi"] != ""
            ]
            drop_fractions = [
                int(row["dropped"]) / int(row["injected"])
                for row in members
                if int(row["injected"]) > 0
            ]
            condition_means[key][policy] = mean
            summaries.append(
                {
                    "workload": workload.name,
                    "rate_label": label,
                    "policy": policy,
                    "heldout_mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "heldout_min": min(values),
                    "drop_fraction_mean": statistics.fmean(drop_fractions),
                    "realized_phi_mean": (
                        statistics.fmean(phis) if phis else ""
                    ),
                    "calibration_oracle_phi": static["condition"][key]["phi"],
                }
            )

    references = {
        key: max(means.values()) for key, means in condition_means.items()
    }
    oracle_means = {
        key: means["oracle_routing_phi"]
        for key, means in condition_means.items()
    }
    for row in summaries:
        key = f"{row['workload']}:{row['rate_label']}"
        row["regret_to_best_observed"] = (
            references[key] - float(row["heldout_mean"])
        )
        row["regret_to_static_yardstick"] = (
            oracle_means[key] - float(row["heldout_mean"])
        )
        if row["realized_phi_mean"] == "":
            row["phi_distance"] = ""
        else:
            row["phi_distance"] = abs(
                float(row["realized_phi_mean"])
                - float(row["calibration_oracle_phi"])
            )
    base.write_rows(out / summary_name, summaries)

    deployable = [p for p in policies if p != "oracle_routing_phi"]
    worst = {}
    for policy in deployable:
        members = [row for row in summaries if row["policy"] == policy]
        worst_member = max(
            members, key=lambda row: float(row["regret_to_best_observed"])
        )
        phi_distances = [
            float(row["phi_distance"])
            for row in members
            if row["phi_distance"] != ""
        ]
        worst[policy] = {
            "worst_regret_to_best_observed": float(
                worst_member["regret_to_best_observed"]
            ),
            "worst_condition": (
                f"{worst_member['workload']}:{worst_member['rate_label']}"
            ),
            "worst_regret_to_static_yardstick": max(
                float(row["regret_to_static_yardstick"]) for row in members
            ),
            "mean_phi_distance": (
                statistics.fmean(phi_distances) if phi_distances else None
            ),
            "max_phi_distance": max(phi_distances) if phi_distances else None,
        }
    ranking = sorted(
        deployable,
        key=lambda policy: worst[policy]["worst_regret_to_best_observed"],
    )
    routing_preserving_set = [
        policy
        for policy in (
            "var_prefill",
            "always",
            "universal_routing_phi",
            "least_ttft",
            "never",
        )
        if policy in deployable
    ]
    routing_preserving_ranking = sorted(
        routing_preserving_set,
        key=lambda policy: worst[policy]["worst_regret_to_best_observed"],
    )

    first, second = ranking[:2]
    first_worst_condition = worst[first]["worst_condition"]
    first_workload, first_rate_label = first_worst_condition.split(":", 1)
    paired_by_policy: dict[str, dict[int, float]] = {}
    for policy in (first, second):
        paired_by_policy[policy] = {
            int(row["seed"]): float(row["goodput"])
            for row in rows
            if row["workload"] == first_workload
            and row["rate_label"] == first_rate_label
            and row["policy"] == policy
        }
    paired_seeds = sorted(
        set(paired_by_policy[first]) & set(paired_by_policy[second])
    )
    paired_differences = [
        paired_by_policy[first][seed] - paired_by_policy[second][seed]
        for seed in paired_seeds
    ]
    paired_mean, paired_low, paired_high = base.ci95(paired_differences)
    paired_top_gap = {
        "condition": first_worst_condition,
        "first_policy": first,
        "second_policy": second,
        "mean_goodput_difference": paired_mean,
        "ci_low": paired_low,
        "ci_high": paired_high,
        "seeds": paired_seeds,
    }
    result = {
        "selected_lambda": lambda_selection(out),
        "universal_phi": static["universal"]["phi"],
        "minimax_ranking": ranking,
        "routing_preserving_minimax_ranking": routing_preserving_ranking,
        "paired_top_gap": paired_top_gap,
        "policies": worst,
    }
    (out / result_name).write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        f"# {title}",
        "",
        f"Selected global lambda: `{result['selected_lambda']}`",
        "",
        "## Frozen calibration choices",
        "",
        "| condition | static oracle phi |",
        "|---|---:|",
    ]
    for key, value in static["condition"].items():
        lines.append(f"| `{key}` | {value['phi']:.1f} |")
    lines.extend(
        [
            "",
            f"Universal minimax static fraction: `{result['universal_phi']:.1f}`.",
            "",
            "## Held-out minimax ranking",
            "",
        "| rank | policy | worst regret | worst condition | static-yardstick worst regret | mean phi distance |",
        "|---:|---|---:|---|---:|---:|",
        ]
    )
    for rank, policy in enumerate(ranking, 1):
        value = worst[policy]
        phi = value["mean_phi_distance"]
        lines.append(
            f"| {rank} | `{policy}` | "
            f"{value['worst_regret_to_best_observed']:.3f} | "
            f"`{value['worst_condition']}` | "
            f"{value['worst_regret_to_static_yardstick']:.3f} | "
            f"{'n/a' if phi is None else f'{phi:.3f}'} |"
        )
    paired = result["paired_top_gap"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"`{first}` ranks first across the complete policy panel; "
            f"`{second}` ranks second. At their shared decisive condition "
            f"`{paired['condition']}`, the paired mean goodput difference "
            f"(`{first}` minus `{second}`) is "
            f"{paired['mean_goodput_difference']:.3f}, with a 95% t interval "
            f"[{paired['ci_low']:.3f}, {paired['ci_high']:.3f}] over "
            f"{len(paired['seeds'])} seeds. "
            "The interval crosses zero, so the observed top-two ordering is not "
            "statistically resolved by this seed count.",
            "",
            "Within the routing-preserving comparison set (policies that retain "
            "the existing decode and prefill routers), the minimax order is: "
            + ", ".join(f"`{policy}`" for policy in routing_preserving_ranking)
            + ".",
            "",
            f"`var_prefill` has mean fraction distance "
            f"{worst['var_prefill']['mean_phi_distance']:.3f}. Fraction distance "
            "is only a mechanism diagnostic: joint policies can achieve better "
            "goodput while using a fraction far from the static oracle because "
            "they also change pod placement.",
            "",
            "The condition-specific static fraction is selected on seed 42 and",
            f"frozen before these {seed_description}. Fraction distance is",
            "diagnostic; the primary ranking is worst goodput regret.",
            "",
        ]
    )
    (out / report_name).write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


def analyze_confirmation(out: Path) -> None:
    analyze(
        out,
        row_files=("confirmation.csv",),
        summary_name="confirmation_summary.csv",
        result_name="confirmation_result.json",
        report_name="CONFIRMATION-RESULT.md",
        title="Minimax var-prefill confirmation result",
        seed_description="eight confirmation seeds",
    )
    analyze(
        out,
        row_files=("evaluation.csv", "confirmation.csv"),
        summary_name="combined_summary.csv",
        result_name="combined_result.json",
        report_name="COMBINED-RESULT.md",
        title="Minimax var-prefill combined held-out result",
        seed_description="twelve held-out and confirmation seeds",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "capacity",
            "static-calibration",
            "lambda-calibration",
            "evaluation",
            "confirmation",
            "analyze",
            "confirmation-analyze",
            "all",
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not args.no_build and args.phase != "analyze":
        base.build_binary()

    phases = (
        ("capacity", run_capacity),
        ("static-calibration", run_static_calibration),
        ("lambda-calibration", run_lambda_calibration),
        ("evaluation", run_evaluation),
        ("confirmation", run_confirmation),
    )
    if args.phase == "all":
        for _, function in phases:
            function(out, args.jobs, args.force)
        analyze(out)
    elif args.phase == "analyze":
        analyze(out)
    elif args.phase == "confirmation-analyze":
        analyze_confirmation(out)
    else:
        dict(phases)[args.phase](out, args.jobs, args.force)


if __name__ == "__main__":
    main()
