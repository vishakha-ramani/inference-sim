#!/usr/bin/env python3
"""Trace which requests var-prefill selects, and why.

This is a mechanism diagnostic, not a tuning pass. It reruns frozen cells with
decision tracing enabled, joins each decision to the realized request outcome,
and decomposes local decisions into VaR-local versus stability-veto causes.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

import run_decisive_campaign as base
import run_minimax_var_prefill as minimax


SEED = 13
CELLS = (
    ("shared", "low"),
    ("shared", "medium"),
    ("shared", "near_high"),
    ("rag", "near_high"),
)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = q * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p10": percentile(values, 0.10),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_map(out: Path) -> dict[tuple[str, str], tuple[float, int]]:
    return {
        (workload.name, label): (rate, requests)
        for workload, label, rate, requests in minimax.conditions(out)
    }


def trace_cell(
    out: Path,
    workload_name: str,
    rate_label: str,
    *,
    force: bool,
) -> tuple[Path, Path]:
    workload = base.WORKLOADS[workload_name]
    rate, requests = condition_map(out)[(workload_name, rate_label)]
    target_map = minimax.targets()[workload_name]
    weight = minimax.lambda_selection(out)
    trace_dir = out / "request_selection"
    tag = f"{workload_name}_{rate_label}_var_prefill_s{SEED}_n{requests}"
    spec_path = trace_dir / "specs" / f"{tag}.yaml"
    metrics_path = trace_dir / "metrics" / f"{tag}.json"
    stdout_path = trace_dir / "stdout" / f"{tag}.txt"
    trace_path = trace_dir / "decisions" / f"{tag}.csv"
    outcome_path = trace_dir / "outcomes" / f"{tag}.csv"
    admission_path = trace_dir / "admission" / f"{tag}.csv"
    for path in (
        spec_path,
        metrics_path,
        stdout_path,
        trace_path,
        outcome_path,
        admission_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    base.make_spec(workload, rate, SEED, requests, spec_path, None)
    if (
        not force
        and metrics_path.exists()
        and stdout_path.exists()
        and trace_path.exists()
        and outcome_path.exists()
        and admission_path.exists()
    ):
        return metrics_path, trace_path

    command = [
        str(base.ROOT / "blis"),
        "run",
        "--model",
        base.MODEL,
        "--workload-spec",
        str(spec_path),
        "--num-requests",
        str(requests),
        *base.topology_args("1p2m"),
        *base.slo_args(target_map),
        *base.policy_args("var_prefill", target_map, weight, None),
        "--seed",
        str(SEED),
        "--trace-level",
        "decisions",
        "--edpp-decision-trace",
        str(trace_path),
        "--pd-outcome-trace",
        str(outcome_path),
        "--edpp-admission-trace",
        str(admission_path),
        "--metrics-path",
        str(metrics_path),
    ]
    proc = subprocess.run(command, cwd=base.ROOT, capture_output=True, text=True)
    stdout_path.write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"trace run failed ({tag}):\n{proc.stderr[-4000:]}\n"
            f"{proc.stdout[-4000:]}"
        )
    return metrics_path, trace_path


def decision_cause(row: dict[str, Any]) -> str:
    if row["skip_reason"]:
        return f"skip:{row['skip_reason']}"
    lhs = float(row["lhs"])
    rhs = float(row["rhs"])
    if lhs <= 0:
        return "var_favors_local"
    if lhs <= rhs:
        return "stability_veto"
    return "disaggregate"


def enrich(
    workload_name: str,
    rate_label: str,
    metrics_path: Path,
    trace_path: Path,
    targets: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    metrics = json.loads(metrics_path.read_text())
    outcomes = {row["requestID"]: row for row in metrics.get("requests", [])}
    trace_rows = read_csv(trace_path)
    enriched: list[dict[str, Any]] = []
    ordered = sorted(trace_rows, key=lambda row: (int(row["clock"]), row["request_id"]))
    count = len(ordered)
    for index, row in enumerate(ordered):
        outcome = outcomes.get(row["request_id"], {})
        slo_class = row["class"] or outcome.get("slo_class", "")
        target = targets.get(slo_class, {})
        completed = bool(outcome)
        good = (
            completed
            and float(outcome.get("ttft_ms", math.inf))
            <= float(target.get("ttft_ms", math.inf))
            and float(outcome.get("itl_ms", math.inf))
            <= float(target.get("itl_ms", math.inf))
            and float(outcome.get("e2e_ms", math.inf))
            <= float(target.get("e2e_ms", math.inf))
        )
        full_prompt = int(outcome.get("num_prefill_tokens", 0))
        ap = int(row["ap"])
        lhs = float(row["lhs"])
        rhs = float(row["rhs"])
        ttft_p = float(row["ttft_p"])
        ttft_d = float(row["ttft_d"])
        enriched.append(
            {
                "workload": workload_name,
                "rate_label": rate_label,
                "request_id": row["request_id"],
                "clock": int(row["clock"]),
                "time_decile": min(9, int(10 * index / max(1, count))),
                "class": slo_class,
                "tenant_id": outcome.get("tenant_id", ""),
                "completed": completed,
                "good": good,
                "disaggregate": row["disaggregate"].lower() == "true",
                "cause": decision_cause(row),
                "shadow_least_ttft_disaggregate": (
                    not row["skip_reason"] and ttft_p < ttft_d
                ),
                "full_prompt_tokens": full_prompt,
                "ap": ap,
                "cached_tokens": max(0, full_prompt - ap),
                "cache_fraction": (
                    max(0, full_prompt - ap) / full_prompt if full_prompt else 0.0
                ),
                "output_tokens": int(outcome.get("num_decode_tokens", 0)),
                "qp": float(row["qp"]),
                "qd": float(row["qd"]),
                "var_benefit": lhs,
                "stability_charge": rhs,
                "margin": lhs - rhs,
                "predicted_ttft_disagg_us": ttft_p,
                "predicted_ttft_local_us": ttft_d,
                "ttft_p_minus_d": ttft_p - ttft_d,
                "var_local_decode": float(row["var_local_decode"]),
                "var_local_colloc_prefill": float(
                    row["var_local_colloc_prefill"]
                ),
                "var_disagg_decode": float(row["var_disagg_decode"]),
                "var_disagg_colloc_prefill": float(
                    row["var_disagg_colloc_prefill"]
                ),
                "var_disagg_prefill_pool": float(
                    row["var_disagg_prefill_pool"]
                ),
                "ttft_ms": outcome.get("ttft_ms", ""),
                "itl_ms": outcome.get("itl_ms", ""),
                "e2e_ms": outcome.get("e2e_ms", ""),
            }
        )
    return enriched


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "share": None,
        "good_rate": (
            sum(bool(row["good"]) for row in rows) / len(rows) if rows else None
        ),
        "disaggregate_rate": (
            sum(bool(row["disaggregate"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "shadow_least_ttft_rate": (
            sum(bool(row["shadow_least_ttft_disaggregate"]) for row in rows)
            / len(rows)
            if rows
            else None
        ),
        "ap": distribution([float(row["ap"]) for row in rows]),
        "cache_fraction": distribution(
            [float(row["cache_fraction"]) for row in rows]
        ),
        "qp": distribution([float(row["qp"]) for row in rows]),
        "qd": distribution([float(row["qd"]) for row in rows]),
        "var_benefit": distribution(
            [float(row["var_benefit"]) for row in rows]
        ),
        "stability_charge": distribution(
            [float(row["stability_charge"]) for row in rows]
        ),
        "margin": distribution([float(row["margin"]) for row in rows]),
        "ttft_p_minus_d": distribution(
            [float(row["ttft_p_minus_d"]) for row in rows]
        ),
        "var_local_decode": distribution(
            [float(row["var_local_decode"]) for row in rows]
        ),
        "var_local_colloc_prefill": distribution(
            [float(row["var_local_colloc_prefill"]) for row in rows]
        ),
        "var_disagg_decode": distribution(
            [float(row["var_disagg_decode"]) for row in rows]
        ),
        "var_disagg_colloc_prefill": distribution(
            [float(row["var_disagg_colloc_prefill"]) for row in rows]
        ),
        "var_disagg_prefill_pool": distribution(
            [float(row["var_disagg_prefill_pool"]) for row in rows]
        ),
    }


def analyze_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    by_cause: dict[str, Any] = {}
    for cause in sorted({str(row["cause"]) for row in rows}):
        members = [row for row in rows if row["cause"] == cause]
        summary = summarize_group(members)
        summary["share"] = len(members) / total if total else None
        by_cause[cause] = summary

    by_decile: dict[str, Any] = {}
    for decile in range(10):
        members = [row for row in rows if int(row["time_decile"]) == decile]
        by_decile[str(decile)] = summarize_group(members)

    disagreements = {
        "var_local_least_ttft_remote": summarize_group(
            [
                row
                for row in rows
                if not row["disaggregate"]
                and row["shadow_least_ttft_disaggregate"]
            ]
        ),
        "var_remote_least_ttft_local": summarize_group(
            [
                row
                for row in rows
                if row["disaggregate"]
                and not row["shadow_least_ttft_disaggregate"]
            ]
        ),
    }
    return {
        "requests": total,
        "overall": summarize_group(rows),
        "by_cause": by_cause,
        "by_time_decile": by_decile,
        "shadow_disagreements": disagreements,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=minimax.DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()

    all_rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "seed": SEED,
        "lambda": minimax.lambda_selection(out),
        "cells": {},
        "limitation": (
            "Shadow least-TTFT is evaluated on the var-prefill trajectory. "
            "It is a same-state mechanism comparison, not a realized "
            "per-request counterfactual-goodput oracle."
        ),
    }
    for workload_name, rate_label in CELLS:
        metrics_path, trace_path = trace_cell(
            out, workload_name, rate_label, force=args.force
        )
        rows = enrich(
            workload_name,
            rate_label,
            metrics_path,
            trace_path,
            minimax.targets()[workload_name],
        )
        all_rows.extend(rows)
        result["cells"][f"{workload_name}:{rate_label}"] = analyze_cell(rows)

    diagnostic_dir = out / "request_selection"
    write_csv(diagnostic_dir / "request_selection_enriched.csv", all_rows)
    (diagnostic_dir / "request_selection_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    lines = [
        "# Request-selection diagnosis",
        "",
        f"Frozen seed: `{SEED}`; lambda: `{result['lambda']}`.",
        "",
        "The static fraction is not used as a request-label oracle. The table",
        "decomposes the actual reduced-policy decisions by their immediate cause.",
        "",
        "| condition | remote | VaR-local | stability veto | cache/other skip | shadow least-TTFT remote | goodput |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, cell in result["cells"].items():
        causes = cell["by_cause"]
        remote = causes.get("disaggregate", {}).get("share")
        var_local = causes.get("var_favors_local", {}).get("share")
        veto = causes.get("stability_veto", {}).get("share")
        skip = sum(
            value["share"]
            for cause, value in causes.items()
            if cause.startswith("skip:")
        )
        overall = cell["overall"]
        lines.append(
            f"| `{key}` | {fmt(remote)} | {fmt(var_local)} | {fmt(veto)} | "
            f"{fmt(skip)} | {fmt(overall['shadow_least_ttft_rate'])} | "
            f"{fmt(overall['good_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Time evolution",
            "",
            "| condition | first-decile remote | last-decile remote | first-decile VaR benefit median | last-decile VaR benefit median | first-decile stability median | last-decile stability median |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, cell in result["cells"].items():
        first = cell["by_time_decile"]["0"]
        last = cell["by_time_decile"]["9"]
        lines.append(
            f"| `{key}` | {fmt(first['disaggregate_rate'])} | "
            f"{fmt(last['disaggregate_rate'])} | "
            f"{fmt(first['var_benefit']['median'])} | "
            f"{fmt(last['var_benefit']['median'])} | "
            f"{fmt(first['stability_charge']['median'])} | "
            f"{fmt(last['stability_charge']['median'])} |"
        )
    lines.extend(
        [
            "",
            "Shadow least-TTFT is evaluated on the `var_prefill` trajectory.",
            "It is a same-state mechanism comparison, not a realized per-request",
            "counterfactual-goodput oracle.",
            "",
        ]
    )
    (diagnostic_dir / "REQUEST-SELECTION.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
