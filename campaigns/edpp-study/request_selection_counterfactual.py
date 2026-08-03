#!/usr/bin/env python3
"""Routing-controlled, one-request counterfactuals for var-prefill.

This diagnostic isolates the P/D selection question:

* replay the observed var-prefill P/D decision for every request;
* by default, pin the decode and prefill placements chosen by the normal
  routers in the observed run, removing route selection as a confound;
* flip exactly one sampled request between local and disaggregated;
* measure the resulting change in total SLO attainment.

The result is local, one-step hindsight regret on the realized trajectory.  It
is not a global request-label oracle and is not used to tune policy parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import run_decisive_campaign as base
import run_minimax_var_prefill as minimax
import trace_request_selection as selection


SAMPLE_SEED = 20260730
SAMPLES_PER_STRATUM = 20
GOODPUT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Stratum:
    workload: str
    rate_label: str
    name: str
    predicate: Callable[[dict[str, Any]], bool]

    @property
    def cell(self) -> str:
        return f"{self.workload}:{self.rate_label}"


STRATA = (
    Stratum(
        "shared",
        "low",
        "stability_veto",
        lambda row: row["cause"] == "stability_veto",
    ),
    Stratum(
        "shared",
        "low",
        "disaggregate",
        lambda row: row["cause"] == "disaggregate",
    ),
    Stratum(
        "shared",
        "medium",
        "stability_veto",
        lambda row: row["cause"] == "stability_veto",
    ),
    Stratum(
        "shared",
        "medium",
        "disaggregate",
        lambda row: row["cause"] == "disaggregate",
    ),
    Stratum(
        "shared",
        "near_high",
        "stability_veto",
        lambda row: row["cause"] == "stability_veto",
    ),
    Stratum(
        "shared",
        "near_high",
        "disaggregate",
        lambda row: row["cause"] == "disaggregate",
    ),
    Stratum(
        "shared",
        "near_high",
        "var_favors_local",
        lambda row: row["cause"] == "var_favors_local",
    ),
    Stratum(
        "rag",
        "near_high",
        "var_local_least_ttft_remote",
        lambda row: (
            not row["disaggregate"]
            and row["shadow_least_ttft_disaggregate"]
        ),
    ),
    Stratum(
        "rag",
        "near_high",
        "var_remote_least_ttft_local",
        lambda row: (
            row["disaggregate"]
            and not row["shadow_least_ttft_disaggregate"]
        ),
    ),
)


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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def load_enriched(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv(path):
        row: dict[str, Any] = dict(raw)
        for field in (
            "completed",
            "good",
            "disaggregate",
            "shadow_least_ttft_disaggregate",
        ):
            row[field] = parse_bool(row[field])
        for field in (
            "clock",
            "time_decile",
            "full_prompt_tokens",
            "ap",
            "cached_tokens",
            "output_tokens",
        ):
            row[field] = int(row[field])
        for field in (
            "cache_fraction",
            "qp",
            "qd",
            "var_benefit",
            "stability_charge",
            "margin",
            "predicted_ttft_disagg_us",
            "predicted_ttft_local_us",
            "ttft_p_minus_d",
        ):
            row[field] = float(row[field])
        rows.append(row)
    return rows


def plan_rows(
    rows: list[dict[str, Any]],
    outcomes: dict[str, dict[str, str]],
    routing_control: str,
) -> list[dict[str, str]]:
    """Build the baseline plan under the requested routing control."""
    if routing_control == "automatic":
        return [
            {
                "request_id": str(row["request_id"]),
                "decode_instance": "",
                "prefill_instance": (
                    "auto" if row["disaggregate"] else "local"
                ),
            }
            for row in sorted(
                rows, key=lambda item: (item["clock"], item["request_id"])
            )
        ]
    return [
        {
            "request_id": str(row["request_id"]),
            "decode_instance": outcomes[str(row["request_id"])][
                "decode_instance"
            ],
            "prefill_instance": (
                outcomes[str(row["request_id"])]["prefill_instance"]
                if row["disaggregate"]
                else "local"
            ),
        }
        for row in sorted(rows, key=lambda item: (item["clock"], item["request_id"]))
    ]


def write_plan(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["request_id", "decode_instance", "prefill_instance"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plan_digest(rows: list[dict[str, str]]) -> str:
    canonical = "\n".join(
        f"{row['request_id']},{row['decode_instance']},{row['prefill_instance']}"
        for row in rows
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def metric_goodput(path: Path) -> float:
    metrics = json.loads(path.read_text())
    if isinstance(metrics, list):
        values = [float(item.get("slo_attainment", 0.0)) for item in metrics]
        return statistics.fmean(values) if values else 0.0
    return float(metrics.get("slo_attainment", 0.0))


def request_good(
    metrics: dict[str, Any],
    request_id: str,
    targets: dict[str, dict[str, float]],
) -> tuple[bool, bool]:
    for request in metrics.get("requests", []):
        if request.get("requestID") != request_id:
            continue
        slo_class = str(request.get("slo_class", ""))
        target = targets.get(slo_class)
        if target is None:
            return True, False
        good = (
            float(request.get("ttft_ms", math.inf))
            <= float(target["ttft_ms"])
            and float(request.get("itl_ms", math.inf))
            <= float(target["itl_ms"])
            and float(request.get("e2e_ms", math.inf))
            <= float(target["e2e_ms"])
        )
        return True, good
    return False, False


def request_ttft_ms(metrics: dict[str, Any], request_id: str) -> float | None:
    for request in metrics.get("requests", []):
        if request.get("requestID") == request_id:
            return float(request.get("ttft_ms", math.nan))
    return None


def stratum_sample(
    candidates: list[dict[str, Any]],
    count: int,
    *,
    seed_key: str,
) -> list[dict[str, Any]]:
    """Sample across time deciles so a stratum is not only early or late."""
    rng = random.Random(f"{SAMPLE_SEED}:{seed_key}")
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in candidates:
        buckets.setdefault(int(row["time_decile"]), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: str(row["request_id"]))
        rng.shuffle(bucket)

    chosen: list[dict[str, Any]] = []
    active = sorted(buckets)
    while active and len(chosen) < min(count, len(candidates)):
        rng.shuffle(active)
        next_active: list[int] = []
        for decile in active:
            bucket = buckets[decile]
            if bucket and len(chosen) < count:
                chosen.append(bucket.pop())
            if bucket:
                next_active.append(decile)
        active = next_active
    return chosen


def cell_tag(workload: str, rate_label: str, requests: int) -> str:
    return f"{workload}_{rate_label}_s{selection.SEED}_n{requests}"


def run_command(
    *,
    spec_path: Path,
    requests: int,
    targets: dict[str, dict[str, float]],
    plan_path: Path,
    metrics_path: Path,
    stdout_path: Path,
    force: bool,
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and metrics_path.exists() and stdout_path.exists():
        # Parse before reuse so truncated artifacts fail loudly.
        metric_goodput(metrics_path)
        return
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
        *base.slo_args(targets),
        "--pd-plan",
        str(plan_path),
        "--seed",
        str(selection.SEED),
        "--metrics-path",
        str(metrics_path),
    ]
    proc = subprocess.run(command, cwd=base.ROOT, capture_output=True, text=True)
    stdout_path.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"counterfactual run failed for {plan_path.name}:\n"
            f"{proc.stderr[-4000:]}\n{proc.stdout[-4000:]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=minimax.DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument(
        "--samples-per-stratum", type=int, default=SAMPLES_PER_STRATUM
    )
    parser.add_argument(
        "--routing-control",
        choices=("pinned", "automatic"),
        default="pinned",
        help=(
            "pinned replays observed placements; automatic leaves both routers "
            "live. Pinned is the cleaner request-selection diagnostic."
        ),
    )
    args = parser.parse_args()
    out = args.out.resolve()
    root = out / "request_selection"
    diagnostic = root / f"counterfactual_{args.routing_control}"
    enriched_path = root / "request_selection_enriched.csv"
    if not enriched_path.exists():
        raise SystemExit(
            f"{enriched_path} is missing; run trace_request_selection.py first"
        )

    all_rows = load_enriched(enriched_path)
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        by_cell.setdefault((row["workload"], row["rate_label"]), []).append(row)

    targets_by_workload = minimax.targets()
    baseline_info: dict[tuple[str, str], dict[str, Any]] = {}
    for workload, rate_label in sorted({(s.workload, s.rate_label) for s in STRATA}):
        rows = by_cell[(workload, rate_label)]
        requests = len(rows)
        tag = cell_tag(workload, rate_label, requests)
        outcome_path = root / "outcomes" / (
            f"{workload}_{rate_label}_var_prefill_"
            f"s{selection.SEED}_n{requests}.csv"
        )
        outcomes = {
            row["request_id"]: row for row in read_csv(outcome_path)
        }
        missing_outcomes = sorted(
            str(row["request_id"])
            for row in rows
            if str(row["request_id"]) not in outcomes
        )
        if missing_outcomes:
            raise RuntimeError(
                f"{workload}:{rate_label} has {len(missing_outcomes)} "
                "decisions without routing outcomes; cannot build a total plan"
            )
        plan = plan_rows(rows, outcomes, args.routing_control)
        remote_prefills = sorted(
            {
                outcome["prefill_instance"]
                for outcome in outcomes.values()
                if parse_bool(outcome["disaggregated"])
            }
        )
        if args.routing_control == "pinned" and len(remote_prefills) != 1:
            raise RuntimeError(
                f"{workload}:{rate_label} expected exactly one prefill "
                f"instance, found {remote_prefills}"
            )
        plan_path = diagnostic / "plans" / f"{tag}_baseline.csv"
        write_plan(plan_path, plan)
        spec_path = root / "specs" / (
            f"{workload}_{rate_label}_var_prefill_"
            f"s{selection.SEED}_n{requests}.yaml"
        )
        traced_metrics_path = root / "metrics" / (
            f"{workload}_{rate_label}_var_prefill_"
            f"s{selection.SEED}_n{requests}.json"
        )
        replay_metrics_path = diagnostic / "baseline" / f"{tag}.json"
        replay_stdout_path = diagnostic / "stdout" / f"{tag}_baseline.txt"
        run_command(
            spec_path=spec_path,
            requests=requests,
            targets=targets_by_workload[workload],
            plan_path=plan_path,
            metrics_path=replay_metrics_path,
            stdout_path=replay_stdout_path,
            force=args.force,
        )
        traced_goodput = metric_goodput(traced_metrics_path)
        replay_goodput = metric_goodput(replay_metrics_path)
        if abs(traced_goodput - replay_goodput) > GOODPUT_TOLERANCE:
            raise RuntimeError(
                "SELF-CONSISTENCY GATE FAILED for "
                f"{workload}:{rate_label}: traced={traced_goodput}, "
                f"routing-preserving replay={replay_goodput}. "
                "Do not interpret one-request deviations."
            )
        baseline_info[(workload, rate_label)] = {
            "requests": requests,
            "tag": tag,
            "rows": rows,
            "plan": plan,
            "plan_path": plan_path,
            "plan_digest": plan_digest(plan),
            "remote_prefill_instance": (
                remote_prefills[0] if len(remote_prefills) == 1 else "auto"
            ),
            "spec_path": spec_path,
            "metrics_path": replay_metrics_path,
            "goodput": replay_goodput,
        }
        print(
            f"self-consistency {workload}:{rate_label}: "
            f"{traced_goodput:.12f} == {replay_goodput:.12f}",
            flush=True,
        )

    samples: list[dict[str, Any]] = []
    for stratum in STRATA:
        rows = by_cell[(stratum.workload, stratum.rate_label)]
        candidates = [row for row in rows if stratum.predicate(row)]
        chosen = stratum_sample(
            candidates,
            args.samples_per_stratum,
            seed_key=f"{stratum.cell}:{stratum.name}",
        )
        if not chosen:
            raise RuntimeError(f"stratum {stratum.cell}:{stratum.name} is empty")
        for row in chosen:
            samples.append(
                {
                    "stratum": stratum.name,
                    "candidate_count": len(candidates),
                    **row,
                }
            )

    work: list[dict[str, Any]] = []
    for sample in samples:
        cell = (sample["workload"], sample["rate_label"])
        info = baseline_info[cell]
        request_id = str(sample["request_id"])
        flipped_to = (
            "local"
            if sample["disaggregate"]
            else info["remote_prefill_instance"]
        )
        flipped_action = "local" if sample["disaggregate"] else "remote"
        safe_request_id = request_id.replace("/", "-")
        tag = (
            f"{info['tag']}__{sample['stratum']}__{safe_request_id}"
            f"__to-{flipped_action}"
        )
        dev_plan = [dict(row) for row in info["plan"]]
        matches = 0
        for row in dev_plan:
            if row["request_id"] == request_id:
                row["prefill_instance"] = flipped_to
                matches += 1
        if matches != 1:
            raise RuntimeError(
                f"expected one plan row for {request_id}, found {matches}"
            )
        plan_path = diagnostic / "plans" / f"{tag}.csv"
        write_plan(plan_path, dev_plan)
        work.append(
            {
                "sample": sample,
                "info": info,
                "flipped_to": flipped_to,
                "flipped_action": flipped_action,
                "plan_path": plan_path,
                "metrics_path": diagnostic / "deviations" / f"{tag}.json",
                "stdout_path": diagnostic / "stdout" / f"{tag}.txt",
            }
        )

    def execute(item: dict[str, Any]) -> dict[str, Any]:
        sample = item["sample"]
        info = item["info"]
        run_command(
            spec_path=info["spec_path"],
            requests=info["requests"],
            targets=targets_by_workload[sample["workload"]],
            plan_path=item["plan_path"],
            metrics_path=item["metrics_path"],
            stdout_path=item["stdout_path"],
            force=args.force,
        )
        return item

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(execute, item) for item in work]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(work):
                print(
                    f"completed {completed}/{len(work)} one-request deviations",
                    flush=True,
                )

    result_rows: list[dict[str, Any]] = []
    for item in work:
        sample = item["sample"]
        info = item["info"]
        baseline_metrics = json.loads(info["metrics_path"].read_text())
        dev_metrics = json.loads(item["metrics_path"].read_text())
        baseline_goodput = float(info["goodput"])
        dev_goodput = metric_goodput(item["metrics_path"])
        delta = dev_goodput - baseline_goodput
        base_completed, base_request_good = request_good(
            baseline_metrics,
            sample["request_id"],
            targets_by_workload[sample["workload"]],
        )
        dev_completed, dev_request_good = request_good(
            dev_metrics,
            sample["request_id"],
            targets_by_workload[sample["workload"]],
        )
        baseline_ttft_ms = request_ttft_ms(
            baseline_metrics, sample["request_id"]
        )
        deviated_ttft_ms = request_ttft_ms(
            dev_metrics, sample["request_id"]
        )
        if sample["disaggregate"]:
            realized_remote_ttft_ms = baseline_ttft_ms
            realized_local_ttft_ms = deviated_ttft_ms
        else:
            realized_local_ttft_ms = baseline_ttft_ms
            realized_remote_ttft_ms = deviated_ttft_ms
        realized_ttft_remote_minus_local_ms = (
            realized_remote_ttft_ms - realized_local_ttft_ms
            if realized_remote_ttft_ms is not None
            and realized_local_ttft_ms is not None
            else None
        )
        predicted_remote_faster = sample["ttft_p_minus_d"] < 0
        realized_remote_faster = (
            realized_ttft_remote_minus_local_ms < 0
            if realized_ttft_remote_minus_local_ms is not None
            else None
        )
        result_rows.append(
            {
                "workload": sample["workload"],
                "rate_label": sample["rate_label"],
                "stratum": sample["stratum"],
                "candidate_count": sample["candidate_count"],
                "request_id": sample["request_id"],
                "time_decile": sample["time_decile"],
                "class": sample["class"],
                "baseline_action": (
                    "remote" if sample["disaggregate"] else "local"
                ),
                "flipped_to": item["flipped_action"],
                "baseline_goodput": baseline_goodput,
                "deviated_goodput": dev_goodput,
                "goodput_delta": delta,
                "beneficial_flip": delta > GOODPUT_TOLERANCE,
                "harmful_flip": delta < -GOODPUT_TOLERANCE,
                "baseline_request_completed": base_completed,
                "baseline_request_good": base_request_good,
                "deviated_request_completed": dev_completed,
                "deviated_request_good": dev_request_good,
                "request_good_delta": int(dev_request_good)
                - int(base_request_good),
                "predicted_local_ttft_ms": (
                    sample["predicted_ttft_local_us"] / 1000.0
                ),
                "predicted_remote_ttft_ms": (
                    sample["predicted_ttft_disagg_us"] / 1000.0
                ),
                "predicted_ttft_remote_minus_local_ms": (
                    sample["ttft_p_minus_d"] / 1000.0
                ),
                "realized_local_ttft_ms": realized_local_ttft_ms,
                "realized_remote_ttft_ms": realized_remote_ttft_ms,
                "realized_ttft_remote_minus_local_ms": (
                    realized_ttft_remote_minus_local_ms
                ),
                "predicted_remote_faster": predicted_remote_faster,
                "realized_remote_faster": realized_remote_faster,
                "ttft_order_correct": (
                    predicted_remote_faster == realized_remote_faster
                    if realized_remote_faster is not None
                    else ""
                ),
                "ap": sample["ap"],
                "cache_fraction": sample["cache_fraction"],
                "qp": sample["qp"],
                "qd": sample["qd"],
                "var_benefit": sample["var_benefit"],
                "stability_charge": sample["stability_charge"],
                "margin": sample["margin"],
                "ttft_p_minus_d": sample["ttft_p_minus_d"],
                "metrics_path": str(item["metrics_path"].relative_to(base.ROOT)),
            }
        )

    result_rows.sort(
        key=lambda row: (
            row["workload"],
            row["rate_label"],
            row["stratum"],
            row["time_decile"],
            row["request_id"],
        )
    )
    write_csv(diagnostic / "counterfactual_requests.csv", result_rows)

    summaries: list[dict[str, Any]] = []
    keys = sorted(
        {
            (row["workload"], row["rate_label"], row["stratum"])
            for row in result_rows
        }
    )
    for workload, rate_label, stratum in keys:
        members = [
            row
            for row in result_rows
            if (
                row["workload"],
                row["rate_label"],
                row["stratum"],
            )
            == (workload, rate_label, stratum)
        ]
        deltas = [float(row["goodput_delta"]) for row in members]
        summaries.append(
            {
                "workload": workload,
                "rate_label": rate_label,
                "stratum": stratum,
                "candidate_count": members[0]["candidate_count"],
                "sample_count": len(members),
                "beneficial_count": sum(
                    bool(row["beneficial_flip"]) for row in members
                ),
                "harmful_count": sum(
                    bool(row["harmful_flip"]) for row in members
                ),
                "neutral_count": sum(
                    not row["beneficial_flip"] and not row["harmful_flip"]
                    for row in members
                ),
                "beneficial_fraction": sum(delta > GOODPUT_TOLERANCE for delta in deltas)
                / len(deltas),
                "mean_goodput_delta": statistics.fmean(deltas),
                "median_goodput_delta": statistics.median(deltas),
                "min_goodput_delta": min(deltas),
                "max_goodput_delta": max(deltas),
                "request_improved_count": sum(
                    int(row["request_good_delta"]) > 0 for row in members
                ),
                "request_harmed_count": sum(
                    int(row["request_good_delta"]) < 0 for row in members
                ),
                "ttft_order_observed": sum(
                    row["ttft_order_correct"] != "" for row in members
                ),
                "ttft_order_correct_count": sum(
                    row["ttft_order_correct"] is True for row in members
                ),
                "predicted_remote_faster_count": sum(
                    bool(row["predicted_remote_faster"]) for row in members
                ),
                "realized_remote_faster_count": sum(
                    row["realized_remote_faster"] is True for row in members
                ),
            }
        )
    write_csv(diagnostic / "counterfactual_summary.csv", summaries)

    report = {
        "diagnostic": (
            f"{args.routing_control}-routing one-request P/D deviations; "
            "local hindsight regret, not a global request-label oracle"
        ),
        "routing_control": args.routing_control,
        "seed": selection.SEED,
        "sample_seed": SAMPLE_SEED,
        "samples_per_stratum": args.samples_per_stratum,
        "goodput_tolerance": GOODPUT_TOLERANCE,
        "baseline_self_consistency": {
            f"{workload}:{rate_label}": {
                "goodput": info["goodput"],
                "requests": info["requests"],
                "plan_sha256": info["plan_digest"],
            }
            for (workload, rate_label), info in sorted(baseline_info.items())
        },
        "summaries": summaries,
    }
    (diagnostic / "counterfactual_result.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    lines = [
        "# Routing-controlled request counterfactuals",
        "",
        (
            "Each run pins the observed P/D choice for every request and flips "
            "exactly one sampled P/D choice."
        ),
        (
            "Decode and prefill placement is "
            + (
                "pinned to the original normal-router outcome."
                if args.routing_control == "pinned"
                else "left to the normal routers."
            )
        ),
        "The result measures local one-step hindsight regret, not a globally",
        "optimal request labeling.",
        "",
        "All baseline replays passed the exact goodput self-consistency gate.",
        "",
        "| condition | sampled decision | candidates | n | flip helps | harms | neutral | mean Δ goodput | own request improves | TTFT order correct | predicted/realized remote-faster |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['workload']}:{row['rate_label']}` | "
            f"`{row['stratum']}` | {row['candidate_count']} | "
            f"{row['sample_count']} | {row['beneficial_count']} | "
            f"{row['harmful_count']} | {row['neutral_count']} | "
            f"{row['mean_goodput_delta']:+.6f} | "
            f"{row['request_improved_count']} | "
            f"{row['ttft_order_correct_count']}/{row['ttft_order_observed']} | "
            f"{row['predicted_remote_faster_count']}/"
            f"{row['realized_remote_faster_count']} |"
        )
    lines.extend(
        [
            "",
            "A positive delta means that changing only that request's P/D choice",
            "improved aggregate SLO attainment after the system evolved normally.",
            "A zero delta does not prove the original choice was uniquely optimal;",
            "the goodput metric is discrete and may be insensitive to small latency",
            "changes away from an SLO boundary.",
            "",
        ]
    )
    (diagnostic / "REQUEST-COUNTERFACTUAL.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
