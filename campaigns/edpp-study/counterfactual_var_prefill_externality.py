#!/usr/bin/env python3
"""Pinned one-request counterfactuals for exact prefill VaR decision changes.

For RAG medium and near-high at the frozen seed:

* replay the exact-prefill VaR trajectory with all observed routes pinned;
* sample requests whose action differs from the legacy-VaR trajectory;
* flip exactly one sampled P/D action;
* measure total-goodput and own-request changes.

This is local hindsight regret on the exact-VaR trajectory, not a global
request-label oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import request_selection_counterfactual as cf
import run_decisive_campaign as base
import run_minimax_var_prefill as minimax
import trace_var_prefill_externality as trace


SAMPLES_PER_DIRECTION = 20
CELLS = ("medium", "near_high")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=minimax.DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument(
        "--samples-per-direction",
        type=int,
        default=SAMPLES_PER_DIRECTION,
    )
    args = parser.parse_args()
    out = args.out.resolve()
    trace_root = out / "var_prefill_externality_trace"
    diagnostic = trace_root / "counterfactual_exact_pinned"
    changed_path = trace_root / "changed_decisions.csv"
    if not changed_path.exists():
        raise SystemExit(
            f"{changed_path} missing; run trace_var_prefill_externality.py first"
        )
    changed = read_csv(changed_path)
    targets = minimax.targets()["rag"]

    condition = {
        label: requests
        for workload, label, _, requests in minimax.conditions(out)
        if workload.name == "rag"
    }
    baseline_info: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    for rate_label in CELLS:
        requests = condition[rate_label]
        tag = (
            f"rag_{rate_label}_var_prefill_nostability_exactvar_"
            f"s{trace.SEED}_n{requests}"
        )
        spec_path = trace_root / "specs" / f"{tag}.yaml"
        metrics_path = trace_root / "metrics" / f"{tag}.json"
        decisions_path = trace_root / "decisions" / f"{tag}.csv"
        outcomes_path = trace_root / "outcomes" / f"{tag}.csv"
        decisions = trace.parse_decisions(decisions_path)
        ordered = sorted(
            decisions.values(),
            key=lambda row: (int(row["clock"]), str(row["request_id"])),
        )
        for index, row in enumerate(ordered):
            row["time_decile"] = min(9, int(10 * index / max(1, len(ordered))))
        outcomes = {
            row["request_id"]: row for row in read_csv(outcomes_path)
        }
        if set(decisions) - set(outcomes):
            raise RuntimeError(
                f"rag:{rate_label}: exact trace has decisions without outcomes"
            )
        plan = cf.plan_rows(ordered, outcomes, "pinned")
        remote_prefills = sorted(
            {
                row["prefill_instance"]
                for row in outcomes.values()
                if cf.parse_bool(row["disaggregated"])
            }
        )
        if len(remote_prefills) != 1:
            raise RuntimeError(
                f"rag:{rate_label}: expected one remote prefill instance, "
                f"found {remote_prefills}"
            )
        plan_path = diagnostic / "plans" / f"{tag}_baseline.csv"
        cf.write_plan(plan_path, plan)
        replay_metrics = diagnostic / "baseline" / f"{tag}.json"
        replay_stdout = diagnostic / "stdout" / f"{tag}_baseline.txt"
        cf.run_command(
            spec_path=spec_path,
            requests=requests,
            targets=targets,
            plan_path=plan_path,
            metrics_path=replay_metrics,
            stdout_path=replay_stdout,
            force=args.force,
        )
        traced_goodput = cf.metric_goodput(metrics_path)
        replay_goodput = cf.metric_goodput(replay_metrics)
        if abs(traced_goodput - replay_goodput) > cf.GOODPUT_TOLERANCE:
            raise RuntimeError(
                f"self-consistency failed rag:{rate_label}: "
                f"trace={traced_goodput}, replay={replay_goodput}"
            )
        print(
            f"self-consistency rag:{rate_label}: "
            f"{traced_goodput:.12f} == {replay_goodput:.12f}",
            flush=True,
        )
        baseline_info[rate_label] = {
            "requests": requests,
            "spec_path": spec_path,
            "metrics_path": replay_metrics,
            "goodput": replay_goodput,
            "plan": plan,
            "remote_prefill": remote_prefills[0],
            "tag": tag,
        }

        changed_ids = {
            row["request_id"]: row
            for row in changed
            if row["rate_label"] == rate_label
        }
        for direction in ("local_to_remote", "remote_to_local"):
            candidates = [
                {
                    **decisions[request_id],
                    "time_decile": decisions[request_id]["time_decile"],
                    "direction": direction,
                }
                for request_id, row in changed_ids.items()
                if row["direction"] == direction
            ]
            chosen = cf.stratum_sample(
                candidates,
                args.samples_per_direction,
                seed_key=f"exact-prefill:{rate_label}:{direction}",
            )
            for row in chosen:
                samples.append(
                    {
                        **row,
                        "rate_label": rate_label,
                        "candidate_count": len(candidates),
                    }
                )

    work: list[dict[str, Any]] = []
    for sample in samples:
        info = baseline_info[sample["rate_label"]]
        request_id = str(sample["request_id"])
        baseline_remote = bool(sample["disaggregate"])
        flipped_action = "local" if baseline_remote else "remote"
        flipped_to = (
            "local" if baseline_remote else info["remote_prefill"]
        )
        plan = [dict(row) for row in info["plan"]]
        matches = 0
        for row in plan:
            if row["request_id"] == request_id:
                row["prefill_instance"] = flipped_to
                matches += 1
        if matches != 1:
            raise RuntimeError(
                f"expected one plan row for {request_id}, found {matches}"
            )
        safe_id = request_id.replace("/", "-")
        tag = (
            f"{info['tag']}__{sample['direction']}__{safe_id}"
            f"__to-{flipped_action}"
        )
        plan_path = diagnostic / "plans" / f"{tag}.csv"
        cf.write_plan(plan_path, plan)
        work.append(
            {
                "sample": sample,
                "info": info,
                "flipped_action": flipped_action,
                "plan_path": plan_path,
                "metrics_path": diagnostic / "deviations" / f"{tag}.json",
                "stdout_path": diagnostic / "stdout" / f"{tag}.txt",
            }
        )

    def execute(item: dict[str, Any]) -> dict[str, Any]:
        info = item["info"]
        cf.run_command(
            spec_path=info["spec_path"],
            requests=info["requests"],
            targets=targets,
            plan_path=item["plan_path"],
            metrics_path=item["metrics_path"],
            stdout_path=item["stdout_path"],
            force=args.force,
        )
        return item

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(execute, item) for item in work]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 10 == 0 or done == len(work):
                print(f"completed {done}/{len(work)} deviations", flush=True)

    result_rows: list[dict[str, Any]] = []
    for item in work:
        sample = item["sample"]
        info = item["info"]
        baseline_metrics = json.loads(info["metrics_path"].read_text())
        deviation_metrics = json.loads(item["metrics_path"].read_text())
        deviation_goodput = cf.metric_goodput(item["metrics_path"])
        delta = deviation_goodput - float(info["goodput"])
        base_completed, base_good = cf.request_good(
            baseline_metrics, sample["request_id"], targets
        )
        dev_completed, dev_good = cf.request_good(
            deviation_metrics, sample["request_id"], targets
        )
        result_rows.append(
            {
                "workload": "rag",
                "rate_label": sample["rate_label"],
                "direction_vs_legacy": sample["direction"],
                "request_id": sample["request_id"],
                "candidate_count": sample["candidate_count"],
                "time_decile": sample["time_decile"],
                "exact_action": (
                    "remote" if sample["disaggregate"] else "local"
                ),
                "flipped_to": item["flipped_action"],
                "baseline_goodput": info["goodput"],
                "deviated_goodput": deviation_goodput,
                "goodput_delta": delta,
                "exact_action_helpful": delta < -cf.GOODPUT_TOLERANCE,
                "exact_action_harmful": delta > cf.GOODPUT_TOLERANCE,
                "neutral": abs(delta) <= cf.GOODPUT_TOLERANCE,
                "baseline_request_completed": base_completed,
                "baseline_request_good": base_good,
                "deviated_request_completed": dev_completed,
                "deviated_request_good": dev_good,
                "request_good_delta_from_flip": int(dev_good) - int(base_good),
                "ap": sample["ap"],
                "qp": sample["qp"],
                "qd": sample["qd"],
                "t_adm_p": sample["t_adm_p"],
                "t_adm_d": sample["t_adm_d"],
                "var_disagg_prefill_pool": sample[
                    "var_disagg_prefill_pool"
                ],
                "var_benefit": sample["lhs"],
                "metrics_path": str(
                    item["metrics_path"].relative_to(base.ROOT)
                ),
            }
        )
    result_rows.sort(
        key=lambda row: (
            row["rate_label"],
            row["direction_vs_legacy"],
            row["time_decile"],
            row["request_id"],
        )
    )
    write_csv(diagnostic / "counterfactual_requests.csv", result_rows)

    summaries: list[dict[str, Any]] = []
    for rate_label in CELLS:
        for direction in ("local_to_remote", "remote_to_local"):
            members = [
                row
                for row in result_rows
                if row["rate_label"] == rate_label
                and row["direction_vs_legacy"] == direction
            ]
            deltas = [float(row["goodput_delta"]) for row in members]
            summaries.append(
                {
                    "condition": f"rag:{rate_label}",
                    "direction_vs_legacy": direction,
                    "candidate_count": (
                        int(members[0]["candidate_count"]) if members else 0
                    ),
                    "sample_count": len(members),
                    "exact_action_helpful": sum(
                        bool(row["exact_action_helpful"]) for row in members
                    ),
                    "exact_action_harmful": sum(
                        bool(row["exact_action_harmful"]) for row in members
                    ),
                    "neutral": sum(bool(row["neutral"]) for row in members),
                    "mean_flip_delta": (
                        statistics.fmean(deltas) if deltas else None
                    ),
                    "median_flip_delta": (
                        statistics.median(deltas) if deltas else None
                    ),
                    "own_request_helped_by_flip": sum(
                        int(row["request_good_delta_from_flip"]) > 0
                        for row in members
                    ),
                    "own_request_harmed_by_flip": sum(
                        int(row["request_good_delta_from_flip"]) < 0
                        for row in members
                    ),
                }
            )
    write_csv(diagnostic / "counterfactual_summary.csv", summaries)
    report = {
        "seed": trace.SEED,
        "diagnostic": (
            "placement-pinned one-request deviations from exact-prefill VaR; "
            "local hindsight regret, not global labels"
        ),
        "samples_per_direction": args.samples_per_direction,
        "baseline_self_consistency": {
            f"rag:{label}": {
                "goodput": info["goodput"],
                "requests": info["requests"],
            }
            for label, info in baseline_info.items()
        },
        "summaries": summaries,
    }
    (diagnostic / "counterfactual_result.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
