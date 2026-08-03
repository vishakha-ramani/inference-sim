#!/usr/bin/env python3
"""One-request forced-action diagnostics for the final public-workload policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import run_decisive_campaign as base
import run_public_load_static_benchmark as benchmark
import run_public_workload_heterogeneity_closeout as public


DEFAULT_OUT = base.CAMPAIGN / "out" / "public_joint_counterfactual_ttft_rollout_v2"
CAPACITY_SOURCE = (
    base.CAMPAIGN
    / "out"
    / "public_load_static_benchmark_ttft_rollout_v2"
    / "capacity_selection.json"
)
PROTOCOL = base.CAMPAIGN / "PUBLIC-JOINT-COUNTERFACTUAL-PROTOCOL.md"
POLICY = "causal_externality_no_capacity_v8"
DIAGNOSTIC_SEED = 1073741827
SAMPLE_PER_CELL = 8
V = 8.0

# Importing the public benchmark already disables the generic timeout. Keep the
# setting explicit here because this runner may also be imported independently.
public.REQUEST_TIMEOUT_SECS = -1


@dataclass(frozen=True)
class Cell:
    hardware: str
    workload: str
    rate_label: str
    rate: float
    requests: int

    @property
    def key(self) -> str:
        return f"{self.hardware}:{self.workload}:{self.rate_label}"

    @property
    def tag(self) -> str:
        return f"{self.hardware}_{self.workload}_{self.rate_label}"


@dataclass(frozen=True)
class FixedTask:
    cell: Cell
    phase: str
    tag: str
    kind: str
    plan_path: Path
    spec_path: Path
    seed: int
    request_id: str = ""
    forced_action: str = ""


def load_capacity() -> dict[str, Any]:
    if not CAPACITY_SOURCE.exists():
        raise SystemExit(f"missing frozen capacity source: {CAPACITY_SOURCE}")
    return json.loads(CAPACITY_SOURCE.read_text())


def make_cells(capacity: dict[str, Any], *, smoke: bool) -> list[Cell]:
    if smoke:
        hardware, workload, rate_label = (
            "h100_homogeneous",
            "interactive",
            benchmark.LOADS[0][0],
        )
        return [
            Cell(
                hardware,
                workload,
                rate_label,
                float(capacity[hardware][workload]["evaluation_rates"][rate_label]),
                40,
            )
        ]
    return [
        Cell(
            hardware,
            workload,
            rate_label,
            float(capacity[hardware][workload]["evaluation_rates"][rate_label]),
            public.WORKLOADS[workload].evaluation_requests,
        )
        for hardware in public.HARDWARES
        for workload in public.WORKLOADS
        for rate_label, _ in benchmark.LOADS
    ]


def online_run(cell: Cell, seed: int, *, smoke: bool) -> public.Run:
    phase = "public_joint_counterfactual_smoke_online" if smoke else "public_joint_counterfactual_online"
    return public.Run(
        phase,
        public.WORKLOADS[cell.workload],
        cell.rate_label,
        cell.rate,
        seed,
        POLICY,
        cell.hardware,
        cell.requests,
    )


def action_id(row: dict[str, str]) -> str:
    placement = "local" if row["local"] == "true" else row["prefill_instance"]
    return f"{row['decode_instance']}:{placement}"


def action_meta(row: dict[str, str]) -> dict[str, Any]:
    local = row["local"] == "true"
    return {
        "action": action_id(row),
        "decode_instance": row["decode_instance"],
        "prefill_instance": "local" if local else row["prefill_instance"],
        "local": local,
    }


def read_candidate_trace(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        raise RuntimeError(f"missing candidate trace: {path}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["request_id"]].append(row)
    return dict(grouped)


def capture_plan(path: Path) -> dict[str, Any]:
    grouped = read_candidate_trace(path)
    plan_rows = []
    requests: dict[str, Any] = {}
    max_score_error = 0.0
    argmin_violations = 0
    candidate_count_violations = 0
    for request_id, rows in grouped.items():
        if len(rows) != 4 or len({action_id(row) for row in rows}) != 4:
            candidate_count_violations += 1
        chosen = [row for row in rows if row["chosen"] == "true"]
        if len(chosen) != 1:
            raise RuntimeError(
                f"{path}: request {request_id} has {len(chosen)} chosen candidates"
            )
        selected = chosen[0]
        selected_score = float(selected["score"])
        best_score = min(float(row["score"]) for row in rows)
        if selected_score > best_score + 1e-10 * max(1.0, abs(best_score)):
            argmin_violations += 1
        for row in rows:
            score = float(row["score"])
            expected = V * (
                float(row["slo_externality"]) - float(row["own_good"])
            )
            max_score_error = max(max_score_error, abs(score - expected))
            if float(row["capacity_total"]) != 0:
                raise RuntimeError(f"{path}: nonzero capacity term in final policy")
        selected_meta = action_meta(selected)
        plan_rows.append(
            {
                "request_id": request_id,
                "decode_instance": selected_meta["decode_instance"],
                "prefill_instance": selected_meta["prefill_instance"],
            }
        )
        requests[request_id] = {
            "chosen": selected_meta,
            "actions": {action_id(row): action_meta(row) for row in rows},
        }
    return {
        "plan_rows": plan_rows,
        "requests": requests,
        "request_count": len(grouped),
        "max_abs_score_identity_error": max_score_error,
        "argmin_violations": argmin_violations,
        "candidate_count_violations": candidate_count_violations,
    }


def write_plan(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("request_id", "decode_instance", "prefill_instance"),
        )
        writer.writeheader()
        writer.writerows(rows)


def sample_hash(cell: str, request_id: str) -> str:
    return hashlib.sha256(f"{cell}|{request_id}".encode()).hexdigest()


def sample_requests(
    cell: Cell, capture: dict[str, Any], sample_count: int
) -> list[dict[str, Any]]:
    strata: dict[str, list[str]] = defaultdict(list)
    for request_id, item in capture["requests"].items():
        strata[item["chosen"]["action"]].append(request_id)
    for request_ids in strata.values():
        request_ids.sort(key=lambda request_id: sample_hash(cell.key, request_id))

    selected: list[str] = []
    for action in sorted(strata):
        selected.extend(strata[action][:2])
    if len(selected) > sample_count:
        selected.sort(key=lambda request_id: sample_hash(cell.key, request_id))
        selected = selected[:sample_count]
    selected_set = set(selected)
    remaining = sorted(
        (
            request_id
            for request_id in capture["requests"]
            if request_id not in selected_set
        ),
        key=lambda request_id: sample_hash(cell.key, request_id),
    )
    selected.extend(remaining[: sample_count - len(selected)])
    if len(selected) != sample_count:
        raise RuntimeError(
            f"{cell.key}: sampled {len(selected)} requests, want {sample_count}"
        )
    return [
        {
            "sample_index": index,
            "request_id": request_id,
            "hash": sample_hash(cell.key, request_id),
            "chosen": capture["requests"][request_id]["chosen"],
            "actions": capture["requests"][request_id]["actions"],
        }
        for index, request_id in enumerate(selected)
    ]


def online_spec_path(out: Path, run: public.Run) -> Path:
    return out / run.phase / "specs" / f"{run.tag()}.yaml"


def execute_fixed(task: FixedTask, out: Path, force: bool) -> dict[str, Any]:
    metrics_path = out / task.phase / "metrics" / f"{task.tag}.json"
    stdout_path = out / task.phase / "stdout" / f"{task.tag}.txt"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    config = public.WORKLOADS[task.cell.workload]
    if not force and metrics_path.exists() and stdout_path.exists():
        row = base.result_row(
            phase=task.phase,
            workload=task.cell.workload,
            rate_label=task.cell.rate_label,
            rate=task.cell.rate,
            seed=task.seed,
            policy=task.kind,
            parameter=None,
            topology=f"1p2d:{task.cell.hardware}",
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
            str(task.spec_path),
            "--num-requests",
            str(task.cell.requests),
            *public.topology_args(task.cell.hardware),
            *base.slo_args(config.targets),
            "--pd-plan",
            str(task.plan_path),
            "--timeout",
            "-1",
            "--seed",
            str(task.seed),
            "--metrics-path",
            str(metrics_path),
        ]
        proc = subprocess.run(
            command, cwd=base.ROOT, capture_output=True, text=True
        )
        stdout_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                f"fixed-plan run failed ({task.tag}):\n"
                f"{proc.stderr[-5000:]}\n{proc.stdout[-5000:]}"
            )
        row = base.result_row(
            phase=task.phase,
            workload=task.cell.workload,
            rate_label=task.cell.rate_label,
            rate=task.cell.rate,
            seed=task.seed,
            policy=task.kind,
            parameter=None,
            topology=f"1p2d:{task.cell.hardware}",
            metrics_path=metrics_path,
            stdout_path=stdout_path,
        )
    row["hardware"] = task.cell.hardware
    row["cell"] = task.cell.key
    row["request_id"] = task.request_id
    row["forced_action"] = task.forced_action
    row["hard_valid"] = public.hard_valid(row)
    return row


def run_fixed_many(
    tasks: Iterable[FixedTask], out: Path, jobs: int, force: bool
) -> list[dict[str, Any]]:
    planned = list(tasks)
    rows = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(execute_fixed, task, out, force): task for task in planned
        }
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/{len(planned)}] {task.tag} "
                f"goodput={row['goodput']:.3f} valid={row['hard_valid']}",
                flush=True,
            )
    return rows


def exact_replay_gate(
    cells: list[Cell], online_rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    fields = (
        "goodput",
        "good_ttft",
        "good_itl",
        "good_e2e",
        "injected",
        "completed",
        "dropped",
        "still_queued",
        "still_running",
        "timed_out",
        "length_capped",
    )
    mismatches = []
    for cell in cells:
        online = next(
            row
            for row in online_rows
            if row["hardware"] == cell.hardware
            and row["workload"] == cell.workload
            and row["rate_label"] == cell.rate_label
        )
        replay = next(row for row in replay_rows if row["cell"] == cell.key)
        differences = {
            field: {"online": online[field], "replay": replay[field]}
            for field in fields
            if online[field] != replay[field]
        }
        if differences:
            mismatches.append({"cell": cell.key, "differences": differences})
    return {
        "fields": list(fields),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def conservative_best_action(
    best_actions: list[str], baseline_action: str, actions: dict[str, dict[str, Any]]
) -> str:
    baseline = actions[baseline_action]

    def rank(action: str) -> tuple[int, str]:
        candidate = actions[action]
        changed = int(candidate["decode_instance"] != baseline["decode_instance"])
        changed += int(candidate["local"] != baseline["local"])
        return changed, action

    return min(best_actions, key=rank)


def request_results(
    cells: list[Cell],
    samples: dict[str, list[dict[str, Any]]],
    replay_rows: list[dict[str, Any]],
    deviation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for cell in cells:
        baseline_goodput = float(
            next(row for row in replay_rows if row["cell"] == cell.key)["goodput"]
        )
        for sample in samples[cell.key]:
            baseline_action = sample["chosen"]["action"]
            action_goodput = {baseline_action: baseline_goodput}
            members = [
                row
                for row in deviation_rows
                if row["cell"] == cell.key
                and row["request_id"] == sample["request_id"]
            ]
            if len(members) != 3:
                raise RuntimeError(
                    f"{cell.key}/{sample['request_id']}: {len(members)} deviations, want 3"
                )
            for row in members:
                action_goodput[row["forced_action"]] = float(row["goodput"])
            if set(action_goodput) != set(sample["actions"]):
                raise RuntimeError(
                    f"{cell.key}/{sample['request_id']}: incomplete action set"
                )
            best_goodput = max(action_goodput.values())
            tolerance = 1e-12
            best_actions = sorted(
                action
                for action, value in action_goodput.items()
                if best_goodput - value <= tolerance
            )
            agrees = baseline_action in best_actions
            regret = max(0.0, best_goodput - baseline_goodput)
            representative = conservative_best_action(
                best_actions, baseline_action, sample["actions"]
            )
            baseline_meta = sample["actions"][baseline_action]
            best_meta = sample["actions"][representative]
            decoder_changed = (
                best_meta["decode_instance"] != baseline_meta["decode_instance"]
            )
            placement_changed = best_meta["local"] != baseline_meta["local"]
            if agrees:
                category = "agreement"
                direction = "unchanged"
            elif decoder_changed and placement_changed:
                category = "decoder_and_placement"
                direction = "local_to_remote" if baseline_meta["local"] else "remote_to_local"
            elif decoder_changed:
                category = "decoder_only"
                direction = "unchanged"
            elif placement_changed:
                category = "placement_only"
                direction = "local_to_remote" if baseline_meta["local"] else "remote_to_local"
            else:
                raise RuntimeError("disagreement did not change an action dimension")
            results.append(
                {
                    "cell": cell.key,
                    "hardware": cell.hardware,
                    "workload": cell.workload,
                    "rate_label": cell.rate_label,
                    "requests_in_trace": cell.requests,
                    "request_id": sample["request_id"],
                    "baseline_action": baseline_action,
                    "best_actions": best_actions,
                    "representative_best_action": representative,
                    "baseline_goodput": baseline_goodput,
                    "best_goodput": best_goodput,
                    "regret": regret,
                    "good_requests_recovered": regret * cell.requests,
                    "agrees": agrees,
                    "error_category": category,
                    "direction": direction,
                    "decoder_changed": decoder_changed,
                    "placement_changed": placement_changed,
                    "action_goodput": action_goodput,
                }
            )
    return results


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"sampled_requests": 0}
    disagreements = [item for item in items if not item["agrees"]]
    regrets = [float(item["regret"]) for item in items]
    return {
        "sampled_requests": len(items),
        "agreement_fraction": sum(bool(item["agrees"]) for item in items) / len(items),
        "positive_regret_fraction": len(disagreements) / len(items),
        "mean_goodput_regret": statistics.fmean(regrets),
        "total_goodput_regret": sum(regrets),
        "mean_good_requests_recovered": statistics.fmean(
            float(item["good_requests_recovered"]) for item in items
        ),
        "total_good_requests_recovered": sum(
            float(item["good_requests_recovered"]) for item in items
        ),
        "local_to_remote": sum(
            item["direction"] == "local_to_remote" for item in disagreements
        ),
        "remote_to_local": sum(
            item["direction"] == "remote_to_local" for item in disagreements
        ),
        "decoder_errors": sum(bool(item["decoder_changed"]) for item in disagreements),
        "placement_errors": sum(
            bool(item["placement_changed"]) for item in disagreements
        ),
    }


def grouped_summary(
    results: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    values = sorted({str(item[field]) for item in results})
    return {
        value: summarize([item for item in results if str(item[field]) == value])
        for value in values
    }


def analyze(
    cells: list[Cell],
    online_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    deviation_rows: list[dict[str, Any]],
    samples: dict[str, list[dict[str, Any]]],
    captures: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gate = exact_replay_gate(cells, online_rows, replay_rows)
    results = request_results(cells, samples, replay_rows, deviation_rows)
    error_categories = {}
    for category in (
        "agreement",
        "decoder_only",
        "placement_only",
        "decoder_and_placement",
    ):
        members = [item for item in results if item["error_category"] == category]
        error_categories[category] = {
            **summarize(members),
            "total_goodput_lost": sum(float(item["regret"]) for item in members),
        }
    return {
        "status": "exact one-request forced-action diagnostic; not a global oracle",
        "routing_value": "smooth TTFT x E2E; reported goodput remains TTFT x ITL x E2E",
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "cell_count": len(cells),
        "online_runs": len(online_rows),
        "replay_gate_runs": len(replay_rows),
        "sampled_requests": len(results),
        "deviation_runs": len(deviation_rows),
        "overall": summarize(results),
        "by_cell": grouped_summary(results, "cell"),
        "by_hardware": grouped_summary(results, "hardware"),
        "by_workload": grouped_summary(results, "workload"),
        "by_load": grouped_summary(results, "rate_label"),
        "by_baseline_action": grouped_summary(results, "baseline_action"),
        "by_error_category": error_categories,
        "request_results": results,
        "replay_gate": gate,
        "trace_validation": {
            "max_abs_score_identity_error": max(
                capture["max_abs_score_identity_error"]
                for capture in captures.values()
            ),
            "argmin_violations": sum(
                capture["argmin_violations"] for capture in captures.values()
            ),
            "candidate_count_violations": sum(
                capture["candidate_count_violations"]
                for capture in captures.values()
            ),
        },
        "hard_invalid_online_runs": sum(
            not bool(row["hard_valid"]) for row in online_rows
        ),
        "hard_invalid_replay_runs": sum(
            not bool(row["hard_valid"]) for row in replay_rows
        ),
        "hard_invalid_deviation_runs": sum(
            not bool(row["hard_valid"]) for row in deviation_rows
        ),
        "deviation_runs_with_drops": sum(
            int(row["dropped"]) > 0 for row in deviation_rows
        ),
        "deviation_runs_with_timeouts": sum(
            int(row["timed_out"]) > 0 for row in deviation_rows
        ),
        "deviation_runs_with_length_caps": sum(
            int(row["length_capped"]) > 0 for row in deviation_rows
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    flattened = []
    for row in rows:
        flattened.append(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def write_report(out: Path, result: dict[str, Any]) -> None:
    overall = result["overall"]
    lines = [
        "# Public joint request counterfactuals",
        "",
        "This is an exact one-request forced-action diagnostic, not a global oracle.",
        "All other request actions remain fixed to the final policy's captured plan.",
        "",
        "## Overall",
        "",
        f"- Sampled decisions: {overall['sampled_requests']} across {result['cell_count']} cells.",
        f"- Agreement with a best forced action: {overall['agreement_fraction']:.1%}.",
        f"- Positive one-request regret: {overall['positive_regret_fraction']:.1%}.",
        f"- Mean goodput regret per sampled decision: {overall['mean_goodput_regret']:.5f}.",
        f"- Mean equivalent good requests recovered: {overall['mean_good_requests_recovered']:.3f}.",
        f"- Local decisions that should be remote: {overall['local_to_remote']}.",
        f"- Remote decisions that should be local: {overall['remote_to_local']}.",
        f"- Decisions with a decoder error: {overall['decoder_errors']}.",
        "",
        "## Error decomposition",
        "",
        "| class | decisions | total goodput lost |",
        "|---|---:|---:|",
    ]
    for category, summary in result["by_error_category"].items():
        lines.append(
            f"| {category} | {summary['sampled_requests']} | "
            f"{summary['total_goodput_lost']:.4f} |"
        )
    lines += [
        "",
        "## By workload",
        "",
        "| workload | decisions | agreement | positive regret | mean regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for workload, summary in result["by_workload"].items():
        lines.append(
            f"| {workload} | {summary['sampled_requests']} | "
            f"{summary['agreement_fraction']:.1%} | "
            f"{summary['positive_regret_fraction']:.1%} | "
            f"{summary['mean_goodput_regret']:.5f} |"
        )
    lines += [
        "",
        "## By fleet",
        "",
        "| fleet | decisions | agreement | positive regret | mean regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for hardware, summary in result["by_hardware"].items():
        lines.append(
            f"| {hardware} | {summary['sampled_requests']} | "
            f"{summary['agreement_fraction']:.1%} | "
            f"{summary['positive_regret_fraction']:.1%} | "
            f"{summary['mean_goodput_regret']:.5f} |"
        )
    trace = result["trace_validation"]
    lines += [
        "",
        "## Validity",
        "",
        f"- Online runs: {result['online_runs']}; replay gates: {result['replay_gate_runs']}; deviation runs: {result['deviation_runs']}.",
        f"- Replay mismatches: {result['replay_gate']['mismatch_count']}.",
        f"- Online argmin violations: {trace['argmin_violations']}.",
        f"- Candidate-count violations: {trace['candidate_count_violations']}.",
        f"- Maximum score-identity error: {trace['max_abs_score_identity_error']:.3g}.",
        f"- Hard-invalid runs: {result['hard_invalid_online_runs'] + result['hard_invalid_replay_runs'] + result['hard_invalid_deviation_runs']}.",
        f"- Deviation runs with drops/timeouts/length caps: {result['deviation_runs_with_drops']}/{result['deviation_runs_with_timeouts']}/{result['deviation_runs_with_length_caps']}.",
    ]
    (out / "PUBLIC-JOINT-COUNTERFACTUAL.md").write_text("\n".join(lines) + "\n")


def run_campaign(out: Path, jobs: int, force: bool, *, smoke: bool) -> None:
    if not PROTOCOL.exists():
        raise SystemExit(f"missing frozen protocol: {PROTOCOL}")
    cells = make_cells(load_capacity(), smoke=smoke)
    seed = 42 if smoke else DIAGNOSTIC_SEED
    sample_count = 2 if smoke else SAMPLE_PER_CELL
    online_runs = [online_run(cell, seed, smoke=smoke) for cell in cells]
    online_rows = public.run_many(online_runs, out, jobs, force)
    public.require_hard_valid(online_rows, "counterfactual online baselines")

    captures: dict[str, dict[str, Any]] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    replay_tasks = []
    plans_by_cell: dict[str, list[dict[str, str]]] = {}
    plan_root = out / ("smoke_plans" if smoke else "plans")
    for cell, run in zip(cells, online_runs):
        capture = capture_plan(public.trace_path(out, run))
        if capture["request_count"] != cell.requests:
            raise RuntimeError(
                f"{cell.key}: captured {capture['request_count']} requests, want {cell.requests}"
            )
        captures[cell.key] = capture
        plans_by_cell[cell.key] = capture["plan_rows"]
        baseline_plan = plan_root / cell.tag / "baseline.csv"
        write_plan(baseline_plan, capture["plan_rows"])
        samples[cell.key] = sample_requests(cell, capture, sample_count)
        replay_phase = (
            "public_joint_counterfactual_smoke_replay"
            if smoke
            else "public_joint_counterfactual_replay"
        )
        replay_tasks.append(
            FixedTask(
                cell,
                replay_phase,
                f"{cell.tag}_baseline_replay_s{seed}",
                "baseline_replay",
                baseline_plan,
                online_spec_path(out, run),
                seed,
            )
        )

    manifest = {
        "status": "frozen before forced-action deviations",
        "seed": seed,
        "sample_per_cell": sample_count,
        "cells": samples,
    }
    manifest_name = "smoke_sample_manifest.json" if smoke else "sample_manifest.json"
    (out / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")

    replay_rows = run_fixed_many(replay_tasks, out, jobs, force)
    public.require_hard_valid(replay_rows, "counterfactual unchanged-plan replay")
    gate = exact_replay_gate(cells, online_rows, replay_rows)
    if gate["mismatch_count"]:
        raise RuntimeError(f"unchanged-plan replay gate failed: {gate}")

    deviation_tasks = []
    for cell, run in zip(cells, online_runs):
        base_rows = plans_by_cell[cell.key]
        for sample in samples[cell.key]:
            baseline_action = sample["chosen"]["action"]
            alternatives = sorted(
                action for action in sample["actions"] if action != baseline_action
            )
            for action_index, action in enumerate(alternatives):
                meta = sample["actions"][action]
                changed = [dict(row) for row in base_rows]
                matches = [
                    row
                    for row in changed
                    if row["request_id"] == sample["request_id"]
                ]
                if len(matches) != 1:
                    raise RuntimeError("deviation plan request is not unique")
                matches[0]["decode_instance"] = meta["decode_instance"]
                matches[0]["prefill_instance"] = meta["prefill_instance"]
                plan_path = (
                    plan_root
                    / cell.tag
                    / f"dev_{sample['sample_index']:02d}_a{action_index}.csv"
                )
                write_plan(plan_path, changed)
                phase = (
                    "public_joint_counterfactual_smoke_deviation"
                    if smoke
                    else "public_joint_counterfactual_deviation"
                )
                deviation_tasks.append(
                    FixedTask(
                        cell,
                        phase,
                        f"{cell.tag}_dev_{sample['sample_index']:02d}_a{action_index}_s{seed}",
                        "one_request_deviation",
                        plan_path,
                        online_spec_path(out, run),
                        seed,
                        sample["request_id"],
                        action,
                    )
                )

    deviation_rows = run_fixed_many(deviation_tasks, out, jobs, force)
    public.require_hard_valid(deviation_rows, "counterfactual deviations")
    result = analyze(
        cells, online_rows, replay_rows, deviation_rows, samples, captures
    )
    expected_cells = 1 if smoke else 18
    expected_samples = expected_cells * sample_count
    expected_deviations = expected_samples * 3
    trace = result["trace_validation"]
    if (
        result["cell_count"] != expected_cells
        or result["sampled_requests"] != expected_samples
        or result["deviation_runs"] != expected_deviations
        or result["replay_gate"]["mismatch_count"]
        or trace["max_abs_score_identity_error"] > 1e-9
        or trace["argmin_violations"]
        or trace["candidate_count_violations"]
        or result["hard_invalid_online_runs"]
        or result["hard_invalid_replay_runs"]
        or result["hard_invalid_deviation_runs"]
        or result["deviation_runs_with_timeouts"]
        or result["deviation_runs_with_length_caps"]
    ):
        raise RuntimeError(f"counterfactual validity gate failed: {result}")

    prefix = "smoke_" if smoke else ""
    public.write_rows(out / f"{prefix}online.csv", online_rows)
    public.write_rows(out / f"{prefix}replay.csv", replay_rows)
    public.write_rows(out / f"{prefix}deviations.csv", deviation_rows)
    write_csv(out / f"{prefix}request_results.csv", result["request_results"])
    result_path = out / ("smoke_result.json" if smoke else "counterfactual_result.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    if not smoke:
        write_report(out, result)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "confirm", "all"), default="all")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = ("smoke", "confirm") if args.stage == "all" else (args.stage,)
    for stage in stages:
        run_campaign(args.out, args.jobs, args.force, smoke=stage == "smoke")


if __name__ == "__main__":
    main()
