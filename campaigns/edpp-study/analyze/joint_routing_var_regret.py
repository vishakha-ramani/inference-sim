#!/usr/bin/env python3
"""Analyze generalized joint-routing candidate traces in VaR score space.

The input is the simulator's native long-form candidate-trace contract. Each
request has one row per feasible joint action and exactly one row marked
``chosen``. Blank ``prefill_instance`` values are the only normalization
performed (to ``local``).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

INVENTORY_COLUMNS = {"instance_id", "capability"}
TRACE_COLUMNS = {
    "request_id", "decode_instance", "prefill_instance", "chosen",
    "var_decode", "var_colloc_prefill", "var_prefill_pool", "var_total",
    "router_decode",
}
VAR_COMPONENTS = ("var_decode", "var_colloc_prefill", "var_prefill_pool")
CAPABILITIES = {"prefill", "mixed"}


def _read(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return list(reader), set(reader.fieldnames)


def derive_topology(inventory_path: Path) -> tuple[str, dict[str, str]]:
    rows, fields = _read(inventory_path)
    missing = INVENTORY_COLUMNS - fields
    if missing:
        raise ValueError(f"{inventory_path}: missing columns {sorted(missing)}")
    instances: dict[str, str] = {}
    for line, row in enumerate(rows, 2):
        instance = row["instance_id"].strip()
        capability = row["capability"].strip().lower()
        if not instance or instance in instances:
            raise ValueError(f"{inventory_path}:{line}: blank or duplicate instance_id")
        if capability not in CAPABILITIES:
            raise ValueError(
                f"{inventory_path}:{line}: capability must be one of {sorted(CAPABILITIES)}"
            )
        instances[instance] = capability
    p = sum(value == "prefill" for value in instances.values())
    m = sum(value == "mixed" for value in instances.values())
    if not p or not m:
        raise ValueError(f"{inventory_path}: joint routing requires >=1 prefill and >=1 mixed")
    return f"{p}P{m}D", instances


def _boolean(value: str, where: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{where}: chosen must be true/false (or 1/0), got {value!r}")


def _number(value: str, where: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{where}: expected numeric value, got {value!r}") from exc
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{where}: value must be finite")
    return result


def load_candidates(trace_path: Path, instances: dict[str, str]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    rows, fields = _read(trace_path)
    missing = TRACE_COLUMNS - fields
    if missing:
        raise ValueError(f"{trace_path}: missing columns {sorted(missing)}")
    components = list(VAR_COMPONENTS)
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for line, row in enumerate(rows, 2):
        where = f"{trace_path}:{line}"
        rid = row["request_id"].strip()
        decode = row["decode_instance"].strip()
        prefill = row["prefill_instance"].strip() or "local"
        if prefill.lower() == "local":
            prefill = "local"
        router_decode = _boolean(row["router_decode"], where)
        if not rid:
            raise ValueError(f"{where}: blank request_id")
        if instances.get(decode) != "mixed":
            raise ValueError(f"{where}: decode_instance {decode!r} is not a mixed instance")
        if prefill != "local" and instances.get(prefill) != "prefill":
            raise ValueError(f"{where}: prefill_instance {prefill!r} is not local or prefill-only")
        key = (rid, decode, prefill)
        if key in seen:
            raise ValueError(f"{where}: duplicate candidate {key}")
        seen.add(key)
        candidate: dict[str, Any] = {
            "request_id": rid,
            "decode_instance": decode,
            "prefill_instance": prefill,
            "is_router_decode": router_decode,
            "chosen": _boolean(row["chosen"], where),
            "predicted_var": _number(row["var_total"], where),
        }
        for component in components:
            candidate[component] = _number(row[component], where)
        by_request[rid].append(candidate)
    if not by_request:
        raise ValueError(f"{trace_path}: no candidate rows")
    decode_instances = sorted(name for name, role in instances.items() if role == "mixed")
    prefill_instances = sorted(name for name, role in instances.items() if role == "prefill")
    expected_actions = {
        (decode, prefill)
        for decode in decode_instances
        for prefill in ["local", *prefill_instances]
    }
    for rid, candidates in by_request.items():
        router_flags = {
            decode: {item["is_router_decode"] for item in candidates
                     if item["decode_instance"] == decode}
            for decode in decode_instances
        }
        if any(len(flags) != 1 for flags in router_flags.values()):
            raise ValueError(f"{trace_path}: request {rid!r} has inconsistent router_decode flags")
        router_decodes = [
            decode for decode, flags in router_flags.items() if True in flags
        ]
        if len(router_decodes) != 1:
            raise ValueError(f"{trace_path}: request {rid!r} must mark exactly one router decode")
        for item in candidates:
            item["router_decode"] = router_decodes[0]
        if sum(item["chosen"] for item in candidates) != 1:
            raise ValueError(f"{trace_path}: request {rid!r} must have exactly one chosen candidate")
        actual_actions = {
            (item["decode_instance"], item["prefill_instance"])
            for item in candidates
        }
        if actual_actions != expected_actions:
            missing_actions = sorted(expected_actions - actual_actions)
            raise ValueError(
                f"{trace_path}: request {rid!r} must enumerate D(P+1)={len(expected_actions)} "
                f"feasible actions; missing {missing_actions}"
            )
    return dict(by_request), components


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def analyze(inventory_path: Path, trace_path: Path) -> dict[str, Any]:
    topology, instances = derive_topology(inventory_path)
    requests, components = load_candidates(trace_path, instances)
    per_request: list[dict[str, Any]] = []
    for rid in sorted(requests):
        candidates = requests[rid]
        chosen = next(item for item in candidates if item["chosen"])
        router_decode = chosen["router_decode"]
        router_slice = [
            item for item in candidates
            if item["decode_instance"] == router_decode
        ]
        global_best = min(candidates, key=lambda item: item["predicted_var"])
        decomposed_best = min(router_slice, key=lambda item: item["predicted_var"])
        joint_regret = max(0.0, chosen["predicted_var"] - global_best["predicted_var"])
        router_induced_regret = max(
            0.0, decomposed_best["predicted_var"] - global_best["predicted_var"]
        )
        strict_disagreement = joint_regret > 1e-9
        item: dict[str, Any] = {
            "request_id": rid,
            "candidate_count": len(candidates),
            "chosen_action": f"{chosen['decode_instance']}|{chosen['prefill_instance']}",
            "best_action": f"{global_best['decode_instance']}|{global_best['prefill_instance']}",
            "chosen_var": chosen["predicted_var"],
            "best_var": global_best["predicted_var"],
            "var_regret": joint_regret,
            "router_induced_var_regret": router_induced_regret,
            "decomposed_action": f"{decomposed_best['decode_instance']}|{decomposed_best['prefill_instance']}",
            "decomposed_var": decomposed_best["predicted_var"],
            "action_disagreement": chosen is not global_best,
            "strict_var_disagreement": strict_disagreement,
            "decode_reversal": decomposed_best["decode_instance"] != global_best["decode_instance"],
            "split_reversal": decomposed_best["prefill_instance"] != global_best["prefill_instance"],
            "locality_reversal": (
                (decomposed_best["prefill_instance"] == "local")
                != (global_best["prefill_instance"] == "local")
            ),
            "router_decode": router_decode,
            "chosen_differs_from_router_decode": (
                chosen["decode_instance"] != chosen["router_decode"]
            ),
            "best_differs_from_router_decode": (
                global_best["decode_instance"] != chosen["router_decode"]
            ),
        }
        for component in components:
            item[f"chosen_{component}"] = chosen[component]
            item[f"best_{component}"] = global_best[component]
            item[f"{component}_delta"] = chosen[component] - global_best[component]
        per_request.append(item)
    positive = [row for row in per_request if row["var_regret"] > 1e-9]
    router_positive = [
        row for row in per_request if row["router_induced_var_regret"] > 1e-9
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "topology": topology,
        "instance_counts": {
            "prefill": sum(value == "prefill" for value in instances.values()),
            "decode_capable": sum(value == "mixed" for value in instances.values()),
        },
        "action_cardinality": (
            sum(value == "mixed" for value in instances.values())
            * (sum(value == "prefill" for value in instances.values()) + 1)
        ),
        "regret_definition": "chosen predicted_var minus minimum feasible predicted_var",
        "router_regret_definition": "minimum predicted_var on the scorer-selected decode minus the global minimum",
        "request_count": len(per_request),
        "mean_var_regret": _mean(row["var_regret"] for row in per_request),
        "total_var_regret": sum(row["var_regret"] for row in per_request),
        "positive_regret_fraction": len(positive) / len(per_request),
        "mean_router_induced_var_regret": _mean(
            row["router_induced_var_regret"] for row in per_request
        ),
        "total_router_induced_var_regret": sum(
            row["router_induced_var_regret"] for row in per_request
        ),
        "positive_router_regret_fraction": len(router_positive) / len(per_request),
        "action_disagreement_fraction": _mean(row["action_disagreement"] for row in per_request),
        "strict_var_disagreement_fraction": _mean(
            row["strict_var_disagreement"] for row in per_request
        ),
        "decode_reversal_fraction": _mean(row["decode_reversal"] for row in per_request),
        "split_reversal_fraction": _mean(row["split_reversal"] for row in per_request),
        "locality_reversal_fraction": _mean(row["locality_reversal"] for row in per_request),
        "chosen_router_decode_disagreement_fraction": _mean(
            row["chosen_differs_from_router_decode"] for row in per_request
        ),
        "best_router_decode_disagreement_fraction": _mean(
            row["best_differs_from_router_decode"] for row in per_request
        ),
        "per_request": per_request,
    }
    for component in components:
        report[f"mean_{component}_delta"] = _mean(
            row[f"{component}_delta"] for row in per_request
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = analyze(args.inventory, args.trace)
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
