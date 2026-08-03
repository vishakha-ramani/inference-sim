#!/usr/bin/env python3
"""Replace adapted Kairos with paper-mode Kairos in frozen secondary studies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import run_public_final_topology_sweep as topology
import run_public_mixed_burst_benchmark as mixed


PROTOCOL = (
    mixed.base.CAMPAIGN / "PUBLIC-PAPER-KAIROS-SECONDARY-EXTENSION-PROTOCOL.md"
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    for row in rows:
        row["hard_valid"] = str(row["hard_valid"]).lower() == "true"
    return rows


def run_mixed(out: Path, jobs: int, force: bool) -> dict[str, Any]:
    static = mixed.load_static(out)
    paper_runs = [
        mixed.Run(
            "paper_kairos_confirmation",
            profile,
            hardware,
            seed,
            mixed.PAPER_KAIROS,
        )
        for hardware in mixed.prior.HARDWARES
        for profile in mixed.PROFILES
        for seed in mixed.CONFIRMATION_SEEDS
    ]
    if len(paper_runs) != 32:
        raise RuntimeError(f"mixed paper-Kairos block has {len(paper_runs)} runs, want 32")
    paper_rows = mixed.run_many(paper_runs, out, jobs, force)
    mixed.require_valid(paper_rows, "mixed paper-Kairos block")
    mixed.write_rows(out / "paper_kairos_confirmation.csv", paper_rows)

    archived = read_rows(out / "confirmation.csv")
    retained = [row for row in archived if row["policy"] != mixed.ADAPTED_KAIROS]
    if len(archived) != 128 or len(retained) != 96:
        raise RuntimeError(
            f"unexpected mixed archive counts: total={len(archived)} retained={len(retained)}"
        )
    combined = retained + paper_rows
    mixed.require_valid(combined, "mixed combined paper-Kairos analysis")
    mixed.write_rows(out / "confirmation_paper_kairos.csv", combined)

    saved = (mixed.KAIROS_POLICY, mixed.DEPLOYABLE, mixed.POLICIES)
    mixed.KAIROS_POLICY = mixed.PAPER_KAIROS
    mixed.DEPLOYABLE = (mixed.FOCAL, "least_ttft_joint", mixed.PAPER_KAIROS)
    mixed.POLICIES = (*mixed.DEPLOYABLE, "static_joint_yardstick")
    try:
        full_runs = mixed.confirmation_runs(static)
        result = mixed.analyze_confirmation(out, full_runs, combined)
        result["status"] = (
            "matched mixed-workload/burst analysis with paper-mode Kairos; "
            "other policy rows reused from the frozen confirmation"
        )
        result["new_paper_kairos_runs"] = len(paper_rows)
        result["reused_frozen_runs"] = len(retained)
        (out / "confirmation_paper_kairos_result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        mixed.write_report(
            out,
            static,
            result,
            "PUBLIC-MIXED-BURST-PAPER-KAIROS.md",
        )
    finally:
        mixed.KAIROS_POLICY, mixed.DEPLOYABLE, mixed.POLICIES = saved
    return result


def run_topology(out: Path, jobs: int, force: bool) -> dict[str, Any]:
    capacity = topology.load_json(out, "capacity_selection.json")
    static = topology.load_json(out, "static_selection.json")
    paper_runs = [
        topology.Run(
            "paper_kairos_confirmation",
            topology_name,
            name,
            float(capacity[topology_name][name]["evaluation_rate"]),
            seed,
            topology.PAPER_KAIROS,
            cfg.evaluation_requests,
        )
        for topology_name in topology.TOPOLOGIES
        for name, cfg in topology.prior.WORKLOADS.items()
        for seed in topology.CONFIRMATION_SEEDS
    ]
    if len(paper_runs) != 36:
        raise RuntimeError(
            f"topology paper-Kairos block has {len(paper_runs)} runs, want 36"
        )
    paper_rows = topology.run_many(paper_runs, out, jobs, force)
    topology.require_terminal(paper_rows, "topology paper-Kairos block")
    topology.write_rows(out / "paper_kairos_confirmation.csv", paper_rows)

    archived = read_rows(out / "confirmation.csv")
    retained = [row for row in archived if row["policy"] != topology.ADAPTED_KAIROS]
    if len(archived) != 144 or len(retained) != 108:
        raise RuntimeError(
            f"unexpected topology archive counts: total={len(archived)} retained={len(retained)}"
        )
    combined = retained + paper_rows
    topology.require_terminal(combined, "topology combined paper-Kairos analysis")
    topology.write_rows(out / "confirmation_paper_kairos.csv", combined)

    saved = (topology.KAIROS_POLICY, topology.DEPLOYABLE, topology.POLICIES)
    topology.KAIROS_POLICY = topology.PAPER_KAIROS
    topology.DEPLOYABLE = (
        topology.FOCAL,
        "least_ttft_joint",
        topology.PAPER_KAIROS,
    )
    topology.POLICIES = (*topology.DEPLOYABLE, "static_joint_yardstick")
    try:
        full_runs = topology.confirmation_runs(capacity, static)
        result = topology.analyze(out, full_runs, combined)
        result["status"] = (
            "matched topology analysis with paper-mode Kairos; other policy "
            "rows reused from the frozen confirmation"
        )
        result["new_paper_kairos_runs"] = len(paper_rows)
        result["reused_frozen_runs"] = len(retained)
        (out / "confirmation_paper_kairos_result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        topology.write_report(
            out,
            capacity,
            static,
            result,
            "PUBLIC-FINAL-TOPOLOGY-PAPER-KAIROS.md",
        )
    finally:
        topology.KAIROS_POLICY, topology.DEPLOYABLE, topology.POLICIES = saved
    return result


def validate_new_runs(mixed_result: dict[str, Any], topology_result: dict[str, Any]) -> None:
    for label, result, new_runs in (
        ("mixed", mixed_result, 32),
        ("topology", topology_result, 36),
    ):
        if result["new_paper_kairos_runs"] != new_runs:
            raise RuntimeError(f"{label} new-run count mismatch")
        for key in (
            "hard_invalid_runs",
            "runs_with_drops",
            "runs_with_timeouts",
            "runs_with_length_caps",
        ):
            if result[key] != 0:
                raise RuntimeError(f"{label} failed {key}: {result[key]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("mixed", "topology", "all"), default="all")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.exists():
        raise SystemExit(f"missing frozen protocol: {PROTOCOL}")

    mixed_result = None
    topology_result = None
    if args.stage in {"mixed", "all"}:
        mixed_result = run_mixed(mixed.DEFAULT_OUT.resolve(), args.jobs, args.force)
        print(json.dumps({"mixed": mixed_result}, indent=2), flush=True)
    if args.stage in {"topology", "all"}:
        topology_result = run_topology(
            topology.DEFAULT_OUT.resolve(), args.jobs, args.force
        )
        print(json.dumps({"topology": topology_result}, indent=2), flush=True)
    if mixed_result is not None and topology_result is not None:
        validate_new_runs(mixed_result, topology_result)
        print("paper-Kairos secondary extension passed all 68 new-run gates", flush=True)


if __name__ == "__main__":
    main()
