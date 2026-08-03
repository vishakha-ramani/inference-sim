"""Self-checks for generalized joint-routing VaR diagnostics."""

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("joint_regret", HERE / "joint_routing_var_regret.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
DRIVER = HERE.parent / "run_joint_routing_diagnostics.py"


def write(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(header); writer.writerows(rows)


TRACE_HEADER = [
    "request_id", "decode_instance", "prefill_instance", "chosen",
    "var_decode", "var_colloc_prefill", "var_prefill_pool", "var_total",
    "router_decode",
]


def candidate(rid, decode, prefill, chosen, total, router="m0"):
    # Components intentionally sum to total so the fixture is easy to inspect.
    return [rid, decode, prefill, chosen, total / 2, total / 4, total / 4, total, decode == router]


def test_generalized_topology_and_telescoping_attribution():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); inv = root / "inventory.csv"; trace = root / "trace.csv"
        write(inv, ["instance_id", "capability"], [
            ["p0", "prefill"], ["p1", "prefill"],
            ["m0", "mixed"], ["m1", "mixed"], ["m2", "mixed"],
        ])
        write(trace, TRACE_HEADER, [
            candidate("r0", "m0", "", "true", 0.9),
            candidate("r0", "m0", "p0", "false", 0.6),
            candidate("r0", "m0", "p1", "false", 0.8),
            candidate("r0", "m1", "local", "false", 0.2),
            candidate("r0", "m1", "p0", "false", 0.3),
            candidate("r0", "m1", "p1", "false", 0.4),
            candidate("r0", "m2", "local", "false", 0.5),
            candidate("r0", "m2", "p0", "false", 0.7),
            candidate("r0", "m2", "p1", "false", 0.4),
        ])
        report = MODULE.analyze(inv, trace)
        row = report["per_request"][0]
        assert report["topology"] == "2P3D"
        assert abs(row["var_regret"] - 0.7) < 1e-12
        assert abs(row["router_induced_var_regret"] - 0.4) < 1e-12
        assert row["action_disagreement"] and row["decode_reversal"]
        assert row["locality_reversal"]
        assert report["action_disagreement_fraction"] == 1.0
        assert row["chosen_action"] == "m0|local"  # blank is normalized
        assert "var_colloc_prefill_delta" in row
        assert "chosen_var_decode" in row and "best_var_prefill_pool" in row


def test_rejects_ambiguous_chosen_rows():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); inv = root / "inventory.csv"; trace = root / "trace.csv"
        write(inv, ["instance_id", "capability"], [["p0", "prefill"], ["m0", "mixed"]])
        write(trace, TRACE_HEADER, [
            candidate("r0", "m0", "local", "false", 0.2),
            candidate("r0", "m0", "p0", "false", 0.1),
        ])
        try:
            MODULE.analyze(inv, trace)
        except ValueError as error:
            assert "exactly one chosen" in str(error)
        else:
            raise AssertionError("ambiguous trace was accepted")


def test_driver_consumes_native_trace_directly():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); inv = root / "inventory.csv"; trace = root / "native.csv"
        write(inv, ["instance_id", "capability"], [["p0", "prefill"], ["m0", "mixed"]])
        write(trace, TRACE_HEADER, [
            candidate("r0", "m0", "", "true", 0.2),
            candidate("r0", "m0", "p0", "false", 0.1),
        ])
        result = subprocess.run(
            [sys.executable, str(DRIVER), "--out", str(root / "out"),
             "--inventory", str(inv), "--trace", str(trace)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads((root / "out" / "var_regret_report.json").read_text())
        assert report["topology"] == "1P1D"
        assert report["per_request"][0]["chosen_action"] == "m0|local"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok ", name)
    print("all passed")
