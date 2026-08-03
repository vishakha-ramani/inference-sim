#!/usr/bin/env python3
"""Run generalized joint-routing diagnostics on a simulator candidate trace."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYZER = HERE / "analyze" / "joint_routing_var_regret.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    inventory, trace = args.inventory, args.trace
    report = args.out / "var_regret_report.json"
    subprocess.run(
        [sys.executable, str(ANALYZER), "--inventory", str(inventory), "--trace", str(trace), "--out", str(report)],
        check=True,
    )
    print(f"diagnostic report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
