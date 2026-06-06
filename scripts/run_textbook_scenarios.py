#!/usr/bin/env python3
"""Run merged textbook + flow scenarios; optional JSON summary on stdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo layout: scripts/ lives next to package pulse/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pulse.knowledge.textbook_scenarios import run_all_scenarios


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable summary (default: human text)",
    )
    args = p.parse_args()
    results = run_all_scenarios(seed=args.seed)
    if args.json:
        payload = [
            {
                "name": r.name,
                "pass_rate": r.pass_rate,
                "source": r.source,
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "value": c.value,
                        "threshold": c.threshold,
                    }
                    for c in r.checks
                ],
            }
            for r in results
        ]
        print(json.dumps(payload, indent=2))
        return

    for r in results:
        print(f"\n{r.name}  pass_rate={r.pass_rate:.2f}")
        print(f"  {r.source}")
        for c in r.checks:
            mark = "OK" if c.passed else "FAIL"
            print(f"  [{mark}] {c.name}: value={c.value:.4g} threshold={c.threshold:.4g}")


if __name__ == "__main__":
    main()
