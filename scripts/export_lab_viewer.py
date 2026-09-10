"""Export the lab-viewer graph and teacher protocol runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from pulse.lab_export import LAB_DIR, write_lab


def main() -> None:
    parser = argparse.ArgumentParser(description="Write lab/graph.json and lab/runs/*.json")
    parser.add_argument(
        "--out", type=Path, default=LAB_DIR,
        help="Lab directory (default: repo lab/)",
    )
    args = parser.parse_args()
    root = write_lab(args.out)
    print(f"wrote graph and runs under {root}")


if __name__ == "__main__":
    main()
