"""Write lab/graph.json from live types."""

from __future__ import annotations

import argparse
from pathlib import Path

from pulse.lab_export import LAB_DIR, write_lab


def main() -> None:
    parser = argparse.ArgumentParser(description="Write lab/graph.json")
    parser.add_argument("--out", type=Path, default=LAB_DIR, help="Lab directory")
    args = parser.parse_args()
    root = write_lab(args.out)
    print(f"wrote graph under {root}")


if __name__ == "__main__":
    main()
