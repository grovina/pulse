"""Serve the live lab viewer (student simulation + static graph)."""

from __future__ import annotations

import argparse
import os

import uvicorn

from pulse.lab_export import write_lab


def main() -> None:
    parser = argparse.ArgumentParser(description="Pulse lab viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=None, help="Optional student checkpoint (.pt)")
    args = parser.parse_args()
    if args.model:
        os.environ["PULSE_LAB_MODEL"] = args.model
    write_lab()
    from pulse.lab_server import app
    from pulse.lab_sim import get_lab_model
    _, meta = get_lab_model()
    kind = "checkpoint" if meta["trained"] else "untrained"
    print(f"lab at http://{args.host}:{args.port}")
    print(f"student {meta['version']} ({kind})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
