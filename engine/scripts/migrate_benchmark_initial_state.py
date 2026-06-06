#!/usr/bin/env python3
"""Upgrade benchmark JSON initial_state vectors to current STATE_DIM.

Chains: 17-D (pre-ACTH) → 18-D → 19-D (hepatic_output after lactate),
then rebuilds the unobserved internal-state tail (indices 19+, iter
55+) from MARKERS typicals — robust to tail reshuffles, not just
growth. Idempotent when already at STATE_DIM.

Run from apps/pulse/engine:  uv run python scripts/migrate_benchmark_initial_state.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ENGINE = Path(__file__).resolve().parent.parent
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from pulse.types import MARKER_INDEX, MARKERS, STATE_DIM  # noqa: E402

ACTH_TYPICAL = float(MARKERS[MARKER_INDEX["acth"]].typical)
HEPATIC_TYPICAL = float(MARKERS[MARKER_INDEX["hepatic_output"]].typical)

FILES = [
    Path("pulse/benchmark.dataset.generated.json"),
    Path("pulse/benchmark.dataset.template.json"),
]


def migrate_episode_initial(state: list) -> list:
    s = list(state)
    if len(s) == 17:
        s = s[:10] + [ACTH_TYPICAL] + s[10:]
    if len(s) == 18:
        s = s[:6] + [HEPATIC_TYPICAL] + s[6:]
    # Indices 0-18 are observed / cold-modelled markers carrying real
    # episode initial values — preserve them. Indices 19+ are unobserved
    # slow internal states (iter 55+) with no ground truth: their initial
    # value is definitionally the marker's `typical` (the model evolves
    # them forward from there). Rebuild the internal tail from typicals so
    # this is robust not just to tail *growth* but to tail *reshuffles*
    # (e.g. iter 56 split lumped glycogen_pool → liver+muscle, which
    # changed the meaning of indices 19-20 — a naive append would have
    # left stale 19-D values in the wrong slots).
    if len(s) >= 19:
        s = s[:19] + [float(MARKERS[i].typical) for i in range(19, STATE_DIM)]
    if len(s) != STATE_DIM:
        raise ValueError(f"initial_state length {len(state)}; cannot reach {STATE_DIM}")
    return s


def main() -> None:
    cwd = Path.cwd()
    if not (cwd / "pulse").is_dir():
        print("Run from apps/pulse/engine (pulse/ directory missing)", file=sys.stderr)
        sys.exit(1)

    for rel in FILES:
        path = cwd / rel
        if not path.is_file():
            print(f"Skip missing {path}", file=sys.stderr)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        episodes = data.get("episodes", [])
        n_migrated = 0
        for ep in episodes:
            raw = ep.get("initial_state")
            if not isinstance(raw, list):
                continue
            if len(raw) != STATE_DIM:
                ep["initial_state"] = migrate_episode_initial(raw)
                n_migrated += 1
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"{path.name}: {len(episodes)} episodes, {n_migrated} migrated to {STATE_DIM}-D")


if __name__ == "__main__":
    main()
