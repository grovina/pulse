"""A run's stated hours as a ceiling on the whole job, not only on training.

Iter 123's probe (trainer-69h6t) was launched with ``--deadline-hours=4``, finished its
three epochs at 3 h 38 min (so the old epoch-boundary check never fired), and then spent
34 minutes on the end-of-run board: 4 h 12 min billed against a 4 h approval. The clock
here starts at process start, covers the board, checkpoint writes and uploads that follow
training, and is checked before every step.
"""

from __future__ import annotations

import time
from typing import Callable

# Time between Cloud Run starting the task and Python importing this module (measured on
# trainer-69h6t: 27 s). Counted against the ceiling.
STARTUP_ALLOWANCE_S = 90.0
# The end-of-run board measured 34 min on the 4 vCPU job (trainer-69h6t); this default
# is that plus the checkpoint saves and uploads and a margin. A run that wrote a phase
# board uses its own measured time when that is longer.
DEFAULT_FINAL_RESERVE_MIN = 45.0
MEASURED_BOARD_MARGIN = 1.25

PROCESS_START = time.time()


class RunDeadline:
    """``hours`` after ``started_at`` the job must be finished, not merely training."""

    def __init__(
        self,
        hours: float,
        *,
        reserve_min: float = DEFAULT_FINAL_RESERVE_MIN,
        started_at: float = PROCESS_START,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.hours = float(hours)
        self.floor_reserve_s = float(reserve_min) * 60.0
        self.started_at = started_at
        self.clock = clock
        self.board_s = 0.0
        self.val_s = 0.0

    @property
    def active(self) -> bool:
        return self.hours > 0.0

    @property
    def ceiling(self) -> float:
        return self.started_at + self.hours * 3600.0 - STARTUP_ALLOWANCE_S

    def note_board(self, seconds: float) -> None:
        self.board_s = max(self.board_s, float(seconds))

    def note_validation(self, seconds: float) -> None:
        self.val_s = max(self.val_s, float(seconds))

    @property
    def reserve_s(self) -> float:
        """What must remain when training stops: a final validation, the board, saves."""
        measured = MEASURED_BOARD_MARGIN * self.board_s + self.val_s
        return max(self.floor_reserve_s, measured)

    def remaining_s(self) -> float:
        return self.ceiling - self.clock()

    def must_stop(self, next_step_s: float = 0.0) -> bool:
        """True when one more step of ``next_step_s`` would eat into the reserve."""
        return self.active and self.remaining_s() - next_step_s < self.reserve_s
