"""The deadline is a ceiling on the whole job: it leaves a reserve, and it checks per step."""

from pulse.training.deadline import STARTUP_ALLOWANCE_S, RunDeadline


class Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_stops_with_the_reserve_left_and_the_reserve_covers_a_mocked_board() -> None:
    clock = Clock()
    d = RunDeadline(4.0, reserve_min=40.0, started_at=0.0, clock=clock)
    steps = 0
    while not d.must_stop(next_step_s=100.0):
        clock.t += 100.0
        steps += 1
    # Training stopped with at least the reserve left (and less than reserve + one step).
    assert d.remaining_s() >= d.reserve_s
    assert d.remaining_s() < d.reserve_s + 100.0
    # A final board of 34 minutes plus a 200 s validation fits before the ceiling.
    clock.t += 200.0 + 34 * 60.0
    assert d.remaining_s() >= 0.0
    assert clock.t <= 4 * 3600.0 - STARTUP_ALLOWANCE_S


def test_checks_inside_an_epoch_not_only_at_its_boundary() -> None:
    clock = Clock()
    d = RunDeadline(1.0, reserve_min=10.0, started_at=0.0, clock=clock)
    clock.t = 3600.0 - STARTUP_ALLOWANCE_S - 10 * 60.0 - 50.0
    assert d.must_stop(next_step_s=60.0)  # an hour-long epoch is mid-flight; one step more is too many
    assert not d.must_stop(next_step_s=40.0)


def test_a_longer_measured_board_raises_the_reserve() -> None:
    d = RunDeadline(4.0, reserve_min=10.0, started_at=0.0, clock=Clock())
    assert d.reserve_s == 600.0
    d.note_board(34 * 60.0)
    d.note_validation(200.0)
    assert d.reserve_s == 1.25 * 34 * 60.0 + 200.0


def test_no_deadline_never_stops() -> None:
    assert not RunDeadline(0.0, clock=Clock(1e9), started_at=0.0).must_stop(1e9)
