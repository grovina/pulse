"""deploy/launch.sh will not start a run without a stated ceiling."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "launch.sh"
ENV = {**os.environ, "TRAINER_CPU": "4", "TRAINER_MEMORY": "16Gi", "PATH": os.environ["PATH"]}


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=ENV)


def test_refuses_without_max_hours() -> None:
    r = _run("--dry-run", "--", "--gcs-bucket=b")
    assert r.returncode != 0 and "--max-hours is required" in r.stderr
    assert "run jobs execute" not in r.stdout


def test_refuses_a_bad_or_zero_ceiling_and_a_second_deadline() -> None:
    assert _run("--max-hours", "0", "--dry-run", "--").returncode != 0
    assert _run("--max-hours", "abc", "--dry-run", "--").returncode != 0
    r = _run("--max-hours", "4", "--dry-run", "--", "--deadline-hours=20")
    assert r.returncode != 0


def test_dry_run_prints_the_timeout_deadline_and_cost_bound() -> None:
    r = _run("--max-hours", "4", "--dry-run", "--", "--gcs-bucket=b", "--n-epochs=3")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "--task-timeout=15000s" in out  # 4 h + 600 s
    assert "--deadline-hours=4" in out and "--async" in out and "--n-epochs=3" in out
    assert "max cost about $1.56" in out  # 15000 s * (4*0.000018 + 16*0.000002)
