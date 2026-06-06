#!/usr/bin/env bash
set -euo pipefail

# Compare two pulse training runs end-to-end.
#
# Usage:
#   compare-runs.sh <new_job_id> [<baseline_job_id>] [--bucket <bucket>] [--out <path>]
#
# If <baseline_job_id> is omitted, the most recent prior job in the bucket
# (any job with both checkpoint.pt and benchmark-report.json) is used.
#
# Downloads checkpoint.pt and benchmark-report.json (when present) for both
# jobs into a temp dir, then invokes pulse.diagnostics compare. Falls back to
# probe-only comparison when one side has no benchmark report.

BUCKET="${GCS_BUCKET:-grovina-pulse}"
OUT=""
NEW_JOB=""
BASELINE_JOB=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)
      BUCKET="$2"; shift 2;;
    --out)
      OUT="$2"; shift 2;;
    --help|-h)
      sed -n '3,15p' "$0"; exit 0;;
    -*)
      echo "Unknown option: $1" >&2; exit 2;;
    *)
      if [[ -z "$NEW_JOB" ]]; then
        NEW_JOB="$1"
      elif [[ -z "$BASELINE_JOB" ]]; then
        BASELINE_JOB="$1"
      else
        echo "Unexpected positional argument: $1" >&2; exit 2
      fi
      shift;;
  esac
done

if [[ -z "$NEW_JOB" ]]; then
  echo "Usage: $0 <new_job_id> [<baseline_job_id>] [--bucket <bucket>] [--out <path>]" >&2
  exit 2
fi

ROOT_PREFIX="gs://${BUCKET}/training/jobs"

# Resolve baseline if not given: most recent job (by ID lexsort) with a
# checkpoint that is not the new job.
if [[ -z "$BASELINE_JOB" ]]; then
  echo "Resolving baseline (most recent prior job with checkpoint.pt)..." >&2
  CANDIDATES="$(gsutil ls "${ROOT_PREFIX}/" 2>/dev/null | awk -F/ '{ print $(NF-1) }' \
    | grep -v "^${NEW_JOB}$" | sort -r)"
  for c in $CANDIDATES; do
    if gsutil -q stat "${ROOT_PREFIX}/${c}/checkpoint.pt"; then
      BASELINE_JOB="$c"
      break
    fi
  done
  if [[ -z "$BASELINE_JOB" ]]; then
    echo "No baseline job with a checkpoint found in ${ROOT_PREFIX}." >&2
    exit 1
  fi
  echo "Baseline: ${BASELINE_JOB}" >&2
fi

WORK_DIR="$(mktemp -d -t pulse-compare-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "Workdir: ${WORK_DIR}" >&2

fetch_run () {
  local job="$1" prefix="$2"
  local root="${ROOT_PREFIX}/${job}"
  if gsutil -q stat "${root}/checkpoint.pt"; then
    gsutil -q cp "${root}/checkpoint.pt" "${WORK_DIR}/${prefix}-checkpoint.pt"
    echo "${WORK_DIR}/${prefix}-checkpoint.pt"
  else
    echo ""
  fi
}

fetch_report () {
  local job="$1" prefix="$2"
  local root="${ROOT_PREFIX}/${job}"
  if gsutil -q stat "${root}/benchmark-report.json"; then
    gsutil -q cp "${root}/benchmark-report.json" "${WORK_DIR}/${prefix}-benchmark.json"
    echo "${WORK_DIR}/${prefix}-benchmark.json"
  else
    echo ""
  fi
}

NEW_CKPT="$(fetch_run "$NEW_JOB" new)"
NEW_REPORT="$(fetch_report "$NEW_JOB" new)"
BASE_CKPT="$(fetch_run "$BASELINE_JOB" baseline)"
BASE_REPORT="$(fetch_report "$BASELINE_JOB" baseline)"

if [[ -z "$NEW_CKPT" && -z "$NEW_REPORT" ]]; then
  echo "New job ${NEW_JOB} has neither checkpoint nor benchmark report." >&2
  exit 1
fi
if [[ -z "$BASE_CKPT" && -z "$BASE_REPORT" ]]; then
  echo "Baseline job ${BASELINE_JOB} has neither checkpoint nor benchmark report." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ENGINE_DIR="${REPO_ROOT}/apps/pulse/engine"
PYTHON="${ENGINE_DIR}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || command -v python)"
fi

ARGS=(-m pulse.diagnostics compare
      --new-label "${NEW_JOB}" --baseline-label "${BASELINE_JOB}")
[[ -n "$NEW_CKPT" ]]    && ARGS+=(--new-checkpoint "$NEW_CKPT")
[[ -n "$NEW_REPORT" ]]  && ARGS+=(--new-report "$NEW_REPORT")
[[ -n "$BASE_CKPT" ]]   && ARGS+=(--baseline-checkpoint "$BASE_CKPT")
[[ -n "$BASE_REPORT" ]] && ARGS+=(--baseline-report "$BASE_REPORT")
[[ -n "$OUT" ]]         && ARGS+=(--out "$OUT")

cd "$ENGINE_DIR"
exec "$PYTHON" "${ARGS[@]}"
