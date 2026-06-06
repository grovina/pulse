#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 1 ]]; then
  BUCKET="${GCS_BUCKET:-grovina-pulse}"
  JOB_ID="$1"
elif [[ $# -eq 2 ]]; then
  BUCKET="${GCS_BUCKET:-$1}"
  JOB_ID="$2"
else
  echo "Usage: $0 <job_id>   (bucket: \$GCS_BUCKET or default grovina-pulse)" >&2
  echo "   or: $0 <bucket> <job_id>" >&2
  exit 1
fi

ROOT="gs://${BUCKET}/training/jobs/${JOB_ID}"
REPORT_PATH="${ROOT}/benchmark-report.json"
CHECKPOINT_PATH="${ROOT}/checkpoint.pt"
META_PATH="${ROOT}/meta.json"

echo "Job artifacts root: ${ROOT}" >&2
if gsutil -q stat "${META_PATH}"; then
  echo "meta.json=present" >&2
else
  echo "meta.json=absent" >&2
fi

if gsutil -q stat "${REPORT_PATH}"; then
  echo "" >&2
  echo "Benchmark report found: ${REPORT_PATH}" >&2
  report_json="$(gsutil cat "${REPORT_PATH}")"

  passed="$(printf '%s' "${report_json}" | jq -r '.gate.passed')"
  overall_mape="$(printf '%s' "${report_json}" | jq -r '.overall_weighted_mape')"
  verifier="$(printf '%s' "${report_json}" | jq -r '.verifier.overall_score')"
  failures="$(printf '%s' "${report_json}" | jq -r '.gate.failures[]?' || true)"

  echo "gate.passed=${passed}" >&2
  echo "overall_weighted_mape=${overall_mape}" >&2
  echo "verifier.overall_score=${verifier}" >&2
  if [[ -n "${failures}" ]]; then
    echo "failures:" >&2
    while IFS= read -r line; do
      [[ -n "${line}" ]] && echo "  - ${line}" >&2
    done <<< "${failures}"
  fi
else
  echo "Benchmark report not found: ${REPORT_PATH}" >&2
fi

if gsutil -q stat "${CHECKPOINT_PATH}"; then
  echo "checkpoint=present (${CHECKPOINT_PATH})" >&2
else
  echo "checkpoint=absent (${CHECKPOINT_PATH})" >&2
fi

