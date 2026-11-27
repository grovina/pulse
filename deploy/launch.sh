#!/usr/bin/env bash
# Launch a trainer run with a hard ceiling on billed time.
#
#   deploy/launch.sh --max-hours 4 [--dry-run] -- --gcs-bucket=... --gcs-object=... <train flags>
#
# --max-hours H is required. It does two things:
#   * train.py gets --deadline-hours=H: the whole job (training, then the final validation,
#     board, checkpoint and uploads) is planned to end by H hours after process start, and
#     training stops early enough to leave --final-reserve-min for those steps.
#   * Cloud Run gets --task-timeout = H hours + MARGIN_S (10 min for startup and a slow
#     upload), so the platform kills the task at the ceiling even if the code misbehaves.
# Everything after `--` goes to the trainer as is (a --deadline-hours there is refused).
# --dry-run prints the command and the cost bound and runs nothing.
#
# Trainer-69h6t (2026-11-27) was launched with --deadline-hours=4 under a 26 h task
# timeout and billed 4 h 12 min: the deadline covered training only, and the 34 min
# board came after it. This script exists so the stated hours is the cap.
#
# Env: PROJECT (grovina-pulse), REGION (europe-west1), JOB (trainer),
# TRAINER_CPU / TRAINER_MEMORY (read from the job when unset; set for offline use).
set -euo pipefail

PROJECT="${PROJECT:-grovina-pulse}"
REGION="${REGION:-europe-west1}"
JOB="${JOB:-trainer}"
MARGIN_S=600

# Cloud Run jobs, tier-1 region (europe-west1), 2026-11: $0.000018 per vCPU-second and
# $0.000002 per GiB-second. Source: https://cloud.google.com/run/pricing
# (Jobs bill the allocated resources for the whole task time.) Check it before trusting a quote.
PRICE_VCPU_S="0.000018"
PRICE_GIB_S="0.000002"

usage() { echo "usage: $0 --max-hours H [--dry-run] -- <trainer args>" >&2; exit 2; }

MAX_HOURS=""
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --max-hours) MAX_HOURS="${2:-}"; shift 2 || usage ;;
    --max-hours=*) MAX_HOURS="${1#*=}"; shift ;;
    --dry-run) DRY=1; shift ;;
    --) shift; break ;;
    *) echo "unknown option: $1 (trainer args go after --)" >&2; usage ;;
  esac
done
[ -n "$MAX_HOURS" ] || { echo "--max-hours is required: it is the ceiling on billed time" >&2; usage; }
[[ "$MAX_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk "BEGIN{exit !($MAX_HOURS > 0)}" \
  || { echo "--max-hours must be a positive number" >&2; usage; }
for a in "$@"; do
  case "$a" in
    --deadline-hours*) echo "do not pass --deadline-hours; --max-hours sets it" >&2; exit 2 ;;
  esac
done

TIMEOUT_S="$(awk "BEGIN{printf \"%d\", $MAX_HOURS*3600 + $MARGIN_S}")"

if [ -z "${TRAINER_CPU:-}" ] || [ -z "${TRAINER_MEMORY:-}" ]; then
  YAML="$(gcloud run jobs describe "$JOB" --project "$PROJECT" --region "$REGION" --format=yaml)"
  TRAINER_CPU="$(printf '%s\n' "$YAML" | awk '/^ *cpu:/{gsub(/[\047"]/,"",$2); print $2; exit}')"
  TRAINER_MEMORY="$(printf '%s\n' "$YAML" | awk '/^ *memory:/{gsub(/[\047"]/,"",$2); print $2; exit}')"
fi
MEM_GIB="${TRAINER_MEMORY%Gi}"
COST="$(awk "BEGIN{printf \"%.2f\", $TIMEOUT_S * ($TRAINER_CPU*$PRICE_VCPU_S + $MEM_GIB*$PRICE_GIB_S)}")"

# --args is comma-split by gcloud; the ^|^ prefix picks a delimiter that cannot occur in a flag.
ARGS="^|^--deadline-hours=${MAX_HOURS}"
for a in "$@"; do ARGS="${ARGS}|${a}"; done

CMD=(gcloud run jobs execute "$JOB" --project "$PROJECT" --region "$REGION" --async
     "--task-timeout=${TIMEOUT_S}s" "--args=${ARGS}")

echo "ceiling: ${MAX_HOURS} h of training job + ${MARGIN_S}s margin = ${TIMEOUT_S}s task timeout"
echo "size: ${TRAINER_CPU} vCPU, ${TRAINER_MEMORY}; max cost about \$${COST}" \
     "(prices \$${PRICE_VCPU_S}/vCPU-s, \$${PRICE_GIB_S}/GiB-s)"
if [ "$DRY" = 1 ]; then
  printf 'dry run: '; printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
"${CMD[@]}"
