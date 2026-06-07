#!/usr/bin/env bash
# gents box first-run hook (best-effort; gents wraps this and never lets a
# failure wedge the box). Wires the gcloud CLI to authenticate as the box's
# GCP service account the same way the host does, so deploys + `gcloud storage`
# work identically inside the box and on the Mac.
#
# Why this exists: the box mounts a read-only SA key and sets
# GOOGLE_APPLICATION_CREDENTIALS, which buys ADC for client libraries (the
# google-cloud-storage SDK pulse uses) — but the gcloud CLI itself needs an
# ACTIVE account (`gcloud auth activate-service-account`), which deploys via
# Cloud Build / Cloud Run rely on. Without this, `gcloud builds submit` dies at
# "No active gcloud auth".
#
# Fully generic: every host-specific value (project, config name, key path) is
# injected by the gents fleet entry as env, so this file carries no grovina IDs
# and the public repo stays runs-anywhere. All steps are idempotent.
set -uo pipefail

SRC=/secrets/gcp/key.json                         # ro mount (host catalog key)
KEY="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/key.json}"
CONFIG="${CLOUDSDK_ACTIVE_CONFIG_NAME:-default}"
PROJECT="${GCP_PROJECT_ID:-}"
SA="pulse-dev@${PROJECT}.iam.gserviceaccount.com"  # convention: the dev SA

if [ ! -f "$SRC" ]; then
  echo "pulse setup: no SA key at $SRC — skipping gcloud auth wiring" >&2
  exit 0
fi

# Copy the ro mount into the writable home volume so ~/.config/gcloud stays
# gent-owned and gcloud can write its config/credentials db underneath.
install -D -m600 "$SRC" "$KEY"

gcloud config configurations create "$CONFIG" --quiet 2>/dev/null || true
export CLOUDSDK_ACTIVE_CONFIG_NAME="$CONFIG"
gcloud auth activate-service-account --key-file="$KEY" --quiet \
  && gcloud config set account "$SA" --quiet \
  && { [ -n "$PROJECT" ] && gcloud config set project "$PROJECT" --quiet; } \
  && echo "pulse setup: gcloud active as $SA (config: $CONFIG, project: $PROJECT)" >&2 \
  || echo "pulse setup: gcloud activation failed (continuing)" >&2
