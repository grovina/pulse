#!/usr/bin/env bash
# Build + deploy Pulse to Cloud Run: the `trainer` job and the `engine`
# inference service. Source-upload build (no GitHub connection required).
#
# Run from the repo root, authenticated as dev@ (the gents box is). Idempotent:
# `run deploy` updates in place; the job is created-or-updated.
#
# The trainer job is the critical path, so it is updated FIRST and pinned to the
# immutable per-build SHA tag (not :latest) for run-to-SHA traceability. The
# engine inference service is updated LAST and best-effort: it currently fails
# to start on PORT=8080, and engine-first under `set -e` used to abort the
# script before the trainer was ever pinned. Set DEPLOY_ENGINE=0 to skip it.
#
# Config via env (defaults match the reference grovina deployment):
#   PROJECT       (default: grovina-pulse)
#   REGION        (default: europe-west1)
#   BUCKET        (default: grovina-pulse-data)
#   REPO          (default: pulse)
#   TASK_TIMEOUT  trainer task timeout, seconds (default: 158400 = 44h; the
#                 current recipe is 58 epochs at aux-every-12 and needs the
#                 headroom iter 98 used, not the old 4-CPU / 24h defaults)
#   DEPLOY_ENGINE 1 to deploy the engine service, 0 to skip (default: 1)
set -euo pipefail

PROJECT="${PROJECT:-grovina-pulse}"
REGION="${REGION:-europe-west1}"
BUCKET="${BUCKET:-grovina-pulse-data}"
REPO="${REPO:-pulse}"
TASK_TIMEOUT="${TASK_TIMEOUT:-158400}"
DEPLOY_ENGINE="${DEPLOY_ENGINE:-1}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AR="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}"

echo "==> Building images (gcloud builds submit)"
# Immutable per-build tag for traceability. The built-in $SHORT_SHA is empty for
# source-upload builds, so we pass the short git SHA explicitly (falls back to
# 'latest' outside a git tree). '-dirty' flags an uncommitted working tree.
TAG="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo latest)"
if [ "$TAG" != latest ] && ! git -C "$ROOT" diff --quiet 2>/dev/null; then
  TAG="${TAG}-dirty"
fi
echo "    image tag: ${TAG}"
# Run the build AS dev@ (a user-managed SA). Without this, `gcloud builds
# submit` uses the Compute Engine default SA, which dev@ cannot actAs. dev@
# holds artifactregistry.writer + storage.admin + logging.logWriter and can
# actAs itself (granted by setup-gcp.sh). Requires non-default build logging,
# satisfied by cloudbuild.yaml's CLOUD_LOGGING_ONLY.
BUILD_SA="pulse-dev@${PROJECT}.iam.gserviceaccount.com"
gcloud builds submit "$ROOT" \
  --project "$PROJECT" \
  --config "$ROOT/deploy/cloudbuild.yaml" \
  --service-account "projects/${PROJECT}/serviceAccounts/${BUILD_SA}" \
  --substitutions "_REGION=${REGION},_REPO=${REPO},_TAG=${TAG}"

echo "==> Creating/updating trainer job (pinned to ${AR}/trainer:${TAG})"
# Per-run hyperparameters are passed at execute time:
#   gcloud run jobs execute trainer --region "$REGION" \
#     --args=--gcs-bucket="$BUCKET",--gcs-object=training/jobs/<id>/model.pt
if gcloud run jobs describe trainer --project "$PROJECT" --region "$REGION" >/dev/null 2>&1; then
  verb=update
else
  verb=create
fi
gcloud run jobs "$verb" trainer \
  --project "$PROJECT" --region "$REGION" \
  --image "${AR}/trainer:${TAG}" \
  --service-account "trainer@${PROJECT}.iam.gserviceaccount.com" \
  --cpu 8 --memory 32Gi \
  --max-retries 0 --task-timeout "${TASK_TIMEOUT}s" \
  --quiet
echo "    trainer pinned to :${TAG}, task-timeout ${TASK_TIMEOUT}s"

# Engine LAST and best-effort: a failed engine deploy (it currently does not
# start on PORT=8080) must not undo or block the trainer update above. Disable
# `set -e` for this block so a non-zero exit only warns.
if [ "$DEPLOY_ENGINE" = 1 ]; then
  echo "==> Deploying engine service (best-effort)"
  set +e
  gcloud run deploy engine \
    --project "$PROJECT" --region "$REGION" \
    --image "${AR}/engine:${TAG}" \
    --service-account "engine@${PROJECT}.iam.gserviceaccount.com" \
    --set-env-vars "MODEL_URI=gs://${BUCKET}/models/prod.pt" \
    --no-allow-unauthenticated \
    --quiet
  engine_rc=$?
  set -e
  if [ "$engine_rc" -ne 0 ]; then
    echo "    WARNING: engine deploy failed (rc=$engine_rc); trainer is unaffected." >&2
  fi
else
  echo "==> Skipping engine service (DEPLOY_ENGINE=0)"
fi

echo "Done. Trainer job pinned to :${TAG}. Engine: $(gcloud run services describe engine --project "$PROJECT" --region "$REGION" --format='value(status.url)' 2>/dev/null || echo '(not deployed / describe failed)')"
