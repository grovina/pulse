#!/usr/bin/env bash
# Build + deploy Pulse to Cloud Run: the `engine` inference service and the
# `trainer` job. Source-upload build (no GitHub connection required).
#
# Run from the repo root, authenticated as dev@ (the gents box is). Idempotent:
# `run deploy` updates in place; the job is created-or-updated.
#
# Config via env (defaults match the reference grovina deployment):
#   PROJECT  (default: grovina-pulse)
#   REGION   (default: europe-west1)
#   BUCKET   (default: grovina-pulse-data)
#   REPO     (default: pulse)
set -euo pipefail

PROJECT="${PROJECT:-grovina-pulse}"
REGION="${REGION:-europe-west1}"
BUCKET="${BUCKET:-grovina-pulse-data}"
REPO="${REPO:-pulse}"

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

echo "==> Deploying engine service"
gcloud run deploy engine \
  --project "$PROJECT" --region "$REGION" \
  --image "${AR}/engine:latest" \
  --service-account "engine@${PROJECT}.iam.gserviceaccount.com" \
  --set-env-vars "MODEL_URI=gs://${BUCKET}/models/prod.pt" \
  --no-allow-unauthenticated \
  --quiet

echo "==> Creating/updating trainer job"
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
  --image "${AR}/trainer:latest" \
  --service-account "trainer@${PROJECT}.iam.gserviceaccount.com" \
  --cpu 4 --memory 16Gi \
  --max-retries 0 --task-timeout 14400s \
  --quiet

echo "Done. Engine: $(gcloud run services describe engine --project "$PROJECT" --region "$REGION" --format='value(status.url)' 2>/dev/null || echo '(describe failed)')"
