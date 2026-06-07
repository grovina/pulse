#!/usr/bin/env bash
# Provision (idempotently) the GCP service accounts + IAM that Pulse needs to
# run on Cloud Run. Safe to re-run anytime — every step tolerates "already
# exists" and IAM bindings are declarative.
#
# Identity model (all in ONE isolated project, so names are short — the project
# already means "pulse"):
#   pulse-dev@ — developer / CI identity. Builds + deploys, reads/writes GCS.
#                This is what the gents box authenticates as. (A bare `dev`
#                is < GCP's 6-char SA-id minimum, hence the qualifier; the
#                runtimes below stay short.)
#   engine@    — runtime SA for the `engine` Cloud Run service (reads the model).
#   trainer@   — runtime SA for the `trainer` Cloud Run job (writes artifacts).
#
# Must run as a human/owner account — NOT as pulse-dev@ (can't grant itself).
#
# Config via env (sensible defaults for the reference grovina deployment):
#   PROJECT  (default: grovina-pulse)
#   REGION   (default: europe-west1)
#   BUCKET   (default: grovina-pulse-data)
set -euo pipefail

PROJECT="${PROJECT:-grovina-pulse}"
REGION="${REGION:-europe-west1}"
BUCKET="${BUCKET:-grovina-pulse-data}"

ACTIVE="$(gcloud config get-value account 2>/dev/null || true)"
case "$ACTIVE" in
  pulse-dev@*) echo "Refusing to run as $ACTIVE — IAM setup needs an owner account." >&2; exit 1 ;;
esac

echo "Project=$PROJECT Region=$REGION Bucket=$BUCKET  (as $ACTIVE)"
NUM="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"

dev="pulse-dev@${PROJECT}.iam.gserviceaccount.com"
engine="engine@${PROJECT}.iam.gserviceaccount.com"
trainer="trainer@${PROJECT}.iam.gserviceaccount.com"

# --- service accounts (idempotent) ----------------------------------------
mk_sa() { # id, display
  gcloud iam service-accounts create "$1" --project "$PROJECT" \
    --display-name "$2" 2>/dev/null || true
}
mk_sa pulse-dev "Pulse dev/CI"
mk_sa engine    "Pulse engine (Cloud Run service runtime)"
mk_sa trainer   "Pulse trainer (Cloud Run job runtime)"

# --- project-level roles for dev@ ------------------------------------------
proj_bind() { # member, role
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member "$1" --role "$2" --condition=None --quiet >/dev/null
}
for role in \
  roles/run.developer \
  roles/cloudbuild.builds.editor \
  roles/artifactregistry.writer \
  roles/storage.admin \
  roles/logging.viewer; do
  proj_bind "serviceAccount:${dev}" "$role"
done

# dev@ deploys services that RUN AS the runtime SAs → needs actAs on them.
sa_user() { # actor, target_sa
  gcloud iam service-accounts add-iam-policy-binding "$2" --project "$PROJECT" \
    --member "serviceAccount:$1" --role roles/iam.serviceAccountUser --quiet >/dev/null
}
sa_user "$dev" "$engine"
sa_user "$dev" "$trainer"

# --- bucket-scoped roles for the runtime SAs --------------------------------
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${engine}"  --role roles/storage.objectViewer --quiet >/dev/null || \
  echo "  (skip: bucket gs://${BUCKET} not created yet — re-run after it exists)" >&2
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${trainer}" --role roles/storage.objectAdmin  --quiet >/dev/null || true

# --- Cloud Build's build identity ------------------------------------------
# `gcloud builds submit` runs as Cloud Build's SA. Ensure it exists and can
# deploy to Cloud Run + push images + act as the runtime SAs.
gcloud beta services identity create --service=cloudbuild.googleapis.com \
  --project "$PROJECT" --quiet >/dev/null 2>&1 || true
cb="${NUM}@cloudbuild.gserviceaccount.com"
proj_bind "serviceAccount:${cb}" roles/run.developer
proj_bind "serviceAccount:${cb}" roles/artifactregistry.writer
sa_user "$cb" "$engine"
sa_user "$cb" "$trainer"
dev_user_on_cb() { # let dev@ trigger builds that run as the build SA
  gcloud iam service-accounts add-iam-policy-binding "$cb" --project "$PROJECT" \
    --member "serviceAccount:${dev}" --role roles/iam.serviceAccountUser --quiet >/dev/null 2>&1 || true
}
dev_user_on_cb

echo "Done. SAs: $dev / $engine / $trainer"
