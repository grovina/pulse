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
  roles/logging.viewer \
  roles/logging.logWriter; do
  proj_bind "serviceAccount:${dev}" "$role"
done

# dev@ deploys services that RUN AS the runtime SAs → needs actAs on them.
sa_user() { # actor, target_sa
  gcloud iam service-accounts add-iam-policy-binding "$2" --project "$PROJECT" \
    --member "serviceAccount:$1" --role roles/iam.serviceAccountUser --quiet >/dev/null
}
sa_user "$dev" "$engine"
sa_user "$dev" "$trainer"
# Builds run AS dev@ (a user-managed SA — see the Cloud Build note below), so
# dev@ must be able to actAs *itself*.
sa_user "$dev" "$dev"

# --- bucket-scoped roles for the runtime SAs --------------------------------
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${engine}"  --role roles/storage.objectViewer --quiet >/dev/null || \
  echo "  (skip: bucket gs://${BUCKET} not created yet — re-run after it exists)" >&2
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${trainer}" --role roles/storage.objectAdmin  --quiet >/dev/null || true

# --- Cloud Build's build identity ------------------------------------------
# Modern `gcloud builds submit` no longer defaults to the legacy Google-managed
# Cloud Build SA ({NUM}@cloudbuild) — it uses the Compute Engine default SA,
# which dev@ is not granted actAs on (and the Google-managed Cloud Build SA
# cannot be passed as a user-specified --service-account). So deploy.sh runs the
# build AS dev@ itself (a user-managed SA dev@ can actAs, granted above), which
# already holds artifactregistry.writer + storage.admin + logging.logWriter.
# Nothing to provision here beyond those project bindings.

echo "Done. SAs: $dev / $engine / $trainer (builds run as $dev)"
