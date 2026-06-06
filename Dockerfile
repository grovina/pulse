# syntax=docker/dockerfile:1

# ===========================================================================
# Stage 1: Install Python dependencies
# ===========================================================================
FROM python:3.12-slim AS deps

WORKDIR /app

COPY pyproject.toml ./

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch && \
    pip install --no-cache-dir numpy fastapi uvicorn google-cloud-storage

# ===========================================================================
# Stage 2: Runtime base (shared between server + train targets)
# ===========================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Force unbuffered stdout/stderr so print() lines reach Cloud Logging in real
# time during long-running training jobs. Otherwise Python line-buffers when
# stdout is a pipe and a 9h run looks silent until the very end.
ENV PYTHONUNBUFFERED=1

COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY pulse/ pulse/

# ===========================================================================
# Stage 3a: Inference server (default target — used by pulse-engine service)
# ===========================================================================
FROM runtime AS server

EXPOSE ${PORT:-8080}

CMD ["sh", "-c", "uvicorn pulse.server:app --host 0.0.0.0 --port ${PORT:-8080}"]

# ===========================================================================
# Stage 3b: Trainer (used by pulse-trainer Cloud Run job)
#
# ENTRYPOINT makes the image behave like calling `python -m pulse.train`
# directly: at execute time pass only flags via `gcloud run jobs execute
# --args=--gcs-bucket=...,--gcs-object=...`. Same shape as running locally.
# ===========================================================================
FROM runtime AS train

ENTRYPOINT ["python", "-m", "pulse.train"]
