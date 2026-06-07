# syntax=docker/dockerfile:1

# ===========================================================================
# Stage 1: Install dependencies with uv (single source of truth: pyproject +
# uv.lock). TORCH selects the PyTorch build — cpu (default; what Cloud Run
# runs) or gpu (CUDA, for NVIDIA hosts):  docker build --build-arg TORCH=gpu …
# ===========================================================================
FROM python:3.12-slim AS deps

ARG TORCH=cpu
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
RUN pip install --no-cache-dir uv

# Resolve deps first (cached layer that only busts when the lock changes), then
# install the project. `--frozen` requires uv.lock to be up to date.
COPY pyproject.toml uv.lock ./
RUN if [ "$TORCH" = "gpu" ]; then G="--no-default-groups --group gpu"; fi; \
    uv sync --frozen --no-install-project $G
COPY pulse/ pulse/
RUN if [ "$TORCH" = "gpu" ]; then G="--no-default-groups --group gpu"; fi; \
    uv sync --frozen $G

# ===========================================================================
# Stage 2: Runtime base (shared between server + train targets) — carries the
# resolved virtualenv and the source. The venv on PATH makes `python` /
# `uvicorn` resolve to the installed deps.
# ===========================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Force unbuffered stdout/stderr so print() lines reach Cloud Logging in real
# time during long-running training jobs. Otherwise Python line-buffers when
# stdout is a pipe and a 9h run looks silent until the very end.
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

COPY --from=deps /app/.venv /app/.venv
COPY --from=deps /app/pulse /app/pulse
# Iter recipes (committed training flag sets). The trainer loads one via
# `--spec train/spec.json`, so the recipe must ship in the image; copied from
# the build context (not the deps stage, which only carries pulse/).
COPY train/ /app/train/

# ===========================================================================
# Stage 3a: Inference server (default target — used by the `engine` service)
# ===========================================================================
FROM runtime AS server

EXPOSE ${PORT:-8080}

CMD ["sh", "-c", "uvicorn pulse.server:app --host 0.0.0.0 --port ${PORT:-8080}"]

# ===========================================================================
# Stage 3b: Trainer (used by the `trainer` Cloud Run job)
#
# ENTRYPOINT makes the image behave like calling `python -m pulse.train`
# directly: at execute time pass only flags via `gcloud run jobs execute
# --args=--gcs-bucket=...,--gcs-object=...`. Same shape as running locally.
# ===========================================================================
FROM runtime AS train

ENTRYPOINT ["python", "-m", "pulse.train"]
