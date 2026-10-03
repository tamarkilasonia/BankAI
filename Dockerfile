# syntax=docker/dockerfile:1
#
# BankAI - one image for the API (backend), the MCP tool server and the init job.
# A prototype with MOCK data only - not a production deployment.
#
#   docker build -t bankai .                          # API + MCP server, no local speech models
#   docker build --build-arg WITH_VOICE=true -t bankai:voice .   # + Whisper and Piper (larger)
#
# Two stages: dependencies are installed into a virtualenv in a builder stage, and only
# that virtualenv and the application files are copied into the final image.

ARG PYTHON_IMAGE=python:3.12.15-slim-bookworm

# --- 1. dependencies ---------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Copied before the code, so a code change does not reinstall every dependency.
COPY requirements.txt requirements-voice.txt ./
ARG WITH_VOICE=false
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
    && if [ "$WITH_VOICE" = "true" ]; then pip install -r requirements-voice.txt; fi

# --- 2. runtime -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="BankAI" \
      org.opencontainers.image.description="Agentic banking voice assistant prototype - MOCK data only"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp

# An unprivileged user. The secrets directory belongs to it, so a Docker volume mounted
# there starts with that owner and mode 0700.
RUN groupadd --system --gid 10001 bankai \
    && useradd --system --uid 10001 --gid bankai --no-create-home --shell /usr/sbin/nologin bankai \
    && mkdir -p /var/lib/bankai/secrets \
    && chown bankai:bankai /var/lib/bankai/secrets \
    && chmod 0700 /var/lib/bankai/secrets

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Only what runs. .dockerignore keeps .env, .git, tests and models out of the build context.
COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY scripts ./scripts
COPY knowledge_base ./knowledge_base
COPY frontend ./frontend
# Compile once at build time: the containers run with a read-only file system.
RUN python -m compileall -q app scripts alembic

USER bankai
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# One worker on purpose: conversations and pending confirmations live in process memory.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--timeout-graceful-shutdown", "10"]
