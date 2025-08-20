# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.4.20 AS uv
FROM python:3.11.5-slim-bookworm AS base

# pull security patches and keep image minimal
RUN apt-get update \
 && apt-get -y upgrade --no-install-recommends \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# bring uv binary
COPY --from=uv /uv /bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf \
    TRANSFORMERS_CACHE=/app/.hf/transformers

# non-root early, so we can COPY with correct ownership (no chown -R)
RUN useradd -m -U appuser
WORKDIR /app

# dependencies (will pick patched manylinux wheels where available)
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN /bin/uv sync --frozen || /bin/uv sync

# application
COPY --chown=appuser:appuser app.py config.py ./
COPY --chown=appuser:appuser models ./models
COPY --chown=appuser:appuser services ./services
COPY --chown=appuser:appuser healthcheck.py /usr/local/bin/healthcheck.py

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python /usr/local/bin/healthcheck.py

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
