# `llm-evalops` is a path dependency (../llm-evalops) outside this build context, so pass it as
# a named context. Train first so models/champion.joblib exists (`make train`):
#   docker build --build-context llm-evalops=../llm-evalops -t fraud-scoring-service:latest .
# ---- build stage: resolve and install dependencies with uv -------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# `../llm-evalops` relative to /app is /llm-evalops.
COPY --from=llm-evalops pyproject.toml README.md /llm-evalops/
COPY --from=llm-evalops src /llm-evalops/src

# Third-party dependencies first, so this layer is cached across source changes.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --no-editable

COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---- runtime stage: slim image with the virtualenv, config and the trained model ----------
FROM python:3.12-slim AS runtime

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app monitoring.yaml evals.yaml ./
COPY --chown=app:app models/champion.joblib ./models/champion.joblib

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/champion.joblib \
    MONITORING_CONFIG=/app/monitoring.yaml

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["uvicorn", "fraudscore.api:app", "--host", "0.0.0.0", "--port", "8000"]
