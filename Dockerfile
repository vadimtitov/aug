FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# System tools + hushed
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git jq procps \
    && curl -fsSL https://raw.githubusercontent.com/vadimtitov/hushed/main/install.sh | bash \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Non-root user created early so we can chown correctly
RUN useradd --create-home appuser

WORKDIR /app

# Install dependencies first (layer-cached until pyproject.toml changes)
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy application source
COPY aug/ ./aug/

# Install the project itself, then hand ownership to appuser
RUN uv sync --no-dev && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["/app/.venv/bin/uvicorn", "aug.app:app", "--host", "0.0.0.0", "--port", "8000"]
