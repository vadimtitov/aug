FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# --- Service dependencies ---
# hushed (https://github.com/vadimtitov/hushed) stores secrets on disk and redacts
# their values from all process output — prevents secrets leaking into LLM context.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git procps \
    && curl -fsSL https://raw.githubusercontent.com/vadimtitov/hushed/main/install.sh | bash \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Agent tools: system utilities available to the agent via run_bash ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        pandoc \
        imagemagick \
        tesseract-ocr \
        poppler-utils \
        ghostscript \
        libimage-exiftool-perl \
        jq \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Non-root user created early so we can chown correctly
RUN useradd --create-home appuser

WORKDIR /app

# Add the venv to PATH so installed binaries (uvicorn etc.) are available without full paths
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies first (layer-cached until pyproject.toml changes)
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project --extra agent-tools

# Copy application source
COPY aug/ ./aug/

# Install the project itself, then hand ownership to appuser
RUN uv sync --no-dev --extra agent-tools && \
    mkdir -p /app/browser-downloads && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "aug.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "25"]
