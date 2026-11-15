# Stealth browser sidecar: Camoufox (a hardened Firefox) behind a tiny HTTP API.
# fetch_page escalates to it when a plain request is blocked by bot protection or the
# page only renders with JavaScript.  Internal network only — no host port.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgtk-3-0 libdbus-glib-1-2 libxt6 libasound2 libx11-xcb1 \
        fonts-liberation fonts-dejavu ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Everything — packages, the fpgen model camoufox writes beside them, the browser build in
# ~/.cache — belongs to the unprivileged user that runs it.
RUN useradd --create-home browser && mkdir /opt/venv && chown browser:browser /opt/venv
USER browser
WORKDIR /home/browser
ENV PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv \
    && pip install "camoufox[geoip]==0.5.9" "fastapi==0.135.1" "uvicorn==0.41.0" "lxml==6.0.2"

# Downloads the Camoufox browser build, its fingerprint model and the GeoIP database.
RUN python -m camoufox fetch

COPY --chown=browser:browser server.py .

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=4)"
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100", "--log-level", "info"]
