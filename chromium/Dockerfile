FROM lscr.io/linuxserver/chromium:latest

# Install socat to proxy CDP from 0.0.0.0:9222 → 127.0.0.1:9221
# Chrome 94+ ignores --remote-debugging-address and only binds CDP to loopback.
# socat makes it reachable from other containers on the Docker network.
RUN apt-get update -qq && apt-get install -y --no-install-recommends socat && rm -rf /var/lib/apt/lists/*

COPY cdp-proxy.sh /custom-cont-init.d/99-cdp-proxy.sh
COPY wrapped-chromium /usr/bin/wrapped-chromium
RUN chmod +x /custom-cont-init.d/99-cdp-proxy.sh /usr/bin/wrapped-chromium
