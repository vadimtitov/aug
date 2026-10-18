FROM lscr.io/linuxserver/chromium:latest

# CDP proxy: 0.0.0.0:9222 → Chrome on 127.0.0.1:9221, via the nginx this image
# already runs for its web UI — see cdp-proxy.conf.
COPY cdp-proxy.conf /etc/nginx/conf.d/cdp-proxy.conf
COPY wrapped-chromium /usr/bin/wrapped-chromium
RUN chmod +x /usr/bin/wrapped-chromium
