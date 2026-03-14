#!/bin/bash
# Start socat as a detached process so this init script exits immediately.
# socat listens on 0.0.0.0:9222 and forwards each connection to 127.0.0.1:9221
# (where Chrome binds CDP). Connections made before Chrome is ready will fail
# individually but socat itself keeps running.
setsid socat TCP-LISTEN:9222,bind=0.0.0.0,fork,reuseaddr TCP:127.0.0.1:9221 &
