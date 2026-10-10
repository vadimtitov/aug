"""Read secrets from hushed at runtime, with a fallback to the process environment.

Secrets added via ``hushed add`` after the process started are not in
``os.environ`` — ``hushed run`` injects them only at process start.  This module
reads them from hushed on demand, so a new OAuth provider can be configured
mid-conversation without restarting the container.
"""

import logging
import os
import re
import subprocess
import tempfile

logger = logging.getLogger(__name__)

# Env-var names are uppercase ASCII letters, digits, and underscores.
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

# Cache: name → value.  Only positive results are cached — a missing secret is
# retried on the next call so that a secret added later is picked up.
_cache: dict[str, str] = {}


def read_secret(name: str) -> str:
    """Read a secret by env-var name.

    Checks ``os.environ`` first (covers secrets present at process start), then
    falls back to hushed (covers secrets added later via ``hushed add``).
    Positive results are cached; empty results are not, so a secret added after
    the first lookup is found on the next call.
    """
    cached = _cache.get(name)
    if cached is not None:
        return cached

    value = os.environ.get(name, "")
    if not value:
        try:
            value = _read_from_hushed(name)
        except Exception:
            logger.warning("failed to read secret %s from hushed", name)
            value = ""

    if value:
        _cache[name] = value
    return value


# --- Private ---


def _read_from_hushed(name: str) -> str:
    """Read a single secret from hushed via a temp file.

    ``hushed run`` redacts stdout/stderr, so we write the value to a temp file
    and read it back — the file gets the real value; only terminal output is
    redacted.
    """
    if not _ENV_NAME_RE.match(name):
        return ""

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            ["hushed", "run", "--", "sh", "-c", f'printf "%s" "${name}" > "{tmp_path}"'],
            check=True,
            capture_output=True,
            timeout=5,
        )
        with open(tmp_path) as f:
            return f.read()
    except Exception:
        logger.warning("failed to read secret %s from hushed", name)
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
