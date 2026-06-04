"""Tests for aug.utils.cdp URL helpers.

Behaviors under test:
  - is_ip_or_localhost recognises IPs and localhost, rejects hostnames
  - resolve_cdp_url leaves IP/localhost URLs untouched
  - resolve_cdp_url rewrites a hostname to its resolved IP (Chrome Host check)
"""

from unittest.mock import patch

from aug.utils.cdp import is_ip_or_localhost, resolve_cdp_url


def test_is_ip_or_localhost():
    assert is_ip_or_localhost("127.0.0.1")
    assert is_ip_or_localhost("10.0.0.5")
    assert is_ip_or_localhost("localhost")
    assert not is_ip_or_localhost("chromium")
    assert not is_ip_or_localhost("example.com")


def test_resolve_cdp_url_leaves_ip_untouched():
    assert resolve_cdp_url("http://127.0.0.1:9222") == "http://127.0.0.1:9222"
    assert resolve_cdp_url("http://localhost:9222") == "http://localhost:9222"


def test_resolve_cdp_url_rewrites_hostname():
    with patch("aug.utils.cdp.socket.gethostbyname", return_value="172.18.0.3") as gethost:
        result = resolve_cdp_url("http://chromium:9222")
    gethost.assert_called_once_with("chromium")
    assert result == "http://172.18.0.3:9222"
