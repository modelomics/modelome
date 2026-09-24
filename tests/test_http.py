from urllib.request import Request

import pytest

from modelome.http import HttpFailure, _redacted_url, _SafeRedirectHandler


def test_sensitive_query_values_are_redacted_from_diagnostic_urls() -> None:
    redacted = _redacted_url(
        "https://api.example.test/works?api_key=top-secret&cursor=opaque&token=also-secret"
    )

    assert "top-secret" not in redacted
    assert "also-secret" not in redacted
    assert "cursor=opaque" in redacted
    assert redacted.count("REDACTED") == 2


def test_cross_origin_redirect_strips_credentials_and_custom_headers() -> None:
    request = Request(
        "https://api.example.test/models",
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer secret",
            "X-API-Key": "secret",
            "If-None-Match": '"private-validator"',
            "Range": "bytes=1024-2047",
            "User-Agent": "registry-test",
        },
    )

    redirected = _SafeRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://cdn.example.test/models",
    )

    assert redirected is not None
    headers = {name.casefold(): value for name, value in redirected.header_items()}
    assert headers == {
        "accept": "application/json",
        "range": "bytes=1024-2047",
        "user-agent": "registry-test",
    }


def test_same_origin_redirect_preserves_auth_with_default_port_equivalence() -> None:
    request = Request(
        "https://api.example.test/models",
        headers={"Authorization": "Bearer secret"},
    )

    redirected = _SafeRedirectHandler().redirect_request(
        request,
        None,
        307,
        "Temporary Redirect",
        {},
        "https://api.example.test:443/v2/models",
    )

    assert redirected is not None
    assert redirected.get_header("Authorization") == "Bearer secret"


def test_redirect_validator_runs_before_the_redirect_is_followed() -> None:
    request = Request("https://public.example/models")

    def reject(url: str) -> None:
        assert url == "http://127.0.0.1/secrets"
        raise HttpFailure("blocked")

    with pytest.raises(HttpFailure, match="blocked"):
        _SafeRedirectHandler(reject).redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://127.0.0.1/secrets",
        )
