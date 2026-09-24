from email.message import Message
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from modelome.http import HttpClient, HttpFailure, _redacted_url, _SafeRedirectHandler


class _Response:
    status = 200

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.headers = Message()
        self.response_url = "https://api.github.com/repos/acme/model/commits/main"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _limit: int) -> bytes:
        return self.body

    @property
    def url(self) -> str:
        return self.response_url


class _Opener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def open(self, _request, timeout: float):
        self.calls += 1
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def _client_with_opener(*responses):
    client = HttpClient(attempts=1, sleep=lambda _delay: None)
    opener = _Opener(responses)
    client._opener = opener
    return client, opener


def test_anonymous_github_commit_get_is_reused_and_explicitly_clearable() -> None:
    client, opener = _client_with_opener(
        _Response(b'{"sha":"abc"}'), _Response(b'{"sha":"abc"}')
    )
    url = "https://api.github.com/repos/acme/model/commits/main"

    first = client.get(url)
    second = client.get(url)

    assert first is second
    assert opener.calls == 1
    client.clear_github_commit_cache()
    client.get(url)
    assert opener.calls == 2


def test_authenticated_github_commit_response_never_enters_or_uses_cache() -> None:
    client, opener = _client_with_opener(
        _Response(b'{"sha":"private"}'),
        _Response(b'{"sha":"public"}'),
        _Response(b'{"sha":"keyed"}'),
    )
    url = "https://api.github.com/repos/acme/model/commits/main"

    private = client.get(url, headers={"Authorization": "Bearer top-secret"})
    public = client.get(url)
    keyed = client.get(url, headers={"X-API-Key": "another-secret"})

    assert private.body == b'{"sha":"private"}'
    assert public.body == b'{"sha":"public"}'
    assert keyed.body == b'{"sha":"keyed"}'
    assert opener.calls == 3


def test_github_commit_errors_and_other_urls_are_not_cached() -> None:
    failure = HTTPError(
        "https://api.github.com/repos/acme/model/commits/main",
        403,
        "rate limited",
        Message(),
        None,
    )
    client, opener = _client_with_opener(failure, failure)
    url = "https://api.github.com/repos/acme/model/commits/main"
    for _ in range(2):
        with pytest.raises(HttpFailure):
            client.get(url)
    assert opener.calls == 2

    other_client, other_opener = _client_with_opener(
        _Response(b"one"), _Response(b"two")
    )
    contents_url = "https://api.github.com/repos/acme/model/contents/README.md"
    assert other_client.get(contents_url).body == b"one"
    assert other_client.get(contents_url).body == b"two"
    assert other_opener.calls == 2


def test_github_commit_cache_requires_exact_path_and_same_origin_response() -> None:
    nested_client, nested_opener = _client_with_opener(
        _Response(b"one"), _Response(b"two")
    )
    nested_url = "https://api.github.com/repos/acme/model/commits/main/statuses"
    assert nested_client.get(nested_url).body == b"one"
    assert nested_client.get(nested_url).body == b"two"
    assert nested_opener.calls == 2

    redirected = _Response(b"redirected")
    redirected.response_url = "https://cdn.example.test/commit"
    same_origin = _Response(b"same-origin")
    client, opener = _client_with_opener(redirected, same_origin)
    commit_url = "https://api.github.com/repos/acme/model/commits/main"
    assert client.get(commit_url).body == b"redirected"
    assert client.get(commit_url).body == b"same-origin"
    assert opener.calls == 2


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
