from __future__ import annotations

from modelome.normalize import extract_url_mentions, extract_urls


def test_url_extractors_skip_malformed_ports_in_untrusted_text() -> None:
    text = (
        "Valid https://example.org/paper and malformed "
        "https://example.org:BEETL/not-a-port."
    )

    assert extract_urls(text) == ["https://example.org/paper"]
    assert extract_url_mentions(text) == [("https://example.org/paper", "text:6-31")]
