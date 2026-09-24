from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.biorxiv import BioRxivSourceAdapter
from modelome.sources.biorxiv_jats_supplementary import (
    BioRxivJatsSupplementSourceAdapter,
)

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
JATS_URL = "https://www.biorxiv.org/content/early/2024/01/03/2024.01.02.123456.source.xml"


class QueueClient:
    max_response_bytes = 32 * 1024 * 1024

    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, headers=None, **_: Any) -> HttpResponse:
        self.calls.append((url, dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(body: bytes, content_type: str = "application/json") -> HttpResponse:
    return HttpResponse(
        status=200,
        headers={"content-type": content_type},
        body=body,
        url="https://fixtures.test/response",
    )


def _details_payload(
    *, count: int = 1, total: int = 1, records: list[dict[str, Any]] | None = None
):
    records = records or [
        {
            "doi": "10.1101/2024.01.02.123456",
            "title": "A model paper",
            "version": "1",
            "date": "2024-01-03",
            "server": "bioRxiv",
            "jatsxml": JATS_URL,
        }
    ]
    return _response(
        json.dumps(
            {
                "messages": [{"status": "ok", "cursor": 0, "count": count, "total": total}],
                "collection": records,
            }
        ).encode()
    )


def _jats(
    *,
    title: str,
    href: str = "supplementary/checkpoint.pt",
    doi: str = "10.1101/2024.01.02.123456",
) -> HttpResponse:
    xml = f'''<article xmlns="http://jats.nlm.nih.gov" xmlns:xlink="http://www.w3.org/1999/xlink">
      <front><article-meta>
        <article-id pub-id-type="doi">{doi}</article-id>
      </article-meta></front>
      <body><supplementary-material xlink:href="{href}">
        <caption><title>{title}</title></caption>
      </supplementary-material></body>
    </article>'''
    return _response(xml.encode(), "application/xml")


def _adapter(client: QueueClient, **options: Any) -> BioRxivJatsSupplementSourceAdapter:
    client.max_response_bytes = options.get("max_jats_bytes", client.max_response_bytes)
    metadata = BioRxivSourceAdapter(
        name="biorxiv",
        url="https://api.biorxiv.org/details",
        server="biorxiv",
        client=client,
        clock=lambda: NOW,
    )
    return BioRxivJatsSupplementSourceAdapter(
        source=metadata,
        client=client,
        minimum_request_interval_seconds=0.34,
        **options,
    )


def test_public_jats_followup_emits_explicit_model_supplement_and_keeps_nested_checkpoint() -> None:
    client = QueueClient(_details_payload(), _jats(title="Pretrained model checkpoint"))
    source = _adapter(client)

    page = source.fetch_page({})

    assert len(page.records) == 1
    resource = page.records[0]
    assert resource.source_record_id == (
        "biorxiv:10.1101/2024.01.02.123456:v1:jats-model-resources"
    )
    link = next(link for link in resource.links if link.relation == "model_artifact")
    assert link.url == (
        "https://www.biorxiv.org/content/early/2024/01/03/supplementary/checkpoint.pt"
    )
    assert link.crawl is False
    assert resource.raw["jatsxml"] == JATS_URL
    assert page.complete is True
    assert page.next_state["metadata_state"]["watermark"] == "2026-08-31"
    assert [url for url, _ in client.calls] == [
        "https://api.biorxiv.org/details/biorxiv/2026-08-25/2026-08-31/0/json",
        JATS_URL,
    ]


def test_public_jats_data_supplement_is_not_projected_as_model() -> None:
    client = QueueClient(_details_payload(), _jats(title="Supplementary data table"))

    page = _adapter(client).fetch_page({})

    assert page.records == ()
    assert len(client.calls) == 2


def test_public_jats_preserves_explicit_external_model_resource_url() -> None:
    client = QueueClient(
        _details_payload(),
        _jats(
            title="Pretrained model weights",
            href="https://zenodo.org/records/1234567",
        ),
    )

    page = _adapter(client).fetch_page({})

    link = next(link for link in page.records[0].links if link.relation == "model_artifact")
    assert link.url == "https://zenodo.org/records/1234567"


def test_public_jats_only_follows_first_party_source_xml_urls() -> None:
    bad_record = {
        "doi": "10.1101/2024.01.02.123456",
        "title": "A model paper",
        "version": "1",
        "date": "2024-01-03",
        "server": "bioRxiv",
        "jatsxml": "https://attacker.example/paper.source.xml",
    }
    client = QueueClient(_details_payload(records=[bad_record]))

    with pytest.raises(ValueError, match="not a first-party source.xml URL"):
        _adapter(client).fetch_page({})
    assert len(client.calls) == 1


def test_public_jats_response_byte_limit_holds_metadata_page_for_retry() -> None:
    client = QueueClient(_details_payload(), _jats(title="Pretrained model checkpoint"))

    with pytest.raises(ValueError, match="JATS response exceeds"):
        _adapter(client, max_jats_bytes=50).fetch_page({})


def test_public_jats_pacing_applies_between_followup_requests() -> None:
    first = {
        "doi": "10.1101/2024.01.02.123456",
        "title": "First model paper",
        "version": "1",
        "date": "2024-01-03",
        "server": "bioRxiv",
        "jatsxml": JATS_URL,
    }
    second_doi = "10.1101/2024.01.02.654321"
    second = {
        **first,
        "doi": second_doi,
        "jatsxml": JATS_URL.replace("123456", "654321"),
    }
    payload = _details_payload(count=2, total=2, records=[first, second])
    xml = _jats(title="Pretrained model checkpoint", doi=second_doi)
    now = [0.0]
    sleeps: list[float] = []

    def sleep(duration: float) -> None:
        sleeps.append(duration)
        now[0] += duration

    client = QueueClient(payload, _jats(title="Pretrained model checkpoint"), xml)
    client.max_response_bytes = 32 * 1024 * 1024
    metadata = BioRxivSourceAdapter(
        name="biorxiv",
        url="https://api.biorxiv.org/details",
        server="biorxiv",
        client=client,
        clock=lambda: NOW,
    )
    source = BioRxivJatsSupplementSourceAdapter(
        source=metadata,
        client=client,
        minimum_request_interval_seconds=0.34,
        monotonic=lambda: now[0],
        sleep=sleep,
    )

    source.fetch_page({})

    assert sleeps == [pytest.approx(0.34), pytest.approx(0.34)]


def test_paper_without_jats_is_skipped_without_losing_association_or_checkpoint() -> None:
    no_jats = {
        "doi": "10.1101/2024.01.02.654321",
        "title": "No JATS paper",
        "version": "1",
        "date": "2024-01-03",
        "server": "bioRxiv",
        "jatsxml": "NA",
    }
    payload = _details_payload(
        count=2,
        total=2,
        records=[
            {
                "doi": "10.1101/2024.01.02.123456",
                "title": "A model paper",
                "version": "1",
                "date": "2024-01-03",
                "server": "bioRxiv",
                "jatsxml": JATS_URL,
            },
            no_jats,
        ],
    )
    client = QueueClient(payload, _jats(title="Pretrained model checkpoint"))

    page = _adapter(client).fetch_page({})

    assert len(page.records) == 1
    assert page.records[0].raw["metadata_source_record_id"] == (
        "biorxiv:10.1101/2024.01.02.123456:v1"
    )
    assert page.next_state["metadata_state"]["watermark"] == "2026-08-31"
    assert [url for url, _ in client.calls] == [
        "https://api.biorxiv.org/details/biorxiv/2026-08-25/2026-08-31/0/json",
        JATS_URL,
    ]


def test_failed_followup_replays_same_metadata_page_before_advancing_checkpoint() -> None:
    metadata_url = "https://api.biorxiv.org/details/biorxiv/2026-08-25/2026-08-31/0/json"

    class RetryOnceClient(QueueClient):
        def __init__(self) -> None:
            super().__init__(
                _details_payload(),
                _details_payload(),
                _jats(title="Pretrained model checkpoint"),
            )
            self.fail_next_jats = True

        def get(self, url: str, *, headers=None, **kwargs: Any) -> HttpResponse:
            if url == JATS_URL and self.fail_next_jats:
                self.fail_next_jats = False
                self.calls.append((url, dict(headers or {})))
                raise TimeoutError("temporary JATS timeout")
            return super().get(url, headers=headers, **kwargs)

    client = RetryOnceClient()
    source = _adapter(client)

    with pytest.raises(TimeoutError, match="temporary JATS timeout"):
        source.fetch_page({})
    retried_page = source.fetch_page({})

    metadata_calls = [url for url, _ in client.calls if url.startswith("https://api.")]
    assert metadata_calls == [metadata_url, metadata_url]
    assert len(retried_page.records) == 1
