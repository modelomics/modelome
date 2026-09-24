from __future__ import annotations

from datetime import UTC, datetime

from test_pmc_source import QueueClient, list_records, record, response

from modelome.sources.pmc import PmcSourceAdapter

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def _run(paragraph: str):
    xml = record().replace(
        "<p>The system learns directly from histology images.</p>", paragraph
    )
    client = QueueClient(response(list_records(xml)))
    source = PmcSourceAdapter(client=client, clock=lambda: NOW, initial_lookback_days=2)
    return source.fetch_page({}).records[0]


def test_zenodo_model_weights_are_retained_as_model_artifact() -> None:
    item = _run(
        '<p>Pretrained model weights are available at '
        '<ext-link ext-link-type="uri" '
        'xlink:href="https://zenodo.org/records/13323293">Zenodo</ext-link>.</p>'
    )

    assert any(
        link.url == "https://zenodo.org/records/13323293"
        and link.relation == "model_artifact"
        and not link.crawl
        for link in item.links
    )
    zenodo = next(
        url for url in item.raw["jats"]["external_urls"]
        if url["url"] == "https://zenodo.org/records/13323293"
    )
    assert zenodo["resource_type"] == "model_artifact"


def test_zenodo_dataset_link_without_model_resource_context_stays_reference() -> None:
    item = _run(
        '<p>Data used in this study are available at '
        '<ext-link ext-link-type="uri" '
        'xlink:href="https://zenodo.org/records/13323293">Zenodo</ext-link>.</p>'
    )

    assert any(
        link.url == "https://zenodo.org/records/13323293"
        and link.relation == "references"
        for link in item.links
    )
    zenodo = next(
        url for url in item.raw["jats"]["external_urls"]
        if url["url"] == "https://zenodo.org/records/13323293"
    )
    assert "resource_type" not in zenodo
