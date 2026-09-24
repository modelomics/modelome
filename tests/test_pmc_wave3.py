from __future__ import annotations

from datetime import UTC, datetime

from test_pmc_source import QueueClient, list_records, record, response

from modelome.sources.pmc import PmcSourceAdapter

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
PMC_FTP_CHECKPOINT = (
    "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/articles/PMC1234567/bin/checkpoint.pt"
)


def _run_supplementary(title: str):
    supplementary = (
        '<supplementary-material id="sm1" mimetype="application" '
        'mime-subtype="octet-stream" '
        f'xlink:href="{PMC_FTP_CHECKPOINT}">'
        f"<caption><title>{title}</title></caption>"
        "</supplementary-material>"
    )
    xml = record().replace(
        "<p>The system learns directly from histology images.</p>",
        "<p>The system learns directly from histology images.</p>" + supplementary,
    )
    client = QueueClient(response(list_records(xml)))
    source = PmcSourceAdapter(client=client, clock=lambda: NOW, initial_lookback_days=2)
    return source.fetch_page({}).records[0]


def test_pmc_ftp_supplement_declared_as_model_checkpoint_is_model_artifact() -> None:
    item = _run_supplementary("Pretrained model checkpoint")

    assert any(
        link.url == PMC_FTP_CHECKPOINT
        and link.relation == "model_artifact"
        and not link.crawl
        for link in item.links
    )
    supplement = next(
        url for url in item.raw["jats"]["external_urls"]
        if url["url"] == PMC_FTP_CHECKPOINT
    )
    assert supplement["resource_type"] == "model_artifact"


def test_pmc_ftp_supplement_declared_as_dataset_remains_reference() -> None:
    item = _run_supplementary("Supplementary data table")

    assert any(
        link.url == PMC_FTP_CHECKPOINT and link.relation == "references"
        for link in item.links
    )
    supplement = next(
        url for url in item.raw["jats"]["external_urls"]
        if url["url"] == PMC_FTP_CHECKPOINT
    )
    assert "resource_type" not in supplement
