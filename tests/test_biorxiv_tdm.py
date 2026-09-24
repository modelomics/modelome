from __future__ import annotations

import io
import zipfile
from typing import Any

import pytest

from modelome.sources.biorxiv_tdm import BioRxivTdmSourceAdapter


def _meca(*, title: str, href: str = "checkpoint.pt") -> bytes:
    article = f'''<article xmlns="http://jats.nlm.nih.gov" xmlns:xlink="http://www.w3.org/1999/xlink">
      <front><article-meta>
        <article-id pub-id-type="doi">10.1101/2024.01.02.123456</article-id>
        <title-group><article-title>Neural biology model</article-title></title-group>
      </article-meta></front>
      <body><supplementary-material xlink:href="{href}">
        <caption><title>{title}</title></caption>
      </supplementary-material></body>
    </article>'''
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.xml", "<manifest><item file='content/article.xml'/></manifest>")
        package.writestr("content/article.xml", article)
        package.writestr("content/checkpoint.pt", b"checkpoint bytes")
    return output.getvalue()


class FakeBody(io.BytesIO):
    pass


class FakeS3:
    def __init__(self, pages: list[dict[str, Any]], objects: dict[str, bytes]) -> None:
        self.pages = pages
        self.objects = objects
        self.list_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.list_calls.append(kwargs)
        return self.pages.pop(0)

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        data = self.objects[kwargs["Key"]]
        return {"Body": FakeBody(data), "ContentLength": len(data)}


def _page(*keys: str, truncated: bool, token: str | None = None):
    result: dict[str, Any] = {
        "Contents": [{"Key": key} for key in keys],
        "IsTruncated": truncated,
    }
    if token:
        result["NextContinuationToken"] = token
    return result


def _source(client: FakeS3, **limits: int) -> BioRxivTdmSourceAdapter:
    return BioRxivTdmSourceAdapter(
        server="biorxiv",
        prefix="Current Content/January_2024/",
        client=client,
        **limits,
    )


def test_tdm_page_extracts_model_member_and_checkpoints_listing() -> None:
    key1 = "Current Content/January_2024/paper.meca"
    key2 = "Current Content/January_2024/another.meca"
    client = FakeS3(
        [
            _page(key1, truncated=True, token="next-page"),
            _page(key2, truncated=False),
        ],
        {key1: _meca(title="Pretrained model checkpoint"), key2: _meca(title="Data table")},
    )
    source = _source(client, max_objects_per_page=1)

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)

    assert first.complete is False
    assert first.next_state["continuation_token"] == "next-page"
    assert first.records[0].source_record_id.startswith("biorxiv:tdm:10.1101/2024.01.02.123456:")
    model_link = next(link for link in first.records[0].links if link.relation == "model_artifact")
    assert model_link.url == "s3://biorxiv-src-monthly/Current%20Content/January_2024/paper.meca"
    assert model_link.locator == "content/article.xml:content/checkpoint.pt"
    assert first.records[0].canonical_url == "https://www.biorxiv.org/content/10.1101/2024.01.02.123456"
    assert second.records == ()
    assert second.complete is True
    assert client.list_calls[0]["RequestPayer"] == "requester"
    assert client.list_calls[1]["ContinuationToken"] == "next-page"
    assert all(call["RequestPayer"] == "requester" for call in client.get_calls)


def test_tdm_extracts_explicit_external_supplementary_model_url() -> None:
    key = "Current Content/January_2024/paper.meca"
    client = FakeS3(
        [_page(key, truncated=False)],
        {key: _meca(
            title="Pretrained model weights",
            href="https://models.example.org/releases/checkpoint.pt",
        )},
    )

    page = _source(client).fetch_page({})

    assert page.records[0].links[-1].url == "https://models.example.org/releases/checkpoint.pt"
    assert page.records[0].links[-1].relation == "model_artifact"


def test_tdm_rejects_archives_over_configured_download_limit() -> None:
    key = "Current Content/January_2024/paper.meca"
    package = _meca(title="Pretrained model checkpoint")
    client = FakeS3([_page(key, truncated=False)], {key: package})

    with pytest.raises(ValueError, match="archive exceeds configured byte limit"):
        _source(client, max_archive_bytes=len(package) - 1).fetch_page({})


def test_tdm_complete_checkpoint_is_a_noop() -> None:
    client = FakeS3([], {})

    page = _source(client).fetch_page(
        {
            "complete": True,
            "bucket": "biorxiv-src-monthly",
            "prefix": "Current Content/January_2024/",
        }
    )

    assert page.complete is True
    assert page.records == ()
    assert client.list_calls == []
