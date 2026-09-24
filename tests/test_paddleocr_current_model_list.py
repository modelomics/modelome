from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.paddleocr_current_model_list import (
    PaddleOcrCurrentModelListSourceAdapter,
)

_REVISION = "4" * 40
_DOCUMENT = b"""
# PaddleOCR current model list

<table>
  <thead>
    <tr><th>\xe6\xa8\xa1\xe5\x9e\x8b</th><th>yaml \xe6\x96\x87\xe4\xbb\xb6</th>
    <th>\xe6\xa8\xa1\xe5\x9e\x8b\xe4\xb8\x8b\xe8\xbd\xbd\xe9\x93\xbe\xe6\x8e\xa5</th><th>\xe8\xb5\x84\xe6\xba\x90</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>DocFoo</td>
      <td><a href="./configs/docfoo.yaml">config</a></td>
      <td>
        <a href="https://weights.example.test/docfoo/train.tar">trained</a>
        <a href="https://weights.example.test/docfoo/inference.tar">inference</a>
      </td>
      <td><a href="https://docs.example.test/docfoo">docs</a></td>
    </tr>
    <tr>
      <td>DocFoo</td>
      <td><a href="https://github.example.test/docfoo_v2.yml">config</a></td>
      <td><a href="https://weights.example.test/docfoo/v2.pdparams">trained</a></td>
      <td></td>
    </tr>
    <tr>
      <td>DocBar</td>
      <td><a href="https://github.example.test/docbar.yaml">config</a></td>
      <td><a href="https://weights.example.test/docbar/inference.tar">inference</a></td>
      <td><a href="">empty</a></td>
    </tr>
  </tbody>
</table>

<table>
  <tr><th>\xe6\xa8\xa1\xe5\x9e\x8b</th><th>\xe6\x96\x87\xe6\xa1\xa3</th></tr>
  <tr><td>Unrelated</td><td><a href="https://weights.example.test/unrelated.tar">weight</a></td></tr>
</table>
"""


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append((url, headers))
        return self.responses.pop(0)


def _response(
    body: bytes, *, url: str = "https://fixtures.test/paddleocr"
) -> HttpResponse:
    return HttpResponse(200, {"etag": '"fixture"'}, body, url)


def _commit() -> HttpResponse:
    return _response(("{\"sha\": \"" + _REVISION + "\"}").encode())


def test_current_paddleocr_html_tables_keep_artifacts_and_configs_row_scoped() -> None:
    client = _QueuedClient(_commit(), _response(_DOCUMENT))
    adapter = PaddleOcrCurrentModelListSourceAdapter(
        repository="example/PaddleOCR",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot
    assert page.upstream_count == 2
    assert page.next_state["artifact_row_count"] == 3
    assert client.calls[1][0].endswith(f"/{_REVISION}/docs/version3.x/model_list.md")
    docfoo = next(record for record in page.records if record.title == "DocFoo")
    assert docfoo.identifiers == (Identifier("paddleocr:current-model", "DocFoo"),)
    assert len(docfoo.releases) == 3
    assert docfoo.models[0].status.value == "released"
    assert len(docfoo.raw["source_rows"]) == 2
    links = {(link.url, link.relation, link.crawl) for link in docfoo.links}
    assert (
        "https://weights.example.test/docfoo/inference.tar",
        "weights",
        False,
    ) in links
    assert (
        "https://raw.githubusercontent.com/example/PaddleOCR/"
        f"{_REVISION}/docs/version3.x/configs/docfoo.yaml",
        "model_config",
        False,
    ) in links
    assert ("https://docs.example.test/docfoo", "related_resource", False) in links
    assert all(link.model_local_ids == (docfoo.models[0].local_id,) for link in docfoo.links)
    docbar = next(record for record in page.records if record.title == "DocBar")
    assert len(docbar.releases) == 1
    assert all("docfoo" not in link.url for link in docbar.links)


def test_current_paddleocr_list_skips_source_when_commit_is_unchanged() -> None:
    client = _QueuedClient(_commit())
    adapter = PaddleOcrCurrentModelListSourceAdapter(client=client)

    page = adapter.fetch_page({"completed_revision": _REVISION, "model_count": 41})

    assert page.records == ()
    assert page.complete
    assert page.upstream_count == 41
    assert len(client.calls) == 1


def test_current_paddleocr_list_rejects_rows_without_direct_artifacts() -> None:
    document = b"""
<table>
  <tr><th>\xe6\xa8\xa1\xe5\x9e\x8b\xe5\x90\x8d\xe7\xa7\xb0</th><th>\xe6\xa8\xa1\xe5\x9e\x8b\xe4\xb8\x8b\xe8\xbd\xbd\xe9\x93\xbe\xe6\x8e\xa5</th></tr>
  <tr><td>Missing</td><td><a href="">download</a></td></tr>
</table>
"""
    client = _QueuedClient(_commit(), _response(document))

    with pytest.raises(ValueError, match="no direct model-artifact rows"):
        PaddleOcrCurrentModelListSourceAdapter(client=client).fetch_page({})
