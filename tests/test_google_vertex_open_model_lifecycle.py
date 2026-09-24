from __future__ import annotations

from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.google_vertex_open_model_lifecycle import (
    GoogleVertexOpenModelLifecycle,
)

URL = "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/deprecations/open-models"


class Client:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response

    def get(
        self, url: str, *, params: Any = None, headers: dict[str, str] | None = None
    ) -> HttpResponse:
        assert url == URL
        assert params is None
        assert headers == {"Accept": "text/markdown"}
        return self.response


def response(body: str, status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, headers={}, body=body.encode(), url=URL)


def test_vertex_managed_open_model_lifecycle_preserves_exact_ids_and_dates() -> None:
    page = GoogleVertexOpenModelLifecycle(client=Client(response(
        "<p>The following table lists lifecycle schedules for managed open models.</p>"
        "<table><tr><th>Model ID</th><th>Deprecation date</th>"
        "<th>Retirement date</th><th>Self-deploy alternative</th></tr>"
        "<tr><td><a name='deepseek-ocr-maas'></a><code>deepseek-ocr-maas</code></td>"
        "<td>July 21, 2026</td><td>October 21, 2026</td>"
        "<td><a href='https://console.cloud.google.com/x'>Self-deploy DeepSeek-OCR</a></td></tr>"
        "<tr><td><code>llama-3.3-70b-instruct-maas</code></td>"
        "<td>July 21, 2026</td><td>October 21, 2026</td>"
        "<td>Self-deploy Llama 3.3</td></tr></table>"
        "<table><tr><td>not-a-managed-model</td></tr></table>"
    ))).fetch_page({})

    assert page.complete and not page.authoritative_snapshot
    assert page.upstream_count == 2
    first = page.records[0]
    assert first.kind is ArtifactKind.CATALOG_RECORD
    assert first.source_record_id == "lifecycle:deepseek-ocr-maas"
    assert first.identifiers == (
        Identifier("google:vertex-managed-open-model", "deepseek-ocr-maas"),
    )
    assert first.models[0].status is ModelStatus.DOCUMENTED
    assert first.raw["provider_lifecycle_status"] == "deprecated"
    assert first.raw["deprecation_date"] == "July 21, 2026"
    assert first.raw["retirement_date"] == "October 21, 2026"
    assert first.raw["self_deploy_alternative"] == "Self-deploy DeepSeek-OCR"


@pytest.mark.parametrize(
    ("body", "status", "error"),
    [
        ("", 503, "HTTP 503"),
        ("Not the expected table", 200, "header was not found"),
        (
            "<table><tr><th>Model ID</th><th>Deprecation date</th>"
            "<th>Retirement date</th><th>Self-deploy alternative</th></tr>"
            "<tr><td>model-x</td><td>Soon</td><td>October 21, 2026</td>"
            "<td>Alternative</td></tr></table>",
            200,
            "invalid lifecycle date",
        ),
        (
            "<table><tr><th>Model ID</th><th>Deprecation date</th>"
            "<th>Retirement date</th><th>Self-deploy alternative</th></tr></table>",
            200,
            "contains no model rows",
        ),
    ],
)
def test_vertex_lifecycle_rejects_unexpected_or_incomplete_responses(
    body: str, status: int, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        GoogleVertexOpenModelLifecycle(
            client=Client(response(body, status))
        ).fetch_page({})
