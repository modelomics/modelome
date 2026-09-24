from __future__ import annotations

from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.nvidia_nim_lifecycle import NvidiaNimLifecycle

URL = "https://docs.nvidia.com/ai-enterprise/lifecycle/latest/eol-notices.html"


class Client:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response

    def get(
        self, url: str, *, params: Any = None, headers: dict[str, str] | None = None
    ) -> HttpResponse:
        assert url == URL
        assert params is None
        assert headers == {"Accept": "text/html"}
        return self.response


def response(body: str, status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, headers={}, body=body.encode(), url=URL)


def test_nvidia_eol_table_emits_exact_nim_component_lifecycle_rows() -> None:
    body = (
        "<table><tr><th>Component</th><th>Current State</th>"
        "<th>Action Required By</th></tr>"
        "<tr><td>NVIDIA NIM™ Retrieval QA E5 Embedding v5</td>"
        "<td>Deprecated</td><td>January 2027 (PB 6)</td></tr>"
        "<tr><td>NVIDIA NIM Llama-3.1-8B-Instruct</td>"
        "<td>Deprecated</td><td>January 2027 (PB6)</td></tr></table>"
        "<table><tr><th>Component</th><th>Current State</th>"
        "<th>Action Required By</th></tr>"
        "<tr><td>NVIDIA NIM Llama-3.1-70b-instruct</td>"
        "<td>End of Support (Model Replaced)</td><td>Immediate</td></tr></table>"
        "<table><tr><th>Component</th><th>Current State</th>"
        "<th>Action Required By</th></tr>"
        "<tr><td>NVIDIA Morpheus</td><td>End of Support</td><td>Immediate</td></tr></table>"
    )
    page = NvidiaNimLifecycle(client=Client(response(body))).fetch_page({})

    assert page.complete and not page.authoritative_snapshot
    assert page.upstream_count == 3
    first = page.records[0]
    assert first.kind is ArtifactKind.CATALOG_RECORD
    assert first.source_record_id == "lifecycle:NVIDIA NIM™ Retrieval QA E5 Embedding v5"
    assert first.identifiers == (
        Identifier(
            "nvidia:ai-enterprise-component",
            "NVIDIA NIM™ Retrieval QA E5 Embedding v5",
        ),
    )
    assert first.models[0].status is ModelStatus.DOCUMENTED
    assert first.raw["provider_lifecycle_status"] == "Deprecated"
    assert first.raw["action_required_by"] == "January 2027 (PB 6)"
    assert page.records[2].raw["provider_lifecycle_status"] == "End of Support (Model Replaced)"


@pytest.mark.parametrize(
    ("body", "status", "error"),
    [
        ("", 503, "HTTP 503"),
        ("<p>No notice table</p>", 200, "lifecycle tables were not found"),
        (
            "<table><tr><th>Component</th><th>Current State</th>"
            "<th>Action Required By</th></tr></table>",
            200,
            "contain no NIM notices",
        ),
    ],
)
def test_nvidia_eol_parser_rejects_unexpected_responses(
    body: str, status: int, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        NvidiaNimLifecycle(client=Client(response(body, status))).fetch_page({})
