from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.alphaxiv import (
    ALPHAXIV_MCP_ENDPOINT,
    ALPHAXIV_MCP_TOOL,
    MCP_PROTOCOL_VERSION,
    AlphaXivExactIdEnricher,
    AlphaXivMcpClient,
    AlphaXivResponseError,
    AlphaXivSafetyError,
    normalize_discovered_arxiv_id,
    plan_alphaxiv_enrichment,
)
from modelome.fetchers import PublicUrlPolicy
from modelome.http import HttpFailure, HttpResponse
from modelome.models import ArtifactKind, Identifier


class FakeMcp:
    def __init__(self, result: Mapping[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append((name, arguments))
        return self.result


class FakePostHttp:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        body: bytes,
        headers: Mapping[str, str],
        redirect_validator: Any,
    ) -> HttpResponse:
        self.calls.append(
            {
                "url": url,
                "body": body,
                "headers": dict(headers),
                "redirect_validator": redirect_validator,
            }
        )
        return self.response


def public_policy() -> PublicUrlPolicy:
    return PublicUrlPolicy(resolver=lambda _host: ("93.184.216.34",))


@pytest.mark.parametrize("value", ["1706.03762", "2608.12345", "math.GT/0309136"])
def test_only_normalized_versionless_arxiv_ids_are_accepted(value: str) -> None:
    assert normalize_discovered_arxiv_id(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        " 1706.03762",
        "arXiv:1706.03762",
        "1706.03762v7",
        "https://arxiv.org/abs/1706.03762",
        "../1706.03762",
        "1706%2e03762",
        "not-a-paper",
    ],
)
def test_non_normalized_or_non_identifier_inputs_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_discovered_arxiv_id(value)


def test_plan_is_an_exact_id_content_read_and_has_no_discovery_arguments() -> None:
    plan = plan_alphaxiv_enrichment("1706.03762")

    assert plan.arxiv_id == "1706.03762"
    assert plan.arxiv_url == "https://arxiv.org/abs/1706.03762"
    assert plan.alphaxiv_url == "https://www.alphaxiv.org/abs/1706.03762"
    assert plan.endpoint == ALPHAXIV_MCP_ENDPOINT
    assert plan.tool == "get_paper_content"
    assert plan.arguments == {
        "url": "https://arxiv.org/abs/1706.03762",
        "fullText": True,
    }
    assert not {"keywords", "question", "query", "recommendations"}.intersection(plan.arguments)


def test_legacy_identifier_path_is_preserved_without_search_resolution() -> None:
    plan = plan_alphaxiv_enrichment("math.GT/0309136")

    assert plan.arguments["url"] == "https://arxiv.org/abs/math.GT/0309136"
    assert plan.alphaxiv_url == "https://www.alphaxiv.org/abs/math.GT/0309136"


def test_enricher_retains_arxiv_identity_and_preserves_alphaxiv_evidence() -> None:
    result = {
        "content": [
            {
                "type": "text",
                "text": (
                    "Paper text. Official implementation: "
                    "https://github.com/tensorflow/tensor2tensor\n"
                    "Unsafe reference http://127.0.0.1/secrets"
                ),
            },
            {"type": "text", "text": "Second extracted page."},
        ],
        "structuredContent": {
            "title": "Attention Is All You Need",
            "metadata": {"project": "https://example.org/project"},
        },
    }
    client = FakeMcp(result)

    record = AlphaXivExactIdEnricher(
        client,
        url_policy=public_policy(),
    ).enrich("1706.03762")

    assert client.calls == [
        (
            ALPHAXIV_MCP_TOOL,
            {"url": "https://arxiv.org/abs/1706.03762", "fullText": True},
        )
    ]
    assert record.source_record_id == "1706.03762"
    assert record.kind is ArtifactKind.PAPER
    assert record.canonical_url == "https://arxiv.org/abs/1706.03762"
    assert record.title == "Attention Is All You Need"
    assert record.identifiers == (Identifier("arxiv", "1706.03762"),)
    assert record.text.endswith("Second extracted page.")
    assert record.raw["alphaxiv_url"] == "https://www.alphaxiv.org/abs/1706.03762"
    assert record.raw["mcp_result"] == result

    links = {link.url: link for link in record.links}
    alpha_link = links["https://www.alphaxiv.org/abs/1706.03762"]
    assert alpha_link.relation == "enriched_by"
    assert alpha_link.crawl is False
    implementation = links["https://github.com/tensorflow/tensor2tensor"]
    assert implementation.relation == "official_implementation"
    assert implementation.crawl is True
    assert "https://example.org/project" in links
    assert "http://127.0.0.1/secrets" not in links

    repeated = AlphaXivExactIdEnricher(
        FakeMcp(result),
        url_policy=public_policy(),
    ).enrich("1706.03762")
    assert repeated == record


def test_enricher_rejects_tool_errors_empty_text_and_oversized_results() -> None:
    with pytest.raises(AlphaXivResponseError, match="not found"):
        AlphaXivExactIdEnricher(
            FakeMcp({"content": [{"type": "text", "text": "not found"}], "isError": True})
        ).enrich("1706.03762")

    with pytest.raises(AlphaXivResponseError, match="no text"):
        AlphaXivExactIdEnricher(FakeMcp({"content": []})).enrich("1706.03762")

    with pytest.raises(AlphaXivResponseError, match="byte limit"):
        AlphaXivExactIdEnricher(
            FakeMcp({"content": [{"type": "text", "text": "x" * 100}]}),
            max_response_bytes=40,
        ).enrich("1706.03762")


def test_mcp_client_sends_only_the_current_exact_id_tool_call() -> None:
    envelope = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": "paper text"}]},
    }
    http = FakePostHttp(
        HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(envelope).encode(),
            url=ALPHAXIV_MCP_ENDPOINT,
        )
    )
    client = AlphaXivMcpClient("axv1_test", http=http)

    result = client.call_tool(
        ALPHAXIV_MCP_TOOL,
        {"url": "https://arxiv.org/abs/1706.03762", "fullText": True},
    )

    assert result == envelope["result"]
    call = http.calls[0]
    assert call["url"] == ALPHAXIV_MCP_ENDPOINT
    assert call["headers"] == {
        "Accept": "application/json, text/event-stream",
        "Authorization": "Bearer axv1_test",
        "Content-Type": "application/json",
        "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        "Mcp-Method": "tools/call",
        "Mcp-Name": ALPHAXIV_MCP_TOOL,
    }
    payload = json.loads(call["body"])
    assert payload["method"] == "tools/call"
    assert payload["params"]["name"] == "get_paper_content"
    assert payload["params"]["arguments"] == {
        "url": "https://arxiv.org/abs/1706.03762",
        "fullText": True,
    }
    assert "discover_papers" not in call["body"].decode()


def test_mcp_client_rejects_search_tools_argument_drift_and_credential_exfiltration() -> None:
    response = HttpResponse(200, {}, b"{}", ALPHAXIV_MCP_ENDPOINT)
    client = AlphaXivMcpClient("axv1_test", http=FakePostHttp(response))

    with pytest.raises(AlphaXivSafetyError, match="only"):
        client.call_tool("discover_papers", {"keywords": ["transformer"]})
    with pytest.raises(AlphaXivSafetyError, match="fullText=true"):
        client.call_tool(
            ALPHAXIV_MCP_TOOL,
            {"url": "https://arxiv.org/abs/1706.03762", "fullText": False},
        )
    with pytest.raises(AlphaXivSafetyError, match="only be sent"):
        AlphaXivMcpClient(
            "axv1_test",
            endpoint="https://attacker.example/mcp/v1",
            http=FakePostHttp(response),
        )
    with pytest.raises(AlphaXivSafetyError, match="invalid alphaXiv API key"):
        AlphaXivMcpClient("axv1_test\r\nX-Evil: yes", http=FakePostHttp(response))


def test_mcp_client_parses_a_bounded_event_stream_response() -> None:
    body = (
        b'event: message\n'
        b'data: {"jsonrpc":"2.0","id":1,"result":{"content":'
        b'[{"type":"text","text":"paper text"}]}}\n\n'
    )
    client = AlphaXivMcpClient(
        "axv1_test",
        http=FakePostHttp(
            HttpResponse(
                status=200,
                headers={"content-type": "text/event-stream; charset=utf-8"},
                body=body,
                url=ALPHAXIV_MCP_ENDPOINT,
            )
        ),
    )

    assert client.call_tool(
        ALPHAXIV_MCP_TOOL,
        {"url": "https://arxiv.org/abs/1706.03762", "fullText": True},
    ) == {"content": [{"type": "text", "text": "paper text"}]}


def test_transport_and_protocol_failures_are_not_silently_materialized() -> None:
    bad_json = FakePostHttp(HttpResponse(200, {}, b"not-json", ALPHAXIV_MCP_ENDPOINT))
    client = AlphaXivMcpClient("axv1_test", http=bad_json)
    with pytest.raises(AlphaXivResponseError, match="invalid JSON"):
        client.call_tool(
            ALPHAXIV_MCP_TOOL,
            {"url": "https://arxiv.org/abs/1706.03762", "fullText": True},
        )

    class FailedHttp:
        def post(self, *_args: Any, **_kwargs: Any) -> HttpResponse:
            raise HttpFailure("network failed")

    with pytest.raises(HttpFailure, match="network failed"):
        AlphaXivMcpClient("axv1_test", http=FailedHttp()).call_tool(
            ALPHAXIV_MCP_TOOL,
            {"url": "https://arxiv.org/abs/1706.03762", "fullText": True},
        )
