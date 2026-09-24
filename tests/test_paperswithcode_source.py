from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import modelome.sources.paperswithcode as pwc_source
from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.paperswithcode import (
    PapersWithCodeEvaluationMethodsSourceAdapter,
    PapersWithCodeLinksSourceAdapter,
    PapersWithCodeValidatedMethodsSourceAdapter,
)

_REVISION = "56cc5c1938678c33dedebf5f74fc4e62e2c35381"
_NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(body: bytes, url: str) -> HttpResponse:
    return HttpResponse(status=200, headers={}, body=body, url=url)


def _metadata() -> bytes:
    return (
        b'{"id":"pwc-archive/links-between-paper-and-code","sha":"'
        + _REVISION.encode()
        + b'","siblings":[{"rfilename":"data/train-00000-of-00001.parquet"}]}'
    )


def _parquet(rows: list[dict[str, Any]]) -> bytes:
    output = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), output)
    return output.getvalue().to_pybytes()


def test_official_rows_create_concrete_paper_and_repository_artifacts() -> None:
    client = QueuedClient(
        _response(_metadata(), "https://huggingface.co/api/datasets/pwc-archive/links"),
        _response(_parquet(_rows()), "https://cas-bridge.xethub.hf.co/data.parquet"),
    )
    source = PapersWithCodeLinksSourceAdapter(client=client, clock=lambda: _NOW)

    page = source.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is False
    assert page.upstream_count == 2
    assert page.next_state["snapshot_revision"] == _REVISION
    assert page.next_state["completed_snapshot_revision"] == _REVISION
    assert len(page.records) == 4
    papers = [record for record in page.records if record.kind is ArtifactKind.PAPER]
    repositories = [
        record for record in page.records if record.kind is ArtifactKind.CODE_REPOSITORY
    ]
    assert all(paper.title == "Source-backed Paper" for paper in papers)
    assert all(Identifier("arxiv", "1706.03762") in paper.identifiers for paper in papers)
    assert {
        link.url
        for paper in papers
        for link in paper.links
        if link.relation == "official_implementation"
    } == {
        "https://github.com/example/first",
        "https://github.com/example/second",
    }
    assert all(link.crawl is False for paper in papers for link in paper.links)
    assert {record.canonical_url for record in repositories} == {
        "https://github.com/example/first",
        "https://github.com/example/second",
    }
    assert all(record.raw["license"] == "CC-BY-SA-4.0" for record in page.records)


def test_unchanged_snapshot_is_a_non_authoritative_noop() -> None:
    client = QueuedClient(
        _response(_metadata(), "https://huggingface.co/api/datasets/pwc-archive/links")
    )
    source = PapersWithCodeLinksSourceAdapter(client=client, clock=lambda: _NOW)

    page = source.fetch_page({"completed_snapshot_revision": _REVISION})

    assert page.records == ()
    assert page.complete is True
    assert page.authoritative_snapshot is False
    assert len(client.calls) == 1


def test_malformed_immutable_rows_are_quarantined_without_replaying_the_batch() -> None:
    rows = _rows()
    rows[1]["paper_title"] = " "
    client = QueuedClient(
        _response(_metadata(), "https://huggingface.co/api/datasets/pwc-archive/links"),
        _response(_parquet(rows), "https://cas-bridge.xethub.hf.co/data.parquet"),
    )
    source = PapersWithCodeLinksSourceAdapter(client=client, clock=lambda: _NOW)

    page = source.fetch_page({})

    assert page.complete is True
    assert page.advance_on_source_issues is True
    assert len(page.issues) == 1
    assert len(page.records) == 2


def test_chunked_scan_advances_by_snapshot_row_without_reusing_paper_claims() -> None:
    rows = _rows()
    payload = _parquet(rows)
    client = QueuedClient(
        *(
            response
            for _ in range(3)
            for response in (
                _response(_metadata(), "https://huggingface.co/api/datasets/pwc-archive/links"),
                _response(payload, "https://cas-bridge.xethub.hf.co/data.parquet"),
            )
        )
    )
    source = PapersWithCodeLinksSourceAdapter(
        client=client,
        clock=lambda: _NOW,
        page_size=1,
    )

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)
    third = source.fetch_page(second.next_state)

    first_paper = next(record for record in first.records if record.kind is ArtifactKind.PAPER)
    second_paper = next(record for record in second.records if record.kind is ArtifactKind.PAPER)
    assert first.complete is False
    assert second.complete is False
    assert third.complete is True
    assert third.records == ()
    assert first.next_state["row_offset"] == 1
    assert second.next_state["row_offset"] == 2
    assert first_paper.source_record_id != second_paper.source_record_id


def test_links_adapter_discovers_and_scans_all_configured_file_shards() -> None:
    revision = "a" * 40
    shard_paths = [
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet",
    ]
    metadata = (
        b'{"id":"pwc-archive/links-between-paper-and-code","sha":"'
        + revision.encode()
        + b'","siblings":['
        + b",".join(
            b'{"rfilename":"' + path.encode() + b'"}' for path in shard_paths
        )
        + b"]}"
    )
    rows = _rows()[:2]
    client = QueuedClient(
        *(
            response
            for row in rows
            for response in (
                _response(metadata, "https://huggingface.co/api/datasets/pwc-archive/links"),
                _response(_parquet([row]), "https://cas-bridge.xethub.hf.co/data.parquet"),
            )
        )
    )
    source = PapersWithCodeLinksSourceAdapter(client=client, clock=lambda: _NOW)

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)

    assert len(first.records) == len(second.records) == 2
    assert first.complete is False
    assert first.next_state["data_paths"] == shard_paths
    assert first.next_state["shard_index"] == 1
    assert second.complete is True
    assert second.next_state["completed_snapshot_revision"] == revision


def test_only_arxiv_verified_nonspam_methods_become_documented_models() -> None:
    revision = "fd7c1cd6bb715116ec3c20e10651616da99ff1aa"
    metadata = (
        b'{"id":"pwc-archive/methods","sha":"'
        + revision.encode()
        + b'","siblings":[{"rfilename":"data/train-00000-of-00001.parquet"}]}'
    )
    atom = b'''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/1802.05957v2</id>
    <title>Spectral Normalization for Generative Adversarial Networks</title>
  </entry>
</feed>'''
    client = QueuedClient(
        _response(metadata, "https://huggingface.co/api/datasets/pwc-archive/methods"),
        _response(_parquet(_method_rows()), "https://cas-bridge.xethub.hf.co/methods.parquet"),
        _response(atom, "https://export.arxiv.org/api/query?id_list=1802.05957"),
    )
    source = PapersWithCodeValidatedMethodsSourceAdapter(
        client=client,
        clock=lambda: _NOW,
        pause=lambda _: None,
    )

    page = source.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert len(page.records) == 1
    record = page.records[0]
    assert record.title == "Spectral Normalization"
    assert record.models[0].status is ModelStatus.DOCUMENTED
    assert record.models[0].confidence == 0.8
    assert Identifier("arxiv", "1802.05957") in record.identifiers
    assert (
        "https://paperswithcode.com/media/methods/spectral-normalization.py",
        "code_reference",
        False,
    ) in {(link.url, link.relation, link.crawl) for link in record.links}
    assert page.next_state["validated_count"] == 1
    assert page.next_state["rejected"]["phone_payload"] == 1
    assert client.calls[2][1] == {"id_list": "1802.05957"}


def test_nonspam_arxiv_linked_methods_can_be_retained_as_candidates_without_lookup() -> None:
    revision = "fd7c1cd6bb715116ec3c20e10651616da99ff1aa"
    metadata = (
        b'{"id":"pwc-archive/methods","sha":"'
        + revision.encode()
        + b'","siblings":[{"rfilename":"data/train-00000-of-00001.parquet"}]}'
    )
    client = QueuedClient(
        _response(metadata, "https://huggingface.co/api/datasets/pwc-archive/methods"),
        _response(_parquet(_method_rows()), "https://cas-bridge.xethub.hf.co/methods.parquet"),
    )
    source = PapersWithCodeValidatedMethodsSourceAdapter(
        name="method-candidates",
        client=client,
        clock=lambda: _NOW,
        admission="candidate",
    )

    page = source.fetch_page({})

    assert len(page.records) == 1
    assert page.records[0].models[0].status is ModelStatus.CANDIDATE
    assert page.records[0].models[0].confidence == 0.4
    assert page.records[0].raw["verified_arxiv_title"] is None
    assert len(client.calls) == 2


def test_validated_methods_adapter_discovers_and_scans_all_file_shards() -> None:
    revision = "c" * 40
    shard_paths = [
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet",
    ]
    metadata = (
        b'{"id":"pwc-archive/methods","sha":"'
        + revision.encode()
        + b'","siblings":['
        + b",".join(
            b'{"rfilename":"' + path.encode() + b'"}' for path in shard_paths
        )
        + b"]}"
    )
    method_rows = [
        _method_rows()[0],
        {
            **_method_rows()[0],
            "url": "https://paperswithcode.com/method/spectral-normalization-variant",
            "name": "Spectral Normalization Variant",
            "full_name": "Spectral Normalization Variant",
        },
    ]
    client = QueuedClient(
        *(
            response
            for row in method_rows
            for response in (
                _response(metadata, "https://huggingface.co/api/datasets/pwc-archive/methods"),
                _response(_parquet([row]), "https://cas-bridge.xethub.hf.co/methods.parquet"),
            )
        )
    )
    source = PapersWithCodeValidatedMethodsSourceAdapter(
        name="method-candidates",
        client=client,
        clock=lambda: _NOW,
        admission="paper_candidate",
    )

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)

    assert len(first.records) == len(second.records) == 1
    assert first.complete is False
    assert first.next_state["data_paths"] == shard_paths
    assert first.next_state["candidate_count"] == 1
    assert second.complete is True
    assert second.next_state["completed_snapshot_revision"] == revision
    assert second.next_state["candidate_count"] == 2
    assert second.authoritative_snapshot is False


def test_paper_linked_methods_without_an_arxiv_source_remain_lower_confidence_candidates() -> None:
    revision = "fd7c1cd6bb715116ec3c20e10651616da99ff1aa"
    metadata = (
        b'{"id":"pwc-archive/methods","sha":"'
        + revision.encode()
        + b'","siblings":[{"rfilename":"data/train-00000-of-00001.parquet"}]}'
    )
    rows = _method_rows()
    rows[0] = {
        **rows[0],
        "source_url": "https://doi.org/10.1000/oceannet-source",
        "source_title": "OceanNet source paper",
    }
    client = QueuedClient(
        _response(metadata, "https://huggingface.co/api/datasets/pwc-archive/methods"),
        _response(_parquet(rows), "https://cas-bridge.xethub.hf.co/methods.parquet"),
    )
    source = PapersWithCodeValidatedMethodsSourceAdapter(
        name="paper-linked-method-candidates",
        client=client,
        clock=lambda: _NOW,
        admission="paper_linked_candidate",
    )

    page = source.fetch_page({})

    assert len(page.records) == 1
    record = page.records[0]
    assert record.models[0].status is ModelStatus.CANDIDATE
    assert record.models[0].confidence == 0.3
    assert record.raw["arxiv_id"] is None
    assert Identifier("arxiv", "1802.05957") not in record.identifiers
    assert (
        "https://doi.org/10.1000/oceannet-source",
        "source_paper",
    ) in {(link.url, link.relation) for link in record.links}
    assert len(client.calls) == 2


def test_evaluation_tables_emit_only_paper_backed_candidate_labels(
    monkeypatch: Any,
) -> None:
    revision = "7dd607a42427a2c27fcedc689a7415df58788c50"
    metadata = (
        b'{"id":"pwc-archive/evaluation-tables","sha":"'
        + revision.encode()
        + b'","siblings":['
        b'{"rfilename":"data/train-00001-of-00002.parquet"},'
        b'{"rfilename":"data/train-00000-of-00002.parquet"}]}'
    )
    rows = [
        {
            "task": "Example task",
            "subtasks": [],
            "datasets": [
                {
                    "sota": {
                        "rows": [
                            {
                                "model_name": "Example Candidate",
                                "paper_url": "https://arxiv.org/abs/1706.03762",
                                "paper_title": "Source-backed paper",
                                "code_links": [
                                    {"url": "https://github.com/example/implementation"},
                                ],
                                "model_links": [
                                    {"url": "https://example.org/models/candidate.safetensors"},
                                    {"url": "https://example.org/models/candidate.safetensors"},
                                    {"url": "javascript:alert(1)"},
                                ],
                            },
                            {
                                "model_name": "Example Candidate",
                                "paper_url": "https://arxiv.org/abs/1706.03762",
                                "paper_title": "Source-backed paper",
                                "code_links": [
                                    {"url": "https://github.com/example/second"},
                                ],
                            },
                            {
                                "model_name": "Call +1 (800) 555-0123 now",
                                "paper_url": "https://arxiv.org/abs/1706.03762",
                                "paper_title": "Source-backed paper",
                            },
                            {
                                "model_name": "No paper",
                                "paper_url": "not-a-url",
                                "paper_title": "Source-backed paper",
                            },
                        ]
                    }
                }
            ],
        }
    ]
    monkeypatch.setattr(pwc_source, "_read_evaluation_rows", lambda *_: rows)
    client = QueuedClient(
        _response(metadata, "https://huggingface.co/api/datasets/pwc-archive/evaluation-tables"),
        _response(b"first", "https://cas-bridge.xethub.hf.co/first.parquet"),
        _response(b"second", "https://cas-bridge.xethub.hf.co/second.parquet"),
    )
    source = PapersWithCodeEvaluationMethodsSourceAdapter(
        client=client,
        clock=lambda: _NOW,
    )

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)

    assert first.complete is False
    assert first.next_state["shard_index"] == 1
    assert first.next_state["candidate_count"] == 1
    assert first.next_state["rejected"] == {
        "invalid_model_name": 1,
        "missing_paper_provenance": 1,
    }
    assert len(first.records) == 1
    record = first.records[0]
    assert record.models[0].status is ModelStatus.CANDIDATE
    assert record.models[0].confidence == 0.35
    assert record.raw["tasks"] == ["Example task"]
    assert {
        link.url for link in record.links if link.relation == "reported_implementation"
    } == {
        "https://github.com/example/implementation",
        "https://github.com/example/second",
    }
    model_artifact_links = [
        link for link in record.links if link.relation == "model_artifact"
    ]
    assert [link.url for link in model_artifact_links] == [
        "https://example.org/models/candidate.safetensors"
    ]
    assert model_artifact_links[0].crawl is False
    assert record.raw["model_artifact_urls"] == [
        "https://example.org/models/candidate.safetensors"
    ]
    assert second.complete is True
    assert second.next_state["completed_snapshot_revision"] == revision
    # The second immutable shard resumes directly; no redundant metadata call.
    assert len(client.calls) == 3


def _rows() -> list[dict[str, Any]]:
    common = {
        "paper_url": "https://paperswithcode.com/paper/source-backed-paper",
        "paper_title": "Source-backed Paper",
        "paper_arxiv_id": "1706.03762v5",
        "paper_url_abs": "https://arxiv.org/abs/1706.03762v5",
        "paper_url_pdf": "https://arxiv.org/pdf/1706.03762v5.pdf",
        "mentioned_in_paper": True,
        "mentioned_in_github": False,
        "framework": "pytorch",
    }
    return [
        {**common, "repo_url": "https://github.com/example/first/tree/main", "is_official": True},
        {**common, "repo_url": "https://github.com/example/second", "is_official": True},
        {
            **common,
            "repo_url": "https://github.com/example/untrusted",
            "is_official": False,
        },
    ]


def _method_rows() -> list[dict[str, Any]]:
    title = "Spectral Normalization for Generative Adversarial Networks"
    shared = {
        "paper": {
            "title": title,
            "url": "https://paperswithcode.com/paper/spectral-normalization-for-generative-adversarial",
        },
        "source_url": "https://arxiv.org/abs/1802.05957v2",
        "source_title": title,
        "code_snippet_url": "https://paperswithcode.com/media/methods/spectral-normalization.py",
        "num_papers": 42,
        "collections": [],
    }
    return [
        {
            **shared,
            "url": "https://paperswithcode.com/method/spectral-normalization",
            "name": "Spectral Normalization",
            "full_name": "Spectral Normalization",
            "description": "A stable normalization method.",
        },
        {
            **shared,
            "url": "https://paperswithcode.com/method/fake-support",
            "name": "Call support now",
            "full_name": "Call support now",
            "description": "Call +1 (800) 555-0123 for immediate help.",
        },
    ]
