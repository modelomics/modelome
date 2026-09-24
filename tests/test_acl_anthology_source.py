from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.acl_anthology import AclAnthologySourceAdapter

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)


class QueueClient:
    def __init__(self, *bodies: bytes) -> None:
        self.bodies = list(bodies)
        self.calls: list[str] = []

    def get(self, url: str, *, headers: Any = None, params: Any = None) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(200, {"content-type": "application/xml"}, self.bodies.pop(0), url)


XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<collection id="2025.acl">
  <volume id="2025.acl-long" type="proceedings">
    <meta><booktitle>Proceedings of ACL 2025</booktitle><year>2025</year></meta>
    <paper id="1">
      <title>Learning with <fixed-case>LLM</fixed-case> Models</title>
      <author><first>Ada</first><last>Researcher</last></author>
      <author><first>Lin</first><last>Scientist</last></author>
      <abstract>We present a neural model with a <i>robust</i> training method.</abstract>
      <doi>10.18653/v1/2025.acl-long.1</doi>
      <url>2025.acl-long.1</url>
    </paper>
  </volume>
</collection>"""


def test_reads_official_collection_xml_as_paper_records_and_skips_unchanged_snapshot() -> None:
    client = QueueClient(XML, XML)
    source = AclAnthologySourceAdapter(
        url="https://aclanthology.org/test.xml", client=client, clock=lambda: NOW
    )

    first = source.fetch_page({})
    same = source.fetch_page(first.next_state)

    assert first.complete is True
    assert first.upstream_count == 1
    assert len(first.records) == 1
    record = first.records[0]
    assert record.kind is ArtifactKind.PAPER
    assert record.source_record_id == "2025.acl-long.1"
    assert record.canonical_url == "https://aclanthology.org/2025.acl-long.1"
    assert record.title == "Learning with LLM Models"
    assert "robust training method" in record.text
    assert record.published_at == "2025-01-01T00:00:00Z"
    assert Identifier("acl-anthology", "2025.acl-long.1") in record.identifiers
    assert Identifier("doi", "10.18653/v1/2025.acl-long.1") in record.identifiers
    assert record.raw["authors"] == ["Ada Researcher", "Lin Scientist"]
    assert record.raw["volume_id"] == "2025.acl-long"
    assert same.complete is True
    assert same.records == ()


def test_recovers_historical_paper_id_from_adjacent_official_xml_comment() -> None:
    # The official 1952.earlymt.xml historical collection includes a paper
    # whose canonical ACL URL is recorded in a comment, without a <url> node.
    xml = b"""<collection id="1952.earlymt">
      <volume id="1">
        <!-- https://aclanthology.org/1952.earlymt-1.6/ -->
        <paper id="6"><title>Historical machine translation paper</title></paper>
      </volume>
    </collection>"""
    page = AclAnthologySourceAdapter(
        url="https://aclanthology.org/1952.earlymt.xml",
        client=QueueClient(xml),
        clock=lambda: NOW,
    ).fetch_page({})

    assert page.upstream_count == 1
    assert len(page.records) == 1
    record = page.records[0]
    assert record.source_record_id == "1952.earlymt-1.6"
    assert record.canonical_url == "https://aclanthology.org/1952.earlymt-1.6"
    assert not page.issues


def test_recovers_current_paper_id_from_embedded_resolved_url_comment() -> None:
    # ACL Anthology's current XML writer retains fully resolved link URLs in
    # comments inside the paper element for human-readable source data.
    xml = b"""<collection id="2026.acl">
      <volume id="2026.acl-long">
        <meta><booktitle>Proceedings of ACL 2026</booktitle><year>2026</year></meta>
        <paper id="1"><title>Paper with resolved URL comment</title>
          <!-- https://aclanthology.org/2026.acl-long.1/ -->
        </paper>
      </volume>
    </collection>"""
    page = AclAnthologySourceAdapter(
        url="https://aclanthology.org/2026.acl.xml",
        client=QueueClient(xml),
        clock=lambda: NOW,
    ).fetch_page({})

    assert [record.source_record_id for record in page.records] == ["2026.acl-long.1"]
    assert not page.issues


def test_rejects_unbounded_or_invalid_collection_payloads() -> None:
    oversized = AclAnthologySourceAdapter(
        url="https://aclanthology.org/test.xml",
        client=QueueClient(XML),
        max_response_bytes=32,
        clock=lambda: NOW,
    )
    with pytest.raises(ValueError, match="exceeds 32 bytes"):
        oversized.fetch_page({})

    invalid = AclAnthologySourceAdapter(
        url="https://aclanthology.org/test.xml",
        client=QueueClient(b"<!DOCTYPE collection [<!ENTITY x 'oops'>]><collection id='x'/>"),
        clock=lambda: NOW,
    )
    with pytest.raises(ValueError, match="must not declare a DTD or entities"):
        invalid.fetch_page({})


def test_enforces_configured_paper_count_cap() -> None:
    duplicate = b"<paper><title>Second paper</title><url>2025.acl-long.2</url></paper>"
    two_papers = XML.replace(b"</volume>", duplicate + b"</volume>")
    source = AclAnthologySourceAdapter(
        url="https://aclanthology.org/test.xml",
        client=QueueClient(two_papers),
        max_papers=1,
        clock=lambda: NOW,
    )
    with pytest.raises(ValueError, match="above max_papers 1"):
        source.fetch_page({})


class ManifestQueueClient:
    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, headers: Any = None, params: Any = None) -> HttpResponse:
        self.calls.append(url)
        item = self.responses.pop(0)
        body = item if isinstance(item, bytes) else json.dumps(item).encode()
        return HttpResponse(200, {"content-type": "application/json"}, body, url)


def _commit(tree_sha: str, commit_sha: str) -> dict[str, Any]:
    return {"sha": commit_sha, "commit": {"tree": {"sha": tree_sha}}}


def _manifest(sha: str, *paths: str) -> dict[str, Any]:
    return {
        "sha": sha,
        "truncated": False,
        "tree": [{"path": path, "type": "blob"} for path in paths]
        + [{"path": "data/xml/schema.rnc", "type": "blob"}],
    }


def test_walks_frozen_manifest_one_collection_at_a_time_and_skips_unchanged_files() -> None:
    xml_2025 = XML
    xml_2024 = (
        XML.replace(b"2025.acl-long.1", b"2024.acl-long.1")
        .replace(b"2025.acl", b"2024.acl")
        .replace(b"2025</year>", b"2024</year>")
    )
    xml_2026 = (
        XML.replace(b"2025.acl-long.1", b"2026.acl-long.1")
        .replace(b"2025.acl", b"2026.acl")
        .replace(b"2025</year>", b"2026</year>")
    )
    paths = ("data/xml/2024.acl.xml", "data/xml/2025.acl.xml")
    expanded_paths = (*paths, "data/xml/2026.acl.xml")
    client = ManifestQueueClient(
        _commit("a" * 40, "d" * 40),
        _manifest("a" * 40, *paths),
        xml_2024,
        xml_2025,
        _commit("a" * 40, "d" * 40),
        _commit("b" * 40, "e" * 40),
        _manifest("b" * 40, *expanded_paths),
        xml_2024,
        xml_2025,
        xml_2026,
    )
    source = AclAnthologySourceAdapter(client=client, clock=lambda: NOW)

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)
    unchanged = source.fetch_page(second.next_state)
    newer_first = source.fetch_page(unchanged.next_state)
    newer_second = source.fetch_page(newer_first.next_state)
    newer_last = source.fetch_page(newer_second.next_state)

    assert first.complete is False
    assert first.records[0].source_record_id == "2024.acl-long.1"
    assert second.complete is True
    assert second.records[0].source_record_id == "2025.acl-long.1"
    assert unchanged.complete is True
    assert unchanged.records == ()
    assert newer_first.complete is False and newer_first.records == ()
    assert newer_second.complete is False and newer_second.records == ()
    assert newer_last.complete is True
    assert [record.source_record_id for record in newer_last.records] == ["2026.acl-long.1"]
    assert newer_last.upstream_count == 3
    assert client.calls == [
        "https://api.github.com/repos/acl-org/acl-anthology/commits/master",
        "https://api.github.com/repos/acl-org/acl-anthology/git/trees/" + "a" * 40 + "?recursive=1",
        "https://raw.githubusercontent.com/acl-org/acl-anthology/"
        + "d" * 40
        + "/data/xml/2024.acl.xml",
        "https://raw.githubusercontent.com/acl-org/acl-anthology/"
        + "d" * 40
        + "/data/xml/2025.acl.xml",
        "https://api.github.com/repos/acl-org/acl-anthology/commits/master",
        "https://api.github.com/repos/acl-org/acl-anthology/commits/master",
        "https://api.github.com/repos/acl-org/acl-anthology/git/trees/" + "b" * 40 + "?recursive=1",
        "https://raw.githubusercontent.com/acl-org/acl-anthology/"
        + "e" * 40
        + "/data/xml/2024.acl.xml",
        "https://raw.githubusercontent.com/acl-org/acl-anthology/"
        + "e" * 40
        + "/data/xml/2025.acl.xml",
        "https://raw.githubusercontent.com/acl-org/acl-anthology/"
        + "e" * 40
        + "/data/xml/2026.acl.xml",
    ]


def test_rejects_truncated_manifest_instead_of_claiming_full_collection_coverage() -> None:
    client = ManifestQueueClient(
        _commit("c" * 40, "f" * 40),
        {"sha": "c" * 40, "truncated": True, "tree": []},
    )
    source = AclAnthologySourceAdapter(client=client, clock=lambda: NOW)
    with pytest.raises(ValueError, match="truncated status"):
        source.fetch_page({})

    missing_status = ManifestQueueClient(
        _commit("c" * 40, "f" * 40),
        {"sha": "c" * 40, "tree": []},
    )
    with pytest.raises(ValueError, match="truncated status"):
        AclAnthologySourceAdapter(client=missing_status, clock=lambda: NOW).fetch_page({})


def test_checkpoint_index_at_path_list_end_is_rejected_before_indexing() -> None:
    paths = ["data/xml/2025.acl.xml"]
    source = AclAnthologySourceAdapter(client=ManifestQueueClient(), clock=lambda: NOW)
    state = {
        "tree_sha": "a" * 40,
        "commit_sha": "b" * 40,
        "collection_paths": paths,
        "collection_index": len(paths),
    }
    with pytest.raises(ValueError, match="outside frozen path list"):
        source.fetch_page(state)


def test_xml_manifest_path_filter_fails_closed_on_nested_collection_paths() -> None:
    client = ManifestQueueClient(
        _commit("c" * 40, "f" * 40),
        {
            "sha": "c" * 40,
            "truncated": False,
            "tree": [{"path": "data/xml/archive/papers.xml", "type": "blob"}],
        },
    )
    with pytest.raises(ValueError, match="unsupported XML collection path"):
        AclAnthologySourceAdapter(client=client, clock=lambda: NOW).fetch_page({})


def test_twenty_thousand_collection_state_bound_is_supported() -> None:
    source = AclAnthologySourceAdapter(
        max_collections=20_000, client=ManifestQueueClient(), clock=lambda: NOW
    )
    assert source.max_collections == 20_000
