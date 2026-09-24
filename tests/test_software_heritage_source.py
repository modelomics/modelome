from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind
from modelome.sources.software_heritage import SoftwareHeritageOriginSourceAdapter

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
BUCKET = "https://softwareheritage.s3.amazonaws.com/"
RELEASE_LIST = "https://docs.softwareheritage.org/_sources/devel/swh-export/graph/dataset.rst.txt"


def _release_list() -> bytes:
    return b"""Full graph datasets
-------------------

.. _graph-dataset-2026-06-04:

2026-06-04
~~~~~~~~~~

Teaser datasets
---------------
"""


def _metadata(**overrides: Any) -> bytes:
    value: dict[str, Any] = {
        "flavor": "full",
        "formats": ["orc"],
        "object_types": ["origin", "revision"],
        "export_start": "2026-06-04T19:11:53.251510+00:00",
        "export_end": "2026-06-09T07:13:50.541733+00:00",
        "tool": {"name": "swh.export", "version": "1.11.7"},
    }
    value.update(overrides)
    return json.dumps(value).encode()


def _xml(
    prefix: str,
    *,
    common_prefixes: tuple[str, ...] = (),
    objects: tuple[dict[str, Any], ...] = (),
    truncated: bool = False,
    next_token: str | None = None,
) -> bytes:
    root = Element("ListBucketResult", xmlns="http://s3.amazonaws.com/doc/2006-03-01/")
    SubElement(root, "Prefix").text = prefix
    SubElement(root, "IsTruncated").text = str(truncated).lower()
    if next_token is not None:
        SubElement(root, "NextContinuationToken").text = next_token
    for value in common_prefixes:
        item = SubElement(root, "CommonPrefixes")
        SubElement(item, "Prefix").text = value
    for value in objects:
        item = SubElement(root, "Contents")
        for key, text in value.items():
            if key == "ChecksumAlgorithm" and isinstance(text, tuple):
                for algorithm in text:
                    SubElement(item, key).text = str(algorithm)
            else:
                SubElement(item, key).text = str(text)
    return tostring(root)


def _object(index: int, *, modified: str = "2026-06-09T07:13:50.000Z") -> dict[str, Any]:
    return {
        "Key": f"graph/2026-06-04/orc/origin/origin-shard-{index}.orc",
        "LastModified": modified,
        "ETag": f'"etag-{index}-16"',
        "ChecksumAlgorithm": ("CRC32",),
        "ChecksumType": "COMPOSITE",
        "Size": 131_000_000 + index,
    }


class Client:
    def __init__(self, *, objects: tuple[dict[str, Any], ...] | None = None) -> None:
        self.objects = objects or (_object(0), _object(1), _object(2))
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> HttpResponse:
        if url == RELEASE_LIST:
            return HttpResponse(status=200, headers={}, body=_release_list(), url=url)
        if url.endswith("/meta/export.json"):
            return HttpResponse(status=200, headers={}, body=_metadata(), url=url)
        assert url == BUCKET
        params = dict(kwargs["params"])
        self.calls.append(params)
        prefix = params["prefix"]
        if prefix == "graph/":
            body = _xml(
                prefix,
                common_prefixes=(
                    "graph/2026-03-02/",
                    "graph/not-a-release/",
                    "graph/2026-06-04/",
                ),
            )
        else:
            body = _xml(prefix, objects=self.objects)
        return HttpResponse(status=200, headers={}, body=body, url=BUCKET)


def _source(client: Client, **kwargs: Any) -> SoftwareHeritageOriginSourceAdapter:
    return SoftwareHeritageOriginSourceAdapter(
        client=client,
        page_size=2,
        clock=lambda: NOW,
        **kwargs,
    )


def test_dynamically_selects_latest_settled_export_and_pages_every_orc() -> None:
    client = Client()
    source = _source(client)

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)
    records = (*first.records, *second.records)

    assert first.complete is False
    assert second.complete is True
    assert first.upstream_count == second.upstream_count == 3
    assert [record.raw["manifest_index"] for record in records] == [0, 1, 2]
    assert all(record.kind is ArtifactKind.CATALOG_RECORD for record in records)
    assert all(record.models == () for record in records)
    assert all(record.raw["github_name_or_keyword_filter"] is False for record in records)
    assert records[0].canonical_url == (
        "https://softwareheritage.s3.amazonaws.com/graph/2026-06-04/orc/origin/origin-shard-0.orc"
    )
    assert records[0].raw["object_etag"] == '"etag-0-16"'
    assert records[0].raw["object_checksum_type"] == "COMPOSITE"
    assert records[0].links[0].crawl is False
    assert second.next_state["watermark"] == "2026-06-04"
    assert "stage" not in second.next_state
    # Begin discovers the release; resume stays pinned to the exact target prefix.
    assert [call["prefix"] for call in client.calls] == [
        "graph/",
        "graph/2026-06-04/orc/origin/",
        "graph/2026-06-04/orc/origin/",
    ]


def test_completed_snapshot_is_idempotent_and_detects_object_inventory_drift() -> None:
    client = Client(objects=(_object(0),))
    source = _source(client)
    completed = source.fetch_page({})
    assert completed.complete

    unchanged = source.fetch_page(completed.next_state)
    assert unchanged.complete
    assert unchanged.records == ()

    client.objects = (_object(0), _object(1))
    with pytest.raises(ValueError, match="immutable origin manifest changed"):
        source.fetch_page(completed.next_state)


def test_resumed_scan_fails_closed_if_frozen_manifest_changes() -> None:
    client = Client()
    source = _source(client)
    first = source.fetch_page({})
    client.objects = (*client.objects, _object(3))

    with pytest.raises(ValueError, match="frozen origin manifest changed"):
        source.fetch_page(first.next_state)


def test_freshly_modified_latest_prefix_is_not_silently_treated_as_complete() -> None:
    client = Client(objects=(_object(0, modified="2026-09-04T11:30:00Z"),))
    source = _source(client, settlement_age_hours=168)

    with pytest.raises(ValueError, match="settlement age"):
        source.fetch_page({})


@pytest.mark.parametrize(
    "metadata,error",
    [
        (_metadata(flavor="teaser"), "not a full graph"),
        (_metadata(formats=["csv"]), "does not publish ORC"),
        (_metadata(object_types=["revision"]), "does not include origins"),
        (_metadata(tool={"name": "other", "version": "1"}), "unexpected tool"),
    ],
)
def test_export_metadata_must_approve_a_completed_full_origin_orc(
    metadata: bytes,
    error: str,
) -> None:
    class MetadataClient(Client):
        def get(self, url: str, **kwargs: Any) -> HttpResponse:
            if url.endswith("/meta/export.json"):
                return HttpResponse(status=200, headers={}, body=metadata, url=url)
            return super().get(url, **kwargs)

    with pytest.raises(ValueError, match=error):
        _source(MetadataClient()).fetch_page({})


def test_coverage_contract_is_an_archive_snapshot_not_a_global_github_census() -> None:
    semantics = _source(Client()).coverage_semantics

    assert semantics["github_name_or_keyword_filter"] is False
    assert semantics["historical_github_census"] is False
    assert semantics["contains_repository_content"] is False
    assert semantics["snapshot_semantics"] is True


def test_paginated_s3_listing_follows_opaque_continuation_tokens() -> None:
    class PaginatedClient(Client):
        def get(self, url: str, **kwargs: Any) -> HttpResponse:
            if url == RELEASE_LIST:
                return HttpResponse(status=200, headers={}, body=_release_list(), url=url)
            if url.endswith("/meta/export.json"):
                return HttpResponse(status=200, headers={}, body=_metadata(), url=url)
            params = dict(kwargs["params"])
            self.calls.append(params)
            prefix = params["prefix"]
            token = params.get("continuation-token")
            if prefix == "graph/" and token is None:
                body = _xml(
                    prefix,
                    common_prefixes=("graph/2026-03-02/",),
                    truncated=True,
                    next_token="opaque+/=token",
                )
            elif prefix == "graph/":
                assert token == "opaque+/=token"
                body = _xml(prefix, common_prefixes=("graph/2026-06-04/",))
            else:
                body = _xml(prefix, objects=(_object(0),))
            return HttpResponse(status=200, headers={}, body=body, url=BUCKET)

    client = PaginatedClient()
    page = _source(client).fetch_page({})

    assert page.complete
    assert page.records[0].raw["release"] == "2026-06-04"
    assert client.calls[1]["continuation-token"] == "opaque+/=token"


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"page_size": 0}, "page size"),
        ({"list_page_size": 1001}, "list page size"),
        ({"settlement_age_hours": 0}, "settlement age"),
        ({"bucket_url": "http://example.org"}, "HTTPS"),
        ({"graph_prefix": "../graph/"}, "safe relative"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any], error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        SoftwareHeritageOriginSourceAdapter(client=Client(), **kwargs)
