from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.kaggle import KaggleModelsSourceAdapter


class _Client:
    def __init__(self, *payloads: Mapping[str, Any]) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        if not self.payloads:
            raise AssertionError("unexpected request")
        return HttpResponse(200, {}, json.dumps(self.payloads.pop(0)).encode(), url)


def test_search_and_owner_filters_are_sent_and_part_of_checkpoint_identity() -> None:
    payload = {"models": [{"ref": "google/gemma"}], "totalResults": 1}
    client = _Client(payload)
    adapter = KaggleModelsSourceAdapter(
        client=client,
        search="gemma",
        owner="google",
        page_size=7,
    )

    page = adapter.fetch_page({})

    assert client.calls[0][1] == {
        "sortBy": "createTime",
        "pageSize": 7,
        "search": "gemma",
        "owner": "google",
    }
    assert page.complete is True
    assert len(page.records) == 1
    assert adapter.checkpoint_signature != KaggleModelsSourceAdapter(
        client=_Client(payload), search="gemma", page_size=7
    ).checkpoint_signature


def test_resume_preserves_search_and_owner_scope() -> None:
    client = _Client(
        {"models": [{"ref": "google/first"}], "nextPageToken": "next", "totalResults": 2},
        {"models": [{"ref": "google/second"}], "totalResults": 2},
    )
    adapter = KaggleModelsSourceAdapter(client=client, search="gemma", owner="google")

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert second.complete is True
    assert client.calls[1][1] == {
        "sortBy": "createTime",
        "pageSize": 100,
        "search": "gemma",
        "owner": "google",
        "pageToken": "next",
    }


def test_incomplete_page_sequence_restarts_instead_of_marking_scan_complete() -> None:
    client = _Client(
        {"models": [{"ref": "google/first"}], "totalResults": 2},
        {"models": [{"ref": "google/first"}], "totalResults": 1},
    )
    adapter = KaggleModelsSourceAdapter(client=client)

    incomplete = adapter.fetch_page({})
    assert not incomplete.complete
    assert incomplete.issues[0].stage == "source_pagination"
    assert incomplete.retry_state == {}
    assert incomplete.next_state == {}

    restarted = adapter.fetch_page(incomplete.retry_state)
    assert restarted.complete
    assert client.calls[1][1] == {"sortBy": "createTime", "pageSize": 100}


def test_model_list_rejects_total_drift_between_pages() -> None:
    client = _Client(
        {
            "models": [{"ref": "google/first"}],
            "nextPageToken": "next",
            "totalResults": 2,
        },
        {"models": [{"ref": "google/second"}], "totalResults": 3},
        {"models": [{"ref": "google/first"}], "nextPageToken": "stable", "totalResults": 3},
    )
    adapter = KaggleModelsSourceAdapter(client=client)

    first = adapter.fetch_page({})
    drift = adapter.fetch_page(first.next_state)
    assert not drift.complete
    assert drift.issues[0].stage == "source_pagination"
    assert "provider total changed during paginated scan" in drift.issues[0].error
    assert drift.retry_state == {}

    adapter.fetch_page(drift.retry_state)
    assert client.calls[2][1] == {"sortBy": "createTime", "pageSize": 100}


def test_model_list_pagination_cycle_fails_instead_of_rereading_pages() -> None:
    client = _Client(
        {"models": [{"ref": "google/first"}], "nextPageToken": "first"},
        {"models": [{"ref": "google/second"}], "nextPageToken": "second"},
        {"models": [{"ref": "google/third"}], "nextPageToken": "first"},
        {"models": [{"ref": "google/first"}]},
    )
    adapter = KaggleModelsSourceAdapter(client=client)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    cycle = adapter.fetch_page(second.next_state)
    assert cycle.issues[0].stage == "source_pagination"
    assert cycle.retry_state == {}

    adapter.fetch_page(cycle.retry_state)
    assert client.calls[3][1] == {"sortBy": "createTime", "pageSize": 100}


def test_version_file_metadata_is_opt_in_and_paginates_first_party_file_rows() -> None:
    client = _Client(
        {
            "models": [
                {
                    "ref": "google/gemma",
                    "instances": [{"id": 12, "slug": "2b", "framework": "PyTorch"}],
                }
            ],
            "totalResults": 1,
        },
        {"instances": [{"id": 12, "slug": "2b", "framework": "PyTorch"}]},
        {"versionList": {"versions": [{"id": 101, "versionNumber": 1}]}},
        {
            "files": [
                {
                    "name": "model.safetensors",
                    "size": 12,
                    "creationDate": "2025-01-01T00:00:00Z",
                }
            ],
            "nextPageToken": "more-files",
        },
        {"files": [{"name": "config.json", "size": 3}]},
    )
    adapter = KaggleModelsSourceAdapter(
        client=client,
        page_size=3,
        include_all_versions=True,
        include_version_files=True,
    )

    page = adapter.fetch_page({})

    assert page.records[0].releases[0].metadata["files"] == (
        {
            "name": "model.safetensors",
            "size": 12,
            "creation_date": "2025-01-01T00:00:00Z",
        },
        {"name": "config.json", "size": 3, "creation_date": None},
    )
    assert client.calls[-2:] == [
        (
            "https://www.kaggle.com/api/v1/models/google/gemma/PyTorch/2b/1/files",
            {"pageSize": 3},
        ),
        (
            "https://www.kaggle.com/api/v1/models/google/gemma/PyTorch/2b/1/files",
            {"pageSize": 3, "pageToken": "more-files"},
        ),
    ]


def test_version_file_metadata_requires_all_version_expansion() -> None:
    with pytest.raises(ValueError, match="requires include_all_versions"):
        KaggleModelsSourceAdapter(client=_Client(), include_version_files=True)


def test_empty_public_version_listing_does_not_invent_latest_release_or_files() -> None:
    client = _Client(
        {
            "models": [
                {
                    "ref": "google/gemma",
                    "instances": [
                        {
                            "id": 12,
                            "slug": "2b",
                            "framework": "PyTorch",
                            "versionNumber": 3,
                        }
                    ],
                }
            ],
            "totalResults": 1,
        },
        {
            "instances": [
                {
                    "id": 12,
                    "slug": "2b",
                    "framework": "PyTorch",
                    "versionNumber": 3,
                }
            ]
        },
        {"versionList": {"versions": []}},
    )
    page = KaggleModelsSourceAdapter(
        client=client,
        include_all_versions=True,
        include_version_files=True,
    ).fetch_page({})

    assert page.records[0].releases == ()
    assert len(client.calls) == 3


def test_private_or_disappeared_latest_version_is_not_resurrected_from_summary() -> None:
    client = _Client(
        {
            "models": [
                {
                    "ref": "google/gemma",
                    "instances": [
                        {
                            "id": 12,
                            "slug": "2b",
                            "framework": "PyTorch",
                            "versionNumber": 2,
                        }
                    ],
                }
            ],
            "totalResults": 1,
        },
        {
            "instances": [
                {
                    "id": 12,
                    "slug": "2b",
                    "framework": "PyTorch",
                    "versionNumber": 2,
                }
            ]
        },
        {
            "versionList": {
                "versions": [
                    {"id": 101, "versionNumber": 1},
                    {"id": 102, "versionNumber": 2, "isPrivate": True},
                ]
            }
        },
    )

    page = KaggleModelsSourceAdapter(
        client=client,
        include_all_versions=True,
    ).fetch_page({})

    assert [(release.version, release.revision) for release in page.records[0].releases] == [
        ("1", "101")
    ]


def test_historical_version_does_not_inherit_latest_version_id() -> None:
    client = _Client(
        {
            "models": [
                {
                    "ref": "google/gemma",
                    "instances": [
                        {
                            "id": 12,
                            "slug": "2b",
                            "framework": "PyTorch",
                            "versionNumber": 2,
                            "versionId": 202,
                        }
                    ],
                }
            ],
            "totalResults": 1,
        },
        {
            "instances": [
                {
                    "id": 12,
                    "slug": "2b",
                    "framework": "PyTorch",
                    "versionNumber": 2,
                    "versionId": 202,
                }
            ]
        },
        {"versionList": {"versions": [{"versionNumber": 1}]}},
    )

    page = KaggleModelsSourceAdapter(
        client=client,
        include_all_versions=True,
    ).fetch_page({})

    historical = page.records[0].releases[0]
    assert historical.version == "1"
    assert historical.revision is None
    assert not any(
        identifier.namespace == "kaggle:model-version"
        and identifier.value == "202"
        for identifier in historical.identifiers
    )


def test_conflicting_variation_ids_are_not_silently_merged() -> None:
    client = _Client(
        {
            "models": [
                {
                    "ref": "google/gemma",
                    "instances": [
                        {"id": 12, "slug": "2b", "framework": "PyTorch"},
                        {"id": 13, "slug": "2b", "framework": "PyTorch"},
                    ],
                }
            ],
            "totalResults": 1,
        },
        {"instances": []},
    )
    page = KaggleModelsSourceAdapter(
        client=client,
        include_all_versions=True,
    ).fetch_page({})

    assert not page.records
    assert len(page.issues) == 1
    assert "variation PyTorch/2b has conflicting IDs 12 and 13" in page.issues[0].error


def test_malformed_model_is_reported_without_dropping_valid_models() -> None:
    client = _Client(
        {
            "models": [
                {"title": "Missing ref"},
                {"ref": "google/gemma", "tags": ["NLP"]},
            ],
            "totalResults": 2,
        }
    )
    page = KaggleModelsSourceAdapter(client=client).fetch_page({})

    assert [record.source_record_id for record in page.records] == ["google/gemma"]
    assert len(page.issues) == 1
    assert page.issues[0].stage == "source_normalize"
    assert page.issues[0].summary["raw"] == {"title": "Missing ref"}


def test_optional_version_expansion_paginates_all_releases_and_builds_version_links() -> None:
    client = _Client(
        {
            "models": [
                {
                    "ref": "google/gemma",
                    "instances": [
                        {
                            "id": 12,
                            "slug": "2b",
                            "framework": "PyTorch",
                            "versionNumber": 3,
                            "versionId": "latest-id",
                            "baseModelInstanceId": 77,
                            "sourceUrl": "https://huggingface.co/google/gemma-2b",
                            "attestationKernelUrl": "https://www.kaggle.com/code/google/attestation",
                            "sigstoreState": "VERIFIED",
                            "baseModelInstanceInformation": {
                                "id": 77,
                                "modelSlug": "base-model",
                                "instanceSlug": "7b",
                                "framework": "PyTorch",
                            },
                            "downloadUrl": "/models/google/gemma/PyTorch/2b/3/download",
                        }
                    ],
                }
            ],
            "totalResults": 1,
        },
        {"instances": [{"id": 12, "slug": "2b", "framework": "PyTorch", "versionNumber": 3}]},
        {
            "versionList": {
                "versions": [
                    {
                        "versionNumber": 1,
                        "id": 101,
                        "url": "/models/google/gemma/PyTorch/2b/1",
                        "totalUncompressedBytes": 10,
                        "isTfHubModel": True,
                    },
                    {"versionNumber": 2, "id": 102, "totalUncompressedBytes": 20},
                    {"versionNumber": 4, "id": 104, "isPrivate": True},
                ]
            },
            "nextPageToken": "older-pages",
        },
        {
            "versionList": {"versions": [{"versionNumber": 3, "id": 103}]},
        },
    )
    adapter = KaggleModelsSourceAdapter(
        client=client,
        page_size=2,
        include_all_versions=True,
    )

    page = adapter.fetch_page({})

    assert len(page.records[0].releases) == 3
    assert [release.version for release in page.records[0].releases] == ["1", "2", "3"]
    assert [release.revision for release in page.records[0].releases] == [
        "101",
        "102",
        "103",
    ]
    assert all(
        any(identifier.value == "12" for identifier in release.identifiers)
        for release in page.records[0].releases
    )
    assert page.records[0].releases[0].metadata["total_uncompressed_bytes"] == 10
    assert page.records[0].releases[0].metadata["is_tfhub_model"] is True
    assert page.records[0].releases[-1].metadata["base_model_instance_id"] == "77"
    assert page.records[0].releases[-1].metadata[
        "base_model_instance_information"
    ] == {
        "id": 77,
        "modelSlug": "base-model",
        "instanceSlug": "7b",
        "framework": "PyTorch",
    }
    assert page.records[0].releases[-1].metadata["source_url"] == (
        "https://huggingface.co/google/gemma-2b"
    )
    assert page.records[0].releases[-1].metadata["attestation_kernel_url"] == (
        "https://www.kaggle.com/code/google/attestation"
    )
    assert page.records[0].releases[-1].metadata["sigstore_state"] == "VERIFIED"
    assert {
        link.url for link in page.records[0].links if link.relation == "source_reference"
    } == {"https://huggingface.co/google/gemma-2b"}
    assert {
        link.url for link in page.records[0].links if link.relation == "attestation"
    } == {"https://www.kaggle.com/code/google/attestation"}
    assert {
        link.url for link in page.records[0].links if link.relation == "weights"
    } == {
        "https://www.kaggle.com/api/v1/models/google/gemma/PyTorch/2b/1/download",
        "https://www.kaggle.com/api/v1/models/google/gemma/PyTorch/2b/2/download",
        "https://www.kaggle.com/api/v1/models/google/gemma/PyTorch/2b/3/download",
    }
    assert {
        link.url for link in page.records[0].links if link.relation == "model_version"
    } == {
        "https://www.kaggle.com/models/google/gemma/PyTorch/2b/1",
        "https://www.kaggle.com/models/google/gemma/PyTorch/2b/2",
        "https://www.kaggle.com/models/google/gemma/PyTorch/2b/3",
    }
    assert client.calls[1:] == [
        (
            "https://www.kaggle.com/api/v1/models/google/gemma/list",
            {"pageSize": 2},
        ),
        (
            "https://www.kaggle.com/api/v1/models/google/gemma/PyTorch/2b/list",
            {"pageSize": 2},
        ),
        (
            "https://www.kaggle.com/api/v1/models/google/gemma/PyTorch/2b/list",
            {"pageSize": 2, "pageToken": "older-pages"},
        ),
    ]


def test_version_pagination_cycle_fails_instead_of_returning_partial_releases() -> None:
    client = _Client(
        {
            "models": [
                {
                    "ref": "google/gemma",
                    "instances": [{"slug": "2b", "framework": "PyTorch"}],
                }
            ],
            "totalResults": 1,
        },
        {"instances": [{"slug": "2b", "framework": "PyTorch"}]},
        {"versionList": {"versions": []}, "nextPageToken": "first"},
        {"versionList": {"versions": []}, "nextPageToken": "second"},
        {"versionList": {"versions": []}, "nextPageToken": "first"},
    )

    page = KaggleModelsSourceAdapter(
        client=client,
        include_all_versions=True,
    ).fetch_page({})

    assert not page.records
    assert len(page.issues) == 1
    assert "version pagination token did not advance" in page.issues[0].error


def test_all_versions_are_scoped_and_paginated_independently_per_variation() -> None:
    client = _Client(
        {
            "models": [
                {
                    "ref": "google/gemma",
                    "instances": [
                        {
                            "id": 12,
                            "slug": "2b",
                            "framework": "PyTorch",
                            "versionNumber": 2,
                        },
                    ],
                }
            ],
            "totalResults": 1,
        },
        {
            "instances": [
                {"id": 12, "slug": "2b", "framework": "PyTorch", "versionNumber": 2}
            ],
            "nextPageToken": "more-variations",
        },
        {"instances": [{"id": 13, "slug": "7b", "framework": "PyTorch", "versionNumber": 1}]},
        {
            "versionList": {
                "versions": [
                    {
                        "id": 101,
                        "versionNumber": 1,
                        "modelInstanceId": 12,
                        "variationSlug": "2b",
                        "framework": "PyTorch",
                    }
                ]
            },
            "nextPageToken": "older",
        },
        {
            "versionList": {
                "versions": [
                    {
                        "id": 102,
                        "versionNumber": 2,
                        "modelInstanceId": 12,
                        "variationSlug": "2b",
                        "framework": "PyTorch",
                    }
                ]
            },
        },
        {
            "versionList": {
                "versions": [
                    {
                        "id": 201,
                        "versionNumber": 1,
                        "modelInstanceId": 13,
                        "variationSlug": "7b",
                        "framework": "PyTorch",
                    }
                ]
            },
        },
    )

    page = KaggleModelsSourceAdapter(
        client=client,
        page_size=1,
        include_all_versions=True,
    ).fetch_page({})

    releases = page.records[0].releases
    assert [(release.metadata["instance_slug"], release.version) for release in releases] == [
        ("2b", "1"),
        ("2b", "2"),
        ("7b", "1"),
    ]
    assert [release.revision for release in releases] == ["101", "102", "201"]
    assert client.calls[1:] == [
        (
            "https://www.kaggle.com/api/v1/models/google/gemma/list",
            {"pageSize": 1},
        ),
        (
            "https://www.kaggle.com/api/v1/models/google/gemma/list",
            {"pageSize": 1, "pageToken": "more-variations"},
        ),
        (
            "https://www.kaggle.com/api/v1/models/google/gemma/PyTorch/2b/list",
            {"pageSize": 1},
        ),
        (
            "https://www.kaggle.com/api/v1/models/google/gemma/PyTorch/2b/list",
            {"pageSize": 1, "pageToken": "older"},
        ),
        (
            "https://www.kaggle.com/api/v1/models/google/gemma/PyTorch/7b/list",
            {"pageSize": 1},
        ),
    ]


def test_version_rows_for_another_variation_are_rejected() -> None:
    client = _Client(
        {
            "models": [
                {
                    "ref": "google/gemma",
                    "instances": [
                        {"id": 12, "slug": "2b", "framework": "PyTorch"}
                    ],
                }
            ],
            "totalResults": 1,
        },
        {"instances": [{"id": 12, "slug": "2b", "framework": "PyTorch"}]},
        {
            "versionList": {
                "versions": [
                    {
                        "id": 201,
                        "versionNumber": 1,
                        "modelInstanceId": 13,
                        "variationSlug": "7b",
                        "framework": "PyTorch",
                    }
                ]
            },
        },
    )

    page = KaggleModelsSourceAdapter(
        client=client,
        include_all_versions=True,
    ).fetch_page({})

    assert not page.records
    assert len(page.issues) == 1
    assert "belongs to model instance 13, expected 12" in page.issues[0].error
