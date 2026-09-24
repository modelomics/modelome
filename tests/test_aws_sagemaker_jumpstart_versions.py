from __future__ import annotations

from typing import Any

import pytest

from modelome.sources.aws_sagemaker_jumpstart_versions import (
    AwsSageMakerJumpStartVersionsSourceAdapter,
)
from modelome.sources.catalog import create_source


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_hub_contents(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("contents", kwargs))
        if "NextToken" not in kwargs:
            return {
                "HubContentSummaries": [
                    {
                        "HubContentName": "meta-llama-3-3-70b-instruct",
                        "HubContentDisplayName": "Meta Llama 3.3 70B Instruct",
                        "HubContentType": "Model",
                        "HubContentVersion": "2.0.0",
                        "HubContentArn": "arn:aws:sagemaker:us-west-2:aws:hub/SageMakerPublicHub",
                        "SupportStatus": "Supported",
                    }
                ],
                "NextToken": "contents-page-2",
            }
        return {
            "HubContentSummaries": [
                {
                    "HubContentName": "amazon-titan-text-express",
                    "HubContentDisplayName": "Amazon Titan Text Express",
                    "HubContentType": "Model",
                    "HubContentVersion": "1.0.0",
                }
            ]
        }

    def list_hub_content_versions(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("versions", kwargs))
        if kwargs["HubContentName"] == "meta-llama-3-3-70b-instruct":
            if "NextToken" not in kwargs:
                return {
                    "HubContentSummaries": [
                        {"HubContentVersion": "1.0.0"},
                    ],
                    "NextToken": "versions-page-2",
                }
            return {"HubContentSummaries": [{"HubContentVersion": "2.0.0"}]}
        return {"HubContentSummaries": [{"HubContentVersion": "1.0.0"}]}


def test_jumpstart_versions_factory_builds_signed_client_from_explicit_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import modelome.sources.catalog as catalog

    config = {
        "name": "aws-sagemaker-jumpstart-versions",
        "adapter": "aws_sagemaker_jumpstart_versions",
        "region": "us-west-2",
        "page_size": 25,
    }
    factory_calls: list[tuple[str | None, dict[str, str]]] = []

    def make_client(*, region_name=None, environ=None):
        factory_calls.append((region_name, dict(environ or {})))
        return _Client()

    monkeypatch.setattr(catalog, "create_signed_sagemaker_client", make_client)
    source = create_source(config, environ={"AWS_REGION": "us-east-1"})
    assert isinstance(source, AwsSageMakerJumpStartVersionsSourceAdapter)
    assert source.page_size == 25
    assert factory_calls == [("us-west-2", {"AWS_REGION": "us-east-1"})]


def test_load_sources_replaces_shared_http_client_with_signed_sagemaker_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import modelome.sources.catalog as catalog
    from modelome.http import HttpClient
    from modelome.sources.catalog import load_sources

    config_path = tmp_path / "jumpstart.toml"
    config_path.write_text(
        """\
[[source]]
name = "aws-sagemaker-jumpstart-versions"
adapter = "aws_sagemaker_jumpstart_versions"
enabled = true
region = "us-west-2"
page_size = 1
"""
    )
    factory_calls: list[tuple[str | None, dict[str, str]]] = []

    def make_client(*, region_name=None, environ=None):
        factory_calls.append((region_name, dict(environ or {})))
        return _Client()

    monkeypatch.setattr(catalog, "create_signed_sagemaker_client", make_client)
    sources = load_sources(
        config_path,
        client=HttpClient(),
        environ={"AWS_DEFAULT_REGION": "eu-west-1"},
    )

    assert isinstance(sources["aws-sagemaker-jumpstart-versions"],
                      AwsSageMakerJumpStartVersionsSourceAdapter)
    assert factory_calls == [("us-west-2", {"AWS_DEFAULT_REGION": "eu-west-1"})]


def test_jumpstart_versions_lists_exact_ids_and_paged_historical_versions() -> None:
    client = _Client()
    adapter = AwsSageMakerJumpStartVersionsSourceAdapter(client=client, page_size=1)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert not first.complete and not first.authoritative_snapshot
    assert first.next_state == {"next_token": "contents-page-2"}
    assert {record.source_record_id for record in first.records} == {
        "meta-llama-3-3-70b-instruct@1.0.0",
        "meta-llama-3-3-70b-instruct@2.0.0",
    }
    old_release = next(
        release
        for record in first.records
        for release in record.releases
        if release.version == "1.0.0"
    )
    assert old_release.identifiers[0].value == "meta-llama-3-3-70b-instruct@1.0.0"
    assert old_release.metadata["hub_name"] == "SageMakerPublicHub"
    assert second.complete and not second.next_state
    assert {record.source_record_id for record in second.records} == {
        "amazon-titan-text-express@1.0.0"
    }
    assert client.calls[0] == (
        "contents",
        {"HubName": "SageMakerPublicHub", "HubContentType": "Model", "MaxResults": 1},
    )
    assert any(kind == "versions" and "NextToken" in kwargs for kind, kwargs in client.calls)


def test_jumpstart_versions_rejects_missing_listed_current_version() -> None:
    class BadClient(_Client):
        def list_hub_content_versions(self, **kwargs: Any) -> dict[str, Any]:
            return {"HubContentSummaries": [{"HubContentVersion": "0.9.0"}]}

    with pytest.raises(ValueError, match="absent from ListHubContentVersions"):
        AwsSageMakerJumpStartVersionsSourceAdapter(client=BadClient(), page_size=1).fetch_page({})


def test_jumpstart_versions_rejects_repeated_contents_token() -> None:
    class BadClient(_Client):
        def list_hub_contents(self, **kwargs: Any) -> dict[str, Any]:
            return {"HubContentSummaries": [], "NextToken": kwargs["NextToken"]}

    with pytest.raises(ValueError, match="repeated NextToken"):
        AwsSageMakerJumpStartVersionsSourceAdapter(client=BadClient()).fetch_page(
            {"next_token": "same"}
        )


def test_jumpstart_versions_rejects_bad_response_shape() -> None:
    class BadClient(_Client):
        def list_hub_contents(self, **kwargs: Any) -> dict[str, Any]:
            return {"HubContentSummaries": None}

    with pytest.raises(ValueError, match="HubContentSummaries must be a list"):
        AwsSageMakerJumpStartVersionsSourceAdapter(client=BadClient()).fetch_page({})
