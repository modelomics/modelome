from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.fetchers import (
    AwsBedrockModelCardFetcher,
    ContentTooLargeError,
    GitHubRepositoryFetcher,
    HuggingFaceModelCardFetcher,
    NvidiaNgcModelCardFetcher,
    OpenAIModelDocumentationFetcher,
    PrivateResourceError,
    PublicUrlPolicy,
    PyTorchHubModelPageFetcher,
    RobotsDeniedError,
    RobotsUnavailableError,
    UnsafeUrlError,
    WebPageFetcher,
    WeightReferenceFetcher,
)
from modelome.http import HttpClient, HttpFailure, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link

PUBLIC_ADDRESS = "93.184.216.34"


class FakeHttp:
    def __init__(self, responses: list[HttpResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, HttpResponse)
        return item


class RedirectAwareHttp(HttpClient):
    def __init__(self, destination: str) -> None:
        self.destination = destination
        self.response_returned = False

    def get(self, url, *, params=None, headers=None, redirect_validator=None):
        assert redirect_validator is not None
        redirect_validator(self.destination)
        self.response_returned = True
        return response("secret", url=self.destination, content_type="text/plain")


class DestinationDenyRobots:
    def __init__(self, denied: str) -> None:
        self.denied = denied
        self.calls: list[str] = []

    def assert_allowed(self, url: str) -> None:
        self.calls.append(url)
        if url == self.denied:
            raise RobotsDeniedError(f"robots.txt disallows {url}")


def response(
    payload: Any,
    *,
    url: str = "https://example.test/",
    content_type: str = "application/json",
    status: int = 200,
) -> HttpResponse:
    if isinstance(payload, bytes):
        body = payload
    elif isinstance(payload, str):
        body = payload.encode()
    else:
        body = json.dumps(payload).encode()
    return HttpResponse(status, {"content-type": content_type}, body, url)


def public_policy(*addresses: str) -> PublicUrlPolicy:
    resolved = addresses or (PUBLIC_ADDRESS,)
    return PublicUrlPolicy(resolver=lambda _host: resolved)


def test_github_fetcher_uses_discovered_repository_identifier() -> None:
    metadata = {
        "id": 123456,
        "node_id": "R_repo_node",
        "full_name": "lab/unseen-model",
        "html_url": "https://github.com/lab/unseen-model",
        "description": "Reference implementation",
        "created_at": "2025-01-01T00:00:00Z",
        "pushed_at": "2025-02-01T00:00:00Z",
        "homepage": "https://lab.example/model-card",
    }
    client = FakeHttp(
        [
            response(metadata, url="https://api.github.com/repos/lab/unseen-model"),
            response(
                b"# Unseen Model\nPaper: https://arxiv.org/abs/2501.00001",
                url="https://api.github.com/repos/lab/unseen-model/readme",
                content_type="text/markdown",
            ),
        ]
    )

    record = GitHubRepositoryFetcher(
        client=client,
        url_policy=public_policy(),
    ).fetch("https://github.com/lab/unseen-model/tree/main/examples")

    assert record.kind == ArtifactKind.CODE_REPOSITORY
    assert record.source_record_id == "lab/unseen-model"
    assert record.identifiers[0].value == "lab/unseen-model"
    assert record.identifiers[1].namespace == "github:repository-id"
    assert record.identifiers[1].value == "123456"
    assert record.identifiers[2].namespace == "github:node-id"
    assert record.identifiers[2].value == "R_repo_node"
    assert record.identifiers[3].namespace == "url"
    assert record.identifiers[3].value.endswith("/tree/main/examples")
    assert any(link.url == "https://arxiv.org/abs/2501.00001" for link in record.links)
    assert client.calls[0][0] == "https://api.github.com/repos/lab/unseen-model"


def test_github_fetcher_retains_requested_and_current_handles_after_rename() -> None:
    metadata = {
        "id": 987654,
        "node_id": "R_renamed_node",
        "full_name": "new-owner/current-name",
        "html_url": "https://github.com/new-owner/current-name",
        "description": "Renamed public repository",
    }
    client = FakeHttp(
        [
            response(metadata, url="https://api.github.com/repos/old-owner/old-name"),
            response(
                "# CurrentName\n\nA deep neural network.",
                url="https://api.github.com/repos/old-owner/old-name/readme",
                content_type="text/markdown",
            ),
        ]
    )

    record = GitHubRepositoryFetcher(
        client=client,
        url_policy=public_policy(),
    ).fetch("https://github.com/old-owner/old-name")

    assert record.canonical_url == "https://github.com/new-owner/current-name"
    assert record.identifiers == (
        Identifier("github:repository", "old-owner/old-name"),
        Identifier("github:repository", "new-owner/current-name"),
        Identifier("github:repository-id", "987654"),
        Identifier("github:node-id", "R_renamed_node"),
        Identifier("url", "https://github.com/old-owner/old-name"),
    )


def test_github_fetcher_rejects_private_repository_before_readme_or_storage() -> None:
    client = FakeHttp(
        [
            response(
                {
                    "full_name": "private-org/confidential",
                    "html_url": "https://github.com/private-org/confidential",
                    "private": True,
                },
                url="https://api.github.com/repos/private-org/confidential",
            )
        ]
    )
    fetcher = GitHubRepositoryFetcher(
        client=client,
        token="broadly-scoped-token",
        url_policy=public_policy(),
    )

    with pytest.raises(PrivateResourceError, match="excluded"):
        fetcher.fetch("https://github.com/private-org/confidential")

    assert len(client.calls) == 1


def test_weight_reference_fetcher_never_downloads_linked_checkpoint() -> None:
    fetcher = WeightReferenceFetcher(url_policy=public_policy())

    assert fetcher.accepts("https://weights.example/releases/model.safetensors?download=1")
    record = fetcher.fetch(
        "https://weights.example/releases/model.safetensors?download=1"
    )

    assert record.kind is ArtifactKind.WEIGHTS
    assert record.title == "model.safetensors"
    assert record.raw == {"reference_only": True, "suffix": ".safetensors"}
    assert record.identifiers[0].namespace == "url"


def test_huggingface_readme_fetcher_materializes_model_card_text() -> None:
    client = FakeHttp(
        [
            response(
                "User-agent: *\nDisallow:\n",
                url="https://huggingface.co/robots.txt",
                content_type="text/plain",
            ),
            response(
                "We introduce CardModel-X, with code at https://github.com/lab/card-x.",
                url="https://huggingface.co/lab/card-x/raw/a1b2c3/README.md",
                content_type="text/markdown",
            ),
        ]
    )
    fetcher = HuggingFaceModelCardFetcher(client, url_policy=public_policy())
    url = "https://huggingface.co/lab/card-x/raw/a1b2c3/README.md"

    assert fetcher.accepts(url)
    record = fetcher.fetch(url)

    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.models[0].name == "lab/card-x"
    assert record.models[0].identifiers[0].value == "lab/card-x"
    assert "CardModel-X" in record.text
    assert record.links[0].url == "https://github.com/lab/card-x"


def test_aws_bedrock_card_fetcher_bridges_card_path_and_serving_id() -> None:
    url = (
        "https://docs.aws.amazon.com/bedrock/latest/userguide/"
        "model-card-example-aurora-net-2.html"
    )
    client = FakeHttp(
        [
            response(
                "User-agent: *\nDisallow:\n",
                url="https://docs.aws.amazon.com/robots.txt",
                content_type="text/plain",
            ),
            response(
                """
                <html><head><title>AuroraNet 2 - Amazon Bedrock</title></head><body>
                  <h1>AuroraNet 2</h1>
                  <p>Use the following model IDs to access this model programmatically.</p>
                  <table><tr><th>Endpoint</th><th>Model ID</th></tr>
                    <tr><td>bedrock-runtime</td><td>example.aurora-net-2-v1:0</td></tr>
                    <tr><td>bedrock-mantle</td><td>example.aurora-net-2-v1:0</td></tr>
                  </table>
                  <a href="https://provider.example/cards/aurora-net-2">model card</a>
                  <code>modelId='example.aurora-net-2-v1:0'</code>
                </body></html>
                """,
                url=url,
                content_type="text/html",
            ),
        ]
    )
    fetcher = AwsBedrockModelCardFetcher(client, url_policy=public_policy())

    assert fetcher.accepts(url)
    record = fetcher.fetch(url)

    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.source_record_id == "aws-bedrock:example-aurora-net-2"
    assert record.title == "AuroraNet 2"
    assert record.models[0].identifiers == (
        Identifier("aws:bedrock-model-card", "example-aurora-net-2"),
        Identifier("aws:bedrock:model", "example.aurora-net-2-v1:0"),
    )
    assert record.raw["bedrock_model_ids"] == ["example.aurora-net-2-v1:0"]
    assert {link.url for link in record.links} == {
        "https://provider.example/cards/aurora-net-2"
    }


def test_openai_model_documentation_fetcher_keeps_exact_id_and_snapshots() -> None:
    url = "https://developers.openai.com/api/docs/models/gpt-4"
    client = FakeHttp(
        [
            response(
                "User-agent: *\nDisallow:\n",
                url="https://developers.openai.com/robots.txt",
                content_type="text/plain",
            ),
            response(
                """
                <html><head><title>GPT-4 Model | OpenAI API</title></head><body>
                  <h1>GPT-4</h1>
                  <p>An older high-intelligence GPT model.</p>
                  <h2>Snapshots</h2>
                  <p>gpt-4</p>
                  <p>gpt-4-0613</p><p>Deprecated</p>
                  <p>gpt-4-0314</p><p>Deprecated</p>
                  <h2>Rate limits</h2>
                  <a href="/api/docs/models/all">Model catalog</a>
                </body></html>
                """,
                url=url,
                content_type="text/html",
            ),
        ]
    )
    fetcher = OpenAIModelDocumentationFetcher(client, url_policy=public_policy())

    assert fetcher.accepts(url)
    assert not fetcher.accepts("https://developers.openai.com/api/docs/models/all")
    record = fetcher.fetch(url)

    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.source_record_id == "openai-documentation:gpt-4"
    assert record.title == "GPT-4"
    assert record.models[0].identifiers == (Identifier("openai:model", "gpt-4"),)
    assert record.models[0].aliases == ("gpt-4",)
    assert [release.version for release in record.releases] == [
        "gpt-4-0613",
        "gpt-4-0314",
    ]
    assert [release.identifiers for release in record.releases] == [
        (Identifier("openai:model-snapshot", "gpt-4-0613"),),
        (Identifier("openai:model-snapshot", "gpt-4-0314"),),
    ]
    assert record.links == ()


def test_pytorch_hub_page_fetcher_retains_only_declared_model_resources() -> None:
    url = "https://pytorch.org/hub/lab_hub-model"
    client = FakeHttp(
        [
            response(
                "User-agent: *\nDisallow:\n",
                url="https://pytorch.org/robots.txt",
                content_type="text/plain",
            ),
            response(
                """
                <html><head><title>Hub Model – PyTorch</title></head><body>
                  <h1>Hub Model</h1>
                  <a href="https://github.com/lab/hub-model/tree/main">Code</a>
                  <a href="https://github.com/lab/hub-model/issues">Issues</a>
                  <a href="https://arxiv.org/abs/2401.12345">Paper</a>
                  <a href="https://huggingface.co/lab/hub-model">Model card</a>
                  <a href="https://weights.example/hub-model.pth">Weights</a>
                  <a href="/hub/">Hub navigation</a>
                  <a href="https://pytorch.org/docs/">Framework navigation</a>
                </body></html>
                """,
                url=url,
                content_type="text/html",
            ),
        ]
    )
    fetcher = PyTorchHubModelPageFetcher(client, url_policy=public_policy())

    assert fetcher.accepts(url)
    assert not fetcher.accepts("https://pytorch.org/hub/")
    record = fetcher.fetch(url)

    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.source_record_id == "pytorch-hub:lab_hub-model"
    assert record.title == "Hub Model"
    assert record.models[0].identifiers == (
        Identifier("pytorch:hub-model-page", "lab_hub-model"),
    )
    assert {(link.url, link.relation) for link in record.links} == {
        ("https://github.com/lab/hub-model/tree/main", "code_reference"),
        ("https://arxiv.org/abs/2401.12345", "paper_reference"),
        ("https://huggingface.co/lab/hub-model", "model_card"),
        ("https://weights.example/hub-model.pth", "weights"),
    }


def test_ngc_version_metadata_fetcher_preserves_declared_card_resources() -> None:
    metadata_url = (
        "https://api.ngc.nvidia.com/v2/models/nvidia/tao/"
        "actionrecognitionnet/versions/trainable_rgb_3d"
    )
    payload = {
        "model": {
            "orgName": "nvidia",
            "teamName": "tao",
            "name": "actionrecognitionnet",
            "displayName": "Action Recognition Net",
            "description": (
                "Paper: https://arxiv.org/abs/2401.12345. "
                "Code: https://github.com/NVIDIA/tao_toolkit. "
                "Guide: https://docs.nvidia.com/tao/guide.html"
            ),
        },
        "modelVersion": {
            "versionId": "trainable_rgb_3d",
            "createdDate": "2024-08-29T17:59:14.698Z",
            "status": "UPLOAD_COMPLETE",
            "hasSignedVersion": True,
        },
    }
    client = FakeHttp([response(payload, url=metadata_url)])
    fetcher = NvidiaNgcModelCardFetcher(client, url_policy=public_policy())

    assert fetcher.accepts(metadata_url)
    record = fetcher.fetch(metadata_url)

    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.source_record_id == "nvidia/tao/actionrecognitionnet:trainable_rgb_3d"
    assert record.canonical_url == (
        "https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/actionrecognitionnet"
    )
    assert record.identifiers == (
        Identifier("ngc:model", "nvidia/tao/actionrecognitionnet"),
        Identifier(
            "ngc:model-version",
            "nvidia/tao/actionrecognitionnet:trainable_rgb_3d",
        ),
    )
    assert record.models[0].name == "Action Recognition Net"
    assert record.releases[0].version == "trainable_rgb_3d"
    assert record.releases[0].metadata == {
        "status": "UPLOAD_COMPLETE",
        "hasSignedVersion": True,
    }
    assert {(link.url, link.relation) for link in record.links} >= {
        ("https://arxiv.org/abs/2401.12345", "paper_reference"),
        ("https://github.com/NVIDIA/tao_toolkit", "code_reference"),
        ("https://docs.nvidia.com/tao/guide.html", "documentation_reference"),
    }
    assert all(
        link.crawl is False
        for link in record.links
        if link.relation in {"model_card", "model_card_metadata"}
    )


def test_ngc_version_metadata_fetcher_rejects_a_mismatched_response_identity() -> None:
    metadata_url = (
        "https://api.ngc.nvidia.com/v2/models/nvidia/tao/"
        "actionrecognitionnet/versions/trainable_rgb_3d"
    )
    payload = {
        "model": {
            "orgName": "nvidia",
            "teamName": "tao",
            "name": "different-model",
        },
        "modelVersion": {"versionId": "trainable_rgb_3d"},
    }
    fetcher = NvidiaNgcModelCardFetcher(
        FakeHttp([response(payload, url=metadata_url)]),
        url_policy=public_policy(),
    )

    with pytest.raises(ValueError, match="does not match the requested version URL"):
        fetcher.fetch(metadata_url)


def test_github_fetcher_skips_an_oversized_optional_readme() -> None:
    metadata = {
        "full_name": "lab/repository",
        "html_url": "https://github.com/lab/repository",
        "description": "Small metadata remains useful.",
    }
    client = FakeHttp(
        [
            response(metadata, url="https://api.github.com/repos/lab/repository"),
            response(
                b"x" * 33,
                url="https://api.github.com/repos/lab/repository/readme",
                content_type="text/markdown",
            ),
        ]
    )

    record = GitHubRepositoryFetcher(
        client=client,
        url_policy=public_policy(),
        max_metadata_bytes=1024,
        max_readme_bytes=32,
    ).fetch("https://github.com/lab/repository")

    assert record.text == "Small metadata remains useful."
    assert "_modelome_readme" not in record.raw


def test_github_fetcher_bounds_base64_decoded_readme() -> None:
    metadata = {
        "full_name": "lab/repository",
        "html_url": "https://github.com/lab/repository",
        "description": "Metadata only.",
    }
    encoded_readme = base64.b64encode(b"x" * 33).decode()
    client = FakeHttp(
        [
            response(metadata, url="https://api.github.com/repos/lab/repository"),
            response(
                {"encoding": "base64", "content": encoded_readme},
                url="https://api.github.com/repos/lab/repository/readme",
            ),
        ]
    )

    record = GitHubRepositoryFetcher(
        client=client,
        url_policy=public_policy(),
        max_metadata_bytes=1024,
        max_readme_bytes=32,
    ).fetch("https://github.com/lab/repository")

    assert record.text == "Metadata only."
    assert "_modelome_readme" not in record.raw


def test_web_fetcher_extracts_visible_evidence_and_safe_links() -> None:
    robots = "User-agent: *\nDisallow:\n"
    page = b"""
        <html><head><title>Provider Model Card</title>
        <meta property="og:type" content="article">
        <link rel="canonical alternate" href="https://provider.example/cards/new-model">
        <style>ignored</style></head><body>
        <p>We introduce Nova-3, a documented system.</p>
        <a href="https://github.com/lab/nova">Code</a>
        <a href="http://127.0.0.1/admin">unsafe</a>
        </body></html>
    """
    client = FakeHttp(
        [
            response(
                robots,
                url="https://provider.example/robots.txt",
                content_type="text/plain",
            ),
            response(
                page,
                url="https://provider.example/redirect?id=1",
                content_type="text/html; charset=utf-8",
            ),
        ]
    )

    record = WebPageFetcher(client, url_policy=public_policy()).fetch(
        "https://provider.example/redirect?id=1"
    )

    assert record.kind == ArtifactKind.BLOG_POST
    assert record.canonical_url == "https://provider.example/cards/new-model"
    assert record.identifiers[-1].namespace == "url"
    assert record.identifiers[-1].value == "https://provider.example/redirect?id=1"
    assert "ignored" not in record.text
    assert "Nova-3" in record.text
    assert [link.url for link in record.links] == ["https://github.com/lab/nova"]
    assert client.calls[0][0] == "https://provider.example/robots.txt"


def test_web_fetcher_retains_publisher_article_metadata_without_crawling_full_text() -> None:
    client = FakeHttp(
        [
            response(
                "User-agent: *\nDisallow:\n",
                url="https://publisher.example/robots.txt",
                content_type="text/plain",
            ),
            response(
                b"""
                <html><head><title>Publisher article</title>
                <meta name="citation_title" content="A model with publisher evidence">
                <meta name="citation_doi" content="doi:10.1038/FeatureNet.1">
                <meta name="citation_abstract" content="We introduce FeatureNet-1.">
                <meta name="citation_pdf_url" content="https://publisher.example/articles/x.pdf">
                </head><body><p>Abstract landing page.</p></body></html>
                """,
                url="https://publisher.example/articles/x",
                content_type="text/html",
            ),
        ]
    )

    record = WebPageFetcher(client, url_policy=public_policy()).fetch(
        "https://publisher.example/articles/x"
    )

    assert record.kind is ArtifactKind.PAPER
    assert Identifier("doi", "10.1038/featurenet.1") in record.identifiers
    assert "We introduce FeatureNet-1." in record.text
    assert record.links[-1] == Link(
        "https://publisher.example/articles/x.pdf",
        relation="full_text",
        locator="meta:citation_pdf_url",
        crawl=False,
    )


def test_web_fetcher_checks_destination_robots_before_following_redirect() -> None:
    destination = "https://destination.example/private/card"
    client = RedirectAwareHttp(destination)
    robots = DestinationDenyRobots(destination)
    fetcher = WebPageFetcher(
        client,
        url_policy=public_policy(),
        robots_policy=robots,
    )

    with pytest.raises(RobotsDeniedError, match="disallows"):
        fetcher.fetch("https://origin.example/card")

    assert client.response_returned is False
    assert robots.calls == ["https://origin.example/card", destination]


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.com/path",
        "http://localhost/path",
        "http://service.localhost/path",
        "http://127.0.0.1/path",
        "http://10.0.0.1/path",
        "http://169.254.169.254/latest/meta-data",
        "http://192.0.2.1/path",
        "http://100.64.0.1/path",
        "http://[::1]/path",
    ],
)
def test_public_url_policy_rejects_non_public_targets(url: str) -> None:
    policy = public_policy()

    with pytest.raises(UnsafeUrlError):
        policy.validate(url)


def test_public_url_policy_rejects_any_private_dns_answer() -> None:
    policy = public_policy(PUBLIC_ADDRESS, "10.0.0.8")

    with pytest.raises(UnsafeUrlError, match="non-public"):
        policy.validate("https://mixed-addresses.example/page")


def test_public_url_policy_is_injectable_and_allows_public_hosts() -> None:
    calls: list[str] = []

    def resolver(host: str) -> tuple[str, ...]:
        calls.append(host)
        return (PUBLIC_ADDRESS,)

    policy = PublicUrlPolicy(resolver=resolver)

    assert policy.validate("https://Public.Example/path#section") == (
        "https://public.example/path"
    )
    assert calls == ["public.example"]


def test_robots_disallow_prevents_page_request() -> None:
    client = FakeHttp(
        [
            response(
                "User-agent: modelome\nDisallow: /private\n",
                url="https://papers.example/robots.txt",
                content_type="text/plain",
            )
        ]
    )
    fetcher = WebPageFetcher(client, url_policy=public_policy())

    with pytest.raises(RobotsDeniedError, match="disallows"):
        fetcher.fetch("https://papers.example/private/model-card")

    assert [call[0] for call in client.calls] == ["https://papers.example/robots.txt"]


def test_robots_policy_is_cached_per_origin() -> None:
    client = FakeHttp(
        [
            response(
                "User-agent: *\nDisallow:\n",
                url="https://papers.example/robots.txt",
                content_type="text/plain",
            ),
            response(
                "<html><title>One</title><p>First</p></html>",
                url="https://papers.example/one",
                content_type="text/html",
            ),
            response(
                "<html><title>Two</title><p>Second</p></html>",
                url="https://papers.example/two",
                content_type="text/html",
            ),
        ]
    )
    fetcher = WebPageFetcher(client, url_policy=public_policy())

    assert fetcher.fetch("https://papers.example/one").title == "One"
    assert fetcher.fetch("https://papers.example/two").title == "Two"

    assert [call[0] for call in client.calls].count("https://papers.example/robots.txt") == 1


def test_robots_failure_behavior_is_explicit() -> None:
    denied_client = FakeHttp([HttpFailure("robots network failure")])
    denied = WebPageFetcher(denied_client, url_policy=public_policy())

    with pytest.raises(RobotsUnavailableError, match="network failure"):
        denied.fetch("https://provider.example/card")

    allowed_client = FakeHttp(
        [
            HttpFailure("robots network failure"),
            response(
                "<html><title>Available</title></html>",
                url="https://provider.example/card",
                content_type="text/html",
            ),
        ]
    )
    allowed = WebPageFetcher(
        allowed_client,
        url_policy=public_policy(),
        robots_failure_mode="allow",
    )

    assert allowed.fetch("https://provider.example/card").title == "Available"


def test_missing_robots_file_allows_page() -> None:
    client = FakeHttp(
        [
            response(
                "not found",
                status=404,
                url="https://provider.example/robots.txt",
                content_type="text/plain",
            ),
            response(
                "<html><title>Available</title></html>",
                url="https://provider.example/card",
                content_type="text/html",
            ),
        ]
    )

    record = WebPageFetcher(client, url_policy=public_policy()).fetch(
        "https://provider.example/card"
    )

    assert record.title == "Available"


def test_web_fetcher_rejects_oversized_page_before_decoding() -> None:
    client = FakeHttp(
        [
            response(
                "User-agent: *\nDisallow:\n",
                url="https://provider.example/robots.txt",
                content_type="text/plain",
            ),
            response(
                b"x" * 33,
                url="https://provider.example/card",
                content_type="text/plain",
            ),
        ]
    )
    fetcher = WebPageFetcher(
        client,
        url_policy=public_policy(),
        max_page_bytes=32,
    )

    with pytest.raises(ContentTooLargeError, match="32 bytes"):
        fetcher.fetch("https://provider.example/card")
