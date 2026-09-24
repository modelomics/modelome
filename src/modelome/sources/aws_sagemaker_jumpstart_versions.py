"""Enumerate SageMaker JumpStart public-hub model version IDs via AWS API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_HUB = "SageMakerPublicHub"
_HUB_CONTENTS = "https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ListHubContents.html"
_HUB_VERSIONS = (
    "https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ListHubContentVersions.html"
)
_CATALOG_URL = (
    "https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-latest.html"
)


class AwsSageMakerJumpStartVersionsSourceAdapter:
    """Enumerate exact public-hub model IDs and every available content version.

    ``client`` is a boto3 SageMaker client or a compatible injected object.
    The AWS API is signed and requires AWS credentials/IAM authorization; this
    adapter does not use or claim anonymous API access. Its sweeps are not
    authoritative snapshots because AWS does not document snapshot isolation
    across the paginated calls.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers Model content and available HubContentVersion values in the "
        "SageMakerPublicHub visible to the configured AWS principal. It requires "
        "AWS credentials, excludes proprietary Marketplace-only offerings and "
        "private hubs, and does not promise a transactionally consistent snapshot."
    )

    def __init__(
        self,
        *,
        name: str = "aws-sagemaker-jumpstart-versions",
        client: Any,
        page_size: int = 50,
        max_versions_per_model: int = 10_000,
        max_records_per_page: int = 20_000,
    ) -> None:
        if (
            not name.strip()
            or not 1 <= page_size <= 100
            or max_versions_per_model < 1
            or max_records_per_page < 1
        ):
            raise ValueError("name and positive bounded page/version/record limits are required")
        if not callable(getattr(client, "list_hub_contents", None)):
            raise ValueError("client must provide list_hub_contents")
        if not callable(getattr(client, "list_hub_content_versions", None)):
            raise ValueError("client must provide list_hub_content_versions")
        self.name = name
        self.client = client
        self.page_size = page_size
        self.max_versions_per_model = max_versions_per_model
        self.max_records_per_page = max_records_per_page
        self.checkpoint_signature = content_hash(
            {
                "adapter": "aws-sagemaker-jumpstart-versions-v1",
                "hub_name": _HUB,
                "hub_content_type": "Model",
                "page_size": page_size,
                "max_versions_per_model": max_versions_per_model,
                "max_records_per_page": max_records_per_page,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        next_token = state.get("next_token")
        if next_token is not None and (not isinstance(next_token, str) or not next_token):
            raise ValueError(f"{self.name}: invalid NextToken checkpoint")
        response = self.client.list_hub_contents(
            HubName=_HUB,
            HubContentType="Model",
            MaxResults=self.page_size,
            **({"NextToken": next_token} if next_token is not None else {}),
        )
        if not isinstance(response, Mapping):
            raise ValueError(f"{self.name}: list_hub_contents response must be an object")
        summaries = response.get("HubContentSummaries")
        if not isinstance(summaries, list):
            raise ValueError(f"{self.name}: HubContentSummaries must be a list")
        new_token = response.get("NextToken")
        if new_token is not None and (not isinstance(new_token, str) or not new_token):
            raise ValueError(f"{self.name}: invalid provider NextToken")
        if new_token == next_token and new_token is not None:
            raise ValueError(f"{self.name}: provider repeated NextToken")

        records: list[SourceRecord] = []
        seen_models: set[str] = set()
        for summary in summaries:
            if not isinstance(summary, Mapping):
                raise ValueError(f"{self.name}: malformed hub content summary")
            model_id = _required_text(summary.get("HubContentName"), "HubContentName")
            if model_id in seen_models:
                raise ValueError(f"{self.name}: duplicate model ID {model_id!r} in page")
            seen_models.add(model_id)
            if summary.get("HubContentType") not in {None, "Model"}:
                raise ValueError(f"{self.name}: unexpected content type for {model_id!r}")
            current_version = _required_text(summary.get("HubContentVersion"), "HubContentVersion")
            versions = self._versions(model_id)
            if current_version not in versions:
                raise ValueError(
                    f"{self.name}: current version {current_version!r} for {model_id!r} "
                    "was absent from ListHubContentVersions"
                )
            display_name = _required_text(
                summary.get("HubContentDisplayName") or model_id,
                "HubContentDisplayName",
            )
            for version in versions:
                records.append(self._record(model_id, display_name, version, summary))
                if len(records) > self.max_records_per_page:
                    raise ValueError(f"{self.name}: records exceed configured page limit")

        complete = new_token is None
        return SourcePage(
            tuple(records),
            {"next_token": new_token} if new_token is not None else {},
            complete,
            upstream_count=None,
            authoritative_snapshot=False,
        )

    def _versions(self, model_id: str) -> tuple[str, ...]:
        token: str | None = None
        seen_tokens: set[str] = set()
        versions: set[str] = set()
        while True:
            response = self.client.list_hub_content_versions(
                HubName=_HUB,
                HubContentType="Model",
                HubContentName=model_id,
                MaxResults=100,
                **({"NextToken": token} if token is not None else {}),
            )
            summaries = (
                response.get("HubContentSummaries") if isinstance(response, Mapping) else None
            )
            if not isinstance(summaries, list):
                raise ValueError(f"{self.name}: version summaries must be a list")
            for summary in summaries:
                if not isinstance(summary, Mapping):
                    raise ValueError(f"{self.name}: malformed version summary for {model_id!r}")
                version = _required_text(summary.get("HubContentVersion"), "HubContentVersion")
                versions.add(version)
                if len(versions) > self.max_versions_per_model:
                    raise ValueError(
                        f"{self.name}: version count for {model_id!r} exceeds configured limit"
                    )
            next_token = response.get("NextToken")
            if next_token is None:
                break
            if not isinstance(next_token, str) or not next_token or next_token in seen_tokens:
                raise ValueError(f"{self.name}: invalid/repeated version NextToken")
            seen_tokens.add(next_token)
            token = next_token
        if not versions:
            raise ValueError(f"{self.name}: no versions returned for {model_id!r}")
        return tuple(sorted(versions))

    def _record(
        self,
        model_id: str,
        display_name: str,
        version: str,
        summary: Mapping[str, Any],
    ) -> SourceRecord:
        model_local_id = f"model:{model_id}"
        identity = Identifier("aws:sagemaker-jumpstart-model", model_id)
        release_identity = Identifier(
            "aws:sagemaker-jumpstart-model-version", f"{model_id}@{version}"
        )
        model = ModelHint(
            model_local_id,
            display_name,
            identifiers=(identity,),
            aliases=(model_id,) if model_id.casefold() != display_name.casefold() else (),
            status=ModelStatus.DOCUMENTED,
        )
        release = ReleaseHint(
            f"release:{model_id}:{version}",
            model_local_id,
            version=version,
            identifiers=(release_identity,),
            metadata={
                "hub_name": _HUB,
                "hub_content_type": "Model",
                "hub_content_version": version,
                "hub_content_arn": summary.get("HubContentArn"),
                "support_status": summary.get("SupportStatus"),
            },
        )
        return SourceRecord(
            source_record_id=f"{model_id}@{version}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(_CATALOG_URL),
            title=f"{display_name} ({model_id}, version {version})",
            raw={
                "model_id": model_id,
                "hub_content_version": version,
                "hub_name": _HUB,
                "hub_content_type": "Model",
                "summary": dict(summary),
            },
            text=f"SageMaker JumpStart public hub model {display_name}; version {version}.",
            identifiers=(identity, release_identity),
            links=(
                Link(_CATALOG_URL, "catalog_documentation", crawl=False),
                Link(_HUB_CONTENTS, "api_documentation", crawl=False),
                Link(_HUB_VERSIONS, "version_api_documentation", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()
