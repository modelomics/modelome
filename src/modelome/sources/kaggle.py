"""Public Kaggle Models catalog ingestion.

The provider's documented ``kaggle models list`` command uses the public
``/api/v1/models/list`` endpoint with an opaque page token.  This adapter retains
each source-declared model, variation/version, download reference, provenance,
training-data link, and external base-model link without downloading model bytes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelRelationHint,
    ModelStatus,
    ReleaseHint,
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash, extract_urls, identifier_from_url


class KaggleModelsSourceAdapter:
    """Page every public model returned by Kaggle's official models listing."""

    def __init__(
        self,
        *,
        name: str = "kaggle-models",
        url: str = "https://www.kaggle.com/api/v1/models/list",
        page_size: int = 100,
        sort_by: str = "createTime",
        search: str | None = None,
        owner: str | None = None,
        include_all_versions: bool = False,
        include_version_files: bool = False,
        artifact_kind: str | ArtifactKind = ArtifactKind.MODEL_CARD,
        client: HttpClient | Any | None = None,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, "catalog URL")
        self.page_size = int(page_size)
        if not 1 <= self.page_size <= 1_000:
            raise ValueError(f"{self.name}: page_size must be from 1 through 1000")
        self.sort_by = _required_text(sort_by, "sort_by")
        self.search = _optional_text(search)
        self.owner = _optional_text(owner)
        if not isinstance(include_all_versions, bool):
            raise ValueError(f"{self.name}: include_all_versions must be a boolean")
        if not isinstance(include_version_files, bool):
            raise ValueError(f"{self.name}: include_version_files must be a boolean")
        if include_version_files and not include_all_versions:
            raise ValueError(
                f"{self.name}: include_version_files requires include_all_versions"
            )
        self.include_all_versions = include_all_versions
        self.include_version_files = include_version_files
        if self.sort_by not in {
            "hotness",
            "downloadCount",
            "voteCount",
            "notebookCount",
            "createTime",
        }:
            raise ValueError(f"{self.name}: unsupported Kaggle model sort {self.sort_by!r}")
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.client = client or HttpClient()
        self.checkpoint_signature = content_hash(
            {
                "adapter": "kaggle-models-v1",
                "url": self.url,
                "page_size": self.page_size,
                "sort_by": self.sort_by,
                "search": self.search,
                "owner": self.owner,
                "include_all_versions": self.include_all_versions,
                "include_version_files": self.include_version_files,
                "artifact_kind": self.artifact_kind.value,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        token = _optional_text(state.get("next_page_token"))
        resuming = token is not None
        raw_items_seen = _state_count(state, "raw_items_seen") if resuming else 0
        scan_total = _state_count(state, "scan_total") if resuming else None
        seen_tokens = _state_tokens(state, "seen_page_tokens") if resuming else set()
        if token is not None and token in seen_tokens:
            raise ValueError(f"{self.name}: pagination token was already consumed")
        params: dict[str, str | int] = {
            "sortBy": self.sort_by,
            "pageSize": self.page_size,
        }
        if self.search is not None:
            params["search"] = self.search
        if self.owner is not None:
            params["owner"] = self.owner
        if token is not None:
            params["pageToken"] = token
        response: HttpResponse = self.client.get(
            self.url,
            params=params,
            headers={"Accept": "application/json"},
        )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: expected a JSON object from {response.url}")
        items = payload.get("models")
        if not _is_sequence(items):
            raise ValueError(f"{self.name}: response models must be a list")
        raw_items_seen = (raw_items_seen or 0) + len(items)
        response_total = _optional_nonnegative_int(payload.get("totalResults"))
        total_drift = (
            response_total is not None
            and scan_total is not None
            and response_total != scan_total
        )
        if response_total is not None and not total_drift:
            scan_total = max(scan_total or 0, response_total)

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for index, item in enumerate(items):
            try:
                records.append(self._record(item, index))
            except (TypeError, ValueError) as error:
                raw = dict(item) if isinstance(item, Mapping) else {"value": repr(item)[:1000]}
                record_id = _optional_text(item.get("ref")) if isinstance(item, Mapping) else None
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            record_id
                            or f"{self.name}:malformed:{content_hash(raw)[:32]}"
                        ),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"index": index, "raw": raw},
                    )
                )

        next_token = _optional_text(payload.get("nextPageToken"))
        if next_token is not None and (
            next_token == token or next_token in seen_tokens
        ):
            return self._restart_page(
                records,
                issues,
                state,
                "pagination token did not advance",
                response_total if response_total is not None else scan_total,
            )
        if total_drift:
            return self._restart_page(
                records,
                issues,
                state,
                f"provider total changed during paginated scan ({scan_total} to {response_total})",
                response_total,
            )
        if next_token is None and scan_total is not None and raw_items_seen < scan_total:
            return self._restart_page(
                records,
                issues,
                state,
                f"pagination ended after {raw_items_seen} model(s), "
                f"before the provider-reported total of {scan_total}",
                response_total if response_total is not None else scan_total,
            )
        next_state: dict[str, Any] = {}
        if next_token is not None:
            consumed_tokens = set(seen_tokens)
            if token is not None:
                consumed_tokens.add(token)
            next_state = {
                "next_page_token": next_token,
                "raw_items_seen": raw_items_seen,
            }
            if consumed_tokens:
                next_state["seen_page_tokens"] = sorted(consumed_tokens)
            if scan_total is not None:
                next_state["scan_total"] = scan_total
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=next_token is None,
            upstream_count=response_total if response_total is not None else scan_total,
            issues=tuple(issues),
        )

    def _restart_page(
        self,
        records: Sequence[SourceRecord],
        issues: Sequence[SourceIssue],
        state: Mapping[str, Any],
        error: str,
        upstream_count: int | None,
    ) -> SourcePage:
        issue_id = content_hash({"state": dict(state), "error": error})[:32]
        issue = SourceIssue(
            source_record_id=f"{self.name}:pagination:{issue_id}",
            stage="source_pagination",
            error=f"{self.name}: {error}; restarting the scan from page one",
            summary={"prior_state": dict(state), "restart_state": {}},
        )
        return SourcePage(
            records=tuple(records),
            next_state={},
            complete=False,
            upstream_count=upstream_count,
            issues=tuple((*issues, issue)),
            retry_state={},
        )

    def _record(self, item: Any, index: int) -> SourceRecord:
        if not isinstance(item, Mapping):
            raise ValueError(f"catalog item {index} is not an object")
        if item.get("isPrivate") is True or item.get("is_private") is True:
            raise ValueError("public list returned a private model")
        model_ref = _required_text(item.get("ref"), f"catalog item {index} ref")
        model_url = _model_url(model_ref, item.get("url"))
        title = _optional_text(item.get("title")) or model_ref
        model_identifier = Identifier("kaggle:model", model_ref)
        local_id = f"{model_ref}#model"
        model = ModelHint(
            local_id=local_id,
            name=title,
            identifiers=(model_identifier,),
            aliases=tuple(
                value
                for value in (_optional_text(item.get("slug")), _optional_text(item.get("ref")))
                if value and value != title
            ),
            status=ModelStatus.RELEASED,
            locator="$.ref",
        )

        links: list[Link] = [Link(model_url, relation="model_page", locator="$.url")]
        text_parts = [
            value
            for value in (
                _optional_text(item.get("subtitle")),
                _optional_text(item.get("description")),
                _optional_text(item.get("provenanceSources")),
            )
            if value
        ]
        for url in extract_urls(text_parts):
            links.append(Link(url, relation="documentation_reference", locator="$.description"))
        provenance = _optional_text(item.get("provenanceSources"))
        if provenance:
            for url in extract_urls(provenance):
                links.append(Link(url, relation="provenance", locator="$.provenanceSources"))
        for index, value in enumerate(_sequence(item.get("modelVersionLinks"))):
            for url in extract_urls(value):
                links.append(
                    Link(
                        url,
                        relation="model_version_reference",
                        locator=f"$.modelVersionLinks[{index}]",
                    )
                )

        releases: list[ReleaseHint] = []
        relations: list[ModelRelationHint] = []
        instances = _sequence(item.get("instances"))
        if self.include_all_versions:
            instances = self._model_instances(model_ref, instances)
        for instance_index, instance in enumerate(instances):
            if not isinstance(instance, Mapping):
                continue
            version_items = (
                self._instance_versions(model_ref, instance)
                if self.include_all_versions
                else ()
            )
            for version_item in _versions_with_latest(instance, version_items):
                release, instance_links, relation = self._instance(
                    model_ref,
                    local_id,
                    instance,
                    instance_index,
                    version_item=version_item,
                )
                releases.append(release)
                links.extend(instance_links)
                if relation is not None:
                    relations.append(relation)

        tags = []
        for tag in _sequence(item.get("tags")):
            if isinstance(tag, Mapping):
                label = _optional_text(tag.get("fullPath")) or _optional_text(tag.get("name"))
            else:
                label = _optional_text(tag)
            if label:
                tags.append(label)
        if tags:
            text_parts.append("tags: " + ", ".join(sorted(set(tags))))

        return SourceRecord(
            source_record_id=model_ref,
            kind=self.artifact_kind,
            canonical_url=model_url,
            title=title,
            raw=dict(item),
            text="\n".join(text_parts),
            published_at=_optional_text(item.get("createTime")),
            modified_at=_optional_text(item.get("updateTime")),
            identifiers=(model_identifier,),
            links=_unique_links(links),
            models=(model,),
            model_relations=tuple(relations),
            releases=tuple(releases),
        )

    def _model_instances(
        self,
        model_ref: str,
        embedded: Sequence[Any],
    ) -> tuple[Any, ...]:
        """Page the first-party variation listing and merge embedded summaries."""
        owner, separator, model_slug = model_ref.partition("/")
        if not separator or not owner or not model_slug:
            raise ValueError(f"invalid Kaggle model ref for variation lookup: {model_ref!r}")
        parts = urlsplit(self.url)
        path = "/".join(quote(part, safe="") for part in (owner, model_slug))
        url = f"{parts.scheme}://{parts.netloc}/api/v1/models/{path}/list"
        token: str | None = None
        seen_tokens: set[str] = set()
        listed: list[Mapping[str, Any]] = []
        while True:
            params: dict[str, str | int] = {"pageSize": self.page_size}
            if token is not None:
                params["pageToken"] = token
            response: HttpResponse = self.client.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
            )
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: variations for {model_ref} returned HTTP {response.status}"
                )
            if len(response.body) > 4 * 1024 * 1024:
                raise ValueError(f"{self.name}: model variations response is too large")
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError(f"{self.name}: model variations response is not an object")
            page_instances = payload.get("instances")
            if not _is_sequence(page_instances):
                raise ValueError(f"{self.name}: model variations are not a list")
            if any(
                value is not None and not isinstance(value, Mapping)
                for value in page_instances
            ):
                raise ValueError(f"{self.name}: model variation row is not an object")
            listed.extend(value for value in page_instances if isinstance(value, Mapping))
            next_token = _optional_text(
                payload.get("nextPageToken", payload.get("next_page_token"))
            )
            if next_token is None:
                break
            if next_token in seen_tokens or next_token == token:
                raise ValueError(f"{self.name}: variation pagination token did not advance")
            seen_tokens.add(next_token)
            token = next_token

        # The public model listing embeds a latest-instance summary, while the
        # dedicated endpoint is cursor-paginated. Merge by variation identity so
        # fields present only in the embedded summary survive.
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for instance in (*embedded, *listed):
            if not isinstance(instance, Mapping):
                continue
            framework = _optional_text(instance.get("framework")) or "unspecified"
            slug = _optional_text(instance.get("slug"))
            if slug is None:
                key = ("id", _optional_text(instance.get("id")) or content_hash(dict(instance)))
            else:
                key = (framework, slug)
            merged[key] = {**merged.get(key, {}), **instance}
        return tuple(merged.values())

    def _instance(
        self,
        model_ref: str,
        model_local_id: str,
        instance: Mapping[str, Any],
        index: int,
        *,
        version_item: Mapping[str, Any] | None = None,
    ) -> tuple[ReleaseHint, tuple[Link, ...], ModelRelationHint | None]:
        parent_instance = instance
        instance = {**parent_instance, **(version_item or {})}
        framework = _optional_text(parent_instance.get("framework")) or "unspecified"
        slug = _optional_text(parent_instance.get("slug")) or f"instance-{index}"
        version = (
            _optional_text(
                (version_item or {}).get(
                    "versionNumber",
                    (version_item or {}).get("version_number"),
                )
            )
            or _optional_text(parent_instance.get("versionNumber"))
            or "unknown"
        )
        instance_ref = f"{model_ref}/{framework}/{slug}"
        release_ref = f"{instance_ref}/{version}"
        version_id = None
        if version_item is not None:
            version_id = (
                _optional_text(version_item.get("versionId"))
                or _optional_text(version_item.get("version_id"))
                or _optional_text(version_item.get("id"))
            )
        version_id = version_id or _optional_text(parent_instance.get("versionId"))
        instance_id = _optional_text(parent_instance.get("id"))
        identifiers = [
            Identifier("kaggle:model-instance", instance_ref),
            Identifier("kaggle:model-instance-version", release_ref),
        ]
        if version_id:
            identifiers.append(Identifier("kaggle:model-version", version_id))
        if instance_id:
            identifiers.append(Identifier("kaggle:model-instance-id", instance_id))
        links: list[Link] = []
        instance_url = _optional_web_url(parent_instance.get("url"), self.url)
        if instance_url:
            links.append(Link(instance_url, relation="model_variant", locator="$.instances.url"))
        if version_item is not None:
            version_path = "/".join(
                (
                    quote(model_ref, safe="/"),
                    quote(framework, safe=""),
                    quote(slug, safe=""),
                    quote(version, safe=""),
                )
            )
            version_url = _optional_web_url(
                version_item.get("url"),
                "https://www.kaggle.com",
            ) or canonicalize_url(f"https://www.kaggle.com/models/{version_path}")
            links.append(
                Link(
                    version_url,
                    relation="model_version",
                    locator="$.instances.versions[].url",
                    crawl=False,
                )
            )
        declared_download_url = (
            "https://www.kaggle.com/api/v1/models/"
            f"{quote(model_ref, safe='/')}/{quote(framework, safe='')}/"
            f"{quote(slug, safe='')}/{quote(version, safe='')}/download"
            if version_item is not None
            else instance.get("downloadUrl")
        )
        download_url = _optional_web_url(declared_download_url, self.url)
        if download_url:
            links.append(
                Link(
                    download_url,
                    relation="weights",
                    locator="$.instances.downloadUrl",
                    crawl=False,
                )
            )
        source_url = _optional_web_url(
            instance.get("sourceUrl", instance.get("source_url")), self.url
        )
        if source_url:
            links.append(
                Link(
                    source_url,
                    relation="source_reference",
                    locator="$.instances.sourceUrl",
                    crawl=False,
                )
            )
        attestation_url = _optional_web_url(
            instance.get("attestationKernelUrl", instance.get("attestation_kernel_url")),
            self.url,
        )
        if attestation_url:
            links.append(
                Link(
                    attestation_url,
                    relation="attestation",
                    locator="$.instances.attestationKernelUrl",
                    crawl=False,
                )
            )
        for dataset_index, value in enumerate(_sequence(instance.get("trainingData"))):
            dataset_url = _optional_web_url(value, self.url)
            if dataset_url:
                links.append(
                    Link(
                        dataset_url,
                        relation="dataset",
                        locator=f"$.instances.trainingData[{dataset_index}]",
                    )
                )

        base_url = _optional_web_url(instance.get("externalBaseModelUrl"), self.url)
        relation = None
        if base_url:
            links.append(
                Link(
                    base_url,
                    relation="base_model",
                    locator="$.instances.externalBaseModelUrl",
                )
            )
            target_name, target_identifiers = _model_identity(base_url)
            relation = ModelRelationHint(
                subject_local_id=model_local_id,
                predicate="base_model",
                target=ModelHint(
                    local_id=f"{model_ref}#base-model-{index}",
                    name=target_name,
                    identifiers=target_identifiers,
                    status=ModelStatus.DOCUMENTED,
                    locator="$.instances.externalBaseModelUrl",
                ),
                locator="$.instances.externalBaseModelUrl",
            )
        version_files = (
            self._version_files(model_ref, framework, slug, version)
            if self.include_version_files
            else ()
        )
        release = ReleaseHint(
            local_id=(
                f"{model_ref}#release:{instance_id or instance_ref}"
                + (f"/{version_id or version}" if version_item is not None else "")
            ),
            model_local_id=model_local_id,
            version=version,
            revision=version_id,
            identifiers=tuple(identifiers),
            metadata={
                "framework": framework,
                "instance_slug": slug,
                "license": _optional_text(instance.get("licenseName")),
                "fine_tunable": instance.get("fineTunable"),
                "model_instance_type": _optional_text(instance.get("modelInstanceType")),
                "base_model_instance_id": _optional_text(
                    instance.get("baseModelInstanceId", instance.get("base_model_instance_id"))
                ),
                "base_model_instance_information": _mapping_or_none(
                    instance.get(
                        "baseModelInstanceInformation",
                        instance.get("base_model_instance_information"),
                    )
                ),
                "source_url": source_url,
                "attestation_kernel_url": attestation_url,
                "sigstore_state": _optional_text(
                    instance.get("sigstoreState", instance.get("sigstore_state"))
                ),
                "is_tfhub_model": _optional_bool(
                    (version_item or {}).get(
                        "isTfhubModel",
                        (version_item or {}).get("is_tfhub_model"),
                    )
                ),
                "files": version_files,
                "total_uncompressed_bytes": _optional_nonnegative_int(
                    instance.get("totalUncompressedBytes")
                ),
                "overview": _optional_text(instance.get("overview")),
                "usage": _optional_text(instance.get("usage")),
            },
            locator=f"$.instances[{index}]",
        )
        return release, _unique_links(links), relation

    def _instance_versions(
        self,
        model_ref: str,
        instance: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        owner, separator, model_slug = model_ref.partition("/")
        if not separator or not owner or not model_slug:
            raise ValueError(f"invalid Kaggle model ref for version lookup: {model_ref!r}")
        framework = _optional_text(instance.get("framework")) or "unspecified"
        slug = _optional_text(instance.get("slug"))
        if slug is None:
            raise ValueError(f"Kaggle model {model_ref!r} instance has no slug")
        parts = urlsplit(self.url)
        instance_path = "/".join(
            quote(part, safe="") for part in (owner, model_slug, framework, slug)
        )
        url = f"{parts.scheme}://{parts.netloc}/api/v1/models/{instance_path}/list"
        token: str | None = None
        seen_tokens: set[str] = set()
        versions: list[Mapping[str, Any]] = []
        while True:
            params: dict[str, str | int] = {"pageSize": self.page_size}
            if token is not None:
                params["pageToken"] = token
            response: HttpResponse = self.client.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
            )
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: versions for {model_ref}/{framework}/{slug} "
                    f"returned HTTP {response.status}"
                )
            if len(response.body) > 4 * 1024 * 1024:
                raise ValueError(f"{self.name}: model instance versions response is too large")
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError(f"{self.name}: model instance versions response is not an object")
            version_list = payload.get("versionList", payload.get("version_list"))
            if isinstance(version_list, Mapping):
                page_versions = version_list.get("versions")
            else:
                page_versions = version_list
            if not _is_sequence(page_versions):
                raise ValueError(f"{self.name}: model instance versions are not a list")
            if any(value is not None and not isinstance(value, Mapping) for value in page_versions):
                raise ValueError(f"{self.name}: model instance version row is not an object")
            versions.extend(value for value in page_versions if isinstance(value, Mapping))
            next_token = _optional_text(
                payload.get("nextPageToken", payload.get("next_page_token"))
            )
            if next_token is None:
                break
            if next_token in seen_tokens or next_token == token:
                raise ValueError(f"{self.name}: version pagination token did not advance")
            seen_tokens.add(next_token)
            token = next_token

        # The versions endpoint is scoped to one instance, but retain the SDK's
        # explicit association fields when present as a guard against malformed
        # or unexpectedly mixed pages. ModelInstanceVersion identifies both its
        # owning instance and variation in the first-party schema.
        expected_instance_id = _optional_text(instance.get("id"))
        for version in versions:
            associated_instance_id = _optional_text(
                version.get("modelInstanceId", version.get("model_instance_id"))
            )
            if (
                expected_instance_id is not None
                and associated_instance_id is not None
                and associated_instance_id != expected_instance_id
            ):
                raise ValueError(
                    f"{self.name}: version row belongs to model instance "
                    f"{associated_instance_id}, expected {expected_instance_id}"
                )
            version_slug = _optional_text(
                version.get("variationSlug", version.get("variation_slug"))
            )
            if version_slug is not None and version_slug != slug:
                raise ValueError(
                    f"{self.name}: version row belongs to variation {version_slug!r}, "
                    f"expected {slug!r}"
                )
            version_framework = _optional_text(version.get("framework"))
            if version_framework is not None and version_framework != framework:
                raise ValueError(
                    f"{self.name}: version row belongs to framework "
                    f"{version_framework!r}, expected {framework!r}"
                )
        return tuple(versions)

    def _version_files(
        self,
        model_ref: str,
        framework: str,
        slug: str,
        version: str,
    ) -> tuple[dict[str, Any], ...]:
        """List files for one exact version when explicitly requested."""
        parts = urlsplit(self.url)
        path = "/".join(
            quote(part, safe="/") for part in (model_ref, framework, slug, version)
        )
        url = f"{parts.scheme}://{parts.netloc}/api/v1/models/{path}/files"
        token: str | None = None
        seen_tokens: set[str] = set()
        files: list[dict[str, Any]] = []
        while True:
            params: dict[str, str | int] = {"pageSize": self.page_size}
            if token is not None:
                params["pageToken"] = token
            response: HttpResponse = self.client.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
            )
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: files for {model_ref}/{framework}/{slug}/{version} "
                    f"returned HTTP {response.status}"
                )
            if len(response.body) > 4 * 1024 * 1024:
                raise ValueError(f"{self.name}: model version files response is too large")
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError(f"{self.name}: model version files response is not an object")
            page_files = payload.get("files")
            if not _is_sequence(page_files):
                raise ValueError(f"{self.name}: model version files are not a list")
            for file in page_files:
                if not isinstance(file, Mapping):
                    raise ValueError(f"{self.name}: model version file row is not an object")
                name = _optional_text(file.get("name"))
                if name is None:
                    raise ValueError(f"{self.name}: model version file has no name")
                files.append(
                    {
                        "name": name,
                        "size": _optional_nonnegative_int(file.get("size")),
                        "creation_date": _optional_text(
                            file.get("creationDate", file.get("creation_date"))
                        ),
                    }
                )
            next_token = _optional_text(
                payload.get("nextPageToken", payload.get("next_page_token"))
            )
            if next_token is None:
                break
            if next_token in seen_tokens or next_token == token:
                raise ValueError(f"{self.name}: model version file pagination did not advance")
            seen_tokens.add(next_token)
            token = next_token
        return tuple(files)


def _versions_with_latest(
    instance: Mapping[str, Any],
    listed: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, ...]:
    """Use every listed version and retain the embedded latest if absent."""
    if not listed:
        return (None,)
    versions: list[Mapping[str, Any]] = []
    seen: dict[str, str | None] = {}
    private_numbers: set[str] = set()
    for version in listed:
        number = _optional_text(version.get("versionNumber", version.get("version_number")))
        if version.get("isPrivate") is True or version.get("is_private") is True:
            if number:
                private_numbers.add(number)
            continue
        version_id = _optional_text(
            version.get("versionId")
        ) or _optional_text(version.get("version_id")) or _optional_text(version.get("id"))
        if number is None:
            raise ValueError("Kaggle model instance version has no version number")
        if number in seen and seen[number] != version_id:
            raise ValueError(f"Kaggle instance has conflicting version rows for {number}")
        if number not in seen:
            versions.append(version)
            seen[number] = version_id
    latest_number = _optional_text(instance.get("versionNumber"))
    if latest_number and latest_number not in seen and latest_number not in private_numbers:
        versions.append(instance)
    return tuple(versions)


def _model_url(model_ref: str, value: Any) -> str:
    supplied = _optional_web_url(value, "https://www.kaggle.com")
    if supplied:
        return supplied
    return canonicalize_url(
        f"https://www.kaggle.com/models/{quote(model_ref, safe='/')}"
    )


def _model_identity(value: str) -> tuple[str, tuple[Identifier, ...]]:
    identifier = identifier_from_url(value)
    if identifier is not None:
        return identifier.value, (identifier,)
    parsed = urlsplit(value)
    components = [component for component in parsed.path.split("/") if component]
    if len(components) >= 3 and components[0] == "models":
        model_ref = "/".join(components[1:3])
        return model_ref, (Identifier("kaggle:model", model_ref),)
    return value, ()


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda link: (link.url, link.relation, link.locator or "", link.crawl),
        )
    )


def _web_url(value: str, label: str) -> str:
    result = canonicalize_url(value)
    parsed = urlsplit(result)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
    return result


def _optional_web_url(value: Any, base_url: str) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    result = canonicalize_url(urljoin(base_url, text))
    parsed = urlsplit(result)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return result


def _required_text(value: Any, label: str) -> str:
    result = _optional_text(value)
    if result is None:
        raise ValueError(f"{label} must be non-empty text")
    return result


def _optional_text(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _state_count(state: Mapping[str, Any], key: str) -> int:
    value = state.get(key)
    if value is None:
        return 0
    parsed = _optional_nonnegative_int(value)
    if parsed is None:
        raise ValueError(f"Kaggle model sync state {key} must be a nonnegative integer")
    return parsed


def _state_tokens(state: Mapping[str, Any], key: str) -> set[str]:
    value = state.get(key, ())
    if not _is_sequence(value) or any(not isinstance(token, str) or not token for token in value):
        raise ValueError(f"Kaggle model sync state {key} must be a list of non-empty tokens")
    return set(value)


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


__all__ = ["KaggleModelsSourceAdapter"]
