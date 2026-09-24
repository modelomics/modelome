"""Public OpenCSG Hub model-catalog ingestion.

OpenCSG's public ``/api/v1/models`` listing provides a large independent model
repository inventory with source-native paths, clone locations, tags, licenses,
and declared Hugging Face or ModelScope mirror paths.  The adapter retains those
claims without cloning repositories or downloading any file.
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
from modelome.normalize import canonicalize_url, content_hash, extract_urls


class OpenCsgModelsSourceAdapter:
    """Page every public model repository returned by the OpenCSG Hub API."""

    def __init__(
        self,
        *,
        name: str = "opencsg-models",
        url: str = "https://hub.opencsg.com/api/v1/models",
        page_size: int = 100,
        sort_by: str = "recently_update",
        artifact_kind: str | ArtifactKind = ArtifactKind.MODEL_CARD,
        client: HttpClient | Any | None = None,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, "catalog URL")
        self.page_size = int(page_size)
        if not 1 <= self.page_size <= 100:
            raise ValueError(f"{self.name}: page_size must be from 1 through 100")
        self.sort_by = _required_text(sort_by, "sort_by")
        if self.sort_by not in {
            "trending",
            "recently_update",
            "most_download",
            "most_favorite",
            "most_star",
        }:
            raise ValueError(f"{self.name}: unsupported OpenCSG model sort {self.sort_by!r}")
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.client = client or HttpClient()
        self.checkpoint_signature = content_hash(
            {
                "adapter": "opencsg-models-v2",
                "url": self.url,
                "page_size": self.page_size,
                "sort_by": self.sort_by,
                "artifact_kind": self.artifact_kind.value,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        page_number = _state_page(state)
        raw_items_seen = _state_count(state, "raw_items_seen")
        scan_total = _state_count_or_none(state, "scan_total")
        response: HttpResponse = self.client.get(
            self.url,
            params={"page": page_number, "per": self.page_size, "sort": self.sort_by},
            headers={"Accept": "application/json"},
        )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: expected a JSON object from {response.url}")
        items = payload.get("data")
        if not _is_sequence(items):
            raise ValueError(f"{self.name}: response data must be a list")
        response_total = _optional_nonnegative_int(payload.get("total"))
        if scan_total is not None and response_total is not None and response_total != scan_total:
            raise ValueError(
                f"{self.name}: provider total changed from {scan_total} to {response_total}; "
                "restart the inventory scan"
            )
        scan_total = response_total if response_total is not None else scan_total
        raw_items_seen += len(items)
        if scan_total is not None and raw_items_seen > scan_total:
            raise ValueError(
                f"{self.name}: received {raw_items_seen} models, beyond provider total {scan_total}"
            )

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for index, item in enumerate(items):
            if isinstance(item, Mapping) and item.get("private") is True:
                continue
            try:
                records.append(self._record(item, index))
            except (TypeError, ValueError) as error:
                raw = dict(item) if isinstance(item, Mapping) else {"value": repr(item)[:1000]}
                record_id = _optional_text(item.get("path")) if isinstance(item, Mapping) else None
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

        if scan_total is not None:
            complete = raw_items_seen == scan_total
            if not complete and not items:
                raise ValueError(
                    f"{self.name}: empty page {page_number} before provider total {scan_total}"
                )
        else:
            complete = len(items) < self.page_size
        next_state: dict[str, Any] = {}
        if not complete:
            next_state = {
                "page": page_number + 1,
                "raw_items_seen": raw_items_seen,
            }
            if scan_total is not None:
                next_state["scan_total"] = scan_total
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=scan_total,
            issues=tuple(issues),
        )

    def _record(self, item: Any, index: int) -> SourceRecord:
        if not isinstance(item, Mapping):
            raise ValueError(f"catalog item {index} is not an object")
        model_path = _required_text(item.get("path"), f"catalog item {index} path")
        if "/" not in model_path:
            raise ValueError(f"catalog item {index} path must contain namespace and name")
        model_url = _model_url(model_path, item.get("url"))
        title = (
            _optional_text(item.get("nickname"))
            or _optional_text(item.get("name"))
            or model_path
        )
        model_local_id = f"{model_path}#model"
        model_identifier = Identifier("opencsg:model", model_path)
        model = ModelHint(
            local_id=model_local_id,
            name=title,
            identifiers=(model_identifier,),
            aliases=tuple(
                value
                for value in (_optional_text(item.get("name")), model_path)
                if value and value != title
            ),
            status=ModelStatus.RELEASED,
            locator="$.path",
        )
        tags = _tag_labels(item.get("tags"))
        description = _optional_text(item.get("description"))
        readme = _optional_text(item.get("readme"))
        text_parts = [part for part in (description, readme) if part]
        if tags:
            text_parts.append("tags: " + ", ".join(sorted(set(tags))))
        if license_name := _optional_text(item.get("license")):
            text_parts.append(f"license: {license_name}")
        if metadata_text := _model_metadata_text(item.get("metadata")):
            text_parts.append(metadata_text)

        links: list[Link] = [Link(model_url, relation="model_page", locator="$.path")]
        for field, value in (("description", description), ("readme", readme)):
            if value:
                links.extend(
                    Link(url, relation="documentation_reference", locator=f"$.{field}")
                    for url in extract_urls(value)
                )
        clone_url = _clone_url(item.get("repository"), model_url)
        if clone_url:
            links.append(
                Link(
                    clone_url,
                    relation="model_repository",
                    locator="$.repository.http_clone_url",
                    crawl=False,
                )
            )

        identifiers = [model_identifier]
        repository_id = _optional_text(item.get("repository_id"))
        if repository_id:
            identifiers.append(Identifier("opencsg:repository", repository_id))
        relations: list[ModelRelationHint] = []
        for field, namespace, base_url in (
            ("hf_path", "huggingface:model", "https://huggingface.co"),
            ("ms_path", "modelscope:model", "https://modelscope.cn/models"),
        ):
            external_path = _optional_text(item.get(field))
            if not external_path:
                continue
            external_url = canonicalize_url(
                f"{base_url.rstrip('/')}/{quote(external_path, safe='/')}"
            )
            links.append(Link(external_url, relation="model_mirror", locator=f"$.{field}"))
            relations.append(
                ModelRelationHint(
                    subject_local_id=model_local_id,
                    predicate="mirrors",
                    target=ModelHint(
                        local_id=f"{model_path}#{field}",
                        name=external_path,
                        identifiers=(Identifier(namespace, external_path),),
                        status=ModelStatus.DOCUMENTED,
                        locator=f"$.{field}",
                    ),
                    locator=f"$.{field}",
                )
            )

        revision = _optional_text(item.get("revision"))
        releases = ()
        if revision:
            releases = (
                ReleaseHint(
                    local_id=f"{model_path}#revision:{revision}",
                    model_local_id=model_local_id,
                    revision=revision,
                    identifiers=(
                        Identifier("opencsg:revision", f"{model_path}@{revision}"),
                    ),
                    locator="$.revision",
                ),
            )

        return SourceRecord(
            source_record_id=model_path,
            kind=self.artifact_kind,
            canonical_url=model_url,
            title=title,
            raw=dict(item),
            text="\n".join(text_parts),
            published_at=_optional_text(item.get("created_at")),
            modified_at=_optional_text(item.get("updated_at")),
            identifiers=tuple(identifiers),
            links=_unique_links(links),
            models=(model,),
            model_relations=tuple(relations),
            releases=releases,
        )


def _model_url(model_path: str, value: Any) -> str:
    supplied = _optional_web_url(value, "https://opencsg.com")
    if supplied:
        return supplied
    return canonicalize_url(
        f"https://opencsg.com/models/{quote(model_path, safe='/')}"
    )


def _clone_url(value: Any, base_url: str) -> str | None:
    if not isinstance(value, Mapping):
        return None
    return _optional_web_url(value.get("http_clone_url"), base_url)


def _tag_labels(value: Any) -> tuple[str, ...]:
    labels: list[str] = []
    for item in _sequence(value):
        if isinstance(item, Mapping):
            label = _optional_text(item.get("show_name")) or _optional_text(item.get("name"))
        else:
            label = _optional_text(item)
        if label:
            labels.append(label)
    return tuple(labels)


def _model_metadata_text(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    fields = (
        ("model_params", "parameters"),
        ("architecture", "architecture"),
        ("tensor_type", "tensor type"),
        ("model_type", "model type"),
        ("mini_gpu_memory_gb", "minimum GPU memory GB"),
        ("mini_gpu_finetune_gb", "minimum GPU finetune memory GB"),
    )
    parts = [
        f"{label}: {text}"
        for key, label in fields
        if (text := _optional_text(value.get(key))) is not None
    ]
    return "model metadata: " + "; ".join(parts) if parts else None


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


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _state_page(state: Mapping[str, Any]) -> int:
    value = state.get("page")
    if value is None:
        return 1
    parsed = _optional_nonnegative_int(value)
    if parsed is None or parsed < 1:
        raise ValueError("OpenCSG model sync state page must be a positive integer")
    return parsed


def _state_count(state: Mapping[str, Any], key: str) -> int:
    value = state.get(key)
    if value is None:
        return 0
    parsed = _optional_nonnegative_int(value)
    if parsed is None:
        raise ValueError(f"OpenCSG model sync state {key} must be a nonnegative integer")
    return parsed


def _state_count_or_none(state: Mapping[str, Any], key: str) -> int | None:
    if state.get(key) is None:
        return None
    return _state_count(state, key)


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


__all__ = ["OpenCsgModelsSourceAdapter"]
