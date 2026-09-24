"""Public OpenRouter model-catalog ingestion.

OpenRouter's ``GET /api/v1/models`` endpoint returns the complete current catalog
when requested with ``output_modalities=all``.  Without that parameter, the
provider defaults to text-output models only. The catalog is valuable because it
records publicly routable closed and open models which do not necessarily have a
downloadable model artifact.  It is intentionally represented as provider-page
evidence, not as a claim that OpenRouter owns the underlying model or weights.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
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
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash, extract_urls


class OpenRouterModelsSourceAdapter:
    """Capture each public model currently listed by OpenRouter exactly once.

    The adapter can request all modalities instead of the endpoint's text-only
    default. The response's ``total_count`` plus null ``links.next`` make an
    accidental partial response a hard failure. A catalog observation can
    disappear on a later run, so the response is an authoritative snapshot of
    *OpenRouter availability*, not a tombstone for the underlying model
    everywhere.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers model IDs publicly available through OpenRouter in one observed "
        "catalog response. The API defaults to text-output models; configure "
        "output_modalities='all' to include image, audio, and embedding models. "
        "It records current routing availability, not a complete historical "
        "catalog, model ownership, or downloadable weights. Removed, private, "
        "and never-routed models are outside this source."
    )

    def __init__(
        self,
        *,
        name: str = "openrouter-models",
        url: str = "https://openrouter.ai/api/v1/models",
        model_page_base_url: str = "https://openrouter.ai",
        output_modalities: str | None = None,
        max_response_bytes: int = 16 * 1024 * 1024,
        client: HttpClient | Any | None = None,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name, "catalog URL")
        self.model_page_base_url = _web_url(
            model_page_base_url,
            self.name,
            "model-page base URL",
        ).rstrip("/")
        self.output_modalities = _optional_modalities(output_modalities, self.name)
        self.max_response_bytes = _positive_int(
            max_response_bytes,
            "max_response_bytes",
            self.name,
        )
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openrouter-models-v1",
                "url": self.url,
                "model_page_base_url": self.model_page_base_url,
                "output_modalities": self.output_modalities,
                "max_response_bytes": self.max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.url,
            params=(
                {"output_modalities": self.output_modalities}
                if self.output_modalities is not None
                else None
            ),
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: catalog exceeds {self.max_response_bytes} bytes"
            )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: catalog response is not a JSON object")
        items = payload.get("data")
        if not _is_sequence(items):
            raise ValueError(f"{self.name}: catalog data is not a JSON list")
        total_count = _nonnegative_int(payload.get("total_count"), "total_count", self.name)
        if len(items) != total_count:
            raise ValueError(
                f"{self.name}: catalog returned {len(items)} item(s), "
                f"but reported {total_count}"
            )
        links = payload.get("links")
        if not isinstance(links, Mapping):
            raise ValueError(f"{self.name}: catalog lacks a pagination links object")
        if _text(links.get("next")):
            raise ValueError(
                f"{self.name}: catalog unexpectedly returned a next page"
            )

        records = tuple(self._record(item, index) for index, item in enumerate(items))
        record_ids = [record.source_record_id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError(f"{self.name}: catalog contains duplicate model IDs")

        return SourcePage(
            records=records,
            next_state={
                "catalog_sha256": content_hash(response.body),
                "model_count": total_count,
            },
            complete=True,
            upstream_count=total_count,
            authoritative_snapshot=self.output_modalities in {None, "all"},
        )

    def _record(self, item: Any, index: int) -> SourceRecord:
        if not isinstance(item, Mapping):
            raise ValueError(f"{self.name}: catalog item {index} is not an object")
        model_id = _required_text(item.get("id"), f"catalog item {index} ID")
        title = _required_text(item.get("name"), f"catalog item {index} name")
        model_identifier = Identifier("openrouter:model", model_id)
        model_local_id = f"{model_id}#model"
        model = ModelHint(
            local_id=model_local_id,
            name=title,
            identifiers=(model_identifier,),
            aliases=_aliases(item, model_id, title),
            status=ModelStatus.RELEASED,
            locator="$.id",
        )
        description = _text(item.get("description"))
        architecture = item.get("architecture")
        if architecture is not None and not isinstance(architecture, Mapping):
            raise ValueError(
                f"{self.name}: catalog item {model_id!r} architecture is not an object"
            )
        architecture = architecture or {}

        model_page = self._model_page_url(model_id)
        links: list[Link] = [
            Link(model_page, relation="model_page", locator="$.id", crawl=False),
        ]
        details = _optional_web_url(
            architecture_value(item, "links", "details"),
            self.url,
        )
        if details:
            links.append(
                Link(
                    details,
                    relation="provider_endpoints",
                    locator="$.links.details",
                    crawl=False,
                )
            )
        if description:
            links.extend(
                Link(url, relation="documentation_reference", locator="$.description")
                for url in extract_urls(description)
            )

        relations: list[ModelRelationHint] = []
        hugging_face_id = _text(item.get("hugging_face_id"))
        if hugging_face_id:
            hugging_face_url = _hugging_face_model_url(hugging_face_id, self.name)
            links.append(
                Link(
                    hugging_face_url,
                    relation="linked_model_artifact",
                    locator="$.hugging_face_id",
                    crawl=False,
                )
            )
            relations.append(
                ModelRelationHint(
                    subject_local_id=model_local_id,
                    predicate="hosted_huggingface_model",
                    target=ModelHint(
                        local_id=f"{model_local_id}#huggingface",
                        name=hugging_face_id,
                        identifiers=(Identifier("huggingface:model", hugging_face_id),),
                        status=ModelStatus.DOCUMENTED,
                        locator="$.hugging_face_id",
                    ),
                    locator="$.hugging_face_id",
                )
            )
        alias_target = item.get("alias_target")
        if alias_target is not None:
            if not isinstance(alias_target, Mapping):
                raise ValueError(
                    f"{self.name}: catalog item {model_id!r} alias target is not an object"
                )
            alias_slug = _required_text(
                alias_target.get("slug"),
                f"catalog item {model_id} alias target slug",
            )
            if alias_slug != model_id:
                alias_name = _text(alias_target.get("name")) or alias_slug
                relations.append(
                    ModelRelationHint(
                        subject_local_id=model_local_id,
                        predicate="alias_of",
                        target=ModelHint(
                            local_id=f"{model_local_id}#alias-target",
                            name=alias_name,
                            identifiers=(Identifier("openrouter:model", alias_slug),),
                            status=ModelStatus.RELEASED,
                            locator="$.alias_target.slug",
                        ),
                        locator="$.alias_target.slug",
                    )
                )

        text = _record_text(description, architecture, item)
        return SourceRecord(
            source_record_id=model_id,
            kind=ArtifactKind.PROVIDER_PAGE,
            canonical_url=model_page,
            title=title,
            raw=dict(item),
            text=text,
            published_at=_timestamp(item.get("created")),
            identifiers=(model_identifier,),
            links=_unique_links(links),
            models=(model,),
            model_relations=tuple(relations),
        )

    def _model_page_url(self, model_id: str) -> str:
        return canonicalize_url(
            f"{self.model_page_base_url}/{quote(model_id, safe='/~:._-')}"
        )


class OpenRouterVideoModelsSourceAdapter:
    """Capture OpenRouter's complete current video-generation model list.

    The first-party ``GET /api/v1/videos/models`` contract says its ``data``
    array contains all available video-generation models (unlike the broader
    models endpoint, whose documented modality filter covers text, image,
    audio, and embeddings). This separate source requires ``OPENROUTER_API_KEY``
    because the video API documents Bearer authentication. The endpoint has no
    pagination or reported total, so completeness relies on its documented
    single-list response contract; byte, row-count, and schema limits fail
    closed. This is a snapshot of current OpenRouter routing availability only.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Captures the current complete list returned by OpenRouter's video-model "
        "catalog endpoint. It is a snapshot of video-generation routing "
        "availability, not a historical catalog or a census of models outside "
        "OpenRouter. The first-party endpoint has no cursor or reported total."
    )

    def __init__(
        self,
        *,
        name: str = "openrouter-video-models",
        url: str = "https://openrouter.ai/api/v1/videos/models",
        model_page_base_url: str = "https://openrouter.ai",
        token: str | None = None,
        max_response_bytes: int = 16 * 1024 * 1024,
        max_models: int = 100_000,
        client: HttpClient | Any | None = None,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name, "catalog URL")
        self.model_page_base_url = _web_url(
            model_page_base_url,
            self.name,
            "model-page base URL",
        ).rstrip("/")
        self._token = _required_text(
            token if token is not None else os.environ.get("OPENROUTER_API_KEY", ""),
            "OPENROUTER_API_KEY",
        )
        self.max_response_bytes = _positive_int(
            max_response_bytes,
            "max_response_bytes",
            self.name,
        )
        self.max_models = _positive_int(max_models, "max_models", self.name)
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openrouter-video-models-v1",
                "url": self.url,
                "model_page_base_url": self.model_page_base_url,
                "max_response_bytes": self.max_response_bytes,
                "max_models": self.max_models,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if state:
            raise ValueError(f"{self.name}: this catalog does not accept pagination state")
        try:
            response: HttpResponse = self.client.get(
                self.url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._token}",
                },
            )
        except Exception as error:
            if self._token in str(error):
                raise RuntimeError(f"{self.name}: catalog request failed") from None
            raise
        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: catalog exceeds {self.max_response_bytes} bytes"
            )
        payload = response.json()
        if not isinstance(payload, Mapping) or not _is_sequence(payload.get("data")):
            raise ValueError(f"{self.name}: catalog data is not a JSON list")
        items = payload["data"]
        if len(items) > self.max_models:
            raise ValueError(f"{self.name}: catalog exceeds {self.max_models} models")
        # The documented endpoint is unpaginated. If a future response begins
        # advertising pagination, do not silently treat the first page as all.
        if "links" in payload:
            links = payload["links"]
            if not isinstance(links, Mapping):
                raise ValueError(f"{self.name}: catalog pagination links are malformed")
            if links.get("next") not in (None, ""):
                raise ValueError(f"{self.name}: catalog unexpectedly returned a next page")
        if any(
            payload.get(key) not in (None, "")
            for key in ("next", "next_url", "nextCursor")
        ):
            raise ValueError(f"{self.name}: catalog unexpectedly returned a next page")

        records = tuple(self._record(item, index) for index, item in enumerate(items))
        record_ids = [record.source_record_id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError(f"{self.name}: catalog contains duplicate model IDs")
        return SourcePage(
            records=records,
            next_state={
                "catalog_sha256": content_hash(response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, item: Any, index: int) -> SourceRecord:
        if not isinstance(item, Mapping):
            raise ValueError(f"{self.name}: catalog item {index} is not an object")
        model_id = _required_text(item.get("id"), f"catalog item {index} ID")
        title = _required_text(item.get("name"), f"catalog item {index} name")
        identifier = Identifier("openrouter:model", model_id)
        model_url = canonicalize_url(
            f"{self.model_page_base_url}/{quote(model_id, safe='/~:._-')}"
        )
        model_local_id = f"{model_id}#model"
        description = _text(item.get("description"))
        text_parts = [description] if description else []
        for key, label in (
            ("supported_durations", "supported durations"),
            ("supported_resolutions", "supported resolutions"),
            ("supported_aspect_ratios", "supported aspect ratios"),
        ):
            values = _text_values(item.get(key))
            if values:
                text_parts.append(f"{label}: {', '.join(values)}")
        if item.get("generate_audio") is True:
            text_parts.append("generates audio: yes")
        return SourceRecord(
            source_record_id=model_id,
            kind=ArtifactKind.PROVIDER_PAGE,
            canonical_url=model_url,
            title=title,
            raw=dict(item),
            text="\n".join(text_parts),
            published_at=_timestamp(item.get("created")),
            identifiers=(identifier,),
            links=(Link(model_url, relation="model_page", locator="$.id", crawl=False),),
            models=(
                ModelHint(
                    local_id=model_local_id,
                    name=title,
                    identifiers=(identifier,),
                    aliases=_aliases(item, model_id, title),
                    status=ModelStatus.RELEASED,
                    locator="$.id",
                ),
            ),
        )


def architecture_value(item: Mapping[str, Any], parent: str, child: str) -> Any:
    value = item.get(parent)
    return value.get(child) if isinstance(value, Mapping) else None


def _record_text(
    description: str,
    architecture: Mapping[str, Any],
    item: Mapping[str, Any],
) -> str:
    parts = [description] if description else []
    if modality := _text(architecture.get("modality")):
        parts.append(f"modality: {modality}")
    for key, label in (
        ("input_modalities", "input modalities"),
        ("output_modalities", "output modalities"),
        ("supported_parameters", "supported parameters"),
    ):
        values = _text_values(architecture.get(key) if key in architecture else item.get(key))
        if values:
            parts.append(f"{label}: {', '.join(values)}")
    if canonical_slug := _text(item.get("canonical_slug")):
        parts.append(f"OpenRouter canonical slug: {canonical_slug}")
    if context_length := _positive_optional_int(item.get("context_length")):
        parts.append(f"context length: {context_length}")
    pricing = item.get("pricing")
    if isinstance(pricing, Mapping):
        prices = tuple(
            f"{key}: {_text(value)}"
            for key in ("prompt", "completion", "image", "request")
            if (value := _text(pricing.get(key)))
        )
        if prices:
            parts.append("pricing: " + ", ".join(prices))
    for key, label in (("tokenizer", "tokenizer"), ("instruct_type", "instruction format")):
        value = _text(architecture.get(key))
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def _positive_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and str(number) == str(value).strip() else None


def _optional_modalities(value: Any, source: str) -> str | None:
    if value is None:
        return None
    modalities = ",".join(part.strip() for part in _text(value).casefold().split(","))
    allowed = {"all", "text", "image", "audio", "embeddings"}
    parts = modalities.split(",")
    if not modalities or (
        modalities != "all" and any(part not in allowed - {"all"} for part in parts)
    ):
        raise ValueError(
            f"{source}: output_modalities must be 'all' or a comma-separated "
            "list of text, image, audio, or embeddings"
        )
    return modalities


def _aliases(item: Mapping[str, Any], model_id: str, title: str) -> tuple[str, ...]:
    candidates = (item.get("canonical_slug"), model_id)
    return tuple(
        value
        for value in dict.fromkeys(_text(candidate) for candidate in candidates)
        if value and value != title
    )


def _timestamp(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return _text(value)
    if seconds < 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, UTC).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return _text(value)


def _hugging_face_model_url(value: str, source: str) -> str:
    if any(character in value for character in "\r\n\x00"):
        raise ValueError(f"{source}: hugging_face_id contains control characters")
    parts = value.split("/")
    if len(parts) != 2 or not all(_text(part) for part in parts):
        raise ValueError(f"{source}: hugging_face_id is not an owner/model identifier")
    return canonicalize_url(f"https://huggingface.co/{quote(value, safe='/._-')}")


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    result = []
    seen = set()
    for link in values:
        key = (link.url, link.relation, link.locator, link.crawl)
        if key not in seen:
            seen.add(key)
            result.append(link)
    return tuple(result)


def _optional_web_url(value: Any, base_url: str) -> str | None:
    text = _text(value)
    if not text:
        return None
    candidate = canonicalize_url(urljoin(base_url, text))
    parts = urlsplit(candidate)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("OpenRouter model link is not an HTTP(S) URL")
    return candidate


def _web_url(value: str, source: str, label: str) -> str:
    result = canonicalize_url(_required_text(value, label))
    parts = urlsplit(result)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"{source}: {label} is not an HTTP(S) URL")
    return result


def _required_text(value: Any, label: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{label} is required")
    if any(character in result for character in "\r\n\x00"):
        raise ValueError(f"{label} contains control characters")
    return result


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _text_values(value: Any) -> tuple[str, ...]:
    if not _is_sequence(value):
        return ()
    return tuple(item for item in (_text(raw) for raw in value) if item)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str))


def _nonnegative_int(value: Any, label: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {label} must be a non-negative integer") from error
    if result < 0 or str(result) != str(value).strip():
        raise ValueError(f"{source}: {label} must be a non-negative integer")
    return result


def _positive_int(value: Any, label: str, source: str) -> int:
    result = _nonnegative_int(value, label, source)
    if result < 1:
        raise ValueError(f"{source}: {label} must be positive")
    return result
