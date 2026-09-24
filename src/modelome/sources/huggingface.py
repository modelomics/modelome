from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
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

Clock = Callable[[], datetime]
_LINK_RE = re.compile(r'<([^>]+)>\s*((?:;\s*[^,]+)*)')
_REL_RE = re.compile(r'\brel\s*=\s*(?:"([^"]+)"|([^;\s,]+))', re.IGNORECASE)
_WEIGHT_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".flax",
    ".gguf",
    ".ggml",
    ".h5",
    ".mlmodel",
    ".mlpackage",
    ".msgpack",
    ".onnx",
    ".pb",
    ".pt",
    ".pt2",
    ".pth",
    ".safetensors",
    ".tflite",
    ".torchscript",
)


def _is_weight_file(filename: str, siblings: Sequence[str] = ()) -> bool:
    folded = filename.casefold()
    # Transformers commonly publish sharded checkpoints with a JSON index.
    # Keep recognition specific to known weight-index names so arbitrary JSON
    # metadata does not become a binary artifact reference.
    if folded.endswith(_WEIGHT_SUFFIXES) or folded.endswith(
        (".safetensors.index.json", ".bin.index.json")
    ):
        return True

    # TensorFlow checkpoints consist of a shared-prefix `.index` file and one
    # or more `.data-00000-of-00001` shards. Recognize only paired components
    # so unrelated repository index/data files remain ordinary files.
    folded_siblings = tuple(value.casefold() for value in siblings)
    if folded.endswith(".index"):
        prefix = folded[: -len(".index")]
        return any(
            re.fullmatch(re.escape(prefix) + r"\.data-\d+-of-\d+", value)
            for value in folded_siblings
        )
    match = re.fullmatch(r"(.+)\.data-\d+-of-\d+", folded)
    if match:
        return f"{match.group(1)}.index" in folded_siblings
    return False


def _utcnow() -> datetime:
    return datetime.now(UTC)


class HuggingFaceSourceAdapter:
    """Enumerate public Hugging Face model repositories.

    A first run follows every RFC 8288 ``Link`` cursor. Later runs walk the
    newest-first listing until reaching the previous ``lastModified`` watermark
    minus a configurable overlap. Cursor state also freezes that boundary so an
    interrupted run resumes the same scan.
    """

    # The Hub cursor is provider-issued and page-level checkpoints remain exact;
    # grouping eight already-fetched pages only amortizes local snapshot writes.
    commit_pages = 8

    def __init__(
        self,
        *,
        name: str = "huggingface",
        url: str = "https://huggingface.co/api/models",
        artifact_kind: str | ArtifactKind = ArtifactKind.MODEL_CARD,
        page_size: int = 100,
        overlap_days: int = 2,
        max_response_bytes: int = 16 * 1024 * 1024,
        token: str | None = None,
        include_private: bool = False,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = name
        self.url = url
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.page_size = int(page_size)
        self.overlap_days = int(overlap_days)
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.token = token
        self.include_private = bool(include_private)
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "huggingface-v1",
                "url": self.url,
                "artifact_kind": self.artifact_kind.value,
                "page_size": self.page_size,
                "overlap_days": self.overlap_days,
                "max_response_bytes": self.max_response_bytes,
                "include_private": self.include_private,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        next_url = _text(state.get("next_url"))
        resuming_scan = bool(next_url)
        raw_items_seen = _state_count(state, "raw_items_seen") if resuming_scan else 0
        scan_total = _state_count(state, "scan_total") if resuming_scan else None
        count_is_complete = not resuming_scan or (
            "raw_items_seen" in state and state.get("raw_count_incomplete") is not True
        )
        prior_watermark = _parse_timestamp(state.get("watermark"))
        if next_url:
            next_url = self._safe_next_url(next_url, self.url)
            cutoff = _parse_timestamp(state.get("cutoff"))
            scan_high = _parse_timestamp(state.get("scan_high_watermark"))
            response: HttpResponse = self.client.get(next_url, headers=headers)
        else:
            cutoff = (
                prior_watermark - timedelta(days=self.overlap_days)
                if prior_watermark is not None
                else None
            )
            scan_high = None
            response = self.client.get(
                self.url,
                params={
                    "limit": self.page_size,
                    "full": "true",
                    "cardData": "true",
                    # The Hub API excludes model configuration from `full`;
                    # it is a separate opt-in (`config=true` in the REST API).
                    "config": "true",
                    "sort": "lastModified",
                    "direction": -1,
                },
                headers=headers,
            )

        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: catalog response exceeds {self.max_response_bytes} bytes"
            )
        payload = response.json()
        if isinstance(payload, Mapping):
            items = payload.get("items") or payload.get("models") or []
        else:
            items = payload
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
            raise ValueError(f"{self.name}: expected a JSON array from {response.url}")
        raw_items_seen = (raw_items_seen or 0) + len(items)
        response_total = _integer_header(response.headers, "x-total-count")
        if response_total is not None and count_is_complete:
            scan_total = max(scan_total or 0, response_total)

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        reached_cutoff = False
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                issues.append(
                    SourceIssue(
                        source_record_id=f"{self.name}:item:{index}",
                        stage="source_normalize",
                        error="TypeError: model result is not a JSON object",
                        summary={"index": index, "value": repr(item)[:1000]},
                    )
                )
                continue
            # An authenticated list can contain repositories visible only to the
            # token. Never persist even their identifiers unless the source was
            # explicitly configured as a private registry.
            if item.get("private") is True and not self.include_private:
                continue
            modified = _parse_timestamp(item.get("lastModified") or item.get("last_modified"))
            if modified is not None and (scan_high is None or modified > scan_high):
                scan_high = modified
            if cutoff is not None and modified is not None and modified < cutoff:
                reached_cutoff = True
                break
            try:
                records.append(self._record(item))
            except (KeyError, TypeError, ValueError) as error:
                raw = dict(item)
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            _text(item.get("id") or item.get("modelId"))
                            or f"{self.name}:malformed:{content_hash(raw)[:32]}"
                        ),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"index": index, "raw": raw},
                    )
                )

        raw_link_next = _link_relation(_header(response.headers, "link"), "next")
        link_next = (
            self._safe_next_url(raw_link_next, response.url or self.url)
            if raw_link_next
            else None
        )
        if (
            not reached_cutoff
            and not link_next
            and scan_total is not None
            and raw_items_seen < scan_total
        ):
            raise ValueError(
                f"{self.name}: pagination ended after {raw_items_seen} raw item(s), "
                f"before the known total of {scan_total}"
            )
        complete = reached_cutoff or not link_next
        now = _isoformat(self.clock())
        if complete:
            watermark = scan_high or prior_watermark or _parse_timestamp(now)
            next_state: dict[str, Any] = {"completed_at": now}
            if watermark is not None:
                next_state["watermark"] = _isoformat(watermark)
        else:
            next_state = {
                "next_url": link_next,
                "started_at": _text(state.get("started_at")) or now,
                "raw_items_seen": raw_items_seen,
            }
            if scan_total is not None:
                next_state["scan_total"] = scan_total
            if not count_is_complete:
                next_state["raw_count_incomplete"] = True
            if prior_watermark is not None:
                next_state["watermark"] = _isoformat(prior_watermark)
            if cutoff is not None:
                next_state["cutoff"] = _isoformat(cutoff)
            if scan_high is not None:
                next_state["scan_high_watermark"] = _isoformat(scan_high)

        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=response_total if response_total is not None else scan_total,
            issues=tuple(issues),
        )

    def _safe_next_url(self, value: str, base_url: str) -> str:
        candidate = canonicalize_url(urljoin(base_url, value))
        if _origin(candidate) != _origin(self.url):
            raise ValueError(f"{self.name}: pagination URL changed origin")
        return candidate

    def _record(self, item: Mapping[str, Any]) -> SourceRecord:
        repo_id = _text(item.get("id")) or _text(item.get("modelId"))
        if not repo_id:
            raise ValueError(f"{self.name}: model result is missing id")
        encoded_id = quote(repo_id, safe="/")
        model_url = canonicalize_url(f"https://huggingface.co/{encoded_id}")
        api_url = canonicalize_url(f"https://huggingface.co/api/models/{encoded_id}")

        identifiers: list[Identifier] = [Identifier("huggingface:model", repo_id)]
        links: list[Link] = [
            Link(model_url, relation="model_page", locator="$.id"),
            Link(api_url, relation="metadata", locator="$.id"),
        ]

        card_data = item.get("cardData") or item.get("card_data")
        config = item.get("config")
        for locator, value in (("$.cardData", card_data), ("$.config", config)):
            for url in extract_urls(value):
                links.append(
                    Link(
                        url,
                        relation=_declared_reference_relation(url),
                        locator=locator,
                    )
                )
                if identifier := identifier_from_url(url):
                    identifiers.append(identifier)

        commit_sha = _text(item.get("sha"))
        revision = commit_sha or "main"
        sibling_filenames = [
            filename
            for sibling in _sequence(item.get("siblings"))
            if isinstance(sibling, Mapping)
            if (filename := _text(sibling.get("rfilename") or sibling.get("path")))
        ]
        for sibling in _sequence(item.get("siblings")):
            if not isinstance(sibling, Mapping):
                continue
            filename = _text(sibling.get("rfilename")) or _text(sibling.get("path"))
            if not filename:
                continue
            if filename.casefold() == "readme.md":
                links.append(
                    Link(
                        canonicalize_url(
                            f"https://huggingface.co/{encoded_id}/raw/"
                            f"{quote(revision, safe='')}/{quote(filename, safe='/')}"
                        ),
                        relation="model_card_source",
                        locator="$.siblings",
                    )
                )
            elif filename.casefold() == "config.json":
                links.append(
                    Link(
                        canonicalize_url(
                            f"https://huggingface.co/{encoded_id}/raw/"
                            f"{quote(revision, safe='')}/{quote(filename, safe='/')}"
                        ),
                        relation="model_config",
                        locator="$.siblings",
                    )
                )
            elif _is_weight_file(filename, sibling_filenames):
                # The list response is the source-declared artifact inventory.
                # Keep a revision-pinned reference to each checkpoint, but do
                # not enqueue or download binary bytes during model discovery.
                links.append(
                    Link(
                        canonicalize_url(
                            f"https://huggingface.co/{encoded_id}/resolve/"
                            f"{quote(revision, safe='')}/{quote(filename, safe='/')}"
                        ),
                        relation="weights",
                        locator="$.siblings",
                        crawl=False,
                    )
                )

        tags = tuple(_text(value) for value in _sequence(item.get("tags")) if _text(value))
        for tag in tags:
            if identifier := _identifier_from_tag(tag):
                identifiers.append(identifier)

        aliases = tuple(
            value
            for value in (_text(item.get("modelId")), _text(item.get("id")))
            if value and value != repo_id
        )
        model = ModelHint(
            local_id=f"{repo_id}#model",
            name=repo_id,
            identifiers=(Identifier("huggingface:model", repo_id),),
            aliases=aliases,
            status=ModelStatus.RELEASED,
            locator="$.id",
        )
        relations = tuple(self._base_model_relations(item, card_data, model.local_id, repo_id))
        release_identifiers = (
            (Identifier("huggingface:revision", f"{repo_id}@{commit_sha}"),)
            if commit_sha
            else ()
        )
        release = ReleaseHint(
            local_id=f"{repo_id}#release:{revision}",
            model_local_id=model.local_id,
            revision=revision,
            identifiers=release_identifiers,
            released_at=_text(item.get("lastModified") or item.get("last_modified")) or None,
            metadata={
                "library_name": _text(item.get("library_name")) or None,
                "pipeline_tag": _text(item.get("pipeline_tag")) or None,
                "weight_files": sorted(
                    filename
                    for filename in sibling_filenames
                    if _is_weight_file(filename, sibling_filenames)
                ),
            },
            locator="$.sha" if commit_sha else "$.id",
        )

        text_parts = []
        if pipeline := _text(item.get("pipeline_tag")):
            text_parts.append(f"pipeline: {pipeline}")
        if tags:
            text_parts.append("tags: " + ", ".join(tags))
        if card_data is not None:
            text_parts.append(json.dumps(card_data, ensure_ascii=False, sort_keys=True))

        return SourceRecord(
            source_record_id=repo_id,
            kind=self.artifact_kind,
            canonical_url=model_url,
            title=repo_id,
            raw=dict(item),
            text="\n".join(text_parts),
            published_at=_text(item.get("createdAt") or item.get("created_at")) or None,
            modified_at=_text(item.get("lastModified") or item.get("last_modified")) or None,
            identifiers=_unique_identifiers(identifiers),
            links=_unique_links(links),
            models=(model,),
            model_relations=relations,
            releases=(release,),
        )

    def _base_model_relations(
        self,
        item: Mapping[str, Any],
        card_data: Any,
        subject_local_id: str,
        repo_id: str,
    ) -> Iterable[ModelRelationHint]:
        candidates: list[tuple[str, str]] = []
        candidates.extend(_named_values(item.get("baseModels"), "$.baseModels"))
        candidates.extend(_named_values(item.get("base_models"), "$.base_models"))
        if isinstance(card_data, Mapping):
            candidates.extend(_named_values(card_data.get("base_model"), "$.cardData.base_model"))
            candidates.extend(_named_values(card_data.get("base_models"), "$.cardData.base_models"))
        for tag in _sequence(item.get("tags")):
            tag_text = _text(tag)
            if tag_text.casefold().startswith("base_model:"):
                candidates.append((tag_text.rsplit(":", maxsplit=1)[-1].strip(), "$.tags"))

        seen: set[str] = set()
        for index, (candidate, locator) in enumerate(candidates):
            name, identifiers = _model_identity(candidate)
            if not name or name == repo_id or name in seen:
                continue
            seen.add(name)
            target = ModelHint(
                local_id=f"{repo_id}#base-model-{index}",
                name=name,
                identifiers=identifiers,
                status=ModelStatus.DOCUMENTED,
                locator=locator,
            )
            yield ModelRelationHint(
                subject_local_id=subject_local_id,
                predicate="base_model",
                target=target,
                locator=locator,
            )


def _named_values(value: Any, locator: str) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(value.strip(), locator)] if value.strip() else []
    if isinstance(value, Mapping):
        candidate = next(
            (_text(value.get(key)) for key in ("id", "name", "model") if _text(value.get(key))),
            "",
        )
        return [(candidate, locator)] if candidate else []
    return [
        pair
        for item in _sequence(value)
        for pair in _named_values(item, locator)
    ]


def _model_identity(value: str) -> tuple[str, tuple[Identifier, ...]]:
    value = value.strip()
    if not value:
        return "", ()
    if (
        (identifier := identifier_from_url(value))
        and identifier.namespace == "huggingface:model"
    ):
        return identifier.value, (identifier,)
    if "/" in value and not any(character.isspace() for character in value):
        return value, (Identifier("huggingface:model", value),)
    return value, ()


def _identifier_from_tag(tag: str) -> Identifier | None:
    prefix, separator, value = tag.partition(":")
    if not separator or not value.strip():
        return None
    namespace = prefix.casefold().strip()
    if namespace == "arxiv":
        return Identifier("arxiv", re.sub(r"v\d+$", "", value.strip()))
    if namespace == "doi":
        return Identifier("doi", value.strip().casefold())
    return None


def _declared_reference_relation(url: str) -> str:
    """Type a direct Hub-card URL only when its canonical URL proves the type."""

    identifier = identifier_from_url(url)
    if identifier is not None and identifier.namespace in {"arxiv", "doi"}:
        return "paper_reference"
    if identifier is not None and identifier.namespace == "github:repository":
        return "code_reference"
    return "metadata_reference"


def _link_relation(header: str | None, relation: str) -> str | None:
    if not header:
        return None
    wanted = relation.casefold()
    for match in _LINK_RE.finditer(header):
        rel_match = _REL_RE.search(match.group(2))
        relations = (rel_match.group(1) or rel_match.group(2) or "") if rel_match else ""
        if wanted in relations.casefold().split():
            return match.group(1)
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.casefold()
    return next((value for key, value in headers.items() if key.casefold() == wanted), None)


def _integer_header(headers: Mapping[str, str], name: str) -> int | None:
    value = _header(headers, name)
    try:
        result = int(value) if value is not None else None
    except ValueError:
        return None
    return result if result is not None and result >= 0 else None


def _state_count(state: Mapping[str, Any], key: str) -> int | None:
    if key not in state:
        return None
    value = state.get(key)
    if isinstance(value, bool):
        raise ValueError(f"invalid Hugging Face checkpoint {key}: {value!r}")
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid Hugging Face checkpoint {key}: {value!r}") from error
    if count < 0:
        raise ValueError(f"invalid Hugging Face checkpoint {key}: {value!r}")
    return count


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _unique_identifiers(values: Iterable[Identifier]) -> tuple[Identifier, ...]:
    return tuple(dict.fromkeys(values))


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    seen: set[tuple[str, str]] = set()
    result = []
    for value in values:
        key = (value.url, value.relation)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    scheme = parts.scheme.casefold()
    port = parts.port
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else None
    return scheme, (parts.hostname or "").casefold(), port


__all__ = ["HuggingFaceSourceAdapter"]
