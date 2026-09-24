from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

from modelome.lake import ParquetLandingZone, ReleaseReceipt
from modelome.models import ArtifactKind, Identifier, Link, SourceRecord


@dataclass(frozen=True, slots=True)
class OpenAireSoftwareProjectionPage:
    release: str
    start_row: int
    next_row: int
    total_rows: int
    rows_examined: int
    records: tuple[SourceRecord, ...]
    complete: bool


class OpenAireSoftwareProjector:
    """Project explicitly typed OpenAIRE software products from landed rows.

    The source-wide graph remains untouched in Parquet. This view only emits
    products whose upstream ``type`` is exactly ``software``; it does not infer
    model identity from names, descriptions, or resource URLs.
    """

    def __init__(
        self,
        landing_zone: ParquetLandingZone,
        *,
        source: str = "openaire-graph",
        dataset: str = "graph",
    ) -> None:
        if not isinstance(landing_zone, ParquetLandingZone):
            raise TypeError("landing_zone must be a ParquetLandingZone")
        self.landing_zone = landing_zone
        self.source = _required_text(source, "source")
        self.dataset = _required_text(dataset, "dataset")

    def page(
        self,
        release: str,
        *,
        start_row: int = 0,
        max_rows: int = 100_000,
        arrow_batch_size: int = 16_384,
    ) -> OpenAireSoftwareProjectionPage:
        release = _required_text(release, "release")
        start_row = _nonnegative_int(start_row, "start_row")
        max_rows = _positive_int(max_rows, "max_rows")
        arrow_batch_size = _positive_int(arrow_batch_size, "arrow_batch_size")
        receipt = self._release(release)
        if start_row > receipt.row_count:
            raise ValueError("start_row is beyond the sealed release row count")

        records: list[SourceRecord] = []
        row_index = 0
        examined = 0
        stopped = False
        for batch in self.landing_zone.iter_release_batches(
            source=self.source,
            dataset=self.dataset,
            release=release,
            columns=("source_record_id", "payload_json", "content_sha256"),
            batch_size=arrow_batch_size,
        ):
            for row in batch.to_pylist():
                if row_index < start_row:
                    row_index += 1
                    continue
                if examined >= max_rows:
                    stopped = True
                    break
                record = _project_row(row)
                if record is not None:
                    records.append(record)
                examined += 1
                row_index += 1
            if stopped:
                break
        next_row = start_row + examined
        return OpenAireSoftwareProjectionPage(
            release=release,
            start_row=start_row,
            next_row=next_row,
            total_rows=receipt.row_count,
            rows_examined=examined,
            records=tuple(records),
            complete=next_row == receipt.row_count,
        )

    def _release(self, release: str) -> ReleaseReceipt:
        matches = tuple(
            item
            for item in self.landing_zone.list_releases(
                source=self.source,
                dataset=self.dataset,
                verify_shards=False,
            )
            if item.release == release
        )
        if len(matches) != 1:
            raise ValueError(
                f"expected one sealed {self.source}/{self.dataset}/{release} release, "
                f"found {len(matches)}"
            )
        return matches[0]


def _project_row(row: Mapping[str, Any]) -> SourceRecord | None:
    payload_json = _required_text(row.get("payload_json"), "payload_json")
    digest = _required_text(row.get("content_sha256"), "content_sha256")
    if hashlib.sha256(payload_json.encode("utf-8")).hexdigest() != digest:
        raise ValueError("OpenAIRE landing row content hash does not match payload_json")
    try:
        payload = json.loads(payload_json)
    except (json.JSONDecodeError, RecursionError):
        raise ValueError("OpenAIRE landing payload_json is invalid") from None
    if not isinstance(payload, Mapping):
        raise ValueError("OpenAIRE landing payload_json must contain an object")
    software = project_openaire_software(payload)
    if software is not None:
        return software
    return project_openaire_relation(payload)


def project_openaire_software(payload: Mapping[str, Any]) -> SourceRecord | None:
    """Project one Graph product using only explicit software type, PIDs, and URLs."""

    if not isinstance(payload, Mapping):
        raise TypeError("OpenAIRE product payload must be a mapping")
    if _text(payload.get("type")) != "software":
        return None
    graph_id = _text(payload.get("id"))
    if graph_id is None:
        return None

    identifiers = [Identifier("openaire:graph-product", graph_id)]
    pids = _sequence(payload.get("pids")) or _sequence(payload.get("pid"))
    canonical_url: str | None = None
    for pid in pids:
        if not isinstance(pid, Mapping):
            continue
        scheme = _text(pid.get("scheme") or pid.get("qualifier"))
        value = _text(pid.get("value") or pid.get("id"))
        if scheme is None or value is None:
            continue
        identifiers.append(Identifier(f"openaire-pid:{scheme.casefold()}", value))
        if scheme.casefold() == "doi" and canonical_url is None:
            canonical_url = f"https://doi.org/{quote(value, safe='/') }"
        elif scheme.casefold() in {"url", "uri"} and canonical_url is None:
            canonical_url = _public_https(value)

    links: list[Link] = []
    instance_pids: list[dict[str, str]] = []
    code_repository_urls = _safe_urls(payload.get("codeRepositoryUrl"))
    documentation_urls = _safe_urls(payload.get("documentationUrl"))
    for url in code_repository_urls:
        links.append(Link(url, relation="code_repository", locator="codeRepositoryUrl"))
    for index, url in enumerate(documentation_urls):
        links.append(
            Link(url, relation="documentation", locator=f"documentationUrl[{index}]")
        )

    instances = _sequence(payload.get("instances")) or _sequence(payload.get("instance"))
    for instance_index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            continue
        instance_pid_values = _sequence(instance.get("pids")) or _sequence(
            instance.get("pid")
        )
        instance_pid_values += _sequence(instance.get("alternateIdentifiers")) or _sequence(
            instance.get("alternateIdentifier")
        )
        for pid in instance_pid_values:
            if not isinstance(pid, Mapping):
                continue
            scheme = _text(pid.get("scheme"))
            value = _text(pid.get("value"))
            if scheme is None or value is None:
                continue
            instance_pids.append({"scheme": scheme, "value": value})
            pid_url = _pid_url(scheme, value)
            if pid_url is not None:
                links.append(
                    Link(
                        pid_url,
                        relation="instance_identifier",
                        locator=f"instance[{instance_index}].pid",
                        crawl=False,
                    )
                )
        urls = _sequence(instance.get("urls")) or _sequence(instance.get("url"))
        if not urls:
            urls = _sequence(instance.get("webresource"))
        for value in urls:
            url_text = _text(value)
            if isinstance(value, Mapping):
                url_text = _text(value.get("url"))
            url = _public_https(url_text) if url_text is not None else None
            if url is not None:
                links.append(
                    Link(
                        url,
                        relation="provider_resource",
                        locator=f"instances[{instance_index}]",
                        crawl=True,
                    )
                )
    if canonical_url is None and links:
        canonical_url = next(
            (
                item.url
                for item in links
                if item.relation == "code_repository"
            ),
            links[0].url,
        )
    if canonical_url is None:
        return None

    title = _title(payload.get("mainTitle")) or _title(payload.get("title")) or graph_id
    unique_identifiers = tuple(dict.fromkeys(identifiers))
    unique_links = tuple(dict.fromkeys(links))
    return SourceRecord(
        source_record_id=f"openaire-graph:software:{quote(graph_id, safe='')}",
        kind=ArtifactKind.OTHER,
        canonical_url=canonical_url,
        title=title,
        identifiers=unique_identifiers,
        links=unique_links,
        raw={
            "record_type": "openaire_software_product",
            "product_type": "software",
            "openaire_graph_id": graph_id,
            "source_wide_filtering": False,
            "model_classification_performed": False,
            "pids": [
                {"namespace": item.namespace, "value": item.value}
                for item in unique_identifiers
                if item.namespace.startswith("openaire-pid:")
            ],
            "provider_urls": [item.url for item in unique_links],
            "code_repository_urls": list(code_repository_urls),
            "documentation_urls": list(documentation_urls),
            "instance_pids": instance_pids,
        },
    )


def project_openaire_relation(payload: Mapping[str, Any]) -> SourceRecord | None:
    """Expose explicit software-to-publication/dataset edges as non-model evidence."""

    if not isinstance(payload, Mapping):
        raise TypeError("OpenAIRE relation payload must be a mapping")
    source_id, source_type = _relation_node(payload, "source")
    target_id, target_type = _relation_node(payload, "target")
    if source_type != "software" or target_type not in {"publication", "dataset", "data"}:
        return None
    relation = payload.get("relType") or payload.get("reltype")
    if isinstance(relation, Mapping):
        predicate = _text(relation.get("name"))
        relation_type = _text(relation.get("type"))
    else:
        predicate = _text(payload.get("relation") or payload.get("predicate"))
        relation_type = None
    if predicate is None:
        return None

    source_url = _research_product_api_url(source_id)
    target_url = _research_product_api_url(target_id)
    row_digest = hashlib.sha256(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    edge_identity = f"{source_id}|{predicate}|{target_id}"
    return SourceRecord(
        source_record_id=f"openaire-graph:relation-evidence:{row_digest}",
        kind=ArtifactKind.OTHER,
        canonical_url=source_url,
        title=f"OpenAIRE software relation: {predicate}",
        identifiers=(
            Identifier("openaire:graph-product", source_id),
            Identifier("openaire:graph-relation-sha256", row_digest),
        ),
        links=(
            Link(
                target_url,
                relation=f"openaire_related_{_relation_slug(predicate)}",
                locator=f"relation:{edge_identity}",
                crawl=False,
            ),
        ),
        raw={
            "record_type": "openaire_related_product_evidence",
            "source_product_id": source_id,
            "source_product_type": source_type,
            "target_product_id": target_id,
            "target_product_type": target_type,
            "relation_type": relation_type,
            "relation_predicate": predicate,
            "relation_provenance": payload.get("provenance"),
            "relation_validated": payload.get("validated"),
            "relation_validation_date": payload.get("validationDate"),
            "source_wide_filtering": False,
            "model_classification_performed": False,
        },
    )


def _title(value: Any) -> str | None:
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, Mapping):
        return _text(value.get("value") or value.get("text"))
    if isinstance(value, (list, tuple)):
        for item in value:
            result = _title(item)
            if result is not None:
                return result
    return None


def _public_https(value: str) -> str | None:
    try:
        parts = urlsplit(value)
        if (
            parts.scheme.casefold() != "https"
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.port not in {None, 443}
        ):
            return None
        return (
            value.rstrip("/")
            if parts.path == "/" and not parts.query and not parts.fragment
            else value
        )
    except ValueError:
        return None


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if value is None:
        return ()
    return (value,)


def _safe_urls(value: Any) -> tuple[str, ...]:
    urls: list[str] = []
    for item in _sequence(value):
        text = _text(item)
        url = _public_https(text) if text is not None else None
        if url is not None and url not in urls:
            urls.append(url)
    return tuple(urls)


def _pid_url(scheme: str, value: str) -> str | None:
    normalized = scheme.casefold()
    if normalized == "doi":
        return f"https://doi.org/{quote(value, safe='/')}"
    if normalized in {"url", "uri"}:
        return _public_https(value)
    return None


def _relation_node(payload: Mapping[str, Any], role: str) -> tuple[str | None, str | None]:
    node = payload.get(role)
    if isinstance(node, Mapping):
        node_id = _text(node.get("id"))
        node_type = _text(node.get("type"))
    else:
        node_id = _text(node)
        node_type = None
    node_type = _text(payload.get(f"{role}Type")) or node_type
    return node_id, node_type.casefold() if node_type is not None else None


def _research_product_api_url(graph_id: str) -> str:
    return f"https://api.openaire.eu/graph/v3/research-products/{quote(graph_id, safe='')}"


def _relation_slug(value: str) -> str:
    slug = "".join(char.casefold() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in slug.split("_") if part)


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip()
    return result or None


def _required_text(value: Any, name: str) -> str:
    result = _text(value)
    if result is None:
        raise ValueError(f"{name} must be nonempty text")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value
