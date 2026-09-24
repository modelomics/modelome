"""Static ingestion for OpenCV's curated DNN sample checkpoint index."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpClient, HttpResponse
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

Clock = Callable[[], datetime]
_TOP_FIELD = re.compile(r"^(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*?)\s*$")
_NESTED_FIELD = re.compile(r"^  (?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*?)\s*$")
_LOAD_FIELD = re.compile(r"^    (?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*?)\s*$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class OpenCVDnnModelIndexSourceAdapter:
    """Enumerate checkpoint URLs declared by OpenCV's ``samples/dnn/models.yml``.

    This is the OpenCV project's own alias-to-download index, separate from the
    ONNX Model Zoo's repository and the OpenCV-owned Hugging Face organization.
    It names model formats such as Caffe, TensorFlow, and Darknet and records
    exact third-party or first-party download URLs without fetching checkpoint
    bytes.
    """

    disable_derived_extraction = True

    def __init__(
        self,
        *,
        name: str = "opencv-dnn-model-index",
        url: str = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/models.yml",
        repository_url: str = "https://github.com/opencv/opencv",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = 1_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, "index URL")
        self.repository_url = _web_url(repository_url, "repository URL")
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "opencv-dnn-model-index-v1",
                "url": self.url,
                "repository_url": self.repository_url,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "top-level aliases with exact load_info.url values",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        headers = {"Accept": "text/yaml,text/plain"}
        if etag := _text(state.get("etag")):
            headers["If-None-Match"] = etag
        if modified := _text(state.get("http_last_modified")):
            headers["If-Modified-Since"] = modified
        response: HttpResponse = self.client.get(self.url, headers=headers)
        checked_at = _isoformat(self.clock())
        if response.status == 304:
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(records=(), next_state=next_state, complete=True)
        if response.status != 200:
            raise ValueError(f"{self.name}: index returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: index exceeds {self.max_response_bytes} bytes")
        entries = _parse_index(response.text(), self.name)
        if not entries:
            raise ValueError(f"{self.name}: index contains no entries with checkpoint URLs")
        if len(entries) > self.max_entries:
            raise ValueError(f"{self.name}: index exceeds {self.max_entries} entries")
        index_hash = content_hash(response.body)
        records = tuple(self._record(entry, index_hash=index_hash) for entry in entries)
        next_state: dict[str, Any] = {
            "checked_at": checked_at,
            "content_hash": index_hash,
            "entry_count": len(records),
        }
        if etag := _header(response.headers, "etag"):
            next_state["etag"] = etag
        if modified := _header(response.headers, "last-modified"):
            next_state["http_last_modified"] = modified
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, entry: _Entry, *, index_hash: str) -> SourceRecord:
        model_id = Identifier("opencv:dnn-sample-model", entry.name)
        model = ModelHint(
            local_id=f"model:{entry.name}",
            name=entry.name,
            identifiers=(model_id,),
            status=ModelStatus.RELEASED,
            locator=entry.locator,
        )
        links = [
            Link(self.url, relation="model_page", locator=entry.locator, crawl=False),
            Link(self.repository_url, relation="source_repository", crawl=False),
            Link(
                entry.url,
                relation="weights",
                locator=f"{entry.locator}:load_info.url",
                crawl=False,
            ),
        ]
        record = SourceRecord(
            source_record_id=f"model:{entry.name}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(self.url),
            title=entry.name,
            raw={
                "index_url": self.url,
                "index_sha256": index_hash,
                "alias": entry.name,
                "load_info": dict(entry.load_info),
                "model_filename": entry.model_filename,
                "config_filename": entry.config_filename,
                "sample": entry.sample,
            },
            text="\n".join(
                part
                for part in (entry.name, entry.model_filename or "", entry.sample or "")
                if part
            ),
            identifiers=(model_id,),
            links=tuple(links),
            models=(model,),
            releases=(
                ReleaseHint(
                    local_id=f"release:{entry.name}",
                    model_local_id=model.local_id,
                    identifiers=(model_id,),
                    metadata={
                        "format": _format(entry.model_filename, entry.url),
                        "checkpoint_url": entry.url,
                        "sha1": entry.load_info.get("sha1"),
                        "download_sha": entry.load_info.get("download_sha"),
                        "download_name": entry.load_info.get("download_name"),
                        "member": entry.load_info.get("member"),
                        "index_sha256": index_hash,
                    },
                    locator=entry.locator,
                ),
            ),
        )
        return record


@dataclass(frozen=True, slots=True)
class _Entry:
    name: str
    url: str
    load_info: Mapping[str, str]
    model_filename: str | None
    config_filename: str | None
    sample: str | None
    locator: str


def _parse_index(document: str, source: str) -> tuple[_Entry, ...]:
    entries: list[_Entry] = []
    current: dict[str, Any] | None = None
    in_load_info = False
    for line_number, line in enumerate(document.splitlines(), start=1):
        if not line or line.lstrip().startswith(("#", "%", "---")):
            continue
        top = _TOP_FIELD.match(line)
        if top is not None:
            if current is not None:
                _append_entry(entries, current, source)
            name = top.group("key")
            if name.casefold() in {"true", "false", "null"}:
                current = None
                continue
            current = {
                "name": name,
                "line": line_number,
                "load_info": {},
                "model": None,
                "config": None,
                "sample": None,
            }
            in_load_info = False
            continue
        if current is None:
            continue
        nested = _NESTED_FIELD.match(line)
        if nested is not None:
            key, raw_value = nested.group("key"), nested.group("value")
            in_load_info = key == "load_info" and not raw_value
            if key in {"model", "config", "sample"}:
                current[key] = _yaml_scalar(raw_value)
            continue
        if in_load_info and (field := _LOAD_FIELD.match(line)) is not None:
            current["load_info"][field.group("key")] = _yaml_scalar(field.group("value"))
    if current is not None:
        _append_entry(entries, current, source)
    names = [entry.name for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError(f"{source}: index contains duplicate model aliases")
    return tuple(entries)


def _append_entry(entries: list[_Entry], raw: Mapping[str, Any], source: str) -> None:
    load_info = raw["load_info"]
    url = _text(load_info.get("url")) if isinstance(load_info, Mapping) else ""
    if not url:
        return
    if not _is_web_url(url):
        raise ValueError(f"{source}: {raw['name']} has an invalid checkpoint URL")
    entries.append(
        _Entry(
            name=raw["name"],
            url=url,
            load_info=dict(load_info),
            model_filename=_optional_text(raw.get("model")),
            config_filename=_optional_text(raw.get("config")),
            sample=_optional_text(raw.get("sample")),
            locator=f"line:{raw['line']}",
        )
    )


def _yaml_scalar(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    return parsed if isinstance(parsed, str) else value


def _format(filename: str | None, url: str) -> str | None:
    from urllib.parse import urlsplit

    candidate = filename or urlsplit(url).path.rsplit("/", 1)[-1]
    if "." not in candidate:
        return None
    return candidate.rsplit(".", 1)[-1].casefold()


def _required_text(value: Any, field: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{field} is required")
    return result


def _optional_text(value: Any) -> str | None:
    result = _text(value)
    return result or None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _web_url(value: Any, field: str) -> str:
    text = _required_text(value, field)
    if not _is_web_url(text):
        raise ValueError(f"{field} must be an HTTP(S) URL")
    return text


def _is_web_url(value: str) -> bool:
    return value.startswith(("https://", "http://"))


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _header(headers: Mapping[str, Any], key: str) -> str:
    return _text(headers.get(key) or headers.get(key.casefold()) or headers.get(key.title()))


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["OpenCVDnnModelIndexSourceAdapter"]
