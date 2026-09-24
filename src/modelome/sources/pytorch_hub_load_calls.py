"""Capture exact ``torch.hub.load(repo:ref, entrypoint, ...)`` examples from Hub cards."""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

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

_INDEX_URL = "https://pytorch.org/hub/"
_NAMESPACE = "pytorch:hub-model-page"
_HUB_PAGE = re.compile(r"^/hub/(?P<slug>[a-z0-9][a-z0-9_-]*)/?$")
_REPO_REF = re.compile(
    r"^(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+):(?P<ref>[A-Za-z0-9_.-]+)$"
)
_ENTRYPOINT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _HubPageParser(HTMLParser):
    def __init__(self, *, max_code_chars: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_code_chars = max_code_chars
        self.anchors: list[tuple[str, str]] = []
        self.title_parts: list[str] = []
        self.code_blocks: list[str] = []
        self._anchor: tuple[str, list[str]] | None = None
        self._pre_depth = 0
        self._code_depth = 0
        self._capture_inline_code = False
        self._code_parts: list[str] | None = None
        self._code_chars = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        attributes = {key.casefold(): value or "" for key, value in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "a":
            self._anchor = (attributes.get("href", ""), [])
        elif tag == "pre":
            if self._pre_depth == 0:
                self._start_code_block()
            self._pre_depth += 1
        elif tag == "code":
            if self._pre_depth == 0 and self._code_parts is None:
                self._capture_inline_code = True
                self._start_code_block()
            self._code_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "title":
            self._in_title = False
        elif tag == "a" and self._anchor is not None:
            href, parts = self._anchor
            self.anchors.append((href.strip(), " ".join("".join(parts).split())))
            self._anchor = None
        elif tag == "pre" and self._pre_depth:
            self._pre_depth -= 1
            if self._pre_depth == 0:
                self._finish_code_block()
        elif tag == "code" and self._code_depth:
            self._code_depth -= 1
            if self._capture_inline_code and self._code_depth == 0:
                self._capture_inline_code = False
                self._finish_code_block()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._anchor is not None:
            self._anchor[1].append(data)
        if self._code_parts is not None and (self._pre_depth or self._capture_inline_code):
            self._code_chars += len(data)
            if self._code_chars > self.max_code_chars:
                raise ValueError(f"Hub card exceeds {self.max_code_chars} code characters")
            self._code_parts.append(data)

    def _start_code_block(self) -> None:
        self._code_parts = []
        self._code_chars = 0

    def _finish_code_block(self) -> None:
        if self._code_parts is not None:
            text = "".join(self._code_parts)
            if text.strip():
                self.code_blocks.append(text)
        self._code_parts = None
        self._code_chars = 0

    def finish(self) -> None:
        if self._anchor is not None:
            href, parts = self._anchor
            self.anchors.append((href.strip(), " ".join("".join(parts).split())))
            self._anchor = None
        if self._code_parts is not None:
            self._finish_code_block()


class PyTorchHubLoadCallSourceAdapter:
    """Supplement PyTorch Hub card identities with exact versioned loader calls.

    Only literal Python calls of the form ``torch.hub.load('owner/repo:ref',
    'entrypoint', ...)`` are retained. The adapter reads the curated index and
    each linked card page, but never follows or downloads model-weight URLs.
    """

    disable_derived_extraction = True

    def __init__(
        self,
        *,
        name: str = "pytorch-hub-load-calls",
        index_url: str = _INDEX_URL,
        page_batch_size: int = 10,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_code_chars: int = 256 * 1024,
        max_pages: int = 1_000,
        client: Any | None = None,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.index_url = _https_url(index_url, "index URL")
        self.page_batch_size = _positive_int(page_batch_size, "page_batch_size")
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_code_chars = _positive_int(max_code_chars, "max_code_chars")
        self.max_pages = _positive_int(max_pages, "max_pages")
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "pytorch-hub-load-calls-v1",
                "index_url": self.index_url,
                "page_batch_size": page_batch_size,
                "max_response_bytes": max_response_bytes,
                "max_code_chars": max_code_chars,
                "max_pages": max_pages,
                "admission": (
                    "literal torch.hub.load calls with exact owner/repo:ref and entrypoint"
                ),
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if "detail_tasks" not in state:
            return self._index_page(state)
        return self._detail_page(state)

    def _index_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(
            self.index_url,
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: index returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: index exceeds {self.max_response_bytes} bytes")
        parser = self._parse(response.text())
        response_url = response.url or self.index_url
        tasks = _card_tasks(parser.anchors, response_url)
        if not tasks:
            raise ValueError(f"{self.name}: index contains no PyTorch Hub model cards")
        if len(tasks) > self.max_pages:
            raise ValueError(f"{self.name}: index exceeds {self.max_pages} model cards")
        checked_at = _isoformat(_utcnow())
        next_state = {
            "detail_tasks": tasks,
            "detail_offset": 0,
            "index_sha256": content_hash(response.body),
            "checked_at": checked_at,
            "entry_count": len(tasks),
        }
        if etag := _header(response.headers, "etag"):
            next_state["etag"] = etag
        return SourcePage(
            records=(),
            next_state=next_state,
            complete=False,
            upstream_count=len(tasks),
        )

    def _detail_page(self, state: Mapping[str, Any]) -> SourcePage:
        tasks = _tasks_from_state(state, self.name, self.max_pages)
        offset = state.get("detail_offset", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset < len(tasks):
            raise ValueError(f"{self.name}: detail offset is invalid")
        batch = tasks[offset : offset + self.page_batch_size]
        records = []
        for task in batch:
            response = self.client.get(
                task["url"],
                headers={"Accept": "text/html,application/xhtml+xml"},
            )
            if response.status != 200:
                raise ValueError(f"{self.name}: card {task['url']} returned HTTP {response.status}")
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: card {task['url']} exceeds response limit")
            parser = self._parse(response.text())
            records.append(self._record(task, parser, response))
        next_offset = offset + len(batch)
        complete = next_offset == len(tasks)
        next_state = dict(state)
        if complete:
            next_state.pop("detail_tasks", None)
            next_state.pop("detail_offset", None)
        else:
            next_state["detail_offset"] = next_offset
        next_state["checked_at"] = _isoformat(_utcnow())
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=len(tasks),
            authoritative_snapshot=complete,
        )

    def _parse(self, text: str) -> _HubPageParser:
        parser = _HubPageParser(max_code_chars=self.max_code_chars)
        parser.feed(text)
        parser.close()
        parser.finish()
        return parser

    def _record(
        self,
        task: Mapping[str, str],
        parser: _HubPageParser,
        response: HttpResponse,
    ) -> SourceRecord:
        slug = task["slug"]
        local_id = _model_local_id(slug)
        identifier = Identifier(_NAMESPACE, slug)
        response_url = response.url or task["url"]
        model_name = task["name"] or " ".join(("".join(parser.title_parts)).split()) or slug
        model = ModelHint(
            local_id=local_id,
            name=model_name,
            identifiers=(identifier,),
            status=ModelStatus.DOCUMENTED,
            locator=f"hub-card:{slug}",
        )
        calls = _load_calls(parser.code_blocks)
        links = [Link(task["url"], relation="model_card", crawl=False, model_local_ids=(local_id,))]
        releases = []
        for ordinal, call in enumerate(calls):
            ref = call["ref"]
            repo_url = f"https://github.com/{call['repo']}/tree/{quote(ref, safe='')}"
            call_id = content_hash(
                {
                    "repo": call["repo"],
                    "ref": ref,
                    "entrypoint": call["entrypoint"],
                    "arguments": call["arguments"],
                }
            )[:24]
            locator = f"hub-card:{slug}:torch.hub.load[{ordinal}]"
            links.append(
                Link(
                    repo_url,
                    relation="official_implementation",
                    locator=locator,
                    crawl=False,
                    model_local_ids=(local_id,),
                )
            )
            if _requests_pretrained_weights(call["arguments"]):
                call_identifier = Identifier("pytorch:hub-load", call_id)
                releases.append(
                    ReleaseHint(
                        local_id=f"release:{slug}:{call_id}",
                        model_local_id=local_id,
                        version=ref,
                        identifiers=(call_identifier,),
                        metadata={
                            "repository": call["repo"],
                            "repository_ref": ref,
                            "entrypoint": call["entrypoint"],
                            "arguments": call["arguments"],
                            "loader": "torch.hub.load",
                        },
                        locator=locator,
                    )
                )
        raw_calls = [
            {
                "repository": call["repo"],
                "repository_ref": call["ref"],
                "entrypoint": call["entrypoint"],
                "arguments": call["arguments"],
                "locator": f"hub-card:{slug}:torch.hub.load[{ordinal}]",
            }
            for ordinal, call in enumerate(calls)
        ]
        content_hash_value = content_hash(response.body)
        return SourceRecord(
            source_record_id=f"{self.name}:model-page:{slug}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(response_url),
            title=model_name,
            raw={
                "index_url": self.index_url,
                "page_slug": slug,
                "source_declared_page_url": task["url"],
                "content_sha256": content_hash_value,
                "torch_hub_load_calls": raw_calls,
            },
            text="\n".join(
                [model_name]
                + [
                    f"torch.hub.load: {call['repo']}:{call['ref']} {call['entrypoint']}"
                    for call in calls
                ]
            ),
            identifiers=(identifier,),
            links=tuple(links),
            models=(model,),
            releases=tuple(releases),
        )


def _load_calls(code_blocks: list[str]) -> tuple[dict[str, Any], ...]:
    calls: list[dict[str, Any]] = []
    seen = set()
    for block_index, block in enumerate(code_blocks):
        try:
            tree = ast.parse(textwrap.dedent(block).strip())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node.func) != "torch.hub.load":
                continue
            if len(node.args) < 2:
                continue
            repo_ref = _literal_string(node.args[0])
            entrypoint = _literal_string(node.args[1])
            match = _REPO_REF.fullmatch(repo_ref)
            if match is None or not _ENTRYPOINT.fullmatch(entrypoint):
                continue
            arguments = {
                f"arg_{index + 3}": _safe_unparse(value)
                for index, value in enumerate(node.args[2:])
            }
            arguments.update(
                {
                    keyword.arg: _safe_unparse(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
            )
            identity = (
                match.group("repo"),
                match.group("ref"),
                entrypoint,
                tuple(sorted(arguments.items())),
            )
            if identity in seen:
                continue
            seen.add(identity)
            calls.append(
                {
                    "repo": match.group("repo"),
                    "ref": match.group("ref"),
                    "entrypoint": entrypoint,
                    "arguments": arguments,
                    "code_block": block_index,
                }
            )
    return tuple(calls)


def _card_tasks(anchors: list[tuple[str, str]], base_url: str) -> list[dict[str, str]]:
    tasks: dict[str, dict[str, str]] = {}
    for href, label in anchors:
        absolute = urljoin(base_url, href)
        parsed = urlsplit(absolute)
        if parsed.scheme != "https" or parsed.netloc.casefold() != "pytorch.org":
            continue
        match = _HUB_PAGE.fullmatch(parsed.path)
        if match is None:
            continue
        slug = match.group("slug")
        tasks.setdefault(
            slug,
            {"slug": slug, "url": f"https://pytorch.org/hub/{slug}", "name": label},
        )
    return list(tasks.values())


def _tasks_from_state(
    state: Mapping[str, Any],
    source: str,
    max_pages: int,
) -> list[dict[str, str]]:
    raw = state.get("detail_tasks")
    if not isinstance(raw, list) or not raw or len(raw) > max_pages:
        raise ValueError(f"{source}: detail task state is invalid")
    tasks = []
    seen = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError(f"{source}: detail task is invalid")
        slug = _required_text(item.get("slug"), "page slug")
        url = _https_url(item.get("url"), "card URL")
        if url != f"https://pytorch.org/hub/{slug}" or slug in seen:
            raise ValueError(f"{source}: detail task URL or slug is invalid")
        seen.add(slug)
        tasks.append({"slug": slug, "url": url, "name": _text(item.get("name"))})
    return tasks


def _model_local_id(slug: str) -> str:
    return f"{_NAMESPACE}:{content_hash(slug)[:24]}#model"


def _requests_pretrained_weights(arguments: Mapping[str, str]) -> bool:
    if arguments.get("pretrained") == "True":
        return True
    return "weights" in arguments and arguments["weights"] not in {"None", "False"}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _literal_string(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    return ""


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparsed>"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _required_text(value: Any, field: str) -> str:
    value = _text(value)
    if not value:
        raise ValueError(f"{field} is required")
    return value


def _https_url(value: Any, field: str) -> str:
    text = _required_text(value, field)
    parsed = urlsplit(text)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{field} must be an HTTPS URL")
    return text


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _header(headers: Mapping[str, Any], key: str) -> str:
    return _text(headers.get(key) or headers.get(key.casefold()) or headers.get(key.title()))


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["PyTorchHubLoadCallSourceAdapter"]
