"""Enumerate FourCastNet checkpoint files from its public NERSC directory index."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

from modelome.http import HttpResponse
from modelome.models import Link, SourcePage
from modelome.normalize import content_hash
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)

_INDEX_URL = "https://portal.nersc.gov/project/m4134/FCN_weights_v0/"
_EXPECTED = {"backbone.ckpt", "precip.ckpt"}


class FourCastNetCheckpointRegistrySourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Enumerate the exact model files in the public index linked by FourCastNet."""

    coverage_limitation = (
        "Covers only backbone.ckpt and precip.ckpt in the public FourCastNet v0 "
        "NERSC directory index named by the first-party NVlabs README. These files "
        "are separate model checkpoints from the same published directory. The "
        "adapter does not resolve redirects, inspect contents, or download bytes."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "nvlabs-fourcastnet-checkpoint-registry")
        kwargs.setdefault("repository", "NVlabs/FourCastNet")
        kwargs.setdefault("branch", "master")
        kwargs.setdefault("source_path", "README.md")
        kwargs.setdefault("provider_namespace", "fourcastnet:checkpoint")
        kwargs.setdefault("max_entries", 2)
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "fourcastnet-public-index-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "index_url": _INDEX_URL,
                "max_response_bytes": self.max_response_bytes,
                "admission": "two exact files in the linked public index",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        response: HttpResponse = self.client.get(
            _INDEX_URL, headers={"Accept": "text/html"}
        )
        if response.status != 200:
            raise ValueError(
                f"{self.name}: public checkpoint index returned HTTP {response.status}"
            )
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: index exceeds {self.max_response_bytes} bytes")
        source_sha256 = content_hash(response.body)
        checkpoints = _parse_index(response.text(), self.name)
        if (
            revision == _text(state.get("completed_revision"))
            and source_sha256 == _text(state.get("completed_index_sha256"))
        ):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            if etag := _header(commit_response.headers, "etag"):
                next_state["commit_etag"] = etag
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )
        records = tuple(self._record(item, revision, response.body) for item in checkpoints)
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "completed_index_sha256": source_sha256,
            "checked_at": checked_at,
            "source_url": _INDEX_URL,
            "source_sha256": source_sha256,
            "model_count": len(records),
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _model_name(self, handle: str) -> str:
        if handle == "precip.ckpt":
            return "FourCastNet precipitation diagnostic"
        return "FourCastNet backbone"

    def _record(self, checkpoint: _Checkpoint, revision: str, source: bytes):
        record = super()._record(checkpoint, revision, source)
        # Preserve README provenance and add the directory index as catalog
        # evidence; each direct file URL remains intact as artifact evidence.
        links = record.links + (
            Link(
                _INDEX_URL,
                relation="checkpoint_index",
                locator=checkpoint.locator,
                crawl=False,
                model_local_ids=(record.models[0].local_id,),
            ),
        )
        raw = dict(record.raw)
        raw["weight_url"] = checkpoint.url
        return replace(
            record,
            links=links,
            raw=raw,
        )


class _IndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = next((value for key, value in attrs if key.casefold() == "href"), None)
        if href:
            self.hrefs.append(href)


def _parse_index(document: str, source: str) -> tuple[_Checkpoint, ...]:
    parser = _IndexParser()
    parser.feed(document)
    parser.close()
    observed: dict[str, str] = {}
    for href in parser.hrefs:
        absolute = urljoin(_INDEX_URL, href)
        parsed = urlsplit(absolute)
        filename = parsed.path.rsplit("/", 1)[-1]
        in_checkpoint_directory = parsed.path.startswith(
            "/project/m4134/FCN_weights_v0/"
        )
        if not filename.endswith(".ckpt") or (
            not in_checkpoint_directory and filename not in _EXPECTED
        ):
            continue
        if (
            parsed.scheme != "https"
            or parsed.netloc != "portal.nersc.gov"
            or parsed.path != f"/project/m4134/FCN_weights_v0/{filename}"
            or parsed.query
            or parsed.fragment
            or filename not in _EXPECTED
            or filename in observed
        ):
            raise ValueError(f"{source}: unexpected or duplicate checkpoint link {href!r}")
        observed[filename] = absolute
    if set(observed) != _EXPECTED:
        raise ValueError(f"{source}: expected exactly the two published FourCastNet files")
    return tuple(
        _Checkpoint(handle=name, url=observed[name], locator=f"nersc-index:{name}")
        for name in sorted(_EXPECTED)
    )
