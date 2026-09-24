"""Enumerate U-TILISE's first-party public checkpoint directory."""

from __future__ import annotations

from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

from modelome.http import HttpClient
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import content_hash

_REPOSITORY = "prs-eth/U-TILISE"
_README = f"https://github.com/{_REPOSITORY}"
_SCRIPT = f"https://github.com/{_REPOSITORY}/blob/main/scripts/download_checkpoints.sh"
_DIRECTORY = "https://share.phys.ethz.ch/~pf/stuckercdata/u-tilise/checkpoints/"
_CHECKPOINTS: dict[str, str] = {
    "utilise_earthnet2021.pth": "U-TILISE cloud removal, EarthNet2021",
    "utilise_sen12mscrts_wo_s1.pth": "U-TILISE cloud removal, SEN12MS-CR-TS without Sentinel-1 SAR",
    "utilise_sen12mscrts_w_s1.pth": "U-TILISE cloud removal, SEN12MS-CR-TS with Sentinel-1 SAR",
}


class UTilizeCheckpointRegistrySourceAdapter:
    """Read only the three checkpoint names documented by the U-TILISE authors."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the three .pth names documented in the first-party U-TILISE README, "
        "and the directory linked by its first-party download script. Other files "
        "in the directory are ignored. The adapter fetches only the HTML listing, "
        "never checkpoint bytes; emitted records may overlap general web catalogs."
    )

    def __init__(
        self,
        *,
        name: str = "prs-eth-u-tilise-checkpoints",
        max_response_bytes: int = 1024 * 1024,
        client: Any | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        if isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "prs-eth-u-tilise-checkpoints-v1",
                "directory": _DIRECTORY,
                "filenames": sorted(_CHECKPOINTS),
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(_DIRECTORY, headers={"Accept": "text/html"})
        if response.status != 200:
            raise ValueError(f"{self.name}: checkpoint index returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: checkpoint index exceeds configured byte limit")
        filenames = _parse_directory(response.text(), self.name)
        digest = content_hash(response.body)
        if digest == state.get("completed_index_sha256"):
            return SourcePage((), dict(state), True, upstream_count=len(filenames))
        records = tuple(_record(self.name, filename, digest) for filename in filenames)
        return SourcePage(
            records,
            {"completed_index_sha256": digest, "checkpoint_count": len(records)},
            True,
            upstream_count=len(filenames),
            authoritative_snapshot=True,
        )


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() == "a":
            self.hrefs.extend(value for key, value in attrs if key.casefold() == "href" and value)


def _parse_directory(document: str, source: str) -> tuple[str, ...]:
    parser = _Links()
    parser.feed(document)
    found: set[str] = set()
    for href in parser.hrefs:
        absolute = urljoin(_DIRECTORY, href)
        parsed = urlsplit(absolute)
        if (parsed.scheme, parsed.netloc, parsed.path.rsplit("/", 1)[0] + "/") != (
            "https",
            "share.phys.ethz.ch",
            urlsplit(_DIRECTORY).path,
        ):
            continue
        filename = parsed.path.rsplit("/", 1)[-1]
        if filename not in _CHECKPOINTS:
            continue
        if parsed.query or parsed.fragment or href != quote(filename, safe=""):
            raise ValueError(f"{source}: non-canonical checkpoint index link {href!r}")
        if filename in found:
            raise ValueError(f"{source}: duplicate checkpoint index link {filename!r}")
        found.add(filename)
    if found != set(_CHECKPOINTS):
        raise ValueError(f"{source}: expected exactly the three documented checkpoint files")
    return tuple(sorted(found))


def _record(source: str, filename: str, index_sha256: str) -> SourceRecord:
    url = urljoin(_DIRECTORY, quote(filename, safe=""))
    model_id = filename.removesuffix(".pth")
    return SourceRecord(
        source_record_id=f"{source}:{filename}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=url,
        title=_CHECKPOINTS[filename],
        raw={
            "repository": _REPOSITORY,
            "filename": filename,
            "directory_index_sha256": index_sha256,
            "index_url": _DIRECTORY,
        },
        identifiers=(Identifier("u-tilise:checkpoint", filename),),
        links=(
            Link(
                url,
                relation="weights",
                locator="first-party checkpoint directory index",
                crawl=False,
                model_local_ids=(model_id,),
            ),
            Link(
                _README,
                relation="documentation",
                locator="U-TILISE README checkpoint list",
                crawl=False,
                model_local_ids=(model_id,),
            ),
            Link(
                _SCRIPT,
                relation="checkpoint_inventory",
                locator="first-party download script directory URL",
                crawl=False,
                model_local_ids=(model_id,),
            ),
        ),
        models=(
            ModelHint(
                local_id=model_id,
                name=_CHECKPOINTS[filename],
                identifiers=(Identifier("u-tilise:model", model_id),),
                locator="first-party README checkpoint filename",
            ),
        ),
        releases=(
            ReleaseHint(
                local_id=f"{model_id}@directory-index",
                model_local_id=model_id,
                identifiers=(Identifier("u-tilise:checkpoint", filename),),
                metadata={"checkpoint_filename": filename, "index_sha256": index_sha256},
                locator="first-party directory index",
            ),
        ),
    )
