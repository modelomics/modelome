"""Official Grounding DINO checkpoint rows and their exact mirror URLs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urlsplit

from modelome.http import HttpClient
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

_REPOSITORY = "IDEA-Research/GroundingDINO"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_MODELS = {
    "GroundingDINO-T": (
        "groundingdino_swint_ogc.pth",
        "v0.1.0-alpha",
        "GroundingDINO_SwinT_OGC.py",
    ),
    "GroundingDINO-B": (
        "groundingdino_swinb_cogcoor.pth",
        "v0.1.0-alpha2",
        "GroundingDINO_SwinB_cfg.py",
    ),
}


class GroundingDINOCheckpointSourceAdapter:
    """Read the two exact variant/checkpoint mappings in the official README."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the official README's GroundingDINO-T and -B checkpoint rows, "
        "including their direct GitHub release assets, Hugging Face mirrors, and "
        "config links. It does not enumerate community fine-tunes, model APIs, or "
        "historical rows removed from the README."
    )

    def __init__(
        self,
        *,
        name: str = "groundingdino-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "main",
        max_readme_bytes: int = 2 * 1024 * 1024,
        max_entries: int = 100,
        client: HttpClient | Any | None = None,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if not name.strip() or not branch.strip() or min(max_readme_bytes, max_entries) <= 0:
            raise ValueError("name, branch, and positive limits are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_readme_bytes = max_readme_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_readme_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "groundingdino-checkpoint-table-v1",
                "repository": repository,
                "branch": branch,
                "readme_bytes": max_readme_bytes,
                "max_entries": max_entries,
                "admission": "official README rows with exact GitHub release and HF mirror",
            }
        )

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}"
        )

    def readme_url(self, revision: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repository}/{revision}/README.md"

    def source_page_url(self, revision: str) -> str:
        return f"https://github.com/{self.repository}/blob/{revision}/README.md"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid repository revision")
        if revision == state.get("completed_revision"):
            count = state.get("model_count")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{self.name}: invalid completed model count")
            return SourcePage((), dict(state), True, upstream_count=count)

        url = self.readme_url(revision)
        response = self.client.get(url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_readme_bytes:
            raise ValueError(f"{self.name}: README exceeds response limit")
        try:
            readme = response.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{self.name}: README is not UTF-8") from exc
        rows = _parse_checkpoint_rows(readme, maximum=self.max_entries)
        if set(rows) != set(_MODELS):
            raise ValueError(
                f"{self.name}: README checkpoint table did not contain the expected variants"
            )
        page_url = self.source_page_url(revision)
        records = tuple(
            self._record(name, rows[name], revision, page_url, response.body) for name in _MODELS
        )
        next_state = {
            "completed_revision": revision,
            "model_count": len(records),
            "source_url": url,
            "source_sha256": content_hash(response.body),
        }
        return SourcePage(
            records,
            next_state,
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        name: str,
        row: tuple[str, str, str, str],
        revision: str,
        page_url: str,
        source: bytes,
    ) -> SourceRecord:
        checkpoint_url, mirror_url, config_url, metric = row
        filename, tag, expected_config = _MODELS[name]
        if (
            _release_coordinates(checkpoint_url) != (tag, filename)
            or _mirror_filename(mirror_url) != filename
            or _config_filename(config_url) != expected_config
        ):
            raise ValueError(f"{self.name}: unsafe or inconsistent checkpoint mapping for {name}")
        model_local_id = f"model:{name}"
        identifier = Identifier("groundingdino:model", name)
        model = ModelHint(
            local_id=model_local_id,
            name=name,
            aliases=(filename,),
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=f"README.md checkpoint table: {name}",
        )
        release = ReleaseHint(
            local_id=f"release:{name}:{tag}",
            model_local_id=model_local_id,
            version=tag,
            identifiers=(Identifier("groundingdino:checkpoint", filename),),
            metadata={
                "checkpoint_filename": filename,
                "github_release_url": checkpoint_url,
                "huggingface_mirror_url": mirror_url,
                "config_url": config_url,
                "reported_coco_box_ap": metric,
            },
            locator=f"README.md checkpoint table: {name}",
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{name}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(page_url),
            title=name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": "README.md",
                "source_sha256": content_hash(source),
                "variant": name,
                "checkpoint_filename": filename,
                "release_tag": tag,
                "github_release_url": checkpoint_url,
                "huggingface_mirror_url": mirror_url,
                "config_url": config_url,
                "reported_coco_box_ap": metric,
            },
            text=(
                f"Official {name} checkpoint {filename}; GitHub release and "
                f"Hugging Face mirror are paired by the same README row."
            ),
            identifiers=(identifier,),
            links=(
                Link(page_url, relation="model_card", crawl=False,
                     locator=model.locator, model_local_ids=(model_local_id,)),
                Link(checkpoint_url, relation="weights", crawl=False,
                     locator=model.locator, model_local_ids=(model_local_id,)),
                Link(mirror_url, relation="weights_mirror", crawl=False,
                     locator=model.locator, model_local_ids=(model_local_id,)),
                Link(config_url, relation="model_config", crawl=False,
                     locator=model.locator, model_local_ids=(model_local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[tuple[str, tuple[str, ...]]]]] = []
        self._table: list[list[tuple[str, tuple[str, ...]]]] | None = None
        self._row: list[tuple[str, tuple[str, ...]]] | None = None
        self._cell_text: list[str] | None = None
        self._cell_links: list[str] | None = None
        self._link: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and self._table is None:
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell_text = []
            self._cell_links = []
        elif tag == "a" and self._cell_links is not None:
            self._link = attributes.get("href")

    def handle_data(self, data: str) -> None:
        if self._cell_text is not None:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            if self._link and self._cell_links is not None:
                self._cell_links.append(self._link)
            self._link = None
        elif tag in {"th", "td"} and self._cell_text is not None and self._cell_links is not None:
            if self._row is not None:
                text = " ".join("".join(self._cell_text).split())
                self._row.append((text, tuple(self._cell_links)))
            self._cell_text = None
            self._cell_links = None
        elif tag == "tr" and self._row is not None:
            if self._table is not None:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def _parse_checkpoint_rows(
    readme: str, *, maximum: int
) -> dict[str, tuple[str, str, str, str]]:
    parser = _TableParser()
    parser.feed(readme)
    target: list[list[tuple[str, tuple[str, ...]]]] | None = None
    header_row: list[tuple[str, tuple[str, ...]]] | None = None
    for table in parser.tables:
        for row in table:
            labels = {cell[0].casefold() for cell in row}
            if {"name", "checkpoint", "config"}.issubset(labels):
                target = table
                header_row = row
                break
        if target is not None:
            break
    if target is None or header_row is None:
        raise ValueError("Grounding DINO README checkpoint table was not recognized")
    headers = [text.casefold() for text, _ in header_row]
    output: dict[str, tuple[str, str, str, str]] = {}
    for row in target:
        cells = {
            headers[index]: (text, links)
            for index, (text, links) in enumerate(row)
            if index < len(headers)
        }
        if "name" not in cells or "checkpoint" not in cells or "config" not in cells:
            continue
        name = cells["name"][0]
        if name not in _MODELS:
            if any(_is_checkpoint_url(link) for link in cells["checkpoint"][1]):
                raise ValueError(
                    f"Grounding DINO README contains an unknown checkpoint row: {name}"
                )
            continue
        checkpoint_links = tuple(
            link for link in cells["checkpoint"][1] if _is_checkpoint_url(link)
        )
        config_links = tuple(link for link in cells["config"][1] if _is_config_url(link))
        metric = cells.get("box ap on coco", ("", ()))[0]
        if len(checkpoint_links) != 2 or len(config_links) != 1 or not metric:
            raise ValueError(f"Grounding DINO README has an incomplete checkpoint row: {name}")
        github = next((link for link in checkpoint_links if _release_coordinates(link)), None)
        mirror = next((link for link in checkpoint_links if _mirror_filename(link)), None)
        if not github or not mirror:
            raise ValueError(f"Grounding DINO README has no GitHub/HF pair for {name}")
        if name in output:
            raise ValueError(f"Grounding DINO README has duplicate checkpoint row: {name}")
        output[name] = (github, mirror, config_links[0], metric)
        if len(output) > maximum:
            raise ValueError("Grounding DINO README checkpoint table exceeds entry limit")
    return output


def _is_checkpoint_url(url: str) -> bool:
    return bool(_release_coordinates(url) or _mirror_filename(url))


def _is_config_url(url: str) -> bool:
    return _config_filename(url) is not None


def _release_coordinates(url: str) -> tuple[str, str] | None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "github.com" or parts.query or parts.fragment:
        return None
    match = re.fullmatch(
        r"/IDEA-Research/GroundingDINO/releases/download/([^/]+)/([^/]+\.pth)", parts.path
    )
    return (match.group(1), match.group(2)) if match else None


def _mirror_filename(url: str) -> str | None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "huggingface.co" or parts.query or parts.fragment:
        return None
    match = re.fullmatch(r"/ShilongLiu/GroundingDINO/resolve/main/([^/]+\.pth)", parts.path)
    return match.group(1) if match else None


def _config_filename(url: str) -> str | None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "github.com" or parts.query or parts.fragment:
        return None
    match = re.fullmatch(
        r"/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/([^/]+\.py)",
        parts.path,
    )
    return match.group(1) if match else None
