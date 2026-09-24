"""Read exact SatMAE checkpoint links from its first-party README tables."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpResponse
from modelome.models import SourcePage
from modelome.normalize import content_hash
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)

_TABLE_ROW = re.compile(r"^\|(?P<cells>.*)\|$")
_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https://zenodo\.org/record/[0-9]+/files/[A-Za-z0-9_.-]+)\)")
_EXPECTED: dict[tuple[str, str, str], str] = {
    ("fmow-nontemporal", "ViT-Large", "Pretrain"): "fmow_pretrain.pth",
    ("fmow-nontemporal", "ViT-Large", "Finetune"): "fmow_finetune.pth",
    ("fmow-temporal", "ViT-Large", "Pretrain"): "pretrain_fmow_temporal.pth",
    ("fmow-temporal", "ViT-Large", "Finetune"): "finetune_fmow_temporal.pth",
    ("fmow-sentinel", "ViT-Base (200 epochs)", "Pretrain"): "pretrain-vit-base-e199.pth",
    ("fmow-sentinel", "ViT-Base (200 epochs)", "Finetune"): "finetune-vit-base-e7.pth",
    ("fmow-sentinel", "ViT-Large (200 epochs)", "Pretrain"): "pretrain-vit-large-e199.pth",
    ("fmow-sentinel", "ViT-Large (200 epochs)", "Finetune"): "finetune-vit-large-e7.pth",
}


class SatMAECheckpointRegistrySourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Enumerate the eight exact Zenodo files linked by the SatMAE README."""

    coverage_limitation = (
        "Covers only the eight checkpoint file links in the first-party "
        "sustainlab-group/SatMAE README tables. Zenodo URLs are retained verbatim; "
        "records may overlap the generic Zenodo catalog, and this adapter does not "
        "claim unique artifact identity, resolve links, or download checkpoint bytes."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "sustainlab-satmae-checkpoint-registry")
        kwargs.setdefault("repository", "sustainlab-group/SatMAE")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", "README.md")
        kwargs.setdefault("provider_namespace", "satmae:checkpoint")
        kwargs.setdefault("max_entries", len(_EXPECTED))
        super().__init__(**kwargs)

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
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

        response: HttpResponse = self.client.get(
            self.raw_url(revision), headers={"Accept": "text/markdown,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds {self.max_response_bytes} bytes")
        checkpoints = _parse_readme(response.text(), self.name, self.source_path)
        records = tuple(self._record(item, revision, response.body) for item in checkpoints)
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
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
        return f"SatMAE checkpoint {handle}"

    def _record(self, checkpoint: _Checkpoint, revision: str, source: bytes):
        record = super()._record(checkpoint, revision, source)
        # Keep the source-declared Zenodo URL without implying it is a unique
        # binary identity or that this adapter verified the target's contents.
        releases = tuple(
            replace(
                release,
                metadata={
                    key: value for key, value in release.metadata.items() if key != "weight_url"
                }
                | {"checkpoint_file_url": checkpoint.url},
            )
            for release in record.releases
        )
        raw = dict(record.raw)
        raw["checkpoint_file_url"] = raw.pop("weight_url")
        return replace(record, raw=raw, releases=releases)


def _parse_readme(document: str, source: str, path: str) -> tuple[_Checkpoint, ...]:
    if document.count("## Temporal SatMAE") != 1 or document.count(
        "## Multi-Spectral SatMAE"
    ) != 1:
        raise ValueError(f"{source}: expected the two SatMAE model sections")
    context = ""
    found: dict[tuple[str, str, str], _Checkpoint] = {}
    for line_number, line in enumerate(document.splitlines(), 1):
        stripped = line.strip()
        if stripped == "#### fMoW Non-Temporal Checkpoints":
            context = "fmow-nontemporal"
            continue
        if stripped == "#### fMoW Temporal Checkpoints":
            context = "fmow-temporal"
            continue
        if stripped == "## Multi-Spectral SatMAE":
            context = "fmow-sentinel"
            continue
        row = _TABLE_ROW.fullmatch(stripped)
        if row is None:
            continue
        cells = [cell.strip() for cell in row.group("cells").split("|")]
        if len(cells) != 4 or cells[0] in {"Model", "---"} or set(cells[0]) <= {"-", ":"}:
            continue
        model_name = cells[0]
        if model_name not in {"ViT-Large", "ViT-Base (200 epochs)", "ViT-Large (200 epochs)"}:
            continue
        if context not in {"fmow-nontemporal", "fmow-temporal", "fmow-sentinel"}:
            continue
        for column, cell in (("Pretrain", cells[2]), ("Finetune", cells[3])):
            key = (context, model_name, column)
            expected = _EXPECTED.get(key)
            if expected is None:
                continue
            match = _LINK.search(cell)
            if match is None or match.group("label").casefold() != "download":
                raise ValueError(
                    f"{source}: missing exact {column.lower()} URL at line {line_number}"
                )
            url = match.group("url")
            filename = urlsplit(url).path.rsplit("/", 1)[-1]
            if filename != expected or key in found:
                raise ValueError(
                    f"{source}: unexpected or duplicate checkpoint URL at line {line_number}"
                )
            handle = f"{context}/{model_name.lower().replace(' ', '-')}/{column.lower()}-{filename}"
            found[key] = _Checkpoint(handle=handle, url=url, locator=f"{path}:line:{line_number}")
    if set(found) != set(_EXPECTED):
        raise ValueError(f"{source}: expected exactly eight SatMAE checkpoint rows")
    return tuple(found[key] for key in _EXPECTED)
