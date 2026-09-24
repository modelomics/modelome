"""PMT's first-party pretrained checkpoint table and published asset mirror."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

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

_REPOSITORY = "Mondo-Robotics/PMT"
_DOCUMENT = "checkpoints/pretrained/README.md"
_ASSET_REPO = "aCodeDog/PMT-assets"
_ASSET_PREFIX = "checkpoints/pretrained/"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*$")
_FILE = re.compile(r"^[A-Za-z0-9_.+-]+\.pt$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PMTPretrainedCheckpointSourceAdapter:
    """Index PMT's exact `.pt` and official SONIC ONNX release assets."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the `.pt` checkpoints and SONIC ONNX components named by the "
        "first-party PMT pretrained README, when those exact files exist in the "
        "linked PMT-assets Hugging Face dataset. It does not fetch weights or "
        "index motion, terrain, or other assets."
    )

    def __init__(
        self,
        *,
        name: str = "pmt-pretrained-checkpoints",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 32,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        for value, label in (
            (max_response_bytes, "max_response_bytes"),
            (max_entries, "max_entries"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "pmt-pretrained-checkpoints-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "asset_repository": _ASSET_REPO,
                "asset_prefix": _ASSET_PREFIX,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "source README table rows joined to exact published files",
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{_REPOSITORY}"

    @property
    def asset_repository_url(self) -> str:
        return f"https://huggingface.co/datasets/{_ASSET_REPO}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        repo_revision = self._github_revision()
        document_url = (
            f"https://raw.githubusercontent.com/{_REPOSITORY}/{repo_revision}/"
            f"{quote(_DOCUMENT, safe='/')}"
        )
        doc_response: HttpResponse = self.client.get(
            document_url, headers={"Accept": "text/markdown,text/plain"}
        )
        self._require_response(doc_response, "first-party checkpoint README")
        entries = _parse_inventory(doc_response.text(), source=self.name, maximum=self.max_entries)
        if not entries:
            raise ValueError(f"{self.name}: first-party table contains no checkpoints")

        asset_revision, files, list_bodies = self._asset_files()
        records: list[SourceRecord] = []
        for entry in entries:
            if entry[0] == "sonic_onnx/":
                sonic_files = self._match_sonic_files(files)
                records.append(
                    self._sonic_record(
                        entry,
                        sonic_files,
                        repo_revision,
                        asset_revision,
                        doc_response.body,
                        document_url,
                    )
                )
                continue
            filename = entry[0]
            path = f"{_ASSET_PREFIX}{filename}"
            item = files.get(path)
            if item is None:
                raise ValueError(
                    f"{self.name}: source-listed checkpoint is missing from asset repo: {path}"
                )
            records.append(
                self._checkpoint_record(
                    entry,
                    item,
                    repo_revision,
                    asset_revision,
                    doc_response.body,
                    document_url,
                )
            )
        if len(records) > self.max_entries:
            raise ValueError(
                f"{self.name}: checkpoint inventory exceeds {self.max_entries} entries"
            )
        checked_at = self.clock()
        if checked_at.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return SourcePage(
            records=tuple(records),
            next_state={
                "checked_at": checked_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "source_revision": repo_revision,
                "asset_revision": asset_revision,
                "document_sha256": content_hash(doc_response.body),
                "asset_listing_sha256": content_hash(b"\n".join(list_bodies)),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _github_revision(self) -> str:
        response: HttpResponse = self.client.get(
            f"https://api.github.com/repos/{_REPOSITORY}/commits/main",
            headers={"Accept": "application/vnd.github+json"},
        )
        self._require_response(response, "GitHub commit endpoint")
        payload = response.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: GitHub did not return a full commit SHA")
        return revision

    def _asset_files(self) -> tuple[str, dict[str, dict[str, Any]], tuple[bytes, ...]]:
        info_response: HttpResponse = self.client.get(
            f"https://huggingface.co/api/datasets/{_ASSET_REPO}",
            headers={"Accept": "application/json"},
        )
        self._require_response(info_response, "HF asset repository metadata")
        info = info_response.json()
        revision = info.get("sha", "") if isinstance(info, Mapping) else ""
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: HF asset repo did not return a full commit SHA")
        response_bodies: list[bytes] = []
        files: dict[str, dict[str, Any]] = {}
        for relative_dir in ("checkpoints/pretrained", "checkpoints/pretrained/sonic_onnx"):
            endpoint = (
                f"https://huggingface.co/api/datasets/{_ASSET_REPO}/tree/{revision}/{relative_dir}"
            )
            response: HttpResponse = self.client.get(
                endpoint,
                params={"recursive": "false", "expand": "false"},
                headers={"Accept": "application/json"},
            )
            self._require_response(response, f"HF asset listing {relative_dir}")
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"{self.name}: HF asset listing is not a list")
            response_bodies.append(response.body)
            for item in payload:
                if not isinstance(item, Mapping) or item.get("type") != "file":
                    continue
                path = item.get("path")
                if not isinstance(path, str) or not path.startswith(_ASSET_PREFIX):
                    continue
                if not _SAFE_PATH.fullmatch(path):
                    raise ValueError(f"{self.name}: invalid asset path in HF listing")
                oid = item.get("oid")
                size = item.get("size")
                if not isinstance(oid, str) or not oid.strip():
                    raise ValueError(f"{self.name}: HF asset file has no object identifier")
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise ValueError(f"{self.name}: HF asset file has invalid size")
                files[path] = {"path": path, "oid": oid, "size": size}
                if len(files) > self.max_entries * 2:
                    raise ValueError(f"{self.name}: asset subtree exceeds file limit")
        return revision, files, tuple(response_bodies)

    def _checkpoint_record(
        self,
        entry: tuple[str, str, str, str, str, int],
        item: dict[str, Any],
        repo_revision: str,
        asset_revision: str,
        document: bytes,
        document_url: str,
    ) -> SourceRecord:
        filename, task_id, network, iteration, reward, line = entry
        model_local_id = f"model:{_slug(task_id)}"
        file_url = self._asset_file_url(asset_revision, item["path"])
        locator = f"{_DOCUMENT}:line:{line}"
        model_id = Identifier("pmt:policy", task_id)
        metadata = {
            "repository": _REPOSITORY,
            "source_revision": repo_revision,
            "document_path": _DOCUMENT,
            "asset_repository": _ASSET_REPO,
            "asset_revision": asset_revision,
            "checkpoint_path": item["path"],
            "asset_oid": item["oid"],
            "size_bytes": item["size"],
            "task_or_gym_id": task_id,
            "network": network,
            "iteration": iteration,
            "reward": reward,
            "source_document_sha256": content_hash(document),
        }
        model = ModelHint(
            local_id=model_local_id,
            name=task_id,
            identifiers=(model_id,),
            aliases=(filename,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{_slug(filename)}",
            model_local_id=model_local_id,
            revision=asset_revision,
            identifiers=(Identifier("pmt:checkpoint", item["path"]),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"pmt:{_slug(filename)}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(file_url),
            title=f"PMT {task_id}",
            raw=metadata,
            text=f"{task_id}; {filename}; {network}; iteration {iteration}; reward {reward}",
            identifiers=(model_id,),
            links=(
                Link(
                    file_url,
                    relation="model_artifact",
                    locator=locator,
                    crawl=False,
                    model_local_ids=(model_local_id,),
                ),
                Link(self.asset_repository_url, relation="asset_repository", crawl=False),
                Link(document_url, relation="model_catalog", locator=locator, crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )

    def _sonic_record(
        self,
        entry: tuple[str, str, str, str, str, int],
        files: tuple[dict[str, Any], ...],
        repo_revision: str,
        asset_revision: str,
        document: bytes,
        document_url: str,
    ) -> SourceRecord:
        _, task_id, network, iteration, reward, line = entry
        names = {item["path"].rsplit("/", 1)[-1] for item in files}
        required = {"model_encoder.onnx", "model_decoder.onnx"}
        if not required.issubset(names):
            raise ValueError(f"{self.name}: published SONIC ONNX encoder/decoder are missing")
        model_local_id = f"model:{_slug(task_id)}"
        locator = f"{_DOCUMENT}:line:{line}"
        model_id = Identifier("pmt:policy", task_id)
        artifact_links = tuple(
            Link(
                self._asset_file_url(asset_revision, item["path"]),
                relation="model_artifact",
                locator=item["path"],
                crawl=False,
                model_local_ids=(model_local_id,),
            )
            for item in files
        )
        metadata = {
            "repository": _REPOSITORY,
            "source_revision": repo_revision,
            "document_path": _DOCUMENT,
            "asset_repository": _ASSET_REPO,
            "asset_revision": asset_revision,
            "checkpoint_path": "checkpoints/pretrained/sonic_onnx/",
            "components": [
                {"path": item["path"], "oid": item["oid"], "size_bytes": item["size"]}
                for item in files
            ],
            "task_or_gym_id": task_id,
            "network": network,
            "iteration": iteration,
            "reward": reward,
            "source_document_sha256": content_hash(document),
        }
        model = ModelHint(
            local_id=model_local_id,
            name=task_id,
            identifiers=(model_id,),
            aliases=("SONIC", "PMT-SONIC"),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{_slug(task_id)}",
            model_local_id=model_local_id,
            revision=asset_revision,
            identifiers=(Identifier("pmt:checkpoint", "checkpoints/pretrained/sonic_onnx/"),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"pmt:{_slug(task_id)}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(self.asset_repository_url),
            title=f"PMT {task_id} (SONIC ONNX)",
            raw=metadata,
            text=f"{task_id}; official SONIC ONNX encoder and decoder",
            identifiers=(model_id,),
            links=(
                *artifact_links,
                Link(self.asset_repository_url, relation="asset_repository", crawl=False),
                Link(document_url, relation="model_catalog", locator=locator, crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )

    def _match_sonic_files(self, files: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        prefix = f"{_ASSET_PREFIX}sonic_onnx/"
        selected = tuple(
            files[key]
            for key in sorted(files)
            if key.startswith(prefix)
            and key.rsplit("/", 1)[-1]
            in {
                "model_encoder.onnx",
                "model_decoder.onnx",
                "observation_config_sonic_release.yaml",
            }
        )
        if len(selected) < 2:
            raise ValueError(f"{self.name}: no published SONIC ONNX files found")
        return selected

    def _asset_file_url(self, revision: str, path: str) -> str:
        return (
            f"https://huggingface.co/datasets/{_ASSET_REPO}/resolve/{revision}/"
            f"{quote(path, safe='/')}"
        )

    def _require_response(self, response: HttpResponse, label: str) -> None:
        if response.status != 200:
            raise ValueError(f"{self.name}: {label} returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: {label} exceeds {self.max_response_bytes} bytes")


def _parse_inventory(
    document: str,
    *,
    source: str,
    maximum: int,
) -> tuple[tuple[str, str, str, str, str, int], ...]:
    active = False
    entries: dict[str, tuple[str, str, str, str, str, int]] = {}
    for line_number, line in enumerate(document.splitlines(), start=1):
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            continue
        if [cell.casefold().strip("` ") for cell in cells[:2]] == ["file", "task / gym id"]:
            active = True
            continue
        if not active:
            continue
        if all(set(cell) <= {"-", ":", " ", "`"} for cell in cells):
            continue
        filename = cells[0].strip("` ")
        task_id = cells[1].strip("` ")
        if filename != "sonic_onnx/" and not _FILE.fullmatch(filename):
            continue
        if not task_id or filename in entries:
            raise ValueError(f"{source}: invalid or duplicate checkpoint row on line {line_number}")
        entry = (
            filename,
            task_id,
            cells[2].strip("` "),
            cells[3].strip("` "),
            cells[4],
            line_number,
        )
        entries[filename] = entry
        if len(entries) > maximum:
            raise ValueError(f"{source}: pretrained table exceeds {maximum} entries")
    return tuple(entries.values())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
