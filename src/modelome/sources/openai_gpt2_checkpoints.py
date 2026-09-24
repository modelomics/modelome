"""Enumerate original OpenAI GPT-2 TensorFlow checkpoint data files.

The archived first-party downloader declares the Azure Blob artifact root and
checkpoint filename. Its developer guide declares the four released model
sizes. This adapter records only the TensorFlow weight shard URL and never
downloads model bytes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

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

_MODEL_COMMAND = re.compile(r"python3\s+download_model\.py\s+(124M|355M|774M|1558M)\b")
_ARTIFACT_ROOT = re.compile(
    r'requests\.get\("(?P<root>https://[^/" ]+(?:/[^/" ]+)*)/"'
    r'\s*\+\s*subdir\s*\+\s*"/"'
)
_WEIGHT_FILE = "model.ckpt.data-00000-of-00001"


class OpenAIGPT2CheckpointSourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Read model sizes and the weight URL template from pinned OpenAI files."""

    coverage_limitation = (
        "Covers the four GPT-2 sizes declared in the archived OpenAI developer "
        "guide and only the exact TensorFlow checkpoint data shard declared by "
        "the first-party downloader. Companion index/meta and tokenizer files "
        "are not emitted as separate checkpoints."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "openai-gpt2-checkpoints")
        kwargs.setdefault("repository", "openai/gpt-2")
        kwargs.setdefault("branch", "master")
        kwargs.setdefault("source_path", "download_model.py")
        kwargs.setdefault("provider_namespace", "openai:gpt2-checkpoint")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openai-gpt2-checkpoints-v1",
                "repository": self.repository,
                "branch": self.branch,
                "paths": ["DEVELOPERS.md", "download_model.py"],
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "four documented model IDs + literal official Azure blob template",
            }
        )

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

        guide = self._get_text(revision, "DEVELOPERS.md")
        downloader = self._get_text(revision, "download_model.py")
        models = tuple(dict.fromkeys(_MODEL_COMMAND.findall(guide)))
        root_match = _ARTIFACT_ROOT.search(downloader)
        if not models or root_match is None or _WEIGHT_FILE not in downloader:
            raise ValueError(f"{self.name}: first-party GPT-2 artifact declarations changed")
        if len(models) > self.max_entries:
            raise ValueError(f"{self.name}: model list exceeds {self.max_entries} entries")
        root = root_match.group("root").rstrip("/")
        if root != "https://openaipublic.blob.core.windows.net/gpt-2":
            raise ValueError(f"{self.name}: unexpected GPT-2 artifact root")
        records = tuple(
            self._record(
                _Checkpoint(
                    handle=model,
                    url=f"{root}/models/{model}/model.ckpt.data-00000-of-00001",
                    locator=f"DEVELOPERS.md:download_model.py:{model}",
                ),
                revision,
                downloader.encode() + b"\0" + guide.encode(),
            )
            for model in models
        )
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "document_url": self.raw_url(revision),
            "document_sha256": content_hash(downloader.encode() + b"\0" + guide.encode()),
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

    def _get_text(self, revision: str, path: str) -> str:
        response: HttpResponse = self.client.get(
            f"https://raw.githubusercontent.com/{self.repository}/{revision}/{path}",
            headers={"Accept": "text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: {path} returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: {path} exceeds {self.max_response_bytes} bytes")
        return response.text()


__all__ = ["OpenAIGPT2CheckpointSourceAdapter"]
