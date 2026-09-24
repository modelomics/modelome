"""Reader for DGL's first-party DGMG tutorial checkpoint example."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import PurePosixPath
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


class DGLCoreTutorialCheckpointSourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Extract DGL's literal DGMG cycle checkpoint URL from its tutorial source."""

    coverage_limitation = (
        "Covers only the direct checkpoint loaded by DGL's first-party DGMG tutorial. "
        "The tutorial identifies the DGMG model and says it generates cycles with "
        "10–20 nodes. This does not enumerate DGL-LifeSci models or other tutorials."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "dgl-core-dgmg-cycle-checkpoint")
        kwargs.setdefault("repository", "dmlc/dgl")
        kwargs.setdefault("branch", "master")
        kwargs.setdefault(
            "source_path", "tutorials/models/3_generative_model/5_dgmg.py"
        )
        kwargs.setdefault("provider_namespace", "dgl:core-checkpoint")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "dgl-core-dgmg-tutorial-checkpoint-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "one literal load_url URL paired with DGMG source class/call",
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

        response: HttpResponse = self.client.get(
            self.raw_url(revision),
            headers={"Accept": "text/x-python,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: tutorial source returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: tutorial source exceeds {self.max_response_bytes} bytes"
            )
        url = _extract_dgmg_checkpoint(response.text(), self.name, self.source_path)
        filename = PurePosixPath(urlsplit(url).path).name
        checkpoint = _Checkpoint(
            handle=filename,
            url=url,
            locator=f"{self.source_path}:DGMG tutorial checkpoint",
        )
        record = self._record(checkpoint, revision, response.body)
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": 1,
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=(record,),
            next_state=next_state,
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _model_name(self, handle: str) -> str:
        return "DGL DGMG cycles (10–20 nodes)"


def _extract_dgmg_checkpoint(document: str, source: str, path: str) -> str:
    try:
        module = ast.parse(document, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{source}: tutorial source is not valid Python: {error.msg}") from error

    has_dgmg_class = any(
        isinstance(node, ast.ClassDef) and node.name == "DGMG" for node in module.body
    )
    has_dgmg_construction = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DGMG"
        for node in ast.walk(module)
    )
    urls = [
        argument.value
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_url"
        for argument in node.args[:1]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    ]
    if not has_dgmg_class or not has_dgmg_construction or len(urls) != 1:
        raise ValueError(f"{source}: expected DGMG class/construction and one literal load_url")
    parsed = urlsplit(urls[0])
    if (
        parsed.scheme != "https"
        or parsed.hostname != "data.dgl.ai"
        or not parsed.path.startswith("/model/")
        or not parsed.path.casefold().endswith(".pth")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{source}: expected one direct DGL .pth model checkpoint URL")
    return urls[0]
