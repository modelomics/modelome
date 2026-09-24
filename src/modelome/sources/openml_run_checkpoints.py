"""Read explicit ONNX checkpoint artifacts attached to OpenML runs.

OpenML's TensorFlow integration can attach a trained ``model.onnx`` file to a
run under the ``onnx_model`` output key. This adapter accepts caller-supplied
run IDs and resolves only that exact artifact reference. It does not sweep the
full run catalog or download artifact bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


class OpenMLRunOnnxCheckpointSourceAdapter:
    """Emit run-scoped models only when OpenML declares an ONNX model file."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only caller-supplied OpenML run IDs whose run metadata declares an "
        "`onnx_model` output file. It does not enumerate all OpenML runs, infer "
        "checkpoint files from predictions, or download binaries."
    )

    def __init__(
        self,
        *,
        run_ids: Sequence[str | int],
        name: str = "openml-run-onnx-checkpoints",
        url: str = "https://www.openml.org/api/v1/json/run",
        artifact_base_url: str = "https://api.openml.org/data/download",
        max_response_bytes: int = 4 * 1024 * 1024,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must be non-empty")
        if not url.startswith("https://") or not artifact_base_url.startswith("https://"):
            raise ValueError("OpenML URLs must use HTTPS")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        ids = tuple(_numeric_id(value, "run ID") for value in run_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("run_ids must be a non-empty sequence of unique IDs")
        self.name = name
        self.url = url.rstrip("/")
        self.artifact_base_url = artifact_base_url.rstrip("/")
        self.run_ids = ids
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash({
            "adapter": "openml-run-onnx-checkpoint-v1",
            "run_ids": ids,
            "url": self.url,
            "artifact_base_url": self.artifact_base_url,
            "max_response_bytes": max_response_bytes,
            "admission": "exact onnx_model output-file reference",
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        index = state.get("index", 0)
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index <= len(self.run_ids)
        ):
            raise ValueError(f"{self.name}: invalid cursor")
        if index == len(self.run_ids):
            return SourcePage(records=(), next_state=dict(state), complete=True)

        run_id = self.run_ids[index]
        response: HttpResponse = self.client.get(
            f"{self.url}/{quote(run_id, safe='')}",
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: run {run_id} returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: run response exceeds {self.max_response_bytes} bytes")
        payload = response.json()
        run = _run_detail(payload, self.name)
        record = self._record(run_id, run)
        next_state = {"index": index + 1}
        return SourcePage(
            records=() if record is None else (record,),
            next_state=next_state,
            complete=index + 1 == len(self.run_ids),
            upstream_count=None,
            authoritative_snapshot=False,
        )

    def _record(self, run_id: str, run: Mapping[str, Any]) -> SourceRecord | None:
        file_id = _onnx_file_id(run, self.name)
        if file_id is None:
            return None
        file_id = _numeric_id(file_id, "ONNX file ID")
        flow_id = _optional_id(run.get("flow_id"), "flow ID")
        flow_name = _text(run.get("flow_name")) or _text(run.get("name"))
        model_name = flow_name or (
            f"OpenML flow {flow_id}" if flow_id else f"OpenML run {run_id} model"
        )
        model_local_id = f"model:{run_id}"
        weights_url = f"{self.artifact_base_url}/{quote(file_id, safe='')}/model.onnx"
        run_url = f"https://www.openml.org/r/{quote(run_id, safe='')}"
        model = ModelHint(
            local_id=model_local_id,
            name=model_name,
            identifiers=(Identifier("openml:trained-run-model", run_id),),
            status=ModelStatus.RELEASED,
            locator=run_url,
        )
        release = ReleaseHint(
            local_id=f"release:{run_id}",
            model_local_id=model_local_id,
            revision=run_id,
            identifiers=(Identifier("openml:file", file_id),),
            metadata={"format": "onnx", "run_id": run_id, "flow_id": flow_id},
        )
        links = [
            Link(run_url, relation="source_run", crawl=False, model_local_ids=(model_local_id,)),
            Link(weights_url, relation="weights", crawl=False, model_local_ids=(model_local_id,)),
        ]
        if flow_id:
            links.append(Link(
                f"https://www.openml.org/f/{quote(flow_id, safe='')}",
                relation="source_implementation",
                crawl=False,
                model_local_ids=(model_local_id,),
            ))
        return SourceRecord(
            source_record_id=f"openml-run-onnx:{run_id}:{file_id}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(weights_url),
            title=f"{model_name} (OpenML run {run_id}, ONNX)",
            raw=dict(run),
            identifiers=(Identifier("openml:run", run_id), Identifier("openml:file", file_id)),
            links=tuple(links),
            models=(model,),
            releases=(release,),
        )


def _run_detail(payload: Any, source: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: run response must be a JSON object")
    run = payload.get("run", payload)
    if isinstance(run, list) and len(run) == 1:
        run = run[0]
    if not isinstance(run, Mapping):
        raise ValueError(f"{source}: response does not contain a run record")
    return run


def _onnx_file_id(run: Mapping[str, Any], source: str) -> str | None:
    outputs: Any = run.get("output_files")
    if outputs is None:
        outputs = run.get("outputfile", run.get("outputfiles"))
    if isinstance(outputs, Mapping):
        candidate = outputs.get("onnx_model") or outputs.get("model.onnx")
        if isinstance(candidate, Mapping):
            candidate = candidate.get("file_id", candidate.get("id"))
        if candidate is None:
            return None
        return _text(candidate)
    if isinstance(outputs, list):
        for output in outputs:
            if not isinstance(output, Mapping):
                continue
            filename = _text(output.get("file_name", output.get("name"))).lower()
            key = _text(output.get("name", output.get("key"))).lower()
            if filename == "model.onnx" or key == "onnx_model":
                return _text(output.get("file_id", output.get("id")))
        return None
    if outputs is not None:
        raise ValueError(f"{source}: run output-files field is malformed")
    return None


def _numeric_id(value: Any, label: str) -> str:
    text = _text(value)
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f"invalid {label}")
    return str(int(text))


def _optional_id(value: Any, label: str) -> str:
    return "" if value is None or _text(value) == "" else _numeric_id(value, label)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


__all__ = ["OpenMLRunOnnxCheckpointSourceAdapter"]
