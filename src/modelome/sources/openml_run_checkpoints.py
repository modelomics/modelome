"""Read explicit ONNX checkpoint artifacts attached to OpenML runs.

OpenML's TensorFlow integration can attach a trained ``model.onnx`` file to a
run under the ``onnx_model`` output key. This adapter accepts caller-supplied
run IDs, or optionally performs a complete but deliberately slow paged census
of run metadata, and resolves only exact artifact references. It does not
download artifact bytes.
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
        return _record_for_run(run_id, run, self.artifact_base_url, self.name)


class OpenMLRunOnnxCensusSourceAdapter(OpenMLRunOnnxCheckpointSourceAdapter):
    """Exhaustively page run IDs, then inspect each run's metadata for ONNX output.

    Each ``fetch_page`` performs at most one upstream request. A run-list page
    is retained in checkpoint state while its IDs are inspected one-by-one,
    allowing the crawl to resume without replaying completed details.
    """

    coverage_limitation = (
        "Covers public OpenML runs enumerated through paged run/list results and "
        "one run-detail lookup per ID. Run/list does not include output files, so "
        "this exhaustive census is intentionally much slower than an ID-scoped scan. "
        "Only exact onnx_model/model.onnx output references are admitted; binaries "
        "are not downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "openml-run-onnx-census",
        list_url: str = "https://www.openml.org/api/v1/json/run/list",
        detail_url: str = "https://www.openml.org/api/v1/json/run",
        page_size: int = 100,
        **kwargs: Any,
    ) -> None:
        if not name.strip() or not list_url.startswith("https://"):
            raise ValueError("source name must be non-empty and list URL must use HTTPS")
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be from 1 through 1000")
        # Initialize shared response handling and record construction; census
        # mode does not use this placeholder ID list.
        super().__init__(run_ids=[1], name=name, url=detail_url, **kwargs)
        self.list_url = list_url.rstrip("/")
        self.page_size = page_size
        self.checkpoint_signature = content_hash({
            "adapter": "openml-run-onnx-census-v1",
            "list_url": self.list_url,
            "detail_url": self.url,
            "artifact_base_url": self.artifact_base_url,
            "page_size": page_size,
            "max_response_bytes": self.max_response_bytes,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        offset = _cursor(state.get("offset", 0), "offset", self.name)
        run_ids = state.get("run_ids", [])
        detail_index = _cursor(state.get("detail_index", 0), "detail_index", self.name)
        listing_complete = state.get("listing_complete", False)
        if not isinstance(listing_complete, bool) or not isinstance(run_ids, list):
            raise ValueError(f"{self.name}: invalid checkpoint state")
        if run_ids:
            ids = tuple(_numeric_id(value, "run ID") for value in run_ids)
            if len(ids) != len(set(ids)) or detail_index > len(ids):
                raise ValueError(f"{self.name}: invalid run-ID page checkpoint")
            if detail_index < len(ids):
                run_id = ids[detail_index]
                response = self.client.get(
                    f"{self.url}/{quote(run_id, safe='')}",
                    headers={"Accept": "application/json"},
                )
                if response.status != 200:
                    raise ValueError(f"{self.name}: run {run_id} returned HTTP {response.status}")
                self._check_body(response, "run")
                record = self._record(run_id, _run_detail(response.json(), self.name))
                next_index = detail_index + 1
                next_state = {
                    "offset": offset,
                    "run_ids": list(ids),
                    "detail_index": next_index,
                    "listing_complete": listing_complete,
                }
                complete = listing_complete and next_index == len(ids)
                return SourcePage(
                    records=() if record is None else (record,),
                    next_state=next_state,
                    complete=complete,
                    upstream_count=None,
                    authoritative_snapshot=False,
                )

        if listing_complete:
            return SourcePage(records=(), next_state=dict(state), complete=True)
        response = self.client.get(
            f"{self.list_url}/limit/{self.page_size}/offset/{offset}",
            headers={"Accept": "application/json"},
        )
        self._check_body(response, "run-list")
        payload = response.json()
        if response.status != 200 and not _is_no_runs_response(payload):
            raise ValueError(f"{self.name}: run list returned HTTP {response.status}")
        ids = () if _is_no_runs_response(payload) else _run_ids(payload, self.name)
        next_offset = offset + len(ids)
        exhausted = len(ids) < self.page_size
        next_state = {
            "offset": next_offset,
            "run_ids": list(ids),
            "detail_index": 0,
            "listing_complete": exhausted,
        }
        return SourcePage(
            records=(),
            next_state=next_state,
            complete=exhausted and not ids,
            upstream_count=None,
            authoritative_snapshot=False,
        )

    def _check_body(self, response: HttpResponse, label: str) -> None:
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: {label} response exceeds "
                f"{self.max_response_bytes} bytes"
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


def _run_ids(payload: Any, source: str) -> tuple[str, ...]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: run-list response must be a JSON object")
    if _is_no_runs_response(payload):
        return ()
    runs = payload.get("runs", payload)
    if isinstance(runs, Mapping):
        runs = runs.get("run", runs.get("runs"))
    if runs is None:
        return ()
    if isinstance(runs, Mapping):
        runs = [runs]
    if not isinstance(runs, list):
        raise ValueError(f"{source}: response does not contain a run list")
    ids: list[str] = []
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError(f"{source}: run-list item must be a JSON object")
        ids.append(_numeric_id(run.get("run_id", run.get("id")), "run ID"))
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source}: run-list page contains duplicate IDs")
    return tuple(ids)


def _is_no_runs_response(payload: Any) -> bool:
    """OpenML's documented run-list no-result error uses code 512."""
    if not isinstance(payload, Mapping):
        return False
    for key in ("error", "detail"):
        error = payload.get(key)
        if isinstance(error, Mapping):
            try:
                if int(error.get("code", -1)) == 512:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _cursor(value: Any, label: str, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{source}: invalid {label} checkpoint")
    return value


def _record_for_run(
    run_id: str,
    run: Mapping[str, Any],
    artifact_base_url: str,
    source: str,
) -> SourceRecord | None:
    file_id = _onnx_file_id(run, source)
    if file_id is None:
        return None
    file_id = _numeric_id(file_id, "ONNX file ID")
    flow_id = _optional_id(run.get("flow_id"), "flow ID")
    flow_name = _text(run.get("flow_name")) or _text(run.get("name"))
    model_name = flow_name or (
        f"OpenML flow {flow_id}" if flow_id else f"OpenML run {run_id} model"
    )
    model_local_id = f"model:{run_id}"
    weights_url = f"{artifact_base_url}/{quote(file_id, safe='')}/model.onnx"
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


__all__ = [
    "OpenMLRunOnnxCheckpointSourceAdapter",
    "OpenMLRunOnnxCensusSourceAdapter",
]
