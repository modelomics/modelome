"""OpenML's public registry of versioned machine-learning flows.

OpenML flows describe serialized implementations of machine-learning
algorithms. This adapter records each flow/version as registry evidence; it
does not instantiate flows, fetch run artifacts, or claim trained weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit

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


class OpenMLFlowRegistrySourceAdapter:
    """Page through public OpenML flows using its documented limit/offset API."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers public OpenML flow (algorithm implementation) records returned by its "
        "paged registry API. It is not a census of trained checkpoints or model runs, "
        "and it does not execute implementations or download artifacts."
    )

    def __init__(
        self,
        *,
        name: str = "openml-flows",
        url: str = "https://www.openml.org/api/v1/json/flow/list",
        page_size: int = 100,
        max_response_bytes: int = 8 * 1024 * 1024,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must be non-empty")
        if page_size < 1 or max_response_bytes < 1:
            raise ValueError("page_size and max_response_bytes must be positive")
        if not url.startswith("https://"):
            raise ValueError("OpenML API URL must use HTTPS")
        self.name = name
        self.url = url.rstrip("/")
        self.page_size = page_size
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openml-flow-registry-v1",
                "url": self.url,
                "page_size": page_size,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        offset = state.get("offset", 0)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError(f"{self.name}: invalid offset checkpoint")
        response: HttpResponse = self.client.get(
            f"{self.url}/limit/{self.page_size}/offset/{offset}",
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: response exceeds {self.max_response_bytes} bytes")
        payload = response.json()
        flows = _flows(payload, self.name)
        records = tuple(self._record(flow) for flow in flows)
        complete = len(flows) < self.page_size
        next_state = {"offset": offset + len(flows)}
        if complete:
            next_state["completed"] = True
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=complete,
            upstream_count=None,
            authoritative_snapshot=False,
        )

    def _record(self, flow: Mapping[str, Any]) -> SourceRecord:
        flow_id = _first_text(flow, "id", "flow_id", "fid")
        name = _first_text(flow, "full_name", "name", "flow_name")
        if not flow_id or not name:
            raise ValueError(f"{self.name}: flow row lacks an ID or name")
        flow_id = quote(flow_id, safe="")
        flow_url = f"https://www.openml.org/f/{flow_id}"
        version = _first_text(flow, "external_version", "version") or None
        model_id = f"model:{flow_id}"
        model = ModelHint(
            local_id=model_id,
            name=name,
            aliases=tuple(
                v
                for v in (_text(flow.get("name")), _text(flow.get("full_name")))
                if v and v != name
            ),
            identifiers=(Identifier("openml:flow", flow_id),),
            status=ModelStatus.DOCUMENTED,
        )
        release = ReleaseHint(
            local_id=f"release:{flow_id}",
            model_local_id=model_id,
            version=version,
            identifiers=(Identifier("openml:flow-release", flow_id),),
            metadata={"flow_id": flow_id, "uploader": _first_text(flow, "uploader")},
        )
        links = [
            Link(flow_url, relation="model_card", crawl=False, model_local_ids=(model_id,)),
            Link(
                "https://www.openml.org/",
                relation="registry",
                crawl=False,
                model_local_ids=(model_id,),
            ),
        ]
        source_url = _optional_http_url(flow.get("source_url"))
        if source_url:
            links.append(
                Link(
                    source_url,
                    relation="official_implementation",
                    locator="$.source_url",
                    crawl=False,
                    model_local_ids=(model_id,),
                )
            )
        binary_url = _optional_http_url(flow.get("binary_url"))
        if binary_url:
            links.append(
                Link(
                    binary_url,
                    relation="serialized_implementation",
                    locator="$.binary_url",
                    crawl=False,
                    model_local_ids=(model_id,),
                )
            )
        return SourceRecord(
            source_record_id=f"openml-flow:{flow_id}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(flow_url),
            title=name,
            raw=dict(flow),
            identifiers=(Identifier("openml:flow", flow_id),),
            links=tuple(links),
            models=(model,),
            releases=(release,),
        )


class OpenMLRunRegistrySourceAdapter:
    """Page over OpenML experiment-run records without treating them as models.

    Runs identify one evaluation of a flow on a task/dataset. The v1 run
    metadata standard allows description, predictions, and optional traces; it
    does not expose a reusable trained checkpoint as a standard run artifact.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers public OpenML experiment runs returned by the paged run/list API. "
        "Records exact run, flow, task, and dataset references. They are experiment "
        "evidence, not model entities or pretrained checkpoints; predictions and "
        "traces are not downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "openml-runs",
        url: str = "https://www.openml.org/api/v1/json/run/list",
        page_size: int = 100,
        max_response_bytes: int = 16 * 1024 * 1024,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must be non-empty")
        if page_size < 1 or max_response_bytes < 1:
            raise ValueError("page_size and max_response_bytes must be positive")
        if not url.startswith("https://"):
            raise ValueError("OpenML API URL must use HTTPS")
        self.name = name
        self.url = url.rstrip("/")
        self.page_size = page_size
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash({
            "adapter": "openml-run-registry-v1",
            "url": self.url,
            "page_size": page_size,
            "max_response_bytes": max_response_bytes,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        offset = state.get("offset", 0)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError(f"{self.name}: invalid offset checkpoint")
        response: HttpResponse = self.client.get(
            f"{self.url}/limit/{self.page_size}/offset/{offset}",
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: response exceeds {self.max_response_bytes} bytes")
        runs = _runs(response.json(), self.name)
        records = tuple(self._record(run) for run in runs)
        complete = len(runs) < self.page_size
        next_state: dict[str, Any] = {"offset": offset + len(runs)}
        if complete:
            next_state["completed"] = True
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=complete,
            upstream_count=None,
            authoritative_snapshot=False,
        )

    def _record(self, run: Mapping[str, Any]) -> SourceRecord:
        run_id = _first_text(run, "id", "run_id")
        if not run_id.isdigit() or int(run_id) < 1:
            raise ValueError(f"{self.name}: run row lacks a valid numeric ID")
        run_id = str(int(run_id))
        run_url = f"https://www.openml.org/r/{quote(run_id, safe='')}"
        flow_id = _optional_numeric_id(run, "flow_id")
        task_id = _optional_numeric_id(run, "task_id")
        dataset_id = _optional_numeric_id(run, "dataset_id", "did")
        flow_name = _first_text(run, "flow_name", "name")
        title = f"OpenML run {run_id}" + (f": {flow_name}" if flow_name else "")
        links = [Link(run_url, relation="experiment_run", crawl=False)]
        if flow_id:
            links.append(Link(
                f"https://www.openml.org/f/{quote(flow_id, safe='')}",
                relation="evaluated_implementation",
                crawl=False,
            ))
        if task_id:
            links.append(Link(
                f"https://www.openml.org/t/{quote(task_id, safe='')}",
                relation="evaluation_task",
                crawl=False,
            ))
        if dataset_id:
            links.append(Link(
                f"https://www.openml.org/d/{quote(dataset_id, safe='')}",
                relation="evaluation_dataset",
                crawl=False,
            ))
        identifiers = [Identifier("openml:run", run_id)]
        return SourceRecord(
            source_record_id=f"openml-run:{run_id}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(run_url),
            title=title,
            raw=dict(run),
            identifiers=tuple(identifiers),
            links=tuple(links),
        )


def _flows(payload: Any, source: str) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: response must be a JSON object")
    value: Any = payload.get("flows", payload.get("flow"))
    if isinstance(value, Mapping):
        # OpenML's JSON API wraps flow rows in `flows.flow`; JSON serializers
        # can omit a repeated child when a valid list is empty.
        if "flow" not in value and "flows" not in value:
            namespace_only = all(str(key).startswith("@xmlns") for key in value)
            if namespace_only:
                return []
            raise ValueError(f"{source}: response does not contain a flow list")
        value = value.get("flow", value.get("flows"))
    if value is None and ("flows" in payload or "flow" in payload):
        return []
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{source}: response does not contain a flow list")
    return list(value)


def _runs(payload: Any, source: str) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: response must be a JSON object")
    value: Any = payload.get("runs", payload.get("run"))
    if isinstance(value, Mapping):
        if "run" not in value and "runs" not in value:
            namespace_only = all(str(key).startswith("@xmlns") for key in value)
            if namespace_only:
                return []
            raise ValueError(f"{source}: response does not contain a run list")
        value = value.get("run", value.get("runs"))
    if value is None and ("runs" in payload or "run" in payload):
        return []
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{source}: response does not contain a run list")
    return list(value)


def _optional_numeric_id(value: Mapping[str, Any], *keys: str) -> str:
    text = _first_text(value, *keys)
    if not text:
        return ""
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f"OpenML run row has invalid {keys[0]}")
    return str(int(text))


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _first_text(value: Mapping[str, Any], *keys: str) -> str:
    return next((text for key in keys if (text := _text(value.get(key)))), "")


def _optional_http_url(value: Any) -> str | None:
    candidate = _text(value)
    parts = urlsplit(candidate)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return None
    try:
        return canonicalize_url(candidate)
    except ValueError:
        return None


__all__ = ["OpenMLFlowRegistrySourceAdapter"]
