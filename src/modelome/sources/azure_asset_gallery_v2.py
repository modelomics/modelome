"""Anonymous paginated reads from Microsoft's Azure ML Asset Catalog V2."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from hashlib import sha256
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

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

_URL = "https://api.catalog.azureml.ms/asset-gallery/v1.0/models"
_DETAIL_BASE = "https://api.catalog.azureml.ms/asset-gallery/v1.0"
_CATALOG_URL = "https://ai.azure.com/catalog/models/"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AzureAssetGalleryV2Adapter:
    """Enumerate public Azure model asset versions using continuation tokens.

    This is the Azure catalog asset namespace, not an inference availability
    list. The public query sends no credentials and enumerates public catalog
    entries returned by the documented Asset Catalog V2 API.
    """

    coverage_limitation = (
        "Covers public records returned by Azure Asset Catalog V2, including "
        "third-party catalog assets; it does not assert that assets are "
        "currently deployable or callable as hosted inference endpoints. The "
        "catalog is mutable during cursor scans, so records omitted by an "
        "unstable scan are not withdrawn by tombstones."
    )

    def __init__(
        self,
        *,
        name: str = "azure-asset-gallery-v2",
        page_size: int = 100,
        max_pages: int = 500,
        max_scan_restarts: int = 3,
        max_response_bytes: int = 8 * 1024 * 1024,
        timeout: float = 30.0,
        client: Any | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("name is required")
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        if max_pages < 1 or max_scan_restarts < 1 or max_response_bytes < 1 or timeout <= 0:
            raise ValueError("pagination, response, and timeout bounds must be positive")
        self.name = name
        self.page_size = page_size
        self.max_pages = max_pages
        self.max_scan_restarts = max_scan_restarts
        self.max_response_bytes = max_response_bytes
        self.timeout = timeout
        self.client = client
        self.checkpoint_signature = content_hash(
            {
                "adapter": "azure-asset-gallery-v2-v1",
                "url": _URL,
                "query": {
                    "order": [{"field": "name", "direction": "asc"}],
                },
                "page_size": page_size,
                "max_pages": max_pages,
                "max_scan_restarts": max_scan_restarts,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any] | None = None) -> SourcePage:
        state = state or {}
        page_number = _state_int(state.get("page", 1), "page")
        if page_number > self.max_pages:
            raise ValueError(f"{self.name}: page limit exceeded")
        token = state.get("continuation_token")
        if token is not None and (not isinstance(token, str) or not token):
            raise ValueError(f"{self.name}: invalid continuation token")
        expected_total = state.get("total_count")
        if expected_total is not None:
            expected_total = _state_int(expected_total, "total_count")
        records_seen = _nonnegative_int(state.get("records_seen", 0), "records_seen")
        scan_restarts = _nonnegative_int(
            state.get("scan_restarts", 0), "scan_restarts"
        )
        seen_cursor_hashes = _hash_list(
            state.get("seen_cursor_hashes", []), "seen_cursor_hashes", self.max_pages
        )
        seen_asset_hashes = _hash_list(
            state.get("seen_asset_hashes", []),
            "seen_asset_hashes",
            self.page_size * self.max_pages,
        )
        if records_seen != len(seen_asset_hashes):
            raise ValueError(f"{self.name}: records_seen does not match prior assets")
        if token is not None and _digest(token) not in seen_cursor_hashes:
            raise ValueError(f"{self.name}: continuation token is not in scan history")

        payload: dict[str, Any] = {
            "order": [{"field": "name", "direction": "asc"}],
            "pageSize": self.page_size,
            "includeTotalResultCount": True,
        }
        if token is not None:
            payload["continuationToken"] = token
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        response_body, status = self._post(body)
        if status != 200:
            raise ValueError(f"{self.name}: model gallery returned HTTP {status}")
        try:
            response = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{self.name}: model gallery returned invalid JSON") from exc
        if not isinstance(response, Mapping):
            raise ValueError(f"{self.name}: model gallery response is not an object")
        total = _nonnegative_int(response.get("totalCount"), "totalCount")
        if expected_total is not None and total != expected_total:
            return self._restart_scan(state, total, "totalCount changed")
        if total > self.page_size * self.max_pages:
            raise ValueError(f"{self.name}: catalog exceeds configured page bound")
        summaries = response.get("summaries")
        if not isinstance(summaries, list) or len(summaries) > self.page_size:
            raise ValueError(f"{self.name}: invalid model summaries page")
        next_token = response.get("continuationToken")
        if next_token is not None and (not isinstance(next_token, str) or not next_token):
            raise ValueError(f"{self.name}: invalid next continuation token")
        if next_token is not None and not summaries:
            raise ValueError(f"{self.name}: empty page has a continuation token")
        next_token_hash = _digest(next_token) if next_token is not None else None
        if next_token_hash is not None and next_token_hash in seen_cursor_hashes:
            raise ValueError(f"{self.name}: continuation token repeated or cycled")
        if len(summaries) > total:
            return self._restart_scan(state, total, "page rows exceed totalCount")

        records = tuple(self._record(item) for item in summaries)
        asset_hashes = [_digest(str(record.raw["assetId"])) for record in records]
        if len(set(asset_hashes)) != len(asset_hashes) or set(asset_hashes) & set(
            seen_asset_hashes
        ):
            return self._restart_scan(state, total, "duplicate asset version across pages")
        records_seen_after = records_seen + len(records)
        if records_seen_after > total:
            return self._restart_scan(state, total, "records_seen exceeds totalCount")
        complete = next_token is None
        if complete and records_seen_after != total:
            return self._restart_scan(
                state, total, "completed scan count does not match totalCount"
            )
        next_seen_cursor_hashes = list(seen_cursor_hashes)
        if next_token_hash is not None:
            next_seen_cursor_hashes.append(next_token_hash)
        next_seen_asset_hashes = [*seen_asset_hashes, *asset_hashes]
        next_state: dict[str, Any] = {
            "page": page_number + 1,
            "total_count": total,
            "records_seen": records_seen_after,
            "scan_restarts": scan_restarts,
            "seen_cursor_hashes": next_seen_cursor_hashes,
            "seen_asset_hashes": next_seen_asset_hashes,
        }
        if next_token is not None:
            next_state["continuation_token"] = next_token
        return SourcePage(
            records,
            next_state,
            complete,
            upstream_count=total,
            authoritative_snapshot=False,
        )

    def _restart_scan(
        self, state: Mapping[str, Any], total: int, reason: str
    ) -> SourcePage:
        restarts = _nonnegative_int(state.get("scan_restarts", 0), "scan_restarts") + 1
        if restarts > self.max_scan_restarts:
            raise ValueError(
                f"{self.name}: catalog remained unstable after "
                f"{self.max_scan_restarts} scan restarts ({reason})"
            )
        return SourcePage(
            records=(),
            next_state={
                "page": 1,
                "total_count": total,
                "records_seen": 0,
                "seen_cursor_hashes": [],
                "seen_asset_hashes": [],
                "scan_restarts": restarts,
            },
            complete=False,
            upstream_count=total,
            authoritative_snapshot=False,
        )

    def _post(self, body: bytes) -> tuple[bytes, int]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.client is not None:
            response = self.client.post(_URL, data=body, headers=headers)
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: response exceeds byte limit")
            return response.body, response.status

        request = Request(_URL, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                if response.geturl() != _URL:
                    raise ValueError(f"{self.name}: unexpected redirect")
                response_body = response.read(self.max_response_bytes + 1)
                status = response.status
        except HTTPError as exc:
            return exc.read(self.max_response_bytes + 1), exc.code
        if len(response_body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: response exceeds byte limit")
        return response_body, status

    @staticmethod
    def _record(item: Any) -> SourceRecord:
        if not isinstance(item, Mapping):
            raise ValueError("azure asset gallery summary is not an object")
        name = _required_text(item.get("name"), "name")
        display_name = _required_text(item.get("displayName"), "displayName")
        registry = _required_text(item.get("registryName"), "registryName")
        version = _required_text(item.get("version"), "version")
        asset_id = _required_text(item.get("assetId"), "assetId")
        expected_id = f"azureml://registries/{registry}/models/{name}/versions/{version}"
        if asset_id != expected_id:
            raise ValueError("azure asset gallery assetId does not match its identity fields")
        model_identity = f"{registry}/{name}"
        local_id = f"{registry}:{name}"
        detail_url = (
            f"{_DETAIL_BASE}/{quote(registry, safe='')}/models/"
            f"{quote(name, safe='')}/version/{quote(version, safe='')}"
        )
        portal_url = f"{_CATALOG_URL}{quote(name, safe='')}?version={quote(version, safe='')}"
        identifiers = (Identifier("azureml:catalog-model", model_identity),)
        release_id = f"{model_identity}@{version}"
        release = ReleaseHint(
            local_id=f"version:{version}",
            model_local_id=local_id,
            version=version,
            identifiers=(Identifier("azureml:catalog-model-version", asset_id),),
            released_at=_optional_text(item.get("createdTime")),
            metadata={"asset_id": asset_id, "registry": registry},
            locator=f"Asset Catalog V2 model version {version}",
        )
        model = ModelHint(
            local_id=local_id,
            name=display_name,
            aliases=(name,),
            identifiers=identifiers,
            status=ModelStatus.DOCUMENTED,
            locator=f"registry={registry}, name={name}, version={version}",
        )
        return SourceRecord(
            source_record_id=f"azureml:asset-version:{release_id}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(detail_url),
            title=f"{display_name} (version {version})",
            raw=dict(item),
            text=(
                f"Azure public model asset {registry}/{name}, version {version}; "
                f"publisher: {_optional_text(item.get('publisher')) or 'unknown'}."
            ),
            identifiers=identifiers + (Identifier("azureml:catalog-model-version", asset_id),),
            links=(Link(portal_url, relation="documents_model", crawl=False),),
            models=(model,),
            releases=(release,),
        )


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"azure asset gallery summary has invalid {field}")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"azure asset gallery response has invalid {field}")
    return value


def _state_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"azure asset gallery state has invalid {field}")
    return value


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _hash_list(value: Any, field: str, maximum: int) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"azure asset gallery state has invalid {field}")
    if any(not isinstance(item, str) or not _SHA256.fullmatch(item) for item in value):
        raise ValueError(f"azure asset gallery state has invalid {field}")
    if len(set(value)) != len(value):
        raise ValueError(f"azure asset gallery state has duplicate {field}")
    return list(value)
