"""Historical spaCy pipeline-package enumeration from its compatibility manifest."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]

_DEFAULT_URL = (
    "https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json"
)
_MANIFEST_PATH = re.compile(
    r"^/explosion/spacy-models/[A-Za-z0-9._-]+/compatibility\.json$"
)
_PACKAGE = re.compile(r"^[a-z][a-z0-9_]{1,127}$")
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9._+-]*)?$")
_COMPATIBILITY = re.compile(r"^[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9._+-]*)?$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SpacyModelsSourceAdapter:
    """Enumerate every public spaCy pipeline and compatible package release.

    spaCy's maintained ``compatibility.json`` is a compact, machine-readable
    historical manifest. It maps each spaCy compatibility series to exact
    pipeline package names and published package versions. This adapter keeps
    that source-declared release history without deriving package-file URLs or
    downloading wheels.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers pipeline package names and versions retained by spaCy's public "
        "compatibility manifest. It does not prove availability of a wheel on a "
        "package index, cover privately distributed pipelines, or enumerate "
        "models outside spaCy's compatibility publication."
    )

    def __init__(
        self,
        *,
        name: str = "spacy-models",
        url: str = _DEFAULT_URL,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_models: int = 10_000,
        max_releases: int = 100_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _manifest_url(url, self.name)
        self.max_response_bytes = _positive_int(
            max_response_bytes, "max_response_bytes", self.name
        )
        self.max_models = _positive_int(max_models, "max_models", self.name)
        self.max_releases = _positive_int(max_releases, "max_releases", self.name)
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "spacy-models-v1",
                "url": self.url,
                "max_response_bytes": self.max_response_bytes,
                "max_models": self.max_models,
                "max_releases": self.max_releases,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.url,
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            raise ValueError(
                f"{self.name}: compatibility manifest returned HTTP {response.status}"
            )
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: compatibility manifest exceeds "
                f"{self.max_response_bytes} bytes"
            )
        payload = response.json()
        models, releases, compatibility = self._parse_manifest(payload)
        manifest_hash = content_hash(response.body)
        checked_at = _isoformat(self.clock())
        next_state = {
            "completed_content_hash": manifest_hash,
            "checked_at": checked_at,
            "model_count": len(models),
            "release_count": len(releases),
        }
        if manifest_hash == _text(state.get("completed_content_hash")):
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=len(releases),
                authoritative_snapshot=False,
            )

        record = SourceRecord(
            source_record_id=f"{self.name}:compatibility",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=self.url,
            title="spaCy pipeline compatibility manifest",
            raw={
                "manifest_url": self.url,
                "response_url": response.url or self.url,
                "content_hash": manifest_hash,
                "content_bytes": len(response.body),
                "compatibility": compatibility,
            },
            text="\n".join(model.name for model in models),
            models=models,
            releases=releases,
        )
        return SourcePage(
            records=(record,),
            next_state=next_state,
            complete=True,
            upstream_count=len(releases),
            authoritative_snapshot=True,
        )

    def _parse_manifest(
        self, payload: Any
    ) -> tuple[tuple[ModelHint, ...], tuple[ReleaseHint, ...], Mapping[str, Any]]:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: compatibility manifest must be an object")
        spacy = payload.get("spacy")
        if not isinstance(spacy, Mapping):
            raise ValueError(f"{self.name}: compatibility manifest is missing spacy map")

        by_package: dict[str, dict[str, set[str]]] = {}
        compatibility_raw: dict[str, dict[str, list[str]]] = {}
        for series, package_map in spacy.items():
            series_name = _required_text(series, "spaCy compatibility series")
            if not _COMPATIBILITY.fullmatch(series_name):
                raise ValueError(
                    f"{self.name}: invalid spaCy compatibility series {series_name!r}"
                )
            if not isinstance(package_map, Mapping):
                raise ValueError(
                    f"{self.name}: compatibility series {series_name!r} is not an object"
                )
            normalized_packages: dict[str, list[str]] = {}
            for package, versions in package_map.items():
                package_name = _required_text(package, "pipeline package")
                if not _PACKAGE.fullmatch(package_name):
                    raise ValueError(
                        f"{self.name}: invalid pipeline package {package_name!r}"
                    )
                if not isinstance(versions, Sequence) or isinstance(
                    versions, (str, bytes, bytearray)
                ):
                    raise ValueError(
                        f"{self.name}: versions for {package_name!r} must be an array"
                    )
                normalized_versions: list[str] = []
                for value in versions:
                    version = _required_text(value, f"version for {package_name}")
                    if not _VERSION.fullmatch(version):
                        raise ValueError(
                            f"{self.name}: invalid package version {version!r}"
                        )
                    normalized_versions.append(version)
                    by_package.setdefault(package_name, {}).setdefault(version, set()).add(
                        series_name
                    )
                if not normalized_versions:
                    raise ValueError(
                        f"{self.name}: {package_name!r} has no declared package versions"
                    )
                normalized_packages[package_name] = sorted(set(normalized_versions))
            if not normalized_packages:
                raise ValueError(
                    f"{self.name}: compatibility series {series_name!r} has no pipelines"
                )
            compatibility_raw[series_name] = dict(sorted(normalized_packages.items()))

        if not by_package:
            raise ValueError(f"{self.name}: compatibility manifest declares no pipelines")
        if len(by_package) > self.max_models:
            raise ValueError(f"{self.name}: manifest exceeds {self.max_models} pipeline models")

        models: list[ModelHint] = []
        releases: list[ReleaseHint] = []
        for package_name, versions in sorted(by_package.items()):
            local_id = f"spacy:{package_name}#model"
            models.append(
                ModelHint(
                    local_id=local_id,
                    name=package_name,
                    identifiers=(Identifier("spacy:model", package_name),),
                    status=ModelStatus.RELEASED,
                    locator=f"spacy.{package_name}",
                )
            )
            for version, series in sorted(versions.items()):
                releases.append(
                    ReleaseHint(
                        local_id=f"spacy:{package_name}#release:{version}",
                        model_local_id=local_id,
                        version=version,
                        identifiers=(
                            Identifier("spacy:package-version", f"{package_name}@{version}"),
                        ),
                        metadata={"spacy_compatibility": sorted(series)},
                        locator=f"spacy.{','.join(sorted(series))}.{package_name}",
                    )
                )
        if len(releases) > self.max_releases:
            raise ValueError(
                f"{self.name}: manifest exceeds {self.max_releases} pipeline releases"
            )
        return tuple(models), tuple(releases), dict(sorted(compatibility_raw.items()))


def _manifest_url(value: str, source: str) -> str:
    url = canonicalize_url(_required_text(value, "compatibility manifest URL"))
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname != "raw.githubusercontent.com"
        or not _MANIFEST_PATH.fullmatch(parts.path)
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            f"{source}: compatibility manifest must be an HTTPS raw spaCy GitHub URL"
        )
    return url


def _required_text(value: Any, label: str) -> str:
    result = str(value).strip() if isinstance(value, str) else ""
    if not result:
        raise ValueError(f"{label} must be a non-empty string")
    return result


def _positive_int(value: Any, label: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {label} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{source}: {label} must be a positive integer")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = ["SpacyModelsSourceAdapter"]
