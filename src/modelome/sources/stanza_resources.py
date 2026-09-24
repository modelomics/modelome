"""Version-pinned historical enumeration of Stanza pretrained resources."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    ModelHint,
    ModelRelationHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_RESOURCE_PATH = re.compile(r"^resources_(?P<version>\d+\.\d+\.\d+)\.json$")
_COORDINATE_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,255}$")
_LANGUAGE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _ModelRelease:
    language: str
    processor: str
    package: str
    md5: str
    dependencies: tuple[tuple[str, str], ...]

    @property
    def coordinate(self) -> str:
        return f"{self.language}/{self.processor}/{self.package}"

    @property
    def local_id(self) -> str:
        return f"stanza:{self.coordinate}#model"


@dataclass(frozen=True, slots=True)
class _PipelinePackage:
    language: str
    package: str
    components: tuple[tuple[str, str], ...]

    @property
    def coordinate(self) -> str:
        return f"{self.language}/package/{self.package}"

    @property
    def local_id(self) -> str:
        return f"stanza:{self.coordinate}#model"


class StanzaResourcesSourceAdapter:
    """Enumerate every versioned Stanza resource manifest at an immutable commit.

    The public ``stanfordnlp/stanza-resources`` repository publishes one
    resources JSON file per Stanza release. Each processor/package object with
    an MD5 checksum is a source-declared pretrained component release. Named
    entries in each language's ``packages`` map are retained as pipeline
    variants with links to their component releases. All retained manifests
    preserve historical records while never constructing or downloading
    artifact-file URLs.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers processor/package releases and named pipeline variants present in "
        "the public Stanza resources "
        "repository at one observed commit. It does not prove that a referenced "
        "model file remains downloadable, enumerate private resources, or establish "
        "coverage of models outside Stanza's maintained resource manifests."
    )

    def __init__(
        self,
        *,
        name: str = "stanza-resources",
        repository: str = "stanfordnlp/stanza-resources",
        branch: str = "main",
        max_tree_bytes: int = 2 * 1024 * 1024,
        max_manifest_bytes: int = 2 * 1024 * 1024,
        max_manifests: int = 100,
        max_models_per_manifest: int = 20_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.max_tree_bytes = _positive_int(max_tree_bytes, "max_tree_bytes", self.name)
        self.max_manifest_bytes = _positive_int(
            max_manifest_bytes, "max_manifest_bytes", self.name
        )
        self.max_manifests = _positive_int(max_manifests, "max_manifests", self.name)
        self.max_models_per_manifest = _positive_int(
            max_models_per_manifest, "max_models_per_manifest", self.name
        )
        self.client = client or HttpClient(
            max_response_bytes=max(self.max_tree_bytes, self.max_manifest_bytes)
        )
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "stanza-resources-v1",
                "repository": self.repository,
                "branch": self.branch,
                "max_tree_bytes": self.max_tree_bytes,
                "max_manifest_bytes": self.max_manifest_bytes,
                "max_manifests": self.max_manifests,
                "max_models_per_manifest": self.max_models_per_manifest,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}"
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
                upstream_count=_nonnegative_int(state.get("model_release_count")),
                authoritative_snapshot=False,
            )

        tree_response = self.client.get(
            self._tree_url(revision),
            headers={"Accept": "application/vnd.github+json"},
        )
        if tree_response.status != 200:
            raise ValueError(f"{self.name}: repository tree returned HTTP {tree_response.status}")
        if len(tree_response.body) > self.max_tree_bytes:
            raise ValueError(
                f"{self.name}: repository tree exceeds {self.max_tree_bytes} bytes"
            )
        manifests = self._manifest_paths(tree_response.json())
        if len(manifests) > self.max_manifests:
            raise ValueError(
                f"{self.name}: repository declares more than "
                f"{self.max_manifests} resource manifests"
            )

        records: list[SourceRecord] = []
        total_releases = 0
        manifest_hashes: dict[str, str] = {}
        for version, path in manifests:
            response = self.client.get(
                self._raw_url(revision, path), headers={"Accept": "application/json"}
            )
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: resource manifest {path} returned HTTP {response.status}"
                )
            if len(response.body) > self.max_manifest_bytes:
                raise ValueError(
                    f"{self.name}: resource manifest {path} exceeds "
                    f"{self.max_manifest_bytes} bytes"
                )
            payload = response.json()
            releases = self._releases(payload, path)
            packages = self._pipeline_packages(payload, path)
            total_releases += len(releases) + len(packages)
            manifest_hash = content_hash(response.body)
            manifest_hashes[path] = manifest_hash
            records.append(
                self._record(
                    version,
                    path,
                    revision,
                    releases,
                    packages,
                    manifest_hash,
                    response,
                    payload,
                )
            )

        next_state = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "manifest_count": len(manifests),
            "model_release_count": total_releases,
            "manifest_hashes": manifest_hashes,
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=True,
            upstream_count=total_releases,
            authoritative_snapshot=True,
        )

    def _revision(self) -> tuple[str, HttpResponse]:
        response: HttpResponse = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit lookup returned HTTP {response.status}")
        if len(response.body) > self.max_tree_bytes:
            raise ValueError(
                f"{self.name}: commit response exceeds {self.max_tree_bytes} bytes"
            )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: commit lookup is not an object")
        revision = _text(payload.get("sha"))
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit lookup returned invalid SHA")
        return revision, response

    def _manifest_paths(self, payload: Any) -> tuple[tuple[str, str], ...]:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: repository tree is not an object")
        if payload.get("truncated") is True:
            raise ValueError(f"{self.name}: repository tree is truncated")
        tree = payload.get("tree")
        if not isinstance(tree, Sequence) or isinstance(tree, (str, bytes, bytearray)):
            raise ValueError(f"{self.name}: repository tree is missing entries")
        entries: list[tuple[str, str]] = []
        seen_versions: set[str] = set()
        for item in tree:
            if not isinstance(item, Mapping) or item.get("type") != "blob":
                continue
            path = _text(item.get("path"))
            match = _RESOURCE_PATH.fullmatch(path)
            if match is None:
                continue
            version = match.group("version")
            if version in seen_versions:
                raise ValueError(
                    f"{self.name}: duplicate resource manifest version {version!r}"
                )
            seen_versions.add(version)
            entries.append((version, path))
        if not entries:
            raise ValueError(f"{self.name}: repository tree contains no resource manifests")
        return tuple(sorted(entries, key=lambda item: _version_key(item[0])))

    def _releases(self, payload: Any, path: str) -> tuple[_ModelRelease, ...]:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: resource manifest {path} is not an object")
        releases: list[_ModelRelease] = []
        seen: set[str] = set()
        for language_value, language_data in payload.items():
            if language_value == "url" and isinstance(language_data, str):
                # Early Stanza manifests retain a source-wide resource-base URL.
                # It is provenance metadata, not a language or a model declaration.
                continue
            language = _language(language_value, self.name, path)
            if not isinstance(language_data, Mapping):
                raise ValueError(
                    f"{self.name}: language {language!r} in {path} is not an object"
                )
            if "alias" in language_data:
                continue
            for processor_value, packages in language_data.items():
                processor = _text(processor_value)
                if processor in {"lang_name", "default_md5", "default_processors", "packages"}:
                    continue
                if not _COORDINATE_PART.fullmatch(processor) or not isinstance(
                    packages, Mapping
                ):
                    continue
                for package_value, metadata in packages.items():
                    package = _text(package_value)
                    if not _COORDINATE_PART.fullmatch(package) or not isinstance(
                        metadata, Mapping
                    ):
                        continue
                    md5 = _text(metadata.get("md5"))
                    if not md5:
                        continue
                    if not _MD5.fullmatch(md5):
                        raise ValueError(
                            f"{self.name}: invalid checksum for "
                            f"{language}/{processor}/{package} in {path}"
                        )
                    release = _ModelRelease(
                        language=language,
                        processor=processor,
                        package=package,
                        md5=md5,
                        dependencies=self._dependencies(
                            metadata.get("dependencies"), language, path
                        ),
                    )
                    if release.coordinate in seen:
                        raise ValueError(
                            f"{self.name}: duplicate model coordinate "
                            f"{release.coordinate!r} in {path}"
                        )
                    seen.add(release.coordinate)
                    releases.append(release)
        if not releases:
            raise ValueError(f"{self.name}: resource manifest {path} declares no models")
        if len(releases) > self.max_models_per_manifest:
            raise ValueError(
                f"{self.name}: resource manifest {path} exceeds "
                f"{self.max_models_per_manifest} model releases"
            )
        return tuple(sorted(releases, key=lambda item: item.coordinate))

    def _pipeline_packages(self, payload: Any, path: str) -> tuple[_PipelinePackage, ...]:
        """Read named Stanza pipeline recipes and their processor/package members."""
        packages: list[_PipelinePackage] = []
        seen: set[str] = set()
        for language_value, language_data in payload.items():
            if not isinstance(language_data, Mapping) or "alias" in language_data:
                continue
            language = _language(language_value, self.name, path)
            raw_packages = language_data.get("packages")
            if not isinstance(raw_packages, Mapping):
                continue
            for package_value, raw_components in raw_packages.items():
                package = _text(package_value)
                if not _COORDINATE_PART.fullmatch(package):
                    raise ValueError(
                        f"{self.name}: invalid pipeline package {package!r} in {path}"
                    )
                if not isinstance(raw_components, Mapping):
                    raise ValueError(
                        f"{self.name}: pipeline package {package!r} in {path} is not an object"
                    )
                components = []
                for processor_value, component_value in raw_components.items():
                    processor = _text(processor_value)
                    component = _text(component_value)
                    if not _COORDINATE_PART.fullmatch(processor) or not _COORDINATE_PART.fullmatch(
                        component
                    ):
                        raise ValueError(
                            f"{self.name}: invalid component in pipeline package {package!r}"
                        )
                    components.append((processor, component))
                if not components:
                    continue
                coordinate = f"{language}/package/{package}"
                if coordinate in seen:
                    raise ValueError(
                        f"{self.name}: duplicate pipeline package {coordinate!r} in {path}"
                    )
                seen.add(coordinate)
                packages.append(
                    _PipelinePackage(
                        language=language,
                        package=package,
                        components=tuple(sorted(set(components))),
                    )
                )
        return tuple(sorted(packages, key=lambda item: item.coordinate))

    def _dependencies(
        self, value: Any, language: str, path: str
    ) -> tuple[tuple[str, str], ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError(f"{self.name}: dependencies in {path} must be an array")
        dependencies: set[tuple[str, str]] = set()
        for item in value:
            if not isinstance(item, Mapping):
                raise ValueError(f"{self.name}: dependency in {path} is not an object")
            processor = _text(item.get("model"))
            package = _text(item.get("package"))
            if not _COORDINATE_PART.fullmatch(processor) or not _COORDINATE_PART.fullmatch(
                package
            ):
                raise ValueError(f"{self.name}: invalid dependency in {path}")
            dependencies.add((processor, package))
        return tuple(sorted(dependencies))

    def _record(
        self,
        version: str,
        path: str,
        revision: str,
        releases: Sequence[_ModelRelease],
        packages: Sequence[_PipelinePackage],
        manifest_hash: str,
        response: HttpResponse,
        payload: Mapping[str, Any],
    ) -> SourceRecord:
        models = tuple(
            ModelHint(
                local_id=item.local_id,
                name=item.coordinate,
                identifiers=(Identifier("stanza:model", item.coordinate),),
                status=ModelStatus.RELEASED,
                locator=f"{item.language}.{item.processor}.{item.package}",
            )
            for item in releases
        )
        release_hints = tuple(
            ReleaseHint(
                local_id=f"{item.local_id}:resources-{version}",
                model_local_id=item.local_id,
                revision=version,
                identifiers=(
                    Identifier(
                        "stanza:resources-manifest-entry",
                        f"{version}:{item.coordinate}@{item.md5}",
                    ),
                ),
                metadata={"md5": item.md5, "resource_manifest": path},
                locator=f"{item.language}.{item.processor}.{item.package}.md5",
            )
            for item in releases
        )
        relations: list[ModelRelationHint] = []
        for item in releases:
            for processor, package in item.dependencies:
                coordinate = f"{item.language}/{processor}/{package}"
                relations.append(
                    ModelRelationHint(
                        subject_local_id=item.local_id,
                        predicate="depends_on",
                        target=ModelHint(
                            local_id=f"stanza:{coordinate}#model",
                            name=coordinate,
                            identifiers=(Identifier("stanza:model", coordinate),),
                            status=ModelStatus.RELEASED,
                            locator=(
                                f"{item.language}.{processor}.{package}.dependency"
                            ),
                        ),
                        locator=f"{item.language}.{item.processor}.{item.package}.dependencies",
                    )
                )
        for package in packages:
            for processor, component_package in package.components:
                coordinate = f"{package.language}/{processor}/{component_package}"
                relations.append(
                    ModelRelationHint(
                        subject_local_id=package.local_id,
                        predicate="contains_component",
                        target=ModelHint(
                            local_id=f"stanza:{coordinate}#model",
                            name=coordinate,
                            identifiers=(Identifier("stanza:model", coordinate),),
                            status=ModelStatus.RELEASED,
                            locator=(
                                f"{package.language}.packages.{package.package}."
                                f"{processor}"
                            ),
                        ),
                        locator=f"{package.language}.packages.{package.package}.{processor}",
                    )
                )
        package_models = tuple(
            ModelHint(
                local_id=item.local_id,
                name=item.coordinate,
                identifiers=(Identifier("stanza:pipeline-package", item.coordinate),),
                status=ModelStatus.RELEASED,
                locator=f"{item.language}.packages.{item.package}",
            )
            for item in packages
        )
        package_releases = tuple(
            ReleaseHint(
                local_id=f"{item.local_id}:resources-{version}",
                model_local_id=item.local_id,
                revision=version,
                identifiers=(
                    Identifier(
                        "stanza:pipeline-package-version",
                        f"{item.coordinate}@{version}",
                    ),
                ),
                metadata={"resource_manifest": path, "component_count": len(item.components)},
                locator=f"{item.language}.packages.{item.package}",
            )
            for item in packages
        )
        raw_url = self._raw_url(revision, path)
        return SourceRecord(
            source_record_id=f"{self.name}:resources-{version}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=raw_url,
            title=f"Stanza resources {version}",
            raw={
                "repository": self.repository,
                "repository_revision": revision,
                "resource_manifest": path,
                "resource_version": version,
                "manifest_hash": manifest_hash,
                "response_url": response.url or raw_url,
                "content_bytes": len(response.body),
                "manifest": payload,
            },
            text="\n".join(
                [item.coordinate for item in releases]
                + [item.coordinate for item in packages]
            ),
            identifiers=(Identifier("stanza:resources-manifest", version),),
            models=(*models, *package_models),
            releases=(*release_hints, *package_releases),
            model_relations=tuple(relations),
        )

    def _tree_url(self, revision: str) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/git/trees/"
            f"{quote(revision, safe='')}?recursive=1"
        )

    def _raw_url(self, revision: str, path: str) -> str:
        return canonicalize_url(
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(path, safe='/')}"
        )


def _repository(value: str) -> str:
    result = _required_text(value, "repository")
    if not _REPOSITORY.fullmatch(result):
        raise ValueError("repository must be an owner/name pair")
    return result


def _language(value: Any, source: str, path: str) -> str:
    result = _text(value)
    if not _LANGUAGE.fullmatch(result):
        raise ValueError(f"{source}: invalid language key {result!r} in {path}")
    return result


def _version_key(value: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in value.split("."))  # type: ignore[return-value]


def _required_text(value: Any, label: str) -> str:
    result = _text(value)
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


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _header(headers: Mapping[str, Any], name: str) -> str:
    wanted = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == wanted:
            return _text(value)
    return ""


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = ["StanzaResourcesSourceAdapter"]
