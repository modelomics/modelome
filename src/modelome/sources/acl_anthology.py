from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit
from xml.etree import ElementTree

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AclAnthologySourceAdapter:
    """Enumerate papers across bounded ACL Anthology XML collections.

    The default mode reads the official Git tree manifest, freezes its XML
    collection paths, and visits one collection file per source page. Per-file
    digests skip unchanged collections while a tree SHA makes interrupted walks
    resumable. Supplying ``url`` retains a fixed single-collection mode.
    """

    def __init__(
        self,
        *,
        name: str = "acl-anthology",
        url: str | None = None,
        repository: str = "acl-org/acl-anthology",
        branch: str = "master",
        manifest_url: str | None = None,
        max_response_bytes: int = 32 * 1024 * 1024,
        max_papers: int = 10_000,
        max_collections: int = 20_000,
        max_abstract_chars: int = 100_000,
        client: HttpClient | Any | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.name = _text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _text(branch, "branch")
        if url is not None and manifest_url is not None:
            raise ValueError(f"{self.name}: configure either url or manifest_url, not both")
        self.collection_url = _https_url(url, self.name) if url is not None else None
        default_manifest = (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
        )
        self.manifest_url = (
            _https_url(manifest_url, self.name)
            if manifest_url is not None
            else (default_manifest if self.collection_url is None else "")
        )
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_papers = _positive_int(max_papers, "max_papers")
        self.max_collections = _positive_int(max_collections, "max_collections")
        if self.max_collections > 20_000:
            raise ValueError(f"{self.name}: max_collections must not exceed 20000")
        self.max_abstract_chars = _positive_int(max_abstract_chars, "max_abstract_chars")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "acl-anthology-collection-xml-v2",
                "repository": self.repository,
                "branch": self.branch,
                "collection_url": self.collection_url,
                "manifest_url": self.manifest_url,
                "max_response_bytes": self.max_response_bytes,
                "max_papers": self.max_papers,
                "max_collections": self.max_collections,
                "max_abstract_chars": self.max_abstract_chars,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if self.collection_url is not None:
            return self._fetch_single_collection(state)
        return self._fetch_manifest_collection(state)

    def _fetch_single_collection(self, state: Mapping[str, Any]) -> SourcePage:
        prior_digest = state.get("snapshot_digest", "")
        if prior_digest and (
            not isinstance(prior_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", prior_digest)
        ):
            raise ValueError(f"{self.name}: invalid snapshot_digest checkpoint")
        response: HttpResponse = self.client.get(
            self.collection_url,
            headers={"Accept": "application/xml, text/xml"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: collection returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: response exceeds {self.max_response_bytes} bytes")
        digest = hashlib.sha256(response.body).hexdigest()
        checked_at = _isoformat(self.clock())
        if digest == prior_digest:
            return SourcePage(
                records=(),
                next_state={"snapshot_digest": digest, "checked_at": checked_at},
                complete=True,
            )

        return self._parse_collection(
            response.body,
            digest=digest,
            state={"snapshot_digest": digest, "checked_at": checked_at},
            source_label=self.collection_url,
            prior_digest=prior_digest,
        )

    def _fetch_manifest_collection(self, state: Mapping[str, Any]) -> SourcePage:
        completed_tree_sha = _optional_text(state.get("completed_tree_sha"))
        digests = _state_string_map(
            state.get("collection_digests", {}), self.name, self.max_collections
        )
        counts = _state_count_map(
            state.get("collection_counts", {}), self.name, self.max_collections
        )
        pending_paths = state.get("collection_paths")
        if pending_paths is None:
            response = self.client.get(
                self.manifest_url,
                headers={"Accept": "application/vnd.github+json, application/json"},
            )
            commit = _json_object(response, self.name, self.max_response_bytes)
            commit_data = commit.get("commit")
            commit_tree = commit_data.get("tree") if isinstance(commit_data, Mapping) else None
            requested_tree_sha = (
                commit_tree.get("sha") if isinstance(commit_tree, Mapping) else None
            )
            commit_sha = commit.get("sha")
            if not isinstance(commit_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
                raise ValueError(f"{self.name}: branch response is missing a valid commit SHA")
            if not isinstance(requested_tree_sha, str) or not re.fullmatch(
                r"[0-9a-f]{40}", requested_tree_sha
            ):
                raise ValueError(f"{self.name}: branch commit is missing a valid tree SHA")
            if requested_tree_sha == completed_tree_sha:
                return SourcePage(
                    records=(),
                    next_state={
                        "completed_tree_sha": requested_tree_sha,
                        "collection_digests": digests,
                        "collection_counts": counts,
                        "checked_at": _isoformat(self.clock()),
                    },
                    complete=True,
                    upstream_count=sum(counts.values()),
                )
            tree_url = (
                f"https://api.github.com/repos/{self.repository}/git/trees/"
                f"{requested_tree_sha}?recursive=1"
            )
            tree_response = self.client.get(
                tree_url,
                headers={"Accept": "application/vnd.github+json, application/json"},
            )
            payload = _json_object(tree_response, self.name, self.max_response_bytes)
            tree_sha = payload.get("sha")
            if tree_sha != requested_tree_sha:
                raise ValueError(
                    f"{self.name}: returned tree SHA did not match requested commit tree"
                )
            if payload.get("truncated") is not False:
                raise ValueError(
                    f"{self.name}: Git tree manifest is missing or has truncated status"
                )
            tree = payload.get("tree")
            if not isinstance(tree, list):
                raise ValueError(f"{self.name}: manifest tree must be an array")
            paths: list[str] = []
            for item in tree:
                if not isinstance(item, Mapping):
                    continue
                path = item.get("path")
                if (
                    not isinstance(path, str)
                    or not path.startswith("data/xml/")
                    or not path.endswith(".xml")
                ):
                    continue
                if item.get("type") != "blob" or not re.fullmatch(
                    r"data/xml/[A-Za-z0-9._-]+\.xml", path
                ):
                    raise ValueError(f"{self.name}: unsupported XML collection path {path!r}")
                paths.append(path)
            if len(paths) != len(set(paths)):
                raise ValueError(f"{self.name}: manifest repeats a collection path")
            paths.sort()
            active_paths = set(paths)
            digests = {path: value for path, value in digests.items() if path in active_paths}
            counts = {path: value for path, value in counts.items() if path in active_paths}
            if len(paths) > self.max_collections:
                raise ValueError(
                    f"{self.name}: manifest has {len(paths)} collections, "
                    f"above max_collections {self.max_collections}"
                )
            pending_paths = paths
            index = 0
            frozen_tree_sha = tree_sha
            frozen_commit_sha = commit_sha
        else:
            frozen_tree_sha = _required_text(state.get("tree_sha"), "tree_sha", self.name)
            frozen_commit_sha = _required_text(state.get("commit_sha"), "commit_sha", self.name)
            if not re.fullmatch(r"[0-9a-f]{40}", frozen_commit_sha):
                raise ValueError(f"{self.name}: invalid frozen commit SHA")
            if not re.fullmatch(r"[0-9a-f]{40}", frozen_tree_sha):
                raise ValueError(f"{self.name}: invalid frozen tree SHA")
            if (
                not isinstance(pending_paths, list)
                or not pending_paths
                or len(pending_paths) > self.max_collections
            ):
                raise ValueError(f"{self.name}: invalid frozen collection path list")
            pending_paths = [_manifest_path(path, self.name) for path in pending_paths]
            if len(pending_paths) != len(set(pending_paths)):
                raise ValueError(f"{self.name}: frozen collection paths contain duplicates")
            index = _nonnegative_int(
                state.get("collection_index", 0), "collection_index", self.name
            )
            if index >= len(pending_paths):
                raise ValueError(f"{self.name}: collection_index is outside frozen path list")
        if not pending_paths:
            next_state = {
                "completed_tree_sha": frozen_tree_sha,
                "collection_digests": digests,
                "collection_counts": counts,
                "checked_at": _isoformat(self.clock()),
            }
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=sum(counts.values()),
            )

        path = _manifest_path(pending_paths[index], self.name)
        collection_url = (
            f"https://raw.githubusercontent.com/{self.repository}/{frozen_commit_sha}/{path}"
        )
        response = self.client.get(
            collection_url,
            headers={"Accept": "application/xml, text/xml"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: collection returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: response exceeds {self.max_response_bytes} bytes")
        digest = hashlib.sha256(response.body).hexdigest()
        next_index = index + 1
        complete = next_index == len(pending_paths)
        if digests.get(path) == digest:
            papers = counts.get(path, 0)
            records: tuple[SourceRecord, ...] = ()
            issues: tuple[SourceIssue, ...] = ()
        else:
            parsed = self._parse_collection(
                response.body,
                digest=digest,
                state={},
                source_label=path,
            )
            records = parsed.records
            issues = parsed.issues
            papers = parsed.upstream_count or 0
            digests[path] = digest
            counts[path] = papers
        next_state: dict[str, Any]
        if complete:
            next_state = {
                "completed_tree_sha": frozen_tree_sha,
                "collection_digests": digests,
                "collection_counts": counts,
                "checked_at": _isoformat(self.clock()),
            }
        else:
            next_state = {
                "tree_sha": frozen_tree_sha,
                "commit_sha": frozen_commit_sha,
                "collection_paths": pending_paths,
                "collection_index": next_index,
                "collection_digests": digests,
                "collection_counts": counts,
            }
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=complete,
            upstream_count=sum(counts.values()) if complete else papers,
            issues=issues,
            advance_on_source_issues=True,
        )

    def _parse_collection(
        self,
        body: bytes,
        *,
        digest: str,
        state: Mapping[str, Any],
        source_label: str,
        prior_digest: str = "",
    ) -> SourcePage:
        if len(body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: response exceeds {self.max_response_bytes} bytes")
        if prior_digest and digest == prior_digest:
            return SourcePage(records=(), next_state=state, complete=True)
        if b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
            raise ValueError(f"{self.name}: collection XML must not declare a DTD or entities")
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as error:
            raise ValueError(f"{self.name}: invalid collection XML: {error}") from None
        if root.tag != "collection":
            raise ValueError(f"{self.name}: expected collection XML root")
        collection_id = root.get("id", "") or source_label.rsplit("/", 1)[-1].removesuffix(".xml")
        paper_rows: list[tuple[ElementTree.Element, str, str, str]] = []
        for volume in root.findall("./volume"):
            volume_id = volume.get("id", "")
            booktitle = _element_text(volume.find("./meta/booktitle"))
            year = _element_text(volume.find("./meta/year"))
            paper_rows.extend(
                (paper, volume_id, booktitle, year) for paper in volume.findall("./paper")
            )
        if len(paper_rows) > self.max_papers:
            raise ValueError(
                f"{self.name}: collection has {len(paper_rows)} papers, "
                f"above max_papers {self.max_papers}"
            )
        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for index, (paper, volume_id, booktitle, year) in enumerate(paper_rows):
            try:
                records.append(
                    self._record(
                        paper,
                        collection_id=collection_id,
                        volume_id=volume_id,
                        booktitle=booktitle,
                        year=year,
                        index=index,
                    )
                )
            except (TypeError, ValueError) as error:
                paper_id = _paper_id(paper)
                issues.append(
                    SourceIssue(
                        source_record_id=paper_id or f"{self.name}:item:{index}",
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"collection_id": collection_id, "index": index},
                    )
                )
        return SourcePage(
            records=tuple(records),
            next_state=dict(state),
            complete=True,
            upstream_count=len(paper_rows),
            issues=tuple(issues),
            advance_on_source_issues=True,
        )

    def _record(
        self,
        paper: ElementTree.Element,
        *,
        collection_id: str,
        volume_id: str,
        booktitle: str,
        year: str,
        index: int,
    ) -> SourceRecord:
        anthology_id = _paper_id(paper)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,127}", anthology_id):
            raise ValueError("paper is missing a valid Anthology URL identifier")
        title_node = paper.find("title")
        title = _element_text(title_node)
        if not title:
            raise ValueError("paper is missing its title")
        authors = [name for node in paper.findall("author") if (name := _person_name(node))]
        abstract = _element_text(paper.find("abstract"))[: self.max_abstract_chars]
        doi = _element_text(paper.find("doi"))
        landing = canonicalize_url(f"https://aclanthology.org/{anthology_id}/")
        pdf = canonicalize_url(f"https://aclanthology.org/{anthology_id}.pdf")
        identifiers = [Identifier("acl-anthology", anthology_id)]
        if doi and re.fullmatch(r"10\.\d{4,9}/\S+", doi, flags=re.IGNORECASE):
            identifiers.append(Identifier("doi", doi.casefold()))
        text = "\n".join(value for value in (title, abstract, booktitle) if value)
        return SourceRecord(
            source_record_id=anthology_id,
            kind=ArtifactKind.PAPER,
            canonical_url=landing,
            title=title,
            text=text,
            published_at=f"{year}-01-01T00:00:00Z" if re.fullmatch(r"\d{4}", year) else None,
            identifiers=tuple(identifiers),
            links=(
                Link(landing, relation="landing_page", locator=f"xml:paper[{index}]", crawl=False),
                Link(pdf, relation="full_text", locator=f"xml:paper[{index}]/url", crawl=False),
            ),
            raw={
                "collection_id": collection_id,
                "volume_id": volume_id,
                "booktitle": booktitle,
                "year": year,
                "authors": authors,
                "abstract": abstract,
                "doi": doi,
            },
        )


def _paper_id(paper: ElementTree.Element) -> str:
    return _element_text(paper.find("url"))


def _element_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _person_name(person: ElementTree.Element) -> str:
    return " ".join(text for key in ("first", "last") if (text := _element_text(person.find(key))))


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _https_url(value: Any, source: str) -> str:
    text = _text(value, "URL")
    parts = urlsplit(text)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError(f"{source}: URL must be HTTPS without credentials")
    return text


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["AclAnthologySourceAdapter"]


def _json_object(response: HttpResponse, source: str, max_bytes: int) -> Mapping[str, Any]:
    if response.status != 200:
        raise ValueError(f"{source}: GitHub manifest returned HTTP {response.status}")
    if len(response.body) > max_bytes:
        raise ValueError(f"{source}: manifest exceeds {max_bytes} bytes")
    try:
        payload = response.json()
    except (ValueError, TypeError):
        raise ValueError(f"{source}: GitHub manifest is not valid JSON") from None
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: GitHub manifest must be a JSON object")
    return payload


def _repository(value: Any) -> str:
    text = _text(value, "repository")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text):
        raise ValueError("repository must be an owner/name pair")
    return text


def _manifest_path(value: Any, source: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"data/xml/[A-Za-z0-9._-]+\.xml", value):
        raise ValueError(f"{source}: invalid collection path in frozen manifest")
    return value


def _state_string_map(value: Any, source: str, max_entries: int) -> dict[str, str]:
    if not isinstance(value, Mapping) or len(value) > max_entries:
        raise ValueError(f"{source}: invalid collection_digests checkpoint")
    result: dict[str, str] = {}
    for key, digest in value.items():
        path = _manifest_path(key, source)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{source}: invalid collection digest")
        result[path] = digest
    return result


def _state_count_map(value: Any, source: str, max_entries: int) -> dict[str, int]:
    if not isinstance(value, Mapping) or len(value) > max_entries:
        raise ValueError(f"{source}: invalid collection_counts checkpoint")
    result: dict[str, int] = {}
    for key, count in value.items():
        path = _manifest_path(key, source)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{source}: invalid collection paper count")
        result[path] = count
    return result


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _required_text(value: Any, field: str, source: str) -> str:
    text = _optional_text(value)
    if not text:
        raise ValueError(f"{source}: {field} must not be empty")
    return text


def _nonnegative_int(value: Any, field: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{source}: invalid {field}")
    return value
