from __future__ import annotations

import base64
import binascii
import html
import ipaddress
import re
import socket
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from modelome.http import HttpClient, HttpFailure, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, extract_urls, identifier_from_url

DEFAULT_MAX_PAGE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_README_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_METADATA_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_ROBOTS_BYTES = 512 * 1024
_README_JSON_OVERHEAD_BYTES = 64 * 1024
AddressResolver = Callable[[str], Iterable[str]]
REFERENCE_WEIGHT_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".gguf", ".h5", ".onnx", ".pt", ".pth", ".safetensors"}
)
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_AWS_BEDROCK_MODEL_CARD_RE = re.compile(
    r"^https://docs\.aws\.amazon\.com/bedrock/latest/userguide/"
    r"model-card-(?P<card_id>[a-z0-9][a-z0-9-]*)\.html$",
    re.IGNORECASE,
)
_AWS_BEDROCK_TABLE_MODEL_ID_RE = re.compile(
    r"\b(?:bedrock-runtime|bedrock-mantle)\s+"
    r"(?P<id>[A-Za-z0-9][A-Za-z0-9._:-]{1,255})",
    re.IGNORECASE,
)
_AWS_BEDROCK_CODE_MODEL_ID_RE = re.compile(
    r"\bmodelId\s*=\s*['\"](?P<id>[A-Za-z0-9][A-Za-z0-9._:-]{1,255})",
    re.IGNORECASE,
)
_OPENAI_MODEL_DOCUMENTATION_RE = re.compile(
    r"^https://developers\.openai\.com/api/docs/models/"
    r"(?P<model_id>[a-z0-9][a-z0-9.-]*)$",
    re.IGNORECASE,
)
_OPENAI_DOCUMENTATION_NON_MODELS = frozenset({"all", "compare"})
_OPENAI_SNAPSHOT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*$", re.IGNORECASE)
_PYTORCH_HUB_MODEL_PAGE_RE = re.compile(
    r"^https://pytorch\.org/hub/(?P<hub_id>[a-z0-9][a-z0-9_-]*)$",
    re.IGNORECASE,
)


class FetchPolicyError(RuntimeError):
    """A fetch was stopped by a crawler policy rather than a network error."""


class UnsafeUrlError(FetchPolicyError, ValueError):
    """A URL could reach a non-public network destination."""


class RobotsDeniedError(FetchPolicyError):
    """The target path is disallowed by the site's robots.txt policy."""


class RobotsUnavailableError(FetchPolicyError):
    """robots.txt could not be evaluated and the configured behavior is deny."""


class ContentTooLargeError(HttpFailure):
    """A response exceeded the fetcher's decoded-content budget."""


class PrivateResourceError(FetchPolicyError):
    """A credential revealed a private resource that this crawl excludes."""


class ArtifactFetcher(Protocol):
    def accepts(self, url: str) -> bool: ...

    def fetch(self, url: str) -> SourceRecord: ...


class RobotsAccessPolicy(Protocol):
    def assert_allowed(self, url: str) -> None: ...


class PublicUrlPolicy:
    """Fail closed unless an HTTP(S) URL resolves entirely to public IP space."""

    def __init__(self, resolver: AddressResolver | None = None) -> None:
        self.resolver = resolver or _resolve_addresses

    def validate(self, url: str, *, resolve: bool = True) -> str:
        if not isinstance(url, str) or not url.strip():
            raise UnsafeUrlError("URL must be a non-empty string")
        try:
            parts = urlsplit(url.strip())
            port = parts.port
        except ValueError as error:
            raise UnsafeUrlError(f"invalid URL: {url!r}") from error

        scheme = parts.scheme.casefold()
        if scheme not in {"http", "https"} or not parts.hostname:
            raise UnsafeUrlError("only absolute HTTP(S) URLs are crawlable")
        if parts.username is not None or parts.password is not None:
            raise UnsafeUrlError("URLs containing credentials are not crawlable")

        host = parts.hostname.casefold().rstrip(".")
        if not host or host == "localhost" or host.endswith(".localhost"):
            raise UnsafeUrlError(f"local hostname is not crawlable: {host or '<empty>'}")
        if "%" in host:
            raise UnsafeUrlError("scoped or escaped hostnames are not crawlable")

        literal = _parse_address(host)
        if literal is not None:
            _require_public_address(literal, host)
        elif resolve:
            try:
                addresses = tuple(self.resolver(host))
            except Exception as error:
                raise UnsafeUrlError(f"could not resolve public host {host!r}: {error}") from error
            if not addresses:
                raise UnsafeUrlError(f"host {host!r} resolved to no addresses")
            for raw_address in addresses:
                address = _parse_address(str(raw_address))
                if address is None:
                    raise UnsafeUrlError(
                        f"resolver returned an invalid address for {host!r}: {raw_address!r}"
                    )
                _require_public_address(address, host)

        netloc = f"[{host}]" if ":" in host else host
        if port is not None:
            netloc = f"{netloc}:{port}"
        sanitized = urlunsplit((scheme, netloc, parts.path, parts.query, ""))
        # The shared canonicalizer does not preserve IPv6 brackets.
        if ":" in host:
            return sanitized
        return canonicalize_url(sanitized)

    def allows(self, url: str, *, resolve: bool = True) -> bool:
        try:
            self.validate(url, resolve=resolve)
        except FetchPolicyError:
            return False
        return True


class WeightReferenceFetcher:
    """Materialize a linked checkpoint as metadata without downloading it."""

    def __init__(self, *, url_policy: PublicUrlPolicy | None = None) -> None:
        self.url_policy = url_policy or PublicUrlPolicy()

    def accepts(self, url: str) -> bool:
        if not self.url_policy.allows(url, resolve=False):
            return False
        return PurePosixPath(urlsplit(url).path).suffix.casefold() in REFERENCE_WEIGHT_SUFFIXES

    def fetch(self, url: str) -> SourceRecord:
        canonical_url = self.url_policy.validate(url, resolve=False)
        filename = PurePosixPath(urlsplit(canonical_url).path).name
        return SourceRecord(
            source_record_id=canonical_url,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonical_url,
            title=filename or canonical_url,
            raw={
                "reference_only": True,
                "suffix": PurePosixPath(urlsplit(canonical_url).path).suffix.casefold(),
            },
            identifiers=(Identifier("url", canonical_url),),
        )


class HuggingFaceModelCardFetcher:
    """Fetch a versioned Hub README as a model-card artifact."""

    def __init__(
        self,
        client: HttpClient | Any | None = None,
        *,
        url_policy: PublicUrlPolicy | None = None,
        robots_policy: RobotsAccessPolicy | None = None,
        robots_failure_mode: str = "deny",
        user_agent: str = "modelome",
        max_page_bytes: int = DEFAULT_MAX_PAGE_BYTES,
        max_robots_bytes: int = DEFAULT_MAX_ROBOTS_BYTES,
    ) -> None:
        self.web = WebPageFetcher(
            client,
            url_policy=url_policy,
            robots_policy=robots_policy,
            robots_failure_mode=robots_failure_mode,
            user_agent=user_agent,
            max_page_bytes=max_page_bytes,
            max_robots_bytes=max_robots_bytes,
        )

    def accepts(self, url: str) -> bool:
        return _huggingface_raw_readme_repository(url) is not None and self.web.accepts(url)

    def fetch(self, url: str) -> SourceRecord:
        repository = _huggingface_raw_readme_repository(url)
        if repository is None:
            raise ValueError(f"not a versioned Hugging Face README URL: {url!r}")
        fetched = self.web.fetch(url)
        repository_identifier = Identifier("huggingface:model", repository)
        identifiers = tuple(dict.fromkeys((repository_identifier, *fetched.identifiers)))
        model = ModelHint(
            local_id=f"{repository}#model-card-model",
            name=repository,
            identifiers=(repository_identifier,),
            status=ModelStatus.RELEASED,
            locator="url:repository",
        )
        return SourceRecord(
            source_record_id=fetched.source_record_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=fetched.canonical_url,
            title=f"{repository} model card",
            raw={**dict(fetched.raw), "model_repository": repository},
            text=fetched.text,
            published_at=fetched.published_at,
            modified_at=fetched.modified_at,
            identifiers=identifiers,
            links=fetched.links,
            models=(model,),
        )


class AwsBedrockModelCardFetcher:
    """Resolve one public AWS Bedrock card without account-scoped APIs.

    The public catalog emits stable ``model-card-*.html`` URLs but does not put
    Bedrock's invocation ID in its catalog row.  Each card's source-declared
    Programmatic Access table supplies that ID.  Retaining both identities on
    one model hint lets a later entry build bridge the catalog card and the
    resolved card exactly, rather than guessing from a display name.
    """

    def __init__(
        self,
        client: HttpClient | Any | None = None,
        *,
        url_policy: PublicUrlPolicy | None = None,
        robots_policy: RobotsAccessPolicy | None = None,
        robots_failure_mode: str = "deny",
        user_agent: str = "modelome",
        max_page_bytes: int = DEFAULT_MAX_PAGE_BYTES,
        max_robots_bytes: int = DEFAULT_MAX_ROBOTS_BYTES,
    ) -> None:
        self.web = WebPageFetcher(
            client,
            url_policy=url_policy,
            robots_policy=robots_policy,
            robots_failure_mode=robots_failure_mode,
            user_agent=user_agent,
            max_page_bytes=max_page_bytes,
            max_robots_bytes=max_robots_bytes,
        )

    def accepts(self, url: str) -> bool:
        return _aws_bedrock_model_card_id(url) is not None and self.web.accepts(url)

    def fetch(self, url: str) -> SourceRecord:
        card_id = _aws_bedrock_model_card_id(url)
        if card_id is None:
            raise ValueError(f"not an AWS Bedrock model-card URL: {url!r}")
        fetched = self.web.fetch(url)
        canonical_card_id = _aws_bedrock_model_card_id(fetched.canonical_url)
        if canonical_card_id is None or canonical_card_id != card_id:
            raise ValueError("AWS Bedrock card redirect changed the requested card identity")

        card_identifier = Identifier("aws:bedrock-model-card", card_id)
        model_ids = _aws_bedrock_model_ids(fetched.text)
        model_identifiers = tuple(
            dict.fromkeys(
                (
                    card_identifier,
                    *(Identifier("aws:bedrock:model", value) for value in model_ids),
                )
            )
        )
        identifiers = tuple(dict.fromkeys((*model_identifiers, *fetched.identifiers)))
        title = _aws_bedrock_card_title(fetched.title, card_id)
        model = ModelHint(
            local_id=f"aws-bedrock:{card_id}#model",
            name=title,
            identifiers=model_identifiers,
            status=ModelStatus.DOCUMENTED,
            locator="url:model-card",
        )
        return SourceRecord(
            source_record_id=f"aws-bedrock:{card_id}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=fetched.canonical_url,
            title=title,
            raw={
                **dict(fetched.raw),
                "bedrock_model_card": card_id,
                "bedrock_model_ids": list(model_ids),
            },
            text=fetched.text,
            published_at=fetched.published_at,
            modified_at=fetched.modified_at,
            identifiers=identifiers,
            links=fetched.links,
            models=(model,),
        )


class OpenAIModelDocumentationFetcher:
    """Resolve one public OpenAI model page without an account API request.

    The official all-models page supplies the exact API model ID in each card's
    URL, including current and deprecated entries. A detail page can also
    declare version-locked snapshots. Keeping the API ID on the resolved card
    makes the public documentation observation merge exactly with an optional
    public-owner Models API observation, while snapshots remain releases rather
    than separate model identities.
    """

    def __init__(
        self,
        client: HttpClient | Any | None = None,
        *,
        url_policy: PublicUrlPolicy | None = None,
        robots_policy: RobotsAccessPolicy | None = None,
        robots_failure_mode: str = "deny",
        user_agent: str = "modelome",
        max_page_bytes: int = DEFAULT_MAX_PAGE_BYTES,
        max_robots_bytes: int = DEFAULT_MAX_ROBOTS_BYTES,
    ) -> None:
        self.web = WebPageFetcher(
            client,
            url_policy=url_policy,
            robots_policy=robots_policy,
            robots_failure_mode=robots_failure_mode,
            user_agent=user_agent,
            max_page_bytes=max_page_bytes,
            max_robots_bytes=max_robots_bytes,
        )

    def accepts(self, url: str) -> bool:
        return _openai_model_documentation_id(url) is not None and self.web.accepts(url)

    def fetch(self, url: str) -> SourceRecord:
        model_id = _openai_model_documentation_id(url)
        if model_id is None:
            raise ValueError(f"not an OpenAI model-documentation URL: {url!r}")
        fetched = self.web.fetch(url)
        canonical_model_id = _openai_model_documentation_id(fetched.canonical_url)
        if canonical_model_id is None or canonical_model_id != model_id:
            raise ValueError("OpenAI model documentation redirect changed the requested identity")

        local_id = f"openai-documentation:{model_id}#model"
        model_identifier = Identifier("openai:model", model_id)
        snapshots = _openai_snapshot_ids(fetched.text, model_id)
        releases = tuple(
            ReleaseHint(
                local_id=f"{local_id}:snapshot:{snapshot}",
                model_local_id=local_id,
                version=snapshot,
                identifiers=(Identifier("openai:model-snapshot", snapshot),),
                locator="heading:Snapshots",
            )
            for snapshot in snapshots
        )
        title = _openai_model_documentation_title(fetched.title, model_id)
        return SourceRecord(
            source_record_id=f"openai-documentation:{model_id}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=fetched.canonical_url,
            title=title,
            raw={
                **dict(fetched.raw),
                "openai_model_id": model_id,
                "openai_model_snapshots": list(snapshots),
            },
            text=fetched.text,
            published_at=fetched.published_at,
            modified_at=fetched.modified_at,
            identifiers=tuple(dict.fromkeys((model_identifier, *fetched.identifiers))),
            # Documentation navigation is not evidence that its links are this
            # model's paper, code, or weights. The catalog's model-scoped link
            # already preserves this page as the provider-card resource.
            links=(),
            models=(
                ModelHint(
                    local_id=local_id,
                    name=title,
                    aliases=(model_id,) if title != model_id else (),
                    identifiers=(model_identifier,),
                    status=ModelStatus.DOCUMENTED,
                    locator="url:model-documentation",
                ),
            ),
            releases=releases,
        )


class PyTorchHubModelPageFetcher:
    """Resolve one curated PyTorch Hub page without treating navigation as evidence.

    PyTorch Hub cards identify reproducibility-oriented model pages but a page's
    navigation is not a resource list for that model. Only direct code,
    publication, Hugging Face card, or checkpoint references are retained. The
    same opaque Hub-page identity as the catalog row makes this bounded detail
    record merge exactly without assuming it is the same as an upstream model.
    """

    def __init__(
        self,
        client: HttpClient | Any | None = None,
        *,
        url_policy: PublicUrlPolicy | None = None,
        robots_policy: RobotsAccessPolicy | None = None,
        robots_failure_mode: str = "deny",
        user_agent: str = "modelome",
        max_page_bytes: int = DEFAULT_MAX_PAGE_BYTES,
        max_robots_bytes: int = DEFAULT_MAX_ROBOTS_BYTES,
    ) -> None:
        self.web = WebPageFetcher(
            client,
            url_policy=url_policy,
            robots_policy=robots_policy,
            robots_failure_mode=robots_failure_mode,
            user_agent=user_agent,
            max_page_bytes=max_page_bytes,
            max_robots_bytes=max_robots_bytes,
        )

    def accepts(self, url: str) -> bool:
        return _pytorch_hub_model_page_id(url) is not None and self.web.accepts(url)

    def fetch(self, url: str) -> SourceRecord:
        hub_id = _pytorch_hub_model_page_id(url)
        if hub_id is None:
            raise ValueError(f"not a PyTorch Hub model-page URL: {url!r}")
        fetched = self.web.fetch(url)
        canonical_hub_id = _pytorch_hub_model_page_id(fetched.canonical_url)
        if canonical_hub_id is None or canonical_hub_id != hub_id:
            raise ValueError("PyTorch Hub redirect changed the requested card identity")

        local_id = f"pytorch-hub:{hub_id}#model"
        model_identifier = Identifier("pytorch:hub-model-page", hub_id)
        title = _pytorch_hub_page_title(fetched.title, hub_id)
        return SourceRecord(
            source_record_id=f"pytorch-hub:{hub_id}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=fetched.canonical_url,
            title=title,
            raw={**dict(fetched.raw), "pytorch_hub_page": hub_id},
            text=fetched.text,
            published_at=fetched.published_at,
            modified_at=fetched.modified_at,
            identifiers=tuple(dict.fromkeys((model_identifier, *fetched.identifiers))),
            links=_pytorch_hub_resource_links(fetched.links),
            models=(
                ModelHint(
                    local_id=local_id,
                    name=title,
                    aliases=(hub_id,) if title != hub_id else (),
                    identifiers=(model_identifier,),
                    status=ModelStatus.DOCUMENTED,
                    locator="url:pytorch-hub-model-page",
                ),
            ),
        )


class NvidiaNgcModelCardFetcher:
    """Fetch one source-declared NGC version card as structured JSON evidence.

    NGC's public version endpoint carries model-card Markdown directly.  This
    fetcher is deliberately limited to exact version metadata URLs emitted by
    :class:`NgcModelsSourceAdapter`; it is not a catalog enumerator and never
    derives or downloads an archive URL from NGC's CLI command.
    """

    _HOST = "api.ngc.nvidia.com"

    def __init__(
        self,
        client: HttpClient | Any | None = None,
        *,
        url_policy: PublicUrlPolicy | None = None,
        max_response_bytes: int = DEFAULT_MAX_PAGE_BYTES,
    ) -> None:
        if max_response_bytes < 1:
            raise ValueError("NGC model-card response limit must be positive")
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.url_policy = url_policy or PublicUrlPolicy()
        self.max_response_bytes = max_response_bytes

    def accepts(self, url: str) -> bool:
        if not self.url_policy.allows(url, resolve=False):
            return False
        return _ngc_version_target(url) is not None

    def fetch(self, url: str) -> SourceRecord:
        metadata_url = self.url_policy.validate(url)
        target = _ngc_version_target(metadata_url)
        if target is None:
            raise ValueError(f"not an NGC model-version metadata URL: {url!r}")
        org_name, team_name, model_name, version_id = target
        response: HttpResponse = self.client.get(
            metadata_url,
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            raise ValueError(f"NGC model metadata returned HTTP {response.status}")
        response_url = self.url_policy.validate(response.url or metadata_url)
        response_target = _ngc_version_target(response_url)
        if response_target != target:
            raise ValueError("NGC metadata redirect changed the requested model version")
        _bounded_body(response, self.max_response_bytes, "NGC model metadata")
        try:
            payload = response.json()
        except (TypeError, ValueError) as error:
            raise ValueError("NGC model metadata is not valid JSON") from error
        if not isinstance(payload, Mapping):
            raise ValueError("NGC model metadata is not a JSON object")
        model = payload.get("model")
        model_version = payload.get("modelVersion")
        if not isinstance(model, Mapping) or not isinstance(model_version, Mapping):
            raise ValueError("NGC model metadata lacks model or modelVersion objects")
        _validate_ngc_metadata_identity(
            model,
            model_version,
            org_name,
            team_name,
            model_name,
            version_id,
        )

        resource_id = "/".join(
            (org_name, *([team_name] if team_name else []), model_name)
        )
        model_identifier = Identifier("ngc:model", resource_id)
        version_identifier = Identifier(
            "ngc:model-version",
            f"{resource_id}:{version_id}",
        )
        local_id = f"{resource_id}#model"
        title = _text(model.get("displayName")) or model_name
        description = _text(model.get("description"))
        card_url = _ngc_model_card_url(org_name, team_name, model_name)
        links = [
            Link(metadata_url, relation="model_card_metadata", locator="url", crawl=False),
            Link(card_url, relation="model_card", locator="url", crawl=False),
        ]
        for declared_url in extract_urls(description):
            if not self.url_policy.allows(declared_url, resolve=False):
                continue
            link_url = self.url_policy.validate(declared_url, resolve=False)
            if link_url in {metadata_url, card_url}:
                continue
            links.append(
                Link(
                    link_url,
                    relation=_ngc_link_relation(link_url),
                    locator="$.model.description",
                )
            )

        release = ReleaseHint(
            local_id=f"{resource_id}#version:{version_id}",
            model_local_id=local_id,
            version=version_id,
            identifiers=(version_identifier,),
            released_at=_text(model_version.get("createdDate")) or None,
            metadata=_ngc_release_metadata(model_version),
            locator="$.modelVersion.versionId",
        )
        return SourceRecord(
            source_record_id=f"{resource_id}:{version_id}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=card_url,
            title=title,
            raw={
                "metadata_url": metadata_url,
                "model": dict(model),
                "model_version": dict(model_version),
            },
            text=description,
            modified_at=_text(model_version.get("createdDate")) or None,
            identifiers=(model_identifier, version_identifier),
            links=_unique_ngc_links(links),
            models=(
                ModelHint(
                    local_id=local_id,
                    name=title,
                    aliases=(resource_id,) if title != resource_id else (),
                    identifiers=(model_identifier,),
                    status=ModelStatus.RELEASED,
                    locator="$.model.name",
                ),
            ),
            releases=(release,),
        )


@dataclass(frozen=True, slots=True)
class _RobotsCacheEntry:
    parser: RobotFileParser | None = None
    failure: str | None = None
    allow_on_failure: bool = False


class RobotsTxtPolicy:
    """Fetch and cache one robots.txt decision table per URL origin.

    ``failure_mode`` is explicit: ``"deny"`` (the default) blocks a page when
    robots.txt cannot be evaluated, while ``"allow"`` continues. A 4xx robots
    response is treated as an unavailable policy and allows access per RFC 9309;
    rate limiting (429) remains a failure.
    """

    def __init__(
        self,
        client: HttpClient | Any,
        *,
        url_policy: PublicUrlPolicy,
        user_agent: str = "modelome",
        failure_mode: str = "deny",
        max_bytes: int = DEFAULT_MAX_ROBOTS_BYTES,
    ) -> None:
        if failure_mode not in {"allow", "deny"}:
            raise ValueError("robots failure_mode must be 'allow' or 'deny'")
        if max_bytes < 1:
            raise ValueError("robots max_bytes must be positive")
        self.client = client
        self.url_policy = url_policy
        self.user_agent = user_agent
        self.failure_mode = failure_mode
        self.max_bytes = max_bytes
        self._cache: dict[str, _RobotsCacheEntry] = {}

    def assert_allowed(self, url: str) -> None:
        target = self.url_policy.validate(url)
        parts = urlsplit(target)
        origin = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        entry = self._cache.get(origin)
        if entry is None:
            entry = self._load(origin)
            self._cache[origin] = entry

        if entry.failure:
            if entry.allow_on_failure or self.failure_mode == "allow":
                return
            raise RobotsUnavailableError(f"robots.txt unavailable for {origin}: {entry.failure}")
        if entry.parser is not None and not entry.parser.can_fetch(self.user_agent, target):
            raise RobotsDeniedError(f"robots.txt disallows {target}")

    def _load(self, origin: str) -> _RobotsCacheEntry:
        robots_url = f"{origin}/robots.txt"
        try:
            self.url_policy.validate(robots_url)
            response: HttpResponse = self.client.get(
                robots_url,
                headers={"Accept": "text/plain,*/*;q=0.1"},
            )
            self.url_policy.validate(response.url or robots_url)
            if response.status >= 400:
                if 400 <= response.status < 500 and response.status != 429:
                    return _RobotsCacheEntry(
                        failure=f"HTTP {response.status}",
                        allow_on_failure=True,
                    )
                raise HttpFailure(f"GET {robots_url} returned HTTP {response.status}")
            body = _bounded_body(response, self.max_bytes, "robots.txt")
            parser = RobotFileParser(robots_url)
            parser.parse(body.decode("utf-8", errors="replace").splitlines())
            return _RobotsCacheEntry(parser=parser)
        except (FetchPolicyError, HttpFailure, OSError, UnicodeError, ValueError) as error:
            status = _http_error_status(error)
            if status is not None and 400 <= status < 500 and status != 429:
                return _RobotsCacheEntry(failure=f"HTTP {status}", allow_on_failure=True)
            return _RobotsCacheEntry(failure=str(error) or type(error).__name__)


class GitHubRepositoryFetcher:
    """Enrich a repository discovered in paper, model-card, or catalog evidence."""

    def __init__(
        self,
        client: HttpClient | Any | None = None,
        *,
        token: str | None = None,
        include_private: bool = False,
        api_url: str = "https://api.github.com",
        url_policy: PublicUrlPolicy | None = None,
        max_metadata_bytes: int = DEFAULT_MAX_METADATA_BYTES,
        max_readme_bytes: int = DEFAULT_MAX_README_BYTES,
    ) -> None:
        if min(max_metadata_bytes, max_readme_bytes) < 1:
            raise ValueError("GitHub response limits must be positive")
        self.client = client or HttpClient(
            max_response_bytes=max(
                max_metadata_bytes,
                _encoded_readme_response_limit(max_readme_bytes),
            )
        )
        self.token = token or ""
        self.include_private = bool(include_private)
        self.api_url = api_url.rstrip("/")
        self.url_policy = url_policy or PublicUrlPolicy()
        self.max_metadata_bytes = max_metadata_bytes
        self.max_readme_bytes = max_readme_bytes

    def accepts(self, url: str) -> bool:
        if not self.url_policy.allows(url, resolve=False):
            return False
        identifier = identifier_from_url(url)
        return identifier is not None and identifier.namespace == "github:repository"

    def fetch(self, url: str) -> SourceRecord:
        safe_url = self.url_policy.validate(url)
        identifier = identifier_from_url(safe_url)
        if identifier is None or identifier.namespace != "github:repository":
            raise ValueError(f"not a GitHub repository URL: {url}")
        repository = identifier.value
        encoded_repository = "/".join(quote(part, safe="") for part in repository.split("/", 1))
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        metadata_url = f"{self.api_url}/repos/{encoded_repository}"
        self.url_policy.validate(metadata_url)
        metadata_response: HttpResponse = self.client.get(metadata_url, headers=headers)
        self.url_policy.validate(metadata_response.url or metadata_url)
        metadata_body = _bounded_body(
            metadata_response,
            self.max_metadata_bytes,
            "GitHub repository metadata",
        )
        try:
            metadata = metadata_response.json()
        except (TypeError, ValueError) as error:
            raise ValueError(f"GitHub returned invalid metadata for {repository}") from error
        if not isinstance(metadata, Mapping):
            raise ValueError(f"GitHub returned a non-object for {repository}")
        # Keep the explicit bound adjacent to decoding even if an injected
        # response implementation ignores its body in ``json()``.
        if len(metadata_body) > self.max_metadata_bytes:
            raise ContentTooLargeError("GitHub repository metadata exceeded its byte limit")
        if metadata.get("private") is True and not self.include_private:
            raise PrivateResourceError("private GitHub repository excluded by crawl policy")

        readme = self._readme(repository, encoded_repository, headers)
        reported_url = str(metadata.get("html_url") or safe_url)
        try:
            canonical_url = self.url_policy.validate(reported_url)
        except UnsafeUrlError:
            canonical_url = safe_url
        raw = dict(metadata)
        if readme:
            raw["_modelome_readme"] = readme

        links = _links_from_values(
            canonical_url,
            metadata.get("homepage"),
            _nested(metadata, "parent", "html_url"),
            _nested(metadata, "source", "html_url"),
            readme,
            url_policy=self.url_policy,
        )
        description = str(metadata.get("description") or "").strip()
        text = "\n\n".join(part for part in (description, readme) if part)
        identifiers = [Identifier("github:repository", repository)]
        current_repository_identifier = identifier_from_url(canonical_url)
        if (
            current_repository_identifier is not None
            and current_repository_identifier.namespace == "github:repository"
            and current_repository_identifier not in identifiers
        ):
            identifiers.append(current_repository_identifier)
        repository_id = metadata.get("id")
        if (
            isinstance(repository_id, int)
            and not isinstance(repository_id, bool)
            and repository_id >= 0
        ):
            identifiers.append(Identifier("github:repository-id", str(repository_id)))
        node_id = _optional_text(metadata.get("node_id"))
        if node_id:
            identifiers.append(Identifier("github:node-id", node_id))
        if safe_url != canonical_url:
            identifiers.append(Identifier("url", safe_url))
        return SourceRecord(
            source_record_id=repository.casefold(),
            kind=ArtifactKind.CODE_REPOSITORY,
            canonical_url=canonical_url,
            title=str(metadata.get("full_name") or repository),
            text=text,
            raw=raw,
            published_at=_optional_text(metadata.get("created_at")),
            modified_at=_optional_text(metadata.get("pushed_at") or metadata.get("updated_at")),
            identifiers=tuple(identifiers),
            links=links,
        )

    def _readme(
        self,
        repository: str,
        encoded_repository: str,
        headers: Mapping[str, str],
    ) -> str:
        readme_headers = dict(headers)
        readme_headers["Accept"] = "application/vnd.github.raw+json"
        readme_url = f"{self.api_url}/repos/{encoded_repository}/readme"
        try:
            self.url_policy.validate(readme_url)
            response: HttpResponse = self.client.get(readme_url, headers=readme_headers)
            self.url_policy.validate(response.url or readme_url)
            content_type = _header(response.headers, "content-type").casefold()
            if "json" not in content_type:
                body = _bounded_body(response, self.max_readme_bytes, "GitHub README")
                return body.decode("utf-8", errors="replace")
            body = _bounded_body(
                response,
                _encoded_readme_response_limit(self.max_readme_bytes),
                "GitHub README response",
            )
            try:
                payload = response.json()
            except (TypeError, ValueError):
                return body.decode("utf-8", errors="replace")
            if not isinstance(payload, Mapping):
                return ""
            encoded = payload.get("content")
            if isinstance(encoded, str) and payload.get("encoding") == "base64":
                compact = re.sub(r"\s+", "", encoded)
                if len(compact) > 4 * ((self.max_readme_bytes + 2) // 3) + 4:
                    raise ContentTooLargeError(
                        f"GitHub README for {repository} exceeded {self.max_readme_bytes} bytes"
                    )
                decoded = base64.b64decode(compact, validate=True)
                if len(decoded) > self.max_readme_bytes:
                    raise ContentTooLargeError(
                        f"GitHub README for {repository} exceeded {self.max_readme_bytes} bytes"
                    )
                return decoded.decode("utf-8", errors="replace")
        except (FetchPolicyError, HttpFailure, ValueError, TypeError, binascii.Error):
            # README enrichment is optional; repository metadata remains useful.
            return ""
        return ""


class WebPageFetcher:
    """Fetch one public web page as evidence, subject to robots.txt."""

    def __init__(
        self,
        client: HttpClient | Any | None = None,
        *,
        url_policy: PublicUrlPolicy | None = None,
        robots_policy: RobotsAccessPolicy | None = None,
        robots_failure_mode: str = "deny",
        user_agent: str = "modelome",
        max_page_bytes: int = DEFAULT_MAX_PAGE_BYTES,
        max_robots_bytes: int = DEFAULT_MAX_ROBOTS_BYTES,
    ) -> None:
        if min(max_page_bytes, max_robots_bytes) < 1:
            raise ValueError("web response limits must be positive")
        self.client = client or HttpClient(
            max_response_bytes=max(max_page_bytes, max_robots_bytes)
        )
        self.url_policy = url_policy or PublicUrlPolicy()
        self.robots_policy = robots_policy or RobotsTxtPolicy(
            self.client,
            url_policy=self.url_policy,
            user_agent=user_agent,
            failure_mode=robots_failure_mode,
            max_bytes=max_robots_bytes,
        )
        self.max_page_bytes = max_page_bytes

    def accepts(self, url: str) -> bool:
        return self.url_policy.allows(url, resolve=False)

    def fetch(self, url: str) -> SourceRecord:
        requested_url = self.url_policy.validate(url)
        self.robots_policy.assert_allowed(requested_url)
        request_headers = {
            "Accept": (
                "text/html,application/xhtml+xml;q=0.9,"
                "application/json;q=0.8,*/*;q=0.1"
            )
        }
        if isinstance(self.client, HttpClient):
            response: HttpResponse = self.client.get(
                requested_url,
                headers=request_headers,
                redirect_validator=self.robots_policy.assert_allowed,
            )
        else:
            response = self.client.get(requested_url, headers=request_headers)
        response_url = self.url_policy.validate(response.url or requested_url)
        if _origin(response_url) != _origin(requested_url):
            self.robots_policy.assert_allowed(response_url)
        body = _bounded_body(response, self.max_page_bytes, "web page")
        content_type = _header(response.headers, "content-type").casefold()
        canonical_url = response_url
        if "html" in content_type or body.lstrip().startswith(b"<"):
            parser = _PageParser(response_url, self.url_policy)
            parser.feed(body.decode("utf-8", errors="replace"))
            title = parser.title or canonical_url
            text = _page_text(parser.metadata, parser.text)
            links = tuple(
                dict.fromkeys(
                    (
                        *(
                            Link(url=item, relation="references", locator="html:link")
                            for item in parser.links
                        ),
                        *_scholarly_links(parser.metadata, self.url_policy),
                    )
                )
            )
            kind = _page_kind(parser.metadata)
            raw: Mapping[str, Any] = {
                "metadata": parser.metadata,
                "canonical_url": parser.canonical_url,
            }
            if parser.canonical_url:
                with suppress(UnsafeUrlError):
                    canonical_url = self.url_policy.validate(parser.canonical_url)
        else:
            title = canonical_url
            text = (
                body.decode("utf-8", errors="replace")
                if "json" in content_type or "text" in content_type
                else ""
            )
            links = tuple(
                Link(url=item, relation="references", locator="body:url")
                for item in extract_urls(text)
                if self.url_policy.allows(item, resolve=False)
            )
            kind = ArtifactKind.OTHER
            raw = {"content_type": content_type, "size": len(body)}

        identifiers: list[Identifier] = []
        if identifier := identifier_from_url(canonical_url):
            identifiers.append(identifier)
        if (
            ("html" in content_type or body.lstrip().startswith(b"<"))
            and (doi := _metadata_doi(parser.metadata))
        ):
            identifiers.append(Identifier("doi", doi))
        for alias in (requested_url, response_url):
            if alias != canonical_url:
                identifiers.append(Identifier("url", alias))
        return SourceRecord(
            source_record_id=canonical_url,
            kind=kind,
            canonical_url=canonical_url,
            title=title,
            text=text,
            raw=raw,
            identifiers=tuple(dict.fromkeys(identifiers)),
            links=links,
        )


class _PageParser(HTMLParser):
    def __init__(self, base_url: str, url_policy: PublicUrlPolicy) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.url_policy = url_policy
        self._in_title = False
        self._hidden_depth = 0
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._links: list[str] = []
        self.metadata: dict[str, str] = {}
        self.canonical_url: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.casefold(): value or "" for key, value in attrs}
        tag = tag.casefold()
        if tag == "title":
            self._in_title = True
        if tag in {"script", "style", "noscript", "svg", "template"}:
            self._hidden_depth += 1
        if tag == "a" and attributes.get("href"):
            self._add_link(attributes["href"])
        if tag == "link" and "canonical" in attributes.get("rel", "").casefold().split():
            candidate = urljoin(self.base_url, attributes.get("href", ""))
            if self.url_policy.allows(candidate, resolve=False):
                self.canonical_url = self.url_policy.validate(candidate, resolve=False)
        if tag == "meta":
            key = attributes.get("property") or attributes.get("name")
            value = attributes.get("content")
            if key and value:
                self.metadata[key.casefold()] = html.unescape(value).strip()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "noscript", "svg", "template"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        value = re.sub(r"\s+", " ", data).strip()
        if not value:
            return
        if self._in_title:
            self._title_parts.append(value)
        if not self._hidden_depth:
            self._text_parts.append(value)

    @property
    def title(self) -> str:
        return " ".join(self._title_parts).strip()

    @property
    def text(self) -> str:
        return "\n".join(self._text_parts)

    @property
    def links(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._links))

    def _add_link(self, href: str) -> None:
        candidate = urljoin(self.base_url, href)
        if self.url_policy.allows(candidate, resolve=False):
            self._links.append(self.url_policy.validate(candidate, resolve=False))


def _page_kind(metadata: Mapping[str, str]) -> ArtifactKind:
    if any(
        metadata.get(key)
        for key in (
            "citation_title",
            "citation_doi",
            "prism.doi",
            "bepress_citation_doi",
        )
    ):
        return ArtifactKind.PAPER
    dc_type = metadata.get("dc.type", "").casefold()
    if "article" in dc_type or "journal" in dc_type:
        return ArtifactKind.PAPER
    declared = " ".join(
        metadata.get(key, "") for key in ("og:type", "article:section", "twitter:label1")
    ).casefold()
    if "article" in declared or "blog" in declared:
        return ArtifactKind.BLOG_POST
    return ArtifactKind.WEB_PAGE


def _page_text(metadata: Mapping[str, str], visible_text: str) -> str:
    """Add publisher-exposed abstract metadata without treating it as full text."""

    metadata_evidence = tuple(
        value
        for value in (
            metadata.get("citation_abstract", ""),
            metadata.get("description", ""),
            metadata.get("og:description", ""),
        )
        if value
    )
    return "\n".join(dict.fromkeys((*metadata_evidence, visible_text)))


def _metadata_doi(metadata: Mapping[str, str]) -> str | None:
    for key in ("citation_doi", "prism.doi", "bepress_citation_doi", "dc.identifier"):
        value = metadata.get(key, "").strip()
        if not value:
            continue
        if (identifier := identifier_from_url(value)) and identifier.namespace == "doi":
            return identifier.value
        normalized = value.casefold().removeprefix("doi:").strip().rstrip(".,;:!?)\"]}")
        if _DOI_RE.fullmatch(normalized):
            return normalized
    return None


def _scholarly_links(
    metadata: Mapping[str, str],
    url_policy: PublicUrlPolicy,
) -> tuple[Link, ...]:
    """Retain explicit publisher full-text locators without scheduling downloads."""

    links: list[Link] = []
    for key in ("citation_pdf_url", "citation_fulltext_html_url"):
        value = metadata.get(key, "")
        if not value or not url_policy.allows(value, resolve=False):
            continue
        links.append(
            Link(
                url=url_policy.validate(value, resolve=False),
                relation="full_text",
                locator=f"meta:{key}",
                crawl=False,
            )
        )
    return tuple(dict.fromkeys(links))


def _nested(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _links_from_values(
    *values: Any,
    url_policy: PublicUrlPolicy,
) -> tuple[Link, ...]:
    urls: list[str] = []
    for value in values:
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            urls.append(value)
        urls.extend(extract_urls(value))
    safe_urls = (
        url_policy.validate(url, resolve=False)
        for url in dict.fromkeys(urls)
        if url_policy.allows(url, resolve=False)
    )
    return tuple(
        Link(url=url, relation="references", locator="repository:metadata")
        for url in dict.fromkeys(safe_urls)
    )


def _resolve_addresses(host: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            result[4][0]
            for result in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        )
    )


def _parse_address(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _require_public_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    host: str,
) -> None:
    if not address.is_global:
        raise UnsafeUrlError(f"host {host!r} resolves to non-public address {address}")


def _bounded_body(response: HttpResponse, limit: int, label: str) -> bytes:
    if len(response.body) > limit:
        raise ContentTooLargeError(f"{label} exceeded {limit} bytes")
    return response.body


def _encoded_readme_response_limit(decoded_limit: int) -> int:
    encoded_content = 4 * ((decoded_limit + 2) // 3)
    return encoded_content + _README_JSON_OVERHEAD_BYTES


def _http_error_status(error: BaseException) -> int | None:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, HTTPError):
            return current.code
        current = current.__cause__
    return None


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    return parts.scheme.casefold(), (parts.hostname or "").casefold(), parts.port


def _huggingface_raw_readme_repository(url: str) -> str | None:
    parts = urlsplit(url)
    if (parts.hostname or "").casefold() not in {"huggingface.co", "www.huggingface.co"}:
        return None
    segments = [unquote(segment) for segment in parts.path.split("/") if segment]
    if (
        len(segments) < 5
        or segments[2].casefold() not in {"raw", "resolve"}
        or segments[-1].casefold() != "readme.md"
    ):
        return None
    return f"{segments[0]}/{segments[1]}"


def _ngc_version_target(url: str) -> tuple[str, str, str, str] | None:
    parts = urlsplit(url)
    if (parts.hostname or "").casefold() != NvidiaNgcModelCardFetcher._HOST:
        return None
    segments = [unquote(segment) for segment in parts.path.split("/") if segment]
    is_version_path = (
        len(segments) >= 2
        and segments[:2] == ["v2", "models"]
        and segments[-2] == "versions"
    )
    if len(segments) == 7 and is_version_path:
        org_name = segments[2]
        team_name = segments[3]
        model_name = segments[4]
        version_id = segments[6]
    elif len(segments) == 6 and is_version_path:
        org_name, model_name, version_id = segments[2], segments[3], segments[5]
        team_name = ""
    else:
        return None
    has_invalid_segment = any(
        not value or any(character.isspace() for character in value)
        for value in (org_name, team_name, model_name, version_id)
    )
    if has_invalid_segment:
        return None
    return org_name, team_name, model_name, version_id


def _validate_ngc_metadata_identity(
    model: Mapping[str, Any],
    model_version: Mapping[str, Any],
    org_name: str,
    team_name: str,
    model_name: str,
    version_id: str,
) -> None:
    expected = (org_name, team_name, model_name, version_id)
    actual = (
        _text(model.get("orgName")),
        _text(model.get("teamName")),
        _text(model.get("name")),
        _text(model_version.get("versionId")),
    )
    if actual != expected:
        raise ValueError("NGC metadata identity does not match the requested version URL")


def _ngc_model_card_url(org_name: str, team_name: str, model_name: str) -> str:
    segments = ["orgs", quote(org_name, safe="")]
    if team_name:
        segments.append(quote(team_name, safe=""))
    segments.extend(("models", quote(model_name, safe="")))
    return canonicalize_url(f"https://catalog.ngc.nvidia.com/{'/'.join(segments)}")


def _aws_bedrock_model_card_id(url: str) -> str | None:
    try:
        canonical_url = canonicalize_url(url)
    except (TypeError, ValueError):
        return None
    match = _AWS_BEDROCK_MODEL_CARD_RE.fullmatch(canonical_url)
    return match.group("card_id").casefold() if match else None


def _aws_bedrock_model_ids(text: str) -> tuple[str, ...]:
    values = []
    for pattern in (_AWS_BEDROCK_TABLE_MODEL_ID_RE, _AWS_BEDROCK_CODE_MODEL_ID_RE):
        for match in pattern.finditer(text):
            value = match.group("id")
            # A Bedrock invocation ID is provider-qualified. This eliminates
            # prose such as "bedrock-runtime endpoint" without a provider
            # vocabulary or a hard-coded model-name list.
            if "." not in value and ":" not in value:
                continue
            values.append(value)
    return tuple(dict.fromkeys(values))


def _aws_bedrock_card_title(value: str, card_id: str) -> str:
    title = value.split(" - Amazon Bedrock", 1)[0].strip()
    return title or card_id


def _openai_model_documentation_id(url: str) -> str | None:
    try:
        canonical_url = canonicalize_url(url)
    except (TypeError, ValueError):
        return None
    match = _OPENAI_MODEL_DOCUMENTATION_RE.fullmatch(canonical_url)
    if match is None:
        return None
    model_id = match.group("model_id").casefold()
    return model_id if model_id not in _OPENAI_DOCUMENTATION_NON_MODELS else None


def _openai_snapshot_ids(text: str, model_id: str) -> tuple[str, ...]:
    """Read only exact snapshot IDs from the page's named snapshot section."""

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    try:
        start = next(index for index, value in enumerate(lines) if value == "Snapshots")
    except StopIteration:
        return ()
    values = []
    prefix = f"{model_id}-"
    for value in lines[start + 1 :]:
        if value in {"Rate limits", "Ask AI"}:
            break
        if (
            _OPENAI_SNAPSHOT_ID_RE.fullmatch(value)
            and value.casefold().startswith(prefix)
        ):
            values.append(value.casefold())
    return tuple(dict.fromkeys(values))


def _openai_model_documentation_title(value: str, model_id: str) -> str:
    title = value.split(" Model | OpenAI API", 1)[0].strip()
    return title or model_id


def _pytorch_hub_model_page_id(url: str) -> str | None:
    try:
        canonical_url = canonicalize_url(url)
    except (TypeError, ValueError):
        return None
    match = _PYTORCH_HUB_MODEL_PAGE_RE.fullmatch(canonical_url)
    return match.group("hub_id").casefold() if match else None


def _pytorch_hub_page_title(value: str, hub_id: str) -> str:
    for separator in (" – PyTorch", " - PyTorch", " | PyTorch"):
        if separator in value:
            value = value.split(separator, 1)[0]
            break
    return value.strip() or hub_id


def _pytorch_hub_resource_links(links: Iterable[Link]) -> tuple[Link, ...]:
    result = []
    for link in links:
        identifier = identifier_from_url(link.url)
        if PurePosixPath(urlsplit(link.url).path).suffix.casefold() in REFERENCE_WEIGHT_SUFFIXES:
            relation = "weights"
        elif identifier is not None and identifier.namespace in {"arxiv", "doi"}:
            relation = "paper_reference"
        elif identifier is not None and identifier.namespace == "huggingface:model":
            relation = "model_card"
        elif _github_source_reference(link.url):
            relation = "code_reference"
        else:
            continue
        result.append(
            Link(
                link.url,
                relation=relation,
                locator=link.locator,
                crawl=link.crawl,
            )
        )
    return tuple(dict.fromkeys(result))


def _github_source_reference(url: str) -> bool:
    """Accept repository and source-tree URLs, but reject issue/navigation URLs."""

    parts = urlsplit(url)
    if (parts.hostname or "").casefold() not in {"github.com", "www.github.com"}:
        return False
    segments = [unquote(segment) for segment in parts.path.split("/") if segment]
    if len(segments) == 2:
        return True
    return len(segments) >= 3 and segments[2] in {"blob", "raw", "tree"}


def _ngc_link_relation(url: str) -> str:
    identifier = identifier_from_url(url)
    if identifier is not None and identifier.namespace in {"arxiv", "doi"}:
        return "paper_reference"
    if identifier is not None and identifier.namespace == "github:repository":
        return "code_reference"
    host = (urlsplit(url).hostname or "").casefold()
    if host in {"gitlab.com", "www.gitlab.com", "bitbucket.org", "www.bitbucket.org"}:
        return "code_reference"
    return "documentation_reference"


def _ngc_release_metadata(model_version: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value
        for key in ("status", "hasSignedVersion")
        if (value := model_version.get(key)) is not None
        and isinstance(value, str | int | float | bool)
    }


def _unique_ngc_links(values: Iterable[Link]) -> tuple[Link, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda link: (
                link.url,
                link.relation,
                link.locator or "",
                link.crawl,
                link.model_local_ids,
            ),
        )
    )


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _header(headers: Mapping[str, str], name: str) -> str:
    wanted = name.casefold()
    return next((value for key, value in headers.items() if key.casefold() == wanted), "")


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


__all__ = [
    "ArtifactFetcher",
    "AwsBedrockModelCardFetcher",
    "ContentTooLargeError",
    "FetchPolicyError",
    "GitHubRepositoryFetcher",
    "HuggingFaceModelCardFetcher",
    "NvidiaNgcModelCardFetcher",
    "OpenAIModelDocumentationFetcher",
    "PyTorchHubModelPageFetcher",
    "PrivateResourceError",
    "PublicUrlPolicy",
    "RobotsAccessPolicy",
    "RobotsDeniedError",
    "RobotsTxtPolicy",
    "RobotsUnavailableError",
    "UnsafeUrlError",
    "WebPageFetcher",
    "WeightReferenceFetcher",
    "REFERENCE_WEIGHT_SUFFIXES",
]
