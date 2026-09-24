from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

from modelome.models import ArtifactKind, Identifier, Link, SourceRecord
from modelome.normalize import canonicalize_url, content_hash
from modelome.sources.eartharxiv import (
    _OAI,
    EarthArxivSourceAdapter,
    _as_utc,
    _first_text,
    _optional_doi,
    _optional_web_url,
    _required_date,
    _required_element_text,
    _texts,
    _unique_links,
    _web_url,
)

_HAL_OAI_IDENTIFIER = re.compile(r"^oai:HAL:(?P<document_id>[^\s/]+)$")
_VERSION = re.compile(r"^(?P<base>.+?)(?P<version>v[1-9]\d*)$")


class HalSourceAdapter(EarthArxivSourceAdapter):
    """Harvest HAL's complete first-party, date-granularity OAI-PMH stream.

    HAL is a repository of research outputs, including manuscripts, journal
    articles, theses, and conference papers.  It is deliberately not queried
    by subject, publication type, institution, author, or model vocabulary.
    A clean state begins at the repository's declared earliest datestamp and
    follows opaque OAI-PMH resumption tokens through that frozen history.
    """

    def __init__(
        self,
        *,
        name: str = "hal",
        url: str = "https://api.archives-ouvertes.fr/oai/hal/",
        web_base_url: str = "https://hal.science",
        artifact_kind: str | ArtifactKind = ArtifactKind.PAPER,
        initial_start_date: str | date = "2002-09-23",
        overlap_days: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            url=url,
            web_base_url=web_base_url,
            artifact_kind=artifact_kind,
            initial_lookback_days=0,
            overlap_days=overlap_days,
            **kwargs,
        )
        self.initial_start_date = _required_date(
            initial_start_date, "initial_start_date", self.name
        )
        self.checkpoint_signature = content_hash(
            {
                "adapter": "hal-oai-dc-v1",
                "url": self.url,
                "web_base_url": self.web_base_url,
                "artifact_kind": self.artifact_kind.value,
                "initial_start_date": self.initial_start_date.isoformat(),
                "overlap_days": self.overlap_days,
            }
        )

    def _window(
        self,
        state: Mapping[str, Any],
        *,
        prior_watermark: date | None,
        closed_through: date,
    ) -> tuple[date, date, bool]:
        raw_start, raw_end = state.get("window_start"), state.get("window_end")
        if raw_start is not None or raw_end is not None:
            return super()._window(
                state,
                prior_watermark=prior_watermark,
                closed_through=closed_through,
            )
        if "resumption_token" in state:
            raise ValueError(f"{self.name}: token checkpoint is missing its frozen window")
        if prior_watermark is None:
            return self.initial_start_date, closed_through, False
        if prior_watermark > closed_through:
            raise ValueError(f"{self.name}: watermark is later than the last closed UTC day")
        return (
            prior_watermark + timedelta(days=1 - self.overlap_days),
            closed_through,
            False,
        )

    def _list_request_parameters(
        self, window_start: date, window_end: date
    ) -> dict[str, str]:
        # HAL declares YYYY-MM-DD OAI-PMH granularity.  Supplying timestamps
        # would let a provider reject the page or round it unexpectedly.
        return {
            "metadataPrefix": "oai_dc",
            "from": window_start.isoformat(),
            "until": window_end.isoformat(),
        }

    def _record(self, element: ET.Element) -> SourceRecord:
        header = element.find(f"{{{_OAI}}}header")
        if header is None:
            raise ValueError("OAI record is missing its header")
        oai_identifier = _required_element_text(header, "identifier", _OAI, self.name)
        document_id = _hal_document_id(oai_identifier)
        datestamp = _oai_datestamp(
            _required_element_text(header, "datestamp", _OAI, self.name), self.name
        )
        fallback_url = canonicalize_url(f"{self.web_base_url}/{document_id}")
        identifiers = _hal_identifiers(oai_identifier, document_id)
        if header.get("status") == "deleted":
            return SourceRecord(
                source_record_id=f"hal:{oai_identifier}",
                kind=self.artifact_kind,
                canonical_url=fallback_url,
                title=f"Withdrawn HAL research output {document_id}",
                raw={
                    "oai_identifier": oai_identifier,
                    "datestamp": datestamp,
                    "sets": _header_sets(header),
                },
                modified_at=datestamp,
                identifiers=tuple(identifiers),
                links=(
                    Link(
                        fallback_url,
                        relation="repository_record",
                        locator="header.identifier",
                    ),
                ),
                deleted=True,
            )

        metadata = element.find(
            f"{{{_OAI}}}metadata/{{http://www.openarchives.org/OAI/2.0/oai_dc/}}dc"
        )
        if metadata is None:
            raise ValueError("OAI record is missing Dublin Core metadata")
        titles = _texts(metadata, "title")
        title = _first_text(metadata, "title", self.name)
        metadata_identifiers = _texts(metadata, "identifier")
        landing_url = next(
            (
                url
                for value in metadata_identifiers
                if (url := _hal_landing_url(value))
            ),
            fallback_url,
        )
        links = [
            Link(
                landing_url,
                relation="repository_record",
                locator="dc.identifier" if landing_url != fallback_url else "header.identifier",
            )
        ]
        for index, value in enumerate(metadata_identifiers):
            if doi := _optional_doi(value):
                identifiers.append(Identifier("doi", doi))
                links.append(
                    Link(
                        canonicalize_url(f"https://doi.org/{doi}"),
                        relation="doi",
                        locator=f"dc.identifier[{index}]",
                    )
                )
            elif url := _optional_web_url(value):
                relation = "repository_record" if url == landing_url else "full_text"
                links.append(Link(url, relation=relation, locator=f"dc.identifier[{index}]"))
        dates = _texts(metadata, "date")
        return SourceRecord(
            source_record_id=f"hal:{oai_identifier}",
            kind=self.artifact_kind,
            canonical_url=landing_url,
            title=title,
            raw={
                "oai_identifier": oai_identifier,
                "document_id": document_id,
                "datestamp": datestamp,
                "sets": _header_sets(header),
                "titles": titles,
                "creators": _texts(metadata, "creator"),
                "dates": dates,
                "identifiers": metadata_identifiers,
                "subjects": _texts(metadata, "subject"),
                "types": _texts(metadata, "type"),
                "rights": _texts(metadata, "rights"),
            },
            text="\n\n".join(_texts(metadata, "description")),
            published_at=next(
                (timestamp for value in dates if (timestamp := _optional_date_timestamp(value))),
                None,
            ),
            modified_at=datestamp,
            identifiers=tuple(dict.fromkeys(identifiers)),
            links=_unique_links(links),
        )

    def _malformed_id(self, element: ET.Element, index: int) -> str:
        identifier = element.findtext(f"{{{_OAI}}}header/{{{_OAI}}}identifier")
        return f"hal:{identifier.strip()}" if identifier else f"{self.name}:malformed:{index}"


def _hal_document_id(oai_identifier: str) -> str:
    match = _HAL_OAI_IDENTIFIER.fullmatch(oai_identifier)
    if match is None:
        raise ValueError(f"HAL OAI identifier is invalid: {oai_identifier!r}")
    return match.group("document_id")


def _hal_identifiers(oai_identifier: str, document_id: str) -> list[Identifier]:
    identifiers = [Identifier("oai:hal", oai_identifier), Identifier("hal:version", document_id)]
    if match := _VERSION.fullmatch(document_id):
        identifiers.append(Identifier("hal:document", match.group("base")))
    else:
        identifiers.append(Identifier("hal:document", document_id))
    return identifiers


def _header_sets(header: ET.Element) -> list[str]:
    return [
        value
        for element in header.findall(f"{{{_OAI}}}setSpec")
        if (value := element.text.strip() if element.text else "")
    ]


def _oai_datestamp(value: str, source: str) -> str:
    try:
        if len(value) == 10:
            parsed = datetime.combine(
                date.fromisoformat(value), datetime.min.time(), tzinfo=UTC
            )
            return parsed.isoformat().replace("+00:00", "Z")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{source}: invalid OAI-PMH datestamp {value!r}") from None
    return _as_utc(parsed).isoformat().replace("+00:00", "Z")


def _optional_date_timestamp(value: str) -> str | None:
    try:
        return _oai_datestamp(value, "HAL Dublin Core date")
    except ValueError:
        # HAL legitimately publishes year-only and free-form bibliographic dates.
        # Keep them in raw evidence without turning a valid page into a retry loop.
        return None


def _hal_landing_url(value: str) -> str:
    url = _optional_web_url(value)
    if not url:
        return ""
    host = (_web_url(url, "HAL identifier").split("//", 1)[1].split("/", 1)[0]).casefold()
    return url if host == "hal.science" or host.endswith(".hal.science") else ""


__all__ = ["HalSourceAdapter"]
