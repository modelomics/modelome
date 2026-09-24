from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import unquote, urlsplit

from modelome.models import ArtifactKind, ModelHint, ModelStatus, SourceRecord
from modelome.normalize import normalize_name


class Extractor(Protocol):
    name: str

    def extract(self, record: SourceRecord) -> Sequence[ModelHint]: ...


@dataclass(frozen=True, slots=True)
class _Candidate:
    raw_name: str
    start: int
    end: int
    confidence: float
    field: str
    quoted: bool = False
    strong_shape: bool = False


class IntroductionCueExtractor:
    """Generate source-backed deep-model candidates from introduction language.

    The extractor knows grammatical cues and generic neural-computation language,
    but no model or architecture names. Broad documents such as papers and blog
    posts must put neural scope in the candidate's sentence. Structured artifacts
    may establish that scope elsewhere in the same document.
    """

    name = "introduction-cues-v4"

    _allowed_kinds = frozenset(
        {
            ArtifactKind.PAPER,
            ArtifactKind.BLOG_POST,
            ArtifactKind.CODE_REPOSITORY,
            ArtifactKind.MODEL_CARD,
            ArtifactKind.PROVIDER_PAGE,
            ArtifactKind.CATALOG_RECORD,
            ArtifactKind.WEB_PAGE,
            ArtifactKind.WEIGHTS,
            ArtifactKind.OTHER,
        }
    )
    _structured_kinds = frozenset(
        {
            ArtifactKind.CODE_REPOSITORY,
            ArtifactKind.MODEL_CARD,
            ArtifactKind.PROVIDER_PAGE,
            ArtifactKind.CATALOG_RECORD,
            ArtifactKind.WEIGHTS,
        }
    )
    _markdown_title_kinds = frozenset(
        {
            ArtifactKind.CODE_REPOSITORY,
            ArtifactKind.MODEL_CARD,
        }
    )

    # A candidate is bounded by syntax, not by a dictionary of known names.
    # Lowercase names require quotation marks; unquoted names need name-like
    # typography and therefore begin with an uppercase letter or a digit.
    _unquoted_name = (
        r"(?P<name>[A-Z0-9][A-Za-z0-9_.+\-]*"
        r"(?:\s+[A-Za-z0-9][A-Za-z0-9_.+\-]*){0,9}?)"
    )
    # Unlike the broad form above, this expression cannot absorb a sentence
    # terminator before a following capitalized word.  It is used for the
    # self-naming construction where a bare model name often ends a sentence.
    _unquoted_terminal_name = (
        r"(?P<name>[A-Z0-9][A-Za-z0-9_+\-]*(?:\.[A-Za-z0-9_+\-]+)*"
        r"(?:\s+[A-Za-z0-9][A-Za-z0-9_+\-]*(?:\.[A-Za-z0-9_+\-]+)*){0,9}?)"
    )
    _quoted_name = (
        r"[\"'\u2018\u201c](?P<name>[\w][\w.+\-]*"
        r"(?:[ \t]+[\w][\w.+\-]*){0,9})[\"'\u2019\u201d]"
    )
    _name_end = (
        r"(?=\s*(?:[,;:()\[\]\u2013\u2014]|"
        r"(?i:\b(?:is|was|are|were|for|that|which|to|with|as)\b)|"
        r"[.!?](?:\s|$)|$))"
    )
    _intro_verb = (
        r"(?i:introduc(?:e|es|ed)|present(?:s|ed)?|propos(?:e|es|ed)|"
        r"develop(?:s|ed)?|describ(?:e|es|ed)|report(?:s|ed)?|"
        r"releas(?:e|es|ed)|design(?:s|ed)?|creat(?:e|es|ed)|"
        r"build|builds|built)"
    )
    _intro_subject = (
        rf"(?:"
        rf"(?:(?i:here|in\s+this\s+(?:paper|work|study|article))\s*,?\s*)?"
        rf"(?i:we)\s+{_intro_verb}"
        rf"|(?i:this\s+(?:paper|work|study|article))\s+{_intro_verb}"
        rf")"
    )
    _entity_head = (
        r"(?i:model|network|architecture|system|framework|method|approach|"
        r"operator|surrogate|learner)"
    )
    _neural_modifier = r"(?:(?i:deep|neural)(?:[\s-]+(?i:deep|neural))?[\s-]+)?"
    _naming_verb = r"(?i:called|named|dubbed|termed)"

    _patterns: tuple[tuple[re.Pattern[str], bool, bool, float], ...] = (
        # Direct introductions: "we developed X, a neural network ..."
        (
            re.compile(rf"\b{_intro_subject}\s+{_quoted_name}{_name_end}", re.MULTILINE),
            True,
            False,
            0.80,
        ),
        (
            re.compile(rf"\b{_intro_subject}\s+{_unquoted_name}{_name_end}", re.MULTILINE),
            False,
            False,
            0.78,
        ),
        # Named model-like entities: "a neural operator, which we called X".
        (
            re.compile(
                rf"\b{_entity_head}\s*,?\s*"
                rf"(?:(?i:(?:which|that)\s+(?:we\s+)?|(?:is|was)\s+))?"
                rf"{_naming_verb}\s+{_quoted_name}{_name_end}",
                re.MULTILINE,
            ),
            True,
            False,
            0.82,
        ),
        (
            re.compile(
                rf"\b{_entity_head}\s*,?\s*"
                rf"(?:(?i:(?:which|that)\s+(?:we\s+)?|(?:is|was)\s+))?"
                rf"{_naming_verb}\s+{_unquoted_name}{_name_end}",
                re.MULTILINE,
            ),
            False,
            False,
            0.80,
        ),
        # The introduced object can be generic, with its name in an appositive:
        # "we propose a network architecture, the X, based on attention ...".
        (
            re.compile(
                rf"\b{_intro_subject}[^.!?\n]{{1,220}}?,\s+"
                rf"(?:(?i:the|called|named|dubbed|termed)\s+)?"
                rf"{_quoted_name}{_name_end}",
                re.MULTILINE,
            ),
            True,
            False,
            0.78,
        ),
        (
            re.compile(
                rf"\b{_intro_subject}[^.!?\n]{{1,220}}?,\s+"
                rf"(?:(?i:the|called|named|dubbed|termed)\s+)?"
                rf"{_unquoted_name}{_name_end}",
                re.MULTILINE,
            ),
            False,
            False,
            0.76,
        ),
        # Self-naming forms are common in abstracts, but their introduced
        # object precedes the name: "we call our neural model X".  Requiring
        # an explicit model-like object keeps this from treating arbitrary
        # quoted prose as a technique name.  ``_has_neural_scope`` still
        # applies sentence-local neural evidence before admitting a candidate.
        (
            re.compile(
                rf"\b(?i:we)\s+(?i:call|name|refer\s+to)\s+"
                rf"(?:(?i:our|the|this)\s+)?"
                rf"{_neural_modifier}{_entity_head}"
                rf"(?:\s+(?i:architecture|model|network|system|framework|"
                rf"method|approach|operator|surrogate|learner)){{0,2}}\s+"
                rf"(?:(?i:as)\s+)?{_quoted_name}{_name_end}",
                re.MULTILINE,
            ),
            True,
            False,
            0.80,
        ),
        (
            re.compile(
                rf"\b(?i:we)\s+(?i:call|name|refer\s+to)\s+"
                rf"(?:(?i:our|the|this)\s+)?"
                rf"{_neural_modifier}{_entity_head}"
                rf"(?:\s+(?i:architecture|model|network|system|framework|"
                rf"method|approach|operator|surrogate|learner)){{0,2}}\s+"
                rf"(?:(?i:as)\s+)?{_unquoted_terminal_name}{_name_end}",
                re.MULTILINE,
            ),
            False,
            False,
            0.78,
        ),
        # A compact apposition is another frequent abstract convention:
        # "Our neural model, X, ...".  Unlike sentence-initial appositives,
        # this is syntactically anchored to an explicit model-like object.
        (
            re.compile(
                rf"\b(?i:(?:our|the|this))\s+"
                rf"{_neural_modifier}{_entity_head}\s*,\s*"
                rf"{_quoted_name}(?=\s*[,\u2013\u2014])",
                re.MULTILINE,
            ),
            True,
            False,
            0.78,
        ),
        (
            re.compile(
                rf"\b(?i:(?:our|the|this))\s+"
                rf"{_neural_modifier}{_entity_head}\s*,\s*"
                rf"{_unquoted_name}(?=\s*[,\u2013\u2014])",
                re.MULTILINE,
            ),
            False,
            False,
            0.76,
        ),
        # Name-first appositives require a stronger shape because sentence-initial
        # prose otherwise looks like a capitalized name.
        (
            re.compile(
                rf"(?:^|[.!?\n]\s*){_quoted_name}\s*[,\u2013\u2014]\s*"
                rf"(?i:an?|the)\s+[^.!?\n]{{1,220}}?[,\u2013\u2014]",
                re.MULTILINE,
            ),
            True,
            True,
            0.74,
        ),
        (
            re.compile(
                rf"(?:^|[.!?\n]\s*){_unquoted_name}\s*[,\u2013\u2014]\s*"
                rf"(?i:an?|the)\s+[^.!?\n]{{1,220}}?[,\u2013\u2014]",
                re.MULTILINE,
            ),
            False,
            True,
            0.72,
        ),
    )
    _quoted_title_prefix = re.compile(
        r"^\s*[\"'\u2018\u201c](?P<name>[\w][\w.+\-]*(?:[ \t]+[\w][\w.+\-]*){0,9})"
        r"[\"'\u2019\u201d]\s*:\s+"
    )
    _title_prefix = re.compile(r"^\s*(?P<name>[^:\n]{2,140}?):\s+")
    _title_family_prefix = re.compile(
        r"^\s*(?P<name>[A-Z][A-Za-z0-9_.+\-]*"
        r"(?:\s+[A-Za-z0-9][A-Za-z0-9_.+\-]*){0,8}?\s+"
        r"(?:[Mm]odels?|[Nn]etworks?|[Oo]perators?|[Aa]rchitectures?))"
        r"\s+(?i:for|to|in|with|from|by|on)\b"
    )
    _title_family_standalone = re.compile(
        r"^\s*(?P<name>[A-Z][A-Za-z0-9_.+\-]*"
        r"(?:\s+[A-Za-z0-9][A-Za-z0-9_.+\-]*){0,8}?\s+"
        r"(?:[Mm]odels?|[Nn]etworks?|[Oo]perators?|[Aa]rchitectures?))\s*$"
    )
    _title_named_verb = re.compile(
        r"^\s*(?P<name>[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*"
        r"(?:[ \-]+\d+(?:\.\d+)?)?)\s+"
        r"(?i:predicts?|enables?|improves?|generates?|learns?|models?)\b"
    )
    _markdown_primary_heading = re.compile(
        r"(?m)^[ \t]{0,3}#[ \t]+"
        r"(?P<name>[A-Za-z0-9][A-Za-z0-9_.+\-]*"
        r"(?:[ \t]+[A-Za-z0-9][A-Za-z0-9_.+\-]*){0,9}?)"
        r"[ \t]*#*[ \t]*$"
    )

    _neural_scope = re.compile(
        r"(?:"
        r"\bdeep[\s-]+learning\b"
        r"|\bdeep[\s-]+neural\b"
        r"|\b(?:artificial[\s-]+)?neural[\s-]+(?:"
        r"networks?|models?|architectures?|operators?|fields?|surrogates?|"
        r"representations?|encoders?|decoders?|learners?|potentials?|states?"
        r")\b"
        r"|\b(?:generative[\s-]+(?:adversarial|diffusion)|denoising[\s-]+diffusion|"
        r"score[\s-]+based)[\s-]+(?:networks?|models?)\b"
        r")",
        re.IGNORECASE,
    )
    _network_architecture = re.compile(r"\bnetwork\s+architecture\b", re.IGNORECASE)
    _attention_mechanism = re.compile(
        r"\battention(?:[\s-]+based)?[\s-]+mechanisms?\b", re.IGNORECASE
    )

    def extract(self, record: SourceRecord) -> Sequence[ModelHint]:
        if record.kind not in self._allowed_kinds:
            return ()

        candidates: list[_Candidate] = []
        quoted_title_match = self._quoted_title_prefix.search(record.title)
        if quoted_title_match:
            candidates.append(
                _Candidate(
                    raw_name=quoted_title_match.group("name"),
                    start=quoted_title_match.start("name"),
                    end=quoted_title_match.end("name"),
                    confidence=0.68,
                    field="title",
                    quoted=True,
                    strong_shape=True,
                )
            )
        else:
            title_match = self._title_prefix.search(record.title)
            if title_match:
                candidates.append(
                    _Candidate(
                        raw_name=title_match.group("name"),
                        start=title_match.start("name"),
                        end=title_match.end("name"),
                        confidence=0.66,
                        field="title",
                        strong_shape=True,
                    )
                )

        for pattern in (self._title_family_prefix, self._title_family_standalone):
            family_match = pattern.search(record.title)
            if family_match and _looks_title_family(family_match.group("name")):
                candidates.append(
                    _Candidate(
                        raw_name=family_match.group("name"),
                        start=family_match.start("name"),
                        end=family_match.end("name"),
                        confidence=0.64,
                        field="title",
                        strong_shape=True,
                    )
                )
        named_verb_match = self._title_named_verb.search(record.title)
        if named_verb_match:
            candidates.append(
                _Candidate(
                    raw_name=named_verb_match.group("name"),
                    start=named_verb_match.start("name"),
                    end=named_verb_match.end("name"),
                    confidence=0.64,
                    field="title",
                    strong_shape=True,
                )
            )

        if record.kind in self._markdown_title_kinds:
            heading_match = self._markdown_primary_heading.search(record.text)
            if heading_match:
                candidates.append(
                    _Candidate(
                        raw_name=heading_match.group("name"),
                        start=heading_match.start("name"),
                        end=heading_match.end("name"),
                        confidence=0.70,
                        field="text",
                        strong_shape=True,
                    )
                )

        for pattern, quoted, strong_shape, confidence in self._patterns:
            for match in pattern.finditer(record.text):
                candidates.append(
                    _Candidate(
                        raw_name=match.group("name"),
                        start=match.start("name"),
                        end=match.end("name"),
                        confidence=confidence,
                        field="text",
                        quoted=quoted,
                        strong_shape=strong_shape,
                    )
                )

        hints: list[ModelHint] = []
        seen: set[str] = set()
        for candidate in candidates:
            name = candidate.raw_name.strip(" \t\r\n\"'\u2018\u2019\u201c\u201d()[]{}")
            normalized = normalize_name(name)
            if (
                not normalized
                or normalized in seen
                or not _looks_distinctive(
                    name,
                    quoted=candidate.quoted,
                    strong_shape=candidate.strong_shape,
                )
                or not self._has_neural_scope(record, candidate)
            ):
                continue
            seen.add(normalized)
            hints.append(
                ModelHint(
                    local_id=(
                        f"{self.name}:{candidate.field}:{candidate.start}:{candidate.end}"
                    ),
                    name=name,
                    status=ModelStatus.CANDIDATE,
                    confidence=candidate.confidence,
                    locator=f"{candidate.field}:{candidate.start}:{candidate.end}",
                )
            )
        return tuple(hints)

    def _has_neural_scope(self, record: SourceRecord, candidate: _Candidate) -> bool:
        if candidate.field == "title" and self._has_named_model_resource(record, candidate):
            return True
        if record.kind in self._structured_kinds:
            context = f"{record.title}\n{record.text}"
        elif candidate.field == "title":
            context = f"{record.title}\n{_first_sentence(record.text)}"
        else:
            context = _containing_sentence(record.text, candidate.start, candidate.end)
        if (
            self._neural_scope.search(context)
            or (
                self._network_architecture.search(context)
                and self._attention_mechanism.search(context)
            )
        ):
            return True
        # Abstracts often split a model's name and its neural description
        # across adjacent sentences ("We introduce X. X is a neural network").
        # Bridge only when the exact name is repeated in the neighboring
        # sentence, so evidence elsewhere in a long paper cannot scope it.
        return candidate.field == "text" and self._named_neighbor_has_neural_scope(
            record.text, candidate
        )

    def _named_neighbor_has_neural_scope(
        self, text: str, candidate: _Candidate
    ) -> bool:
        left_boundary = list(re.finditer(r"[.!?\n]", text[: candidate.start]))
        sentence_start = left_boundary[-1].end() if left_boundary else 0
        right_match = re.search(r"[.!?\n]", text[candidate.end :])
        sentence_end = (
            candidate.end + right_match.start() if right_match else len(text)
        )
        neighbors: list[str] = []
        if sentence_start:
            before = text[: sentence_start - 1]
            prior_boundary = list(re.finditer(r"[.!?\n]", before))
            prior_start = prior_boundary[-1].end() if prior_boundary else 0
            neighbors.append(text[prior_start : sentence_start - 1])
        if right_match:
            next_start = sentence_end + 1
            next_boundary = re.search(r"[.!?\n]", text[next_start :])
            next_end = (
                next_start + next_boundary.start() if next_boundary else len(text)
            )
            neighbors.append(text[next_start:next_end])
        name_pattern = re.compile(
            r"(?<![\w])"
            + r"\s+".join(
                re.escape(part) for part in candidate.raw_name.split()
            )
            + r"(?![\w])",
            re.IGNORECASE,
        )
        return any(
            name_pattern.search(sentence)
            and (
                self._neural_scope.search(sentence)
                or (
                    self._network_architecture.search(sentence)
                    and self._attention_mechanism.search(sentence)
                )
            )
            for sentence in neighbors
        )

    @staticmethod
    def _has_named_model_resource(record: SourceRecord, candidate: _Candidate) -> bool:
        """Accept an explicit model resource only when its URL names the candidate."""

        candidate_name = normalize_name(candidate.raw_name)
        compact_candidate_name = candidate_name.replace(" ", "")
        for link in record.links:
            if link.relation not in {"model_card", "weights"}:
                continue
            path = unquote(urlsplit(link.url).path)
            for segment in path.split("/"):
                segment_name = normalize_name(segment.replace("_", " "))
                if segment_name and (
                    segment_name == candidate_name
                    or segment_name.replace(" ", "") == compact_candidate_name
                ):
                    return True
        return False


def extract_model_hints(
    record: SourceRecord,
    extractors: Iterable[Extractor],
) -> tuple[ModelHint, ...]:
    hints: list[ModelHint] = list(record.models)
    occupied = {hint.local_id for hint in hints}
    for extractor in extractors:
        for hint in extractor.extract(record):
            if hint.local_id not in occupied:
                hints.append(hint)
                occupied.add(hint.local_id)
    return tuple(hints)


def _first_sentence(value: str) -> str:
    boundary = re.search(r"[.!?\n]", value)
    return value if boundary is None else value[: boundary.end()]


def _containing_sentence(value: str, start: int, end: int) -> str:
    left_boundaries = tuple(re.finditer(r"[.!?\n]", value[:start]))
    left = left_boundaries[-1].end() if left_boundaries else 0
    right_boundary = re.search(r"[.!?\n]", value[end:])
    right = end + right_boundary.end() if right_boundary else len(value)
    return value[left:right]


def _looks_distinctive(
    value: str,
    *,
    quoted: bool = False,
    strong_shape: bool = False,
) -> bool:
    words = value.split()
    if not 2 <= len(value) <= 140 or not 1 <= len(words) <= 10:
        return False
    if not any(character.isalpha() for character in value):
        return False
    if not all(character.isalnum() or character in "_.+- " for character in value):
        return False
    if quoted:
        return True

    has_digit = any(character.isdigit() for character in value)
    has_internal_upper = any(
        character.isupper() for word in words for character in word[1:]
    )
    has_acronym = any(len(word) > 1 and word.isupper() for word in words)
    capitalized_words = sum(word[0].isupper() for word in words if word)
    if strong_shape:
        return has_digit or has_internal_upper or has_acronym or capitalized_words >= 2
    return has_digit or any(character.isupper() for character in value)


def _looks_title_family(value: str) -> bool:
    """Reject generic paper headings while retaining named architecture families."""

    words = [word.casefold() for word in value.split()]
    if not words or words[0] in {"a", "an", "the", "this", "that"}:
        return False
    generic_words = {
        "deep",
        "neural",
        "network",
        "networks",
        "model",
        "models",
        "operator",
        "operators",
        "architecture",
        "architectures",
    }
    return any(word not in generic_words for word in words)
