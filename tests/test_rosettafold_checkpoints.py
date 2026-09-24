from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.sources.rosettafold_checkpoints import RoseTTAFoldCheckpointAdapter

REV = "a" * 40


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def json_response(value: Any, url: str) -> HttpResponse:
    return HttpResponse(200, {}, json.dumps(value).encode(), url)


def text_response(value: str, url: str) -> HttpResponse:
    return HttpResponse(200, {}, value.encode(), url)


def test_rosettafold_archives_are_mapped_to_exact_official_downloads() -> None:
    rf_repo = "RosettaCommons/RoseTTAFold"
    rf2_repo = "uw-ipd/RoseTTAFold2"
    rf_api = f"https://api.github.com/repos/{rf_repo}/commits/main"
    rf2_api = f"https://api.github.com/repos/{rf2_repo}/commits/main"
    rf_raw = f"https://raw.githubusercontent.com/{rf_repo}/{REV}/README.md"
    rf2_raw = f"https://raw.githubusercontent.com/{rf2_repo}/{REV}/README.md"
    client = QueueClient(
        json_response({"sha": REV}, rf_api),
        json_response({"sha": REV}, rf2_api),
        text_response(
            "[Update] includes RF2t.pt for RoseTTAFold-2track. "
            "wget https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz",
            rf_raw,
        ),
        text_response(
            "wget https://files.ipd.uw.edu/dimaio/RF2_jan24.tgz", rf2_raw
        ),
    )

    page = RoseTTAFoldCheckpointAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert len(page.records) == 2
    records = {record.raw["repository"]: record for record in page.records}
    rf = records[rf_repo]
    assert rf.raw["archive_filename"] == "weights.tar.gz"
    assert rf.raw["archive_url"] == (
        "https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz"
    )
    models = {model.name: model for model in rf.models}
    assert set(models) == {"RoseTTAFold", "RoseTTAFold-2track"}
    rf2track_release = next(
        release for release in rf.releases
        if release.model_local_id == "model:RoseTTAFold-2track"
    )
    assert rf2track_release.metadata["checkpoint_filename"] == "RF2t.pt"
    assert rf2track_release.metadata["archive_url"] == rf.raw["archive_url"]
    assert len(rf.links[1].model_local_ids) == 2
    assert rf.releases[0].metadata["archive_contents_enumerated"] is False
    rf2 = records[rf2_repo]
    assert rf2.raw["archive_filename"] == "RF2_jan24.tgz"
    assert rf2.raw["archive_url"] == "https://files.ipd.uw.edu/dimaio/RF2_jan24.tgz"
    assert client.calls == [rf_api, rf2_api, rf_raw, rf2_raw]


def test_shared_archive_release_identifier_keeps_rosettafold_variants_distinct() -> None:
    rf_repo = "RosettaCommons/RoseTTAFold"
    rf2_repo = "uw-ipd/RoseTTAFold2"
    rf_api = f"https://api.github.com/repos/{rf_repo}/commits/main"
    rf2_api = f"https://api.github.com/repos/{rf2_repo}/commits/main"
    rf_raw = f"https://raw.githubusercontent.com/{rf_repo}/{REV}/README.md"
    rf2_raw = f"https://raw.githubusercontent.com/{rf2_repo}/{REV}/README.md"
    client = QueueClient(
        json_response({"sha": REV}, rf_api),
        json_response({"sha": REV}, rf2_api),
        text_response(
            "[Update] includes RF2t.pt for RoseTTAFold-2track. "
            "wget https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz",
            rf_raw,
        ),
        text_response(
            "wget https://files.ipd.uw.edu/dimaio/RF2_jan24.tgz", rf2_raw
        ),
    )
    page = RoseTTAFoldCheckpointAdapter(client=client).fetch_page({})
    record = next(record for record in page.records if record.raw["repository"] == rf_repo)

    result = build_entries(
        [source_record_to_entry_seed(record, source="rosettafold-checkpoint-bundles")]
    )

    assert len(result.entries) == 2
    entries = {entry.canonical_name: entry for entry in result.entries}
    assert set(entries) == {"RoseTTAFold", "RoseTTAFold-2track"}
    expected_archive = "https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz"
    for name, entry in entries.items():
        assert len(entry.releases) == 1
        release = entry.releases[0]
        assert any(
            identifier.namespace == "rosettafold:weights-archive"
            and identifier.value == expected_archive
            for identifier in release.identifiers
        )
        release_metadata = json.loads(release.metadata_json)
        assert release_metadata["archive_url"] == expected_archive
        assert any(
            resource.url == expected_archive
            and resource.relation == "weights"
            and resource.model_local_id == entry.members[0].local_id
            for resource in entry.resources
        )
        if name == "RoseTTAFold-2track":
            assert release_metadata["checkpoint_filename"] == "RF2t.pt"
            assert any(
                identifier.namespace == "rosettafold:checkpoint"
                and identifier.value == "RF2t.pt"
                for identifier in release.identifiers
            )
        else:
            assert release_metadata["checkpoint_filename"] is None
            assert any(
                identifier.namespace == "rosettafold:archive-member"
                and identifier.value == "weights.tar.gz"
                for identifier in release.identifiers
            )


def test_rosettafold_archive_records_skip_unchanged_readmes() -> None:
    rf_repo = "RosettaCommons/RoseTTAFold"
    rf2_repo = "uw-ipd/RoseTTAFold2"
    client = QueueClient(
        json_response({"sha": REV}, f"https://api.github.com/repos/{rf_repo}/commits/main"),
        json_response({"sha": REV}, f"https://api.github.com/repos/{rf2_repo}/commits/main"),
    )

    page = RoseTTAFoldCheckpointAdapter(client=client).fetch_page(
        {"completed_revisions": {rf_repo: REV, rf2_repo: REV}}
    )

    assert page.records == ()
    assert page.next_state["completed_revisions"] == {rf_repo: REV, rf2_repo: REV}
    assert len(client.calls) == 2


def test_rosettafold_revision_change_refreshes_full_snapshot() -> None:
    rf_repo = "RosettaCommons/RoseTTAFold"
    rf2_repo = "uw-ipd/RoseTTAFold2"
    changed = "b" * 40
    old_state = {"completed_revisions": {rf_repo: REV, rf2_repo: REV}}
    client = QueueClient(
        json_response({"sha": changed}, f"https://api.github.com/repos/{rf_repo}/commits/main"),
        json_response({"sha": REV}, f"https://api.github.com/repos/{rf2_repo}/commits/main"),
        text_response(
            "RF2t.pt https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz",
            f"https://raw.githubusercontent.com/{rf_repo}/{changed}/README.md",
        ),
        text_response(
            "https://files.ipd.uw.edu/dimaio/RF2_jan24.tgz",
            f"https://raw.githubusercontent.com/{rf2_repo}/{REV}/README.md",
        ),
    )

    page = RoseTTAFoldCheckpointAdapter(client=client).fetch_page(old_state)

    assert page.authoritative_snapshot
    assert {record.raw["repository"] for record in page.records} == {rf_repo, rf2_repo}
    assert page.next_state["completed_revisions"] == {rf_repo: changed, rf2_repo: REV}
    assert len(client.calls) == 4


def test_rosettafold_adapter_rejects_missing_official_asset_link() -> None:
    rf_repo = "RosettaCommons/RoseTTAFold"
    client = QueueClient(
        json_response({"sha": REV}, f"https://api.github.com/repos/{rf_repo}/commits/main"),
        json_response({"sha": REV}, "https://api.github.com/repos/uw-ipd/RoseTTAFold2/commits/main"),
        text_response("No model download here", f"https://raw.githubusercontent.com/{rf_repo}/{REV}/README.md"),
    )

    with pytest.raises(ValueError, match="expected first-party asset link missing"):
        RoseTTAFoldCheckpointAdapter(client=client).fetch_page({})
