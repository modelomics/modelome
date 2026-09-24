from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.conceptnet_numberbatch import ConceptNetNumberbatchSourceAdapter

_REVISION = "d" * 40
_README = b"\n".join(
    (
        b"## Downloads",
        b"Version | Multilingual | English-only | HDF5",
        b"--- | --- | --- | ---",
        b"| **19.08**| [numberbatch-19.08.txt.gz][nb1908-main] | "
        b"[numberbatch-en-19.08.txt.gz][nb1908-en] | [19.08/mini.h5][nb1908-mini] |",
        b"| 17.06 | [numberbatch-17.06.txt.gz][nb1706-main] | "
        b"[numberbatch-en-17.06.txt.gz][nb1706-en] | [17.06/mini.h5][nb1706-mini] |",
        b"| 17.04 | [numberbatch-17.04.txt.gz][nb1704-main] | "
        b"[numberbatch-en-17.04b.txt.gz][nb1704-en] | [17.05/mini.h5][nb1704-mini] |",
        b"| 17.02 | [numberbatch-17.02.txt.gz][nb1704-main] | "
        b"[numberbatch-en-17.02.txt.gz][nb1702-en] | |",
        b"| 16.09 | | | [16.09/numberbatch.h5][nb1609-h5] |",
        b"[nb1908-main]: https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/"
        b"numberbatch-19.08.txt.gz",
        b"[nb1908-en]: https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/"
        b"numberbatch-en-19.08.txt.gz",
        b"[nb1908-mini]: http://conceptnet.s3.amazonaws.com/precomputed-data/2016/"
        b"numberbatch/19.08/mini.h5",
        b"[nb1706-main]: https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/"
        b"numberbatch-17.06.txt.gz",
        b"[nb1706-en]: https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/"
        b"numberbatch-en-17.06.txt.gz",
        b"[nb1706-mini]: http://conceptnet.s3.amazonaws.com/precomputed-data/2016/"
        b"numberbatch/17.06/mini.h5",
        b"[nb1704-main]: https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/"
        b"numberbatch-17.04.txt.gz",
        b"[nb1704-en]: https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/"
        b"numberbatch-en-17.04b.txt.gz",
        b"[nb1704-mini]: http://conceptnet.s3.amazonaws.com/precomputed-data/2016/"
        b"numberbatch/17.05/mini.h5",
        b"[nb1702-en]: http://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/"
        b"numberbatch-en-17.02.txt.gz",
        b"[nb1609-h5]: http://conceptnet.s3.amazonaws.com/precomputed-data/2016/"
        b"numberbatch/16.09/numberbatch.h5",
        b"## License and attribution",
        b"CC-By-SA 4.0",
    )
)


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        body = (
            f'{{"sha":"{_REVISION}"}}'.encode()
            if "/commits/" in url
            else _README
        )
        return HttpResponse(200, {}, body, url)


def test_official_release_table_enumerates_exact_numberbatch_archives() -> None:
    config = tomllib.loads(
        (Path(__file__).parents[1] / "config/proposals/conceptnet_numberbatch.toml").read_text()
    )["source"][0]
    assert config["enabled"] is False
    client = _FakeClient()
    adapter = ConceptNetNumberbatchSourceAdapter(
        name=config["name"],
        repository=config["repository"],
        branch=config["branch"],
        provider_namespace=config["provider_namespace"],
        client=client,
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 11
    records = {record.source_record_id: record for record in page.records}
    assert len(records) == 11
    assert "model:17.02:multilingual:text-vectors" not in records
    assert "model:17.02:english-only:text-vectors" in records
    assert "model:16.09:full-hdf5:hdf5" in records
    release = records["model:19.08:multilingual:text-vectors"]
    assert release.canonical_url == (
        "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/"
        "numberbatch-19.08.txt.gz"
    )
    assert release.releases[0].metadata["license_url"] == (
        "https://creativecommons.org/licenses/by-sa/4.0/"
    )
    assert len(client.calls) == 2
