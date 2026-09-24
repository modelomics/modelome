from __future__ import annotations

from modelome.sources.dryad_model_candidates import DryadModelCandidatesSourceAdapter


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _Client:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, *, params=None, headers=None):
        self.calls.append((url, params))
        return _Response(self.payloads.pop(0))


def _dataset(doi="10.5061/dryad.b2rbnzsq4"):
    return {
        "identifier": f"doi:{doi}",
        "title": "Segmentation with deep learning models",
        "abstract": "Files contain model weights and network architecture.",
        "relatedWorks": [
            {
                "relationship": "primary_article",
                "identifierType": "DOI",
                "identifier": "10.1000/article.1",
            }
        ],
        "_links": {"stash:version": {"href": "/api/v2/versions/123"}},
    }


def _file(name, desc, file_id="81"):
    return {
        "path": name,
        "description": desc,
        "size": 2048,
        "mimeType": "application/octet-stream",
        "digest": "deadbeef",
        "digestType": "md5",
        "_links": {
            "self": {"href": f"/api/v2/files/{file_id}"},
            "stash:download": {"href": f"/api/v2/files/{file_id}/download"},
        },
    }


def test_emits_candidate_only_for_explicit_model_weight_file():
    client = _Client(
        [
            {
                "count": 1,
                "total": 1,
                "_embedded": {"stash:datasets": [_dataset()]},
            },
            {
                "count": 2,
                "total": 2,
                "_embedded": {
                    "stash:files": [
                        _file("unet.h5", "Weights for the trained model."),
                        _file("input.csv", "Input data."),
                    ]
                },
            },
        ]
    )
    page = DryadModelCandidatesSourceAdapter(client=client).fetch_page({})

    assert page.complete is False
    assert page.upstream_count == 1
    assert len(page.records) == 1
    record = page.records[0]
    assert record.source_record_id == "dryad:10.5061/dryad.b2rbnzsq4"
    assert record.models[0].status.value == "candidate"
    assert record.models[0].identifiers == ()
    assert record.releases[0].identifiers[0].value == "81"
    assert record.releases[0].metadata["filename"] == "unet.h5"
    assert record.links[-1].url.endswith("/files/81/download")
    assert record.links[-1].crawl is False
    assert record.links[0].url == "https://doi.org/10.1000/article.1"
    assert client.calls[0][1] == {
        "q": '"deep learning models"',
        "page": 1,
        "per_page": 1,
    }


def test_rejects_generic_weight_file_without_model_description():
    client = _Client(
        [
            {
                "count": 1,
                "total": 1,
                "_embedded": {"stash:datasets": [_dataset()]},
            },
            {
                "count": 1,
                "total": 1,
                "_embedded": {"stash:files": [_file("weights.h5", "Calibration weights.")]},
            },
        ]
    )

    page = DryadModelCandidatesSourceAdapter(client=client).fetch_page({})

    assert page.records == ()


def test_paginates_dataset_search_by_page_number():
    client = _Client(
        [
            {
                "count": 1,
                "total": 2,
                "_embedded": {
                    "stash:datasets": [
                        {
                            "identifier": "doi:10.5061/dryad.example",
                            "title": "A dataset without model metadata",
                        }
                    ]
                },
            },
        ]
    )
    adapter = DryadModelCandidatesSourceAdapter(client=client, page_size=1)

    page = adapter.fetch_page({})

    assert page.complete is False
    assert page.next_state == {
        "query_index": 0,
        "query": '"deep learning models"',
        "records_seen": 1,
        "page": 2,
    }


def test_advances_to_next_phrase_after_query_exhaustion():
    client = _Client(
        [
            {
                "count": 1,
                "total": 1,
                "_embedded": {
                    "stash:datasets": [
                        {
                            "identifier": "doi:10.5061/dryad.example",
                            "title": "A dataset without model metadata",
                        }
                    ]
                },
            },
        ]
    )
    adapter = DryadModelCandidatesSourceAdapter(client=client, page_size=1)

    page = adapter.fetch_page({})

    assert page.complete is False
    assert page.next_state == {
        "query_index": 1,
        "query": '"neural network weights"',
        "records_seen": 0,
    }


def test_reads_model_file_after_first_file_metadata_page():
    files = [_file(f"data-{index}.csv", "Input data.", str(index)) for index in range(100)]
    candidate = _file("model-weights.h5", "Trained neural network weights.", "101")
    client = _Client(
        [
            {
                "count": 1,
                "total": 1,
                "_embedded": {"stash:datasets": [_dataset()]},
            },
            {
                "count": 100,
                "total": 101,
                "_links": {"next": {"href": "/api/v2/versions/123/files?page=2"}},
                "_embedded": {"stash:files": files},
            },
            {
                "count": 1,
                "total": 101,
                "_embedded": {"stash:files": [candidate]},
            },
        ]
    )

    page = DryadModelCandidatesSourceAdapter(client=client).fetch_page({})

    assert len(page.records) == 1
    assert page.records[0].releases[0].metadata["filename"] == "model-weights.h5"
    assert client.calls[1][0] == "https://datadryad.org/api/v2/versions/123/files"
    assert client.calls[2][0] == "https://datadryad.org/api/v2/versions/123/files?page=2"


def test_fixed_doi_route_reads_one_record_and_exact_file_metadata():
    client = _Client(
        [
            {
                **_dataset("10.5061/dryad.ns1rn8ptd"),
                "title": "Supplementary information for a continuous-score machine learning model",
                "abstract": (
                    "This dataset contains weights of the trained machine learning models "
                    "used in the manuscript."
                ),
            },
            {
                "count": 1,
                "total": 1,
                "_embedded": {
                    "stash:files": [
                        _file(
                            "bcch_weights_epoch499.tar",
                            "",
                            "991",
                        )
                    ]
                },
            },
        ]
    )
    adapter = DryadModelCandidatesSourceAdapter(
        client=client,
        dataset_doi="10.5061/dryad.ns1rn8ptd",
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.upstream_count == 1
    assert len(page.records) == 1
    assert page.records[0].source_record_id == "dryad:10.5061/dryad.ns1rn8ptd"
    assert page.records[0].releases[0].identifiers[0].value == "991"
    assert client.calls[0][0] == (
        "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.ns1rn8ptd"
    )
    assert client.calls[1][0] == "https://datadryad.org/api/v2/versions/123/files"
