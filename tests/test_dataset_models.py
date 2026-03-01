import pytest


def test_external_import_request_has_dedupe_and_chunk_defaults():
    pytest.importorskip("pydantic")
    from routes.dataset_models import ExternalImportRequest

    body = ExternalImportRequest(url="https://example.com/api")
    assert body.dedupe_against_existing is True
    assert body.dedupe_within_payload is True
    assert body.chunk_size == 200
    assert body.max_records == 2000
