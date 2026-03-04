import pytest


def test_external_import_request_has_dedupe_and_chunk_defaults():
    pytest.importorskip("pydantic")
    from routes.dataset_models import ExternalImportRequest

    body = ExternalImportRequest(url="https://example.com/api")
    assert body.dedupe_against_existing is True
    assert body.dedupe_within_payload is True
    assert body.chunk_size == 200
    assert body.max_records == 2000


def test_batch_generation_auto_merge_defaults_and_validation():
    pytest.importorskip("pydantic")
    from app.core.enums import AgentType
    from routes.dataset_models import BatchGeneration

    body = BatchGeneration(
        amount=3,
        topics=["Security"],
        agent_types=[AgentType.qa],
        ex_amt=10,
    )
    assert body.auto_merge_related is False
    assert body.auto_merge_similarity_threshold == 0.65

    with pytest.raises(ValueError, match="auto_merge_similarity_threshold"):
        BatchGeneration(
            amount=3,
            topics=["Security"],
            agent_types=[AgentType.qa],
            ex_amt=10,
            auto_merge_similarity_threshold=0,
        )

