import pytest


def test_is_transient_error_for_timeout_and_connection_markers():
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("sqlmodel")
    from agent.automation import _is_transient_error

    assert _is_transient_error(TimeoutError("request timeout"))
    assert _is_transient_error(httpx.ConnectError("connection reset by peer"))
    assert _is_transient_error(RuntimeError("temporarily unavailable"))


def test_is_transient_error_for_non_transient_runtime_error():
    pytest.importorskip("sqlmodel")
    from agent.automation import _is_transient_error

    assert not _is_transient_error(RuntimeError("schema validation failed"))


def test_suggest_topic_count_bounds():
    pytest.importorskip("sqlmodel")
    from agent.automation import _suggest_topic_count

    assert _suggest_topic_count(1) == 1
    assert 1 <= _suggest_topic_count(25) <= 25


def test_validate_generation_request_rejects_bad_concurrency():
    pytest.importorskip("sqlmodel")
    from agent.automation import _validate_generation_request

    with pytest.raises(ValueError, match="max_concurrency"):
        _validate_generation_request(
            amount=1,
            ex_amt=1,
            max_concurrency=0,
            max_retries=0,
            retry_backoff_seconds=0.0,
        )


def test_retry_delay_seconds_uses_exponential_backoff():
    pytest.importorskip("sqlmodel")
    from agent.automation import _retry_delay_seconds

    assert _retry_delay_seconds(attempts=1, retry_backoff_seconds=2.0) == 2.0
    assert _retry_delay_seconds(attempts=3, retry_backoff_seconds=2.0) == 8.0


def test_build_run_plan_item_shapes_expected_payload():
    pytest.importorskip("sqlmodel")
    from agent.automation import _build_run_plan_item
    from agent.types import AgentType

    item = _build_run_plan_item(
        item_index=2,
        run_id="run-123",
        requested_topic="base topic",
        topic="planned topic",
        resolved_agent=AgentType.qa,
        ex_amt=25,
        max_retries=1,
        retry_backoff_seconds=2.0,
        source_material="context",
        model="test-model",
        seed=42,
        slot_key="slot-a",
    )

    assert item["dataset_key"] == "run-123: planned topic"
    assert item["agent"] == "qa"
    assert item["seed"] == 42

