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
