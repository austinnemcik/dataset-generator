import asyncio
from uuid import uuid4

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


def test_summarize_display_status_marks_cancelled_inflight_batch_as_stopping():
    pytest.importorskip("sqlmodel")
    from agent.batch_summary import summarize_display_status

    assert (
        summarize_display_status(
            batch_status="cancelled",
            queued=0,
            running=1,
            completed=0,
            failed=0,
            total=1,
        )
        == "stopping"
    )


def test_summarize_display_status_keeps_stopping_state():
    pytest.importorskip("sqlmodel")
    from agent.batch_summary import summarize_display_status

    assert (
        summarize_display_status(
            batch_status="cancelled",
            queued=1,
            running=0,
            completed=0,
            failed=0,
            total=1,
        )
        == "stopping"
    )


def test_result_from_item_preserves_non_terminal_statuses():
    pytest.importorskip("sqlmodel")
    from types import SimpleNamespace

    from agent.batch_summary import result_from_item

    queued_item = SimpleNamespace(
        item_index=0,
        run_id="queued-run",
        dataset_key="queued-run: topic",
        agent="qa",
        status="queued",
        attempts=0,
        error_type=None,
        error=None,
        created_dataset_id=None,
        topic="topic",
        requested_topic="topic",
        slot_key="slot",
    )
    running_item = SimpleNamespace(**{**queued_item.__dict__, "status": "running", "run_id": "running-run"})

    assert result_from_item(queued_item)["status"] == "queued"
    assert result_from_item(running_item)["status"] == "running"


def test_build_generation_plan_skips_topic_variation_when_disabled(monkeypatch):
    pytest.importorskip("sqlmodel")
    import agent.automation as automation

    calls: list[str] = []

    async def fake_run_topic_variation_agent(**kwargs):
        calls.append(kwargs["topic"])
        return ["should not be used"]

    monkeypatch.setattr(automation, "run_topic_variation_agent", fake_run_topic_variation_agent)

    plan = asyncio.run(
        automation.build_generation_plan(
            amount=3,
            topic="friendly support chats",
            allow_topic_variations=False,
        )
    )

    assert calls == []
    assert plan["planned_topics"] == ["friendly support chats"]
    assert {item["topic"] for item in plan["run_plan"]} == {"friendly support chats"}


def test_start_generation_can_queue_without_waiting(monkeypatch):
    pytest.importorskip("sqlmodel")
    import agent.automation as automation

    monkeypatch.setattr(
        automation,
        "build_generation_plan",
        lambda **kwargs: asyncio.sleep(
            0,
            result={
                "planned_topics": ["queued topic"],
                "run_plan": [
                    {
                        "item_index": 0,
                        "run_id": "queued-run-id",
                        "dataset_key": "queued-run-id: queued topic",
                        "slot_key": "slot-a",
                        "requested_topic": "queued topic",
                        "topic": "queued topic",
                        "agent": "qa",
                        "ex_amt": 2,
                        "max_retries": 1,
                        "retry_backoff_seconds": 1.0,
                        "source_material": None,
                        "model": "test-model",
                        "seed": 1,
                    }
                ],
            },
        ),
    )
    monkeypatch.setattr(automation, "create_batch_run", lambda **kwargs: "queued-batch-id")
    monkeypatch.setattr(
        automation,
        "get_batch_run_status",
        lambda batch_run_id: {
            "batch_run_id": batch_run_id,
            "status": "queued",
            "requested_runs": 1,
            "saved": 0,
            "failed": 0,
            "queued": 1,
            "running": 0,
            "results": [],
        },
    )

    started: list[str] = []

    async def fake_resume_batch_run(batch_run_id: str, *, max_concurrency: int | None = None):
        started.append(batch_run_id)
        await asyncio.sleep(0)
        return {"batch_run_id": batch_run_id}

    monkeypatch.setattr(automation, "resume_batch_run", fake_resume_batch_run)

    summary = asyncio.run(
        automation.start_generation(
            amount=1,
            topic="queued topic",
            ex_amt=2,
            wait_for_completion=False,
        )
    )

    assert summary["batch_run_id"] == "queued-batch-id"
    assert summary["status"] == "queued"
    assert started == ["queued-batch-id"]


def test_execute_batch_item_does_not_save_after_cancel(monkeypatch):
    pytest.importorskip("sqlmodel")
    import agent.automation as automation
    from agent.batch_store import create_batch_run, pending_item_ids_for_batch, stop_batch_run

    run_id = f"test-batch-cancel-save-{uuid4().hex}"
    item_run_id = f"test-batch-cancel-item-{uuid4().hex}"
    create_batch_run(
        request_payload={"topic": "cancel guard", "max_concurrency": 1},
        run_plan=[
            {
                "item_index": 0,
                "run_id": item_run_id,
                "dataset_key": f"{item_run_id}: cancel guard",
                "slot_key": "cancel-guard-slot",
                "requested_topic": "cancel guard",
                "topic": "cancel guard",
                "agent": "qa",
                "ex_amt": 2,
                "max_retries": 0,
                "retry_backoff_seconds": 1.0,
                "source_material": None,
                "model": "test-model",
                "seed": None,
            }
        ],
        run_id=run_id,
    )

    async def fake_generate_dataset(**kwargs):
        stop_batch_run(run_id)
        return ([{"instruction": "hi", "response": "hello"}], "prompt")

    async def fake_save_responses(**kwargs):
        raise AssertionError("save_responses should not run after cancellation")

    monkeypatch.setattr(automation, "generate_dataset", fake_generate_dataset)
    monkeypatch.setattr(automation, "save_responses", fake_save_responses)

    _, pending_ids = pending_item_ids_for_batch(run_id)
    result = asyncio.run(automation._execute_batch_item(pending_ids[0]))

    assert result is not None
    assert result["status"] == "failed"
    assert result["error_type"] == "cancelled"


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


def test_pending_item_ids_for_batch_reads_request_payload_without_name_error():
    pytest.importorskip("sqlmodel")
    from agent.batch_store import create_batch_run, pending_item_ids_for_batch

    run_id = f"test-batch-json-load-{uuid4().hex}"
    request_payload = {"topic": "personality pilot", "max_concurrency": 2}
    run_plan = [
        {
            "item_index": 0,
            "run_id": f"test-item-1-{uuid4().hex}",
            "dataset_key": "test-item-1: personality pilot",
            "slot_key": "slot-1",
            "requested_topic": "personality pilot",
            "topic": "personality pilot",
            "agent": "conversation",
            "ex_amt": 5,
            "max_retries": 1,
            "retry_backoff_seconds": 2.0,
            "source_material": None,
            "model": "test-model",
            "seed": None,
        }
    ]

    create_batch_run(request_payload=request_payload, run_plan=run_plan, run_id=run_id)
    loaded_request, pending_ids = pending_item_ids_for_batch(run_id)

    assert loaded_request["topic"] == "personality pilot"
    assert len(pending_ids) == 1


def test_prepare_item_attempt_reads_source_material_mode_from_request_payload():
    pytest.importorskip("sqlmodel")
    from agent.batch_store import create_batch_run, pending_item_ids_for_batch, prepare_item_attempt

    run_id = f"test-batch-source-mode-{uuid4().hex}"
    create_batch_run(
        request_payload={
            "topic": "warm conversation",
            "max_concurrency": 1,
            "source_material_mode": "style_only",
            "grading_lens": "voice_alignment",
        },
        run_plan=[
            {
                "item_index": 0,
                "run_id": f"test-source-mode-item-{uuid4().hex}",
                "dataset_key": "test-source-mode-item: warm conversation",
                "slot_key": "style-only-slot",
                "requested_topic": "warm conversation",
                "topic": "warm conversation",
                "agent": "conversation",
                "ex_amt": 5,
                "max_retries": 1,
                "retry_backoff_seconds": 2.0,
                "source_material": "seed examples",
                "model": "test-model",
                "seed": None,
            }
        ],
        run_id=run_id,
    )

    _, pending_ids = pending_item_ids_for_batch(run_id)
    prepared, exhausted = prepare_item_attempt(pending_ids[0])

    assert exhausted is False
    assert prepared["source_material_mode"] == "style_only"
    assert prepared["grading_lens"] == "voice_alignment"


def test_delete_batch_run_removes_completed_run():
    pytest.importorskip("sqlmodel")
    from sqlmodel import Session, select

    from agent.batch_store import create_batch_run, delete_batch_run, get_batch_run_status
    from app.core.database import BatchRun, BatchRunItem, engine

    run_id = f"test-batch-delete-helper-{uuid4().hex}"
    item_run_id = f"test-delete-item-1-{uuid4().hex}"
    create_batch_run(
        request_payload={"topic": "cleanup run", "max_concurrency": 1},
        run_plan=[
            {
                "item_index": 0,
                "run_id": item_run_id,
                "dataset_key": f"{item_run_id}: cleanup run",
                "slot_key": "cleanup-slot",
                "requested_topic": "cleanup run",
                "topic": "cleanup run",
                "agent": "conversation",
                "ex_amt": 5,
                "max_retries": 1,
                "retry_backoff_seconds": 2.0,
                "source_material": None,
                "model": "test-model",
                "seed": None,
            }
        ],
        run_id=run_id,
    )

    with Session(engine) as session:
        batch_run = session.exec(select(BatchRun).where(BatchRun.run_id == run_id)).first()
        item = session.exec(select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)).first()
        batch_run.status = "completed"
        batch_run.completed_runs = 1
        batch_run.queued_runs = 0
        item.status = "completed"
        session.add(batch_run)
        session.add(item)
        session.commit()

    deleted_summary = delete_batch_run(run_id)

    assert deleted_summary is not None
    assert deleted_summary["batch_run_id"] == run_id
    assert get_batch_run_status(run_id) is None


def test_delete_batch_run_allows_cancelled_inflight_record_removal():
    pytest.importorskip("sqlmodel")
    from sqlmodel import Session, select

    from agent.batch_store import create_batch_run, delete_batch_run, get_batch_run_status
    from app.core.database import BatchRun, BatchRunItem, engine

    run_id = f"test-batch-delete-cancelled-{uuid4().hex}"
    item_run_id = f"test-delete-cancelled-item-{uuid4().hex}"
    create_batch_run(
        request_payload={"topic": "stuck cancel", "max_concurrency": 1},
        run_plan=[
            {
                "item_index": 0,
                "run_id": item_run_id,
                "dataset_key": f"{item_run_id}: stuck cancel",
                "slot_key": "stuck-cancel-slot",
                "requested_topic": "stuck cancel",
                "topic": "stuck cancel",
                "agent": "conversation",
                "ex_amt": 5,
                "max_retries": 1,
                "retry_backoff_seconds": 2.0,
                "source_material": None,
                "model": "test-model",
                "seed": None,
            }
        ],
        run_id=run_id,
    )

    with Session(engine) as session:
        batch_run = session.exec(select(BatchRun).where(BatchRun.run_id == run_id)).first()
        item = session.exec(select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)).first()
        batch_run.status = "cancelled"
        batch_run.running_runs = 1
        batch_run.queued_runs = 0
        item.status = "running"
        session.add(batch_run)
        session.add(item)
        session.commit()

    deleted_summary = delete_batch_run(run_id)

    assert deleted_summary is not None
    assert deleted_summary["batch_run_id"] == run_id
    assert get_batch_run_status(run_id) is None


def test_delete_terminal_batch_runs_removes_only_terminal_runs():
    pytest.importorskip("sqlmodel")
    from sqlmodel import Session, select

    from agent.batch_store import create_batch_run, delete_terminal_batch_runs, get_batch_run_status
    from app.core.database import BatchRun, engine

    completed_run_id = f"test-batch-clear-completed-{uuid4().hex}"
    running_run_id = f"test-batch-clear-running-{uuid4().hex}"

    create_batch_run(
        request_payload={"topic": "cleanup complete", "max_concurrency": 1},
        run_plan=[
            {
                "item_index": 0,
                "run_id": f"test-complete-item-{uuid4().hex}",
                "dataset_key": "test-complete-item: cleanup complete",
                "slot_key": "cleanup-complete-slot",
                "requested_topic": "cleanup complete",
                "topic": "cleanup complete",
                "agent": "conversation",
                "ex_amt": 5,
                "max_retries": 1,
                "retry_backoff_seconds": 2.0,
                "source_material": None,
                "model": "test-model",
                "seed": None,
            }
        ],
        run_id=completed_run_id,
    )
    create_batch_run(
        request_payload={"topic": "cleanup running", "max_concurrency": 1},
        run_plan=[
            {
                "item_index": 0,
                "run_id": f"test-running-item-{uuid4().hex}",
                "dataset_key": "test-running-item: cleanup running",
                "slot_key": "cleanup-running-slot",
                "requested_topic": "cleanup running",
                "topic": "cleanup running",
                "agent": "conversation",
                "ex_amt": 5,
                "max_retries": 1,
                "retry_backoff_seconds": 2.0,
                "source_material": None,
                "model": "test-model",
                "seed": None,
            }
        ],
        run_id=running_run_id,
    )

    with Session(engine) as session:
        completed_run = session.exec(select(BatchRun).where(BatchRun.run_id == completed_run_id)).first()
        running_run = session.exec(select(BatchRun).where(BatchRun.run_id == running_run_id)).first()
        completed_run.status = "completed"
        running_run.status = "running"
        session.add(completed_run)
        session.add(running_run)
        session.commit()

    result = delete_terminal_batch_runs()

    assert completed_run_id in result["deleted_run_ids"]
    assert running_run_id not in result["deleted_run_ids"]
    assert get_batch_run_status(completed_run_id) is None
    assert get_batch_run_status(running_run_id) is not None

