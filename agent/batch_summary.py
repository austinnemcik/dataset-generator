import json
from collections import defaultdict

from app.core.database import BatchRun, BatchRunItem, utcnow


def json_dump(value: dict | list) -> str:
    return json.dumps(value)


def json_load(raw: str | None, fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def result_from_item(item: BatchRunItem) -> dict:
    if item.status == "completed":
        result_status = "saved"
    elif item.status == "failed":
        result_status = "failed"
    else:
        result_status = item.status
    return {
        "index": item.item_index,
        "run_id": item.run_id,
        "dataset_key": item.dataset_key,
        "agent": item.agent,
        "status": result_status,
        "attempts": item.attempts,
        "error_type": item.error_type,
        "error": item.error,
        "dataset_id": item.created_dataset_id,
        "topic": item.topic,
        "requested_topic": item.requested_topic,
        "slot_key": item.slot_key,
    }


def summarize_display_status(*, batch_status: str, queued: int, running: int, completed: int, failed: int, total: int) -> str:
    if batch_status == "cancelled" and (running > 0 or queued > 0):
        return "stopping"
    if batch_status == "paused":
        return "paused"
    if batch_status == "cancelled":
        return "cancelled"
    if running > 0:
        return "running"
    if queued > 0:
        return "running" if (completed > 0 or failed > 0) else "queued"
    if failed == total and total > 0:
        return "failed"
    return "completed"


def update_batch_counts(items: list[BatchRunItem], batch_run: BatchRun):
    queued = sum(1 for item in items if item.status == "queued")
    running = sum(1 for item in items if item.status == "running")
    completed = sum(1 for item in items if item.status == "completed")
    failed = sum(1 for item in items if item.status == "failed")

    batch_run.total_runs = len(items)
    batch_run.queued_runs = queued
    batch_run.running_runs = running
    batch_run.completed_runs = completed
    batch_run.failed_runs = failed
    batch_run.updated_at = utcnow()

    if batch_run.status == "paused":
        batch_run.completed_at = None
        return
    if batch_run.status == "cancelled":
        if running == 0:
            batch_run.completed_at = batch_run.completed_at or utcnow()
        return

    if running > 0:
        batch_run.status = "running"
        if not batch_run.started_at:
            batch_run.started_at = utcnow()
        batch_run.completed_at = None
    elif queued > 0:
        batch_run.status = "running" if (completed > 0 or failed > 0) else "queued"
        batch_run.completed_at = None
    elif failed == len(items) and len(items) > 0:
        batch_run.status = "failed"
        batch_run.completed_at = utcnow()
    else:
        batch_run.status = "completed"
        batch_run.completed_at = utcnow()


def build_batch_summary(batch_run: BatchRun, items: list[BatchRunItem]) -> dict:
    request_payload = json_load(batch_run.request_json, {})
    results = [result_from_item(item) for item in sorted(items, key=lambda entry: entry.item_index)]
    agent_usage: defaultdict[str, int] = defaultdict(int)
    topic_usage: defaultdict[str, int] = defaultdict(int)
    slot_usage: defaultdict[str, int] = defaultdict(int)
    dataset_keys: list[str] = []
    run_ids: list[str] = []

    for item in items:
        if item.status != "completed":
            continue
        agent_usage[item.agent] += 1
        topic_usage[item.topic] += 1
        if item.slot_key:
            slot_usage[item.slot_key] += 1
        dataset_keys.append(item.dataset_key)
        run_ids.append(item.run_id)

    slot_details: dict[str, dict] = {}
    for item in items:
        key = item.slot_key or f"{item.requested_topic}:{item.agent}"
        if key not in slot_details:
            slot_details[key] = {
                "slot_key": key,
                "requested_topic": item.requested_topic,
                "selected_agent": item.agent,
                "requested_runs": 0,
                "saved": 0,
                "failed": 0,
            }
        slot_details[key]["requested_runs"] += 1
        if item.status == "completed":
            slot_details[key]["saved"] += 1
        elif item.status == "failed":
            slot_details[key]["failed"] += 1

    summary = {
        "batch_run_id": batch_run.run_id,
        "request_group_id": request_payload.get("request_group_id"),
        "status": summarize_display_status(
            batch_status=batch_run.status,
            queued=batch_run.queued_runs,
            running=batch_run.running_runs,
            completed=batch_run.completed_runs,
            failed=batch_run.failed_runs,
            total=batch_run.total_runs,
        ),
        "raw_status": batch_run.status,
        "requested_runs": batch_run.total_runs,
        "generated": batch_run.completed_runs,
        "saved": batch_run.completed_runs,
        "failed": batch_run.failed_runs,
        "queued": batch_run.queued_runs,
        "running": batch_run.running_runs,
        "is_paused": batch_run.status == "paused",
        "is_cancelled": batch_run.status == "cancelled",
        "seed": request_payload.get("seed"),
        "max_concurrency": request_payload.get("max_concurrency"),
        "max_retries": request_payload.get("max_retries"),
        "retry_backoff_seconds": request_payload.get("retry_backoff_seconds"),
        "topic": request_payload.get("topic"),
        "planned_topics": request_payload.get("planned_topics", []),
        "requested_topic": request_payload.get("topic"),
        "source_material_mode": request_payload.get("source_material_mode", "content_and_style"),
        "grading_lens": request_payload.get("grading_lens", "balanced_quality"),
        "agent_usage": dict(agent_usage),
        "topic_usage": dict(topic_usage),
        "slot_usage": dict(slot_usage),
        "run_ids": run_ids,
        "dataset_keys": dataset_keys,
        "results": results,
        "requested_agent": request_payload.get("agent"),
        "random_agent": request_payload.get("random_agent", False),
        "per_slot_summary": list(slot_details.values()),
        "created_at": batch_run.created_at.isoformat() if batch_run.created_at else None,
        "updated_at": batch_run.updated_at.isoformat() if batch_run.updated_at else None,
        "started_at": batch_run.started_at.isoformat() if batch_run.started_at else None,
        "completed_at": batch_run.completed_at.isoformat() if batch_run.completed_at else None,
    }
    batch_run.summary_json = json_dump(summary)
    return summary



