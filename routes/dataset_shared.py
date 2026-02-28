import json

from fastapi import Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from database import Dataset, TrainingExample
from generics import get_latest_grading_result, get_run_costs
from sse_starlette.sse import EventSourceResponse

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    class EventSourceResponse(StreamingResponse):
        def __init__(self, content, status_code: int = 200, headers: dict | None = None):
            merged_headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            if headers:
                merged_headers.update(headers)
            super().__init__(
                content=content,
                status_code=status_code,
                media_type="text/event-stream",
                headers=merged_headers,
            )


def resolve_source_material(
    source_material: str | list[int | str] | None,
    session: Session,
) -> tuple[str | None, list[int], int]:
    if source_material is None:
        return None, [], 0
    if isinstance(source_material, str):
        cleaned = source_material.strip()
        return (cleaned or None), [], (1 if cleaned else 0)
    if not isinstance(source_material, list):
        raise ValueError("source_material must be a string or list of dataset IDs/text blocks.")

    dataset_ids: list[int] = []
    text_blocks: list[str] = []
    seen_ids: set[int] = set()
    for item in source_material:
        if isinstance(item, int):
            if item in seen_ids:
                continue
            seen_ids.add(item)
            dataset_ids.append(item)
            continue
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                text_blocks.append(cleaned)
            continue
        raise ValueError("source_material list values must be integers or strings.")

    if not dataset_ids and not text_blocks:
        raise ValueError("source_material list cannot be empty.")

    context_lines = [
        "Use the following prior material as grounding context.",
        "Preserve consistent facts, style, and character details where relevant.",
    ]

    if text_blocks:
        for idx, block in enumerate(text_blocks, start=1):
            context_lines.append("")
            context_lines.append(f"TEXT BLOCK {idx}:")
            context_lines.append(block)

    if dataset_ids:
        datasets = session.exec(
            select(Dataset).where(Dataset.id.in_(dataset_ids)).order_by(Dataset.id)
        ).all()
        found_ids = {ds.id for ds in datasets if ds.id is not None}
        missing_ids = [dataset_id for dataset_id in dataset_ids if dataset_id not in found_ids]
        if missing_ids:
            raise ValueError(f"source_material dataset IDs not found: {missing_ids}")

        examples = session.exec(
            select(TrainingExample)
            .where(TrainingExample.dataset_id.in_(dataset_ids))
            .order_by(TrainingExample.dataset_id, TrainingExample.id)
        ).all()

        examples_by_dataset: dict[int, list[TrainingExample]] = {}
        for example in examples:
            examples_by_dataset.setdefault(example.dataset_id, []).append(example)

        for dataset in datasets:
            context_lines.append("")
            context_lines.append(f"DATASET {dataset.id}: {dataset.name}")
            if dataset.description:
                context_lines.append(f"DESCRIPTION: {dataset.description}")
            for idx, example in enumerate(examples_by_dataset.get(dataset.id, []), start=1):
                context_lines.append(f"EXAMPLE {idx} INSTRUCTION: {example.instruction}")
                context_lines.append(f"EXAMPLE {idx} RESPONSE: {example.response}")

    return "\n".join(context_lines), dataset_ids, len(text_blocks)


def parse_embedding(embedding_raw: str | None) -> list[float] | None:
    if not embedding_raw:
        return None
    try:
        parsed = json.loads(embedding_raw)
        if not isinstance(parsed, list) or not parsed:
            return None
        return [float(x) for x in parsed]
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def sse_message(*, data: dict, event: str | None = None, event_id: str | None = None) -> str:
    lines: list[str] = []
    if event_id:
        lines.append(f"id: {event_id}")
    if event:
        lines.append(f"event: {event}")
    payload = json.dumps(data, separators=(",", ":"))
    for line in payload.splitlines() or ["{}"]:
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def build_stream_item_payload(item: dict) -> dict:
    run_id = item.get("run_id")
    grading = get_latest_grading_result(run_id) if run_id else None
    costs = get_run_costs(run_id) if run_id else {
        "generation_cost": 0.0,
        "grading_cost": 0.0,
        "total_cost": 0.0,
    }
    return {
        "run_id": run_id,
        "dataset_id": item.get("dataset_id"),
        "dataset_key": item.get("dataset_key"),
        "status": item.get("status"),
        "topic": item.get("topic"),
        "requested_topic": item.get("requested_topic"),
        "agent": item.get("agent"),
        "attempts": item.get("attempts"),
        "error_type": item.get("error_type"),
        "error": item.get("error"),
        "score": grading.get("score") if grading else None,
        "grader_model": grading.get("grader_model") if grading else None,
        "accepted_count": grading.get("accepted_count") if grading else None,
        "rejected_count": grading.get("rejected_count") if grading else None,
        "cost": costs["total_cost"],
        "generation_cost": costs["generation_cost"],
        "grading_cost": costs["grading_cost"],
    }


def parse_last_event_index(last_event_id: str | None) -> int:
    if not last_event_id:
        return -1
    if not last_event_id.startswith("item:"):
        return -1
    parts = last_event_id.split(":")
    if len(parts) < 2:
        return -1
    try:
        return int(parts[1])
    except (TypeError, ValueError):
        return -1
