from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.core.database import Dataset, SourceDocument, TrainingExample
from app.core.generics import get_latest_grading_result, get_run_costs
from app.core.utils import parse_embedding, parse_last_event_index, sse_message

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


def _parse_source_material_items(
    source_material: list[int | str],
) -> tuple[list[int], list[int], list[str]]:
    dataset_ids: list[int] = []
    document_ids: list[int] = []
    text_blocks: list[str] = []
    seen_ids: set[int] = set()

    for item in source_material:
        if isinstance(item, int):
            if item in seen_ids:
                continue
            seen_ids.add(item)
            dataset_ids.append(item)
            continue

        if not isinstance(item, str):
            raise ValueError("source_material list values must be integers or strings.")

        cleaned = item.strip()
        if not cleaned:
            continue

        lowered = cleaned.casefold()
        if lowered.startswith("doc:") or lowered.startswith("document:"):
            _, _, raw_id = cleaned.partition(":")
            try:
                document_id = int(raw_id.strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid source document reference: {cleaned}") from exc
            if document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            document_ids.append(document_id)
            continue

        text_blocks.append(cleaned)

    return dataset_ids, document_ids, text_blocks


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

    dataset_ids, document_ids, text_blocks = _parse_source_material_items(source_material)

    if not dataset_ids and not document_ids and not text_blocks:
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

    if document_ids:
        documents = session.exec(
            select(SourceDocument).where(SourceDocument.id.in_(document_ids)).order_by(SourceDocument.id)
        ).all()
        found_doc_ids = {doc.id for doc in documents if doc.id is not None}
        missing_doc_ids = [document_id for document_id in document_ids if document_id not in found_doc_ids]
        if missing_doc_ids:
            raise ValueError(f"source_material document IDs not found: {missing_doc_ids}")

        for document in documents:
            context_lines.append("")
            context_lines.append(f"DOCUMENT {document.id}: {document.name}")
            for chunk in sorted(document.chunks, key=lambda entry: entry.chunk_index):
                context_lines.append(f"CHUNK {chunk.chunk_index}:")
                context_lines.append(chunk.content)

    return "\n".join(context_lines), dataset_ids, len(text_blocks)


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
        "category": grading.get("category") if grading else None,
        "grader_model": grading.get("grader_model") if grading else None,
        "accepted_count": grading.get("accepted_count") if grading else None,
        "rejected_count": grading.get("rejected_count") if grading else None,
        "cost": costs["total_cost"],
        "generation_cost": costs["generation_cost"],
        "grading_cost": costs["grading_cost"],
    }




