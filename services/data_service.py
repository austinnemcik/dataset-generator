import json
import os
import tempfile
from pathlib import Path

import httpx
from funkybob import RandomNameGenerator
from sqlmodel import Session, select

import app.core.logger as logger
from app.core.database import (
    Dataset,
    ExportHistory,
    ImportHistory,
    SourceChunk,
    SourceDocument,
    TrainingExample,
)
from app.core.generics import TimedLabel, timer
from app.core.utils import chunk_text, parse_file_for_examples, parse_file_for_material, safe_json_dump
from routes.data_processing import detect_format, normalize_import_records
from routes.dataset_models import ExternalImportRequest, IngestExamples
from services.embedding_service import embed_text, embedding_json, is_semantic_duplicate
from services.import_service import (
    build_external_examples,
    fetch_external_payload,
    normalize_external_records,
)


def ingest_examples(session: Session, body: IngestExamples) -> dict:
    incoming_count = len(body.example) if body.example else 0
    dataset_name = body.dataset_name or str(RandomNameGenerator())
    dataset_description = body.dataset_description
    dataset_model = body.model
    dataset_category = body.category.strip() if body.category else None
    source_run_id = body.run_id
    generation_cost = float(body.generation_cost or 0.0)
    grading_cost = float(body.grading_cost or 0.0)
    total_cost = float(body.total_cost or 0.0)
    errors = 0
    example_amount = 0
    too_short_count = 0
    duplicate_count = 0
    processing_error_count = 0
    kept_count = 0
    preview_rejects: list[str] = []

    logger.log_event(
        "ingest.start",
        dataset_name=dataset_name,
        incoming_examples=incoming_count,
        model=dataset_model,
        prompt_len=len(body.prompt or ""),
        run_id=source_run_id,
    )

    if source_run_id:
        existing_dataset = session.exec(
            select(Dataset).where(Dataset.source_run_id == source_run_id)
        ).first()
        if existing_dataset:
            logger.log_event(
                "ingest.reused",
                dataset_id=existing_dataset.id,
                run_id=source_run_id,
            )
            return {
                "dataset_id": existing_dataset.id,
                "reused": True,
                "errors": 0,
                "count": len(existing_dataset.examples),
                "message": "Dataset already existed for this run.",
            }

    dataset = Dataset(
        name=dataset_name,
        description=dataset_description,
        category=dataset_category,
        model=dataset_model,
        source_run_id=source_run_id,
        generation_cost=generation_cost,
        grading_cost=grading_cost,
        total_cost=total_cost,
        examples=[],
    )
    existing_embeddings: list[list[float]] = []
    if not body.example:
        raise ValueError("No examples provided in ingest payload")

    with timer(label=TimedLabel.INGEST_REQUEST):
        for idx, ex in enumerate(body.example):
            try:
                prompt = body.prompt
                instruction = ex.instruction
                response = ex.response
                if len(instruction) < 2 or len(response) < 2 or len(prompt) < 2:
                    errors += 1
                    too_short_count += 1
                    if len(preview_rejects) < 5:
                        preview_rejects.append(
                            f"idx={idx}:short instruction_len={len(instruction)} response_len={len(response)} prompt_len={len(prompt)}"
                        )
                    logger.log_event(
                        "ingest.example_rejected",
                        "WARNING",
                        idx=idx,
                        reason="too_short",
                        run_id=source_run_id,
                    )
                    continue
                embedding_list = embed_text(ex.instruction)
                embedding_str = embedding_json(embedding_list)
                if embedding_list is None or embedding_str is None:
                    raise ValueError("Failed to build a valid embedding for example instruction")
                if is_semantic_duplicate(embedding_list, existing_embeddings):
                    errors += 1
                    duplicate_count += 1
                    if len(preview_rejects) < 5:
                        preview_rejects.append(f"idx={idx}:duplicate")
                    continue
                existing_embeddings.append(embedding_list)
                dataset.examples.append(
                    TrainingExample(
                        prompt=prompt,
                        instruction=instruction,
                        response=response,
                        embedding=embedding_str,
                    )
                )
                example_amount += 1
                kept_count += 1
            except Exception as e:
                errors += 1
                processing_error_count += 1
                if len(preview_rejects) < 5:
                    preview_rejects.append(
                        f"idx={idx}:processing_error type={type(e).__name__} message={str(e)[:120]}"
                    )
                logger.log_event(
                    "ingest.example_error",
                    "ERROR",
                    error=str(e),
                    error_type=type(e).__name__,
                    idx=idx,
                    run_id=source_run_id,
                )
                continue

    if len(dataset.examples) < 1:
        logger.log_event(
            "ingest.rejected",
            "WARNING",
            dataset_name=dataset_name,
            duplicates=duplicate_count,
            incoming=incoming_count,
            kept=kept_count,
            preview_rejects=preview_rejects,
            processing_errors=processing_error_count,
            run_id=source_run_id,
            too_short=too_short_count,
        )
        raise ValueError("No valid examples found after ingest validation")

    session.add(dataset)
    session.commit()
    logger.log_event(
        "ingest.committed",
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        duplicates=duplicate_count,
        errors=errors,
        incoming=incoming_count,
        kept=kept_count,
        processing_errors=processing_error_count,
        run_id=source_run_id,
        too_short=too_short_count,
    )
    return {
        "dataset_id": dataset.id,
        "reused": False,
        "errors": errors,
        "count": example_amount,
        "message": "Dataset successfully parsed and saved to database.",
    }


def export_single_dataset(session: Session, dataset_id: int) -> tuple[str, str]:
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError("No dataset found for this ID")

    examples = session.exec(
        select(TrainingExample).where(TrainingExample.dataset_id == dataset.id)
    ).all()

    lines = []
    for example in examples:
        formatted = {
            "conversations": [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": example.instruction},
                {"from": "gpt", "value": example.response},
            ]
        }
        lines.append(json.dumps(formatted))

    jsonl_content = "\n".join(lines)
    filepath = os.path.join(tempfile.gettempdir(), f"{dataset.name}.jsonl")
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(jsonl_content)
    logger.log_event(
        "dataset_export.file_built",
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        example_count=len(examples),
        path=filepath,
    )
    return filepath, f"{dataset.name}.jsonl"


def get_export_history_rows(session: Session, limit: int) -> list[dict]:
    rows = session.exec(select(ExportHistory).order_by(ExportHistory.created_at.desc()).limit(limit)).all()
    return [
        {
            "id": row.id,
            "status": row.status,
            "export_format": row.export_format,
            "dataset_ids": json.loads(row.dataset_ids_json),
            "options": json.loads(row.options_json) if row.options_json else {},
            "output_filename": row.output_filename,
            "has_artifact": bool(row.output_path and os.path.exists(row.output_path)),
            "total_examples": row.total_examples,
            "train_examples": row.train_examples,
            "val_examples": row.val_examples,
            "error": row.error,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


def get_import_history_rows(session: Session, limit: int) -> list[dict]:
    rows = session.exec(select(ImportHistory).order_by(ImportHistory.created_at.desc()).limit(limit)).all()
    return [
        {
            "id": row.id,
            "status": row.status,
            "source_url": row.source_url,
            "method": row.method,
            "detected_format": row.detected_format,
            "dataset_id": row.dataset_id,
            "dataset_name": row.dataset_name,
            "source_label": row.source_label,
            "fetched_records": row.fetched_records,
            "normalized_records": row.normalized_records,
            "imported_records": row.imported_records,
            "duplicate_records": row.duplicate_records,
            "invalid_records": row.invalid_records,
            "error": row.error,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


async def import_external_dataset_workflow(session: Session, body: ExternalImportRequest) -> dict:
    logger.log_event(
        "external_import.start",
        chunk_size=body.chunk_size,
        preview_only=body.preview_only,
        url=body.url,
    )
    if body.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0.")
    if body.dedupe_threshold <= 0 or body.dedupe_threshold > 1:
        raise ValueError("dedupe_threshold must be in the range (0, 1].")
    if body.preview_limit <= 0 or body.preview_limit > 100:
        raise ValueError("preview_limit must be in the range 1-100.")
    if body.chunk_size <= 0 or body.chunk_size > 500:
        raise ValueError("chunk_size must be in the range 1-500.")

    payload = await fetch_external_payload(body)
    detected_format, records, normalized_rows, invalid_records = normalize_external_records(body, payload)
    if len(records) > body.max_records:
        raise ValueError(f"records exceeds max_records ({body.max_records}). Split into smaller payloads.")

    imported_examples, duplicate_records, invalid_records = build_external_examples(
        session,
        body,
        normalized_rows,
        initial_invalid_records=invalid_records,
    )

    if body.preview_only:
        logger.log_event(
            "external_import.preview_ready",
            detected_format=detected_format,
            duplicate_records=duplicate_records,
            fetched_records=len(records),
            importable_records=len(imported_examples),
            invalid_records=invalid_records,
            url=body.url,
        )
        return {
            "preview_only": True,
            "detected_format": detected_format,
            "fetched_records": len(records),
            "normalized_records": len(normalized_rows),
            "importable_records": len(imported_examples),
            "duplicate_records": duplicate_records,
            "invalid_records": invalid_records,
            "chunk_size": body.chunk_size,
            "dedupe_against_existing": body.dedupe_against_existing,
            "dedupe_within_payload": body.dedupe_within_payload,
            "sample_records": [
                {
                    "instruction": example.instruction,
                    "response": example.response,
                }
                for example in imported_examples[: body.preview_limit]
            ],
        }

    if not imported_examples:
        raise ValueError("No importable records remained after normalization and deduplication.")

    history = ImportHistory(
        status="running",
        source_url=body.url,
        method=body.method.upper(),
        dataset_name=body.dataset_name,
        request_headers_json=safe_json_dump(body.headers),
        request_body_json=safe_json_dump(body.body),
        field_mapper_json=safe_json_dump(body.field_mapper),
        prompt=body.prompt,
        source_label=body.source_label,
    )
    session.add(history)
    session.commit()
    session.refresh(history)

    try:
        dataset = Dataset(
            name=body.dataset_name or str(RandomNameGenerator()),
            description=body.dataset_description,
            model=body.model,
            examples=imported_examples,
        )
        session.add(dataset)
        session.flush()

        history.status = "completed"
        history.detected_format = detected_format
        history.dataset_id = dataset.id
        history.dataset_name = dataset.name
        history.fetched_records = len(records)
        history.normalized_records = len(normalized_rows)
        history.imported_records = len(imported_examples)
        history.duplicate_records = duplicate_records
        history.invalid_records = invalid_records
        history.error = None

        session.add(history)
        session.commit()
    except Exception as e:
        session.rollback()
        history.status = "failed"
        history.error = str(e)
        session.add(history)
        session.commit()
        logger.log_event(
            "external_import.failed",
            "ERROR",
            error=str(e),
            history_id=history.id,
            url=body.url,
        )
        raise

    logger.log_event(
        "external_import.completed",
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        detected_format=detected_format,
        duplicate_records=duplicate_records,
        fetched_records=len(records),
        history_id=history.id,
        imported_records=len(imported_examples),
        invalid_records=invalid_records,
        url=body.url,
    )
    return {
        "import_history_id": history.id,
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "detected_format": detected_format,
        "fetched_records": len(records),
        "normalized_records": len(normalized_rows),
        "imported_records": len(imported_examples),
        "duplicate_records": duplicate_records,
        "invalid_records": invalid_records,
        "chunk_size": body.chunk_size,
        "dedupe_against_existing": body.dedupe_against_existing,
        "dedupe_within_payload": body.dedupe_within_payload,
        "preview_only": False,
    }


def import_example_file_workflow(
    session: Session,
    *,
    filename: str,
    contents: bytes,
    dataset_name: str | None,
    dataset_description: str | None,
    model: str | None,
    prompt: str,
    dedupe_threshold: float = 0.8,
    dedupe_against_existing: bool = True,
    dedupe_within_payload: bool = True,
    source_label: str | None = None,
) -> dict:
    from routes.dataset_models import ExternalImportRequest

    raw_records = parse_file_for_examples(contents, filename)
    temp_request = ExternalImportRequest(
        url=f"upload://{filename}",
        dataset_name=dataset_name,
        dataset_description=dataset_description,
        model=model,
        prompt=prompt,
        source_label=source_label,
        dedupe_threshold=dedupe_threshold,
        dedupe_against_existing=dedupe_against_existing,
        dedupe_within_payload=dedupe_within_payload,
        max_records=max(len(raw_records), 1),
        chunk_size=min(max(len(raw_records), 1), 200),
    )
    detected_format = detect_format(raw_records, temp_request.field_mapper)
    normalized_rows, invalid_records = normalize_import_records(
        raw_records,
        detected_format=detected_format,
        field_mapper=temp_request.field_mapper,
    )
    imported_examples, duplicate_records, invalid_records = build_external_examples(
        session,
        temp_request,
        normalized_rows,
        initial_invalid_records=invalid_records,
    )
    if not imported_examples:
        raise ValueError("No importable examples remained after normalization and deduplication.")

    dataset = Dataset(
        name=dataset_name or Path(filename).stem or str(RandomNameGenerator()),
        description=dataset_description,
        model=model,
        examples=imported_examples,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    logger.log_event(
        "file_example_import.completed",
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        detected_format=detected_format,
        duplicate_records=duplicate_records,
        imported_records=len(imported_examples),
        invalid_records=invalid_records,
        source=filename,
    )
    return {
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "detected_format": detected_format,
        "fetched_records": len(raw_records),
        "normalized_records": len(normalized_rows),
        "imported_records": len(imported_examples),
        "duplicate_records": duplicate_records,
        "invalid_records": invalid_records,
    }


def store_source_document_workflow(
    session: Session,
    *,
    filename: str,
    contents: bytes,
    chunk_char_size: int = 2000,
    chunk_overlap: int = 200,
) -> dict:
    text = parse_file_for_material(contents, filename)
    chunks = chunk_text(text, chunk_char_size=chunk_char_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No document chunks were produced from uploaded file.")

    document = SourceDocument(
        name=Path(filename).name,
        file_type=Path(filename).suffix.lower().lstrip(".") or "text",
        char_count=len(text),
        chunks=[
            SourceChunk(
                chunk_index=index,
                content=chunk,
                char_count=len(chunk),
            )
            for index, chunk in enumerate(chunks)
        ],
    )
    session.add(document)
    session.commit()
    session.refresh(document)
    logger.log_event(
        "source_document.stored",
        chunk_count=len(chunks),
        char_count=document.char_count,
        document_id=document.id,
        file_type=document.file_type,
        name=document.name,
    )
    return {
        "document_id": document.id,
        "name": document.name,
        "file_type": document.file_type,
        "char_count": document.char_count,
        "chunk_count": len(chunks),
        "chunk_char_size": chunk_char_size,
        "chunk_overlap": chunk_overlap,
        "source_material_ref": f"doc:{document.id}",
    }


