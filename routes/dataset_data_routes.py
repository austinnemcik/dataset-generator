import json
import os

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
import httpx
from fastapi.responses import FileResponse, JSONResponse
from funkybob import RandomNameGenerator
from sqlmodel import Session, select
from sqlalchemy import func, or_

from agent import generate_dataset, save_responses
from agent.grading import CATEGORY_TAXONOMY
from app.core.database import Dataset, ExportHistory, ImportHistory, SourceDocument, TrainingExample, get_session
from app.core.generics import new_run_id, response_builder
from routes.dataset_models import (
    ExportRequest,
    ExternalImportRequest,
    Generation,
    IngestExamples,
    ScraperIntakeRequest,
    UpdateTrainingExample,
)
from routes.dataset_shared import resolve_source_material
from routes.data_processing import scraper_reference_card, upload_reference_card
from services.data_service import (
    export_single_dataset,
    get_export_history_rows,
    get_import_history_rows,
    import_example_file_workflow,
    import_external_dataset_workflow,
    ingest_examples,
    store_source_document_workflow,
)
from services.export_service import run_export_request
from services.import_service import build_scraper_examples
import app.core.logger as logger


def _log_route_error(event: str, exc: Exception, *, log_type: str = "ERROR", **fields):
    logger.log_event(event, log_type, error=str(exc), error_type=type(exc).__name__, **fields)


def _category_rows() -> list[dict[str, str]]:
    return [
        {"value": category, "label": category.replace("_", " ").title()}
        for category in CATEGORY_TAXONOMY
    ]


def _document_summary(document: SourceDocument) -> dict:
    return {
        "id": document.id,
        "name": document.name,
        "file_type": document.file_type,
        "char_count": document.char_count,
        "chunk_count": len(document.chunks),
        "created_at": document.created_at.isoformat() if document.created_at else None,
        "source_material_ref": f"doc:{document.id}",
    }


def _dataset_summary(dataset: Dataset, example_count: int = 0) -> dict:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "category": dataset.category,
        "model": dataset.model,
        "example_count": example_count,
        "generation_cost": round(float(dataset.generation_cost or 0.0), 8),
        "grading_cost": round(float(dataset.grading_cost or 0.0), 8),
        "total_cost": round(float(dataset.total_cost or 0.0), 8),
    }


def _dataset_example_preview(examples: list[TrainingExample], limit: int = 3) -> list[dict]:
    return [
        {
            "id": example.id,
            "instruction": example.instruction,
            "response": example.response,
        }
        for example in examples[:limit]
    ]


# Registers the full dataset/data API surface for CRUD, intake, import/export,
# cost summaries, and source document operations. The route bodies stay here,
# while heavier ingest/import/export workflows are delegated into services.
def register_data_routes(router: APIRouter):

    @router.get("/categories")
    def list_dataset_categories():
        return response_builder(
            success=True,
            message="Successfully returned dataset categories.",
            statusCode=200,
            data={
                "categories": _category_rows(),
                "count": len(CATEGORY_TAXONOMY),
            },
        )

    @router.get("")
    def list_datasets(
        limit: int = Query(default=100, ge=1, le=500),
        q: str | None = Query(default=None),
        category: str | None = Query(default=None),
        session: Session = Depends(get_session),
    ):
        try:
            stmt = select(Dataset).order_by(Dataset.id.desc())
            if q:
                query = f"%{q.strip()}%"
                stmt = stmt.where(
                    or_(
                        Dataset.name.ilike(query),
                        Dataset.description.ilike(query),
                        Dataset.category.ilike(query),
                        Dataset.model.ilike(query),
                    )
                )
            if category:
                normalized_category = category.strip().lower()
                if normalized_category == "uncategorized":
                    stmt = stmt.where(Dataset.category.is_(None))
                else:
                    stmt = stmt.where(Dataset.category == category)
            datasets = session.exec(stmt.limit(limit)).all()
            dataset_ids = [dataset.id for dataset in datasets if dataset.id is not None]
            count_rows = session.exec(
                select(TrainingExample.dataset_id, func.count(TrainingExample.id))
                .where(TrainingExample.dataset_id.in_(dataset_ids))
                .group_by(TrainingExample.dataset_id)
            ).all() if dataset_ids else []
            example_counts = {dataset_id: int(count) for dataset_id, count in count_rows}
            rows = [_dataset_summary(dataset, example_counts.get(dataset.id, 0)) for dataset in datasets]
            return response_builder(
                success=True,
                message="Successfully returned datasets.",
                statusCode=200,
                data={"datasets": rows, "count": len(rows), "query": q, "category": category},
            )
        except Exception as e:
            _log_route_error("dataset_list.unexpected_error", e, limit=limit, query=q, category=category)
            return response_builder(
                success=False,
                message="An error occurred while fetching datasets.",
                statusCode=500,
            )

    @router.get("/{dataset_id}")
    def get_dataset_detail(
        dataset_id: int,
        example_offset: int = Query(default=0, ge=0),
        example_limit: int = Query(default=3, ge=1, le=20),
        session: Session = Depends(get_session),
    ):
        try:
            dataset = session.get(Dataset, dataset_id)
            if not dataset:
                raise ValueError("Dataset not found.")
            example_count = session.exec(
                select(func.count(TrainingExample.id)).where(TrainingExample.dataset_id == dataset_id)
            ).one()
            examples = session.exec(
                select(TrainingExample)
                .where(TrainingExample.dataset_id == dataset_id)
                .order_by(TrainingExample.id)
                .offset(example_offset)
                .limit(example_limit)
            ).all()
            return response_builder(
                success=True,
                message="Successfully returned dataset detail.",
                statusCode=200,
                data={
                    "dataset": {
                        **_dataset_summary(dataset, int(example_count)),
                        "examples_preview": _dataset_example_preview(examples),
                        "examples_preview_offset": example_offset,
                        "examples_preview_limit": example_limit,
                    }
                },
            )
        except ValueError as e:
            _log_route_error("dataset_detail.validation_failed", e, dataset_id=dataset_id)
            return response_builder(success=False, message=str(e), statusCode=404)
        except Exception as e:
            _log_route_error("dataset_detail.unexpected_error", e, dataset_id=dataset_id)
            return response_builder(
                success=False,
                message="An error occurred while fetching dataset detail.",
                statusCode=500,
            )

    @router.get("/intake/reference")
    def intake_reference_card():
        return response_builder(
            success=True,
            message="Intake reference loaded.",
            statusCode=200,
            data={"cards": [scraper_reference_card(), upload_reference_card()]},
        )

    @router.post("/intake/upload")
    async def intake_upload_file(
        file: UploadFile = File(...),
        intake_mode: str = Form(...),
        dataset_name: str | None = Form(default=None),
        dataset_description: str | None = Form(default=None),
        model: str | None = Form(default=None),
        prompt: str = Form(default="Imported file dataset"),
        dedupe_threshold: float = Form(default=0.8),
        dedupe_against_existing: bool = Form(default=True),
        dedupe_within_payload: bool = Form(default=True),
        chunk_char_size: int = Form(default=2000),
        chunk_overlap: int = Form(default=200),
        session: Session = Depends(get_session),
    ):
        try:
            if not file.filename:
                raise ValueError("Uploaded file must include a filename.")
            contents = await file.read()
            if not contents:
                raise ValueError("Uploaded file is empty.")

            normalized_mode = intake_mode.strip().lower()
            if normalized_mode == "examples":
                result = import_example_file_workflow(
                    session,
                    filename=file.filename,
                    contents=contents,
                    dataset_name=dataset_name,
                    dataset_description=dataset_description,
                    model=model,
                    prompt=prompt,
                    dedupe_threshold=dedupe_threshold,
                    dedupe_against_existing=dedupe_against_existing,
                    dedupe_within_payload=dedupe_within_payload,
                )
                return response_builder(
                    success=True,
                    message="Successfully imported uploaded examples as dataset.",
                    statusCode=201,
                    data={"intake_mode": normalized_mode, **result},
                )

            if normalized_mode not in {"source_material", "pretraining_data"}:
                raise ValueError("intake_mode must be one of: examples, source_material, pretraining_data.")
            if chunk_char_size <= 0:
                raise ValueError("chunk_char_size must be greater than 0.")
            if chunk_overlap < 0 or chunk_overlap >= chunk_char_size:
                raise ValueError("chunk_overlap must be non-negative and smaller than chunk_char_size.")

            result = store_source_document_workflow(
                session,
                filename=file.filename,
                contents=contents,
                chunk_char_size=chunk_char_size,
                chunk_overlap=chunk_overlap,
            )
            return response_builder(
                success=True,
                message="Successfully stored uploaded source document.",
                statusCode=201,
                data={"intake_mode": normalized_mode, **result},
            )
        except ValueError as e:
            session.rollback()
            _log_route_error("file_upload.validation_failed", e, filename=file.filename, intake_mode=intake_mode)
            return response_builder(success=False, message=str(e), statusCode=400)
        except Exception as e:
            session.rollback()
            _log_route_error("file_upload.unexpected_error", e, filename=file.filename, intake_mode=intake_mode)
            return response_builder(
                success=False,
                message="An unexpected error occurred while ingesting uploaded file.",
                statusCode=500,
            )

    @router.get("/documents")
    def list_source_documents(
        limit: int = Query(default=25, ge=1, le=200),
        session: Session = Depends(get_session),
    ):
        try:
            documents = session.exec(
                select(SourceDocument).order_by(SourceDocument.created_at.desc()).limit(limit)
            ).all()
            rows = [_document_summary(document) for document in documents]
            return response_builder(
                success=True,
                message="Successfully returned source documents.",
                statusCode=200,
                data={"documents": rows, "count": len(rows)},
            )
        except Exception as e:
            _log_route_error("source_document_list.unexpected_error", e, limit=limit)
            return response_builder(
                success=False,
                message="An error occurred while fetching source documents.",
                statusCode=500,
            )

    @router.get("/documents/{document_id}")
    def get_source_document(document_id: int, session: Session = Depends(get_session)):
        try:
            document = session.get(SourceDocument, document_id)
            if not document:
                raise ValueError("Source document not found.")
            return response_builder(
                success=True,
                message="Successfully returned source document.",
                statusCode=200,
                data={
                    "document": _document_summary(document),
                    "chunks": [
                        {
                            "id": chunk.id,
                            "chunk_index": chunk.chunk_index,
                            "char_count": chunk.char_count,
                            "content": chunk.content,
                        }
                        for chunk in sorted(document.chunks, key=lambda item: item.chunk_index)
                    ],
                },
            )
        except ValueError as e:
            _log_route_error("source_document_get.validation_failed", e, document_id=document_id)
            return response_builder(success=False, message=str(e), statusCode=404)
        except Exception as e:
            _log_route_error("source_document_get.unexpected_error", e, document_id=document_id)
            return response_builder(
                success=False,
                message="An error occurred while fetching source document.",
                statusCode=500,
            )

    @router.delete("/documents/{document_id}")
    def delete_source_document(document_id: int, session: Session = Depends(get_session)):
        try:
            document = session.get(SourceDocument, document_id)
            if not document:
                raise ValueError("Source document not found.")
            document_name = document.name
            session.delete(document)
            session.commit()
            return response_builder(
                success=True,
                message=f"Successfully removed source document {document_name}",
                statusCode=200,
            )
        except ValueError as e:
            _log_route_error("source_document_delete.validation_failed", e, document_id=document_id)
            return response_builder(success=False, message=str(e), statusCode=404)
        except Exception as e:
            session.rollback()
            _log_route_error("source_document_delete.unexpected_error", e, document_id=document_id)
            return response_builder(
                success=False,
                message="An error occurred while removing source document.",
                statusCode=500,
            )

    @router.post("/intake/scraper")
    def intake_scraper_text(
        body: ScraperIntakeRequest,
        session: Session = Depends(get_session),
    ):
        try:
            imported_examples, duplicate_records, invalid_records = build_scraper_examples(session, body)

            if body.preview_only:
                return response_builder(
                    success=True,
                    message="Successfully previewed scraper intake.",
                    statusCode=200,
                    data={
                        "preview_only": True,
                        "received_records": len(body.records),
                        "importable_records": len(imported_examples),
                        "duplicate_records": duplicate_records,
                        "invalid_records": invalid_records,
                        "chunk_size": body.chunk_size,
                        "dedupe_against_existing": body.dedupe_against_existing,
                        "dedupe_within_payload": body.dedupe_within_payload,
                        "sample_records": [
                            {"instruction": ex.instruction, "response": ex.response}
                            for ex in imported_examples[: body.preview_limit]
                        ],
                    },
                )

            if not imported_examples:
                raise ValueError("No importable records remained after normalization and deduplication.")

            dataset = Dataset(
                name=body.dataset_name or str(RandomNameGenerator()),
                description=body.dataset_description,
                model=body.model,
                examples=imported_examples,
            )
            session.add(dataset)
            session.commit()
            session.refresh(dataset)

            return response_builder(
                success=True,
                message="Successfully imported scraper text dataset.",
                statusCode=201,
                data={
                    "dataset_id": dataset.id,
                    "dataset_name": dataset.name,
                    "received_records": len(body.records),
                    "imported_records": len(imported_examples),
                    "duplicate_records": duplicate_records,
                    "invalid_records": invalid_records,
                    "chunk_size": body.chunk_size,
                    "dedupe_against_existing": body.dedupe_against_existing,
                    "dedupe_within_payload": body.dedupe_within_payload,
                },
            )
        except ValueError as e:
            session.rollback()
            _log_route_error("scraper_intake.validation_failed", e)
            return response_builder(success=False, message=str(e), statusCode=400)
        except Exception as e:
            session.rollback()
            _log_route_error("scraper_intake.unexpected_error", e)
            return response_builder(
                success=False,
                message="An unexpected error occurred while importing scraper text.",
                statusCode=500,
            )

    @router.post("/ingest")
    def ingest_example(body: IngestExamples, session: Session = Depends(get_session)):
        try:
            result = ingest_examples(session, body)
            return response_builder(
                success=True,
                message=result["message"],
                errors=result["errors"],
                count=result["count"],
                statusCode=200 if result["reused"] else 201,
                data={"dataset_id": result["dataset_id"], "reused": result["reused"]},
            )
        except ValueError as e:
            _log_route_error("ingest.validation_failed", e, run_id=body.run_id)
            return response_builder(success=False, message=str(e), statusCode=400)
        except Exception as e:
            _log_route_error("ingest.unexpected_error", e, run_id=body.run_id)
            return response_builder(
                success=False,
                message="An error occurred while ingesting examples.",
                statusCode=500,
            )

    @router.get("/{dataset_id}/export")
    def export_dataset(dataset_id: int, session: Session = Depends(get_session)):
        try:
            filepath, filename = export_single_dataset(session, dataset_id)
            return FileResponse(filepath, filename=filename)
        except ValueError as e:
            _log_route_error("dataset_export.validation_failed", e, dataset_id=dataset_id)
            return response_builder(success=False, message=str(e), statusCode=404)
        except Exception as e:
            _log_route_error("dataset_export.unexpected_error", e, dataset_id=dataset_id)
            return response_builder(
                success=False,
                message="An error occurred while exporting dataset.",
                statusCode=500,
            )

    @router.get("/amount/{dataset_amount}")
    def all_datasets(dataset_amount: int, session: Session = Depends(get_session)):
        amount = dataset_amount or 5
        datasets = session.exec(select(Dataset).order_by(Dataset.id).limit(amount)).all()

        details = []
        amt = 0
        for dataset in datasets:
            formatted = {
                "dataset": [
                    {"name": dataset.name},
                    {"description": dataset.description},
                    {"id": dataset.id},
                    {"category": dataset.category},
                    {"model": dataset.model},
                    {"generation_cost": dataset.generation_cost},
                    {"grading_cost": dataset.grading_cost},
                    {"total_cost": dataset.total_cost},
                ]
            }
            details.append(json.dumps(formatted))
            amt += 1

        return JSONResponse(
            {
                "success": True,
                "message": f"Successfully returned {amt} datasets",
                "datasets": details,
            }
        )

    @router.get("/costs/summary")
    def cost_summary(
        session: Session = Depends(get_session),
        limit: int | None = Query(default=None, ge=1, le=10000),
        model: str | None = Query(default=None),
    ):
        try:
            stmt = select(Dataset).order_by(Dataset.id)
            if model:
                stmt = stmt.where(Dataset.model == model)
            if limit:
                stmt = stmt.limit(limit)
            datasets = session.exec(stmt).all()
            overall = {"dataset_count": 0, "generation_cost": 0.0, "grading_cost": 0.0, "total_cost": 0.0}
            by_model: dict[str, dict] = {}
            by_dataset: list[dict] = []

            for ds in datasets:
                model_key = ds.model or "unknown"
                overall["dataset_count"] += 1
                overall["generation_cost"] += float(ds.generation_cost or 0.0)
                overall["grading_cost"] += float(ds.grading_cost or 0.0)
                overall["total_cost"] += float(ds.total_cost or 0.0)

                if model_key not in by_model:
                    by_model[model_key] = {
                        "dataset_count": 0,
                        "generation_cost": 0.0,
                        "grading_cost": 0.0,
                        "total_cost": 0.0,
                    }
                by_model[model_key]["dataset_count"] += 1
                by_model[model_key]["generation_cost"] += float(ds.generation_cost or 0.0)
                by_model[model_key]["grading_cost"] += float(ds.grading_cost or 0.0)
                by_model[model_key]["total_cost"] += float(ds.total_cost or 0.0)

                by_dataset.append(
                    {
                        "id": ds.id,
                        "name": ds.name,
                        "category": ds.category,
                        "model": ds.model,
                        "generation_cost": round(float(ds.generation_cost or 0.0), 8),
                        "grading_cost": round(float(ds.grading_cost or 0.0), 8),
                        "total_cost": round(float(ds.total_cost or 0.0), 8),
                    }
                )

            overall = {k: (round(v, 8) if isinstance(v, float) else v) for k, v in overall.items()}
            for model_key, vals in by_model.items():
                by_model[model_key] = {k: (round(v, 8) if isinstance(v, float) else v) for k, v in vals.items()}

            return JSONResponse(
                {
                    "success": True,
                    "message": "Successfully returned cost summary",
                    "filters": {"model": model, "limit": limit},
                    "overall": overall,
                    "by_model": by_model,
                    "by_dataset": by_dataset,
                }
            )
        except Exception as e:
            _log_route_error("cost_summary.unexpected_error", e, limit=limit, model=model)
            return response_builder(
                success=False,
                message="An error occurred while building cost summary.",
                statusCode=500,
            )

    @router.post("/generate")
    async def get_dataset(body: Generation, session: Session = Depends(get_session)):
        agent_type = body.agent_type
        topic = body.topic
        amount = body.amount
        source_material = body.source_material
        source_material_mode = body.source_material_mode
        conversation_length_mode = body.conversation_length_mode
        model = body.model
        run_id = new_run_id()
        dataset_key = f"{run_id}:{topic}"
        try:
            resolved_source_material, _, _ = resolve_source_material(source_material, session)
            if body.model:
                dataset, prompt = await generate_dataset(
                    agent_type=agent_type,
                    topic=topic,
                    amt=amount,
                    source_material=resolved_source_material,
                    source_material_mode=source_material_mode,
                    conversation_length_mode=conversation_length_mode,
                    model=model,
                    run_id=run_id,
                    dataset_key=dataset_key,
                )
            else:
                dataset, prompt = await generate_dataset(
                    agent_type=agent_type,
                    topic=topic,
                    amt=amount,
                    source_material=resolved_source_material,
                    source_material_mode=source_material_mode,
                    conversation_length_mode=conversation_length_mode,
                    run_id=run_id,
                    dataset_key=dataset_key,
                )
            await save_responses(
                agent_type=agent_type,
                examples=dataset,
                prompt=prompt,
                topic=topic,
                model=model,
                amount=amount,
                source_material=resolved_source_material,
                run_id=run_id,
                dataset_key=dataset_key,
            )
            return response_builder(success=True, message="Successfully generated dataset", statusCode=201)
        except ValueError as e:
            _log_route_error("dataset_generate.validation_failed", e, topic=topic, run_id=run_id)
            return response_builder(success=False, message=str(e), statusCode=400)
        except Exception as e:
            _log_route_error("dataset_generate.unexpected_error", e, topic=topic, run_id=run_id)
            return response_builder(
                success=False,
                message="An unexpected error occurred while generating dataset.",
                statusCode=500,
            )

    @router.delete("/remove/{dataset_id}")
    def delete_dataset(dataset_id: int, session: Session = Depends(get_session)):
        try:
            dataset = session.get(Dataset, dataset_id)
            if not dataset:
                raise ValueError("Dataset not found")
            history_rows = session.exec(
                select(ImportHistory).where(ImportHistory.dataset_id == dataset_id)
            ).all()
            for history in history_rows:
                history.dataset_id = None
                session.add(history)
            session.delete(dataset)
            session.commit()
            return response_builder(
                success=True,
                message=f"Successfully removed Dataset {dataset.name}",
                statusCode=200,
            )
        except ValueError as e:
            _log_route_error("dataset_delete.validation_failed", e, dataset_id=dataset_id)
            return response_builder(success=False, message=str(e), statusCode=404)
        except Exception as e:
            _log_route_error("dataset_delete.unexpected_error", e, dataset_id=dataset_id)
            return response_builder(
                success=False,
                message="An error occurred while removing dataset.",
                statusCode=500,
            )

    @router.delete("/examples/{example_id}")
    def delete_dataset_example(example_id: int, session: Session = Depends(get_session)):
        try:
            example = session.get(TrainingExample, example_id)
            if not example:
                raise ValueError("Training example not found.")

            dataset = session.get(Dataset, example.dataset_id)
            if not dataset:
                raise ValueError("Parent dataset not found.")

            dataset_id = dataset.id
            session.delete(example)
            session.commit()

            remaining_examples = session.exec(
                select(func.count(TrainingExample.id)).where(TrainingExample.dataset_id == dataset_id)
            ).one()
            return response_builder(
                success=True,
                message="Training example removed.",
                statusCode=200,
                data={"dataset_id": dataset_id, "remaining_examples": int(remaining_examples)},
            )
        except ValueError as e:
            _log_route_error("dataset_example_delete.validation_failed", e, example_id=example_id)
            return response_builder(success=False, message=str(e), statusCode=404)
        except Exception as e:
            session.rollback()
            _log_route_error("dataset_example_delete.unexpected_error", e, example_id=example_id)
            return response_builder(
                success=False,
                message="An error occurred while removing the training example.",
                statusCode=500,
            )

    @router.put("/examples/{example_id}")
    def update_dataset_example(
        example_id: int,
        body: UpdateTrainingExample,
        session: Session = Depends(get_session),
    ):
        try:
            example = session.get(TrainingExample, example_id)
            if not example:
                raise ValueError("Training example not found.")

            example.instruction = body.instruction
            example.response = body.response
            # Existing embeddings no longer match edited text.
            example.embedding = None
            session.add(example)
            session.commit()
            session.refresh(example)

            return response_builder(
                success=True,
                message="Training example updated.",
                statusCode=200,
                data={
                    "example": {
                        "id": example.id,
                        "dataset_id": example.dataset_id,
                        "instruction": example.instruction,
                        "response": example.response,
                    }
                },
            )
        except ValueError as e:
            _log_route_error("dataset_example_update.validation_failed", e, example_id=example_id)
            return response_builder(success=False, message=str(e), statusCode=404)
        except Exception as e:
            session.rollback()
            _log_route_error("dataset_example_update.unexpected_error", e, example_id=example_id)
            return response_builder(
                success=False,
                message="An error occurred while updating the training example.",
                statusCode=500,
            )

    @router.post("/export")
    def export_datasets(body: ExportRequest, session: Session = Depends(get_session)):
        try:
            output_path, history, stats = run_export_request(session, body)
            return FileResponse(
                output_path,
                filename=history.output_filename or os.path.basename(output_path),
                media_type="application/octet-stream",
                headers={
                    "X-Export-History-Id": str(history.id),
                    "X-Export-Format": history.export_format,
                    "X-Export-Examples": str(history.total_examples),
                    "X-Export-Deduped": str(stats["deduped_examples"]),
                },
            )
        except ValueError as e:
            _log_route_error("dataset_export_batch.validation_failed", e)
            return response_builder(success=False, message=str(e), statusCode=400)
        except Exception as e:
            _log_route_error("dataset_export_batch.unexpected_error", e)
            return response_builder(
                success=False,
                message="An unexpected error occurred while exporting datasets.",
                statusCode=500,
            )

    @router.get("/exports/history")
    def export_history(limit: int = Query(default=25, ge=1, le=200), session: Session = Depends(get_session)):
        try:
            rows = get_export_history_rows(session, limit)
            return response_builder(
                success=True,
                message="Successfully returned export history.",
                statusCode=200,
                data={
                    "exports": rows,
                    "count": len(rows),
                },
            )
        except Exception as e:
            _log_route_error("export_history.unexpected_error", e, limit=limit)
            return response_builder(
                success=False,
                message="An error occurred while fetching export history.",
                statusCode=500,
            )

    @router.get("/exports/{export_id}/download")
    def download_export_artifact(export_id: int, session: Session = Depends(get_session)):
        try:
            history = session.get(ExportHistory, export_id)
            if not history:
                raise ValueError("Export history entry not found.")
            if not history.output_path or not os.path.exists(history.output_path):
                raise FileNotFoundError("Export artifact not found on disk.")
            return FileResponse(
                history.output_path,
                filename=history.output_filename or os.path.basename(history.output_path),
                media_type="application/octet-stream",
            )
        except ValueError as e:
            _log_route_error("export_download.validation_failed", e, export_id=export_id)
            return response_builder(success=False, message=str(e), statusCode=404)
        except FileNotFoundError as e:
            _log_route_error("export_download.missing_artifact", e, export_id=export_id)
            return response_builder(success=False, message=str(e), statusCode=410)
        except Exception as e:
            _log_route_error("export_download.unexpected_error", e, export_id=export_id)
            return response_builder(
                success=False,
                message="An error occurred while downloading export artifact.",
                statusCode=500,
            )

    @router.post("/exports/{export_id}/rerun")
    def rerun_export(export_id: int, session: Session = Depends(get_session)):
        try:
            history = session.get(ExportHistory, export_id)
            if not history:
                raise ValueError("Export history entry not found.")
            options = json.loads(history.options_json) if history.options_json else {}
            rerun_request = ExportRequest(
                dataset_ids=json.loads(history.dataset_ids_json),
                export_format=history.export_format,
                min_score=options.get("min_score"),
                dedupe_pass=bool(options.get("dedupe_pass", False)),
                shuffle=bool(options.get("shuffle", False)),
                train_val_split=options.get("train_val_split"),
                max_examples=options.get("max_examples"),
            )
            output_path, new_history, stats = run_export_request(session, rerun_request)
            return FileResponse(
                output_path,
                filename=new_history.output_filename or os.path.basename(output_path),
                media_type="application/octet-stream",
                headers={
                    "X-Export-History-Id": str(new_history.id),
                    "X-Reexport-Of": str(history.id),
                    "X-Export-Format": new_history.export_format,
                    "X-Export-Examples": str(new_history.total_examples),
                    "X-Export-Deduped": str(stats["deduped_examples"]),
                },
            )
        except ValueError as e:
            _log_route_error("export_rerun.validation_failed", e, export_id=export_id)
            return response_builder(success=False, message=str(e), statusCode=404)
        except Exception as e:
            _log_route_error("export_rerun.unexpected_error", e, export_id=export_id)
            return response_builder(
                success=False,
                message="An error occurred while rerunning export.",
                statusCode=500,
            )

    @router.post("/import/external")
    async def import_external_dataset(
        body: ExternalImportRequest,
        session: Session = Depends(get_session),
    ):
        try:
            result = await import_external_dataset_workflow(session, body)
            if result["preview_only"]:
                return response_builder(
                    success=True,
                    message="Successfully previewed external dataset import.",
                    statusCode=200,
                    data=result,
                )
            return response_builder(
                success=True,
                message="Successfully imported external dataset.",
                statusCode=201,
                data=result,
            )
        except ValueError as e:
            session.rollback()
            _log_route_error("external_import.validation_failed", e, url=body.url)
            return response_builder(success=False, message=str(e), statusCode=400)
        except httpx.HTTPError as e:
            session.rollback()
            _log_route_error("external_import.request_failed", e, url=body.url)
            return response_builder(
                success=False,
                message="External import request failed.",
                statusCode=502,
            )
        except Exception as e:
            session.rollback()
            _log_route_error("external_import.unexpected_error", e, url=body.url)
            return response_builder(
                success=False,
                message="An unexpected error occurred while importing dataset.",
                statusCode=500,
            )

    @router.get("/imports/history")
    def import_history(limit: int = Query(default=25, ge=1, le=200), session: Session = Depends(get_session)):
        try:
            rows = get_import_history_rows(session, limit)
            return response_builder(
                success=True,
                message="Successfully returned import history.",
                statusCode=200,
                data={
                    "imports": rows,
                    "count": len(rows),
                },
            )
        except Exception as e:
            _log_route_error("import_history.unexpected_error", e, limit=limit)
            return response_builder(
                success=False,
                message="An error occurred while fetching import history.",
                statusCode=500,
            )



