import json
import os
import tempfile

from fastapi import APIRouter, Depends, Query
import httpx
from fastapi.responses import FileResponse, JSONResponse
from funkybob import RandomNameGenerator
from sqlmodel import Session, select

from agent import generate_dataset, get_embedding, is_duplicate, save_responses
from database import Dataset, ExportHistory, ImportHistory, TrainingExample, get_session
from generics import TimedLabel, new_run_id, response_builder, timer
from routes.dataset_models import (
    ExportRequest,
    ExternalImportRequest,
    Generation,
    IngestExamples,
    ScraperIntakeRequest,
)
from routes.dataset_shared import resolve_source_material
from routes.data_processing import safe_json_dump, scraper_reference_card
from services.export_service import run_export_request
from services.import_service import (
    build_external_examples,
    build_scraper_examples,
    fetch_external_payload,
    normalize_external_records,
)
import logger


def register_data_routes(router: APIRouter):

    @router.get("/intake/reference")
    def intake_reference_card():
        return response_builder(
            success=True,
            message="Intake reference loaded.",
            statusCode=200,
            data={"cards": [scraper_reference_card()]},
        )

    @router.post("/intake/scraper")
    def intake_scraper_text(
        body: ScraperIntakeRequest,
        session: Session = Depends(get_session),
    ):
        try:
            if body.dedupe_threshold <= 0 or body.dedupe_threshold > 1:
                raise ValueError("dedupe_threshold must be in the range (0, 1].")
            if body.preview_limit <= 0 or body.preview_limit > 100:
                raise ValueError("preview_limit must be in the range 1-100.")
            if body.chunk_size <= 0 or body.chunk_size > 500:
                raise ValueError("chunk_size must be in the range 1-500.")
            if body.response_char_limit < 32 or body.response_char_limit > 8000:
                raise ValueError("response_char_limit must be in the range 32-8000.")
            if not body.records:
                raise ValueError("records must contain at least one item.")
            if len(body.records) > body.max_records:
                raise ValueError(f"records exceeds max_records ({body.max_records}). Split into smaller payloads.")

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
            logger.saveToLog(f"[intake_scraper_text] Validation failed: {e}", "ERROR")
            return response_builder(success=False, message=str(e), statusCode=400)
        except Exception as e:
            session.rollback()
            logger.saveToLog(f"[intake_scraper_text] Unexpected error: {e}", "ERROR")
            return response_builder(
                success=False,
                message="An unexpected error occurred while importing scraper text.",
                statusCode=500,
            )

    @router.post("/ingest")
    def ingest_example(body: IngestExamples, session: Session = Depends(get_session)):
        try:
            incoming_count = len(body.example) if body.example else 0
            dataset_name = body.dataset_name or str(RandomNameGenerator())
            dataset_description = body.dataset_description
            dataset_model = body.model
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

            logger.saveToLog(
                (
                    "[ingest_example] Starting ingest "
                    f"dataset_name={dataset_name!r} "
                    f"model={dataset_model!r} "
                    f"incoming_examples={incoming_count} "
                    f"prompt_len={len(body.prompt or '')}"
                ),
                "INFO",
            )

            if source_run_id:
                existing_dataset = session.exec(
                    select(Dataset).where(Dataset.source_run_id == source_run_id)
                ).first()
                if existing_dataset:
                    logger.saveToLog(
                        (
                            "[ingest_example] Reusing existing ingest result "
                            f"dataset_id={existing_dataset.id} "
                            f"source_run_id={source_run_id}"
                        ),
                        "INFO",
                    )
                    return response_builder(
                        success=True,
                        message="Dataset already existed for this run.",
                        errors=0,
                        count=len(existing_dataset.examples),
                        statusCode=200,
                        data={"dataset_id": existing_dataset.id, "reused": True},
                    )

            dataset = Dataset(
                name=dataset_name,
                description=dataset_description,
                model=dataset_model,
                source_run_id=source_run_id,
                generation_cost=generation_cost,
                grading_cost=grading_cost,
                total_cost=total_cost,
                examples=[],
            )
            existing_embeddings = []
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
                            logger.saveToLog(
                                "Discarding example with reason: BAD RESPONSE.. continuing",
                                "WARNING",
                            )
                            continue
                        embedding = get_embedding(ex.instruction)
                        embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
                        embedding_str = json.dumps(embedding_list)
                        if is_duplicate(embedding_list, existing_embeddings):
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
                        logger.saveToLog(
                            f"[ingest_example] Example processing failed idx={idx} type={type(e).__name__} error={e}",
                            "ERROR",
                        )
                        continue

            if len(dataset.examples) < 1:
                logger.saveToLog(
                    (
                        "[ingest_example] Rejecting ingest with zero survivors "
                        f"dataset_name={dataset_name!r} incoming={incoming_count} kept={kept_count} "
                        f"too_short={too_short_count} duplicates={duplicate_count} "
                        f"processing_errors={processing_error_count} preview_rejects={preview_rejects}"
                    ),
                    "WARNING",
                )
                raise ValueError("No valid examples found after ingest validation")

            session.add(dataset)
            session.commit()
            logger.saveToLog(
                (
                    "[ingest_example] Ingest committed "
                    f"dataset_id={dataset.id} dataset_name={dataset.name!r} incoming={incoming_count} "
                    f"kept={kept_count} too_short={too_short_count} duplicates={duplicate_count} "
                    f"processing_errors={processing_error_count} errors={errors}"
                ),
                "INFO",
            )
            return response_builder(
                success=True,
                message="Dataset successfully parsed and saved to database.",
                errors=errors,
                count=example_amount,
                statusCode=201,
                data={"dataset_id": dataset.id, "reused": False},
            )
        except ValueError as e:
            logger.saveToLog(f"[ingest_example] Validation failed: {e}", "ERROR")
            return response_builder(success=False, message=str(e), statusCode=400)
        except Exception as e:
            logger.saveToLog(f"[ingest_example] Unexpected error: {e}", "ERROR")
            return response_builder(
                success=False,
                message="An error occurred while ingesting examples.",
                statusCode=500,
            )

    @router.get("/{dataset_id}/export")
    def export_dataset(dataset_id: int, session: Session = Depends(get_session)):
        try:
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
            return FileResponse(filepath, filename=f"{dataset.name}.jsonl")
        except ValueError as e:
            logger.saveToLog(f"[export_dataset] Validation failed: {e}", "ERROR")
            return response_builder(success=False, message=str(e), statusCode=404)
        except Exception as e:
            logger.saveToLog(f"[export_dataset] Unexpected error: {e}", "ERROR")
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
            logger.saveToLog(f"[cost_summary] Unexpected error: {e}", "ERROR")
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
            logger.saveToLog(f"[get_dataset] Validation failed: {e}", "ERROR")
            return response_builder(success=False, message=str(e), statusCode=400)
        except Exception as e:
            logger.saveToLog(f"[get_dataset] Unexpected generation error: {e}", "ERROR")
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
            logger.saveToLog(f"[delete_dataset] Validation failed: {e}", "ERROR")
            return response_builder(success=False, message=str(e), statusCode=404)
        except Exception as e:
            logger.saveToLog(f"[delete_dataset] Unexpected error: {e}", "ERROR")
            return response_builder(
                success=False,
                message="An error occurred while removing dataset.",
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
            logger.saveToLog(f"[export_datasets] Validation failed: {e}", "ERROR")
            return response_builder(success=False, message=str(e), statusCode=400)
        except Exception as e:
            logger.saveToLog(f"[export_datasets] Unexpected error: {e}", "ERROR")
            return response_builder(
                success=False,
                message="An unexpected error occurred while exporting datasets.",
                statusCode=500,
            )

    @router.get("/exports/history")
    def export_history(limit: int = Query(default=25, ge=1, le=200), session: Session = Depends(get_session)):
        try:
            rows = session.exec(select(ExportHistory).order_by(ExportHistory.created_at.desc()).limit(limit)).all()
            return response_builder(
                success=True,
                message="Successfully returned export history.",
                statusCode=200,
                data={
                    "exports": [
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
                    ],
                    "count": len(rows),
                },
            )
        except Exception as e:
            logger.saveToLog(f"[export_history] Unexpected error: {e}", "ERROR")
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
            logger.saveToLog(f"[download_export_artifact] Validation failed: {e}", "ERROR")
            return response_builder(success=False, message=str(e), statusCode=404)
        except FileNotFoundError as e:
            logger.saveToLog(f"[download_export_artifact] Missing artifact: {e}", "ERROR")
            return response_builder(success=False, message=str(e), statusCode=410)
        except Exception as e:
            logger.saveToLog(f"[download_export_artifact] Unexpected error: {e}", "ERROR")
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
            logger.saveToLog(f"[rerun_export] Validation failed: {e}", "ERROR")
            return response_builder(success=False, message=str(e), statusCode=404)
        except Exception as e:
            logger.saveToLog(f"[rerun_export] Unexpected error: {e}", "ERROR")
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
        history: ImportHistory | None = None

        try:
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
                return response_builder(
                    success=True,
                    message="Successfully previewed external dataset import.",
                    statusCode=200,
                    data={
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
                    },
                )

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
            return response_builder(
                success=True,
                message="Successfully imported external dataset.",
                statusCode=201,
                data={
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
                },
            )
        except ValueError as e:
            session.rollback()
            if history:
                history.status = "failed"
                history.error = str(e)
                session.add(history)
                session.commit()
            logger.saveToLog(f"[import_external_dataset] Validation failed: {e}", "ERROR")
            return response_builder(success=False, message=str(e), statusCode=400)
        except httpx.HTTPError as e:
            session.rollback()
            if history:
                history.status = "failed"
                history.error = str(e)
                session.add(history)
                session.commit()
            logger.saveToLog(f"[import_external_dataset] Request failed: {e}", "ERROR")
            return response_builder(
                success=False,
                message="External import request failed.",
                statusCode=502,
            )
        except Exception as e:
            session.rollback()
            if history:
                history.status = "failed"
                history.error = str(e)
                session.add(history)
                session.commit()
            logger.saveToLog(f"[import_external_dataset] Unexpected error: {e}", "ERROR")
            return response_builder(
                success=False,
                message="An unexpected error occurred while importing dataset.",
                statusCode=500,
            )

    @router.get("/imports/history")
    def import_history(limit: int = Query(default=25, ge=1, le=200), session: Session = Depends(get_session)):
        try:
            rows = session.exec(select(ImportHistory).order_by(ImportHistory.created_at.desc()).limit(limit)).all()
            return response_builder(
                success=True,
                message="Successfully returned import history.",
                statusCode=200,
                data={
                    "imports": [
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
                    ],
                    "count": len(rows),
                },
            )
        except Exception as e:
            logger.saveToLog(f"[import_history] Unexpected error: {e}", "ERROR")
            return response_builder(
                success=False,
                message="An error occurred while fetching import history.",
                statusCode=500,
            )
