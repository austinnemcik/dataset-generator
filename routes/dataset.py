from fastapi.responses import JSONResponse, FileResponse
from fastapi import Depends, APIRouter, Query
from pydantic import BaseModel
from datetime import datetime
from sqlmodel import Session, select
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from database import get_session, Dataset, TrainingExample
from funkybob import RandomNameGenerator
import asyncio
import json
import tempfile
import os
from generics import response_builder, TimedLabel, timer, new_run_id
from agent import (
    AgentType,
    run_naming_agent,
    get_embedding,
    cosine_similarity,
    is_duplicate,
    generate_dataset,
    save_responses,
)
from agent.automation import get_batch_run_status, start_generation
import logger


class Example(BaseModel):
    instruction: str
    response: str


class IngestExamples(BaseModel):
    example: list[Example] | None = None
    prompt: str
    dataset_id: int
    run_id: str | None = None
    dataset_description: str
    dataset_name: str
    model: str | None = None
    generation_cost: float | None = None
    grading_cost: float | None = None
    total_cost: float | None = None


class BatchGeneration(BaseModel):
    amount: int
    agent_types: list[AgentType] | None = None
    topics: list[str]
    ex_amt: int
    random_agent: bool = False
    max_concurrency: int = 3
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    seed: int | None = None
    source_material: str | None = (
        None  # allow passing in source material to guide the dataset gen if applicable
    )

    model: str | None = None


class Generation(BaseModel):
    agent_type: AgentType
    topic: str
    amount: int
    source_material: str | None = (
        None  # allow passing in source material to guide the dataset gen if applicable
    )
    model: str | None = None


class MergeRequest(BaseModel):
    dataset_ids: list[int] | None = None
    dataset_similarity_threshold: float = 0.65
    delete_originals: bool = False


data_router = APIRouter(prefix="/dataset", tags=["dataset"])


@data_router.post("/batch/generate")
async def get_multiple_datasets(body: BatchGeneration):
    agent_types = body.agent_types
    topics = body.topics
    amount = body.amount
    ex_amt = body.ex_amt
    random_agent = body.random_agent
    max_concurrency = body.max_concurrency
    max_retries = body.max_retries
    retry_backoff_seconds = body.retry_backoff_seconds
    seed = body.seed
    source_material = body.source_material
    model = body.model
    if amount > 250 or not amount > 0: 
        return response_builder(
        success=False, 
        message="Bad generation amount.  Values 1-250 are acceptable.",
        statusCode=400
    )
    if ex_amt > 50 or not ex_amt > 0:
        return response_builder(
            success=False, 
            message="Bad example amount per dataset. Values 1-50 are allowed",
            statusCode=400
        )
    if max_concurrency > 50 or not max_concurrency > 0:
        return response_builder(
            success=False,
            message="Bad max_concurrency. Values 1-50 are allowed.",
            statusCode=400,
        )

    normalized_topics: list[str] = []
    for t in topics:
        if not isinstance(t, str):
            continue
        cleaned = t.strip()
        if cleaned:
            normalized_topics.append(cleaned)

    # De-duplicate topic list while preserving order.
    deduped_topics: list[str] = []
    seen_topics: set[str] = set()
    for t in normalized_topics:
        key = t.casefold()
        if key in seen_topics:
            continue
        seen_topics.add(key)
        deduped_topics.append(t)

    if not deduped_topics:
        return response_builder(
            success=False,
            message="No valid topic(s) provided.",
            statusCode=400,
        )
    if len(deduped_topics) > 5:
        return response_builder(
            success=False,
            message="Too many topics. Maximum allowed is 5.",
            statusCode=400,
        )

    active_agents: list[AgentType] = []
    if agent_types:
        seen_agents: set[AgentType] = set()
        for a in agent_types:
            if a in seen_agents:
                continue
            seen_agents.add(a)
            active_agents.append(a)

    if not random_agent and not active_agents:
        return response_builder(
            success=False,
            message="agent_types must contain at least one value when random_agent is false.",
            statusCode=400,
        )

    iteration_agents: list[AgentType | None] = [None] if random_agent else active_agents
    allocation_slots: list[dict] = []
    for t_idx, topic_name in enumerate(deduped_topics):
        for a_idx, selected_agent in enumerate(iteration_agents):
            allocation_slots.append(
                {
                    "topic": topic_name,
                    "agent": selected_agent,
                    "topic_index": t_idx,
                    "agent_index": a_idx,
                    "amount": 0,
                }
            )

    for i in range(amount):
        allocation_slots[i % len(allocation_slots)]["amount"] += 1

    run_allocations = [slot for slot in allocation_slots if slot["amount"] > 0]

    topic_alloc_map: dict[str, int] = {}
    agent_alloc_map: dict[str, int] = {}
    for slot in run_allocations:
        topic_name = slot["topic"]
        agent_name = slot["agent"].value if slot["agent"] else "random"
        topic_alloc_map[topic_name] = topic_alloc_map.get(topic_name, 0) + slot["amount"]
        agent_alloc_map[agent_name] = agent_alloc_map.get(agent_name, 0) + slot["amount"]

    with timer(TimedLabel.BATCH_GENERATION):
        try:
            per_topic_summaries: list[dict] = []
            slot_parallelism = min(len(run_allocations), max_concurrency)
            # Keep total concurrency roughly within the requested budget.
            per_slot_concurrency = max(1, max_concurrency // max(1, slot_parallelism))

            async def _run_slot(slot: dict):
                selected_agent = slot["agent"]
                topic_seed = (
                    seed + (slot["topic_index"] * 100) + slot["agent_index"]
                    if seed is not None
                    else None
                )
                slot_max_concurrency = min(per_slot_concurrency, slot["amount"])
                summary = await start_generation(
                    amount=slot["amount"],
                    topic=slot["topic"],
                    agent=selected_agent,
                    model=model,
                    source_material=source_material,
                    ex_amt=ex_amt,
                    random_agent=random_agent,
                    max_concurrency=slot_max_concurrency,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    seed=topic_seed,
                    slot_key=f"{slot['topic']}|{selected_agent.value if selected_agent else 'random'}",
                )
                summary["selected_agent"] = selected_agent.value if selected_agent else None
                return summary

            slot_tasks = [asyncio.create_task(_run_slot(slot)) for slot in run_allocations]
            per_topic_summaries = await asyncio.gather(*slot_tasks)
        except ValueError as e:
            return response_builder(
                success=False,
                message=f"Encountered an error during generation run... See error {e}",
                statusCode=400,
            )
        except Exception as e:
            return response_builder(
                success=False,
                message=f"An unexpected error occurred during the batch generation.  See error: {e}",
                statusCode=500,
            )
    aggregate_summary = {
        "requested_runs": sum(s.get("requested_runs", 0) for s in per_topic_summaries),
        "generated": sum(s.get("generated", 0) for s in per_topic_summaries),
        "saved": sum(s.get("saved", 0) for s in per_topic_summaries),
        "failed": sum(s.get("failed", 0) for s in per_topic_summaries),
        "requested_topics": deduped_topics,
        "requested_agents": [a.value for a in active_agents],
        "random_agent_requested": random_agent,
        "random_agent_applied": random_agent,
        "topic_allocations": [
            {"topic": topic_name, "amount": amount_alloc}
            for topic_name, amount_alloc in topic_alloc_map.items()
        ],
        "agent_allocations": agent_alloc_map,
        "run_allocations": [
            {
                "topic": slot["topic"],
                "agent": slot["agent"].value if slot["agent"] else "random",
                "amount": slot["amount"],
            }
            for slot in run_allocations
        ],
        "slot_parallelism": min(len(run_allocations), max_concurrency),
        "per_topic_summaries": per_topic_summaries,
        "run_ids": [
            run_id
            for s in per_topic_summaries
            for run_id in s.get("run_ids", [])
        ],
        "batch_run_ids": [s.get("batch_run_id") for s in per_topic_summaries if s.get("batch_run_id")],
        "dataset_keys": [
            key
            for s in per_topic_summaries
            for key in s.get("dataset_keys", [])
        ],
        "results": [
            r
            for s in per_topic_summaries
            for r in s.get("results", [])
        ],
    }

    if aggregate_summary["saved"] > 0:
        return response_builder(
            success=True,
            message="Successfully completed batch generation run.",
            statusCode=201,
            data=aggregate_summary
        )
    else: 
        return response_builder(
        success=False, 
        message="Generation finished.. but received zero successful dataset generations",
        statusCode=422,
        data=aggregate_summary,
    )


@data_router.get("/batch/{run_id}")
def get_batch_run(run_id: str):
    try:
        summary = get_batch_run_status(run_id)
        if not summary:
            return response_builder(
                success=False,
                message="Batch run not found.",
                statusCode=404,
            )
        return response_builder(
            success=True,
            message="Successfully returned batch run status.",
            statusCode=200,
            data=summary,
        )
    except Exception as e:
        logger.saveToLog(f"[get_batch_run] Unexpected error: {e}", "ERROR")
        return response_builder(
            success=False,
            message="An error occurred while fetching batch run status.",
            statusCode=500,
        )


@data_router.post("/ingest")
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
                    embedding_list = (
                        embedding.tolist() if hasattr(embedding, "tolist") else embedding
                    )
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
                        (
                            "[ingest_example] Example processing failed "
                            f"idx={idx} type={type(e).__name__} error={e}"
                        ),
                        "ERROR",
                    )
                    continue

        if len(dataset.examples) < 1:
            logger.saveToLog(
                (
                    "[ingest_example] Rejecting ingest with zero survivors "
                    f"dataset_name={dataset_name!r} "
                    f"incoming={incoming_count} kept={kept_count} "
                    f"too_short={too_short_count} duplicates={duplicate_count} "
                    f"processing_errors={processing_error_count} "
                    f"preview_rejects={preview_rejects}"
                ),
                "WARNING",
            )
            raise ValueError("No valid examples found after ingest validation")

        session.add(dataset)
        session.commit()
        logger.saveToLog(
            (
                "[ingest_example] Ingest committed "
                f"dataset_id={dataset.id} "
                f"dataset_name={dataset.name!r} "
                f"incoming={incoming_count} kept={kept_count} "
                f"too_short={too_short_count} duplicates={duplicate_count} "
                f"processing_errors={processing_error_count} "
                f"errors={errors}"
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
        return response_builder(
            success=False,
            message=str(e),
            statusCode=400,
        )
    except Exception as e:
        logger.saveToLog(f"[ingest_example] Unexpected error: {e}", "ERROR")
        return response_builder(
            success=False,
            message="An error occurred while ingesting examples.",
            statusCode=500,
        )


@data_router.get("/{dataset_id}/export")
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


@data_router.get("/amount/{dataset_amount}")
def all_datasets(dataset_amount: int, session: Session = Depends(get_session)):
    amount = dataset_amount
    if not dataset_amount:
        amount = 5
        # return 5 if we don't get an amount specified.
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


@data_router.get("/costs/summary")
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
        overall = {
            "dataset_count": 0,
            "generation_cost": 0.0,
            "grading_cost": 0.0,
            "total_cost": 0.0,
        }
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

        overall = {
            k: (round(v, 8) if isinstance(v, float) else v) for k, v in overall.items()
        }
        for model_key, vals in by_model.items():
            by_model[model_key] = {
                k: (round(v, 8) if isinstance(v, float) else v) for k, v in vals.items()
            }

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


@data_router.post("/generate")
async def get_dataset(body: Generation):
    agent_type = body.agent_type
    topic = body.topic
    amount = body.amount
    source_material = body.source_material
    model = body.model
    run_id = new_run_id()
    dataset_key = f"{run_id}:{topic}"
    try:
        if body.model:
            dataset, prompt = await generate_dataset(
                agent_type=agent_type,
                topic=topic,
                amt=amount,
                source_material=source_material,
                model=model,
                run_id=run_id,
                dataset_key=dataset_key,
            )
        else:
            dataset, prompt = await generate_dataset(
                agent_type=agent_type,
                topic=topic,
                amt=amount,
                source_material=source_material,
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
            source_material=source_material,
            run_id=run_id,
            dataset_key=dataset_key,
        )
        return response_builder(
            success=True, message="Successfully generated dataset", statusCode=201
        )
    except ValueError as e:
        logger.saveToLog(f"[get_dataset] Validation failed: {e}", "ERROR")
        return response_builder(
            success=False,
            message=str(e),
            statusCode=400,
        )
    except Exception as e:
        logger.saveToLog(f"[get_dataset] Unexpected generation error: {e}", "ERROR")
        return response_builder(
            success=False,
            message="An unexpected error occurred while generating dataset.",
            statusCode=500,
        )


@data_router.delete("/remove/{dataset_id}")
def delete_dataset(dataset_id: int, session: Session = Depends(get_session)):
    try:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise ValueError("Dataset not found")
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


def _parse_embedding(embedding_raw: str | None) -> list[float] | None:
    if not embedding_raw:
        return None
    try:
        parsed = json.loads(embedding_raw)
        if not isinstance(parsed, list) or not parsed:
            return None
        return [float(x) for x in parsed]
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _discover_merge_pools(
    session: Session, dataset_similarity_threshold: float, merge_run_id: str | None = None
) -> tuple[list[list[int]], int]:
    datasets = session.exec(select(Dataset).order_by(Dataset.id)).all()
    centroids: list[tuple[int, list[float]]] = []
    skipped_without_embeddings = 0

    for ds in datasets:
        examples = session.exec(
            select(TrainingExample).where(TrainingExample.dataset_id == ds.id)
        ).all()
        vectors: list[list[float]] = []
        for ex in examples:
            emb = _parse_embedding(ex.embedding)
            if emb is not None:
                vectors.append(emb)
        if not vectors:
            skipped_without_embeddings += 1
            continue

        dims = len(vectors[0])
        valid_vectors = [v for v in vectors if len(v) == dims]
        if not valid_vectors:
            skipped_without_embeddings += 1
            continue
        centroid = [
            sum(values) / len(valid_vectors) for values in zip(*valid_vectors, strict=False)
        ]
        centroids.append((ds.id, centroid))

    pools: list[list[int]] = []
    assigned: set[int] = set()
    for i, (dataset_id, centroid) in enumerate(centroids):
        if dataset_id in assigned:
            continue
        pool = [dataset_id]
        assigned.add(dataset_id)
        for j in range(i + 1, len(centroids)):
            other_id, other_centroid = centroids[j]
            if other_id in assigned:
                continue
            similarity = float(cosine_similarity(centroid, other_centroid))
            if similarity >= dataset_similarity_threshold:
                pool.append(other_id)
                assigned.add(other_id)
        if len(pool) > 1:
            pools.append(pool)
    run_tag = f"[run_id={merge_run_id}] " if merge_run_id else ""
    logger.saveToLog(
        (
            f"[merge_datasets] {run_tag}Discovery complete "
            f"threshold={dataset_similarity_threshold} "
            f"centroid_datasets={len(centroids)} "
            f"pools_found={len(pools)} "
            f"skipped_without_embeddings={skipped_without_embeddings}"
        ),
        "INFO",
    )
    return pools, skipped_without_embeddings


@data_router.post("/merge")
async def merge_datasets(body: MergeRequest, session: Session = Depends(get_session)):
    errors = 0
    merged_pool_count = 0
    total_examples_before_dedupe = 0
    total_examples_after_dedupe = 0
    total_deduped = 0
    created_dataset_ids: list[int] = []
    merge_run_id = new_run_id()

    try:
        log_prefix = f"[merge_datasets][run_id={merge_run_id}]"
        logger.saveToLog(
            (
                f"{log_prefix} Starting merge request "
                f"explicit_ids_count={(len(body.dataset_ids) if body.dataset_ids else 0)} "
                f"delete_originals={body.delete_originals} "
                f"dataset_similarity_threshold={body.dataset_similarity_threshold}"
            ),
            "INFO",
        )

        # Force an early DB check so connection failures fail fast and clearly.
        session.exec(text("SELECT 1")).all()

        if body.dataset_similarity_threshold <= 0 or body.dataset_similarity_threshold > 1:
            raise ValueError("dataset_similarity_threshold must be in the range (0, 1].")

        if body.dataset_ids:
            # Explicit merge request path: keep old behavior of using caller-selected IDs.
            seen_ids: set[int] = set()
            explicit_pool = []
            for dataset_id in body.dataset_ids:
                if dataset_id in seen_ids:
                    continue
                seen_ids.add(dataset_id)
                explicit_pool.append(dataset_id)
            pools = [explicit_pool] if explicit_pool else []
            skipped_without_embeddings = 0
            discovery_mode = False
            logger.saveToLog(
                f"{log_prefix} Using explicit merge pool: {explicit_pool}",
                "INFO",
            )
        else:
            # Auto-discovery path: find candidate pools by centroid similarity.
            pools, skipped_without_embeddings = _discover_merge_pools(
                session, body.dataset_similarity_threshold, merge_run_id
            )
            discovery_mode = True

        if not pools:
            logger.saveToLog(
                (
                    f"{log_prefix} No merge candidates found "
                    f"(discovery_mode={discovery_mode}, "
                    f"threshold={body.dataset_similarity_threshold})"
                ),
                "WARNING",
            )
            return response_builder(
                success=False,
                statusCode=404,
                message="No merge candidates found.",
                data={
                    "discovery_mode": discovery_mode,
                    "merge_run_id": merge_run_id,
                    "dataset_similarity_threshold": body.dataset_similarity_threshold,
                    "pools_found": 0,
                    "skipped_without_embeddings": skipped_without_embeddings,
                },
            )

        for pool_idx, pool in enumerate(pools, start=1):
            logger.saveToLog(
                f"{log_prefix} Processing pool_index={pool_idx} dataset_ids={pool}",
                "INFO",
            )
            all_datasets: list[Dataset] = []
            all_examples: list[TrainingExample] = []
            for dataset_id in pool:
                ds = session.get(Dataset, dataset_id)
                if not ds:
                    errors += 1
                    continue
                all_datasets.append(ds)
                examples = session.exec(
                    select(TrainingExample).where(TrainingExample.dataset_id == dataset_id)
                ).all()
                all_examples.extend(examples)

            if discovery_mode and len(all_datasets) < 2:
                errors += 1
                logger.saveToLog(
                    (
                        f"{log_prefix} Skipping pool_index={pool_idx} after fetch because it has "
                        f"fewer than 2 valid datasets in discovery mode. pool={pool}"
                    ),
                    "WARNING",
                )
                continue
            if not all_examples:
                errors += 1
                logger.saveToLog(
                    f"{log_prefix} Skipping pool_index={pool_idx} with no examples. pool={pool}",
                    "WARNING",
                )
                continue

            pool_generation_cost = round(
                sum(float(ds.generation_cost or 0.0) for ds in all_datasets), 8
            )
            pool_grading_cost = round(
                sum(float(ds.grading_cost or 0.0) for ds in all_datasets), 8
            )
            pool_total_cost = round(
                sum(float(ds.total_cost or 0.0) for ds in all_datasets), 8
            )
            pool_models = sorted(
                {str(ds.model).strip() for ds in all_datasets if getattr(ds, "model", None)}
            )
            merged_model = ",".join(pool_models) if pool_models else None

            total_examples_before_dedupe += len(all_examples)
            deduped_examples: list[TrainingExample] = []
            existing_embeddings: list[list[float]] = []
            deduped_in_pool = 0
            for ex in all_examples:
                parsed_embedding = _parse_embedding(ex.embedding)
                if parsed_embedding is not None and is_duplicate(
                    parsed_embedding, existing_embeddings, threshold=0.8
                ):
                    deduped_in_pool += 1
                    continue
                if parsed_embedding is not None:
                    existing_embeddings.append(parsed_embedding)
                deduped_examples.append(ex)

            if not deduped_examples:
                errors += 1
                logger.saveToLog(
                    f"{log_prefix} Pool_index={pool_idx} fully deduped, no surviving examples. pool={pool}",
                    "WARNING",
                )
                continue

            naming_examples = [
                {
                    "instruction": ex.instruction,
                    "response": ex.response,
                }
                for ex in deduped_examples
            ]
            meta = await run_naming_agent(naming_examples)
            if not meta or "name" not in meta or "description" not in meta:
                errors += 1
                logger.saveToLog(
                    f"{log_prefix} Naming agent returned invalid metadata. pool_index={pool_idx} pool={pool}",
                    "ERROR",
                )
                continue

            merged_examples = [
                TrainingExample(
                    prompt=ex.prompt,
                    instruction=ex.instruction,
                    response=ex.response,
                    embedding=ex.embedding,
                )
                for ex in deduped_examples
            ]
            dataset = Dataset(
                name=meta["name"],
                description=meta["description"],
                model=merged_model,
                generation_cost=pool_generation_cost,
                grading_cost=pool_grading_cost,
                total_cost=pool_total_cost,
                examples=merged_examples,
            )
            session.add(dataset)
            session.flush()

            if body.delete_originals is True:
                for ds in all_datasets:
                    session.delete(ds)

            merged_pool_count += 1
            total_examples_after_dedupe += len(deduped_examples)
            total_deduped += deduped_in_pool
            created_dataset_ids.append(dataset.id)
            logger.saveToLog(
                (
                    f"{log_prefix} Pool merged "
                    f"pool_index={pool_idx} "
                    f"pool={pool} "
                    f"new_dataset_id={dataset.id} "
                    f"new_dataset_name={dataset.name!r} "
                    f"models={pool_models if pool_models else ['unknown']} "
                    f"generation_cost={pool_generation_cost} "
                    f"grading_cost={pool_grading_cost} "
                    f"total_cost={pool_total_cost} "
                    f"examples_before={len(all_examples)} "
                    f"examples_after={len(deduped_examples)} "
                    f"deduped={deduped_in_pool}"
                ),
                "INFO",
            )

        if merged_pool_count < 1:
            session.rollback()
            logger.saveToLog(
                (
                    f"{log_prefix} Merge completed with zero successful pools; rolled back. "
                    f"pools_requested={len(pools)} errors={errors}"
                ),
                "WARNING",
            )
            return response_builder(
                success=False,
                statusCode=422,
                message="No pools were successfully merged.",
                errors=errors,
                data={
                    "discovery_mode": discovery_mode,
                    "merge_run_id": merge_run_id,
                    "pools_requested": len(pools),
                    "skipped_without_embeddings": skipped_without_embeddings,
                },
            )

        session.commit()
        logger.saveToLog(
            (
                f"{log_prefix} Merge committed successfully "
                f"discovery_mode={discovery_mode} "
                f"pools_requested={len(pools)} "
                f"pools_merged={merged_pool_count} "
                f"created_dataset_ids={created_dataset_ids} "
                f"examples_before={total_examples_before_dedupe} "
                f"examples_after={total_examples_after_dedupe} "
                f"deduped_examples={total_deduped} "
                f"errors={errors}"
            ),
            "INFO",
        )
        return response_builder(
            success=True,
            statusCode=201,
            message=f"Successfully merged {merged_pool_count} pool(s).",
            errors=errors,
            data={
                "discovery_mode": discovery_mode,
                "merge_run_id": merge_run_id,
                "dataset_similarity_threshold": body.dataset_similarity_threshold,
                "pools_requested": len(pools),
                "pools_merged": merged_pool_count,
                "created_dataset_ids": created_dataset_ids,
                "examples_before_dedupe": total_examples_before_dedupe,
                "examples_after_dedupe": total_examples_after_dedupe,
                "deduped_examples": total_deduped,
                "skipped_without_embeddings": skipped_without_embeddings,
            },
        )
    except ValueError as e:
        logger.saveToLog(f"[merge_datasets][run_id={merge_run_id}] Validation failed: {e}", "ERROR")
        return response_builder(success=False, statusCode=400, message=str(e))
    except OperationalError as e:
        session.rollback()
        logger.saveToLog(f"[merge_datasets][run_id={merge_run_id}] Database unavailable: {e}", "ERROR")
        return response_builder(
            success=False,
            statusCode=503,
            message="Database is unavailable. Start the database and retry.",
        )
    except Exception as e:
        session.rollback()
        logger.saveToLog(
            f"[merge_datasets][run_id={merge_run_id}] Unexpected error: {type(e).__name__}: {e}",
            "ERROR",
        )
        return response_builder(
            success=False,
            statusCode=500,
            message="An unexpected error occurred while merging datasets",
        )
