from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from agent import cosine_similarity, is_duplicate, run_naming_agent
from database import Dataset, ImportHistory, TrainingExample, get_session
from generics import new_run_id, response_builder
from routes.dataset_models import MergeRequest
from routes.dataset_shared import parse_embedding
import logger


def discover_merge_pools(
    session: Session, dataset_similarity_threshold: float, merge_run_id: str | None = None
) -> tuple[list[list[int]], int]:
    datasets = session.exec(select(Dataset).order_by(Dataset.id)).all()
    centroids: list[tuple[int, list[float]]] = []
    skipped_without_embeddings = 0

    for ds in datasets:
        examples = session.exec(select(TrainingExample).where(TrainingExample.dataset_id == ds.id)).all()
        vectors: list[list[float]] = []
        for ex in examples:
            emb = parse_embedding(ex.embedding)
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
        centroid = [sum(values) / len(valid_vectors) for values in zip(*valid_vectors, strict=False)]
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
            f"threshold={dataset_similarity_threshold} centroid_datasets={len(centroids)} "
            f"pools_found={len(pools)} skipped_without_embeddings={skipped_without_embeddings}"
        ),
        "INFO",
    )
    return pools, skipped_without_embeddings


def register_merge_routes(router: APIRouter):
    @router.post("/merge")
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
                    f"{log_prefix} Starting merge request explicit_ids_count={(len(body.dataset_ids) if body.dataset_ids else 0)} "
                    f"delete_originals={body.delete_originals} dataset_similarity_threshold={body.dataset_similarity_threshold}"
                ),
                "INFO",
            )

            session.exec(text("SELECT 1")).all()

            if body.dataset_similarity_threshold <= 0 or body.dataset_similarity_threshold > 1:
                raise ValueError("dataset_similarity_threshold must be in the range (0, 1].")

            if body.dataset_ids:
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
                logger.saveToLog(f"{log_prefix} Using explicit merge pool: {explicit_pool}", "INFO")
            else:
                pools, skipped_without_embeddings = discover_merge_pools(
                    session, body.dataset_similarity_threshold, merge_run_id
                )
                discovery_mode = True

            if not pools:
                logger.saveToLog(
                    (
                        f"{log_prefix} No merge candidates found "
                        f"(discovery_mode={discovery_mode}, threshold={body.dataset_similarity_threshold})"
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
                logger.saveToLog(f"{log_prefix} Processing pool_index={pool_idx} dataset_ids={pool}", "INFO")
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
                    logger.saveToLog(f"{log_prefix} Skipping pool_index={pool_idx} with no examples. pool={pool}", "WARNING")
                    continue

                pool_generation_cost = round(sum(float(ds.generation_cost or 0.0) for ds in all_datasets), 8)
                pool_grading_cost = round(sum(float(ds.grading_cost or 0.0) for ds in all_datasets), 8)
                pool_total_cost = round(sum(float(ds.total_cost or 0.0) for ds in all_datasets), 8)
                pool_models = sorted({str(ds.model).strip() for ds in all_datasets if getattr(ds, "model", None)})
                merged_model = ",".join(pool_models) if pool_models else None

                total_examples_before_dedupe += len(all_examples)
                deduped_examples: list[TrainingExample] = []
                existing_embeddings: list[list[float]] = []
                deduped_in_pool = 0
                for ex in all_examples:
                    parsed_embedding = parse_embedding(ex.embedding)
                    if parsed_embedding is not None and is_duplicate(parsed_embedding, existing_embeddings, threshold=0.8):
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

                naming_examples = [{"instruction": ex.instruction, "response": ex.response} for ex in deduped_examples]
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
                        history_rows = session.exec(
                            select(ImportHistory).where(ImportHistory.dataset_id == ds.id)
                        ).all()
                        for history in history_rows:
                            history.dataset_id = None
                            session.add(history)
                        session.delete(ds)

                merged_pool_count += 1
                total_examples_after_dedupe += len(deduped_examples)
                total_deduped += deduped_in_pool
                created_dataset_ids.append(dataset.id)
                logger.saveToLog(
                    (
                        f"{log_prefix} Pool merged pool_index={pool_idx} pool={pool} new_dataset_id={dataset.id} "
                        f"new_dataset_name={dataset.name!r} models={pool_models if pool_models else ['unknown']} "
                        f"generation_cost={pool_generation_cost} grading_cost={pool_grading_cost} "
                        f"total_cost={pool_total_cost} examples_before={len(all_examples)} "
                        f"examples_after={len(deduped_examples)} deduped={deduped_in_pool}"
                    ),
                    "INFO",
                )

            if merged_pool_count < 1:
                session.rollback()
                logger.saveToLog(
                    f"{log_prefix} Merge completed with zero successful pools; rolled back. pools_requested={len(pools)} errors={errors}",
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
                    f"{log_prefix} Merge committed successfully discovery_mode={discovery_mode} "
                    f"pools_requested={len(pools)} pools_merged={merged_pool_count} "
                    f"created_dataset_ids={created_dataset_ids} examples_before={total_examples_before_dedupe} "
                    f"examples_after={total_examples_after_dedupe} deduped_examples={total_deduped} errors={errors}"
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
