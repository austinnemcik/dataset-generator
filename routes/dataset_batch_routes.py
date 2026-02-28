import asyncio

from fastapi import APIRouter, Depends, Query, Request
from sqlmodel import Session, select

from agent import pause_batch_run, restart_failed_batch_run, resume_batch_run, stop_batch_run
from agent.automation import get_batch_run_status, start_generation
from database import BatchRun, engine, get_session
from generics import TimedLabel, response_builder, timer
from routes.dataset_models import BatchGeneration
from routes.dataset_shared import (
    EventSourceResponse,
    build_stream_item_payload,
    parse_last_event_index,
    resolve_source_material,
    sse_message,
)
import logger


def register_batch_routes(router: APIRouter):
    @router.post("/batch/generate")
    async def get_multiple_datasets(body: BatchGeneration, session: Session = Depends(get_session)):
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
                statusCode=400,
            )
        if ex_amt > 50 or not ex_amt > 0:
            return response_builder(
                success=False,
                message="Bad example amount per dataset. Values 1-50 are allowed",
                statusCode=400,
            )
        if max_concurrency > 50 or not max_concurrency > 0:
            return response_builder(
                success=False,
                message="Bad max_concurrency. Values 1-50 are allowed.",
                statusCode=400,
            )
        try:
            (
                resolved_source_material,
                source_material_dataset_ids,
                source_material_text_block_count,
            ) = resolve_source_material(source_material, session)
        except ValueError as e:
            return response_builder(success=False, message=str(e), statusCode=400)

        normalized_topics: list[str] = []
        for t in topics:
            if not isinstance(t, str):
                continue
            cleaned = t.strip()
            if cleaned:
                normalized_topics.append(cleaned)

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

        active_agents = []
        if agent_types:
            seen_agents = set()
            for agent in agent_types:
                if agent in seen_agents:
                    continue
                seen_agents.add(agent)
                active_agents.append(agent)

        if not random_agent and not active_agents:
            return response_builder(
                success=False,
                message="agent_types must contain at least one value when random_agent is false.",
                statusCode=400,
            )

        iteration_agents = [None] if random_agent else active_agents
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
                slot_parallelism = min(len(run_allocations), max_concurrency)
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
                        source_material=resolved_source_material,
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
            "source_material_dataset_ids": source_material_dataset_ids,
            "source_material_text_block_count": source_material_text_block_count,
            "source_material_mode": (
                "mixed"
                if source_material_dataset_ids and source_material_text_block_count > 0
                else (
                    "dataset_ids"
                    if source_material_dataset_ids
                    else ("text" if resolved_source_material else None)
                )
            ),
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
            "run_ids": [run_id for s in per_topic_summaries for run_id in s.get("run_ids", [])],
            "batch_run_ids": [s.get("batch_run_id") for s in per_topic_summaries if s.get("batch_run_id")],
            "dataset_keys": [key for s in per_topic_summaries for key in s.get("dataset_keys", [])],
            "results": [r for s in per_topic_summaries for r in s.get("results", [])],
        }

        if aggregate_summary["saved"] > 0:
            return response_builder(
                success=True,
                message="Successfully completed batch generation run.",
                statusCode=201,
                data=aggregate_summary,
            )
        return response_builder(
            success=False,
            message="Generation finished.. but received zero successful dataset generations",
            statusCode=422,
            data=aggregate_summary,
        )

    @router.get("/batch/{run_id}")
    def get_batch_run(run_id: str):
        try:
            summary = get_batch_run_status(run_id)
            if not summary:
                return response_builder(success=False, message="Batch run not found.", statusCode=404)
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

    @router.get("/batch")
    def list_batch_runs(status: str | None = Query(default=None), limit: int = Query(default=25, ge=1, le=200)):
        try:
            with Session(engine) as session:
                stmt = select(BatchRun).order_by(BatchRun.created_at.desc()).limit(limit)
                if status:
                    stmt = stmt.where(BatchRun.status == status)
                runs = session.exec(stmt).all()

            data = []
            for run in runs:
                summary = get_batch_run_status(run.run_id) or {}
                data.append(
                    {
                        "run_id": run.run_id,
                        "status": run.status,
                        "requested_runs": run.total_runs,
                        "saved": run.completed_runs,
                        "failed": run.failed_runs,
                        "queued": run.queued_runs,
                        "running": run.running_runs,
                        "topic": summary.get("topic"),
                        "requested_agent": summary.get("requested_agent"),
                        "created_at": run.created_at.isoformat() if run.created_at else None,
                        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
                        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                    }
                )

            return response_builder(
                success=True,
                message="Successfully returned batch runs.",
                statusCode=200,
                data={"runs": data, "count": len(data), "status_filter": status, "limit": limit},
            )
        except Exception as e:
            logger.saveToLog(f"[list_batch_runs] Unexpected error: {e}", "ERROR")
            return response_builder(
                success=False,
                message="An error occurred while listing batch runs.",
                statusCode=500,
            )

    @router.get("/batch/{run_id}/stream")
    async def stream_batch_run(run_id: str, request: Request):
        initial_summary = get_batch_run_status(run_id)
        if not initial_summary:
            return response_builder(success=False, message="Batch run not found.", statusCode=404)

        async def event_generator():
            last_event_index = parse_last_event_index(request.headers.get("last-event-id"))
            emitted_terminal_runs: set[str] = set()
            for item in initial_summary.get("results", []):
                item_index = item.get("index")
                item_run_id = item.get("run_id")
                if (
                    isinstance(item_index, int)
                    and item_index <= last_event_index
                    and item.get("status") in {"saved", "failed"}
                    and item_run_id
                ):
                    emitted_terminal_runs.add(item_run_id)
            last_progress_snapshot = None

            yield sse_message(
                event="batch_snapshot",
                event_id=f"{run_id}:snapshot",
                data={
                    "run_id": run_id,
                    "status": initial_summary.get("status"),
                    "saved": initial_summary.get("saved"),
                    "failed": initial_summary.get("failed"),
                    "queued": initial_summary.get("queued"),
                    "running": initial_summary.get("running"),
                    "requested_runs": initial_summary.get("requested_runs"),
                },
            )

            while True:
                if await request.is_disconnected():
                    break

                summary = get_batch_run_status(run_id)
                if not summary:
                    yield sse_message(
                        event="error",
                        event_id=f"{run_id}:missing",
                        data={"run_id": run_id, "message": "Batch run not found."},
                    )
                    break

                progress_snapshot = (
                    summary.get("status"),
                    summary.get("saved"),
                    summary.get("failed"),
                    summary.get("queued"),
                    summary.get("running"),
                )
                if progress_snapshot != last_progress_snapshot:
                    last_progress_snapshot = progress_snapshot
                    yield sse_message(
                        event="progress",
                        event_id=f"{run_id}:progress:{summary.get('saved')}:{summary.get('failed')}:{summary.get('queued')}:{summary.get('running')}",
                        data={
                            "run_id": run_id,
                            "status": summary.get("status"),
                            "saved": summary.get("saved"),
                            "failed": summary.get("failed"),
                            "queued": summary.get("queued"),
                            "running": summary.get("running"),
                            "requested_runs": summary.get("requested_runs"),
                        },
                    )

                for item in summary.get("results", []):
                    item_run_id = item.get("run_id")
                    if not item_run_id or item_run_id in emitted_terminal_runs:
                        continue
                    if item.get("status") not in {"saved", "failed"}:
                        continue
                    item_index = item.get("index")
                    emitted_terminal_runs.add(item_run_id)
                    yield sse_message(
                        event="dataset_complete",
                        event_id=(f"item:{item_index}:{item.get('status')}" if isinstance(item_index, int) else item_run_id),
                        data=build_stream_item_payload(item),
                    )

                if summary.get("status") in {"completed", "failed", "cancelled"}:
                    yield sse_message(
                        event="batch_complete",
                        event_id=f"{run_id}:complete",
                        data={
                            "run_id": run_id,
                            "status": summary.get("status"),
                            "saved": summary.get("saved"),
                            "failed": summary.get("failed"),
                            "queued": summary.get("queued"),
                            "running": summary.get("running"),
                            "requested_runs": summary.get("requested_runs"),
                        },
                    )
                    break

                yield ": keep-alive\n\n"
                await asyncio.sleep(1.0)

        return EventSourceResponse(event_generator())

    @router.post("/batch/{run_id}/resume")
    async def resume_batch_endpoint(run_id: str):
        try:
            summary = await resume_batch_run(run_id)
            return response_builder(success=True, message="Batch run resumed.", statusCode=200, data=summary)
        except ValueError as e:
            return response_builder(success=False, message=str(e), statusCode=404)
        except Exception as e:
            logger.saveToLog(f"[resume_batch_endpoint] Unexpected error: {e}", "ERROR")
            return response_builder(
                success=False,
                message="An error occurred while resuming the batch run.",
                statusCode=500,
            )

    @router.post("/batch/{run_id}/pause")
    def pause_batch_endpoint(run_id: str):
        try:
            summary = pause_batch_run(run_id)
            if not summary:
                return response_builder(success=False, message="Batch run not found.", statusCode=404)
            return response_builder(
                success=True,
                message="Batch run paused. In-flight items may still finish.",
                statusCode=200,
                data=summary,
            )
        except Exception as e:
            logger.saveToLog(f"[pause_batch_endpoint] Unexpected error: {e}", "ERROR")
            return response_builder(
                success=False,
                message="An error occurred while pausing the batch run.",
                statusCode=500,
            )

    @router.post("/batch/{run_id}/stop")
    def stop_batch_endpoint(run_id: str):
        try:
            summary = stop_batch_run(run_id)
            if not summary:
                return response_builder(success=False, message="Batch run not found.", statusCode=404)
            return response_builder(
                success=True,
                message="Batch run stop requested. In-flight items may still finish or be marked cancelled.",
                statusCode=200,
                data=summary,
            )
        except Exception as e:
            logger.saveToLog(f"[stop_batch_endpoint] Unexpected error: {e}", "ERROR")
            return response_builder(
                success=False,
                message="An error occurred while stopping the batch run.",
                statusCode=500,
            )

    @router.post("/batch/{run_id}/restart-failed")
    async def restart_failed_batch_endpoint(run_id: str):
        try:
            reset_summary = restart_failed_batch_run(run_id)
            if not reset_summary:
                return response_builder(success=False, message="Batch run not found.", statusCode=404)
            resumed_summary = await resume_batch_run(run_id)
            return response_builder(
                success=True,
                message="Failed batch items requeued and resumed.",
                statusCode=200,
                data={"reset_summary": reset_summary, "current_summary": resumed_summary},
            )
        except ValueError as e:
            return response_builder(success=False, message=str(e), statusCode=400)
        except Exception as e:
            logger.saveToLog(f"[restart_failed_batch_endpoint] Unexpected error: {e}", "ERROR")
            return response_builder(
                success=False,
                message="An error occurred while restarting failed batch items.",
                statusCode=500,
            )
