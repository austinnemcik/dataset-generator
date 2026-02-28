import asyncio
import json
import random
from collections import defaultdict

import httpx
from sqlmodel import Session, select

from database import BatchRun, BatchRunItem, Dataset, engine, utcnow
from generics import new_run_id
from logger import saveToLog

from .generation import generate_dataset
from .naming import run_topic_variation_agent
from .persistence import save_responses
from .types import AgentType

_TERMINAL_ITEM_STATUSES = {"completed", "failed"}
_ACTIVE_BATCH_RUNS: set[str] = set()
_ACTIVE_BATCH_RUNS_GUARD = asyncio.Lock()


def get_random_agent(rng: random.Random | None = None) -> AgentType:
    picker = rng or random
    while True:
        choice = picker.choice(list(AgentType))
        if choice != AgentType.domain_specialist:
            return choice


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
        return True
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError)):
        return True
    message = str(exc).lower()
    transient_markers = (
        "timeout",
        "temporar",
        "rate limit",
        "connection reset",
        "connection aborted",
        "completion call failed",
        "ingest api call failed",
    )
    return any(marker in message for marker in transient_markers)


def _suggest_topic_count(amount: int) -> int:
    return max(1, min(amount, max(3, int(amount**0.5) * 2)))


def _load_existing_dataset_names(limit: int = 250) -> list[str]:
    try:
        with Session(engine) as session:
            datasets = session.exec(select(Dataset).order_by(Dataset.id.desc()).limit(limit)).all()
        names = [str(ds.name).strip() for ds in datasets if getattr(ds, "name", None)]
        return [n for n in names if n]
    except Exception as e:
        saveToLog(f"[start_generation] Failed to load existing dataset names: {e}", "WARNING")
        return []


def _json_dump(value: dict | list) -> str:
    return json.dumps(value)


def _json_load(raw: str | None, fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _update_batch_counts(session: Session, batch_run: BatchRun):
    items = session.exec(
        select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
    ).all()
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


def _result_from_item(item: BatchRunItem) -> dict:
    return {
        "index": item.item_index,
        "run_id": item.run_id,
        "dataset_key": item.dataset_key,
        "agent": item.agent,
        "status": "saved" if item.status == "completed" else "failed",
        "attempts": item.attempts,
        "error_type": item.error_type,
        "error": item.error,
        "dataset_id": item.created_dataset_id,
        "topic": item.topic,
        "requested_topic": item.requested_topic,
        "slot_key": item.slot_key,
    }


def _build_batch_summary(batch_run: BatchRun, items: list[BatchRunItem]) -> dict:
    request_payload = _json_load(batch_run.request_json, {})
    results = [_result_from_item(item) for item in sorted(items, key=lambda entry: entry.item_index)]
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
        "status": batch_run.status,
        "requested_runs": batch_run.total_runs,
        "generated": batch_run.completed_runs,
        "saved": batch_run.completed_runs,
        "failed": batch_run.failed_runs,
        "queued": batch_run.queued_runs,
        "running": batch_run.running_runs,
        "seed": request_payload.get("seed"),
        "max_concurrency": request_payload.get("max_concurrency"),
        "max_retries": request_payload.get("max_retries"),
        "retry_backoff_seconds": request_payload.get("retry_backoff_seconds"),
        "topic": request_payload.get("topic"),
        "planned_topics": request_payload.get("planned_topics", []),
        "requested_topic": request_payload.get("topic"),
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
    batch_run.summary_json = _json_dump(summary)
    return summary


async def build_generation_plan(
    *,
    amount: int,
    topic: str,
    agent: AgentType | None = None,
    model: str | None = None,
    source_material: str | None = None,
    ex_amt: int = 20,
    random_agent: bool = False,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.0,
    seed: int | None = None,
    slot_key: str | None = None,
) -> dict:
    if amount <= 0 or ex_amt <= 0:
        raise ValueError("amount and ex_amt must both be greater than 0")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds cannot be negative")

    rng = random.Random(seed) if seed is not None else random.Random()
    planner_run_id = new_run_id()
    existing_names = _load_existing_dataset_names()
    topic_count = _suggest_topic_count(amount)
    try:
        planned_topics = await run_topic_variation_agent(
            topic=topic,
            topic_count=topic_count,
            existing_dataset_names=existing_names,
            run_id=planner_run_id,
            model=model,
        )
    except Exception as e:
        saveToLog(
            f"[start_generation] Topic planning failed, falling back to base topic: {e}",
            "WARNING",
        )
        planned_topics = [topic]

    if not planned_topics:
        planned_topics = [topic]
    rng.shuffle(planned_topics)

    run_plan: list[dict] = []
    for i in range(amount):
        run_id = new_run_id()
        run_topic = planned_topics[i % len(planned_topics)]
        resolved_agent = get_random_agent(rng) if random_agent else (agent or AgentType.qa)
        dataset_key = f"{run_id}: {run_topic}"
        run_plan.append(
            {
                "item_index": i,
                "run_id": run_id,
                "dataset_key": dataset_key,
                "slot_key": slot_key,
                "requested_topic": topic,
                "topic": run_topic,
                "agent": resolved_agent.value,
                "ex_amt": ex_amt,
                "max_retries": max_retries,
                "retry_backoff_seconds": retry_backoff_seconds,
                "source_material": source_material,
                "model": model,
                "seed": seed,
            }
        )

    return {"planned_topics": planned_topics, "run_plan": run_plan}


def create_batch_run(*, request_payload: dict, run_plan: list[dict], run_id: str | None = None) -> str:
    batch_run_id = run_id or new_run_id()
    with Session(engine) as session:
        batch_run = BatchRun(
            run_id=batch_run_id,
            status="queued",
            request_json=_json_dump(request_payload),
            total_runs=len(run_plan),
            queued_runs=len(run_plan),
            running_runs=0,
            completed_runs=0,
            failed_runs=0,
        )
        session.add(batch_run)
        session.flush()

        for item in run_plan:
            session.add(
                BatchRunItem(
                    batch_run_id=batch_run.id,
                    item_index=int(item["item_index"]),
                    run_id=str(item["run_id"]),
                    dataset_key=str(item["dataset_key"]),
                    slot_key=item.get("slot_key"),
                    requested_topic=str(item["requested_topic"]),
                    topic=str(item["topic"]),
                    agent=str(item["agent"]),
                    ex_amt=int(item["ex_amt"]),
                    seed=item.get("seed"),
                    status="queued",
                    attempts=0,
                    max_retries=int(item["max_retries"]),
                    retry_backoff_seconds=float(item["retry_backoff_seconds"]),
                    source_material=item.get("source_material"),
                    model=item.get("model"),
                )
            )

        session.flush()
        _update_batch_counts(session, batch_run)
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        _build_batch_summary(batch_run, items)
        session.add(batch_run)
        session.commit()

    return batch_run_id


def get_batch_run_status(batch_run_id: str) -> dict | None:
    with Session(engine) as session:
        batch_run = session.exec(select(BatchRun).where(BatchRun.run_id == batch_run_id)).first()
        if not batch_run:
            return None
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        _update_batch_counts(session, batch_run)
        summary = _build_batch_summary(batch_run, items)
        session.add(batch_run)
        session.commit()
        return summary


def _claim_existing_dataset_if_present(item_id: int) -> dict | None:
    with Session(engine) as session:
        item = session.get(BatchRunItem, item_id)
        if not item:
            return None
        batch_run = session.get(BatchRun, item.batch_run_id)
        existing_dataset = session.exec(
            select(Dataset).where(Dataset.source_run_id == item.run_id)
        ).first()
        if not existing_dataset:
            return None
        item.status = "completed"
        item.created_dataset_id = existing_dataset.id
        item.error = None
        item.error_type = None
        item.completed_at = item.completed_at or utcnow()
        item.updated_at = utcnow()
        item.result_json = _json_dump(_result_from_item(item))
        _update_batch_counts(session, batch_run)
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        _build_batch_summary(batch_run, items)
        session.add(item)
        session.add(batch_run)
        session.commit()
        return _result_from_item(item)


def _prepare_item_attempt(item_id: int) -> tuple[dict, bool] | None:
    with Session(engine) as session:
        item = session.get(BatchRunItem, item_id)
        if not item:
            return None
        batch_run = session.get(BatchRun, item.batch_run_id)
        if item.status in _TERMINAL_ITEM_STATUSES:
            return None

        item.attempts += 1
        item.status = "running"
        item.started_at = item.started_at or utcnow()
        item.updated_at = utcnow()
        item.error = None
        item.error_type = None
        _update_batch_counts(session, batch_run)
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        _build_batch_summary(batch_run, items)
        session.add(item)
        session.add(batch_run)
        session.commit()
        session.refresh(item)
        return (
            {
                "id": item.id,
                "batch_run_id": batch_run.run_id,
                "item_index": item.item_index,
                "run_id": item.run_id,
                "dataset_key": item.dataset_key,
                "requested_topic": item.requested_topic,
                "topic": item.topic,
                "agent": item.agent,
                "ex_amt": item.ex_amt,
                "attempts": item.attempts,
                "max_retries": item.max_retries,
                "retry_backoff_seconds": item.retry_backoff_seconds,
                "source_material": item.source_material,
                "model": item.model,
                "slot_key": item.slot_key,
            },
            item.attempts > (item.max_retries + 1),
        )


def _finalize_item_success(item_id: int, dataset_id: int | None = None) -> dict | None:
    with Session(engine) as session:
        item = session.get(BatchRunItem, item_id)
        if not item:
            return None
        batch_run = session.get(BatchRun, item.batch_run_id)
        item.status = "completed"
        item.created_dataset_id = dataset_id
        item.error = None
        item.error_type = None
        item.completed_at = utcnow()
        item.updated_at = utcnow()
        item.result_json = _json_dump(_result_from_item(item))
        _update_batch_counts(session, batch_run)
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        _build_batch_summary(batch_run, items)
        session.add(item)
        session.add(batch_run)
        session.commit()
        return _result_from_item(item)


def _finalize_item_failure(item_id: int, *, error_type: str, error: str, terminal: bool) -> dict | None:
    with Session(engine) as session:
        item = session.get(BatchRunItem, item_id)
        if not item:
            return None
        batch_run = session.get(BatchRun, item.batch_run_id)
        item.error_type = error_type
        item.error = error
        item.updated_at = utcnow()
        if terminal:
            item.status = "failed"
            item.completed_at = utcnow()
        else:
            item.status = "queued"
        item.result_json = _json_dump(_result_from_item(item))
        _update_batch_counts(session, batch_run)
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        _build_batch_summary(batch_run, items)
        session.add(item)
        session.add(batch_run)
        session.commit()
        return _result_from_item(item)


async def _execute_batch_item(item_id: int) -> dict | None:
    claimed = _claim_existing_dataset_if_present(item_id)
    if claimed:
        return claimed

    while True:
        prepared = _prepare_item_attempt(item_id)
        if not prepared:
            return None
        item_data, exhausted = prepared
        if exhausted:
            return _finalize_item_failure(
                item_id,
                error_type="runtime_error",
                error="Retry budget exhausted before execution resumed",
                terminal=True,
            )

        resolved_agent = AgentType(item_data["agent"])
        try:
            print(f"Running {item_data['run_id']} with agent {resolved_agent}")
            dataset, prompt = await generate_dataset(
                amt=item_data["ex_amt"],
                topic=item_data["topic"],
                agent_type=resolved_agent,
                model=item_data["model"],
                source_material=item_data["source_material"],
                run_id=item_data["run_id"],
                dataset_key=item_data["dataset_key"],
            )
            ingest_result = await save_responses(
                agent_type=resolved_agent,
                examples=dataset,
                prompt=prompt,
                topic=item_data["topic"],
                model=item_data["model"],
                amount=item_data["ex_amt"],
                source_material=item_data["source_material"],
                run_id=item_data["run_id"],
                dataset_key=item_data["dataset_key"],
            )
            payload = ingest_result.get("data", {}) if isinstance(ingest_result, dict) else {}
            dataset_id = payload.get("dataset_id") if isinstance(payload, dict) else None
            return _finalize_item_success(item_id, dataset_id=dataset_id)
        except ValueError as e:
            saveToLog(
                f"[start_generation] Validation error run_id={item_data['run_id']} attempt={item_data['attempts']}: {e}",
                "WARNING",
            )
            return _finalize_item_failure(
                item_id,
                error_type="validation_error",
                error=str(e),
                terminal=True,
            )
        except RuntimeError as e:
            saveToLog(
                f"[start_generation] Runtime error run_id={item_data['run_id']} attempt={item_data['attempts']}: {e}",
                "WARNING",
            )
            terminal = (not _is_transient_error(e)) or (
                item_data["attempts"] >= (item_data["max_retries"] + 1)
            )
            result = _finalize_item_failure(
                item_id,
                error_type="runtime_error",
                error=str(e),
                terminal=terminal,
            )
            if terminal:
                return result
            await asyncio.sleep(item_data["retry_backoff_seconds"] * (2 ** (item_data["attempts"] - 1)))
        except Exception as e:
            saveToLog(
                f"[start_generation] Unexpected error run_id={item_data['run_id']} attempt={item_data['attempts']}: {e}",
                "WARNING",
            )
            terminal = (not _is_transient_error(e)) or (
                item_data["attempts"] >= (item_data["max_retries"] + 1)
            )
            result = _finalize_item_failure(
                item_id,
                error_type="runtime_error",
                error=str(e),
                terminal=terminal,
            )
            if terminal:
                return result
            await asyncio.sleep(item_data["retry_backoff_seconds"] * (2 ** (item_data["attempts"] - 1)))


async def resume_batch_run(batch_run_id: str, *, max_concurrency: int | None = None) -> dict:
    async with _ACTIVE_BATCH_RUNS_GUARD:
        if batch_run_id in _ACTIVE_BATCH_RUNS:
            current = get_batch_run_status(batch_run_id)
            if current is not None:
                return current
        _ACTIVE_BATCH_RUNS.add(batch_run_id)

    try:
        with Session(engine) as session:
            batch_run = session.exec(select(BatchRun).where(BatchRun.run_id == batch_run_id)).first()
            if not batch_run:
                raise ValueError("Batch run not found")

            request_payload = _json_load(batch_run.request_json, {})
            effective_concurrency = max_concurrency or int(request_payload.get("max_concurrency", 3) or 3)
            if effective_concurrency <= 0:
                effective_concurrency = 1

            items = session.exec(
                select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
            ).all()
            for item in items:
                if item.status == "running":
                    item.status = "queued"
                    item.updated_at = utcnow()
            _update_batch_counts(session, batch_run)
            _build_batch_summary(batch_run, items)
            session.add(batch_run)
            session.commit()

            pending_item_ids = [item.id for item in items if item.status not in _TERMINAL_ITEM_STATUSES]

        if not pending_item_ids:
            summary = get_batch_run_status(batch_run_id)
            if summary is None:
                raise ValueError("Batch run not found")
            return summary

        queue: asyncio.Queue[int] = asyncio.Queue()
        for item_id in pending_item_ids:
            queue.put_nowait(item_id)

        completed = 0
        progress_every = max(1, len(pending_item_ids) // 10)
        progress_lock = asyncio.Lock()

        async def _worker(worker_id: int):
            nonlocal completed
            while True:
                try:
                    item_id = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    await _execute_batch_item(item_id)
                    async with progress_lock:
                        completed += 1
                        if completed % progress_every == 0 or completed == len(pending_item_ids):
                            saveToLog(
                                f"[resume_batch_run] batch_run_id={batch_run_id} progress={completed}/{len(pending_item_ids)} worker={worker_id}",
                                "INFO",
                            )
                finally:
                    queue.task_done()

        worker_count = min(effective_concurrency, len(pending_item_ids))
        workers = [asyncio.create_task(_worker(i + 1)) for i in range(worker_count)]
        await queue.join()
        await asyncio.gather(*workers)

        summary = get_batch_run_status(batch_run_id)
        if summary is None:
            raise ValueError("Batch run not found")

        saveToLog(
            f"[resume_batch_run] Completed batch_run_id={batch_run_id} saved={summary['saved']} failed={summary['failed']}",
            "INFO",
        )
        return summary
    finally:
        async with _ACTIVE_BATCH_RUNS_GUARD:
            _ACTIVE_BATCH_RUNS.discard(batch_run_id)


def list_incomplete_batch_runs() -> list[str]:
    with Session(engine) as session:
        runs = session.exec(
            select(BatchRun).where(BatchRun.status.in_(["queued", "running"])).order_by(BatchRun.created_at)
        ).all()
        return [run.run_id for run in runs]


async def resume_incomplete_batch_runs() -> list[str]:
    resumed: list[str] = []
    for batch_run_id in list_incomplete_batch_runs():
        resumed.append(batch_run_id)
        asyncio.create_task(resume_batch_run(batch_run_id))
    return resumed


async def start_generation(
    amount: int,
    topic: str,
    agent: AgentType | None = None,
    model: str | None = None,
    source_material: str | None = None,
    ex_amt: int = 20,
    random_agent: bool = False,
    max_concurrency: int = 3,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.0,
    seed: int | None = None,
    batch_run_id: str | None = None,
    slot_key: str | None = None,
):
    if amount <= 0 or ex_amt <= 0:
        raise ValueError("amount and ex_amt must both be greater than 0")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than 0")
    if max_concurrency > 50:
        raise ValueError("max_concurrency must be less than or equal to 50")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds cannot be negative")

    if batch_run_id:
        return await resume_batch_run(batch_run_id, max_concurrency=max_concurrency)

    plan = await build_generation_plan(
        amount=amount,
        topic=topic,
        agent=agent,
        model=model,
        source_material=source_material,
        ex_amt=ex_amt,
        random_agent=random_agent,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        seed=seed,
        slot_key=slot_key,
    )
    request_payload = {
        "topic": topic,
        "agent": agent.value if agent else None,
        "model": model,
        "source_material": source_material,
        "amount": amount,
        "ex_amt": ex_amt,
        "random_agent": random_agent,
        "max_concurrency": max_concurrency,
        "max_retries": max_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
        "seed": seed,
        "planned_topics": plan["planned_topics"],
        "slot_key": slot_key,
    }
    batch_run_id = create_batch_run(
        request_payload=request_payload,
        run_plan=plan["run_plan"],
    )
    return await resume_batch_run(batch_run_id, max_concurrency=max_concurrency)
