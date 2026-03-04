import asyncio
import random

import httpx

from app.core.generics import new_run_id
from app.core.logger import saveToLog

from .batch_store import (
    batch_run_control_state,
    claim_existing_dataset_if_present,
    create_batch_run,
    delete_batch_run as _delete_batch_run,
    delete_terminal_batch_runs as _delete_terminal_batch_runs,
    finalize_item_failure,
    finalize_item_success,
    get_batch_run_status,
    list_incomplete_batch_runs as _list_incomplete_batch_runs,
    load_existing_dataset_names,
    pause_batch_run,
    pending_item_ids_for_batch,
    prepare_item_attempt,
    restart_failed_batch_run,
    stop_batch_run,
)
from .generation import generate_dataset
from .naming import run_topic_variation_agent
from .persistence import save_responses
from .types import AgentType

_ACTIVE_BATCH_RUNS: set[str] = set()
_ACTIVE_BATCH_RUNS_GUARD = asyncio.Lock()


def _validate_generation_request(
    *,
    amount: int,
    ex_amt: int,
    max_concurrency: int | None = None,
    max_retries: int,
    retry_backoff_seconds: float,
) -> None:
    if amount <= 0 or ex_amt <= 0:
        raise ValueError("amount and ex_amt must both be greater than 0")
    if max_concurrency is not None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0")
        if max_concurrency > 50:
            raise ValueError("max_concurrency must be less than or equal to 50")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds cannot be negative")


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


def _retry_delay_seconds(*, attempts: int, retry_backoff_seconds: float) -> float:
    return retry_backoff_seconds * (2 ** (attempts - 1))


def _build_run_plan_item(
    *,
    item_index: int,
    run_id: str,
    requested_topic: str,
    topic: str,
    resolved_agent: AgentType,
    ex_amt: int,
    max_retries: int,
    retry_backoff_seconds: float,
    source_material: str | None,
    model: str | None,
    seed: int,
    slot_key: str | None,
) -> dict:
    return {
        "item_index": item_index,
        "run_id": run_id,
        "dataset_key": f"{run_id}: {topic}",
        "slot_key": slot_key,
        "requested_topic": requested_topic,
        "topic": topic,
        "agent": resolved_agent.value,
        "ex_amt": ex_amt,
        "max_retries": max_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
        "source_material": source_material,
        "model": model,
        "seed": seed,
    }


def _build_batch_request_payload(
    *,
    topic: str,
    agent: AgentType | None,
    model: str | None,
    source_material: str | None,
    source_material_mode: str,
    conversation_length_mode: str,
    amount: int,
    ex_amt: int,
    random_agent: bool,
    max_concurrency: int,
    max_retries: int,
    retry_backoff_seconds: float,
    seed: int | None,
    planned_topics: list[str],
    slot_key: str | None,
    request_group_id: str | None,
) -> dict:
    return {
        "request_group_id": request_group_id,
        "topic": topic,
        "agent": agent.value if agent else None,
        "model": model,
        "source_material": source_material,
        "source_material_mode": source_material_mode,
        "conversation_length_mode": conversation_length_mode,
        "amount": amount,
        "ex_amt": ex_amt,
        "random_agent": random_agent,
        "max_concurrency": max_concurrency,
        "max_retries": max_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
        "seed": seed,
        "planned_topics": planned_topics,
        "slot_key": slot_key,
    }


async def build_generation_plan(
    *,
    amount: int,
    topic: str,
    agent: AgentType | None = None,
    model: str | None = None,
    source_material: str | None = None,
    source_material_mode: str = "content_and_style",
    conversation_length_mode: str = "varied",
    ex_amt: int = 20,
    random_agent: bool = False,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.0,
    seed: int | None = None,
    slot_key: str | None = None,
    allow_topic_variations: bool = True,
) -> dict:
    _validate_generation_request(
        amount=amount,
        ex_amt=ex_amt,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

    rng = random.Random(seed) if seed is not None else random.Random()
    planner_run_id = new_run_id()
    try:
        existing_names = load_existing_dataset_names()
    except Exception as e:
        saveToLog(f"[start_generation] Failed to load existing dataset names: {e}", "WARNING")
        existing_names = []
    if allow_topic_variations:
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
    else:
        planned_topics = [topic]

    if not planned_topics:
        planned_topics = [topic]
    rng.shuffle(planned_topics)

    run_plan: list[dict] = []
    for i in range(amount):
        run_id = new_run_id()
        run_topic = planned_topics[i % len(planned_topics)]
        resolved_agent = get_random_agent(rng) if random_agent else (agent or AgentType.qa)
        run_seed = rng.randint(0, 2_147_483_647)
        run_plan.append(
            _build_run_plan_item(
                item_index=i,
                run_id=run_id,
                requested_topic=topic,
                topic=run_topic,
                resolved_agent=resolved_agent,
                ex_amt=ex_amt,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                source_material=source_material,
                model=model,
                seed=run_seed,
                slot_key=slot_key,
            )
        )

    return {"planned_topics": planned_topics, "run_plan": run_plan}


# Executes one persisted batch item from claim through finalization.
# This function coordinates retry handling, generation, ingest, and
# cooperative pause/stop behavior against the current batch control state.
async def _execute_batch_item(item_id: int) -> dict | None:
    claimed = claim_existing_dataset_if_present(item_id)
    if claimed:
        return claimed

    while True:
        control_state = batch_run_control_state(item_id)
        if not control_state:
            return None
        _, batch_status = control_state
        if batch_status == "paused":
            return None
        if batch_status == "cancelled":
            return finalize_item_failure(
                item_id,
                error_type="cancelled",
                error="Stopped by user before execution",
                terminal=True,
            )

        prepared = prepare_item_attempt(item_id)
        if not prepared:
            return None
        item_data, exhausted = prepared
        if exhausted:
            return finalize_item_failure(
                item_id,
                error_type="runtime_error",
                error="Retry budget exhausted before execution resumed",
                terminal=True,
            )

        resolved_agent = AgentType(item_data["agent"])
        try:
            saveToLog(
                f"[start_generation] Running run_id={item_data['run_id']} agent={resolved_agent.value}",
                "INFO",
            )
            dataset, prompt = await generate_dataset(
                amt=item_data["ex_amt"],
                topic=item_data["topic"],
                agent_type=resolved_agent,
                model=item_data["model"],
                source_material=item_data["source_material"],
                source_material_mode=item_data["source_material_mode"],
                conversation_length_mode=item_data["conversation_length_mode"],
                run_id=item_data["run_id"],
                dataset_key=item_data["dataset_key"],
                seed=item_data["seed"],
            )
            control_state = batch_run_control_state(item_id)
            if control_state and control_state[1] == "cancelled":
                return finalize_item_failure(
                    item_id,
                    error_type="cancelled",
                    error="Stop requested before dataset save completed",
                    terminal=True,
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
            control_state = batch_run_control_state(item_id)
            if control_state and control_state[1] == "cancelled":
                return finalize_item_failure(
                    item_id,
                    error_type="cancelled",
                    error="Stop requested while item was already running",
                    terminal=True,
                )
            return finalize_item_success(item_id, dataset_id=dataset_id)
        except ValueError as e:
            saveToLog(
                f"[start_generation] Validation error run_id={item_data['run_id']} attempt={item_data['attempts']}: {e}",
                "WARNING",
            )
            return finalize_item_failure(
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
            result = finalize_item_failure(
                item_id,
                error_type="runtime_error",
                error=str(e),
                terminal=terminal,
            )
            if terminal:
                return result
            await asyncio.sleep(
                _retry_delay_seconds(
                    attempts=item_data["attempts"],
                    retry_backoff_seconds=item_data["retry_backoff_seconds"],
                )
            )
        except Exception as e:
            saveToLog(
                f"[start_generation] Unexpected error run_id={item_data['run_id']} attempt={item_data['attempts']}: {e}",
                "WARNING",
            )
            terminal = (not _is_transient_error(e)) or (
                item_data["attempts"] >= (item_data["max_retries"] + 1)
            )
            result = finalize_item_failure(
                item_id,
                error_type="runtime_error",
                error=str(e),
                terminal=terminal,
            )
            if terminal:
                return result
            await asyncio.sleep(
                _retry_delay_seconds(
                    attempts=item_data["attempts"],
                    retry_backoff_seconds=item_data["retry_backoff_seconds"],
                )
            )


async def resume_batch_run(batch_run_id: str, *, max_concurrency: int | None = None) -> dict:
    async with _ACTIVE_BATCH_RUNS_GUARD:
        if batch_run_id in _ACTIVE_BATCH_RUNS:
            current = get_batch_run_status(batch_run_id)
            if current is not None:
                return current
        _ACTIVE_BATCH_RUNS.add(batch_run_id)

    try:
        batch_state, pending_item_ids = pending_item_ids_for_batch(batch_run_id)
        if isinstance(batch_state, dict):
            request_payload = batch_state
        else:
            request_payload = {}

        if "status" in request_payload and not pending_item_ids:
            return request_payload

        effective_concurrency = max_concurrency or int(request_payload.get("max_concurrency", 3) or 3)
        if effective_concurrency <= 0:
            effective_concurrency = 1

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
    return _list_incomplete_batch_runs()


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
    source_material_mode: str = "content_and_style",
    conversation_length_mode: str = "varied",
    ex_amt: int = 20,
    random_agent: bool = False,
    max_concurrency: int = 3,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.0,
    seed: int | None = None,
    batch_run_id: str | None = None,
    slot_key: str | None = None,
    allow_topic_variations: bool = True,
    request_group_id: str | None = None,
    wait_for_completion: bool = True,
):
    _validate_generation_request(
        amount=amount,
        ex_amt=ex_amt,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

    if batch_run_id:
        return await resume_batch_run(batch_run_id, max_concurrency=max_concurrency)

    plan = await build_generation_plan(
        amount=amount,
        topic=topic,
        agent=agent,
        model=model,
        source_material=source_material,
        source_material_mode=source_material_mode,
        conversation_length_mode=conversation_length_mode,
        ex_amt=ex_amt,
        random_agent=random_agent,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        seed=seed,
        slot_key=slot_key,
        allow_topic_variations=allow_topic_variations,
    )
    request_payload = _build_batch_request_payload(
        topic=topic,
        agent=agent,
        model=model,
        source_material=source_material,
        source_material_mode=source_material_mode,
        conversation_length_mode=conversation_length_mode,
        amount=amount,
        ex_amt=ex_amt,
        random_agent=random_agent,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        seed=seed,
        planned_topics=plan["planned_topics"],
        slot_key=slot_key,
        request_group_id=request_group_id,
    )
    batch_run_id = create_batch_run(
        request_payload=request_payload,
        run_plan=plan["run_plan"],
        run_id=new_run_id(),
    )
    if not wait_for_completion:
        asyncio.create_task(resume_batch_run(batch_run_id, max_concurrency=max_concurrency))
        summary = get_batch_run_status(batch_run_id)
        if summary is None:
            raise ValueError("Batch run not found after creation")
        return summary
    return await resume_batch_run(batch_run_id, max_concurrency=max_concurrency)


def delete_batch_run(batch_run_id: str) -> dict | None:
    summary = _delete_batch_run(batch_run_id)
    if summary is not None:
        _ACTIVE_BATCH_RUNS.discard(batch_run_id)
    return summary


def delete_terminal_batch_runs() -> dict:
    result = _delete_terminal_batch_runs()
    for run_id in result.get("deleted_run_ids", []):
        _ACTIVE_BATCH_RUNS.discard(run_id)
    return result



