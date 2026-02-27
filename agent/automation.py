import asyncio
import random
from collections import defaultdict

import httpx

from generics import new_run_id
from logger import saveToLog

from .generation import generate_dataset
from .persistence import save_responses
from .types import AgentType


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
):
    if amount <= 0 or ex_amt <= 0:
        raise ValueError("amount and ex_amt must both be greater than 0")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than 0")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds cannot be negative")

    rng = random.Random(seed) if seed is not None else random.Random()
    semaphore = asyncio.Semaphore(max_concurrency)
    agent_counts: defaultdict[str, int] = defaultdict(int)
    run_plan: list[dict] = []
    run_results: list[dict] = []

    for i in range(amount):
        run_id = new_run_id()
        dataset_key = f"{run_id}: {topic}"
        if random_agent:
            resolved_agent = get_random_agent(rng)
        else:
            resolved_agent = agent or AgentType.qa
        run_plan.append(
            {
                "index": i,
                "run_id": run_id,
                "dataset_key": dataset_key,
                "agent": resolved_agent,
            }
        )

    async def _execute_run(plan_entry: dict) -> dict:
        async with semaphore:
            run_id = plan_entry["run_id"]
            dataset_key = plan_entry["dataset_key"]
            resolved_agent: AgentType = plan_entry["agent"]
            last_error: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    dataset, prompt = await generate_dataset(
                        amt=ex_amt,
                        topic=topic,
                        agent_type=resolved_agent,
                        model=model,
                        source_material=source_material,
                        run_id=run_id,
                        dataset_key=dataset_key,
                    )
                    await save_responses(
                        agent_type=resolved_agent,
                        examples=dataset,
                        prompt=prompt,
                        topic=topic,
                        model=model,
                        amount=ex_amt,
                        source_material=source_material,
                        run_id=run_id,
                        dataset_key=dataset_key,
                    )
                    return {
                        "index": plan_entry["index"],
                        "run_id": run_id,
                        "dataset_key": dataset_key,
                        "agent": resolved_agent.value,
                        "status": "saved",
                        "attempts": attempt + 1,
                        "error_type": None,
                        "error": None,
                    }
                except ValueError as e:
                    saveToLog(
                        f"[start_generation] Validation error run_id={run_id} attempt={attempt + 1}: {e}",
                        "WARNING",
                    )
                    return {
                        "index": plan_entry["index"],
                        "run_id": run_id,
                        "dataset_key": dataset_key,
                        "agent": resolved_agent.value,
                        "status": "failed",
                        "attempts": attempt + 1,
                        "error_type": "validation_error",
                        "error": str(e),
                    }
                except RuntimeError as e:
                    last_error = e
                    is_retryable = _is_transient_error(e)
                    saveToLog(
                        f"[start_generation] Runtime error run_id={run_id} attempt={attempt + 1}: {e}",
                        "WARNING",
                    )
                    if not is_retryable or attempt >= max_retries:
                        break
                    await asyncio.sleep(retry_backoff_seconds * (2**attempt))
                except Exception as e:
                    last_error = e
                    is_retryable = _is_transient_error(e)
                    saveToLog(
                        f"[start_generation] Unexpected error run_id={run_id} attempt={attempt + 1}: {e}",
                        "WARNING",
                    )
                    if not is_retryable or attempt >= max_retries:
                        break
                    await asyncio.sleep(retry_backoff_seconds * (2**attempt))

            return {
                "index": plan_entry["index"],
                "run_id": run_id,
                "dataset_key": dataset_key,
                "agent": resolved_agent.value,
                "status": "failed",
                "attempts": max_retries + 1,
                "error_type": "runtime_error",
                "error": str(last_error) if last_error else "Unknown error",
            }

    tasks = [asyncio.create_task(_execute_run(entry)) for entry in run_plan]
    run_results = await asyncio.gather(*tasks)
    run_results.sort(key=lambda item: item["index"])

    total_generated = sum(1 for r in run_results if r["status"] == "saved")
    total_saved = total_generated
    failed = amount - total_saved
    dataset_keys = [r["dataset_key"] for r in run_results if r["status"] == "saved"]
    run_ids = [r["run_id"] for r in run_results if r["status"] == "saved"]

    for result in run_results:
        if result["status"] == "saved":
            agent_counts[result["agent"]] += 1

    summary = {
        "requested_runs": amount,
        "generated": total_generated,
        "saved": total_saved,
        "failed": failed,
        "topic": topic,
        "seed": seed,
        "max_concurrency": max_concurrency,
        "max_retries": max_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
        "agent_usage": dict(agent_counts),
        "run_ids": run_ids,
        "dataset_keys": dataset_keys,
        "results": run_results,
    }

    saveToLog(
        f"[start_generation] Completed batch topic={topic} requested={amount} "
        f"saved={total_saved} failed={failed} agent_usage={dict(agent_counts)}",
        "INFO",
    )
    return summary
