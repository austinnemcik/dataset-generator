import httpx
import logger
import math

from generics import get_run_costs, new_run_id

from .grading import run_grading_agent
from .naming import run_naming_agent
from .settings import DEFAULT_MODEL, MIN_SAVE_RATIO, SERVER_URL
from .types import AgentType


async def save_responses(
    *,
    prompt: str,
    examples: list[dict],
    topic: str,
    model: str = DEFAULT_MODEL,
    amount: int,
    agent_type: AgentType,
    source_material: str | None = None,
    run_id: str | None = None,
    dataset_key: str | None = None,
):
    model = model or DEFAULT_MODEL
    run_id = run_id or new_run_id()
    dataset_key = dataset_key or topic
    meta = await run_naming_agent(
        examples,
        run_id=run_id,
        dataset_key=dataset_key,
        topic=topic,
        model=model,
    )
    if not meta:
        logger.saveToLog("[save_responses] Naming agent failed.. aborting", "ERROR")
        raise ValueError("Naming agent returned no metadata")

    graded = await run_grading_agent(
        topic=topic,
        model=model,
        examples=examples,
        agent_type=agent_type,
        source_material=source_material,
        run_id=run_id,
        dataset_key=dataset_key,
    )
    required_min = max(1, math.ceil((amount or len(examples) or 1) * MIN_SAVE_RATIO))
    if len(graded) < required_min:
        logger.saveToLog(
            f"[save_responses] Valid examples below threshold. valid={len(graded)} required_min={required_min}. Aborting ingest.",
            "ERROR",
        )
        raise ValueError(
            f"Valid examples below threshold ({len(graded)}/{amount}); need at least {required_min}"
        )
    payload = {
        "dataset_name": meta["name"],
        "dataset_description": meta["description"],
        "dataset_id": 0,
        "model": model,
        "example": graded,
        "prompt": prompt,
    }
    costs = get_run_costs(run_id)
    payload["generation_cost"] = costs["generation_cost"]
    payload["grading_cost"] = costs["grading_cost"]
    payload["total_cost"] = costs["total_cost"]

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{SERVER_URL}/dataset/ingest", json=payload, timeout=30
            )
            res.raise_for_status()
            result = res.json()
            logger.saveToLog("[save_responses] Ingested response", "INFO")
            return result
    except httpx.HTTPStatusError as e:
        logger.saveToLog(f"[save_responses] Failed to POST to /ingest, {e}", "ERROR")
        raise RuntimeError("Ingest API returned an HTTP error") from e
    except Exception as e:
        logger.saveToLog(f"[save_responses] Failed to POST to /ingest, {e}", "ERROR")
        raise RuntimeError("Ingest API call failed") from e

