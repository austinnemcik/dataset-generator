import httpx
import app.core.logger as logger
import math

from app.core.generics import get_run_costs, new_run_id

from .grading import run_grading_agent
from .naming import run_naming_agent
from .settings import SERVER_URL, get_client_settings
from .types import AgentType


async def save_responses(
    *,
    prompt: str,
    examples: list[dict],
    topic: str,
    model: str | None = None,
    amount: int,
    agent_type: AgentType,
    source_material: str | None = None,
    run_id: str | None = None,
    dataset_key: str | None = None,
):
    client_settings = get_client_settings()
    model = model or client_settings.default_model
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

    graded_result = await run_grading_agent(
        topic=topic,
        model=model,
        examples=examples,
        agent_type=agent_type,
        source_material=source_material,
        run_id=run_id,
        dataset_key=dataset_key,
    )
    graded_examples = graded_result.get("accepted_examples", [])
    graded_category = graded_result.get("category")
    required_min = max(1, math.ceil((amount or len(examples) or 1) * client_settings.min_save_ratio))
    if len(graded_examples) < required_min:
        logger.saveToLog(
            f"[save_responses] Valid examples below threshold. valid={len(graded_examples)} required_min={required_min}. Aborting ingest.",
            "ERROR",
        )
        raise ValueError(
            f"Valid examples below threshold ({len(graded_examples)}/{amount}); need at least {required_min}"
        )
    payload = {
        "dataset_name": meta["name"],
        "dataset_description": meta["description"],
        "dataset_id": 0,
        "run_id": run_id,
        "category": graded_category,
        "model": model,
        "example": graded_examples,
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




