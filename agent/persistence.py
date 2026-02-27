import httpx
import logger

from .grading import run_grading_agent
from .naming import run_naming_agent
from .settings import DEFAULT_MODEL, SERVER_URL
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
):
    model = model or DEFAULT_MODEL
    meta = await run_naming_agent(examples)
    if not meta:
        logger.saveToLog("[save_responses] Naming agent failed.. aborting", "ERROR")
        raise ValueError("Naming agent returned no metadata")

    graded = await run_grading_agent(
        topic=topic,
        model=model,
        examples=examples,
        agent_type=agent_type,
        source_material=source_material,
    )
    if not len(graded) > 0:
        logger.saveToLog(
            "[save_responses] Received 0 valid examples back from grading agent... aborting",
            "ERROR",
        )
        raise ValueError("No valid examples returned by grading")
    payload = {
        "dataset_name": meta["name"],
        "dataset_description": meta["description"],
        "dataset_id": 0,
        "example": graded,
        "prompt": prompt,
    }

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

