import json

import logger
from generics import TimedLabel, timer

from .llm import run_agent_async
from .parsing import parse_json_with_fallback
from .prompts import load_prompt
from .settings import MAX_NAMING_JSON_RETRIES, NAMING_MODEL


async def run_naming_agent(
    examples: list[dict],
    *,
    run_id: str | None = None,
    dataset_key: str | None = None,
    topic: str | None = None,
    model: str | None = None,
):
    naming_model = NAMING_MODEL
    parse_errors: list[str] = []

    naming_prompt = f"""
    Generate a dataset name and description for the following examples.
    Return STRICT JSON: {{"name": "...", "description": "..."}}
    Do NOT wrap your output in markdown code fences.
    Do NOT include ```json, ``` or any backticks.

    Examples JSON:
    {json.dumps(examples)}
    """

    with timer(label=TimedLabel.NAMING_CALL):
        for attempt in range(MAX_NAMING_JSON_RETRIES + 1):
            retry_suffix = ""
            if attempt > 0:
                retry_suffix = (
                    "\n\nYour previous output was invalid for this task. "
                    'Return ONLY one valid JSON object in shape {"name":"...","description":"..."}. '
                    "No prose, no markdown fences, no truncation. "
                    "Both values must be non-empty strings."
                )

            naming_result = await run_agent_async(
                system_prompt=load_prompt("naming"),
                user_prompt=f"{naming_prompt}{retry_suffix}",
                model=naming_model,
                run_id=run_id,
                dataset_key=dataset_key,
                topic=topic,
                stage="naming",
            )

            try:
                meta = parse_json_with_fallback(naming_result)
            except json.JSONDecodeError as e:
                parse_errors.append(str(e))
                logger.saveToLog(
                    f"[run_naming_agent] Parse failure attempt={attempt + 1}/{MAX_NAMING_JSON_RETRIES + 1}: {e}",
                    "WARNING",
                )
                continue

            if not isinstance(meta, dict):
                logger.saveToLog(
                    f"[run_naming_agent] Parsed payload is not an object attempt={attempt + 1}/{MAX_NAMING_JSON_RETRIES + 1}",
                    "WARNING",
                )
                continue

            name = str(meta.get("name", "")).strip()
            description = str(meta.get("description", "")).strip()
            if name and description:
                return {"name": name, "description": description}

            logger.saveToLog(
                f"[run_naming_agent] Missing required metadata attempt={attempt + 1}/{MAX_NAMING_JSON_RETRIES + 1}",
                "WARNING",
            )

    logger.saveToLog("[run_naming_agent] No valid metadata after retries", "ERROR")
    if parse_errors:
        raise ValueError(f"Naming agent returned malformed JSON: {parse_errors[-1]}")
    raise ValueError("Naming agent returned no metadata")


async def run_topic_variation_agent(
    *,
    topic: str,
    topic_count: int,
    existing_dataset_names: list[str],
    run_id: str | None = None,
    model: str | None = None,
) -> list[str]:
    if topic_count <= 1:
        return [topic]

    sample_names = existing_dataset_names[:200]
    planner_prompt = f"""
Generate {topic_count} distinct dataset topics derived from this base topic:
"{topic}"

Avoid repeating the intent of existing dataset names below.

Return STRICT JSON in this shape:
{{"topics": ["topic 1", "topic 2", "..."]}}

Rules:
- Exactly {topic_count} entries.
- Each topic must be short and specific.
- No markdown or code fences.

Existing dataset names JSON:
{json.dumps(sample_names)}
"""

    result = await run_agent_async(
        system_prompt=load_prompt("topic_planner"),
        user_prompt=planner_prompt,
        model=model,
        run_id=run_id,
        dataset_key=f"{run_id}:topic_planning" if run_id else None,
        topic=topic,
        stage="topic_planning",
    )

    parsed = parse_json_with_fallback(result)
    topics: list[str] = []
    if isinstance(parsed, dict):
        raw = parsed.get("topics", [])
        if isinstance(raw, list):
            topics = [str(item).strip() for item in raw if str(item).strip()]
    elif isinstance(parsed, list):
        topics = [str(item).strip() for item in parsed if str(item).strip()]

    # Normalize to exact length and keep base topic present as fallback.
    deduped: list[str] = []
    seen: set[str] = set()
    for t in topics:
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
        if len(deduped) >= topic_count:
            break

    if not deduped:
        return [topic]
    if len(deduped) < topic_count:
        while len(deduped) < topic_count:
            deduped.append(topic)
    return deduped[:topic_count]
