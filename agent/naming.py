import json

import app.core.logger as logger
from app.core.generics import TimedLabel, timer

from .llm import run_agent_async
from .parsing import parse_json_with_fallback
from .prompts import load_prompt
from .settings import get_client_settings


def _parse_naming_metadata(raw_result: str) -> dict[str, str] | None:
    meta = parse_json_with_fallback(raw_result)
    if not isinstance(meta, dict):
        return None

    name = str(meta.get("name", "")).strip()
    description = str(meta.get("description", "")).strip()
    if not name or not description:
        return None
    return {"name": name, "description": description}


def _normalize_topic_list(parsed: object, *, topic_count: int, fallback_topic: str) -> list[str]:
    topics: list[str] = []
    if isinstance(parsed, dict):
        raw = parsed.get("topics", [])
        if isinstance(raw, list):
            topics = [str(item).strip() for item in raw if str(item).strip()]
    elif isinstance(parsed, list):
        topics = [str(item).strip() for item in parsed if str(item).strip()]

    deduped: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        key = topic.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(topic)
        if len(deduped) >= topic_count:
            break

    if not deduped:
        return [fallback_topic]
    while len(deduped) < topic_count:
        deduped.append(fallback_topic)
    return deduped[:topic_count]


async def run_naming_agent(
    examples: list[dict],
    *,
    run_id: str | None = None,
    dataset_key: str | None = None,
    topic: str | None = None,
    model: str | None = None,
):
    client_settings = get_client_settings()
    naming_model = client_settings.naming_model
    max_naming_json_retries = client_settings.max_naming_json_retries
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
        for attempt in range(max_naming_json_retries + 1):
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
                meta = _parse_naming_metadata(naming_result)
            except json.JSONDecodeError as e:
                parse_errors.append(str(e))
                logger.saveToLog(
                    f"[run_naming_agent] Parse failure attempt={attempt + 1}/{max_naming_json_retries + 1}: {e}",
                    "WARNING",
                )
                continue

            if meta is None:
                logger.saveToLog(
                    f"[run_naming_agent] Missing required metadata attempt={attempt + 1}/{max_naming_json_retries + 1}",
                    "WARNING",
                )
                continue
            return meta

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
    return _normalize_topic_list(parsed, topic_count=topic_count, fallback_topic=topic)



