import json

import logger
from generics import TimedLabel

from .llm import run_agent_async
from .parsing import parse_json_with_fallback
from .prompts import build_prompt, load_prompt
from .settings import DEFAULT_MODEL, MAX_GENERATION_RETRIES
from .types import AgentType


async def generate_dataset(
    topic: str,
    amt: int,
    source_material: str | None = None,
    agent_type: AgentType = AgentType.qa,
    model: str = DEFAULT_MODEL,
    run_id: str | None = None,
    dataset_key: str | None = None,
    seed: int | None = None,
) -> list[dict]:
    model = model or DEFAULT_MODEL
    system = load_prompt(agent_type.value)
    prompt = build_prompt(agent_type, topic, amt, source_material, seed=seed)
    parse_errors: list[str] = []
    for attempt in range(MAX_GENERATION_RETRIES + 1):
        retry_suffix = ""
        if attempt > 0:
            retry_suffix = (
                "\n\nYour previous output was invalid JSON for this task. "
                "Return ONLY one valid JSON array of objects in shape "
                '[{"instruction":"...","response":"..."}]. '
                "No prose, no markdown fences, no truncation."
            )
        current_prompt = f"{prompt}{retry_suffix}"
        result = await run_agent_async(
            system_prompt=system,
            user_prompt=current_prompt,
            label=TimedLabel.CHAT_COMPLETION,
            model=model,
            run_id=run_id,
            dataset_key=dataset_key,
            topic=topic,
            stage="generation",
        )

        try:
            examples = parse_json_with_fallback(result, require_top_level_list=True)
        except json.JSONDecodeError as e:
            parse_errors.append(str(e))
            logger.saveToLog(
                f"[generate_dataset] Parse failure attempt={attempt + 1}/{MAX_GENERATION_RETRIES + 1}: {e}",
                "WARNING",
            )
            continue

        valid = [
            ex
            for ex in examples
            if isinstance(ex, dict)
            and isinstance(ex.get("instruction"), str)
            and isinstance(ex.get("response"), str)
            and len(ex["instruction"]) > 2
            and len(ex["response"]) > 2
        ]
        if valid:
            return valid, prompt

        logger.saveToLog(
            f"[generate_dataset] Parsed array but no valid items attempt={attempt + 1}/{MAX_GENERATION_RETRIES + 1}",
            "WARNING",
        )

    logger.saveToLog("[generate_dataset] No valid examples after parsing/filtering", "ERROR")
    if parse_errors:
        raise ValueError(f"Agent returned malformed JSON: {parse_errors[-1]}")
    raise ValueError("No valid examples generated")
