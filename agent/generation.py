import json

import logger
from generics import TimedLabel

from .llm import run_agent_async
from .parsing import parse_json_with_fallback
from .prompts import build_prompt, load_prompt
from .settings import DEFAULT_MODEL
from .types import AgentType


async def generate_dataset(
    topic: str,
    amt: int,
    source_material: str | None = None,
    agent_type: AgentType = AgentType.qa,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    model = model or DEFAULT_MODEL
    system = load_prompt(agent_type.value)
    prompt = build_prompt(agent_type, topic, amt, source_material)
    result = await run_agent_async(
        system_prompt=system,
        user_prompt=prompt,
        label=TimedLabel.CHAT_COMPLETION,
        model=model,
    )

    try:
        examples = parse_json_with_fallback(result)
    except json.JSONDecodeError:
        logger.saveToLog("Agent returned malformed JSON", "WARNING")
        raise ValueError("Agent returned malformed JSON")

    valid = [
        ex
        for ex in examples
        if isinstance(ex, dict)
        and "instruction" in ex
        and "response" in ex
        and len(ex["instruction"]) > 2
        and len(ex["response"]) > 2
    ]
    if not valid:
        logger.saveToLog("[generate_dataset] No valid examples after parsing/filtering", "ERROR")
        raise ValueError("No valid examples generated")
    return valid, prompt
