import json

import app.core.logger as logger
from app.core.generics import TimedLabel

from .llm import run_agent_async
from .parsing import parse_json_with_fallback
from .prompts import build_prompt, load_prompt
from .settings import get_client_settings
from .types import AgentType


def _normalize_conversation_example(example: dict) -> dict | None:
    instruction = str(example.get("instruction", "")).strip()
    response = str(example.get("response", "")).strip()
    if not instruction or not response:
        return None

    if response.lower().startswith("assistant:"):
        response = response.split(":", 1)[1].strip()
        if not response:
            return None

    lines = [line.strip() for line in instruction.splitlines() if line.strip()]
    if len(lines) < 1:
        return None

    expected_role = "user"
    for line in lines:
        if line.startswith("User:"):
            role = "user"
            content = line[len("User:") :].strip()
        elif line.startswith("Assistant:"):
            role = "assistant"
            content = line[len("Assistant:") :].strip()
        else:
            return None

        if not content or role != expected_role:
            return None
        expected_role = "assistant" if role == "user" else "user"

    last_line = lines[-1]
    if not last_line.startswith("User:"):
        return None

    return {
        "instruction": "\n".join(lines),
        "response": response,
    }


def _normalize_generated_example(example: dict, *, agent_type: AgentType) -> dict | None:
    if not isinstance(example, dict):
        return None
    if agent_type == AgentType.conversation:
        return _normalize_conversation_example(example)

    instruction = example.get("instruction")
    response = example.get("response")
    if not isinstance(instruction, str) or not isinstance(response, str):
        return None
    cleaned_instruction = instruction.strip()
    cleaned_response = response.strip()
    if len(cleaned_instruction) <= 2 or len(cleaned_response) <= 2:
        return None
    return {
        "instruction": cleaned_instruction,
        "response": cleaned_response,
    }


async def generate_dataset(
    topic: str,
    amt: int,
    source_material: str | None = None,
    source_material_mode: str = "content_and_style",
    conversation_length_mode: str = "varied",
    agent_type: AgentType = AgentType.qa,
    model: str | None = None,
    run_id: str | None = None,
    dataset_key: str | None = None,
    seed: int | None = None,
) -> list[dict]:
    client_settings = get_client_settings()
    model = model or client_settings.default_model
    max_generation_retries = client_settings.max_generation_retries
    system = load_prompt(agent_type.value)
    prompt = build_prompt(
        agent_type,
        topic,
        amt,
        source_material,
        source_material_mode=source_material_mode,
        conversation_length_mode=conversation_length_mode,
        seed=seed,
    )
    parse_errors: list[str] = []
    for attempt in range(max_generation_retries + 1):
        retry_suffix = ""
        if attempt > 0:
            retry_suffix = (
                "\n\nYour previous output was invalid JSON for this task. "
                "Return ONLY one valid JSON array of objects in shape "
                '[{"instruction":"...","response":"..."}]. '
                "No prose, no markdown fences, no truncation."
            )
            if agent_type == AgentType.conversation:
                retry_suffix += (
                    " For conversation examples, the instruction must end on a User: line and the final assistant reply"
                    " must appear only in response, not inside instruction."
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
                f"[generate_dataset] Parse failure attempt={attempt + 1}/{max_generation_retries + 1}: {e}",
                "WARNING",
            )
            continue

        valid = [
            normalized
            for ex in examples
            if (normalized := _normalize_generated_example(ex, agent_type=agent_type)) is not None
        ]
        if valid:
            return valid, prompt

        logger.saveToLog(
            f"[generate_dataset] Parsed array but no valid items attempt={attempt + 1}/{max_generation_retries + 1}",
            "WARNING",
        )

    logger.saveToLog("[generate_dataset] No valid examples after parsing/filtering", "ERROR")
    if parse_errors:
        raise ValueError(f"Agent returned malformed JSON: {parse_errors[-1]}")
    raise ValueError("No valid examples generated")



