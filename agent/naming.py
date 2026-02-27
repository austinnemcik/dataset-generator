import json

from generics import TimedLabel, timer

from .llm import run_agent_async
from .parsing import parse_json_with_fallback
from .prompts import load_prompt


async def run_naming_agent(
    examples: list[dict],
    *,
    run_id: str | None = None,
    dataset_key: str | None = None,
    topic: str | None = None,
    model: str | None = None,
):
    meta = None
    with timer(label=TimedLabel.NAMING_CALL):
        naming_prompt = f"""
        Generate a dataset name and description for the following examples.
        Return STRICT JSON: {{"name": "...", "description": "..."}}
        Do NOT wrap your output in markdown code fences.
        Do NOT include ```json, ``` or any backticks.

        Examples JSON:
        {json.dumps(examples)}
        """
        naming_result = await run_agent_async(
            system_prompt=load_prompt("naming"),
            user_prompt=naming_prompt,
            model=model,
            run_id=run_id,
            dataset_key=dataset_key,
            topic=topic,
            stage="naming",
        )

        try:
            meta = parse_json_with_fallback(naming_result)
            if meta == "" or not meta:
                raise ValueError("Naming metadata is empty")
        except json.JSONDecodeError as e:
            raise ValueError("Naming agent returned invalid JSON") from e
    return meta
