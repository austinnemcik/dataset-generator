from .types import AgentType


def load_prompt(name: str) -> str:
    try:
        with open(f"prompts/{name}_agent.txt", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing prompt file: prompts/{name}_agent.txt")


def build_prompt(
    agent: AgentType, topic: str, amt: int, source_material: str | None = None
) -> str:
    base = f"""Generate {amt} diverse examples about {topic}, make sure to follow your instructions about formatting and return NOTHING besides JSON.
Returning, or explaining your answers would be considered a FAILURE.
JSON ARRAY ONLY.
Do NOT wrap your output in markdown code fences.
Do NOT include ```json, ``` or any backticks."""
    if agent == AgentType.domain_specialist:
        return f"""
Using ONLY the following source material, generate {amt} training examples.

SOURCE MATERIAL:
{source_material}

Ensure you follow your instructions about formatting your response, and ONLY return JSON.
Do NOT wrap your output in markdown code fences.
Do NOT include ```json, ``` or any backticks.
"""
    return base

