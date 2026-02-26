import json
import requests
from dotenv import load_dotenv
import os
from enum import Enum
from openai import OpenAI
import numpy as np
import asyncio
from generics import timer, TimedLabel
import httpx
import logger

load_dotenv()
_SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
_API_KEY = os.getenv("OPENROUTER_API_KEY")
THRESHOLD = 0.8
DEFAULT_MODEL = "minimax/minimax-m2.5"
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_API_KEY)


class AgentType(str, Enum):
    qa = "qa"
    instruction_following = "instruction_following"
    domain_specialist = "domain_specialist"
    style = "style"
    adversarial = "adversarial"
    conversation = "conversation"


def get_embedding(text: str):
    response = client.embeddings.create(
        model="openai/text-embedding-3-small", input=text
    )
    return response.data[0].embedding


def load_prompt(name: str) -> str:
    with open(f"prompts/{name}_agent.txt", "r", encoding="utf-8") as file:
        return file.read()


def cosine_similarity(a: list[float], b: list[float]):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def is_duplicate(
    new_embedding: list[float],
    existing_embeddings: list[list[float]],
    threshold: float = THRESHOLD,
):
    for existing in existing_embeddings:
        if cosine_similarity(new_embedding, existing) >= threshold:
            return True
    return False


async def run_agent_async(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    label: TimedLabel | None = None,
):
    if label:
        with timer(label):
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
    else:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    logger.saveToLog(f"Successfully called OpenRouter API.  System Prompt: {system_prompt}, User Prompt: {user_prompt}, Model Used: {model}, Tokens Used: {response.usage.total_tokens},  \n Response: {response.choices[0].message.content}")
    return response.choices[0].message.content


async def save_responses(*, prompt: str, examples: list[dict]):
    meta = await run_naming_agent(examples)

    if not meta:
        print("[save_responses] Naming agent failed.. aborting")
        return
    # Build ingest payload
    payload = {
        "dataset_name": meta["name"],
        "dataset_description": meta["description"],
        "dataset_id": 0,
        "example": examples,
        "prompt": prompt,
    }
    print(payload)

    # POST to /ingest
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{_SERVER_URL}/dataset/ingest", json=payload, timeout=30
            )
            res.raise_for_status()
            result = res.json()
            print(f"[save_responses] Ingest response: {result}")
            return result
    except httpx.HTTPStatusError as e:
        print(f"[save_responses] Failed to POST to /ingest: {e}")


async def generate_dataset(
    topic: str,
    amt: int,
    source_material: str | None = None,
    agent_type: AgentType = AgentType.qa,
) -> list[dict]:
    system = load_prompt(agent_type.value)
    prompt = build_prompt(agent_type, topic, amt, source_material)
    result = await run_agent_async(
        system_prompt=system, user_prompt=prompt, label=TimedLabel.CHAT_COMPLETION
    )

    try:
        examples = json.loads(result)
    except json.JSONDecodeError:
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
    return valid, prompt


def build_prompt(
    agent: AgentType, topic: str, amt: int, source_material: str | None = None
) -> str:
    base = f"Generate {amt} diverse examples about {topic}, make sure to follow your instructions about formatting and return NOTHING besides JSON. Returning, or explaining your answers would be considered a FAILURE. JSON ARRAY ONLY"
    if agent == AgentType.domain_specialist:
        return f"""
Using ONLY the following source material, generate {amt} training examples. 

SOURCE MATERIAL:
{source_material}

Ensure you follow your instructions about formatting your response,  and ONLY return only JSON
"""
    return base


async def run_naming_agent(examples: list[dict]):
    # Ask naming_agent for name + description
    naming_prompt = f"""
    Generate a dataset name and description for the following examples.
    Return STRICT JSON: {{"name": "...", "description": "..."}}

    Examples JSON:
    {json.dumps(examples)}
    """
    naming_result = await run_agent_async(
        system_prompt=load_prompt("naming"), user_prompt=naming_prompt
    )

    try:
        meta = json.loads(naming_result)
        return meta
    except json.JSONDecodeError as e:
        print(f"[save_responses] Failed to parse naming_agent output: {e}")
        print("naming_agent output was:", naming_result)
        return
