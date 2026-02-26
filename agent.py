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
import re

load_dotenv()
_SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
_API_KEY = os.getenv("OPENROUTER_API_KEY")
THRESHOLD = 0.8
DEFAULT_MODEL = "google/gemini-3-flash-preview"
MODEL_PRICING = {}
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
    try:
        with open(f"prompts/{name}_agent.txt", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as e:
        logger.saveToLog(f"Couldn't find instructions for {name}_agent in /prompts", "ERROR")
        return # in the future we could response with a fallback instruction,  but probably better to just let it error.  we could handle it more gracefully in the future though.  maybe v2


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
            logger.saveToLog("Discarding duplicate example", "INFO")
            return True

    return False


def load_pricing():
    try:
        res = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {_API_KEY}"},
        )
        res.raise_for_status()
    except requests.RequestException as e:
        logger.saveToLog(f"Failed to reach OpenRouter pricing API: {e}", "ERROR")
    for model in res.json()["data"]:
        try:
            MODEL_PRICING[model["id"]] = {
                "prompt": float(model["pricing"]["prompt"]),
                "completion": float(model["pricing"]["completion"]),
            }
        except KeyError or ValueError:
            logger.saveToLog(f"Failed to get valid pricing data for {model}")
            continue


def calculate_price(input_tokens: int, output_tokens: int, model: str):
    pricing = MODEL_PRICING.get(model)
    if not pricing or pricing["prompt"]  < 0:
        logger.saveToLog(f"Pricing not found for model: {model}", "WARNING")
    input_price = pricing["prompt"] * input_tokens
    output_price = pricing["completion"] * output_tokens
    total_price = input_price + output_price
    return f"${total_price}", f"${input_price}", f"${output_price}"


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
    logger.saveToLog(
        f"Successfully called OpenRouter API.  System Prompt: {system_prompt}, User Prompt: {user_prompt}, Model Used: {model}, Tokens Used: {response.usage.total_tokens}, Total Cost: {calculate_price(response.usage.prompt_tokens, response.usage.completion_tokens, model)}  \n Response: {response.choices[0].message.content}", "INFO"
    )
    return response.choices[0].message.content


def _strip_markdown_fences(text: str) -> str:
    content = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", content, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content


def _parse_json_with_fallback(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = _strip_markdown_fences(raw)
        if cleaned != raw:
            return json.loads(cleaned)
        raise


async def save_responses(*, prompt: str, examples: list[dict], topic: str, model: str = DEFAULT_MODEL, amount: int):
    meta = await run_naming_agent(examples)
    if not meta:
        logger.saveToLog("[save_responses] Naming agent failed.. aborting", "ERROR")
        return
    # Build ingest payload
    payload = {
        "dataset_name": meta["name"],
        "dataset_description": meta["description"],
        "dataset_id": 0,
        "example": examples,
        "prompt": prompt,
    }
    grading = await run_grading_agent(topic=topic, model=model, examples=examples)
    # POST to /ingest
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{_SERVER_URL}/dataset/ingest", json=payload, timeout=30
            )
            res.raise_for_status()
            result = res.json()
            logger.saveToLog("[save_responses] Ingested response", "INFO")
            return result
    except httpx.HTTPStatusError as e:
        logger.saveToLog(f"[save_responses] Failed to POST to /ingest, {e}", "ERROR")


async def generate_dataset(
    topic: str,
    amt: int,
    source_material: str | None = None,
    agent_type: AgentType = AgentType.qa,
    model: str = DEFAULT_MODEL
) -> list[dict]:
    system = load_prompt(agent_type.value)
    prompt = build_prompt(agent_type, topic, amt, source_material)
    result = await run_agent_async(
        system_prompt=system, user_prompt=prompt, label=TimedLabel.CHAT_COMPLETION, model=model
    )

    try:
        examples = _parse_json_with_fallback(result)
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
    return valid, prompt


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


async def run_naming_agent(examples: list[dict]):
    # Ask naming_agent for name + description
    naming_prompt = f"""
    Generate a dataset name and description for the following examples.
    Return STRICT JSON: {{"name": "...", "description": "..."}}
    Do NOT wrap your output in markdown code fences.
    Do NOT include ```json, ``` or any backticks.

    Examples JSON:
    {json.dumps(examples)}
    """
    naming_result = await run_agent_async(
        system_prompt=load_prompt("naming"), user_prompt=naming_prompt
    )

    try:
        meta = _parse_json_with_fallback(naming_result)
        return meta
    except json.JSONDecodeError as e:
        logger.saveToLog(f"[save_responses] Failed to parse naming_agent output: {e}, Naming Agent output was: {naming_result}", "ERROR")
        return

async def run_grading_agent(examples: list[dict], model: str, topic: str ):
    length = 1 if len(examples) == 20 else 0
    valid = []
    embeddings = await asyncio.gather(*[                                                                                                                                                                   
      asyncio.to_thread(get_embedding, ex["instruction"])                                                                                                                                                
      for ex in examples                                                                                                                                                                                 
  ])  
    for idx, ex in enumerate(examples):
        current = embeddings[idx]
        prior = embeddings[:idx]
        is_dup = int(is_duplicate(new_embedding=current, existing_embeddings=prior))
        try: 
            grading_result = await run_agent_async(system_prompt=load_prompt("grading"), user_prompt=f"""
            USER PROMPT TEMPLATE                                                                                                                                                                                   
        Grade this single instruction/response pair using the rubric in your instructions.                                                                                                                                          
                                                                                                                                                                                                                
        Metadata:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        - model: {model}                                                                                                                                                                                     
        - topic: {topic}                                                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                
        Dataset-level checks:                                                                                                                                                                                  
        - exact_item_count_20: {length}                                                                                                                                                                      
        - duplicate_instruction: {is_dup}                                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                
        Candidate item JSON:                                                                                                                                                                                   
        {ex}                                                                                                                                                                                          
                                                                                                                                                                                                                
        Return exactly one JSON object using the required schema.                                                                                                                                              
        No markdown. No extra text.           
                                            """)
            loaded = json.loads(grading_result)
            score = loaded["normalized_score_0_10"]
            notes = loaded["notes"]
            if score >= 8.0:
                valid.append(ex)
        except: 
            continue
    return valid

load_pricing()
