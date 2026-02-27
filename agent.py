import json
import requests
from dotenv import load_dotenv
import os
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI
import numpy as np
import asyncio
from generics import timer, TimedLabel
import httpx
import logger
import re
from routes import dataset

load_dotenv()
_SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
_API_KEY = os.getenv("OPENROUTER_API_KEY")
THRESHOLD = 0.8
DEFAULT_MODEL = "google/gemini-3-flash-preview"
MODEL_PRICING = {}
MIN_GRADING_SCORE = 8.0
MIN_RESPONSE_CHAR_LENGTH = 40
MAX_GRADING_JSON_RETRIES = 2
MAX_LOW_QUALITY_RETRIES = 1
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_API_KEY)


class AgentType(str, Enum):
    qa = "qa"
    instruction_following = "instruction_following"
    domain_specialist = "domain_specialist"
    style = "style"
    adversarial = "adversarial"
    conversation = "conversation"


def get_embedding(text: str):
    with timer(TimedLabel.EMBEDDING_CALL):
        response = client.embeddings.create(
            model="openai/text-embedding-3-small", input=text
        )
    return response.data[0].embedding


def load_prompt(name: str) -> str:
    try:
        with open(f"prompts/{name}_agent.txt", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as e:
        logger.saveToLog(
            f"Couldn't find instructions for {name}_agent in /prompts", "ERROR"
        )
        return  # in the future we could response with a fallback instruction,  but probably better to just let it error.  we could handle it more gracefully in the future though.  maybe v2


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
    if not pricing or pricing["prompt"] < 0:
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
        f"Successfully called OpenRouter API.  System Prompt: {system_prompt}, User Prompt: {user_prompt}, Model Used: {model}, Tokens Used: {response.usage.total_tokens}, Total Cost: {calculate_price(response.usage.prompt_tokens, response.usage.completion_tokens, model)}  \n Response: {response.choices[0].message.content}",
        "INFO",
    )
    return response.choices[0].message.content


def _strip_markdown_fences(text: str) -> str:
    content = text.strip()
    match = re.match(
        r"^```(?:json)?\s*(.*?)\s*```$", content, flags=re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return content


def _strip_common_artifacts(text: str) -> str:
    cleaned = text.lstrip("\ufeff").strip()
    # Remove common invisible chars that break parsing.
    cleaned = cleaned.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    cleaned = cleaned.replace("\ufeff", "")
    # Remove leading language labels like "json\n{...}".
    cleaned = re.sub(r"^\s*json\s*\n", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _extract_first_json_value(text: str) -> str:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text, idx=i)
            return text[i:end]
        except json.JSONDecodeError:
            continue
    return text


def _normalize_json_like_text(text: str) -> str:
    # Convert common smart-quote artifacts into plain quotes.
    normalized = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    # Remove JS-style comments.
    normalized = re.sub(r"/\*.*?\*/", "", normalized, flags=re.DOTALL)
    normalized = re.sub(r"^\s*//.*?$", "", normalized, flags=re.MULTILINE)
    # Remove trailing commas before object/array close.
    normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
    return normalized


def _parse_json_with_fallback(raw: str):
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(text: str):
        if text not in seen:
            seen.add(text)
            candidates.append(text)

    base = _strip_common_artifacts(raw)
    no_fence = _strip_markdown_fences(base)
    extracted = _extract_first_json_value(no_fence)
    normalized = _normalize_json_like_text(extracted)

    add_candidate(raw)
    add_candidate(base)
    add_candidate(no_fence)
    add_candidate(extracted)
    add_candidate(normalized)

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e

    if last_error:
        raise last_error
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise e


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "unknown"


def _write_grading_audit(payload: dict):
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    topic_slug = _slugify(payload.get("topic", "unknown"))
    out = logs_dir / f"grading_audit_{stamp}_{topic_slug}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.saveToLog(f"[run_grading_agent] Saved grading audit: {out}", "INFO")


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
    meta = await run_naming_agent(examples)
    if not meta:
        logger.saveToLog("[save_responses] Naming agent failed.. aborting", "ERROR")
        return
    
    graded = await run_grading_agent(
        topic=topic,
        model=model,
        examples=examples,
        agent_type=agent_type,
        source_material=source_material,
    )
    # Build ingest payload
    if not len(graded) > 0:
        logger.saveToLog("[save_responses] Received 0 valid examples back from grading agent... aborting", "ERROR")
        return
    payload = {
        "dataset_name": meta["name"],
        "dataset_description": meta["description"],
        "dataset_id": 0,
        "example": graded,
        "prompt": prompt,
    }

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
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    system = load_prompt(agent_type.value)
    prompt = build_prompt(agent_type, topic, amt, source_material)
    result = await run_agent_async(
        system_prompt=system,
        user_prompt=prompt,
        label=TimedLabel.CHAT_COMPLETION,
        model=model,
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
            system_prompt=load_prompt("naming"), user_prompt=naming_prompt
        )

        try:
            meta = _parse_json_with_fallback(naming_result)
            if meta == "" or not meta:
                return
        except json.JSONDecodeError as e:
            logger.saveToLog(
                f"[save_responses] Failed to parse naming_agent output: {e}, Naming Agent output was: {naming_result}",
                "ERROR",
            )
    return meta


async def run_grading_agent(
    examples: list[dict],
    model: str,
    topic: str,
    agent_type: AgentType,
    source_material: str | None = None,
):
    async def _grade_with_json_retries(candidate: dict, duplicate_instruction: int, length: int):
        retries = 0
        last_error = None
        while retries <= MAX_GRADING_JSON_RETRIES:
            retry_note = ""
            if retries > 0:
                retry_note = (
                    "\n\nPrevious output failed JSON parsing. "
                    "Return ONLY one valid raw JSON object. No markdown, no code fences."
                )
            grading_result = await run_agent_async(
                system_prompt=load_prompt("grading"),
                user_prompt=f"""
            USER PROMPT TEMPLATE
            Grade this single instruction/response pair using the rubric in your instructions.

            Metadata:
            - model: {model}
            - topic: {topic}

            Dataset-level checks:
            - exact_item_count_20: {length}
            - duplicate_instruction: {duplicate_instruction}

            Candidate item JSON:
            {json.dumps(candidate)}

            Return exactly one JSON object using the required schema.
            No markdown. No extra text.{retry_note}
            """,
            )
            try:
                return _parse_json_with_fallback(grading_result), retries, None
            except json.JSONDecodeError as e:
                last_error = e
                retries += 1
        return None, retries - 1, f"grading_json_parse_error: {last_error}"

    async def _regenerate_low_quality_example(candidate: dict, reason: str):
        retry_prompt = f"""
Improve this single training example for topic "{topic}".
Return ONLY a JSON array with exactly 1 object:
[{{"instruction":"...","response":"..."}}]

Requirements:
- Keep the instruction-response pair aligned to the topic.
- Ensure response is detailed, accurate, and at least {MIN_RESPONSE_CHAR_LENGTH} characters.
- Response must be a JSON string value.
- Do not include markdown/code fences.

Example to improve:
{json.dumps(candidate)}

Failure reason:
{reason}
"""
        system = load_prompt(agent_type.value)
        result = await run_agent_async(
            system_prompt=system,
            user_prompt=retry_prompt,
            model=model,
        )
        repaired = _parse_json_with_fallback(result)
        if not isinstance(repaired, list) or len(repaired) < 1:
            raise ValueError("Low-quality retry did not return a JSON array with one item")
        item = repaired[0]
        if not isinstance(item, dict) or "instruction" not in item or "response" not in item:
            raise ValueError("Low-quality retry returned invalid item shape")
        return item

    valid = []
    invalid = []
    audit_rows = []
    failure_counts = {
        "grading_json_parse_error": 0,
        "low_score": 0,
        "short_response": 0,
        "exception": 0,
        "accepted": 0,
    }
    with timer(label=TimedLabel.GRADING_CALL):
        if len(examples) > 0:
            length = 1 if len(examples) == 20 else 0
            embeddings = await asyncio.gather(
                *[asyncio.to_thread(get_embedding, ex["instruction"]) for ex in examples]
            )
            for idx, ex in enumerate(examples):
                candidate = ex
                prior = embeddings[:idx]
                is_dup = int(is_duplicate(new_embedding=embeddings[idx], existing_embeddings=prior))
                low_quality_retries = 0

                try:
                    while True:
                        loaded, json_retries, parse_failure = await _grade_with_json_retries(
                            candidate=candidate,
                            duplicate_instruction=is_dup,
                            length=length,
                        )
                        if parse_failure:
                            failure_counts["grading_json_parse_error"] += 1
                            invalid.append(candidate)
                            audit_rows.append(
                                {
                                    "idx": idx,
                                    "accepted": False,
                                    "score": 0.0,
                                    "reason": "grading_json_parse_error",
                                    "json_retries": json_retries,
                                    "low_quality_retries": low_quality_retries,
                                    "instruction": candidate.get("instruction", ""),
                                }
                            )
                            logger.saveToLog(
                                f"[run_grading_agent] Could not parse grading output for idx={idx}: {parse_failure}",
                                "ERROR",
                            )
                            break

                        score = float(loaded.get("normalized_score_0_10", 0))
                        notes = str(loaded.get("notes", ""))
                        response_text = str(candidate.get("response", "")).strip()
                        too_short = len(response_text) < MIN_RESPONSE_CHAR_LENGTH
                        low_score = score < MIN_GRADING_SCORE

                        if too_short or low_score:
                            reasons = []
                            if too_short:
                                reasons.append("short_response")
                            if low_score:
                                reasons.append("low_score")
                            reason_text = ",".join(reasons)

                            if low_quality_retries < MAX_LOW_QUALITY_RETRIES:
                                low_quality_retries += 1
                                try:
                                    candidate = await _regenerate_low_quality_example(
                                        candidate, f"{reason_text}; notes={notes}"
                                    )
                                    continue
                                except Exception as regen_e:
                                    logger.saveToLog(
                                        f"[run_grading_agent] Low-quality retry failed for idx={idx}: {regen_e}",
                                        "ERROR",
                                    )

                            if too_short:
                                failure_counts["short_response"] += 1
                            if low_score:
                                failure_counts["low_score"] += 1
                            invalid.append(candidate)
                            audit_rows.append(
                                {
                                    "idx": idx,
                                    "accepted": False,
                                    "score": score,
                                    "reason": reason_text,
                                    "notes": notes,
                                    "json_retries": json_retries,
                                    "low_quality_retries": low_quality_retries,
                                    "instruction": candidate.get("instruction", ""),
                                }
                            )
                            logger.saveToLog(
                                message=f"Grading agent rejected example {candidate} with reason: {notes}"
                            )
                            break

                        valid.append(candidate)
                        failure_counts["accepted"] += 1
                        audit_rows.append(
                            {
                                "idx": idx,
                                "accepted": True,
                                "score": score,
                                "reason": "",
                                "notes": notes,
                                "json_retries": json_retries,
                                "low_quality_retries": low_quality_retries,
                                "instruction": candidate.get("instruction", ""),
                            }
                        )
                        break
                except Exception as e:
                    failure_counts["exception"] += 1
                    invalid.append(candidate)
                    audit_rows.append(
                        {
                            "idx": idx,
                            "accepted": False,
                            "score": 0.0,
                            "reason": "exception",
                            "notes": str(e),
                            "json_retries": 0,
                            "low_quality_retries": low_quality_retries,
                            "instruction": candidate.get("instruction", ""),
                        }
                    )
                    logger.saveToLog(
                        f"An error occurred when trying to grade example: {candidate}. See exception: {e}",
                        "ERROR",
                    )
                    continue

    _write_grading_audit(
        {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "model": model,
            "input_count": len(examples),
            "accepted_count": len(valid),
            "rejected_count": len(invalid),
            "failure_counts": failure_counts,
            "rows": audit_rows,
        }
    )
    return valid

load_pricing()
