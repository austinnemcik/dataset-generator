import json
import re
from datetime import datetime, timezone
from pathlib import Path

import asyncio
import logger
from generics import TimedLabel, saveScore, timer

from .embeddings import get_embedding, is_duplicate
from .llm import run_agent_async
from .parsing import parse_json_with_fallback
from .prompts import load_prompt
from .settings import (
    DEFAULT_MODEL,
    MAX_GRADING_JSON_RETRIES,
    MAX_LOW_QUALITY_RETRIES,
    MIN_GRADING_SCORE,
    MIN_RESPONSE_CHAR_LENGTH,
)
from .types import AgentType


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


async def run_grading_agent(
    examples: list[dict],
    model: str,
    topic: str,
    agent_type: AgentType,
    source_material: str | None = None,
):
    model = model or DEFAULT_MODEL

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
                return parse_json_with_fallback(grading_result), retries, None
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
        repaired = parse_json_with_fallback(result)
        if not isinstance(repaired, list) or len(repaired) < 1:
            raise ValueError("Low-quality retry did not return a JSON array with one item")
        item = repaired[0]
        if not isinstance(item, dict) or "instruction" not in item or "response" not in item:
            raise ValueError("Low-quality retry returned invalid item shape")
        return item

    async def _grade_entire_dataset(dataset_examples: list[dict]) -> tuple[float, str]:
        if not dataset_examples:
            return 0.0, "No accepted examples to grade at dataset level"
        result = await run_agent_async(
            system_prompt=load_prompt("grading"),
            user_prompt=f"""
Grade the ENTIRE dataset as a whole (not a single row) using the rubric.

Metadata:
- model: {model}
- topic: {topic}

Dataset-level checks:
- exact_item_count_20: {1 if len(dataset_examples) == 20 else 0}
- duplicate_instruction: 0

Candidate item JSON:
{json.dumps(dataset_examples)}

Return exactly one JSON object using the required schema.
No markdown. No extra text.
""",
        )
        loaded = parse_json_with_fallback(result)
        score = float(loaded.get("normalized_score_0_10", 0))
        notes = str(loaded.get("notes", ""))
        return score, notes

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
                is_dup = int(
                    is_duplicate(new_embedding=embeddings[idx], existing_embeddings=prior)
                )
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
    dataset_score = 0.0
    dataset_notes = ""
    try:
        dataset_score, dataset_notes = await _grade_entire_dataset(valid)
    except Exception as e:
        logger.saveToLog(f"[run_grading_agent] Dataset-level grading failed: {e}", "ERROR")
        dataset_score = 0.0
        dataset_notes = f"dataset_level_grading_failed: {e}"

    saveScore(
        "grading_dataset_score",
        dataset_score,
        metadata={
            "topic": topic,
            "model": model,
            "input_count": len(examples),
            "accepted_count": len(valid),
            "rejected_count": len(invalid),
            "notes": dataset_notes,
        },
    )
    if not valid:
        logger.saveToLog("[run_grading_agent] No examples passed grading", "ERROR")
    return valid
