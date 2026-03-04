import json

import app.core.logger as logger
from app.core.generics import TimedLabel, saveScore, timer

from .llm import run_agent_async
from .parsing import parse_json_with_fallback
from .prompts import load_prompt
from .settings import get_client_settings
from .types import AgentType

CATEGORY_TAXONOMY = (
    "technology_software",
    "science_math",
    "business_finance_law",
    "health_medicine",
    "education_tutoring",
    "creative_media",
    "communication_support",
    "humanities_social_science",
    "lifestyle_practical",
    "general_reasoning",
)


def _normalize_category(raw_category: object) -> str | None:
    if not isinstance(raw_category, str):
        return None
    cleaned = raw_category.strip().lower()
    return cleaned if cleaned in CATEGORY_TAXONOMY else None


def _empty_grading_result(notes: str) -> dict:
    return {
        "accepted_examples": [],
        "dataset_score": 0.0,
        "notes": notes,
        "category": None,
    }


def _grading_score_metadata(
    *,
    topic: str,
    model: str,
    input_count: int,
    accepted_count: int,
    rejected_count: int,
    notes: str,
    category: str | None,
    run_id: str | None,
    dataset_key: str | None,
) -> dict:
    grader_settings = get_client_settings()
    return {
        "topic": topic,
        "model": model,
        "grader_model": grader_settings.grading_model,
        "category": category,
        "input_count": input_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "notes": notes,
        "run_id": run_id,
        "dataset_key": dataset_key,
    }


def _row_map(rows: object) -> dict[int, dict]:
    if not isinstance(rows, list):
        return {}
    out: dict[int, dict] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("idx"), int):
            out[row["idx"]] = row
    return out


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _evaluate_loaded_rows(batch_examples: list[dict], loaded: dict) -> tuple[list[dict], list[dict], float, str, str | None]:
    grading_settings = get_client_settings()
    row_map = _row_map(loaded.get("rows", []))

    accepted_local: list[dict] = []
    rejected_local: list[dict] = []
    for idx, ex in enumerate(batch_examples):
        row = row_map.get(idx, {})
        row_score = _safe_float(row.get("score_0_10", 0))
        accept_flag = int(row.get("accept", 0)) if isinstance(row, dict) else 0
        reason = str(row.get("reason", "")) if isinstance(row, dict) else "missing_row"
        short_response = len(str(ex.get("response", "")).strip()) < grading_settings.min_response_char_length
        if accept_flag == 1 and row_score >= grading_settings.min_grading_score and not short_response:
            accepted_local.append(ex)
        else:
            rejected_local.append(
                {
                    "idx": idx,
                    "example": ex,
                    "score": row_score,
                    "reason": reason if reason else ("short_response" if short_response else "low_score"),
                }
            )

    dataset_score = _safe_float(loaded.get("dataset_score_0_10", 0))
    notes = str(loaded.get("notes", ""))
    category = _normalize_category(loaded.get("category"))
    return accepted_local, rejected_local, dataset_score, notes, category


def _build_rejected_payload(rejected: list[dict]) -> list[dict]:
    return [
        {
            "original_idx": row["idx"],
            "instruction": row["example"].get("instruction", ""),
            "response": row["example"].get("response", ""),
            "failure_reason": row["reason"],
            "failure_score": row["score"],
        }
        for row in rejected
    ]


# Grades a batch end-to-end, including one regeneration pass for rejected rows.
# It is responsible for parser retries, row-level acceptance decisions,
# score persistence, and returning the filtered examples plus dataset metadata.
async def run_grading_agent(
    examples: list[dict],
    model: str,
    topic: str,
    agent_type: AgentType,
    source_material: str | None = None,
    run_id: str | None = None,
    dataset_key: str | None = None,
):
    grading_settings = get_client_settings()
    model = model or grading_settings.default_model

    async def _grade_batch_with_json_retries(
        batch_examples: list[dict],
        *,
        stage: str,
        tag: str,
    ):
        retries = 0
        last_error = None
        while retries <= grading_settings.max_grading_json_retries:
            retry_note = ""
            if retries > 0:
                retry_note = (
                    "\n\nPrevious output failed JSON parsing. "
                    "Return ONLY one valid raw JSON object. No markdown/code fences."
                )
            user_prompt = f"""
Grade this ENTIRE dataset in one pass and return row-level judgments.

Metadata:
- model: {model}
- topic: {topic}
- pass_tag: {tag}

Input dataset JSON:
{json.dumps(batch_examples)}

Return ONLY this JSON object schema:
{{
  "model": "{model}",
  "topic": "{topic}",
  "category": "{CATEGORY_TAXONOMY[0]}",
  "dataset_score_0_10": 0.0,
  "notes": "string",
  "rows": [
    {{"idx": 0, "score_0_10": 0.0, "accept": 0, "reason": "string"}}
  ]
}}

Rules:
- rows length must equal input dataset length.
- idx must be 0-based and align to input order.
- category must be exactly one of: {", ".join(CATEGORY_TAXONOMY)}.
- accept = 1 only when score_0_10 >= {grading_settings.min_grading_score}.
- No markdown. No extra keys outside schema.{retry_note}
"""
            result = await run_agent_async(
                system_prompt=load_prompt("grading"),
                user_prompt=user_prompt,
                model=grading_settings.grading_model,
                label=TimedLabel.GRADING_CALL,
                run_id=run_id,
                dataset_key=dataset_key,
                topic=topic,
                stage=stage,
            )
            try:
                loaded = parse_json_with_fallback(result)
                return loaded, retries, None
            except json.JSONDecodeError as e:
                last_error = e
                retries += 1
        return None, retries - 1, f"grading_batch_json_parse_error: {last_error}"

    async def _regenerate_rejected_batch(rejected_rows: list[dict]) -> dict[int, dict]:
        # Returns mapping {original_idx: regenerated_example}
        prompt = f"""
You will receive rejected training examples for topic "{topic}".
Regenerate ONLY those examples and improve quality.

Return ONLY a JSON array with this exact shape:
[{{"original_idx": 3, "instruction": "...", "response": "..."}}]

Rules:
- Return exactly {len(rejected_rows)} items.
- Keep each item aligned to its original_idx.
- response must be a JSON string value.
- No markdown/code fences.

Rejected examples:
{json.dumps(rejected_rows)}
"""
        result = await run_agent_async(
            system_prompt=load_prompt(agent_type.value),
            user_prompt=prompt,
            model=model,
            run_id=run_id,
            dataset_key=dataset_key,
            topic=topic,
            stage="regeneration_batch",
        )
        parsed = parse_json_with_fallback(result)
        if not isinstance(parsed, list):
            raise ValueError("Regeneration batch did not return an array")
        out: dict[int, dict] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = item.get("original_idx")
            if not isinstance(idx, int):
                continue
            if "instruction" not in item or "response" not in item:
                continue
            out[idx] = {
                "instruction": str(item["instruction"]),
                "response": str(item["response"]),
            }
        return out

    if not examples:
        logger.saveToLog("[run_grading_agent] No examples passed for grading", "ERROR")
        return _empty_grading_result("No examples passed for grading")

    with timer(label=TimedLabel.GRADING_CALL):
        loaded, json_retries, parse_failure = await _grade_batch_with_json_retries(
            examples, stage="grading_batch", tag="initial"
        )
    if parse_failure or not isinstance(loaded, dict):
        logger.saveToLog(
            f"[run_grading_agent] Initial batch grading parse failed after retries={json_retries}: {parse_failure}",
            "ERROR",
        )
        saveScore(
            "grading_dataset_score",
            0.0,
            metadata=_grading_score_metadata(
                topic=topic,
                model=model,
                input_count=len(examples),
                accepted_count=0,
                rejected_count=len(examples),
                notes=str(parse_failure),
                category=None,
                run_id=run_id,
                dataset_key=dataset_key,
            ),
        )
        return _empty_grading_result(str(parse_failure))

    accepted, rejected, dataset_score, dataset_notes, dataset_category = _evaluate_loaded_rows(examples, loaded)

    # One regeneration pass for rejected items, then one re-grade pass.
    if rejected:
        rejected_payload = _build_rejected_payload(rejected)
        try:
            regenerated_map = await _regenerate_rejected_batch(rejected_payload)
            regen_examples = []
            for r in rejected:
                idx = r["idx"]
                if idx in regenerated_map:
                    regen_examples.append(regenerated_map[idx])

            if regen_examples:
                loaded_regen, retries_regen, parse_fail_regen = await _grade_batch_with_json_retries(
                    regen_examples, stage="grading_regeneration_batch", tag="regenerated"
                )
                if not parse_fail_regen and isinstance(loaded_regen, dict):
                    regen_accepted, _, _, _, _ = _evaluate_loaded_rows(regen_examples, loaded_regen)
                    # Map back accepted regenerated examples to original positions.
                    for ex in regen_examples:
                        if ex in regen_accepted:
                            accepted.append(ex)
                else:
                    logger.saveToLog(
                        f"[run_grading_agent] Regenerated batch grading parse failed retries={retries_regen}: {parse_fail_regen}",
                        "ERROR",
                    )
        except Exception as e:
            logger.saveToLog(f"[run_grading_agent] Regeneration batch failed: {e}", "ERROR")

    rejected_count = max(0, len(examples) - len(accepted))
    saveScore(
        "grading_dataset_score",
        dataset_score,
        metadata=_grading_score_metadata(
            topic=topic,
            model=model,
            input_count=len(examples),
            accepted_count=len(accepted),
            rejected_count=rejected_count,
            notes=dataset_notes,
            category=dataset_category,
            run_id=run_id,
            dataset_key=dataset_key,
        ),
    )
    logger.saveToLog(
        (
            f"[run_grading_agent] Batch summary generator_model={model} grader_model={grading_settings.grading_model} "
            f"topic={topic} category={dataset_category} input={len(examples)} accepted={len(accepted)} "
            f"rejected={rejected_count} dataset_score={dataset_score}"
        ),
        "INFO",
    )
    if not accepted:
        logger.saveToLog("[run_grading_agent] No examples passed grading", "ERROR")
    return {
        "accepted_examples": accepted,
        "dataset_score": dataset_score,
        "notes": dataset_notes,
        "category": dataset_category,
    }




