import json

import logger
from generics import TimedLabel, saveScore, timer

from .llm import run_agent_async
from .parsing import parse_json_with_fallback
from .prompts import load_prompt
from .settings import (
    DEFAULT_MODEL,
    GRADING_MODEL,
    MAX_GRADING_JSON_RETRIES,
    MIN_GRADING_SCORE,
    MIN_RESPONSE_CHAR_LENGTH,
)
from .types import AgentType


async def run_grading_agent(
    examples: list[dict],
    model: str,
    topic: str,
    agent_type: AgentType,
    source_material: str | None = None,
    run_id: str | None = None,
    dataset_key: str | None = None,
):
    model = model or DEFAULT_MODEL

    async def _grade_batch_with_json_retries(
        batch_examples: list[dict],
        *,
        stage: str,
        tag: str,
    ):
        retries = 0
        last_error = None
        while retries <= MAX_GRADING_JSON_RETRIES:
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
  "dataset_score_0_10": 0.0,
  "notes": "string",
  "rows": [
    {{"idx": 0, "score_0_10": 0.0, "accept": 0, "reason": "string"}}
  ]
}}

Rules:
- rows length must equal input dataset length.
- idx must be 0-based and align to input order.
- accept = 1 only when score_0_10 >= {MIN_GRADING_SCORE}.
- No markdown. No extra keys outside schema.{retry_note}
"""
            result = await run_agent_async(
                system_prompt=load_prompt("grading"),
                user_prompt=user_prompt,
                model=GRADING_MODEL,
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

    def _evaluate_rows(batch_examples: list[dict], loaded: dict):
        rows = loaded.get("rows", [])
        if not isinstance(rows, list):
            rows = []
        row_map = {}
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("idx"), int):
                row_map[row["idx"]] = row

        accepted_local = []
        rejected_local = []
        for idx, ex in enumerate(batch_examples):
            row = row_map.get(idx, {})
            try:
                row_score = float(row.get("score_0_10", 0))
            except (TypeError, ValueError):
                row_score = 0.0
            accept_flag = int(row.get("accept", 0)) if isinstance(row, dict) else 0
            reason = str(row.get("reason", "")) if isinstance(row, dict) else "missing_row"
            short_response = len(str(ex.get("response", "")).strip()) < MIN_RESPONSE_CHAR_LENGTH
            if accept_flag == 1 and row_score >= MIN_GRADING_SCORE and not short_response:
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
        try:
            dataset_score = float(loaded.get("dataset_score_0_10", 0))
        except (TypeError, ValueError):
            dataset_score = 0.0
        notes = str(loaded.get("notes", ""))
        return accepted_local, rejected_local, dataset_score, notes

    if not examples:
        logger.saveToLog("[run_grading_agent] No examples passed for grading", "ERROR")
        return []

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
            metadata={
                "topic": topic,
                "model": model,
                "grader_model": GRADING_MODEL,
                "input_count": len(examples),
                "accepted_count": 0,
                "rejected_count": len(examples),
                "notes": str(parse_failure),
                "run_id": run_id,
                "dataset_key": dataset_key,
            },
        )
        return []

    accepted, rejected, dataset_score, dataset_notes = _evaluate_rows(examples, loaded)

    # One regeneration pass for rejected items, then one re-grade pass.
    if rejected:
        rejected_payload = [
            {
                "original_idx": r["idx"],
                "instruction": r["example"].get("instruction", ""),
                "response": r["example"].get("response", ""),
                "failure_reason": r["reason"],
                "failure_score": r["score"],
            }
            for r in rejected
        ]
        try:
            regenerated_map = await _regenerate_rejected_batch(rejected_payload)
            regen_examples = []
            regen_original_indices = []
            for r in rejected:
                idx = r["idx"]
                if idx in regenerated_map:
                    regen_examples.append(regenerated_map[idx])
                    regen_original_indices.append(idx)

            if regen_examples:
                loaded_regen, retries_regen, parse_fail_regen = await _grade_batch_with_json_retries(
                    regen_examples, stage="grading_regeneration_batch", tag="regenerated"
                )
                if not parse_fail_regen and isinstance(loaded_regen, dict):
                    regen_accepted, _, _, _ = _evaluate_rows(regen_examples, loaded_regen)
                    # Map back accepted regenerated examples to original positions.
                    for local_idx, ex in enumerate(regen_examples):
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
        metadata={
            "topic": topic,
            "model": model,
            "grader_model": GRADING_MODEL,
            "input_count": len(examples),
            "accepted_count": len(accepted),
            "rejected_count": rejected_count,
            "notes": dataset_notes,
            "run_id": run_id,
            "dataset_key": dataset_key,
        },
    )
    logger.saveToLog(
        f"[run_grading_agent] Batch summary generator_model={model} grader_model={GRADING_MODEL} topic={topic} input={len(examples)} accepted={len(accepted)} rejected={rejected_count} dataset_score={dataset_score}",
        "INFO",
    )
    if not accepted:
        logger.saveToLog("[run_grading_agent] No examples passed grading", "ERROR")
    return accepted

