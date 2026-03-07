import json
import os
import random
import re
import zipfile

from sqlmodel import Session, select

from app.core.config import get_settings
from app.core.database import Dataset, ExportHistory, TrainingExample
from app.core.generics import get_latest_grading_result, new_run_id
from routes.dataset_models import ExportRequest
from services.embedding_service import embed_text, is_semantic_duplicate, parse_embedding_json


_USER_LINE_RE = re.compile(r"^User:\s*(.+)$")
_ASSISTANT_LINE_RE = re.compile(r"^Assistant:\s*(.+)$")


def _dataset_passes_score_filter(dataset: Dataset, min_score: float | None) -> bool:
    if min_score is None:
        return True
    if not dataset.source_run_id:
        return False
    grading = get_latest_grading_result(dataset.source_run_id)
    if not grading:
        return False
    try:
        score = float(grading.get("score", 0))
    except (TypeError, ValueError):
        return False
    return score >= min_score


def _format_sharegpt(example: TrainingExample) -> dict:
    conversations = [{"from": "system", "value": "You are a helpful assistant."}]
    parsed_messages = _parse_instruction_as_conversation(example.instruction)
    if parsed_messages:
        conversations.extend(parsed_messages)
        conversations.append({"from": "gpt", "value": example.response})
        return {"conversations": conversations}
    return {
        "conversations": conversations
        + [
            {"from": "human", "value": example.instruction},
            {"from": "gpt", "value": example.response},
        ]
    }


def _format_chatml(example: TrainingExample) -> dict:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    parsed_messages = _parse_instruction_as_conversation(example.instruction, user_role="user", assistant_role="assistant")
    if parsed_messages:
        messages.extend(parsed_messages)
        messages.append({"role": "assistant", "content": example.response})
        return {"messages": messages}
    return {
        "messages": messages
        + [
            {"role": "user", "content": example.instruction},
            {"role": "assistant", "content": example.response},
        ]
    }


def _format_alpaca(example: TrainingExample) -> dict:
    return {
        "instruction": example.instruction,
        "input": "",
        "output": example.response,
    }


def _parse_instruction_as_conversation(
    instruction: str,
    *,
    user_role: str = "human",
    assistant_role: str = "gpt",
) -> list[dict[str, str]] | None:
    lines = [line.strip() for line in instruction.splitlines() if line.strip()]
    if not lines:
        return None

    messages: list[dict[str, str]] = []
    expected_role = user_role
    saw_assistant_turn = False

    for line in lines:
        user_match = _USER_LINE_RE.match(line)
        assistant_match = _ASSISTANT_LINE_RE.match(line)
        if user_match:
            role = user_role
            content = user_match.group(1).strip()
        elif assistant_match:
            role = assistant_role
            content = assistant_match.group(1).strip()
        else:
            return None

        if role != expected_role or not content:
            return None
        if role == assistant_role:
            saw_assistant_turn = True
        messages.append(
            {
                ("from" if user_role == "human" else "role"): role,
                ("value" if user_role == "human" else "content"): content,
            }
        )
        expected_role = assistant_role if role == user_role else user_role

    if messages[-1][("from" if user_role == "human" else "role")] != user_role:
        return None
    if not saw_assistant_turn and len(messages) == 1:
        return messages
    return messages


def _serialize_export_record(example: TrainingExample, export_format: str) -> dict:
    fmt = export_format.lower()
    if fmt == "sharegpt":
        return _format_sharegpt(example)
    if fmt == "chatml":
        return _format_chatml(example)
    if fmt == "alpaca":
        return _format_alpaca(example)
    raise ValueError("export_format must be one of: sharegpt, chatml, alpaca")


def _prepare_export_examples(
    session: Session,
    *,
    dataset_ids: list[int],
    min_score: float | None,
    dedupe_pass: bool,
    shuffle_examples: bool,
    max_examples: int | None,
) -> tuple[list[TrainingExample], dict]:
    datasets = session.exec(select(Dataset).where(Dataset.id.in_(dataset_ids)).order_by(Dataset.id)).all()
    found_ids = {dataset.id for dataset in datasets if dataset.id is not None}
    missing_ids = [dataset_id for dataset_id in dataset_ids if dataset_id not in found_ids]
    if missing_ids:
        raise ValueError(f"Dataset IDs not found: {missing_ids}")

    eligible_datasets = [dataset for dataset in datasets if _dataset_passes_score_filter(dataset, min_score)]
    examples = session.exec(
        select(TrainingExample).where(TrainingExample.dataset_id.in_([dataset.id for dataset in eligible_datasets]))
    ).all() if eligible_datasets else []

    stats = {
        "dataset_count": len(datasets),
        "eligible_dataset_count": len(eligible_datasets),
        "score_filtered_dataset_count": len(datasets) - len(eligible_datasets),
        "input_examples": len(examples),
        "deduped_examples": 0,
    }

    if dedupe_pass:
        deduped: list[TrainingExample] = []
        existing_embeddings: list[list[float]] = []
        for example in examples:
            embedding_list = parse_embedding_json(example.embedding) if example.embedding else embed_text(example.instruction)
            if embedding_list is None:
                deduped.append(example)
                continue
            if is_semantic_duplicate(embedding_list, existing_embeddings):
                stats["deduped_examples"] += 1
                continue
            existing_embeddings.append(embedding_list)
            deduped.append(example)
        examples = deduped

    if shuffle_examples:
        random.shuffle(examples)
    if max_examples is not None:
        examples = examples[:max_examples]

    stats["output_examples"] = len(examples)
    return examples, stats


def _write_jsonl(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def _export_artifact_dir() -> str:
    base_dir = get_settings().export_artifact_dir
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _build_export_files(
    *,
    examples: list[TrainingExample],
    export_format: str,
    train_val_split: float | None,
    base_name: str,
) -> tuple[str, int, int]:
    rows = [_serialize_export_record(example, export_format) for example in examples]
    safe_base = base_name.replace(" ", "-")
    artifact_dir = _export_artifact_dir()
    if train_val_split is None:
        filepath = os.path.join(artifact_dir, f"{safe_base}.jsonl")
        _write_jsonl(filepath, rows)
        return filepath, len(rows), 0

    val_count = int(len(rows) * train_val_split)
    train_count = len(rows) - val_count
    train_rows = rows[:train_count]
    val_rows = rows[train_count:]
    train_path = os.path.join(artifact_dir, f"{safe_base}-train.jsonl")
    val_path = os.path.join(artifact_dir, f"{safe_base}-val.jsonl")
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    zip_path = os.path.join(artifact_dir, f"{safe_base}-split.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(train_path, arcname=os.path.basename(train_path))
        archive.write(val_path, arcname=os.path.basename(val_path))
    return zip_path, len(train_rows), len(val_rows)


def _normalized_output_filename(output_path: str) -> str:
    filename = os.path.basename(output_path)
    if os.path.splitext(filename)[1]:
        return filename
    if zipfile.is_zipfile(output_path):
        return f"{filename}.zip"
    return f"{filename}.jsonl"


def run_export_request(session: Session, body: ExportRequest) -> tuple[str, ExportHistory, dict]:
    dataset_ids = []
    seen_ids: set[int] = set()
    for dataset_id in body.dataset_ids:
        if dataset_id in seen_ids:
            continue
        seen_ids.add(dataset_id)
        dataset_ids.append(dataset_id)
    if not dataset_ids:
        raise ValueError("dataset_ids must contain at least one value.")
    if body.train_val_split is not None and not (0 < body.train_val_split < 1):
        raise ValueError("train_val_split must be between 0 and 1.")
    if body.max_examples is not None and body.max_examples <= 0:
        raise ValueError("max_examples must be greater than 0.")

    examples, stats = _prepare_export_examples(
        session,
        dataset_ids=dataset_ids,
        min_score=body.min_score,
        dedupe_pass=body.dedupe_pass,
        shuffle_examples=body.shuffle,
        max_examples=body.max_examples,
    )
    if not examples:
        raise ValueError("No examples available after export filtering.")

    export_id = new_run_id()[:12]
    output_path, train_count, val_count = _build_export_files(
        examples=examples,
        export_format=body.export_format,
        train_val_split=body.train_val_split,
        base_name=f"dataset-export-{export_id}",
    )
    history = ExportHistory(
        status="completed",
        export_format=body.export_format.lower(),
        dataset_ids_json=json.dumps(dataset_ids),
        options_json=json.dumps(
            {
                "min_score": body.min_score,
                "dedupe_pass": body.dedupe_pass,
                "shuffle": body.shuffle,
                "train_val_split": body.train_val_split,
                "max_examples": body.max_examples,
            }
        ),
        output_filename=_normalized_output_filename(output_path),
        output_path=output_path,
        total_examples=len(examples),
        train_examples=train_count,
        val_examples=val_count,
    )
    session.add(history)
    session.commit()
    session.refresh(history)
    return output_path, history, stats


