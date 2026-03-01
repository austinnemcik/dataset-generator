import json
import os
import random
import zipfile

from sqlmodel import Session, select

from agent import get_embedding, is_duplicate
from config import get_settings
from database import Dataset, ExportHistory, TrainingExample
from generics import get_latest_grading_result, new_run_id
from routes.dataset_models import ExportRequest


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
    return {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": example.instruction},
            {"from": "gpt", "value": example.response},
        ]
    }


def _format_chatml(example: TrainingExample) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
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
            if example.embedding:
                try:
                    embedding_list = json.loads(example.embedding)
                except (TypeError, ValueError, json.JSONDecodeError):
                    embedding_list = None
            else:
                embedding = get_embedding(example.instruction)
                embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
            if not isinstance(embedding_list, list) or not embedding_list:
                deduped.append(example)
                continue
            comparable = [existing for existing in existing_embeddings if len(existing) == len(embedding_list)]
            if is_duplicate(embedding_list, comparable):
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
        output_filename=os.path.basename(output_path),
        output_path=output_path,
        total_examples=len(examples),
        train_examples=train_count,
        val_examples=val_count,
    )
    session.add(history)
    session.commit()
    session.refresh(history)
    return output_path, history, stats
