import enum
import json
import uuid
from fastapi.responses import JSONResponse
from contextlib import contextmanager
import time
import os
from datetime import datetime, timezone


def valid_http(statusCode):
    if not isinstance(statusCode, int):
        return False
    return statusCode > 100 and statusCode <= 599


def response_builder(
    *,
    success: bool,
    message: str,
    count: int | None = None,
    errors: int | None = None,
    statusCode: int = 200,
    data: dict | None = None
):
    status = statusCode
    if not (valid_http(statusCode)):
        status = 500

    return JSONResponse(
        {"success": success, "message": message, "amount": count, "errors": errors, "data": data},
        status_code=status,
    )


class TimedLabel(enum.Enum):
    CHAT_COMPLETION = "chat_completion"
    NAMING_CALL = "naming_completion"
    INGEST_REQUEST = "ingest_api"
    GRADING_CALL = "grading_completion"
    EMBEDDING_CALL = "embedding_completion"
    BATCH_GENERATION = "batch_generation"


@contextmanager
def timer(label: TimedLabel):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        saveTime(label.value, elapsed)


def saveTime(label: str, seconds: float):
    rows = _load_speed_rows()
    summary = _load_speed_summary()

    bucket = rows.setdefault(label, [])
    bucket.append(
        {
            "seconds": seconds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    all_times = [e["seconds"] for e in bucket if isinstance(e.get("seconds"), (int, float))]

    def avg(values):
        return sum(values) / len(values) if values else None

    summary[label] = {
        "last_5": avg(all_times[-5:]),
        "last_10": avg(all_times[-10:]),
        "all_time": avg(all_times),
    }

    _write_speed_rows(rows)
    _write_speed_summary(summary)


def saveScore(label: str, score: float, metadata: dict | None = None):
    rows = _load_grading_rows()
    summary = _load_grading_summary()

    safe_score = max(0.0, min(10.0, float(score)))

    bucket = rows.setdefault(label, [])
    entry = {
        "score": safe_score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        entry.update(metadata)
    bucket.append(entry)

    all_scores = [e.get("score") for e in bucket if isinstance(e.get("score"), (int, float))]

    def avg(values):
        return sum(values) / len(values) if values else None

    summary[label] = {
        "last_5": avg(all_scores[-5:]),
        "last_10": avg(all_scores[-10:]),
        "all_time": avg(all_scores),
    }

    _write_grading_rows(rows)
    _write_grading_summary(summary)


def saveCost(
    *,
    run_id: str,
    model: str,
    stage: str,
    usd_total: float,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    topic: str | None = None,
    dataset_key: str | None = None,
):
    rows = _load_speed_rows()
    summary = _load_speed_summary()
    bucket = rows.setdefault("api_cost", [])
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset_key": dataset_key,
        "topic": topic,
        "model": model,
        "stage": stage,
        "usd_total": float(usd_total),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }
    bucket.append(entry)

    totals = {}
    totals["usd_total"] = round(
        sum(float(e.get("usd_total", 0.0)) for e in bucket), 8
    )
    totals["tokens_total"] = int(
        sum(int(e.get("total_tokens", 0)) for e in bucket)
    )

    by_model = {}
    by_dataset = {}
    for e in bucket:
        m = e.get("model", "UNKNOWN")
        d = e.get("dataset_key") or e.get("run_id") or "UNKNOWN"
        by_model.setdefault(m, {"usd_total": 0.0, "tokens_total": 0, "calls": 0})
        by_dataset.setdefault(d, {"usd_total": 0.0, "tokens_total": 0, "calls": 0})
        by_model[m]["usd_total"] += float(e.get("usd_total", 0.0))
        by_model[m]["tokens_total"] += int(e.get("total_tokens", 0))
        by_model[m]["calls"] += 1
        by_dataset[d]["usd_total"] += float(e.get("usd_total", 0.0))
        by_dataset[d]["tokens_total"] += int(e.get("total_tokens", 0))
        by_dataset[d]["calls"] += 1

    for table in (by_model, by_dataset):
        for _, v in table.items():
            v["usd_total"] = round(v["usd_total"], 8)

    totals["by_model"] = by_model
    totals["by_dataset"] = by_dataset
    summary["api_cost"] = totals

    _write_speed_rows(rows)
    _write_speed_summary(summary)


def get_run_costs(run_id: str) -> dict:
    rows = _load_speed_rows()
    entries = rows.get("api_cost", [])
    generation_stages = {"generation", "naming", "regeneration_batch"}
    grading_stages = {"grading_batch", "grading_regeneration_batch"}
    generation_cost = 0.0
    grading_cost = 0.0
    total_cost = 0.0
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("run_id") != run_id:
            continue
        cost = float(e.get("usd_total", 0.0))
        stage = str(e.get("stage", ""))
        total_cost += cost
        if stage in generation_stages:
            generation_cost += cost
        elif stage in grading_stages:
            grading_cost += cost
    return {
        "generation_cost": round(generation_cost, 8),
        "grading_cost": round(grading_cost, 8),
        "total_cost": round(total_cost, 8),
    }


def new_run_id() -> str:
    return str(uuid.uuid4())


def _speed_benchmark_path() -> str:
    return "logs/speed_benchmark.json"


def _grading_benchmark_path() -> str:
    return "logs/grading_benchmark.json"


def _load_json_file(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:
            return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}


def _write_json_file(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _normalize_benchmark_document(data: dict) -> dict:
    if not isinstance(data, dict):
        return {"rows": {}, "summary": {}}

    if "rows" in data or "summary" in data:
        rows = data.get("rows", {})
        summary = data.get("summary", {})
        return {
            "rows": rows if isinstance(rows, dict) else {},
            "summary": summary if isinstance(summary, dict) else {},
        }

    rows = {}
    summary = {}
    for label, payload in data.items():
        if not isinstance(payload, dict):
            continue
        entries = payload.get("entries", [])
        averages = payload.get("averages", {})
        if isinstance(entries, list):
            rows[label] = entries
        if isinstance(averages, dict):
            summary[label] = averages
    return {"rows": rows, "summary": summary}


def _load_benchmark_document(path: str) -> dict:
    return _normalize_benchmark_document(_load_json_file(path))


def _write_benchmark_document(path: str, rows: dict, summary: dict):
    _write_json_file(path, {"summary": summary, "rows": rows})


def _load_speed_rows() -> dict:
    return _load_benchmark_document(_speed_benchmark_path())["rows"]


def _write_speed_rows(data: dict):
    doc = _load_benchmark_document(_speed_benchmark_path())
    _write_benchmark_document(_speed_benchmark_path(), data, doc["summary"])


def _load_speed_summary() -> dict:
    return _load_benchmark_document(_speed_benchmark_path())["summary"]


def _write_speed_summary(data: dict):
    doc = _load_benchmark_document(_speed_benchmark_path())
    _write_benchmark_document(_speed_benchmark_path(), doc["rows"], data)


def _load_grading_rows() -> dict:
    return _load_benchmark_document(_grading_benchmark_path())["rows"]


def _write_grading_rows(data: dict):
    doc = _load_benchmark_document(_grading_benchmark_path())
    _write_benchmark_document(_grading_benchmark_path(), data, doc["summary"])


def _load_grading_summary() -> dict:
    return _load_benchmark_document(_grading_benchmark_path())["summary"]


def _write_grading_summary(data: dict):
    doc = _load_benchmark_document(_grading_benchmark_path())
    _write_benchmark_document(_grading_benchmark_path(), doc["rows"], data)
