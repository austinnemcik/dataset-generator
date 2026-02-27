import enum
import json
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
    statusCode: int = 200
):
    status = statusCode
    if not (valid_http(statusCode)):
        status = 500

    return JSONResponse(
        {"success": success, "message": message, "amount": count, "errors": errors},
        status_code=status,
    )


class TimedLabel(enum.Enum):
    CHAT_COMPLETION = "chat_completion"
    NAMING_CALL = "naming_completion"
    INGEST_REQUEST = "ingest_api"
    GRADING_CALL = "grading_completion"
    EMBEDDING_CALL = "embedding_completion"


@contextmanager
def timer(label: TimedLabel):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        saveTime(label.value, elapsed)


def saveTime(label: str, seconds: float):
    BENCHMARK_FILE = "logs/benchmark.json"
    # Create the logs/ directory if it doesn't already exist
    os.makedirs(os.path.dirname(BENCHMARK_FILE), exist_ok=True)

    # Start with an empty dict, then try to load existing data from the file
    data = {}
    if os.path.exists(BENCHMARK_FILE):
        with open(BENCHMARK_FILE, "r") as f:
            try:
                data = json.load(f)  # Parses the JSON file into a Python dict
            except (json.JSONDecodeError, ValueError):
                data = {}  # If the file is missing or corrupted, start fresh

    bucket = data.setdefault(label, {"entries": [], "averages": {}})
    bucket["entries"].append({
        "seconds": seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    all_times = [e["seconds"] for e in bucket["entries"]]
    def avg(values): return sum(values) / len(values) if values else None
    bucket["averages"] = {
        "last_5": avg(all_times[-5:]),
        "last_10": avg(all_times[-10:]),
        "all_time": avg(all_times),
    }

    # Write the updated dict back to the file as JSON
    with open(BENCHMARK_FILE, "w") as f:
        json.dump(data, f, indent=2)  # indent=2 makes the output easier to read


def saveScore(label: str, score: float, metadata: dict | None = None):
    BENCHMARK_FILE = "logs/benchmark.json"
    os.makedirs(os.path.dirname(BENCHMARK_FILE), exist_ok=True)

    data = {}
    if os.path.exists(BENCHMARK_FILE):
        with open(BENCHMARK_FILE, "r") as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                data = {}

    # Clamp score to expected grading range.
    safe_score = max(0.0, min(10.0, float(score)))

    bucket = data.setdefault(label, {"entries": [], "averages": {}})
    entry = {
        "score": safe_score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        entry.update(metadata)
    bucket["entries"].append(entry)

    all_scores = [e.get("score") for e in bucket["entries"] if isinstance(e.get("score"), (int, float))]

    def avg(values):
        return sum(values) / len(values) if values else None

    bucket["averages"] = {
        "last_5": avg(all_scores[-5:]),
        "last_10": avg(all_scores[-10:]),
        "all_time": avg(all_scores),
    }

    with open(BENCHMARK_FILE, "w") as f:
        json.dump(data, f, indent=2)
