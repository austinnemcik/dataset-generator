import enum
import json
from fastapi.responses import JSONResponse
from contextlib import contextmanager
import time
import os

def valid_http(statusCode):
    if not isinstance(statusCode, int):
        return False
    return (statusCode > 100 and statusCode <= 599)

def response_builder(
        *,
        success: bool, 
        message: str, 
        count: int | None = None, 
        errors: int | None = None, 
        statusCode: int = 200):
    status = statusCode
    if not (valid_http(statusCode)):
        status = 500

    return JSONResponse({
        "success": success,
        "message": message,
        "amount": count,
        "errors": errors
    }, status_code=status)

class TimedLabel(enum.Enum):
    CHAT_COMPLETION = "chat_completion"
    NAMING_CALL = "naming_completion"
    INGEST_REQUEST = "ingest_api"

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
        with open(BENCHMARK_FILE, 'r') as f:
            try:
                data = json.load(f)  # Parses the JSON file into a Python dict
            except (json.JSONDecodeError, ValueError):
                data = {}  # If the file is missing or corrupted, start fresh

    # If this label doesn't exist in the dict yet, create an empty list for it,
    # then append the new duration. setdefault() is a dict method that does both
    # in one step: get the existing list, or create it if it's not there yet.
    data.setdefault(label, []).append(seconds)

    # Write the updated dict back to the file as JSON
    with open(BENCHMARK_FILE, 'w') as f:
        json.dump(data, f, indent=2)  # indent=2 makes the output easier to read 
