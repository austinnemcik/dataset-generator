from fastapi import APIRouter, Depends
from app.core.generics import response_builder
from app.core.logger import saveToLog
from agent.settings import get_models_cached, API_KEY
from app.core.utils import get_path_value
import requests

utils_router = APIRouter(prefix="/utils", tags=["utils"])


@utils_router.get("/models")
def get_models(query: str, limit: int = 50):
    q = query.lower().strip()
    try:
        models = get_models_cached()
    except RuntimeError as e:
        return response_builder(success=False, message=str(e), statusCode=500)
    if limit <= 0:
        limit = len(models)
    if not q:
        return [m["id"] for m in models[:limit]]
    results = [m for m in models if q in m["id"].lower()]

    results.sort(key=lambda m: m["name"])

    return [m["id"] for m in results[:limit]]


@utils_router.get("/credits")
def get_credits(index: int = 0):
    if not API_KEY:
        return response_builder(
            success=False, message="Can't find API Key.  Check .env", statusCode=404
        )
    try:
        res = requests.get(
            "https://openrouter.ai/api/v1/credits",
            headers={
                "Authorization": f"Bearer {API_KEY}",
            },
            timeout=15,
        )
        res.raise_for_status()
    except requests.RequestException as e:
        saveToLog(f"Failed to reach OpenRouter credits API: {e}", "ERROR")
        return response_builder(
            success=False,
            message=f"Failed to reach OpenRouter credits API: {e}",
            statusCode=404,
        )

    data = res.json()
    try:
        total_credits = float(get_path_value(data, "data.total_credits"))
        total_usage = float(get_path_value(data, "data.total_usage"))
    except TypeError as e:
        return response_builder(success=False, message="Bad response from OpenRouter", statusCode=404)
    if total_credits is None or total_usage is None:
        return response_builder(
            success=False,
            message=f"Failed to get a response from OpenRouter credits API",
            statusCode=404,
        )
    balance = total_credits - total_usage

    if index == 0:
        return f"{balance:.2f}"
    elif index == 1:
        return f"{total_credits:.2f}"
    elif index == 2:
        return f"{total_usage:.2f}"
    else:
        return response_builder(
            success=False,
            message=f"Invalid index: {index}, available index is 0: Balance, 1: Total Credits, 2: Total Usage",
        )
