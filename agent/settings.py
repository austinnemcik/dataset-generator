import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

import logger

load_dotenv()

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "z-ai/glm-5"
GRADING_MODEL = "google/gemini-2.5-flash"
THRESHOLD = 0.8

MIN_GRADING_SCORE = 8.0
MIN_RESPONSE_CHAR_LENGTH = 40
MAX_GRADING_JSON_RETRIES = 2
MAX_LOW_QUALITY_RETRIES = 1
MAX_GENERATION_RETRIES = 1
MIN_SAVE_RATIO = 0.8

MODEL_PRICING: dict[str, dict[str, float]] = {}
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)


def load_pricing():
    try:
        res = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        res.raise_for_status()
    except requests.RequestException as e:
        logger.saveToLog(f"Failed to reach OpenRouter pricing API: {e}", "ERROR")
        return
    for model in res.json()["data"]:
        try:
            MODEL_PRICING[model["id"]] = {
                "prompt": float(model["pricing"]["prompt"]),
                "completion": float(model["pricing"]["completion"]),
            }
        except (KeyError, ValueError, TypeError):
            logger.saveToLog(f"Failed to get valid pricing data for {model}")
            continue


def calculate_price(input_tokens: int, output_tokens: int, model: str):
    pricing = MODEL_PRICING.get(model)
    if not pricing or pricing["prompt"] < 0:
        logger.saveToLog(f"Pricing not found for model: {model}", "WARNING")
        return "$0.0", "$0.0", "$0.0"
    input_price = pricing["prompt"] * input_tokens
    output_price = pricing["completion"] * output_tokens
    total_price = input_price + output_price
    return f"${total_price}", f"${input_price}", f"${output_price}"

