import os
import requests
from openai import OpenAI
import app.core.logger as logger
from app.core.config import get_settings
from sqlmodel import select, Session
from app.core.database import get_session, ClientSettings, engine
from fastapi import Depends
from pydantic import BaseModel

_settings = get_settings()
_client_settings: ClientSettings | None = None

SERVER_URL = _settings.server_url
API_KEY = os.getenv("OPENROUTER_API_KEY")

class ClientSettingsUpdate(BaseModel):
    default_model: str | None = None
    grading_model: str | None = None
    naming_model: str | None = None
    threshold: float | None = None
    min_grading_score: float | None = None
    min_response_char_length: int | None = None
    max_grading_json_retries: int | None = None
    max_naming_json_retries: int | None = None
    max_low_quality_retries: int | None = None
    max_generation_retries: int | None = None
    min_save_ratio: float | None = None

class ClientSettingsRead(BaseModel):
    default_model: str 
    grading_model: str 
    naming_model: str 
    threshold: float 
    min_grading_score: float 
    min_response_char_length: int 
    max_grading_json_retries: int
    max_naming_json_retries: int
    max_low_quality_retries: int
    max_generation_retries: int 
    min_save_ratio: float 
    model_config = {"from_attributes": True}


MODEL_PRICING: dict[str, dict[str, float]] = {}
MODEL_CARDS: list[dict] = []

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)


def load_models():
    if not API_KEY:
        logger.saveToLog("OPENROUTER_API_KEY is not configured; skipping pricing refresh.", "WARNING")
        return

    try:
        res = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=15,
        )
        res.raise_for_status()
    except requests.RequestException as e:
        logger.saveToLog(f"Failed to reach OpenRouter pricing API: {e}", "ERROR")
        return

    data = res.json().get("data", [])

    MODEL_PRICING.clear()
    MODEL_CARDS.clear()

    for model in data:
        try:
            model_id = model["id"]
            pricing = model["pricing"]

            MODEL_PRICING[model_id] = {
                "prompt": float(pricing["prompt"]),
                "completion": float(pricing["completion"]),
            }

            provider, name = model_id.split("/", 1)

            MODEL_CARDS.append({
                "id": model_id,
                "provider": provider,
                "name": name,
            })
        except (KeyError, ValueError, TypeError):
            logger.saveToLog(f"Failed to get valid pricing data for {model}")
            continue


def calculate_price(input_tokens: int, output_tokens: int, model: str) -> tuple[float, float, float]:
    pricing = MODEL_PRICING.get(model)
    if not pricing or pricing["prompt"] < 0:
        logger.saveToLog(f"Pricing not found for model: {model}", "WARNING")
        return 0.0, 0.0, 0.0
    input_price = pricing["prompt"] * input_tokens
    output_price = pricing["completion"] * output_tokens
    total_price = input_price + output_price
    return total_price, input_price, output_price

def load_client_settings_on_startup():
    global _client_settings
    with Session(engine) as session:
        _client_settings = session.exec(select(ClientSettings)).first()
        if _client_settings is None:
            _client_settings = ClientSettings()
            session.add(_client_settings)
            session.commit()
            session.refresh(_client_settings)
        return _client_settings

# fastapi route use only
def _update_client_settings(payload: ClientSettingsUpdate, session: Session = Depends(get_session)):
    global _client_settings
    settings = session.exec(select(ClientSettings)).first()

    if settings is None:
        raise RuntimeError("Client settings not initialized")

    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise ValueError("No fields provided for update")

    updates.pop("id", None)
    updates.pop("created_at", None)
    updates.pop("updated_at", None)

    for key, value in updates.items():
        setattr(settings, key, value)

    session.commit()
    session.refresh(settings)
    _client_settings = settings
    return_model = ClientSettingsRead.model_validate(settings)
    return return_model, settings


def get_client_settings() -> ClientSettingsRead:
    global _client_settings
    if _client_settings is None:
        _client_settings = load_client_settings_on_startup()
    if _client_settings is None:
        raise RuntimeError("Client settings not initialized")
    return ClientSettingsRead.model_validate(_client_settings)

def get_models_cached() -> list:
    if not MODEL_CARDS:
        raise RuntimeError("Models not initialized")
    return MODEL_CARDS  
