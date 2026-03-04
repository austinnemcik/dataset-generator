from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from app.core.database import get_session, Dataset, TrainingExample, ClientSettings
from sqlalchemy import func
import json
from pathlib import Path
from app.core.utils import get_path_value
from agent.settings import get_client_settings, ClientSettingsUpdate, ClientSettingsRead, _update_client_settings
from app.core.generics import response_builder

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])


BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "logs" / "speed_benchmark.json"


def _format_metric(value):
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return value


def _get_benchmark_data():
    if not file_path.exists():
        raise FileNotFoundError("File not found")
    
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

        embedding_time = get_path_value(data, "summary.embedding_completion.all_time")
        ingest_time = get_path_value(data, "summary.ingest_api.all_time")
        grading_time = get_path_value(data, "summary.grading_completion.all_time")
        api_cost = get_path_value(data, "summary.api_cost.usd_total")
        token_total = get_path_value(data, "summary.api_cost.tokens_total")
    return embedding_time, ingest_time, grading_time, api_cost, token_total


@dashboard_router.get("/")
def get_dashboard(session: Session = Depends(get_session)):
    payload = {}
    payload["datasets"] = session.exec(select(func.count()).select_from(Dataset)).one()
    payload["training_examples"] = session.exec(select(func.count()).select_from(TrainingExample)).one()
    payload["embedding_time"], payload["ingest_time"], payload["grading_time"], payload["api_cost"], payload["tokens_total"] = _get_benchmark_data()
    payload["embedding_time"] = _format_metric(payload["embedding_time"])
    payload["ingest_time"] = _format_metric(payload["ingest_time"])
    payload["grading_time"] = _format_metric(payload["grading_time"])
    payload["api_cost"] = _format_metric(payload["api_cost"])
    print(payload)
    return payload

@dashboard_router.get("/settings")
def return_client_settings():
    return get_client_settings()

@dashboard_router.patch("/settings", response_model=ClientSettingsRead)
def update_client_settings(
    payload: ClientSettingsUpdate,
    session: Session = Depends(get_session),
    ):
        try: 
            safe_settings, settings = _update_client_settings(payload=payload, session=session)
            return safe_settings
        except ValueError as e:
            return response_builder(success=False, message=f"Error updating settings value. {e}")
    
    
    
        

