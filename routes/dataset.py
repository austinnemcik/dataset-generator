from fastapi import APIRouter

from routes.dataset_batch_routes import register_batch_routes
from routes.dataset_data_routes import register_data_routes
from routes.dataset_merge_routes import register_merge_routes


data_router = APIRouter(prefix="/dataset", tags=["dataset"])

register_batch_routes(data_router)
register_data_routes(data_router)
register_merge_routes(data_router)
