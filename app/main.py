from pathlib import Path
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Support direct execution via `python app/main.py` by ensuring the project
# root is on sys.path before importing sibling top-level packages.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from agent.automation import resume_incomplete_batch_runs
from agent.settings import load_pricing
from app.core.database import init_db
from routes.dataset import data_router
from routes.dashboard import dashboard_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    load_pricing()
    await resume_incomplete_batch_runs()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(data_router)
app.include_router(dashboard_router)


@app.get('/')
def get_root():
    return {"message": "Data Processing API"}


@app.get("/health")
def get_health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
