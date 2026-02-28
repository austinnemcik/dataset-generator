import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import text
from sqlmodel import Field, Relationship, SQLModel, Session, create_engine

load_dotenv()
_DATABASE_URL = os.getenv("DATABASE_URL")

_engine_kwargs = {"pool_pre_ping": True}
if _DATABASE_URL and _DATABASE_URL.startswith("postgresql"):
    # Fail fast when Postgres is unavailable instead of hanging for minutes.
    _engine_kwargs["connect_args"] = {"connect_timeout": 5}

engine = create_engine(_DATABASE_URL, **_engine_kwargs) # what we use to actually query postgres


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def init_db():
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        try:
            conn.execute(text("ALTER TABLE dataset ADD COLUMN IF NOT EXISTS source_run_id VARCHAR"))
        except Exception:
            pass
        try:
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_dataset_source_run_id "
                    "ON dataset (source_run_id)"
                )
            )
        except Exception:
            pass


def get_session():
    with Session(engine) as session:
        yield session


class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True) # this will automatically set an id if None
    name: str
    description: str | None = None
    model: str | None = None
    source_run_id: str | None = Field(default=None, index=True, unique=True)
    generation_cost: float = 0.0
    grading_cost: float = 0.0
    total_cost: float = 0.0
    examples: list["TrainingExample"] = Relationship(back_populates="dataset", sa_relationship_kwargs={"cascade": "all, delete-orphan"})

class TrainingExample(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    prompt: str
    instruction: str
    response: str
    dataset_id: int = Field(foreign_key="dataset.id")
    dataset: "Dataset" = Relationship(back_populates="examples")
    embedding: str | None = None 


class BatchRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, unique=True)
    status: str = Field(default="queued", index=True)
    request_json: str
    summary_json: str | None = None
    total_runs: int = 0
    queued_runs: int = 0
    running_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    items: list["BatchRunItem"] = Relationship(
        back_populates="batch_run",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class BatchRunItem(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    batch_run_id: int = Field(foreign_key="batchrun.id", index=True)
    item_index: int = Field(index=True)
    run_id: str = Field(index=True, unique=True)
    dataset_key: str
    slot_key: str | None = None
    requested_topic: str
    topic: str
    agent: str
    ex_amt: int
    seed: int | None = None
    status: str = Field(default="queued", index=True)
    attempts: int = 0
    max_retries: int = 0
    retry_backoff_seconds: float = 0.0
    source_material: str | None = None
    model: str | None = None
    error_type: str | None = None
    error: str | None = None
    created_dataset_id: int | None = None
    result_json: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    batch_run: "BatchRun" = Relationship(back_populates="items")

