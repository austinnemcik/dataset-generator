from typing import Any

from pydantic import BaseModel

from agent import AgentType


class Example(BaseModel):
    instruction: str
    response: str


class IngestExamples(BaseModel):
    example: list[Example] | None = None
    prompt: str
    dataset_id: int
    run_id: str | None = None
    dataset_description: str
    dataset_name: str
    model: str | None = None
    generation_cost: float | None = None
    grading_cost: float | None = None
    total_cost: float | None = None


class BatchGeneration(BaseModel):
    amount: int
    agent_types: list[AgentType] | None = None
    topics: list[str]
    ex_amt: int
    random_agent: bool = False
    max_concurrency: int = 3
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    seed: int | None = None
    source_material: str | list[int | str] | None = None
    model: str | None = None


class Generation(BaseModel):
    agent_type: AgentType
    topic: str
    amount: int
    source_material: str | list[int | str] | None = None
    model: str | None = None


class MergeRequest(BaseModel):
    dataset_ids: list[int] | None = None
    dataset_similarity_threshold: float = 0.65
    delete_originals: bool = False


class ExternalImportRequest(BaseModel):
    url: str
    method: str = "GET"
    headers: dict[str, str] | None = None
    body: dict | list | str | None = None
    field_mapper: dict[str, str] | None = None
    dataset_name: str | None = None
    dataset_description: str | None = None
    model: str | None = None
    prompt: str = "Imported external dataset"
    source_label: str | None = None
    timeout_seconds: float = 30.0
    dedupe_threshold: float = 0.8
    preview_only: bool = False
    preview_limit: int = 10


class ExportRequest(BaseModel):
    dataset_ids: list[int]
    export_format: str = "sharegpt"
    min_score: float | None = None
    dedupe_pass: bool = False
    shuffle: bool = False
    train_val_split: float | None = None
    max_examples: int | None = None


class ScraperTextRecord(BaseModel):
    text: str
    source_url: str | None = None
    title: str | None = None
    metadata: dict[str, Any] | None = None


class ScraperIntakeRequest(BaseModel):
    records: list[ScraperTextRecord]
    dataset_name: str | None = None
    dataset_description: str | None = None
    model: str | None = None
    prompt: str = "Imported scraper text"
    dedupe_threshold: float = 0.8
    preview_only: bool = False
    preview_limit: int = 10
