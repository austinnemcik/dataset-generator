from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

from app.core.enums import AgentType


def _clean_required_text(value: str, *, field_name: str, min_length: int = 2) -> str:
    cleaned = value.strip()
    if len(cleaned) < min_length:
        raise ValueError(f"{field_name} must contain at least {min_length} non-space characters.")
    return cleaned


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
    category: str | None = None
    model: str | None = None
    generation_cost: float | None = None
    grading_cost: float | None = None
    total_cost: float | None = None

    @field_validator("example")
    @classmethod
    def validate_examples(cls, value: list[Example] | None) -> list[Example] | None:
        if not value:
            raise ValueError("example must contain at least one item.")
        return value

    @field_validator("prompt", "dataset_description", "dataset_name")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _clean_required_text(value, field_name="value")


class BatchGeneration(BaseModel):
    amount: int
    request_group_id: str | None = None
    agent_types: list[AgentType] | None = None
    topics: list[str]
    allow_topic_variations: bool = False
    ex_amt: int
    auto_merge_related: bool = False
    auto_merge_similarity_threshold: float = 0.65
    random_agent: bool = False
    max_concurrency: int = 3
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    seed: int | None = None
    source_material: str | list[int | str] | None = None
    source_material_mode: Literal["style_only", "content_and_style"] = "content_and_style"
    source_material_example_limit: int = 250
    source_material_example_selection: Literal["first", "random"] = "random"
    grading_lens: Literal["balanced_quality", "voice_alignment"] = "balanced_quality"
    conversation_length_mode: Literal["varied", "short", "balanced", "long"] = "varied"
    model: str | None = None

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value: int) -> int:
        if value < 1 or value > 250:
            raise ValueError("amount must be in the range 1-250.")
        return value

    @field_validator("ex_amt")
    @classmethod
    def validate_ex_amt(cls, value: int) -> int:
        if value < 1 or value > 50:
            raise ValueError("ex_amt must be in the range 1-50.")
        return value

    @field_validator("auto_merge_similarity_threshold")
    @classmethod
    def validate_auto_merge_similarity_threshold(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("auto_merge_similarity_threshold must be in the range (0, 1].")
        return value

    @field_validator("max_concurrency")
    @classmethod
    def validate_max_concurrency(cls, value: int) -> int:
        if value < 1 or value > 50:
            raise ValueError("max_concurrency must be in the range 1-50.")
        return value

    @field_validator("source_material_example_limit")
    @classmethod
    def validate_source_material_example_limit(cls, value: int) -> int:
        if value < 1 or value > 500:
            raise ValueError("source_material_example_limit must be in the range 1-500.")
        return value

    @field_validator("topics")
    @classmethod
    def normalize_topics(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen_topics: set[str] = set()
        for topic in value:
            if not isinstance(topic, str):
                continue
            cleaned = topic.strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen_topics:
                continue
            seen_topics.add(key)
            normalized.append(cleaned)
        if not normalized:
            raise ValueError("topics must contain at least one non-empty value.")
        if len(normalized) > 5:
            raise ValueError("topics may contain at most 5 unique values.")
        return normalized

    @field_validator("agent_types")
    @classmethod
    def normalize_agent_types(cls, value: list[AgentType] | None) -> list[AgentType] | None:
        if not value:
            return None
        normalized: list[AgentType] = []
        seen_agents: set[AgentType] = set()
        for agent in value:
            if agent in seen_agents:
                continue
            seen_agents.add(agent)
            normalized.append(agent)
        return normalized

    @model_validator(mode="after")
    def validate_agent_selection(self):
        if not self.random_agent and not self.agent_types:
            raise ValueError("agent_types must contain at least one value when random_agent is false.")
        return self


class Generation(BaseModel):
    agent_type: AgentType
    topic: str
    amount: int
    source_material: str | list[int | str] | None = None
    source_material_mode: Literal["style_only", "content_and_style"] = "content_and_style"
    conversation_length_mode: Literal["varied", "short", "balanced", "long"] = "varied"
    model: str | None = None

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        return _clean_required_text(value, field_name="topic")

    @field_validator("amount")
    @classmethod
    def validate_generation_amount(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("amount must be greater than 0.")
        return value


class UpdateTrainingExample(BaseModel):
    instruction: str
    response: str

    @field_validator("instruction", "response")
    @classmethod
    def validate_example_text(cls, value: str) -> str:
        return _clean_required_text(value, field_name="value")


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
    dedupe_against_existing: bool = True
    dedupe_within_payload: bool = True
    max_records: int = 2000
    chunk_size: int = 200
    preview_only: bool = False
    preview_limit: int = 10

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_seconds must be greater than 0.")
        return value

    @field_validator("dedupe_threshold")
    @classmethod
    def validate_dedupe_threshold(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("dedupe_threshold must be in the range (0, 1].")
        return value

    @field_validator("chunk_size")
    @classmethod
    def validate_external_chunk_size(cls, value: int) -> int:
        if value <= 0 or value > 500:
            raise ValueError("chunk_size must be in the range 1-500.")
        return value

    @field_validator("preview_limit")
    @classmethod
    def validate_external_preview_limit(cls, value: int) -> int:
        if value <= 0 or value > 100:
            raise ValueError("preview_limit must be in the range 1-100.")
        return value

    @field_validator("max_records")
    @classmethod
    def validate_external_max_records(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_records must be greater than 0.")
        return value


class ExportRequest(BaseModel):
    dataset_ids: list[int]
    export_format: str = "sharegpt"
    min_score: float | None = None
    dedupe_pass: bool = False
    shuffle: bool = False
    train_val_split: float | None = None
    max_examples: int | None = None

    @field_validator("dataset_ids")
    @classmethod
    def validate_dataset_ids(cls, value: list[int]) -> list[int]:
        deduped: list[int] = []
        seen: set[int] = set()
        for dataset_id in value:
            if dataset_id <= 0:
                raise ValueError("dataset_ids must contain only positive integers.")
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            deduped.append(dataset_id)
        if not deduped:
            raise ValueError("dataset_ids must contain at least one value.")
        return deduped

    @field_validator("export_format")
    @classmethod
    def validate_export_format(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"sharegpt", "chatml", "alpaca"}:
            raise ValueError("export_format must be one of: sharegpt, chatml, alpaca.")
        return cleaned

    @field_validator("train_val_split")
    @classmethod
    def validate_train_val_split(cls, value: float | None) -> float | None:
        if value is not None and not (0 < value < 1):
            raise ValueError("train_val_split must be between 0 and 1.")
        return value

    @field_validator("max_examples")
    @classmethod
    def validate_max_examples(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_examples must be greater than 0.")
        return value


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
    dedupe_against_existing: bool = True
    dedupe_within_payload: bool = True
    max_records: int = 2000
    chunk_size: int = 200
    response_char_limit: int = 160
    preview_only: bool = False
    preview_limit: int = 10

    @field_validator("records")
    @classmethod
    def validate_records(cls, value: list[ScraperTextRecord]) -> list[ScraperTextRecord]:
        if not value:
            raise ValueError("records must contain at least one item.")
        return value

    @field_validator("dedupe_threshold")
    @classmethod
    def validate_scraper_dedupe_threshold(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("dedupe_threshold must be in the range (0, 1].")
        return value

    @field_validator("chunk_size")
    @classmethod
    def validate_scraper_chunk_size(cls, value: int) -> int:
        if value <= 0 or value > 500:
            raise ValueError("chunk_size must be in the range 1-500.")
        return value

    @field_validator("response_char_limit")
    @classmethod
    def validate_response_char_limit(cls, value: int) -> int:
        if value < 32 or value > 8000:
            raise ValueError("response_char_limit must be in the range 32-8000.")
        return value

    @field_validator("preview_limit")
    @classmethod
    def validate_scraper_preview_limit(cls, value: int) -> int:
        if value <= 0 or value > 100:
            raise ValueError("preview_limit must be in the range 1-100.")
        return value

    @field_validator("max_records")
    @classmethod
    def validate_scraper_max_records(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_records must be greater than 0.")
        return value

    @model_validator(mode="after")
    def validate_scraper_record_count(self):
        if len(self.records) > self.max_records:
            raise ValueError(
                f"records exceeds max_records ({self.max_records}). Split into smaller payloads."
            )
        return self



