import json

import httpx
from sqlmodel import Session, select

from agent import get_embedding, is_duplicate
from database import TrainingExample
from routes.data_processing import detect_format, extract_records, normalize_import_records, normalize_scraper_text
from routes.dataset_models import ExternalImportRequest, ScraperIntakeRequest


def load_existing_import_embeddings(session: Session) -> list[list[float]]:
    embeddings: list[list[float]] = []
    examples = session.exec(select(TrainingExample.embedding).where(TrainingExample.embedding.is_not(None))).all()
    for embedding_raw in examples:
        try:
            parsed = json.loads(embedding_raw)
            if isinstance(parsed, list) and parsed:
                embeddings.append([float(x) for x in parsed])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return embeddings


def iter_chunks(items: list, chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


async def fetch_external_payload(body: ExternalImportRequest):
    method = body.method.upper()
    headers = body.headers or {}
    request_kwargs = {"headers": headers, "timeout": body.timeout_seconds}
    if body.body is not None:
        if isinstance(body.body, (dict, list)):
            request_kwargs["json"] = body.body
        else:
            request_kwargs["content"] = str(body.body)
    async with httpx.AsyncClient() as client:
        response = await client.request(method, body.url, **request_kwargs)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise ValueError("External import endpoint did not return JSON.") from exc


def build_scraper_examples(session: Session, body: ScraperIntakeRequest) -> tuple[list[TrainingExample], int, int]:
    existing_embeddings = load_existing_import_embeddings(session) if body.dedupe_against_existing else []
    payload_embeddings: list[list[float]] = []
    duplicate_records = 0
    invalid_records = 0
    imported_examples: list[TrainingExample] = []

    for chunk in iter_chunks(body.records, body.chunk_size):
        for record in chunk:
            parsed = normalize_scraper_text(record.text, body.response_char_limit)
            if not parsed:
                invalid_records += 1
                continue
            instruction, response = parsed
            if record.source_url or record.title:
                context_parts = []
                if record.title:
                    context_parts.append(f"Title: {record.title}")
                if record.source_url:
                    context_parts.append(f"Source: {record.source_url}")
                if context_parts:
                    instruction = f"{instruction}\n\n" + "\n".join(context_parts)

            embedding = get_embedding(instruction)
            embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
            if not isinstance(embedding_list, list) or not embedding_list:
                invalid_records += 1
                continue

            comparable_embeddings: list[list[float]] = []
            if body.dedupe_against_existing:
                comparable_embeddings.extend(
                    existing for existing in existing_embeddings if len(existing) == len(embedding_list)
                )
            if body.dedupe_within_payload:
                comparable_embeddings.extend(
                    existing for existing in payload_embeddings if len(existing) == len(embedding_list)
                )

            if comparable_embeddings and is_duplicate(
                embedding_list,
                comparable_embeddings,
                threshold=body.dedupe_threshold,
            ):
                duplicate_records += 1
                continue

            if body.dedupe_within_payload:
                payload_embeddings.append(embedding_list)
            imported_examples.append(
                TrainingExample(
                    prompt=body.prompt,
                    instruction=instruction,
                    response=response,
                    embedding=json.dumps(embedding_list),
                )
            )

    return imported_examples, duplicate_records, invalid_records


def normalize_external_records(
    body: ExternalImportRequest,
    payload,
) -> tuple[str, list, list[dict], int]:
    records = extract_records(payload)
    detected_format = detect_format(records, body.field_mapper)
    normalized_rows, invalid_records = normalize_import_records(
        records,
        detected_format=detected_format,
        field_mapper=body.field_mapper,
    )
    return detected_format, records, normalized_rows, invalid_records


def build_external_examples(
    session: Session,
    body: ExternalImportRequest,
    normalized_rows: list[dict],
    *,
    initial_invalid_records: int,
) -> tuple[list[TrainingExample], int, int]:
    existing_embeddings = load_existing_import_embeddings(session) if body.dedupe_against_existing else []
    payload_embeddings: list[list[float]] = []
    duplicate_records = 0
    invalid_records = initial_invalid_records
    imported_examples: list[TrainingExample] = []

    for chunk in iter_chunks(normalized_rows, body.chunk_size):
        for row in chunk:
            instruction = str(row.get("instruction", "")).strip()
            response = str(row.get("response", "")).strip()
            if len(instruction) < 2 or len(response) < 2:
                invalid_records += 1
                continue
            embedding = get_embedding(instruction)
            embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
            if not isinstance(embedding_list, list) or not embedding_list:
                invalid_records += 1
                continue

            comparable_embeddings: list[list[float]] = []
            if body.dedupe_against_existing:
                comparable_embeddings.extend(
                    existing for existing in existing_embeddings if len(existing) == len(embedding_list)
                )
            if body.dedupe_within_payload:
                comparable_embeddings.extend(
                    existing for existing in payload_embeddings if len(existing) == len(embedding_list)
                )

            if comparable_embeddings and is_duplicate(
                embedding_list,
                comparable_embeddings,
                threshold=body.dedupe_threshold,
            ):
                duplicate_records += 1
                continue

            if body.dedupe_within_payload:
                payload_embeddings.append(embedding_list)
            if body.dedupe_against_existing:
                existing_embeddings.append(embedding_list)

            imported_examples.append(
                TrainingExample(
                    prompt=body.prompt,
                    instruction=instruction,
                    response=response,
                    embedding=json.dumps(embedding_list),
                )
            )

    return imported_examples, duplicate_records, invalid_records
