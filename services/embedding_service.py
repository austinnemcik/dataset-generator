import json

from sqlmodel import Session, select

from agent import get_embedding, is_duplicate
from app.core.database import TrainingExample


def normalize_embedding_vector(raw_embedding) -> list[float] | None:
    if raw_embedding is None:
        return None
    vector = raw_embedding.tolist() if hasattr(raw_embedding, "tolist") else raw_embedding
    if not isinstance(vector, list) or not vector:
        return None
    try:
        return [float(value) for value in vector]
    except (TypeError, ValueError):
        return None


def embedding_json(raw_embedding) -> str | None:
    vector = normalize_embedding_vector(raw_embedding)
    if vector is None:
        return None
    return json.dumps(vector)


def parse_embedding_json(raw_embedding: str | None) -> list[float] | None:
    if not raw_embedding:
        return None
    try:
        return normalize_embedding_vector(json.loads(raw_embedding))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def embed_text(text: str) -> list[float] | None:
    return normalize_embedding_vector(get_embedding(text))


def comparable_embeddings(
    embedding: list[float] | None,
    existing_embeddings: list[list[float]],
) -> list[list[float]]:
    if embedding is None:
        return []
    return [existing for existing in existing_embeddings if len(existing) == len(embedding)]


def is_semantic_duplicate(
    embedding: list[float] | None,
    existing_embeddings: list[list[float]],
    *,
    threshold: float = 0.8,
) -> bool:
    if embedding is None:
        return False
    comparable = comparable_embeddings(embedding, existing_embeddings)
    if not comparable:
        return False
    return is_duplicate(embedding, comparable, threshold=threshold)


def load_training_example_embeddings(session: Session) -> list[list[float]]:
    embeddings: list[list[float]] = []
    rows = session.exec(
        select(TrainingExample.embedding).where(TrainingExample.embedding.is_not(None))
    ).all()
    for raw_embedding in rows:
        parsed = parse_embedding_json(raw_embedding)
        if parsed is not None:
            embeddings.append(parsed)
    return embeddings


