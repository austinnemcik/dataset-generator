import numpy as np
from app.core.generics import TimedLabel, timer
from sentence_transformers import SentenceTransformer
import app.core.logger as logger

from .settings import get_client_settings

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.log_event(
            "embedding_model.load_start",
            level="INFO",
            model_name="all-MiniLM-L6-v2",
            device="cuda",
        )
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        logger.log_event(
            "embedding_model.load_complete",
            level="INFO",
            model_name="all-MiniLM-L6-v2",
            device="cuda",
        )
    return _model

def get_embedding(text: str):
    model = get_embedding_model()
    with timer(TimedLabel.EMBEDDING_CALL):
        embedding = model.encode(text)
    return embedding


def cosine_similarity(a: list[float], b: list[float]):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def is_duplicate(
    new_embedding: list[float],
    existing_embeddings: list[list[float]],
    threshold: float | None = None,
):
    threshold = threshold if threshold is not None else get_client_settings().threshold
    for existing in existing_embeddings:
        if cosine_similarity(new_embedding, existing) >= threshold:
            logger.saveToLog("Discarding duplicate example", "INFO")
            return True
    return False


