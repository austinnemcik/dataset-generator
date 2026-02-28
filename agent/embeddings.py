import numpy as np
from generics import TimedLabel, timer
import torch
from sentence_transformers import SentenceTransformer
import logger

from .settings import THRESHOLD, client

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

def get_embedding(text: str):
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
    threshold: float = THRESHOLD,
):
    for existing in existing_embeddings:
        if cosine_similarity(new_embedding, existing) >= threshold:
            logger.saveToLog("Discarding duplicate example", "INFO")
            return True
    return False