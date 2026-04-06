"""
Embedder — генерирует векторы через Voyage AI.

Модель: voyage-3 (рекомендована Anthropic для использования с Claude)
Размерность: 1024
Цена: ~$0.06 / 1M токенов (200M бесплатно на старте)

Документация: https://docs.voyageai.com/docs/embeddings
"""
import logging

import voyageai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import config

logger = logging.getLogger(__name__)

_client: voyageai.Client | None = None


def get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    return _client


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _embed_batch(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """
    Получить эмбеддинги для батча текстов.

    input_type:
        "document" — для индексируемых чанков
        "query"    — для поисковых запросов (важно для качества поиска!)
    """
    result = get_client().embed(
        texts=texts,
        model=config.EMBEDDING_MODEL,
        input_type=input_type,
    )
    return result.embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Получить эмбеддинги для списка документов (чанков).
    Автоматически разбивает на батчи по EMBEDDING_BATCH_SIZE.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    batch_size = config.EMBEDDING_BATCH_SIZE

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.debug(f"Embedding batch {i // batch_size + 1}: {len(batch)} texts")
        embeddings = _embed_batch(batch, input_type="document")
        all_embeddings.extend(embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Получить эмбеддинг для поискового запроса.
    Использует input_type="query" — это важно для качества поиска в Voyage AI.
    """
    return _embed_batch([query], input_type="query")[0]
