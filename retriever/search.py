"""
Search — гибридный поиск по базе знаний.

Два режима:
1. Vector search — косинусное сходство через pgvector (семантический поиск)
2. Hybrid search — vector + full-text BM25 (лучше для точных совпадений: имена, термины, код)

Гибридный поиск использует Reciprocal Rank Fusion (RRF) для объединения результатов.
"""
import logging
from dataclasses import dataclass
from typing import Any

from core.db import get_conn, get_cursor
from core.embedder import embed_query

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    id: str
    title: str
    content: str          # текст чанка
    source_type: str      # 'notion' | 'url' | 'file' | 'manual'
    source_url: str | None
    chunk_index: int
    chunk_total: int
    score: float          # итоговый score (выше = релевантнее)
    metadata: dict


def _vector_search(
    query_embedding: list[float],
    top_k: int,
    source_filter: str | None,
    conn,
) -> list[dict]:
    """Поиск по векторному сходству."""
    sql = """
        SELECT
            id::text,
            title,
            content,
            source_type,
            source_url,
            chunk_index,
            chunk_total,
            metadata,
            1 - (embedding <=> %s::vector) AS score
        FROM documents
        WHERE embedding IS NOT NULL
        {source_filter}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """.format(
        source_filter="AND source_type = %s" if source_filter else ""
    )

    params: list[Any] = [query_embedding]
    if source_filter:
        params.append(source_filter)
    params.extend([query_embedding, top_k * 2])  # берём больше для RRF

    with get_cursor(conn) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _fulltext_search(
    query: str,
    top_k: int,
    source_filter: str | None,
    conn,
) -> list[dict]:
    """Full-text поиск (русский + английский)."""
    sql = """
        SELECT
            id::text,
            title,
            content,
            source_type,
            source_url,
            chunk_index,
            chunk_total,
            metadata,
            ts_rank(
                to_tsvector('russian', content) || to_tsvector('english', content),
                websearch_to_tsquery('russian', %s) || websearch_to_tsquery('english', %s)
            ) AS score
        FROM documents
        WHERE
            (
                to_tsvector('russian', content) @@ websearch_to_tsquery('russian', %s)
                OR
                to_tsvector('english', content) @@ websearch_to_tsquery('english', %s)
            )
            {source_filter}
        ORDER BY score DESC
        LIMIT %s
    """.format(
        source_filter="AND source_type = %s" if source_filter else ""
    )

    params: list[Any] = [query, query, query, query]
    if source_filter:
        params.append(source_filter)
    params.append(top_k * 2)

    with get_cursor(conn) as cur:
        try:
            cur.execute(sql, params)
            return cur.fetchall()
        except Exception as e:
            logger.warning(f"Full-text search failed (query may be too short): {e}")
            return []


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    fts_results: list[dict],
    k: int = 60,  # константа RRF (обычно 60)
    vector_weight: float = 0.7,
    fts_weight: float = 0.3,
) -> list[tuple[dict, float]]:
    """
    Reciprocal Rank Fusion — объединяет два списка результатов в один.
    Score = vector_weight * (1 / (k + rank_v)) + fts_weight * (1 / (k + rank_f))
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    # Ранжируем векторные результаты
    for rank, doc in enumerate(vector_results):
        doc_id = doc["id"]
        docs[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank + 1)

    # Ранжируем full-text результаты
    for rank, doc in enumerate(fts_results):
        doc_id = doc["id"]
        docs[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0) + fts_weight / (k + rank + 1)

    # Сортируем по убыванию score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(docs[doc_id], score) for doc_id, score in ranked]


def search(
    query: str,
    top_k: int = 5,
    source_filter: str | None = None,
    hybrid: bool = True,
) -> list[SearchResult]:
    """
    Основная функция поиска.

    Args:
        query: Поисковый запрос на любом языке
        top_k: Сколько результатов вернуть
        source_filter: Фильтр по типу источника ('notion', 'url', 'file', 'manual')
        hybrid: True = vector + full-text, False = только vector

    Returns:
        Список SearchResult, отсортированных по релевантности
    """
    if not query.strip():
        return []

    logger.info(f"Search: '{query}' top_k={top_k} source={source_filter} hybrid={hybrid}")

    # Эмбеддинг запроса
    query_embedding = embed_query(query)

    with get_conn() as conn:
        vector_results = _vector_search(query_embedding, top_k, source_filter, conn)

        if hybrid and len(query.split()) >= 2:
            fts_results = _fulltext_search(query, top_k, source_filter, conn)
            ranked = _reciprocal_rank_fusion(vector_results, fts_results)
        else:
            ranked = [(doc, doc["score"]) for doc in vector_results]

    # Берём топ-K и конвертируем в SearchResult
    results = []
    for doc, score in ranked[:top_k]:
        results.append(SearchResult(
            id=doc["id"],
            title=doc["title"] or "Untitled",
            content=doc["content"],
            source_type=doc["source_type"],
            source_url=doc.get("source_url"),
            chunk_index=doc["chunk_index"],
            chunk_total=doc["chunk_total"],
            score=round(score, 4),
            metadata=doc["metadata"] or {},
        ))

    logger.info(f"Found {len(results)} results")
    return results


def format_results_for_claude(results: list[SearchResult]) -> str:
    """
    Форматировать результаты поиска для передачи в контекст Claude.
    Возвращает строку с пронумерованными чанками и attribution.
    """
    if not results:
        return "Ничего не найдено в базе знаний."

    parts = []
    for i, r in enumerate(results, 1):
        source_info = ""
        if r.source_url:
            source_info = f" ([источник]({r.source_url}))"

        chunk_info = ""
        if r.chunk_total > 1:
            chunk_info = f" [часть {r.chunk_index + 1}/{r.chunk_total}]"

        parts.append(
            f"**[{i}] {r.title}**{chunk_info}{source_info}\n"
            f"```\n{r.content}\n```"
        )

    return "\n\n".join(parts)


if __name__ == "__main__":
    # Быстрый тест: python -m retriever.search "твой запрос"
    import sys
    logging.basicConfig(level=logging.INFO)

    query = " ".join(sys.argv[1:]) or "как работает auth"
    results = search(query, top_k=5)

    print(f"\n=== Results for: '{query}' ===\n")
    print(format_results_for_claude(results))
