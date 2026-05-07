"""
LightRAG-style global retrieval — поиск по community summaries.
"""
import logging
from dataclasses import dataclass

from core.db import get_conn, get_cursor
from core.embedder import embed_query

logger = logging.getLogger(__name__)


@dataclass
class CommunityResult:
    community_id: str
    title: str
    summary: str
    score: float


def search_global(query: str, top_k: int = 3) -> list[CommunityResult]:
    """
    Global retrieval — поиск по community summaries через vector similarity.
    Для обобщающих вопросов: 'какова архитектура?', 'что мы знаем о X?'
    """
    query_embedding = embed_query(query)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                """
                SELECT
                    community_id,
                    title,
                    summary,
                    1 - (embedding <=> %s::vector) AS score
                FROM community_summaries
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()

    return [
        CommunityResult(
            community_id=row["community_id"],
            title=row["title"],
            summary=row["summary"],
            score=round(row["score"], 4),
        )
        for row in rows
    ]
