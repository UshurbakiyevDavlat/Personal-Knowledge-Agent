"""
Graph Builder — читает documents из PostgreSQL и строит граф на основе
cosine similarity между embedding векторами.

Ноды: документы, wiki-файлы, user_facts
Рёбра: cosine_similarity > EDGE_THRESHOLD через pgvector LATERAL join (HNSW-ускоренный)
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

EDGE_THRESHOLD = 0.75
MAX_NODES = 500
MAX_EDGES_PER_NODE = 5

COLOR_MAP = {
    "notion": "#5B8AF5",
    "url": "#F5A623",
    "file": "#7ED321",
    "wiki": "#9013FE",
    "manual": "#D0021B",
    "fact": "#FF6B6B",
}


def build_graph_data(source_filter: Optional[str] = None, limit: int = MAX_NODES) -> dict:
    """
    Построить данные графа для визуализации.

    Returns:
        dict с 'nodes': list, 'edges': list, 'stats': dict
    """
    from core.db import get_conn, get_cursor

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # Один представитель на source_id (chunk_index = 0)
            base_query = """
                SELECT DISTINCT ON (source_type, source_id)
                    id::text AS id,
                    source_type,
                    source_id,
                    source_url,
                    title,
                    chunk_total
                FROM documents
                WHERE chunk_index = 0
                  AND embedding IS NOT NULL
            """
            params: list = []
            if source_filter:
                base_query += " AND source_type = %s"
                params.append(source_filter)
            base_query += " LIMIT %s"
            params.append(limit)

            cur.execute(base_query, params)
            docs = cur.fetchall()

            nodes = []
            doc_ids: list[str] = []

            for doc in docs:
                nodes.append({
                    "id": doc["id"],
                    "label": (doc["title"] or "Untitled")[:50],
                    "source_type": doc["source_type"],
                    "source_url": doc["source_url"],
                    "color": COLOR_MAP.get(doc["source_type"], "#AAAAAA"),
                    "size": min(10 + (doc["chunk_total"] or 1) * 2, 30),
                })
                doc_ids.append(doc["id"])

            # user_facts как отдельные ноды
            cur.execute("SELECT key, value, category FROM user_facts LIMIT 50")
            for fact in cur.fetchall():
                nodes.append({
                    "id": f"fact_{fact['key']}",
                    "label": f"📌 {fact['key']}: {fact['value'][:30]}",
                    "source_type": "fact",
                    "source_url": None,
                    "color": COLOR_MAP["fact"],
                    "size": 12,
                })

            edges = []
            if len(doc_ids) > 1:
                cur.execute(
                    """
                    WITH base AS (
                        SELECT id, embedding
                        FROM documents
                        WHERE id::text = ANY(%s) AND chunk_index = 0
                    )
                    SELECT DISTINCT
                        LEAST(b.id::text, nn.id::text)    AS source,
                        GREATEST(b.id::text, nn.id::text) AS target,
                        1 - (b.embedding <=> nn.embedding) AS similarity
                    FROM base b
                    JOIN LATERAL (
                        SELECT id, embedding
                        FROM documents d_inner
                        WHERE chunk_index = 0
                          AND d_inner.id != b.id
                        ORDER BY b.embedding <=> d_inner.embedding
                        LIMIT %s
                    ) nn ON true
                    WHERE 1 - (b.embedding <=> nn.embedding) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (doc_ids, MAX_EDGES_PER_NODE, EDGE_THRESHOLD, len(doc_ids) * MAX_EDGES_PER_NODE),
                )
                for row in cur.fetchall():
                    edges.append({
                        "source": row["source"],
                        "target": row["target"],
                        "weight": round(row["similarity"], 3),
                    })

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "sources": list({n["source_type"] for n in nodes}),
        },
    }
