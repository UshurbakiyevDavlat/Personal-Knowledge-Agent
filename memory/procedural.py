"""
Procedural Memory — хранение и поиск именованных workflows.
Паттерн: таблица workflows с JSONB steps + vector embedding.
"""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def save_workflow(
    name: str,
    trigger: str,
    steps: list[dict],
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """
    Сохранить или обновить workflow.

    Returns:
        dict с 'action': 'created' | 'updated', 'id': int
    """
    from core.db import get_conn, get_cursor
    from core.embedder import embed_query

    embed_text = f"{name}. {trigger}. {description or ''}"
    embedding = embed_query(embed_text)
    tags_list = tags or []
    steps_json = json.dumps(steps, ensure_ascii=False)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute("SELECT id FROM workflows WHERE name = %s", (name,))
            existing = cur.fetchone()

            if existing:
                cur.execute(
                    """
                    UPDATE workflows
                    SET trigger = %s, description = %s, steps = %s::jsonb,
                        tags = %s, embedding = %s::vector
                    WHERE id = %s
                    """,
                    (trigger, description, steps_json, tags_list, embedding, existing["id"]),
                )
                return {"action": "updated", "id": existing["id"], "name": name}
            else:
                cur.execute(
                    """
                    INSERT INTO workflows (name, trigger, description, steps, tags, embedding)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s::vector)
                    RETURNING id
                    """,
                    (name, trigger, description, steps_json, tags_list, embedding),
                )
                row = cur.fetchone()
                return {"action": "created", "id": row["id"], "name": name}


def find_workflows(
    query: str,
    top_k: int = 3,
    tags: Optional[list[str]] = None,
) -> list[dict]:
    """
    Найти workflows по смыслу запроса.

    Returns:
        Список workflows, отсортированных по релевантности
    """
    from core.db import get_conn, get_cursor
    from core.embedder import embed_query

    query_embedding = embed_query(query)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            sql = """
                SELECT id, name, trigger, description, steps, tags,
                       run_count, last_used_at,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM workflows
                WHERE embedding IS NOT NULL
            """
            params: list = [query_embedding]

            if tags:
                sql += " AND tags && %s"
                params.append(tags)

            sql += " ORDER BY similarity DESC LIMIT %s"
            params.append(top_k)

            cur.execute(sql, params)
            rows = cur.fetchall()

    results = []
    for row in rows:
        steps = row["steps"] if isinstance(row["steps"], list) else json.loads(row["steps"])
        results.append({
            "id": row["id"],
            "name": row["name"],
            "trigger": row["trigger"],
            "description": row["description"],
            "steps": steps,
            "tags": row["tags"] or [],
            "run_count": row["run_count"],
            "similarity": round(row["similarity"], 3),
        })

    return results


def mark_workflow_used(workflow_id: int) -> None:
    """Инкрементировать run_count и обновить last_used_at."""
    from core.db import get_conn, get_cursor

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                "UPDATE workflows SET run_count = run_count + 1, last_used_at = NOW() WHERE id = %s",
                (workflow_id,),
            )


def list_all_workflows(tags: Optional[list[str]] = None) -> list[dict]:
    """Получить все workflows, опционально фильтр по тегам."""
    from core.db import get_conn, get_cursor

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            sql = """
                SELECT id, name, trigger, description, steps, tags, run_count, last_used_at
                FROM workflows
            """
            params: list = []
            if tags:
                sql += " WHERE tags && %s"
                params.append(tags)
            sql += " ORDER BY run_count DESC, created_at DESC"

            cur.execute(sql, params)
            rows = cur.fetchall()

    results = []
    for row in rows:
        steps = row["steps"] if isinstance(row["steps"], list) else json.loads(row["steps"])
        results.append({
            "id": row["id"],
            "name": row["name"],
            "trigger": row["trigger"],
            "description": row["description"],
            "steps_count": len(steps),
            "tags": row["tags"] or [],
            "run_count": row["run_count"],
            "last_used_at": row["last_used_at"].isoformat() if row["last_used_at"] else None,
        })

    return results
