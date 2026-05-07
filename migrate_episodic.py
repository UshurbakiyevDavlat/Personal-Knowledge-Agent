"""
Миграция: создаёт таблицы episodic_events, kg_entities, kg_relations, community_summaries
и переносит существующие user_facts в episodic_events.

Запуск: python migrate_episodic.py
Безопасно запускать повторно (идемпотентно).
"""
from core.db import get_conn, get_cursor

MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS episodic_events (
    id          SERIAL PRIMARY KEY,
    fact_key    VARCHAR(200)    NOT NULL,
    fact_value  TEXT            NOT NULL,
    category    VARCHAR(100)    NOT NULL DEFAULT 'general',
    confidence  REAL            DEFAULT 1.0,
    valid_from  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    valid_to    TIMESTAMPTZ,
    invalid_at  TIMESTAMPTZ,
    source      VARCHAR(100)    DEFAULT 'manual',
    context     TEXT,
    created_at  TIMESTAMPTZ     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS episodic_events_active_idx
    ON episodic_events (fact_key, valid_from)
    WHERE valid_to IS NULL AND invalid_at IS NULL;

CREATE INDEX IF NOT EXISTS episodic_events_key_idx
    ON episodic_events (fact_key, valid_from DESC);

CREATE TABLE IF NOT EXISTS kg_entities (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(300)    NOT NULL,
    type        VARCHAR(100)    NOT NULL DEFAULT 'concept',
    description TEXT,
    doc_id      VARCHAR(500),
    created_at  TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS kg_relations (
    id          SERIAL PRIMARY KEY,
    source_name VARCHAR(300)    NOT NULL,
    relation    VARCHAR(100)    NOT NULL,
    target_name VARCHAR(300)    NOT NULL,
    doc_id      VARCHAR(500),
    created_at  TIMESTAMPTZ     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS kg_relations_source_idx ON kg_relations (source_name);
CREATE INDEX IF NOT EXISTS kg_relations_target_idx ON kg_relations (target_name);

CREATE TABLE IF NOT EXISTS community_summaries (
    id              SERIAL PRIMARY KEY,
    community_id    VARCHAR(100)    NOT NULL UNIQUE,
    title           TEXT            NOT NULL,
    summary         TEXT            NOT NULL,
    entity_count    INTEGER         DEFAULT 0,
    embedding       vector(1024),
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     DEFAULT NOW()
);
"""


def migrate():
    print("🔄 Применяем миграцию...")

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # Создаём таблицы
            for stmt in MIGRATION_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)

            # Мигрируем существующие user_facts → episodic_events
            cur.execute("SELECT key, value, category, confidence, created_at FROM user_facts")
            facts = cur.fetchall()

            migrated = 0
            for fact in facts:
                cur.execute(
                    """
                    INSERT INTO episodic_events (fact_key, fact_value, category, confidence, valid_from, source)
                    VALUES (%s, %s, %s, %s, %s, 'migrated')
                    ON CONFLICT DO NOTHING
                    """,
                    (fact["key"], fact["value"], fact["category"], fact["confidence"], fact["created_at"]),
                )
                if cur.rowcount:
                    migrated += 1

    print(f"✅ Таблицы созданы")
    print(f"✅ Мигрировано {migrated} фактов из user_facts в episodic_events")


if __name__ == "__main__":
    migrate()
