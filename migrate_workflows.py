"""
Миграция: создаёт таблицу workflows для Procedural Memory.

Запуск: python migrate_workflows.py
Безопасно запускать повторно (идемпотентно).
"""
from core.db import get_conn, get_cursor

STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS workflows (
        id              SERIAL          PRIMARY KEY,
        name            VARCHAR(200)    NOT NULL UNIQUE,
        trigger         TEXT            NOT NULL,
        description     TEXT,
        steps           JSONB           NOT NULL,
        tags            TEXT[]          DEFAULT '{}',
        embedding       vector(1024),
        run_count       INTEGER         DEFAULT 0,
        last_used_at    TIMESTAMPTZ,
        created_at      TIMESTAMPTZ     DEFAULT NOW(),
        updated_at      TIMESTAMPTZ     DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS workflows_embedding_idx
        ON workflows USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """,
    """
    CREATE INDEX IF NOT EXISTS workflows_name_idx
        ON workflows (name)
    """,
    """
    CREATE OR REPLACE FUNCTION update_workflows_updated_at()
    RETURNS TRIGGER AS $$
    BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
    $$ LANGUAGE plpgsql
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_trigger WHERE tgname = 'workflows_updated_at_trigger'
        ) THEN
            EXECUTE 'CREATE TRIGGER workflows_updated_at_trigger
                BEFORE UPDATE ON workflows
                FOR EACH ROW EXECUTE FUNCTION update_workflows_updated_at()';
        END IF;
    END$$
    """,
]


def migrate():
    print("🔄 Применяем миграцию workflows...")

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            for stmt in STATEMENTS:
                cur.execute(stmt.strip())

    print("✅ Таблица workflows создана")


if __name__ == "__main__":
    migrate()
