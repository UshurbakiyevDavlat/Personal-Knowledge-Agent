-- ============================================================
-- Personal Knowledge Agent — Database Schema
-- База: knowledge_agent (отдельная, не связана с проектами)
-- ============================================================

-- 1. Создать базу (выполни от имени суперюзера):
--    CREATE DATABASE knowledge_agent;
--    \c knowledge_agent

-- 2. Установить расширение
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- Основная таблица — все проиндексированные чанки
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Источник
    source_type VARCHAR(50)   NOT NULL,   -- 'notion' | 'url' | 'file' | 'manual'
    source_id   VARCHAR(500),             -- notion page id / file path / url
    source_url  VARCHAR(2000),            -- ссылка на оригинал (для attribution)
    title       VARCHAR(500),

    -- Контент
    content     TEXT          NOT NULL,   -- текст чанка (500 токенов ±)
    chunk_index INTEGER       NOT NULL,   -- порядковый номер чанка внутри документа
    chunk_total INTEGER       NOT NULL,   -- сколько всего чанков в документе

    -- Вектор
    embedding   vector(1024),             -- Voyage AI voyage-3

    -- Метаданные (любые доп. поля: язык, теги, breadcrumb и т.д.)
    metadata    JSONB         DEFAULT '{}',

    -- Временны́е метки
    indexed_at  TIMESTAMPTZ   DEFAULT NOW(),
    updated_at  TIMESTAMPTZ   DEFAULT NOW()
);

-- ============================================================
-- Индексы
-- ============================================================

-- HNSW — быстрый приближённый поиск по косинусному сходству
-- (лучше IVFFlat для малых/средних коллекций, не требует предварительного обучения)
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search — для гибридного поиска (точные совпадения, имена, термины)
CREATE INDEX IF NOT EXISTS documents_fts_ru_idx
    ON documents USING gin(to_tsvector('russian', content));

CREATE INDEX IF NOT EXISTS documents_fts_en_idx
    ON documents USING gin(to_tsvector('english', content));

-- Для инкрементальной индексации (проверка "что изменилось")
CREATE INDEX IF NOT EXISTS documents_source_idx
    ON documents (source_type, source_id);

CREATE INDEX IF NOT EXISTS documents_updated_idx
    ON documents (updated_at DESC);

-- ============================================================
-- Эпизодическая память — факты о пользователе
-- ============================================================
CREATE TABLE IF NOT EXISTS user_facts (
    id          SERIAL PRIMARY KEY,
    category    VARCHAR(100)  NOT NULL,  -- 'preference' | 'skill' | 'project' | 'personal'
    key         VARCHAR(200)  NOT NULL UNIQUE,
    value       TEXT          NOT NULL,
    confidence  REAL          DEFAULT 1.0,
    created_at  TIMESTAMPTZ   DEFAULT NOW(),
    updated_at  TIMESTAMPTZ   DEFAULT NOW()
);

-- ============================================================
-- Лог индексации — отслеживание что и когда индексировалось
-- ============================================================
CREATE TABLE IF NOT EXISTS index_log (
    id          SERIAL PRIMARY KEY,
    source_type VARCHAR(50)   NOT NULL,
    source_id   VARCHAR(500)  NOT NULL,
    status      VARCHAR(20)   NOT NULL,  -- 'success' | 'failed' | 'skipped'
    chunks_count INTEGER      DEFAULT 0,
    error_msg   TEXT,
    indexed_at  TIMESTAMPTZ   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS index_log_source_idx
    ON index_log (source_type, source_id, indexed_at DESC);

-- ============================================================
-- Полезные функции
-- ============================================================

-- Обновлять updated_at автоматически
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER user_facts_updated_at
    BEFORE UPDATE ON user_facts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- Temporal Episodic Memory (Graphiti pattern)
-- Факты никогда не удаляются — помечаются invalid_at
-- ============================================================
CREATE TABLE IF NOT EXISTS episodic_events (
    id          SERIAL PRIMARY KEY,
    fact_key    VARCHAR(200)    NOT NULL,
    fact_value  TEXT            NOT NULL,
    category    VARCHAR(100)    NOT NULL DEFAULT 'general',
    confidence  REAL            DEFAULT 1.0,

    -- Bi-temporal timestamps
    valid_from  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    valid_to    TIMESTAMPTZ,                               -- NULL = сейчас актуально
    invalid_at  TIMESTAMPTZ,                               -- когда узнали что устарел

    source      VARCHAR(100)    DEFAULT 'manual',
    context     TEXT,

    created_at  TIMESTAMPTZ     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS episodic_events_active_idx
    ON episodic_events (fact_key, valid_from)
    WHERE valid_to IS NULL AND invalid_at IS NULL;

CREATE INDEX IF NOT EXISTS episodic_events_key_idx
    ON episodic_events (fact_key, valid_from DESC);

-- ============================================================
-- Knowledge Graph — plain tables (вместо Apache AGE)
-- ============================================================
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

-- Community summaries — LLM-генерированные обобщения по кластерам сущностей
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

CREATE INDEX IF NOT EXISTS community_summaries_embedding_idx
    ON community_summaries USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TRIGGER community_summaries_updated_at
    BEFORE UPDATE ON community_summaries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- Procedural Memory — именованные workflows (как делать X)
-- ============================================================
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
);

CREATE INDEX IF NOT EXISTS workflows_embedding_idx
    ON workflows USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS workflows_name_idx
    ON workflows (name);

CREATE OR REPLACE FUNCTION update_workflows_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER workflows_updated_at_trigger
    BEFORE UPDATE ON workflows
    FOR EACH ROW EXECUTE FUNCTION update_workflows_updated_at();
