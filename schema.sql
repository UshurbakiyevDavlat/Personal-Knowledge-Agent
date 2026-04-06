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
