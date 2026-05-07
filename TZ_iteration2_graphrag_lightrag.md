═══════════════════════════════════════════
ТЗ: Итерация 2 — GraphRAG Layer (LightRAG pattern + plain PostgreSQL)
Проект: Personal-Knowledge-Agent · Python 3.12 + PostgreSQL 17
Дата: 2026-05-07
═══════════════════════════════════════════

## Цель
Добавить граф знаний поверх существующей vector БД (без замены) — извлечь entities+relations из документов при индексации, хранить в обычных таблицах PostgreSQL, добавить community summaries и dual-level retrieval по паттерну LightRAG.

## Контекст
Сейчас: vector search (semantic) + FTS (keyword) → RRF. Проблема: на "глобальных" вопросах ("какова общая архитектура Adash?") vector search возвращает разрозненные чанки без synthesis. GraphRAG/LightRAG решает это через: (1) entity extraction при индексации, (2) community detection (Louvain), (3) community summaries — LLM-сгенерированные обобщения по каждому кластеру, (4) dual-level retrieval — local (entity-level) + global (community-level).

**Решение по хранению**: Apache AGE НЕ установлен и недоступен в этом Postgres (проверено: только plpgsql + vector 0.8.2). Вместо AGE используем plain PostgreSQL таблицы `kg_entities` и `kg_relations` (adjacency list). Для community detection — NetworkX + python-louvain в Python. Никакого Cypher, никакого дополнительного сервиса. Функционально идентично AGE для наших задач, архитектурно проще.

## Предусловия перед началом
1. `pip install anthropic networkx python-louvain` (AGE НЕ нужен)
2. Запустить `python -c "import schema; schema.init_db()"` или добавить новые таблицы через migrate

## Файлы

**Создать:**
- `graph_kg/extractor.py` — LLM-based entity+relation extraction
- `graph_kg/communities.py` — Leiden community detection + summary generation
- `graph_kg/retriever.py` — dual-level retrieval (local + global)
- `graph_kg/__init__.py` — пустой

**Трогать:**
- `schema.sql` — добавить таблицы kg_entities, kg_relations, community_summaries
- `indexer/notion_indexer.py` — вызывать entity extraction после сохранения чанков
- `retriever/search.py` — добавить `dual_level_search()` рядом с `search()`
- `agent_server/server.py` — добавить `kb_search_global()` для community-level поиска
- `config.py` — добавить конфигурацию KG

**Не трогать:**
- `core/embedder.py`, `core/chunker.py`, `core/db.py` — не трогаем
- `memory/episodic.py` — не трогаем

## Реализация

### 1. schema.sql — plain PostgreSQL KG tables + community_summaries

Добавить в конец (AGE не нужен — используем обычные таблицы):
```sql
-- Knowledge Graph: сущности (entities)
CREATE TABLE IF NOT EXISTS kg_entities (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(300) NOT NULL,
    type        VARCHAR(100) NOT NULL DEFAULT 'concept',
    description TEXT,
    doc_ids     TEXT[]          DEFAULT '{}',   -- из каких документов извлечена
    created_at  TIMESTAMPTZ     DEFAULT NOW(),
    updated_at  TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE(name, type)
);

-- Knowledge Graph: связи (relations)
CREATE TABLE IF NOT EXISTS kg_relations (
    id          SERIAL PRIMARY KEY,
    source_id   INTEGER     NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    target_id   INTEGER     NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    relation    VARCHAR(100) NOT NULL,  -- uses, implements, depends_on, etc.
    weight      REAL        DEFAULT 1.0,
    doc_id      TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS kg_entities_name_idx ON kg_entities (name);
CREATE INDEX IF NOT EXISTS kg_relations_source_idx ON kg_relations (source_id);
CREATE INDEX IF NOT EXISTS kg_relations_target_idx ON kg_relations (target_id);

-- Community summaries — LLM-генерированные обобщения кластеров
CREATE TABLE IF NOT EXISTS community_summaries (
    id              SERIAL PRIMARY KEY,
    community_id    VARCHAR(100) NOT NULL UNIQUE,
    title           TEXT NOT NULL,          -- краткое название темы
    summary         TEXT NOT NULL,          -- LLM summary
    entity_count    INTEGER DEFAULT 0,
    doc_ids         TEXT[],                 -- UUID документов в кластере
    embedding       vector(1024),           -- эмбеддинг summary для vector search
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS community_summaries_embedding_idx
    ON community_summaries USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

### 2. graph_kg/extractor.py — entity+relation extraction

```python
"""
Entity and relation extraction using Claude Haiku.
Вызывается при индексации новых документов.
"""
import json
import logging
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Из текста ниже извлеки все важные сущности и связи между ними.

Сущности (entities): люди, проекты, технологии, концепции, организации, методы.
Связи (relations): uses, implements, depends_on, created_by, part_of, related_to, contrasts_with.

Верни ТОЛЬКО JSON без markdown:
{
  "entities": [
    {"name": "FastAPI", "type": "technology", "description": "Python web framework"}
  ],
  "relations": [
    {"source": "AdashAI", "relation": "uses", "target": "FastAPI"}
  ]
}

Текст:
{text}"""


@dataclass
class Entity:
    name: str
    type: str
    description: str


@dataclass  
class Relation:
    source: str
    relation: str
    target: str


def extract_entities_relations(text: str, title: str) -> tuple[list[Entity], list[Relation]]:
    """
    Извлечь сущности и связи из текста через Claude Haiku.
    
    Returns:
        (entities, relations) — пустые списки при ошибке
    """
    if len(text) < 100:
        return [], []
    
    # Берём первые 2000 символов (Haiku fast+cheap)
    truncated = text[:2000]
    
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": EXTRACTION_PROMPT.format(text=truncated)
            }]
        )
        
        raw = response.content[0].text.strip()
        # Убрать markdown если есть
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        
        data = json.loads(raw)
        
        entities = [
            Entity(name=e["name"], type=e.get("type", "concept"), description=e.get("description", ""))
            for e in data.get("entities", [])
        ]
        relations = [
            Relation(source=r["source"], relation=r["relation"], target=r["target"])
            for r in data.get("relations", [])
            if r.get("source") and r.get("target")
        ]
        
        logger.info(f"Extracted {len(entities)} entities, {len(relations)} relations from '{title}'")
        return entities, relations
        
    except Exception as e:
        logger.warning(f"Entity extraction failed for '{title}': {e}")
        return [], []


def save_to_pg(entities: list[Entity], relations: list[Relation], doc_id: str, conn) -> None:
    """
    Сохранить entities и relations в plain PostgreSQL таблицы.
    Upsert по (name, type) — безопасно вызывать многократно.
    """
    if not entities:
        return
    
    with conn.cursor() as cur:
        # Upsert entities, собираем id-маппинг
        entity_ids: dict[str, int] = {}
        for entity in entities:
            cur.execute("""
                INSERT INTO kg_entities (name, type, description, doc_ids)
                VALUES (%s, %s, %s, ARRAY[%s])
                ON CONFLICT (name, type) DO UPDATE
                    SET description = COALESCE(EXCLUDED.description, kg_entities.description),
                        doc_ids = array_append(kg_entities.doc_ids, %s),
                        updated_at = NOW()
                RETURNING id
            """, (entity.name, entity.type, entity.description, doc_id, doc_id))
            row = cur.fetchone()
            entity_ids[entity.name] = row[0]
        
        # Insert relations (пропускаем если source/target не в entity_ids)
        for rel in relations:
            src_id = entity_ids.get(rel.source)
            tgt_id = entity_ids.get(rel.target)
            if src_id is None or tgt_id is None:
                continue
            cur.execute("""
                INSERT INTO kg_relations (source_id, target_id, relation, doc_id)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (src_id, tgt_id, rel.relation, doc_id))
```

### 3. graph_kg/communities.py — community detection + summaries

```python
"""
Community detection на KG + LLM summary generation.
Запускается периодически (не при каждой индексации).
"""
import json
import logging
from collections import defaultdict

import anthropic
import networkx as nx
from community import best_partition  # python-louvain

from core.db import get_conn, get_cursor
from core.embedder import embed_texts

logger = logging.getLogger(__name__)

COMMUNITY_SUMMARY_PROMPT = """Ты эксперт по анализу знаний. Ниже список связанных концептов/сущностей из одной тематической группы.

Сущности:
{entities}

Напиши краткое (2-3 предложения) резюме того, какая тема или область знаний объединяет эти концепты. 
Начни с конкретного названия темы.
"""


def build_networkx_from_pg(conn) -> nx.Graph:
    """Загрузить граф из plain PostgreSQL таблиц в NetworkX для community detection."""
    G = nx.Graph()
    
    with conn.cursor() as cur:
        # Вершины
        cur.execute("SELECT id, name FROM kg_entities")
        for row in cur.fetchall():
            G.add_node(row[0], name=row[1])
        
        # Рёбра
        cur.execute("SELECT source_id, target_id, weight FROM kg_relations")
        for row in cur.fetchall():
            G.add_edge(row[0], row[1], weight=row[2])
    
    return G


def generate_community_summaries() -> int:
    """
    Построить community detection, сгенерировать LLM summaries, сохранить в БД.
    
    Returns:
        Количество созданных/обновлённых communities
    """
    client = anthropic.Anthropic()
    
    with get_conn() as conn:
        G = build_networkx_from_pg(conn)
        
        if G.number_of_nodes() < 5:
            logger.info("Граф слишком мал для community detection")
            return 0
        
        # Louvain community detection
        partition = best_partition(G)  # {node: community_id}
        
        # Группируем узлы по community
        communities: dict[int, list[str]] = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        # Фильтруем маленькие (< 3 узла)
        communities = {k: v for k, v in communities.items() if len(v) >= 3}
        
        summaries_created = 0
        for comm_id, entities in communities.items():
            entities_text = "\n".join(f"- {e}" for e in entities[:30])
            
            # Генерируем summary через Claude Haiku
            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=256,
                    messages=[{
                        "role": "user", 
                        "content": COMMUNITY_SUMMARY_PROMPT.format(entities=entities_text)
                    }]
                )
                summary_text = response.content[0].text.strip()
                title = summary_text.split(".")[0][:100]
                
            except Exception as e:
                logger.warning(f"Summary generation failed for community {comm_id}: {e}")
                title = f"Тема {comm_id}"
                summary_text = f"Кластер из {len(entities)} концептов"
            
            # Эмбеддинг summary для vector search
            embedding = embed_texts([summary_text])[0]
            
            with get_cursor(conn) as cur:
                cur.execute("""
                    INSERT INTO community_summaries 
                        (community_id, title, summary, entity_count, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (community_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        summary = EXCLUDED.summary,
                        entity_count = EXCLUDED.entity_count,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                """, (str(comm_id), title, summary_text, len(entities), embedding))
            
            summaries_created += 1
        
        logger.info(f"Generated {summaries_created} community summaries")
        return summaries_created
```

### 4. graph_kg/retriever.py — dual-level retrieval

```python
"""
LightRAG-style dual-level retrieval:
- Local: entity-level (существующий vector+graph search)
- Global: community-level (поиск по community summaries)
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
    Global retrieval — поиск по community summaries.
    Для обобщающих вопросов ("какова архитектура?", "что мы знаем о X?").
    """
    query_embedding = embed_query(query)
    
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                SELECT
                    community_id,
                    title,
                    summary,
                    1 - (embedding <=> %s::vector) AS score
                FROM community_summaries
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            
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
```

### 5. agent_server/server.py — добавить `kb_search_global`

```python
from graph_kg.retriever import search_global

@mcp.tool(annotations={"readOnlyHint": True})
def kb_search_global(query: str, top_k: int = 3) -> str:
    """
    Глобальный поиск по тематическим кластерам базы знаний.
    Используй для обобщающих вопросов: 'какова архитектура системы', 
    'что мы знаем о безопасности', 'обзор всего по теме X'.
    
    В отличие от kb_search (поиск точных фактов), kb_search_global 
    возвращает синтезированные обобщения по тематическим кластерам.

    Args:
        query: Тема или вопрос
        top_k: Количество кластеров (1-5)

    Returns:
        Обобщения по релевантным тематическим кластерам
    """
    results = search_global(query=query, top_k=min(top_k, 5))
    
    if not results:
        return "Тематические кластеры ещё не построены. Запусти kb_rebuild_communities."
    
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"**[Кластер {i}: {r.title}]** (релевантность: {r.score})\n{r.summary}")
    
    return "\n\n".join(parts)


@mcp.tool(annotations={"readOnlyHint": False})
def kb_rebuild_communities() -> str:
    """
    Перестроить тематические кластеры (community summaries) из графа знаний.
    Запускай после значительного обновления базы знаний (раз в неделю).
    
    Returns:
        Количество созданных кластеров
    """
    from graph_kg.communities import generate_community_summaries
    count = generate_community_summaries()
    return f"✅ Построено {count} тематических кластеров"
```

### 6. indexer/notion_indexer.py — вызов entity extraction

В функции `index_page`, после успешного вызова `_save_chunks`, добавить (в отдельном try/except чтобы не блокировать индексацию):

```python
# Entity extraction для KG (async, не блокирует индексацию)
try:
    from graph_kg.extractor import extract_entities_relations, save_to_pg
    entities, relations = extract_entities_relations(full_text, title)
    if entities:
        save_to_pg(entities, relations, page_id, conn)
except Exception as e:
    logger.warning(f"KG extraction skipped for '{title}': {e}")
    # Не re-raise — индексация должна работать без KG
```

### 7. requirements.txt — добавить зависимости

```
python-louvain>=0.16
# Apache AGE НЕ нужен — используем plain PostgreSQL таблицы
```

### 8. config.py — добавить конфигурацию KG

```python
# Knowledge Graph
KG_ENABLED: bool = os.getenv("KG_ENABLED", "true").lower() == "true"
KG_MIN_COMMUNITY_SIZE: int = int(os.getenv("KG_MIN_COMMUNITY_SIZE", "3"))
```

## Стандарты
- **Karpathy**: Simple — не заменяем существующий RAG, добавляем слой поверх. Surgical — entity extraction в try/except, не блокирует индексацию. Thoughtful — dual retrieval, human выбирает kb_search vs kb_search_global.
- **Dev**: `dev-standards:ai-llm` — Haiku для extraction (дёшево), batch где можно; `dev-standards:postgresql` — plain таблицы, стандартный psycopg2, никакого AGE.
- **Проект**: все ошибки KG не блокируют основной RAG pipeline.

## Что НЕ делать
- **НЕ мигрировать на Neo4j** — plain PostgreSQL таблицы достаточно, один сервис
- **НЕ делать entity extraction синхронным блокирующим** — только try/except wrapper
- **НЕ удалять** существующий kb_search — dual retrieval дополняет, не заменяет
- **НЕ запускать** generate_community_summaries при каждой индексации — только по требованию или по расписанию

## Критерий готовности
- [ ] После `schema.sql` миграции: таблицы `kg_entities`, `kg_relations` существуют в БД
- [ ] После индексации Notion-страницы: в `kg_entities` появляются строки с entities
- [ ] `kb_rebuild_communities()` — создаёт ≥1 community_summary
- [ ] `kb_search_global("архитектура AdashAI")` — возвращает summary, не пустую строку
- [ ] Ошибка KG extraction НЕ ломает обычную индексацию Notion
- [ ] Karpathy review: код thoughtful, simple, surgical, goal-driven
