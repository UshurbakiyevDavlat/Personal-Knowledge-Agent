═══════════════════════════════════════════
ТЗ: Итерация 3 — Procedural Memory (Workflow & Skills KG)
Проект: Personal-Knowledge-Agent · Python 3.12 + PostgreSQL 17
Дата: 2026-05-07
═══════════════════════════════════════════

## Цель
Добавить процедурную память — хранение "как делать X" в виде именованных workflows. Агент может записать успешный workflow (цепочка шагов), затем найти его при следующем похожем запросе. Это 4-й уровень памяти по нейробиологической аналогии: Working → Episodic → Semantic → **Procedural** (навыки/процедуры).

## Контекст
Сейчас KB хранит факты (user_facts) и документы (documents), но не знает "как делать вещи". Если агент однажды разобрался "как деплоить AdashAI на VPS" — это знание теряется. Процедурная память решает это: `kb_save_workflow(name, steps, trigger)` записывает именованный набор шагов; `kb_find_workflow(query)` находит ближайший по смыслу workflow. Хранится в отдельной таблице `workflows` с embedding на name+trigger для поиска. Простейший паттерн — не граф, а таблица с JSONB steps + vector embedding.

## Файлы

**Трогать:**
- `schema.sql` — добавить таблицу `workflows`
- `agent_server/server.py` — добавить 3 новых MCP tool
- `memory/procedural.py` — создать новый модуль (по аналогии с episodic.py)

**Не трогать:**
- `memory/episodic.py` — не затронут
- `retriever/search.py` — workflows не участвуют в kb_search
- `core/chunker.py` — не нужен (workflows не чанкуются)

## Реализация

### 1. schema.sql — таблица workflows

Добавить в конец schema.sql:

```sql
-- Procedural memory — именованные workflows (как делать X)
CREATE TABLE IF NOT EXISTS workflows (
    id              SERIAL          PRIMARY KEY,
    name            VARCHAR(200)    NOT NULL UNIQUE,   -- "Deploy AdashAI to VPS"
    trigger         TEXT            NOT NULL,          -- "когда нужно задеплоить AdashAI"
    description     TEXT,                              -- краткое описание что делает workflow
    steps           JSONB           NOT NULL,          -- [{step: 1, action: "...", notes: "..."}, ...]
    tags            TEXT[]          DEFAULT '{}',      -- теги для фильтрации
    
    -- Embedding на name + trigger + description для семантического поиска
    embedding       vector(1024),
    
    -- Метаданные
    run_count       INTEGER         DEFAULT 0,         -- сколько раз использовали
    last_used_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- Индекс для семантического поиска
CREATE INDEX IF NOT EXISTS workflows_embedding_idx
    ON workflows USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Индекс для текстового поиска
CREATE INDEX IF NOT EXISTS workflows_name_idx
    ON workflows (name);

-- Trigger для updated_at
CREATE OR REPLACE FUNCTION update_workflows_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER workflows_updated_at_trigger
    BEFORE UPDATE ON workflows
    FOR EACH ROW EXECUTE FUNCTION update_workflows_updated_at();
```

### 2. memory/procedural.py — логика работы с workflows

```python
"""
Procedural Memory — хранение и поиск именованных workflows.
Паттерн: таблица workflows с JSONB steps + vector embedding.
"""
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
    
    Args:
        name: Уникальное имя workflow ("Deploy AdashAI to VPS")
        trigger: Когда применять ("когда нужно задеплоить AdashAI")
        steps: Список шагов [{"step": 1, "action": "...", "notes": "..."}]
        description: Краткое описание
        tags: Теги для группировки
    
    Returns:
        dict с 'action': 'created' | 'updated', 'id': int
    """
    from core.db import get_conn, get_cursor
    from core.embedder import embed_query
    
    # Эмбеддинг на name + trigger + description
    embed_text = f"{name}. {trigger}. {description or ''}"
    embedding = embed_query(embed_text)
    
    tags_list = tags or []
    
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # Проверить существующий
            cur.execute("SELECT id FROM workflows WHERE name = %s", (name,))
            existing = cur.fetchone()
            
            import json
            steps_json = json.dumps(steps, ensure_ascii=False)
            
            if existing:
                cur.execute("""
                    UPDATE workflows
                    SET trigger = %s, description = %s, steps = %s::jsonb,
                        tags = %s, embedding = %s::vector
                    WHERE id = %s
                """, (trigger, description, steps_json, tags_list, str(embedding), existing["id"]))
                return {"action": "updated", "id": existing["id"], "name": name}
            else:
                cur.execute("""
                    INSERT INTO workflows (name, trigger, description, steps, tags, embedding)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s::vector)
                    RETURNING id
                """, (name, trigger, description, steps_json, tags_list, str(embedding)))
                row = cur.fetchone()
                return {"action": "created", "id": row["id"], "name": name}


def find_workflows(
    query: str,
    top_k: int = 3,
    tags: Optional[list[str]] = None,
) -> list[dict]:
    """
    Найти workflows по смыслу запроса.
    
    Args:
        query: Что хочешь сделать ("деплой AdashAI")
        top_k: Количество результатов
        tags: Фильтр по тегам (optional)
    
    Returns:
        Список workflows, отсортированных по релевантности
    """
    from core.db import get_conn, get_cursor
    from core.embedder import embed_query
    
    query_embedding = embed_query(query)
    
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            base_sql = """
                SELECT id, name, trigger, description, steps, tags,
                       run_count, last_used_at,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM workflows
                WHERE embedding IS NOT NULL
            """
            params = [str(query_embedding)]
            
            if tags:
                base_sql += " AND tags && %s"
                params.append(tags)
            
            base_sql += " ORDER BY similarity DESC LIMIT %s"
            params.append(top_k)
            
            cur.execute(base_sql, params)
            rows = cur.fetchall()
    
    results = []
    for row in rows:
        import json
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


def mark_workflow_used(workflow_id: int):
    """Инкрементировать run_count и обновить last_used_at."""
    from core.db import get_conn, get_cursor
    
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                UPDATE workflows
                SET run_count = run_count + 1, last_used_at = NOW()
                WHERE id = %s
            """, (workflow_id,))


def list_all_workflows(tags: Optional[list[str]] = None) -> list[dict]:
    """Получить все workflows (без embedding-поиска), опционально фильтр по тегам."""
    from core.db import get_conn, get_cursor
    
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            sql = """
                SELECT id, name, trigger, description, steps, tags, run_count, last_used_at
                FROM workflows
            """
            params = []
            if tags:
                sql += " WHERE tags && %s"
                params.append(tags)
            sql += " ORDER BY run_count DESC, created_at DESC"
            
            cur.execute(sql, params)
            rows = cur.fetchall()
    
    results = []
    for row in rows:
        import json
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
```

### 3. agent_server/server.py — добавить 3 новых MCP tool

Добавить импорт:
```python
from memory.procedural import save_workflow, find_workflows, mark_workflow_used, list_all_workflows
```

Добавить инструменты:

```python
@mcp.tool(
    annotations={"readOnlyHint": False, "openWorldHint": False},
)
def kb_save_workflow(
    name: str,
    trigger: str,
    steps: list,
    description: str | None = None,
    tags: list | None = None,
) -> str:
    """
    Сохранить именованный workflow — последовательность шагов для выполнения задачи.
    
    Используй когда: успешно выполнил задачу и хочешь запомнить как → чтобы в следующий раз
    не изобретать заново. Workflow ищутся по смыслу через kb_find_workflow.
    
    Args:
        name: Уникальное имя workflow (например: "Deploy AdashAI to Hetzner VPS")
        trigger: Когда применять (например: "деплой новой версии AdashAI на сервер")
        steps: Список шагов [{"step": 1, "action": "git pull && docker-compose build", "notes": "занимает ~3 мин"}]
        description: Краткое описание что делает workflow
        tags: Теги для группировки (например: ["devops", "adash", "docker"])
    
    Returns:
        Подтверждение сохранения
    
    Примеры:
        kb_save_workflow(
            name="Deploy AdashAI to Hetzner",
            trigger="когда нужно задеплоить новую версию AdashAI",
            steps=[
                {"step": 1, "action": "ssh root@hetzner-ip", "notes": "ключ в ~/.ssh/hetzner"},
                {"step": 2, "action": "cd /opt/adash && git pull origin main"},
                {"step": 3, "action": "docker-compose up -d --build", "notes": "~3 мин"},
                {"step": 4, "action": "docker-compose logs -f --tail=50"}
            ],
            tags=["devops", "adash"]
        )
    """
    try:
        result = save_workflow(
            name=name,
            trigger=trigger,
            steps=steps,
            description=description,
            tags=tags,
        )
        emoji = "✅" if result["action"] == "created" else "🔄"
        action_text = "Сохранён" if result["action"] == "created" else "Обновлён"
        return (
            f"{emoji} Workflow {action_text}: '{name}'\n"
            f"  Шагов: {len(steps)}\n"
            f"  Триггер: {trigger}\n"
            f"  Теги: {', '.join(tags or []) or 'нет'}"
        )
    except Exception as e:
        logger.error(f"kb_save_workflow error: {e}", exc_info=True)
        return f"❌ Ошибка: {e}"


@mcp.tool(
    annotations={"readOnlyHint": True},
)
def kb_find_workflow(
    query: str,
    top_k: int = 3,
) -> str:
    """
    Найти подходящий workflow по смыслу задачи.
    
    Ищет семантически близкие workflows через vector similarity.
    Если нашёл подходящий — разверни и выполни его шаги.
    
    Args:
        query: Что хочешь сделать ("задеплоить adash", "настроить nginx proxy")
        top_k: Сколько вариантов показать (по умолчанию 3)
    
    Returns:
        Список подходящих workflows с шагами
    
    Примеры:
        kb_find_workflow("задеплоить AdashAI")
        kb_find_workflow("настроить SSL сертификат")
    """
    try:
        results = find_workflows(query=query, top_k=top_k)
        
        if not results:
            return f"Workflows по запросу '{query}' не найдены.\nИспользуй kb_save_workflow чтобы сохранить первый."
        
        lines = [f"🔍 Workflows для: '{query}'\n"]
        for i, wf in enumerate(results, 1):
            relevance_bar = "█" * int(wf["similarity"] * 10) + "░" * (10 - int(wf["similarity"] * 10))
            lines.append(f"**{i}. {wf['name']}**")
            lines.append(f"   Релевантность: {relevance_bar} {wf['similarity']:.0%}")
            lines.append(f"   Триггер: {wf['trigger']}")
            if wf["description"]:
                lines.append(f"   Описание: {wf['description']}")
            if wf["tags"]:
                lines.append(f"   Теги: {', '.join(wf['tags'])}")
            lines.append(f"   Использован: {wf['run_count']} раз\n")
            lines.append(f"   **Шаги:**")
            for step in wf["steps"]:
                step_num = step.get("step", "?")
                action = step.get("action", "")
                notes = step.get("notes", "")
                line = f"   {step_num}. {action}"
                if notes:
                    line += f"\n      _({notes})_"
                lines.append(line)
            lines.append("")
            
            # Инкрементировать использование
            mark_workflow_used(wf["id"])
        
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"kb_find_workflow error: {e}", exc_info=True)
        return f"❌ Ошибка: {e}"


@mcp.tool(
    annotations={"readOnlyHint": True},
)
def kb_list_workflows(
    tags: list | None = None,
) -> str:
    """
    Показать все сохранённые workflows с кратким описанием.
    
    Args:
        tags: Фильтр по тегам (например: ["devops"] покажет только devops-workflows)
    
    Returns:
        Список всех workflows, отсортированных по частоте использования
    """
    try:
        workflows = list_all_workflows(tags=tags)
        
        if not workflows:
            msg = "Процедурная память пуста."
            if tags:
                msg += f" Нет workflows с тегами: {', '.join(tags)}"
            msg += "\nИспользуй kb_save_workflow чтобы записать первый workflow."
            return msg
        
        lines = [f"📋 Procedural Memory ({len(workflows)} workflows):\n"]
        for wf in workflows:
            tags_str = f" [{', '.join(wf['tags'])}]" if wf["tags"] else ""
            last_used = f", последний: {wf['last_used_at'][:10]}" if wf["last_used_at"] else ""
            lines.append(f"  🔧 **{wf['name']}**{tags_str}")
            lines.append(f"     Триггер: {wf['trigger']}")
            lines.append(f"     Шагов: {wf['steps_count']} | Использован: {wf['run_count']} раз{last_used}")
            lines.append("")
        
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"kb_list_workflows error: {e}", exc_info=True)
        return f"❌ Ошибка: {e}"
```

## Стандарты
- **Karpathy**: Simple — таблица + JSONB steps, никаких графовых баз. Surgical — только schema.sql + новый memory/procedural.py + 3 tools. Goal-driven — workflows найдены за 1 вызов, шаги показываются сразу.
- **Dev**: `dev-standards:postgresql` — HNSW индекс на embedding как в documents, TIMESTAMPTZ, JSONB для гибкости steps.
- **Dev**: `dev-standards:ai-llm` — embed_query переиспользован (тот же voyage-3), не делаем ещё одного клиента.
- **Проект**: синхронный psycopg2, get_conn()/get_cursor(), patterns как в episodic.py.

## Что НЕ делать
- **НЕ делать workflow как граф** — JSONB steps достаточно, AGE граф для этого излишен
- **НЕ автоматически создавать workflows** из диалога — только явное kb_save_workflow (это Итерация 4)
- **НЕ добавлять workflows в kb_search** — они ищутся только через kb_find_workflow
- **НЕ делать шаги строго типизированными** — JSONB позволяет любую структуру step, не ломаем
- **НЕ удалять workflow** если run_count > 0 — защита от случайного удаления ценного знания (предупреждение, не запрет)

## Критерий готовности
- [ ] `kb_save_workflow(name="Test deploy", trigger="деплой", steps=[{"step":1,"action":"ssh server"}])` — возвращает "✅ Workflow Сохранён"
- [ ] Повторный вызов с тем же именем — "🔄 Workflow Обновлён" (не дублирует)
- [ ] `kb_find_workflow("задеплоить сервер")` — находит "Test deploy" с similarity > 0.6
- [ ] `kb_find_workflow("какой-то нерелевантный запрос")` — возвращает результаты (по лучшему совпадению) или "не найдены"
- [ ] `kb_list_workflows()` — показывает все workflows отсортированные по run_count
- [ ] `kb_list_workflows(tags=["devops"])` — фильтрует по тегу
- [ ] После `kb_find_workflow` — run_count для найденного workflow инкрементирован на 1
- [ ] `kb_open_graph()` — ноды workflows НЕ отображаются (они не в documents), граф стабилен
- [ ] Karpathy review: код thoughtful, simple, surgical, goal-driven
