═══════════════════════════════════════════
ТЗ: Итерация 2 — Temporal Memory (Graphiti паттерн)
Проект: Personal-Knowledge-Agent · Python 3.12 + PostgreSQL 17
Дата: 2026-05-07
═══════════════════════════════════════════

## Цель
Апгрейдить `user_facts` (episodic memory) до темпоральной памяти с bi-temporal семантикой: каждый факт знает когда был актуален, конфликты разрешаются автоматически, старые факты не удаляются — помечаются invalidated.

## Контекст
Сейчас `user_facts` — простая key-value таблица с upsert. При обновлении факта старое значение теряется навсегда. Нет истории: "раньше я думал X, теперь Y". Нет conflict resolution. Нет decay для устаревших фактов. Паттерн Graphiti/Zep: каждый факт имеет `valid_from`, `valid_to`, `invalid_at` — facts never die, they just become invalid. Это даёт +15 баллов на LongMemEval (Zep paper arXiv:2501.13956).

## Файлы

**Трогать:**
- `schema.sql` — добавить таблицу `episodic_events` + новые поля в `user_facts`
- `memory/episodic.py` — рефакторинг upsert_fact + добавить temporal query functions
- `agent_server/server.py` — обновить `kb_update_fact`, добавить `kb_get_fact_history`

**Создать:**
- `migrate_episodic.py` — миграция существующих фактов в новую схему

**Не трогать:**
- `retriever/search.py` — не затронут
- `indexer/` — не затронут

## Реализация

### 1. schema.sql — temporal episodic memory

Добавить в конец:
```sql
-- Temporal facts — bi-temporal episodic memory (Graphiti pattern)
-- Старые факты НЕ удаляются, они invalidated
CREATE TABLE IF NOT EXISTS episodic_events (
    id              SERIAL PRIMARY KEY,
    fact_key        VARCHAR(200)    NOT NULL,
    fact_value      TEXT            NOT NULL,
    category        VARCHAR(100)    NOT NULL DEFAULT 'general',
    confidence      REAL            DEFAULT 1.0,
    
    -- Bi-temporal timestamps
    valid_from      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),   -- когда стало актуальным
    valid_to        TIMESTAMPTZ,                               -- когда перестало (NULL = сейчас актуально)
    invalid_at      TIMESTAMPTZ,                               -- когда мы узнали что факт устарел
    
    -- Источник факта
    source          VARCHAR(100)    DEFAULT 'manual',         -- 'manual' | 'extracted' | 'inferred'
    context         TEXT,                                      -- контекст откуда взят факт
    
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- Индекс для быстрого получения актуальных фактов
CREATE INDEX IF NOT EXISTS episodic_events_active_idx
    ON episodic_events (fact_key, valid_from)
    WHERE valid_to IS NULL AND invalid_at IS NULL;

CREATE INDEX IF NOT EXISTS episodic_events_key_idx
    ON episodic_events (fact_key, valid_from DESC);
```

### 2. migrate_episodic.py — миграция существующих фактов

```python
"""
Мигрирует существующие user_facts в episodic_events.
Запуск: python migrate_episodic.py
"""
from core.db import get_conn, get_cursor

def migrate():
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # Читаем все текущие факты
            cur.execute("SELECT key, value, category, confidence, created_at FROM user_facts")
            facts = cur.fetchall()
            
            for fact in facts:
                cur.execute("""
                    INSERT INTO episodic_events (fact_key, fact_value, category, confidence, valid_from, source)
                    VALUES (%s, %s, %s, %s, %s, 'migrated')
                    ON CONFLICT DO NOTHING
                """, (fact["key"], fact["value"], fact["category"], fact["confidence"], fact["created_at"]))
            
            print(f"✅ Мигрировано {len(facts)} фактов в episodic_events")

if __name__ == "__main__":
    migrate()
```

### 3. memory/episodic.py — temporal upsert

Добавить новые функции:

```python
def upsert_temporal_fact(
    key: str,
    value: str,
    category: str = "general",
    confidence: float = 1.0,
    context: str | None = None,
) -> dict:
    """
    Создать или обновить факт с темпоральной семантикой.
    
    Если факт с таким key уже существует и значение изменилось:
    - Помечает старый как invalid (valid_to = NOW(), invalid_at = NOW())
    - Создаёт новый с valid_from = NOW()
    
    Если значение то же самое — просто обновляет confidence.
    
    Returns:
        dict с 'action': 'created' | 'updated' | 'unchanged'
    """
    from core.db import get_conn, get_cursor
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc)
    
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # Получить текущий актуальный факт
            cur.execute("""
                SELECT id, fact_value FROM episodic_events
                WHERE fact_key = %s AND valid_to IS NULL AND invalid_at IS NULL
                ORDER BY valid_from DESC
                LIMIT 1
            """, (key,))
            current = cur.fetchone()
            
            if current is None:
                # Новый факт
                cur.execute("""
                    INSERT INTO episodic_events (fact_key, fact_value, category, confidence, context, valid_from)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (key, value, category, confidence, context, now))
                
                # Также обновить user_facts для обратной совместимости
                cur.execute("""
                    INSERT INTO user_facts (key, value, category, confidence)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """, (key, value, category, confidence))
                
                return {"action": "created", "key": key, "value": value}
            
            elif current["fact_value"] == value:
                # Значение то же — обновляем confidence
                cur.execute("""
                    UPDATE episodic_events SET confidence = %s WHERE id = %s
                """, (confidence, current["id"]))
                return {"action": "unchanged", "key": key, "value": value}
            
            else:
                # Значение изменилось — invalidate старое, создать новое
                cur.execute("""
                    UPDATE episodic_events
                    SET valid_to = %s, invalid_at = %s
                    WHERE id = %s
                """, (now, now, current["id"]))
                
                cur.execute("""
                    INSERT INTO episodic_events (fact_key, fact_value, category, confidence, context, valid_from)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (key, value, category, confidence, context, now))
                
                # Обновить user_facts
                cur.execute("""
                    UPDATE user_facts SET value = %s, updated_at = NOW() WHERE key = %s
                """, (value, key))
                
                return {
                    "action": "updated",
                    "key": key,
                    "old_value": current["fact_value"],
                    "new_value": value,
                }


def get_fact_history(key: str) -> list[dict]:
    """
    Получить полную историю изменений факта.
    
    Returns:
        Список записей от новейшей к старейшей с временными метками
    """
    from core.db import get_conn, get_cursor
    
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                SELECT fact_key, fact_value, category, confidence, valid_from, valid_to, invalid_at, source
                FROM episodic_events
                WHERE fact_key = %s
                ORDER BY valid_from DESC
            """, (key,))
            rows = cur.fetchall()
    
    result = []
    for row in rows:
        status = "✅ актуально" if row["valid_to"] is None and row["invalid_at"] is None else "❌ устарело"
        result.append({
            "value": row["fact_value"],
            "status": status,
            "valid_from": row["valid_from"].isoformat() if row["valid_from"] else None,
            "valid_to": row["valid_to"].isoformat() if row["valid_to"] else None,
            "source": row["source"],
        })
    return result
```

### 4. agent_server/server.py — обновить kb_update_fact, добавить kb_get_fact_history

Заменить в `kb_update_fact`:
```python
from memory.episodic import upsert_temporal_fact

result = upsert_temporal_fact(key=key, value=value, category=category)
action = result["action"]

if action == "created":
    return f"✅ Сохранено: [{category}] {key} = {value}"
elif action == "updated":
    return f"🔄 Обновлено: [{category}] {key}\n  Было: {result['old_value']}\n  Стало: {value}"
else:
    return f"ℹ️ Без изменений: [{category}] {key} = {value}"
```

Добавить новый tool:
```python
@mcp.tool(annotations={"readOnlyHint": True})
def kb_get_fact_history(key: str) -> str:
    """
    Показать историю изменений конкретного факта в эпизодической памяти.
    
    Args:
        key: Ключ факта (например: 'current_project', 'preferred_stack')
    
    Returns:
        Хронология изменений факта с временными метками
    """
    from memory.episodic import get_fact_history
    
    history = get_fact_history(key)
    if not history:
        return f"Факт '{key}' не найден в памяти."
    
    lines = [f"📜 История факта: {key}\n"]
    for entry in history:
        lines.append(f"  {entry['status']} {entry['value']}")
        lines.append(f"    с {entry['valid_from']} | источник: {entry['source']}")
        if entry['valid_to']:
            lines.append(f"    до {entry['valid_to']}")
        lines.append("")
    
    return "\n".join(lines)
```

## Стандарты
- **Karpathy**: Simple — новая таблица рядом со старой (user_facts остаётся для совместимости). Surgical — меняем только memory/episodic.py и server.py. Thoughtful — bi-temporal, данные не теряются.
- **Dev**: `dev-standards:postgresql` — правильные временны́е типы (TIMESTAMPTZ), индекс WHERE для актуальных фактов (partial index).
- **Проект**: синхронный psycopg2, get_conn()/get_cursor(), все функции через episodic.py.

## Что НЕ делать
- **НЕ удалять user_facts таблицу** — обратная совместимость с kb_get_facts
- **НЕ делать episodic_events partitioned** — пока не нужно
- **НЕ автоматически извлекать факты из диалогов** — это отдельная задача (Итерация 3)

## Критерий готовности
- [ ] `kb_update_fact(key="current_project", value="AdashAI v2")` дважды — второй раз: "🔄 Обновлено: Было X, Стало Y"
- [ ] `kb_get_fact_history("current_project")` — показывает оба значения с датами
- [ ] Старый `kb_get_facts()` — работает как прежде (через user_facts)
- [ ] Старые факты из user_facts мигрированы в episodic_events (после `python migrate_episodic.py`)
- [ ] Karpathy review: код thoughtful, simple, surgical, goal-driven
