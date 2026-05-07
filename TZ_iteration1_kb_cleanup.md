═══════════════════════════════════════════
ТЗ: KB Cleanup — человекочитаемое удаление и дедупликация
Проект: Personal-Knowledge-Agent · Python 3.12 + FastAPI + psycopg2 + pgvector
Дата: 2026-05-07
═══════════════════════════════════════════

## Цель
Сделать так чтобы устаревшие документы из KB можно было удалять без знания внутренних source_id — по source_type, дате, названию или URL. Добавить дедупликацию в kb_add_url.

## Контекст
Сейчас `kb_delete(source_id, source_type)` требует внутренний UUID/Notion page ID — агент его не знает для manual/url документов. В `kb_add_document` поле `source_id` вообще не заполняется. `kb_add_url` не проверяет дубликаты: повторный вызов создаёт дублирующие чанки. `kb_list_sources` обрезает на 20 записей на тип без возможности пагинации.

## Файлы

**Трогать:**
- `agent_server/server.py` — добавить 2 новых инструмента (`kb_cleanup`, `kb_delete_by_title`), исправить `kb_add_document` и `kb_add_url`, исправить `kb_list_sources`
- `retriever/search.py` — ничего не трогать (не нужно)

**Не трогать:**
- `core/embedder.py` — не затронут
- `core/chunker.py` — не затронут
- `indexer/notion_indexer.py` — не затронут
- `schema.sql` — схема не меняется, поле `source_id VARCHAR(500)` уже есть

**Создать:** ничего нового

## Реализация

### 1. Исправить `kb_add_document` — заполнять source_id

В `kb_add_document` добавить вычисление source_id для manual-документов.
После строки `resolved_date = doc_date or date.today().isoformat()` добавить:

```python
import hashlib
# source_id = хэш от title + первых 200 символов текста
doc_source_id = source_url or hashlib.md5(f"{title}:{text[:200]}".encode()).hexdigest()
```

В INSERT добавить поле source_id:
```python
cur.execute(
    """
    INSERT INTO documents
        (source_type, source_id, source_url, title, content, chunk_index, chunk_total, embedding, metadata)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """,
    (source_type, doc_source_id, source_url, title, chunk_text, i, len(chunks), embedding, meta),
)
```

### 2. Исправить `kb_add_url` — дедупликация

В `kb_add_url`, ПЕРЕД вызовом `kb_add_document`, добавить проверку дубликата:

```python
from core.db import get_conn, get_cursor

# Проверяем — уже есть в базе?
with get_conn() as conn:
    with get_cursor(conn) as cur:
        cur.execute(
            "SELECT COUNT(*), MAX(indexed_at) FROM documents WHERE source_id = %s",
            (url,)
        )
        row = cur.fetchone()
        if row and row["count"] > 0:
            # Удаляем старые чанки перед переиндексацией
            cur.execute("DELETE FROM documents WHERE source_id = %s", (url,))
            logger.info(f"kb_add_url: удалено {row['count']} старых чанков для {url}")
```

Пояснение: это идемпотентное поведение — повторное добавление URL обновляет документ, а не дублирует.

### 3. Исправить `kb_list_sources` — пагинация + sort

Заменить сигнатуру:
```python
def kb_list_sources(
    source_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "title",  # 'title' | 'date' | 'chunks'
) -> str:
```

В SQL добавить `LIMIT %s OFFSET %s` и ORDER BY в зависимости от sort_by:
- `sort_by="date"` → `ORDER BY MAX(indexed_at) DESC`
- `sort_by="chunks"` → `ORDER BY COUNT(*) DESC`
- иначе → `ORDER BY source_type, title`

Убрать ограничение `items[:20]`, вместо него использовать SQL LIMIT/OFFSET.

Добавить в вывод строку пагинации:
```
Показано: 1-50 из 127 документов. Используй offset=50 для следующей страницы.
```

### 4. Добавить `kb_cleanup` — массовое удаление по фильтрам

Новый MCP tool в `agent_server/server.py`:

```python
@mcp.tool(
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False},
)
def kb_cleanup(
    source_type: str | None = None,
    older_than_days: int | None = None,
    title_contains: str | None = None,
    dry_run: bool = True,
) -> str:
    """
    Массовое удаление устаревших или ненужных документов из базы знаний.

    Args:
        source_type: Удалить только этот тип ('notion' | 'url' | 'file' | 'manual')
        older_than_days: Удалить документы старше N дней (по indexed_at)
        title_contains: Удалить документы, в названии которых есть это слово (ILIKE)
        dry_run: True = только показать что будет удалено, False = реально удалить

    Returns:
        Список документов к удалению + количество чанков. При dry_run=False — подтверждение.

    Примеры:
        kb_cleanup(source_type="manual", older_than_days=30, dry_run=True)
        kb_cleanup(title_contains="апрель", dry_run=False)
        kb_cleanup(source_type="url", older_than_days=90, dry_run=False)
    """
    from core.db import get_conn, get_cursor

    conditions = []
    params = []

    if source_type:
        conditions.append("source_type = %s")
        params.append(source_type)

    if older_than_days is not None:
        # ВАЖНО: psycopg2 не параметризует %s внутри строковых литералов SQL.
        # INTERVAL '%s days' — это баг. Правильно: вычислить cutoff в Python.
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        conditions.append("indexed_at < %s")
        params.append(cutoff)

    if title_contains:
        conditions.append("title ILIKE %s")
        params.append(f"%{title_contains}%")

    if not conditions:
        return "❌ Нужен хотя бы один фильтр: source_type, older_than_days или title_contains"

    where = "WHERE " + " AND ".join(conditions)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # Сначала показываем что будет удалено
            cur.execute(
                f"""
                SELECT source_type, title, source_id, COUNT(*) as chunks, MAX(indexed_at) as last_indexed
                FROM documents {where}
                GROUP BY source_type, title, source_id
                ORDER BY last_indexed
                LIMIT 50
                """,
                params,
            )
            rows = cur.fetchall()

            # Подсчёт общего количества чанков
            cur.execute(f"SELECT COUNT(*) as total FROM documents {where}", params)
            total = cur.fetchone()["total"]

    if not rows:
        return "✅ Ничего не найдено по заданным фильтрам."

    lines = [f"{'🔍 Найдено (dry_run)' if dry_run else '🗑️ Удалено'}: {total} чанков\n"]
    for row in rows:
        date_str = row["last_indexed"].strftime("%Y-%m-%d") if row["last_indexed"] else "—"
        lines.append(f"  • [{row['source_type']}] {row['title'] or 'Untitled'} — {row['chunks']} чанков | {date_str}")

    if len(rows) == 50:
        lines.append(f"  ... и ещё (показаны первые 50)")

    if dry_run:
        lines.append(f"\n💡 Для реального удаления вызови с dry_run=False")
        return "\n".join(lines)

    # Реальное удаление
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"DELETE FROM documents {where}", params)
            deleted = cur.rowcount

    lines.append(f"\n✅ Удалено {deleted} чанков")
    return "\n".join(lines)
```

### 5. Добавить `kb_delete_by_title` — удаление по названию

Новый MCP tool — простой вариант для быстрого удаления конкретного документа по названию:

```python
@mcp.tool(
    annotations={"readOnlyHint": False, "destructiveHint": True},
)
def kb_delete_by_title(title: str, source_type: str | None = None) -> str:
    """
    Удалить документ из базы знаний по названию (точное или частичное совпадение).

    Args:
        title: Название документа (ILIKE поиск — частичное совпадение)
        source_type: Уточнить тип источника (опционально)

    Returns:
        Список удалённых документов и количество чанков

    Пример:
        kb_delete_by_title("Статус апрель 2026")
        kb_delete_by_title("баги", source_type="manual")
    """
    from core.db import get_conn, get_cursor

    params: list = [f"%{title}%"]
    where = "WHERE title ILIKE %s"
    if source_type:
        where += " AND source_type = %s"
        params.append(source_type)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # Показываем что найдено
            cur.execute(
                f"SELECT DISTINCT title, source_type, COUNT(*) as chunks FROM documents {where} GROUP BY title, source_type",
                params,
            )
            rows = cur.fetchall()

            if not rows:
                return f"Документ с названием '{title}' не найден."

            cur.execute(f"DELETE FROM documents {where}", params)
            deleted = cur.rowcount

    lines = [f"✅ Удалено {deleted} чанков:"]
    for row in rows:
        lines.append(f"  • [{row['source_type']}] {row['title']} ({row['chunks']} чанков)")
    return "\n".join(lines)
```

## Стандарты
- **Karpathy**: Simple — не вводим новых таблиц, не меняем схему. Surgical — трогаем только agent_server/server.py. Goal-driven — решаем конкретную боль пользователя с удалением.
- **Dev**: `dev-standards:python-api` — async-паттерны не нужны (psycopg2 sync), конвенции проекта — get_conn()/get_cursor() как context managers, возвращать строки для MCP.
- **Проект**: все функции возвращают `str`, используют `get_conn()` + `get_cursor()` из `core/db.py`, логи через `logger = logging.getLogger(__name__)`.

## Что НЕ делать
- **НЕ менять схему schema.sql** — поле source_id уже есть, просто не заполнялось
- **НЕ добавлять async/await** — весь проект синхронный psycopg2, не asyncpg
- **НЕ трогать** `retriever/search.py`, `core/`, `indexer/` — они не нужны для этой задачи
- **НЕ удалять** старый `kb_delete` — оставить для обратной совместимости (Notion page ID)
- **НЕ делать безвозвратное удаление без dry_run защиты** в kb_cleanup

## Критерий готовности
- [ ] `kb_cleanup(title_contains="апрель", dry_run=True)` — показывает список без удаления
- [ ] `kb_cleanup(title_contains="апрель", dry_run=False)` — реально удаляет, возвращает счётчик
- [ ] `kb_delete_by_title("Статус апрель")` — удаляет по частичному совпадению
- [ ] `kb_add_url(url)` при повторном вызове — обновляет документ, не дублирует (проверить COUNT(*) до и после)
- [ ] `kb_list_sources(limit=10, offset=0, sort_by="date")` — работает с пагинацией
- [ ] `kb_add_document(text, title)` — заполняет source_id (проверить SELECT source_id FROM documents WHERE title = '...')
- [ ] Старый `kb_delete(source_id, source_type)` — всё ещё работает для Notion
- [ ] Karpathy review: код thoughtful, simple, surgical, goal-driven
- [ ] Нет сломанных зависимостей — только agent_server/server.py изменён
