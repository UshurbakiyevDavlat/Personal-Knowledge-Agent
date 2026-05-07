"""
Knowledge Agent MCP Server

Транспорт: stdio (локальный, для Claude Code / Cowork)
SDK: FastMCP (Python)

Инструменты:
  kb_search          — семантический поиск по базе знаний
  kb_add_document    — добавить текст вручную
  kb_add_url         — проиндексировать веб-страницу
  kb_index_notion    — переиндексировать Notion (всё или конкретные страницы)
  kb_list_sources    — список проиндексированных источников
  kb_delete          — удалить документ из базы
  kb_get_facts       — получить эпизодическую память
  kb_update_fact     — обновить факт в эпизодической памяти

Запуск: python -m mcp.server
Или напрямую: python mcp/server.py
"""
import logging
import sys

# Логи ТОЛЬКО в stderr — stdout зарезервирован для MCP протокола
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("knowledge-agent-mcp")

from mcp.server.fastmcp import FastMCP

from config import config
from core.chunker import chunk_document
from core.db import health_check
from core.embedder import embed_texts
from indexer.file_indexer import read_file
from indexer.notion_indexer import index_page, run_full_index
from memory.episodic import (
    delete_fact,
    format_facts_for_context,
    get_all_facts,
    get_fact_history,
    upsert_fact,
    upsert_temporal_fact,
)
from memory.procedural import (
    find_workflows,
    list_all_workflows,
    mark_workflow_used,
    save_workflow,
)
from retriever.search import format_results_for_claude, search
from wiki_agent.manager import (
    ingest_url as wiki_ingest_url_impl,
    lint_wiki as wiki_lint_impl,
    process_inbox as wiki_process_inbox_impl,
)

mcp = FastMCP(
    name="knowledge-agent",
    instructions=(
        "Personal Knowledge Agent — база знаний Давлата. "
        "Используй kb_search для поиска информации из Notion, статей и заметок. "
        "Ищи перед ответом на вопросы о проектах, технологиях, архитектурных решениях."
    ),
    host="0.0.0.0",
    port=8000,
)


# ══════════════════════════════════════════════
# kb_search — основной инструмент поиска
# ══════════════════════════════════════════════

@mcp.tool(
    annotations={"readOnlyHint": True, "openWorldHint": False},
)
def kb_search(
    query: str,
    top_k: int = 5,
    source_filter: str | None = None,
    date_from: str | None = None,
    rerank: bool = True,
) -> str:
    """
    Поиск по личной базе знаний (Notion, статьи, заметки).
    Используй ВСЕГДА когда нужен контекст о проектах, технологиях или решениях.

    Args:
        query: Вопрос или ключевые слова для поиска (на русском или английском)
        top_k: Количество результатов (1-10, по умолчанию 5)
        source_filter: Фильтр по источнику: 'notion' | 'url' | 'file' | 'manual'
        date_from: Фильтр по дате — вернуть только записи не раньше этой даты (YYYY-MM-DD)
        rerank: Применить Voyage rerank-2 после RRF (по умолчанию True)

    Returns:
        Релевантные фрагменты из базы знаний с ссылками на источники
    """
    top_k = max(1, min(top_k, 10))
    results = search(
        query=query, top_k=top_k, source_filter=source_filter,
        hybrid=True, date_from=date_from, rerank=rerank,
    )
    return format_results_for_claude(results)


# ══════════════════════════════════════════════
# kb_add_document — добавить текст вручную
# ══════════════════════════════════════════════

@mcp.tool(
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
)
def kb_add_document(
    text: str,
    title: str,
    source_type: str = "manual",
    source_url: str | None = None,
    doc_date: str | None = None,
) -> str:
    """
    Добавить текст в базу знаний вручную (статья, заметка, конспект).

    Args:
        text: Текст для индексации
        title: Заголовок документа
        source_type: Тип источника ('manual', 'file', 'url')
        source_url: Ссылка на оригинал (опционально)
        doc_date: Дата документа в формате YYYY-MM-DD (по умолчанию — сегодня)

    Returns:
        Подтверждение с количеством созданных чанков
    """
    import hashlib
    from datetime import date
    from psycopg2.extras import Json
    from core.db import get_conn, get_cursor

    chunks = chunk_document(text)
    if not chunks:
        return "Ошибка: текст пустой или слишком короткий для индексации."

    chunk_texts = [c.text for c in chunks]
    embeddings = embed_texts(chunk_texts)

    resolved_date = doc_date or date.today().isoformat()
    doc_source_id = source_url or hashlib.md5(f"{title}:{text[:200]}".encode()).hexdigest()
    meta = Json({"doc_date": resolved_date})

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
                cur.execute(
                    """
                    INSERT INTO documents
                        (source_type, source_id, source_url, title, content, chunk_index, chunk_total, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (source_type, doc_source_id, source_url, title, chunk_text, i, len(chunks), embedding, meta),
                )

    return f"✅ Добавлено: '{title}' — {len(chunks)} чанк(ов) | 📅 {resolved_date}"


# ══════════════════════════════════════════════
# kb_add_url — проиндексировать веб-страницу
# ══════════════════════════════════════════════

@mcp.tool(
    annotations={"readOnlyHint": False, "openWorldHint": True},
)
def kb_add_url(url: str) -> str:
    """
    Загрузить и проиндексировать веб-страницу по URL.

    Args:
        url: Полный URL страницы (https://...)

    Returns:
        Подтверждение индексации или сообщение об ошибке
    """
    import httpx
    from markdownify import markdownify
    from bs4 import BeautifulSoup
    from core.db import get_conn, get_cursor

    try:
        response = httpx.get(url, follow_redirects=True, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Убираем скрипты и стили
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Берём заголовок страницы
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        # Конвертируем в Markdown
        main = soup.find("main") or soup.find("article") or soup.find("body")
        html_content = str(main) if main else response.text
        text = markdownify(html_content, heading_style="ATX")

        # Дедупликация: удалить старые чанки если URL уже проиндексирован
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                cur.execute(
                    "SELECT COUNT(*) as cnt FROM documents WHERE source_id = %s",
                    (url,),
                )
                existing = cur.fetchone()["cnt"]
                if existing:
                    cur.execute("DELETE FROM documents WHERE source_id = %s", (url,))
                    logger.info(f"kb_add_url: удалено {existing} старых чанков для {url}")

        return kb_add_document(text=text, title=title, source_type="url", source_url=url)

    except httpx.HTTPError as e:
        return f"Ошибка загрузки URL: {e}"
    except Exception as e:
        return f"Ошибка обработки страницы: {e}"


# ══════════════════════════════════════════════
# kb_add_file — индексировать локальный файл
# ══════════════════════════════════════════════

@mcp.tool(
    annotations={"readOnlyHint": False, "openWorldHint": False},
)
def kb_add_file(
    path: str,
    title: str | None = None,
    doc_date: str | None = None,
) -> str:
    """
    Индексировать локальный файл в базу знаний.
    Поддерживаемые форматы: .md, .txt, .pdf

    Args:
        path: Путь к файлу (абсолютный или относительный от CWD)
        title: Заголовок документа (по умолчанию — имя файла)
        doc_date: Дата документа YYYY-MM-DD (по умолчанию — сегодня)

    Returns:
        Подтверждение индексации или сообщение об ошибке

    Примеры:
        kb_add_file("/Users/dushu/notes/architecture.md")
        kb_add_file("/Users/dushu/docs/research.pdf", title="GraphRAG Research 2025")
    """
    try:
        from pathlib import Path
        from core.db import get_conn, get_cursor

        file_title, text = read_file(path)
        resolved_title = title or file_title
        source_id = str(Path(path).resolve())

        # Идемпотентность: удаляем старые чанки если файл переиндексируется
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                cur.execute(
                    "DELETE FROM documents WHERE source_type = 'file' AND source_id = %s",
                    (source_id,),
                )
                old_count = cur.rowcount
                if old_count:
                    logger.info(f"kb_add_file: удалено {old_count} старых чанков для {source_id}")

        return kb_add_document(
            text=text,
            title=resolved_title,
            source_type="file",
            source_url=source_id,
            doc_date=doc_date,
        )

    except FileNotFoundError as e:
        return f"❌ {e}"
    except ValueError as e:
        return f"❌ {e}"
    except ImportError as e:
        return f"❌ {e}"
    except Exception as e:
        logger.error(f"kb_add_file error: {e}", exc_info=True)
        return f"❌ Ошибка индексации файла: {e}"


# ══════════════════════════════════════════════
# kb_index_notion — переиндексировать Notion
# ══════════════════════════════════════════════

@mcp.tool(
    annotations={"readOnlyHint": False, "openWorldHint": True},
)
def kb_index_notion(
    page_ids: list[str] | None = None,
    force: bool = False,
) -> str:
    """
    Запустить индексацию Notion. По умолчанию — только изменённые страницы.

    Args:
        page_ids: Список конкретных Page ID для индексации. Если пусто — берёт корневые из конфига.
        force: Переиндексировать всё, даже если страница не изменилась.

    Returns:
        Статистика: сколько страниц проиндексировано / пропущено / ошибок
    """
    stats = run_full_index(page_ids=page_ids, force=force)
    return (
        f"✅ Notion индексация завершена:\n"
        f"  • Проиндексировано: {stats['indexed']}\n"
        f"  • Пропущено (не изменились): {stats['skipped']}\n"
        f"  • Ошибок: {stats['failed']}"
    )


# ══════════════════════════════════════════════
# kb_list_sources — список источников
# ══════════════════════════════════════════════

@mcp.tool(
    annotations={"readOnlyHint": True},
)
def kb_list_sources(
    source_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "title",
) -> str:
    """
    Показать список проиндексированных источников.

    Args:
        source_type: Фильтр: 'notion' | 'url' | 'file' | 'manual'. Если пусто — все.
        limit: Количество записей на страницу (по умолчанию 50)
        offset: Смещение для пагинации (по умолчанию 0)
        sort_by: Сортировка: 'title' | 'date' | 'chunks'

    Returns:
        Список источников с количеством чанков и строкой пагинации
    """
    from core.db import get_conn, get_cursor

    order_clause = {
        "date": "ORDER BY MAX(indexed_at) DESC",
        "chunks": "ORDER BY COUNT(*) DESC",
    }.get(sort_by, "ORDER BY source_type, title")

    where = "WHERE source_type = %s" if source_type else ""
    params: list = []
    if source_type:
        params.append(source_type)

    # Общее количество для пагинации
    count_sql = f"SELECT COUNT(*) as total FROM (SELECT 1 FROM documents {where} GROUP BY source_type, title, source_url) sub"
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(count_sql, params)
            total = cur.fetchone()["total"]

            sql = f"""
                SELECT
                    source_type,
                    title,
                    source_url,
                    COUNT(*) as chunks,
                    MAX(indexed_at) as last_indexed
                FROM documents
                {where}
                GROUP BY source_type, title, source_url
                {order_clause}
                LIMIT %s OFFSET %s
            """
            cur.execute(sql, params + [limit, offset])
            rows = cur.fetchall()

    if not rows and offset == 0:
        return "База знаний пуста."

    lines = []
    for row in rows:
        url_part = f" → {row['source_url']}" if row["source_url"] else ""
        date_part = f" | 🕐 {row['last_indexed'].strftime('%Y-%m-%d')}" if row["last_indexed"] else ""
        lines.append(f"  [{row['source_type']}] {row['title'] or 'Untitled'} [{row['chunks']} чанков]{date_part}{url_part}")

    start = offset + 1
    end = offset + len(rows)
    pagination = f"\nПоказано: {start}-{end} из {total}."
    if end < total:
        pagination += f" Используй offset={end} для следующей страницы."

    return f"📚 Источники в базе знаний:\n" + "\n".join(lines) + pagination


# ══════════════════════════════════════════════
# kb_delete — удалить документ
# ══════════════════════════════════════════════

@mcp.tool(
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True},
)
def kb_delete(source_id: str, source_type: str = "notion") -> str:
    """
    Удалить все чанки документа из базы знаний.

    Args:
        source_id: ID документа (для Notion — Page ID, для других — source_id)
        source_type: Тип источника ('notion' | 'url' | 'file' | 'manual')

    Returns:
        Подтверждение удаления
    """
    from core.db import get_conn, get_cursor

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                "DELETE FROM documents WHERE source_type = %s AND source_id = %s",
                (source_type, source_id),
            )
            deleted = cur.rowcount

    if deleted > 0:
        return f"✅ Удалено {deleted} чанков (source_id={source_id})"
    return f"Документ не найден (source_id={source_id}, type={source_type})"


# ══════════════════════════════════════════════
# kb_get_facts — получить эпизодическую память
# ══════════════════════════════════════════════

@mcp.tool(
    annotations={"readOnlyHint": True},
)
def kb_get_facts() -> str:
    """
    Получить все факты из эпизодической памяти (предпочтения, навыки, контекст).

    Returns:
        Список всех сохранённых фактов
    """
    facts = get_all_facts()
    if not facts:
        return "Эпизодическая память пуста."
    return format_facts_for_context(facts)


# ══════════════════════════════════════════════
# kb_update_fact — обновить факт
# ══════════════════════════════════════════════

@mcp.tool(
    annotations={"readOnlyHint": False, "idempotentHint": True},
)
def kb_update_fact(
    key: str,
    value: str,
    category: str = "general",
) -> str:
    """
    Создать или обновить факт в эпизодической памяти.
    Используй чтобы сохранить что-то важное о пользователе.

    Args:
        key: Уникальный ключ факта (например: 'preferred_framework', 'current_project')
        value: Значение
        category: Категория: 'skill' | 'preference' | 'project' | 'personal' | 'general'

    Returns:
        Подтверждение сохранения
    """
    result = upsert_temporal_fact(key=key, value=value, category=category)
    action = result["action"]
    if action == "created":
        return f"✅ Сохранено: [{category}] {key} = {value}"
    elif action == "updated":
        return f"🔄 Обновлено: [{category}] {key}\n  Было: {result['old_value']}\n  Стало: {value}"
    else:
        return f"ℹ️ Без изменений: [{category}] {key} = {value}"


# ══════════════════════════════════════════════
# kb_get_fact_history — история изменений факта
# ══════════════════════════════════════════════

@mcp.tool(annotations={"readOnlyHint": True})
def kb_get_fact_history(key: str) -> str:
    """
    Показать историю изменений конкретного факта в эпизодической памяти.

    Args:
        key: Ключ факта (например: 'current_project', 'preferred_stack')

    Returns:
        Хронология изменений факта с временными метками
    """
    history = get_fact_history(key)
    if not history:
        return f"Факт '{key}' не найден в памяти."

    lines = [f"📜 История факта: {key}\n"]
    for entry in history:
        lines.append(f"  {entry['status']} {entry['value']}")
        lines.append(f"    с {entry['valid_from']} | источник: {entry['source']}")
        if entry["valid_to"]:
            lines.append(f"    до {entry['valid_to']}")
        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════
# kb_cleanup — массовое удаление по фильтрам
# ══════════════════════════════════════════════

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
    """
    from datetime import datetime, timedelta, timezone
    from core.db import get_conn, get_cursor

    conditions = []
    params: list = []

    if source_type:
        conditions.append("source_type = %s")
        params.append(source_type)

    if older_than_days is not None:
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

            cur.execute(f"SELECT COUNT(*) as total FROM documents {where}", params)
            total = cur.fetchone()["total"]

    if not rows:
        return "✅ Ничего не найдено по заданным фильтрам."

    label = "🔍 Найдено (dry_run)" if dry_run else "🗑️ Удалено"
    lines = [f"{label}: {total} чанков\n"]
    for row in rows:
        date_str = row["last_indexed"].strftime("%Y-%m-%d") if row["last_indexed"] else "—"
        lines.append(f"  • [{row['source_type']}] {row['title'] or 'Untitled'} — {row['chunks']} чанков | {date_str}")

    if len(rows) == 50:
        lines.append("  ... и ещё (показаны первые 50)")

    if dry_run:
        lines.append("\n💡 Для реального удаления вызови с dry_run=False")
        return "\n".join(lines)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"DELETE FROM documents {where}", params)
            deleted = cur.rowcount

    lines.append(f"\n✅ Удалено {deleted} чанков")
    return "\n".join(lines)


# ══════════════════════════════════════════════
# kb_delete_by_title — удаление по названию
# ══════════════════════════════════════════════

@mcp.tool(
    annotations={"readOnlyHint": False, "destructiveHint": True},
)
def kb_delete_by_title(title: str, source_type: str | None = None) -> str:
    """
    Удалить документ из базы знаний по названию (частичное совпадение).

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


# ══════════════════════════════════════════════
# Wiki (Karpathy-style markdown knowledge)
# ══════════════════════════════════════════════

@mcp.tool(annotations={"readOnlyHint": False, "openWorldHint": True})
def wiki_ingest_url(url: str, topic: str) -> str:
    """
    Загрузить статью по URL, извлечь инсайты и добавить в wiki-файл темы.

    Если wiki/{topic}.md существует — инсайты ДОБАВЛЯЮТСЯ в конец.
    Если нет — создаётся новый файл.
    Файл автоматически переиндексируется в KB (source_type='wiki').

    Args:
        url: URL статьи для загрузки
        topic: Название темы (станет именем файла: "RAG Best Practices" → rag_best_practices.md)

    Returns:
        Подтверждение с названием файла и количеством добавленных слов

    Примеры:
        wiki_ingest_url("https://arxiv.org/abs/2501.13956", "Temporal Memory Graphiti")
        wiki_ingest_url("https://blog.langchain.dev/langgraph/", "LangGraph Architecture")
    """
    try:
        result = wiki_ingest_url_impl(
            url=url,
            topic=topic,
            api_key=config.ANTHROPIC_API_KEY,
            voyage_api_key=config.VOYAGE_API_KEY,
        )
        action_emoji = "📄" if result["action"] == "created" else "📝"
        action_text = "создан" if result["action"] == "created" else "обновлён"
        return (
            f"{action_emoji} Wiki {action_text}: {result['file']}\n"
            f"  Добавлено: {result['words_added']} слов\n"
            f"  Тема: {result['topic']}"
        )
    except Exception as e:
        logger.error(f"wiki_ingest_url error: {e}", exc_info=True)
        return f"❌ Ошибка: {e}"


@mcp.tool(annotations={"readOnlyHint": False, "openWorldHint": True})
def wiki_process_inbox() -> str:
    """
    Обработать все URL из wiki/INBOX.md.

    Формат строк в INBOX.md:
        https://example.com/article | Название темы

    После обработки — строки удаляются из INBOX.
    Файлы с ошибками помечаются комментарием # ERROR:.

    Returns:
        Отчёт об обработанных и проблемных URL
    """
    try:
        result = wiki_process_inbox_impl(
            api_key=config.ANTHROPIC_API_KEY,
            voyage_api_key=config.VOYAGE_API_KEY,
        )
        lines = ["📥 Wiki Inbox обработан:"]
        lines.append(f"  ✅ Обработано: {len(result['processed'])} URL")
        for item in result["processed"]:
            lines.append(f"    — {item['topic']}: {item['result']['action']}")
        if result["errors"]:
            lines.append(f"  ❌ Ошибки: {len(result['errors'])}")
            for err in result["errors"]:
                lines.append(f"    — {err['url']}: {err['error']}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"wiki_process_inbox error: {e}", exc_info=True)
        return f"❌ Ошибка: {e}"


@mcp.tool(annotations={"readOnlyHint": False, "openWorldHint": False})
def wiki_lint() -> str:
    """
    Аудит wiki-базы знаний:
    - Находит файлы не обновлявшиеся > 90 дней
    - Находит потенциальные дубли тем
    - Обновляет wiki/README.md (индекс всех тем)
    - Переиндексирует все wiki-файлы в KB

    Returns:
        Отчёт с состоянием wiki, устаревшими файлами и дублями
    """
    try:
        result = wiki_lint_impl(voyage_api_key=config.VOYAGE_API_KEY)
        lines = ["🔍 Wiki Lint завершён:"]
        lines.append(f"  📁 Всего файлов: {result['total_files']}")
        lines.append(f"  📄 README.md обновлён: {'✅' if result['readme_updated'] else '❌'}")
        if result["stale"]:
            lines.append(f"\n  ⚠️ Устаревших (>90 дней): {result['stale_count']}")
            for s in result["stale"]:
                lines.append(f"    — {s['file']} (последнее: {s['last_modified']})")
        else:
            lines.append("  ✅ Устаревших файлов нет")
        if result["duplicates"]:
            lines.append(f"\n  🔄 Потенциальных дублей: {len(result['duplicates'])}")
            for d in result["duplicates"]:
                lines.append(f"    — {d['file_a']} ↔ {d['file_b']} ({int(d['similarity']*100)}% похожи)")
        else:
            lines.append("  ✅ Дублей не обнаружено")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"wiki_lint error: {e}", exc_info=True)
        return f"❌ Ошибка: {e}"


@mcp.tool(annotations={"readOnlyHint": True})
def wiki_list_topics() -> str:
    """
    Показать все темы в wiki/ с краткой статистикой.

    Returns:
        Список wiki-файлов с размером и датой последнего обновления
    """
    from datetime import datetime, timezone
    from pathlib import Path

    wiki_dir = Path("wiki")
    if not wiki_dir.exists():
        return "Wiki пуста. Используй wiki_ingest_url чтобы добавить первую статью."

    files = sorted(wiki_dir.glob("*.md"))
    topic_files = [f for f in files if f.name not in ("README.md", "INBOX.md")]

    if not topic_files:
        return "Wiki пуста. Используй wiki_ingest_url чтобы добавить первую статью."

    lines = [f"📚 Wiki ({len(topic_files)} тем):\n"]
    for f in topic_files:
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        word_count = len(f.read_text(encoding="utf-8").split())
        topic_name = f.stem.replace("_", " ").title()
        lines.append(f"  📄 {topic_name}")
        lines.append(f"     Файл: {f.name} | Слов: {word_count} | Обновлён: {mtime.strftime('%Y-%m-%d')}")

    return "\n".join(lines)


# ══════════════════════════════════════════════
# Visualization (Knowledge Graph Web UI)
# ══════════════════════════════════════════════

@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
def kb_open_graph(port: int = 7331) -> str:
    """
    Открыть интерактивный граф базы знаний в браузере.

    Запускает локальный веб-сервер и открывает visualization/graph.html
    с D3.js force-directed графом: ноды = документы/wiki/факты,
    рёбра = cosine similarity > 0.75.

    Args:
        port: Порт для веб-сервера (по умолчанию 7331)

    Returns:
        URL открытого интерфейса
    """
    try:
        from visualization.api import run_viz_server
        url = run_viz_server(port=port)
        return (
            f"🕸️ Knowledge Graph открыт: {url}\n\n"
            "Функции:\n"
            "  🔍 Поиск по заголовку документа\n"
            "  🎨 Фильтр по типу источника (Notion / Wiki / URL / Файлы / Факты)\n"
            "  🖱️ Drag & zoom, hover для просмотра деталей\n"
            "  🔗 Hover по ноде → открыть источник"
        )
    except ImportError:
        return "❌ Flask не установлен. Выполни: pip install flask"
    except Exception as e:
        logger.error(f"kb_open_graph error: {e}", exc_info=True)
        return f"❌ Ошибка запуска граф-интерфейса: {e}"


# ══════════════════════════════════════════════
# Procedural Memory (Workflows)
# ══════════════════════════════════════════════

@mcp.tool(annotations={"readOnlyHint": False, "openWorldHint": False})
def kb_save_workflow(
    name: str,
    trigger: str,
    steps: list,
    description: str | None = None,
    tags: list | None = None,
) -> str:
    """
    Сохранить именованный workflow — последовательность шагов для выполнения задачи.

    Используй когда: успешно выполнил задачу и хочешь запомнить как → чтобы в следующий
    раз не изобретать заново. Workflows ищутся по смыслу через kb_find_workflow.

    Args:
        name: Уникальное имя workflow (например: "Deploy AdashAI to Hetzner VPS")
        trigger: Когда применять (например: "деплой новой версии AdashAI на сервер")
        steps: Список шагов [{"step": 1, "action": "git pull && docker-compose build", "notes": "~3 мин"}]
        description: Краткое описание что делает workflow
        tags: Теги для группировки (например: ["devops", "docker"])

    Returns:
        Подтверждение сохранения
    """
    try:
        result = save_workflow(name=name, trigger=trigger, steps=steps, description=description, tags=tags)
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


@mcp.tool(annotations={"readOnlyHint": True})
def kb_find_workflow(query: str, top_k: int = 3) -> str:
    """
    Найти подходящий workflow по смыслу задачи.

    Ищет семантически близкие workflows через vector similarity.
    Если нашёл подходящий — разверни и выполни его шаги.

    Args:
        query: Что хочешь сделать ("задеплоить adash", "настроить nginx proxy")
        top_k: Сколько вариантов показать (по умолчанию 3)

    Returns:
        Список подходящих workflows с шагами
    """
    try:
        results = find_workflows(query=query, top_k=top_k)

        if not results:
            return (
                f"Workflows по запросу '{query}' не найдены.\n"
                "Используй kb_save_workflow чтобы сохранить первый."
            )

        lines = [f"🔍 Workflows для: '{query}'\n"]
        for i, wf in enumerate(results, 1):
            bar = "█" * int(wf["similarity"] * 10) + "░" * (10 - int(wf["similarity"] * 10))
            lines.append(f"**{i}. {wf['name']}**")
            lines.append(f"   Релевантность: {bar} {wf['similarity']:.0%}")
            lines.append(f"   Триггер: {wf['trigger']}")
            if wf["description"]:
                lines.append(f"   Описание: {wf['description']}")
            if wf["tags"]:
                lines.append(f"   Теги: {', '.join(wf['tags'])}")
            lines.append(f"   Использован: {wf['run_count']} раз\n")
            lines.append("   **Шаги:**")
            for step in wf["steps"]:
                step_num = step.get("step", "?")
                action = step.get("action", "")
                notes = step.get("notes", "")
                line = f"   {step_num}. {action}"
                if notes:
                    line += f"\n      _({notes})_"
                lines.append(line)
            lines.append("")
            mark_workflow_used(wf["id"])

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"kb_find_workflow error: {e}", exc_info=True)
        return f"❌ Ошибка: {e}"


@mcp.tool(annotations={"readOnlyHint": True})
def kb_list_workflows(tags: list | None = None) -> str:
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


# ══════════════════════════════════════════════
# GraphRAG global search
# ══════════════════════════════════════════════

@mcp.tool(annotations={"readOnlyHint": True})
def kb_search_global(query: str, top_k: int = 3) -> str:
    """
    Глобальный поиск по Knowledge Graph (GraphRAG).

    Ищет по community summaries — обобщённым описаниям кластеров знаний.
    Подходит для широких вопросов: 'какова архитектура?', 'что мы знаем о X?'

    В отличие от kb_search (поиск по чанкам), этот инструмент даёт
    высокоуровневую картину — что и как связано в базе знаний.

    Args:
        query: Вопрос или тема для поиска
        top_k: Количество community summaries (1-10, по умолчанию 3)

    Returns:
        Список релевантных community summaries с оценками релевантности

    Пример:
        kb_search_global("архитектура проекта")
        kb_search_global("технологический стек", top_k=5)
    """
    from graph_kg.retriever import search_global

    top_k = max(1, min(10, top_k))
    results = search_global(query, top_k=top_k)

    if not results:
        return "Community summaries не найдены. Запустите kb_rebuild_communities для построения графа знаний."

    lines = [f"## Глобальный поиск: «{query}»\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. {r.title} (score: {r.score})")
        lines.append(r.summary)
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def kb_rebuild_communities(min_size: int = 3) -> str:
    """
    Перестроить Knowledge Graph community summaries.

    Анализирует сущности и связи из всех проиндексированных документов,
    запускает алгоритм Louvain для обнаружения кластеров,
    генерирует summary для каждого кластера через Claude Haiku,
    сохраняет с эмбеддингами для глобального поиска.

    Запускать:
    - После массовой переиндексации Notion
    - При добавлении большого количества новых документов
    - Если kb_search_global возвращает нерелевантные результаты

    Args:
        min_size: Минимальный размер community (по умолчанию 3 сущности)

    Returns:
        Количество созданных/обновлённых community summaries
    """
    from graph_kg.communities import generate_community_summaries

    count = generate_community_summaries(min_size=min_size)

    if count == 0:
        return (
            "Community summaries не созданы. Возможные причины:\n"
            "• Граф пуст — нет сущностей в kg_entities\n"
            "• Все communities меньше min_size\n"
            "• Переиндексируйте Notion через kb_index_notion"
        )

    return f"✅ Построено {count} community summaries. Используйте kb_search_global для глобального поиска."


# ══════════════════════════════════════════════
# Запуск
# ══════════════════════════════════════════════

if __name__ == "__main__":
    # Проверяем подключение к БД перед запуском
    if not health_check():
        logger.error("Cannot connect to database. Check DATABASE_URL in .env")
        sys.exit(1)

    logger.info("Knowledge Agent MCP server starting (stdio transport)...")
    mcp.run(transport="stdio")
