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
from indexer.notion_indexer import index_page, run_full_index
from memory.episodic import (
    delete_fact,
    format_facts_for_context,
    get_all_facts,
    upsert_fact,
)
from retriever.search import format_results_for_claude, search

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
) -> str:
    """
    Поиск по личной базе знаний (Notion, статьи, заметки).
    Используй ВСЕГДА когда нужен контекст о проектах, технологиях или решениях.

    Args:
        query: Вопрос или ключевые слова для поиска (на русском или английском)
        top_k: Количество результатов (1-10, по умолчанию 5)
        source_filter: Фильтр по источнику: 'notion' | 'url' | 'file' | 'manual'

    Returns:
        Релевантные фрагменты из базы знаний с ссылками на источники
    """
    top_k = max(1, min(top_k, 10))
    results = search(query=query, top_k=top_k, source_filter=source_filter, hybrid=True)
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
) -> str:
    """
    Добавить текст в базу знаний вручную (статья, заметка, конспект).

    Args:
        text: Текст для индексации
        title: Заголовок документа
        source_type: Тип источника ('manual', 'file', 'url')
        source_url: Ссылка на оригинал (опционально)

    Returns:
        Подтверждение с количеством созданных чанков
    """
    from core.db import get_conn, get_cursor

    chunks = chunk_document(text)
    if not chunks:
        return "Ошибка: текст пустой или слишком короткий для индексации."

    chunk_texts = [c.text for c in chunks]
    embeddings = embed_texts(chunk_texts)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
                cur.execute(
                    """
                    INSERT INTO documents
                        (source_type, source_url, title, content, chunk_index, chunk_total, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (source_type, source_url, title, chunk_text, i, len(chunks), embedding),
                )

    return f"✅ Добавлено: '{title}' — {len(chunks)} чанк(ов)"


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

        # Индексируем как обычный документ
        return kb_add_document(text=text, title=title, source_type="url", source_url=url)

    except httpx.HTTPError as e:
        return f"Ошибка загрузки URL: {e}"
    except Exception as e:
        return f"Ошибка обработки страницы: {e}"


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
def kb_list_sources(source_type: str | None = None) -> str:
    """
    Показать список проиндексированных источников.

    Args:
        source_type: Фильтр: 'notion' | 'url' | 'file' | 'manual'. Если пусто — все.

    Returns:
        Список источников с количеством чанков
    """
    from core.db import get_conn, get_cursor

    sql = """
        SELECT
            source_type,
            title,
            source_url,
            COUNT(*) as chunks,
            MAX(indexed_at) as last_indexed
        FROM documents
        {where}
        GROUP BY source_type, title, source_url
        ORDER BY source_type, title
    """.format(
        where="WHERE source_type = %s" if source_type else ""
    )

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(sql, (source_type,) if source_type else ())
            rows = cur.fetchall()

    if not rows:
        return "База знаний пуста."

    # Группируем по source_type
    by_type: dict[str, list] = {}
    for row in rows:
        stype = row["source_type"]
        by_type.setdefault(stype, []).append(row)

    lines = [f"📚 Всего источников: {len(rows)}\n"]
    for stype, items in sorted(by_type.items()):
        total_chunks = sum(r["chunks"] for r in items)
        lines.append(f"**{stype.upper()}** ({len(items)} документов, {total_chunks} чанков)")
        for item in items[:20]:  # ограничиваем вывод
            url_part = f" → {item['source_url']}" if item["source_url"] else ""
            lines.append(f"  • {item['title'] or 'Untitled'} [{item['chunks']} чанков]{url_part}")
        if len(items) > 20:
            lines.append(f"  ... и ещё {len(items) - 20}")
        lines.append("")

    return "\n".join(lines)


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
    fact = upsert_fact(key=key, value=value, category=category)
    return f"✅ Сохранено: [{fact.category}] {fact.key} = {fact.value}"


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
