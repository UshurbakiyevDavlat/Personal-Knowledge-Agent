"""
Notion Indexer — забирает страницы из Notion и индексирует их в pgvector.

Алгоритм:
1. Рекурсивно обходит страницы начиная с NOTION_ROOT_PAGE_IDS
2. Каждую страницу конвертирует в plain text
3. Разбивает на чанки → генерирует эмбеддинги → сохраняет в БД
4. Инкрементальная индексация: пропускает страницы, которые не изменились
"""
import logging
from datetime import datetime, timezone
from typing import Any

from notion_client import Client
from notion_client.errors import APIResponseError

from config import config
from core.chunker import chunk_document
from core.db import get_conn, get_cursor
from core.embedder import embed_texts

logger = logging.getLogger(__name__)

_notion: Client | None = None


def get_notion() -> Client:
    global _notion
    if _notion is None:
        _notion = Client(auth=config.NOTION_API_KEY)
    return _notion


# ──────────────────────────────────────────────
# Конвертация блоков Notion в plain text
# ──────────────────────────────────────────────

def _rich_text_to_str(rich_text: list[dict]) -> str:
    return "".join(rt.get("plain_text", "") for rt in rich_text)


def _block_to_text(block: dict) -> str:
    """Конвертировать один блок Notion в строку."""
    btype = block.get("type", "")
    data = block.get(btype, {})

    match btype:
        case "paragraph":
            return _rich_text_to_str(data.get("rich_text", []))
        case "heading_1":
            return "# " + _rich_text_to_str(data.get("rich_text", []))
        case "heading_2":
            return "## " + _rich_text_to_str(data.get("rich_text", []))
        case "heading_3":
            return "### " + _rich_text_to_str(data.get("rich_text", []))
        case "bulleted_list_item" | "numbered_list_item":
            return "- " + _rich_text_to_str(data.get("rich_text", []))
        case "to_do":
            checked = "✅" if data.get("checked") else "☐"
            return f"{checked} " + _rich_text_to_str(data.get("rich_text", []))
        case "toggle":
            return _rich_text_to_str(data.get("rich_text", []))
        case "quote":
            return "> " + _rich_text_to_str(data.get("rich_text", []))
        case "callout":
            return _rich_text_to_str(data.get("rich_text", []))
        case "code":
            lang = data.get("language", "")
            code = _rich_text_to_str(data.get("rich_text", []))
            return f"```{lang}\n{code}\n```"
        case "divider":
            return "---"
        case "table_row":
            cells = data.get("cells", [])
            return " | ".join(_rich_text_to_str(cell) for cell in cells)
        case _:
            # child_page, image, embed и прочее — пропускаем
            return ""


def _fetch_blocks(page_id: str) -> list[dict]:
    """Получить все блоки страницы (с пагинацией)."""
    notion = get_notion()
    blocks: list[dict] = []
    cursor = None

    while True:
        kwargs: dict[str, Any] = {"block_id": page_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor

        response = notion.blocks.children.list(**kwargs)
        blocks.extend(response.get("results", []))

        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")

    return blocks


def _page_to_text(page_id: str, title: str) -> str:
    """Конвертировать страницу Notion в plain text."""
    blocks = _fetch_blocks(page_id)
    lines = [f"# {title}", ""]

    for block in blocks:
        line = _block_to_text(block)
        if line:
            lines.append(line)

        # Если у блока есть дочерние элементы — рекурсивно разворачиваем
        if block.get("has_children") and block.get("type") not in ("child_page", "child_database"):
            child_blocks = _fetch_blocks(block["id"])
            for cb in child_blocks:
                child_line = _block_to_text(cb)
                if child_line:
                    lines.append("  " + child_line)

    return "\n".join(lines)


def _get_page_title(page: dict) -> str:
    """Извлечь заголовок страницы."""
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            return _rich_text_to_str(prop.get("title", []))
    return "Untitled"


def _get_page_url(page: dict) -> str:
    return page.get("url", "")


def _get_last_edited(page: dict) -> datetime:
    ts = page.get("last_edited_time", "")
    if ts:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return datetime.now(timezone.utc)


# ──────────────────────────────────────────────
# Работа с БД
# ──────────────────────────────────────────────

def _get_indexed_at(source_id: str) -> datetime | None:
    """Когда последний раз индексировали эту страницу."""
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                "SELECT MAX(indexed_at) as indexed_at FROM documents WHERE source_type = 'notion' AND source_id = %s",
                (source_id,),
            )
            row = cur.fetchone()
            if row and row["indexed_at"]:
                return row["indexed_at"]
    return None


def _delete_page_chunks(source_id: str, conn) -> int:
    """Удалить старые чанки страницы перед переиндексацией."""
    with get_cursor(conn) as cur:
        cur.execute(
            "DELETE FROM documents WHERE source_type = 'notion' AND source_id = %s",
            (source_id,),
        )
        return cur.rowcount


def _save_chunks(
    page_id: str,
    title: str,
    url: str,
    chunks_text: list[str],
    embeddings: list[list[float]],
    conn,
) -> int:
    """Сохранить чанки в БД. Возвращает количество сохранённых чанков."""
    total = len(chunks_text)
    with get_cursor(conn) as cur:
        for i, (text, embedding) in enumerate(zip(chunks_text, embeddings)):
            cur.execute(
                """
                INSERT INTO documents
                    (source_type, source_id, source_url, title, content, chunk_index, chunk_total, embedding)
                VALUES
                    ('notion', %s, %s, %s, %s, %s, %s, %s)
                """,
                (page_id, url, title, text, i, total, embedding),
            )
    return total


def _log_index(source_id: str, status: str, chunks_count: int = 0, error: str | None = None, conn=None) -> None:
    def _do(c):
        with get_cursor(c) as cur:
            cur.execute(
                """
                INSERT INTO index_log (source_type, source_id, status, chunks_count, error_msg)
                VALUES ('notion', %s, %s, %s, %s)
                """,
                (source_id, status, chunks_count, error),
            )

    if conn:
        _do(conn)
    else:
        with get_conn() as c:
            _do(c)


# ──────────────────────────────────────────────
# Индексация одной страницы
# ──────────────────────────────────────────────

def index_page(page_id: str, force: bool = False) -> dict:
    """
    Проиндексировать одну страницу Notion.

    Args:
        page_id: ID страницы Notion
        force: Переиндексировать даже если не изменилась

    Returns:
        {"status": "indexed"|"skipped"|"failed", "chunks": int, "title": str}
    """
    notion = get_notion()

    try:
        page = notion.pages.retrieve(page_id)
    except APIResponseError as e:
        logger.error(f"Failed to fetch page {page_id}: {e}")
        _log_index(page_id, "failed", error=str(e))
        return {"status": "failed", "chunks": 0, "title": "unknown"}

    title = _get_page_title(page)
    url = _get_page_url(page)
    last_edited = _get_last_edited(page)

    # Проверяем нужна ли переиндексация
    if not force:
        indexed_at = _get_indexed_at(page_id)
        if indexed_at and indexed_at >= last_edited:
            logger.debug(f"Skipping '{title}' — not changed since {indexed_at}")
            return {"status": "skipped", "chunks": 0, "title": title}

    logger.info(f"Indexing: '{title}' ({page_id})")

    try:
        # Конвертируем в текст
        text = _page_to_text(page_id, title)
        if not text.strip():
            logger.warning(f"Page '{title}' is empty, skipping")
            return {"status": "skipped", "chunks": 0, "title": title}

        # Chunking
        chunks = chunk_document(text)
        if not chunks:
            return {"status": "skipped", "chunks": 0, "title": title}

        # Добавляем заголовок страницы в начало каждого чанка —
        # это критично для качества поиска: чанк знает откуда он
        chunk_texts = [f"[{title}]\n{c.text}" for c in chunks]

        # Embedding
        embeddings = embed_texts(chunk_texts)

        # Сохраняем в БД (атомарно: сначала удаляем старое, потом вставляем новое)
        with get_conn() as conn:
            deleted = _delete_page_chunks(page_id, conn)
            if deleted > 0:
                logger.debug(f"Deleted {deleted} old chunks for '{title}'")
            count = _save_chunks(page_id, title, url, chunk_texts, embeddings, conn)
            _log_index(page_id, "success", count, conn=conn)

        logger.info(f"✅ Indexed '{title}': {count} chunks")
        return {"status": "indexed", "chunks": count, "title": title}

    except Exception as e:
        logger.error(f"Failed to index '{title}': {e}", exc_info=True)
        _log_index(page_id, "failed", error=str(e))
        return {"status": "failed", "chunks": 0, "title": title}


# ──────────────────────────────────────────────
# Рекурсивный обход дерева страниц
# ──────────────────────────────────────────────

def _get_subpages(page_id: str) -> list[str]:
    """Получить ID дочерних страниц."""
    blocks = _fetch_blocks(page_id)
    subpage_ids = []
    for block in blocks:
        if block.get("type") == "child_page":
            subpage_ids.append(block["id"])
    return subpage_ids


def index_tree(root_page_id: str, force: bool = False, _depth: int = 0) -> dict:
    """
    Рекурсивно проиндексировать страницу и все её дочерние страницы.

    Returns:
        {"indexed": int, "skipped": int, "failed": int}
    """
    stats = {"indexed": 0, "skipped": 0, "failed": 0}

    if _depth > 10:  # Защита от глубокой рекурсии
        logger.warning(f"Max depth reached at {root_page_id}, stopping")
        return stats

    result = index_page(root_page_id, force=force)
    stats[result["status"]] += 1

    # Рекурсивно индексируем дочерние страницы
    try:
        subpages = _get_subpages(root_page_id)
        for subpage_id in subpages:
            sub_stats = index_tree(subpage_id, force=force, _depth=_depth + 1)
            for key in stats:
                stats[key] += sub_stats[key]
    except Exception as e:
        logger.error(f"Failed to get subpages of {root_page_id}: {e}")

    return stats


def run_full_index(page_ids: list[str] | None = None, force: bool = False) -> dict:
    """
    Запустить полную индексацию всех корневых страниц.

    Args:
        page_ids: Список ID страниц. Если None — берёт из конфига NOTION_ROOT_PAGE_IDS
        force: Переиндексировать всё без проверки дат

    Returns:
        {"indexed": int, "skipped": int, "failed": int}
    """
    roots = page_ids or config.NOTION_ROOT_PAGE_IDS
    if not roots:
        logger.warning("No root page IDs configured. Set NOTION_ROOT_PAGE_IDS in .env")
        return {"indexed": 0, "skipped": 0, "failed": 0}

    total_stats = {"indexed": 0, "skipped": 0, "failed": 0}
    logger.info(f"Starting Notion index: {len(roots)} root page(s), force={force}")

    for root_id in roots:
        stats = index_tree(root_id, force=force)
        for key in total_stats:
            total_stats[key] += stats[key]

    logger.info(
        f"Index complete: {total_stats['indexed']} indexed, "
        f"{total_stats['skipped']} skipped, {total_stats['failed']} failed"
    )
    return total_stats


if __name__ == "__main__":
    # Быстрый тест: python -m indexer.notion_indexer
    import sys
    logging.basicConfig(level=logging.INFO)

    page_ids = [arg for arg in sys.argv[1:] if not arg.startswith("--")] or None
    stats = run_full_index(page_ids=page_ids, force="--force" in sys.argv)
    print(f"\nResult: {stats}")
