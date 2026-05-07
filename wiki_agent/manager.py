"""
Wiki Manager — операции над папкой wiki/:
- ingest_url: загрузить URL → извлечь инсайты → добавить/обновить wiki-файл
- process_inbox: обработать список URL из wiki/INBOX.md
- lint: найти дубли, устаревшие файлы, обновить README
"""
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

WIKI_DIR = Path("wiki")


def _ensure_wiki_dir():
    WIKI_DIR.mkdir(exist_ok=True)
    inbox = WIKI_DIR / "INBOX.md"
    if not inbox.exists():
        inbox.write_text(
            "# Wiki Inbox\n\nДобавляй URL для обработки по одному на строку:\n\n",
            encoding="utf-8",
        )


def _get_wiki_file(topic: str) -> Path:
    from wiki_agent.extractor import sanitize_topic_name
    return WIKI_DIR / (sanitize_topic_name(topic) + ".md")


def ingest_url(url: str, topic: str, api_key: str, voyage_api_key: str) -> dict:
    """
    Загрузить URL, извлечь инсайты, добавить в wiki-файл темы.

    Если файл темы существует — инсайты ДОБАВЛЯЮТСЯ в конец.
    Если нет — создаётся новый файл с заголовком темы.

    Returns:
        dict с 'action': 'created' | 'appended', 'file': path, 'words_added': int
    """
    import httpx
    from wiki_agent.extractor import extract_insights_from_text

    _ensure_wiki_dir()

    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        raise ValueError(f"Не удалось загрузить URL: {e}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    insights = extract_insights_from_text(
        text=text,
        url=url,
        topic=topic,
        date=today,
        api_key=api_key,
    )

    wiki_file = _get_wiki_file(topic)

    if not wiki_file.exists():
        header = f"# {topic}\n\n_Создан: {today}_\n\n---\n\n"
        wiki_file.write_text(header + insights + "\n", encoding="utf-8")
        action = "created"
    else:
        existing = wiki_file.read_text(encoding="utf-8")
        separator = f"\n\n---\n_Обновлено: {today}_\n\n"
        wiki_file.write_text(existing + separator + insights + "\n", encoding="utf-8")
        action = "appended"

    words_added = len(insights.split())
    logger.info(f"wiki.ingest_url: {action} {wiki_file.name} (+{words_added} слов)")

    _reindex_wiki_file(wiki_file, topic)

    return {
        "action": action,
        "file": str(wiki_file),
        "topic": topic,
        "words_added": words_added,
    }


def process_inbox(api_key: str, voyage_api_key: str) -> dict:
    """
    Обработать все URL из wiki/INBOX.md.

    Формат INBOX.md:
        url | topic
        https://... | RAG Best Practices

    После обработки — очистить обработанные строки из INBOX.

    Returns:
        dict с 'processed': list, 'errors': list
    """
    _ensure_wiki_dir()
    inbox = WIKI_DIR / "INBOX.md"

    if not inbox.exists():
        return {"processed": [], "errors": ["INBOX.md не найден"]}

    content = inbox.read_text(encoding="utf-8")
    lines = content.split("\n")

    processed = []
    errors = []
    remaining_lines = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith("#") or line_stripped.startswith("_"):
            remaining_lines.append(line)
            continue

        if "|" not in line_stripped:
            remaining_lines.append(line)
            continue

        parts = line_stripped.split("|", 1)
        url = parts[0].strip()
        topic = parts[1].strip() if len(parts) > 1 else "misc"

        if not url.startswith("http"):
            remaining_lines.append(line)
            continue

        try:
            result = ingest_url(url, topic, api_key, voyage_api_key)
            processed.append({"url": url, "topic": topic, "result": result})
        except Exception as e:
            errors.append({"url": url, "error": str(e)})
            remaining_lines.append(f"# ERROR: {line_stripped}  ← {e}")

    inbox.write_text("\n".join(remaining_lines) + "\n", encoding="utf-8")
    return {"processed": processed, "errors": errors}


def lint_wiki(voyage_api_key: str) -> dict:
    """
    Аудит wiki/:
    1. Найти файлы не обновлявшиеся > 90 дней
    2. Найти потенциальные дубли по именам файлов (fuzzy)
    3. Обновить wiki/README.md с индексом всех тем
    4. Вернуть статистику
    """
    from datetime import timedelta

    _ensure_wiki_dir()
    now = datetime.now(timezone.utc)
    stale_threshold = now - timedelta(days=90)

    wiki_files = sorted(WIKI_DIR.glob("*.md"))
    topic_files = [f for f in wiki_files if f.name not in ("README.md", "INBOX.md")]

    stale = []
    index_entries = []

    for f in topic_files:
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        content = f.read_text(encoding="utf-8")
        word_count = len(content.split())

        entry = {
            "file": f.name,
            "words": word_count,
            "last_modified": mtime.strftime("%Y-%m-%d"),
        }
        index_entries.append(entry)

        if mtime < stale_threshold:
            stale.append(entry)

    # Jaccard similarity по словам в имени файла
    duplicates = []
    names = [f.stem for f in topic_files]
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            words_a = set(name_a.split("_"))
            words_b = set(name_b.split("_"))
            if not words_a or not words_b:
                continue
            jaccard = len(words_a & words_b) / len(words_a | words_b)
            if jaccard > 0.7:
                duplicates.append({
                    "file_a": name_a,
                    "file_b": name_b,
                    "similarity": round(jaccard, 2),
                })

    readme_lines = [
        "# Wiki Index\n",
        f"_Обновлено: {now.strftime('%Y-%m-%d')}_\n",
        f"_Всего тем: {len(topic_files)}_\n\n",
        "## Темы\n",
    ]
    for entry in sorted(index_entries, key=lambda x: x["file"]):
        name = entry["file"].replace("_", " ").replace(".md", "")
        readme_lines.append(
            f"- [{name}](./{entry['file']}) — {entry['words']} слов, обновлено {entry['last_modified']}"
        )

    if stale:
        readme_lines.append("\n## ⚠️ Устаревшие (>90 дней)\n")
        for s in stale:
            readme_lines.append(f"- {s['file']} (последнее обновление: {s['last_modified']})")

    readme = WIKI_DIR / "README.md"
    readme.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    for f in topic_files:
        _reindex_wiki_file(f, f.stem.replace("_", " ").title())

    return {
        "total_files": len(topic_files),
        "stale_count": len(stale),
        "stale": stale,
        "duplicates": duplicates,
        "readme_updated": True,
    }


def _reindex_wiki_file(wiki_file: Path, topic: str) -> None:
    """Переиндексировать wiki-файл в KB (idempotent: удалить старые чанки, вставить новые)."""
    try:
        import hashlib
        from core.chunker import chunk_document
        from core.db import get_conn, get_cursor
        from core.embedder import embed_texts

        text = wiki_file.read_text(encoding="utf-8")
        if not text.strip():
            return

        source_id = hashlib.md5(str(wiki_file.resolve()).encode()).hexdigest()

        chunks = chunk_document(text)
        if not chunks:
            return

        chunk_texts = [c.text for c in chunks]
        embeddings = embed_texts(chunk_texts)

        with get_conn() as conn:
            with get_cursor(conn) as cur:
                cur.execute(
                    "DELETE FROM documents WHERE source_type = 'wiki' AND source_id = %s",
                    (source_id,),
                )
                for i, (chunk_text, emb) in enumerate(zip(chunk_texts, embeddings)):
                    cur.execute(
                        """
                        INSERT INTO documents
                            (source_type, source_id, title, content, chunk_index, chunk_total, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        ("wiki", source_id, f"Wiki: {topic}", chunk_text, i, len(chunks), emb),
                    )
    except Exception as e:
        logger.warning(f"wiki._reindex_wiki_file failed for {wiki_file.name}: {e}")
