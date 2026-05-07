═══════════════════════════════════════════
ТЗ: Итерация 3 — Karpathy Wiki (AI-managed markdown knowledge)
Проект: Personal-Knowledge-Agent · Python 3.12 + PostgreSQL 17
Дата: 2026-05-07
═══════════════════════════════════════════

## Цель
Добавить Karpathy-style Wiki — папку `wiki/` с markdown-файлами, которые поддерживаются AI-агентом: автоматически обновляются из URL/статей, линтуются на дубли и устаревшие факты, и участвуют в поиске как отдельный высокоприоритетный источник. Паттерн из Karpathy April 2026 gist: wiki ~70x эффективнее RAG для стабильного знания.

## Контекст
Текущая KB хранит всё как векторные чанки — эффективно для поиска, но плохо для стабильного, структурированного знания. Karpathy предлагает иное: папка wiki/ с markdown-файлами, куда агент сам вносит извлечённые инсайты из URL/статей. Файлы редактируются инкрементально — не пересоздаются. Это позволяет: (1) накапливать знание без дублирования, (2) делать explicit структуру тем (каждый .md = тема), (3) давать агенту прямой доступ к чтению файлов без поиска. При этом wiki/ файлы тоже индексируются в KB как source_type='wiki' — поиск охватывает оба слоя.

## Файлы

**Создать:**
- `wiki/` — папка в корне проекта, пустая при инициализации
- `wiki/README.md` — автогенерируемый индекс тем (обновляется при /wiki_lint)
- `wiki_agent/manager.py` — логика управления wiki (ingest, process, lint)
- `wiki_agent/extractor.py` — извлечение инсайтов из URL через LLM

**Трогать:**
- `agent_server/server.py` — добавить 4 новых MCP tool
- `indexer/file_indexer.py` — переиспользовать для индексации wiki-файлов в KB

**Не трогать:**
- `retriever/search.py` — wiki документы попадают в поиск через стандартный source_type фильтр
- `schema.sql` — source_type='wiki' уже поддерживается VARCHAR
- `core/chunker.py`, `core/embedder.py` — без изменений

## Реализация

### 1. wiki_agent/extractor.py — извлечение инсайтов из URL

```python
"""
Wiki Extractor — читает URL и извлекает структурированные инсайты для wiki.
Использует Claude Haiku для дешёвого извлечения.
"""
import logging
import re
from pathlib import Path
import anthropic

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """Ты — ассистент для ведения базы знаний в формате wiki.

Прочитай статью и извлеки из неё структурированные инсайты.

Правила:
- Пиши кратко и по существу (не пересказывай, а выделяй суть)
- Используй markdown: заголовки ##, списки, **важное**
- Структурируй по подтемам если статья широкая
- В конце добавь раздел ## Источник с URL и датой
- НЕ включай тривиальное и общеизвестное
- Максимум 500 слов

URL: {url}
Дата: {date}

Текст статьи:
{text}

---
Выдай markdown-инсайты:"""


def extract_insights_from_text(
    text: str,
    url: str,
    topic: str,
    date: str,
    api_key: str,
) -> str:
    """
    Извлечь ключевые инсайты из текста статьи через Claude Haiku.
    
    Returns:
        Markdown-текст с инсайтами, готовый для добавления в wiki-файл
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    # Обрезать текст до разумного размера (избегаем длинных промптов)
    truncated = text[:8000] if len(text) > 8000 else text
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": EXTRACT_PROMPT.format(
                url=url,
                date=date,
                text=truncated,
            )
        }]
    )
    
    return response.content[0].text.strip()


def sanitize_topic_name(topic: str) -> str:
    """
    Нормализовать имя темы в snake_case для имени файла.
    Пример: "RAG Best Practices 2025" → "rag_best_practices_2025"
    """
    # Нижний регистр, пробелы и спецсимволы → underscore
    name = topic.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s-]+", "_", name)
    name = name.strip("_")
    return name or "misc"
```

### 2. wiki_agent/manager.py — управление wiki-файлами

```python
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
from typing import Optional

logger = logging.getLogger(__name__)

WIKI_DIR = Path("wiki")


def _ensure_wiki_dir():
    """Создать wiki/ если не существует."""
    WIKI_DIR.mkdir(exist_ok=True)
    inbox = WIKI_DIR / "INBOX.md"
    if not inbox.exists():
        inbox.write_text("# Wiki Inbox\n\nДобавляй URL для обработки по одному на строку:\n\n", encoding="utf-8")


def _get_wiki_file(topic: str) -> Path:
    """Получить путь к wiki-файлу для темы."""
    from wiki_agent.extractor import sanitize_topic_name
    filename = sanitize_topic_name(topic) + ".md"
    return WIKI_DIR / filename


def ingest_url(
    url: str,
    topic: str,
    api_key: str,
    voyage_api_key: str,
) -> dict:
    """
    Загрузить URL, извлечь инсайты, добавить в wiki-файл темы.
    
    Если файл темы существует — инсайты ДОБАВЛЯЮТСЯ в конец (не перезаписываются).
    Если нет — создаётся новый файл с заголовком темы.
    
    Returns:
        dict с 'action': 'created' | 'appended', 'file': path, 'words_added': int
    """
    import httpx
    from wiki_agent.extractor import extract_insights_from_text
    _ensure_wiki_dir()
    
    # Загрузить URL
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        # Простое извлечение текста (без зависимости от BS4)
        text = resp.text
        # Убрать HTML-теги простым regex (достаточно для большинства статей)
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
        # Создать новый файл с заголовком
        header = f"# {topic}\n\n_Создан: {today}_\n\n---\n\n"
        wiki_file.write_text(header + insights + "\n", encoding="utf-8")
        action = "created"
    else:
        # Добавить в конец
        existing = wiki_file.read_text(encoding="utf-8")
        separator = f"\n\n---\n_Обновлено: {today}_\n\n"
        wiki_file.write_text(existing + separator + insights + "\n", encoding="utf-8")
        action = "appended"
    
    words_added = len(insights.split())
    logger.info(f"wiki.ingest_url: {action} {wiki_file.name} (+{words_added} слов)")
    
    # Переиндексировать wiki-файл в KB
    _reindex_wiki_file(wiki_file, topic, voyage_api_key)
    
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
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("_"):
            remaining_lines.append(line)
            continue
        
        if "|" not in line:
            remaining_lines.append(line)
            continue
        
        parts = line.split("|", 1)
        url = parts[0].strip()
        topic = parts[1].strip() if len(parts) > 1 else "misc"
        
        if not url.startswith("http"):
            remaining_lines.append(line)
            continue
        
        try:
            result = ingest_url(url, topic, api_key, voyage_api_key)
            processed.append({"url": url, "topic": topic, "result": result})
            # Не добавляем в remaining — строка обработана
        except Exception as e:
            errors.append({"url": url, "error": str(e)})
            remaining_lines.append(f"# ERROR: {line}  ← {e}")
    
    # Перезаписать INBOX без обработанных строк
    inbox.write_text("\n".join(remaining_lines) + "\n", encoding="utf-8")
    
    return {"processed": processed, "errors": errors}


def lint_wiki(voyage_api_key: str) -> dict:
    """
    Аудит wiki/:
    1. Найти файлы не обновлявшиеся > 90 дней
    2. Найти потенциальные дубли по именам файлов (fuzzy)
    3. Обновить wiki/README.md с индексом всех тем
    4. Вернуть статистику
    
    Returns:
        dict с 'stale': list, 'duplicates': list, 'total_files': int, 'readme_updated': bool
    """
    _ensure_wiki_dir()
    
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    stale_threshold = now - timedelta(days=90)
    
    wiki_files = sorted(WIKI_DIR.glob("*.md"))
    # Исключить служебные файлы
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
    
    # Простой поиск дублей: файлы у которых >70% совпадение в имени (без расширения)
    duplicates = []
    names = [f.stem for f in topic_files]
    for i, name_a in enumerate(names):
        for name_b in names[i+1:]:
            # Jaccard similarity по словам в имени
            words_a = set(name_a.split("_"))
            words_b = set(name_b.split("_"))
            if len(words_a) == 0 or len(words_b) == 0:
                continue
            jaccard = len(words_a & words_b) / len(words_a | words_b)
            if jaccard > 0.7:
                duplicates.append({"file_a": name_a, "file_b": name_b, "similarity": round(jaccard, 2)})
    
    # Обновить README.md
    readme_lines = [
        "# Wiki Index\n",
        f"_Обновлено: {now.strftime('%Y-%m-%d')}_\n",
        f"_Всего тем: {len(topic_files)}_\n\n",
        "## Темы\n",
    ]
    for entry in sorted(index_entries, key=lambda x: x["file"]):
        readme_lines.append(f"- [{entry['file'].replace('_', ' ').replace('.md', '')}](./{entry['file']}) — {entry['words']} слов, обновлено {entry['last_modified']}")
    
    if stale:
        readme_lines.append("\n## ⚠️ Устаревшие (>90 дней)\n")
        for s in stale:
            readme_lines.append(f"- {s['file']} (последнее обновление: {s['last_modified']})")
    
    readme = WIKI_DIR / "README.md"
    readme.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    
    # Переиндексировать все изменённые wiki-файлы в KB (batch)
    for f in topic_files:
        _reindex_wiki_file(f, f.stem.replace("_", " ").title(), voyage_api_key)
    
    return {
        "total_files": len(topic_files),
        "stale_count": len(stale),
        "stale": stale,
        "duplicates": duplicates,
        "readme_updated": True,
    }


def _reindex_wiki_file(wiki_file: Path, topic: str, voyage_api_key: str):
    """
    Переиндексировать wiki-файл в KB (documents таблицу).
    Удаляет старые чанки файла, вставляет новые.
    """
    try:
        from indexer.file_indexer import read_file
        from core.db import get_conn, get_cursor
        from core.chunker import chunk_document
        from core.embedder import embed_texts
        import hashlib
        from datetime import date
        
        _, text = read_file(str(wiki_file))
        source_id = hashlib.md5(str(wiki_file.resolve()).encode()).hexdigest()
        
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                # Удалить старые чанки
                cur.execute(
                    "DELETE FROM documents WHERE source_type = 'wiki' AND source_id = %s",
                    (source_id,)
                )
                
                # Разбить на чанки и эмбеддировать
                chunks = chunk_document(text)
                if not chunks:
                    return
                    
                embeddings = embed_texts(chunks)
                today = date.today().isoformat()
                
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    cur.execute("""
                        INSERT INTO documents
                            (source_type, source_id, title, content, chunk_index, chunk_total,
                             embedding, metadata, indexed_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, NOW())
                    """, (
                        "wiki",
                        source_id,
                        f"Wiki: {topic}",
                        chunk,
                        i,
                        len(chunks),
                        str(emb),
                        '{"wiki": true}',
                    ))
    except Exception as e:
        logger.warning(f"wiki._reindex_wiki_file failed for {wiki_file.name}: {e}")
```

### 3. agent_server/server.py — добавить 4 новых MCP tool

Добавить импорт после остальных:
```python
from wiki_agent.manager import ingest_url as wiki_ingest_url_impl
from wiki_agent.manager import process_inbox as wiki_process_inbox_impl
from wiki_agent.manager import lint_wiki as wiki_lint_impl
```

Добавить конфиг-переменную (если не в config.py):
```python
# В config.py добавить:
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
```

Добавить 4 новых tool в server.py:

```python
@mcp.tool(
    annotations={"readOnlyHint": False, "openWorldHint": True},
)
def wiki_ingest_url(
    url: str,
    topic: str,
) -> str:
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
        return (
            f"{action_emoji} Wiki {'создан' if result['action'] == 'created' else 'обновлён'}: "
            f"{result['file']}\n"
            f"  Добавлено: {result['words_added']} слов\n"
            f"  Тема: {result['topic']}"
        )
    except Exception as e:
        logger.error(f"wiki_ingest_url error: {e}", exc_info=True)
        return f"❌ Ошибка: {e}"


@mcp.tool(
    annotations={"readOnlyHint": False, "openWorldHint": True},
)
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
        lines = [f"📥 Wiki Inbox обработан:"]
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


@mcp.tool(
    annotations={"readOnlyHint": False, "openWorldHint": False},
)
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
        lines = [f"🔍 Wiki Lint завершён:"]
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


@mcp.tool(
    annotations={"readOnlyHint": True},
)
def wiki_list_topics() -> str:
    """
    Показать все темы в wiki/ с краткой статистикой.
    
    Returns:
        Список wiki-файлов с размером и датой последнего обновления
    """
    from pathlib import Path
    from datetime import datetime, timezone
    
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
        content = f.read_text(encoding="utf-8")
        word_count = len(content.split())
        topic_name = f.stem.replace("_", " ").title()
        lines.append(f"  📄 {topic_name}")
        lines.append(f"     Файл: {f.name} | Слов: {word_count} | Обновлён: {mtime.strftime('%Y-%m-%d')}")
    
    return "\n".join(lines)
```

### 4. .env.example — добавить переменную

```
ANTHROPIC_API_KEY=sk-ant-...
```

(Нужен для Claude Haiku вызовов при извлечении инсайтов)

### 5. requirements.txt — добавить зависимость

```
anthropic>=0.25.0
httpx>=0.27.0
```

(httpx для синхронных HTTP запросов в extractor; anthropic уже может быть установлен — проверить)

## Стандарты
- **Karpathy**: Simple — wiki/ это просто папка с .md файлами, максимально прозрачно. Surgical — только server.py и новые файлы wiki_agent/. Thoughtful — инсайты добавляются (append), не перезаписываются; fallback если URL не доступен.
- **Dev**: `dev-standards:python-api` — graceful errors, lazy imports для тяжёлых зависимостей, идемпотентный lint.
- **Dev**: `dev-standards:ai-llm` — Haiku (дешевле), обрезка текста до 8000 символов, retry не нужен (wiki не критично real-time).
- **Проект**: синхронный psycopg2 через get_conn()/get_cursor(), chunk_document + embed_texts переиспользуются, source_type='wiki'.

## Что НЕ делать
- **НЕ использовать BeautifulSoup** — достаточно regex-стриппинга тегов, не добавляем зависимость
- **НЕ делать watcher** на wiki/ — только явные вызовы tools (wiki_ingest_url, wiki_lint)
- **НЕ перезаписывать весь wiki-файл при добавлении** — только append секции
- **НЕ делать автоматическое слияние дублей** — только выявлять и сообщать пользователю
- **НЕ требовать anthropic SDK** при старте — lazy import в extractor.py
- **НЕ падать если wiki/ пуста** — все операции должны работать с пустой папкой

## Критерий готовности
- [ ] `wiki_ingest_url("https://arxiv.org/...", "LightRAG")` — создаёт `wiki/lightrag.md` с инсайтами
- [ ] Повторный вызов того же URL с той же темой — добавляет секцию в конец файла (не перезаписывает)
- [ ] `wiki_process_inbox()` после добавления 2 строк в INBOX.md — обрабатывает оба URL, очищает INBOX
- [ ] `wiki_lint()` — создаёт/обновляет `wiki/README.md` с индексом тем
- [ ] `wiki_list_topics()` — показывает все .md файлы в wiki/ с числом слов
- [ ] `kb_search("LightRAG")` — находит wiki-чанки (source_type='wiki') наряду с KB-чанками
- [ ] При недоступном URL — wiki_ingest_url возвращает "❌ Не удалось загрузить URL", не падает
- [ ] При отсутствии ANTHROPIC_API_KEY — понятная ошибка, не трейсбэк
- [ ] Karpathy review: код thoughtful, simple, surgical, goal-driven
