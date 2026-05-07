═══════════════════════════════════════════
ТЗ: File Indexer — kb_add_file (локальные файлы в KB)
Проект: Personal-Knowledge-Agent · Python 3.12
Дата: 2026-05-07
═══════════════════════════════════════════

## Цель
Добавить инструмент `kb_add_file` для индексации локальных файлов (.md, .txt, .pdf) в KB — расширить coverage с "только Notion" до локальной файловой системы.

## Контекст
Сейчас KB индексирует только Notion. Поле `source_type='file'` в схеме уже есть, но нет ни одного индексера для локальных файлов. Нужно минимальное решение: принять путь к файлу, прочитать, разбить на чанки, записать в documents. Поддержать .md/.txt (нативно) и .pdf (через pypdf).

## Файлы

**Создать:**
- `indexer/file_indexer.py` — функция `index_file(path, title)` → list[str] (тексты чанков)

**Трогать:**
- `agent_server/server.py` — добавить MCP tool `kb_add_file`

**Не трогать:**
- `core/chunker.py` — используем как есть
- `core/embedder.py` — используем как есть
- `schema.sql` — схема поддерживает source_type='file' уже

## Реализация

### 1. indexer/file_indexer.py — новый файл

```python
"""
File Indexer — читает локальные файлы и возвращает текст для индексации.
Поддерживаемые форматы: .md, .txt, .pdf
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_file(path: str) -> tuple[str, str]:
    """
    Читает файл и возвращает (title, text).
    
    Args:
        path: Абсолютный или относительный путь к файлу
        
    Returns:
        (title, text) — заголовок (имя файла без расширения) и текстовый контент
        
    Raises:
        FileNotFoundError: если файл не существует
        ValueError: если формат файла не поддерживается
    """
    p = Path(path).resolve()
    
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    suffix = p.suffix.lower()
    title = p.stem  # имя файла без расширения
    
    if suffix in (".md", ".txt"):
        text = p.read_text(encoding="utf-8", errors="replace")
        
    elif suffix == ".pdf":
        try:
            import pypdf
        except ImportError:
            raise ImportError("Для индексации PDF установи: pip install pypdf")
        
        reader = pypdf.PdfReader(str(p))
        pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[Стр. {i+1}]\n{page_text}")
        text = "\n\n".join(pages)
        
    else:
        raise ValueError(f"Неподдерживаемый формат: {suffix}. Поддерживается: .md, .txt, .pdf")
    
    if not text.strip():
        raise ValueError(f"Файл пустой или не удалось извлечь текст: {path}")
    
    logger.info(f"read_file: {p.name} ({len(text)} символов)")
    return title, text
```

### 2. agent_server/server.py — добавить `kb_add_file`

Добавить импорт в начало файла (после остальных imports):
```python
from indexer.file_indexer import read_file
```

Добавить новый MCP tool после `kb_add_url`:

```python
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
        file_title, text = read_file(path)
        resolved_title = title or file_title
        
        # source_id = абсолютный путь (для дедупликации и удаления)
        from pathlib import Path
        source_id = str(Path(path).resolve())
        
        # Удаляем старые чанки если файл переиндексируется
        from core.db import get_conn, get_cursor
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                cur.execute(
                    "DELETE FROM documents WHERE source_type = 'file' AND source_id = %s",
                    (source_id,)
                )
                old_count = cur.rowcount
                if old_count:
                    logger.info(f"kb_add_file: удалено {old_count} старых чанков для {source_id}")
        
        return kb_add_document(
            text=text,
            title=resolved_title,
            source_type="file",
            source_url=source_id,  # для attribution
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
```

### 3. requirements.txt — добавить pypdf

В конец файла:
```
pypdf>=4.0.0
```

(pypdf — чистый Python, без системных зависимостей, хороший extract_text для большинства PDF)

## Стандарты
- **Karpathy**: Simple — один файл, одна функция read_file(), переиспользуем chunk_document + embed_texts из существующего кода. Surgical — только два файла тронуты.
- **Dev**: `dev-standards:python-api` — graceful errors с понятными сообщениями, идемпотентность (повторное добавление = обновление).
- **Проект**: конвенция kb_add_document вызывается для записи (DRY), source_type='file', source_id=абсолютный путь.

## Что НЕ делать
- **НЕ делать watcher на папку** — это отдельная задача (будет в итерации 2)
- **НЕ поддерживать .docx** в этой итерации — усложнит, низкий приоритет
- **НЕ хранить бинарный файл** в БД — только извлечённый текст
- **НЕ падать если pypdf не установлен** — возвращать понятное сообщение с инструкцией установки

## Критерий готовности
- [ ] `kb_add_file("/path/to/file.md")` — индексирует файл, возвращает "✅ Добавлено: 'filename' — N чанков"
- [ ] `kb_add_file("/path/to/file.pdf")` — работает с PDF
- [ ] Повторный вызов с тем же путём — обновляет документ, не дублирует (проверить COUNT(*))
- [ ] `kb_add_file("/nonexistent.md")` — возвращает "❌ Файл не найден", не падает
- [ ] `kb_add_file("/file.xlsx")` — возвращает "❌ Неподдерживаемый формат"
- [ ] Karpathy review: код thoughtful, simple, surgical, goal-driven
