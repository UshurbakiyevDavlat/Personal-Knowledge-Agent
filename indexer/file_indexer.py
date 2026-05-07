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
        ValueError: если формат файла не поддерживается или файл пустой
    """
    p = Path(path).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    suffix = p.suffix.lower()
    title = p.stem

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
                pages.append(f"[Стр. {i + 1}]\n{page_text}")
        text = "\n\n".join(pages)

    else:
        raise ValueError(
            f"Неподдерживаемый формат: {suffix}. Поддерживается: .md, .txt, .pdf"
        )

    if not text.strip():
        raise ValueError(f"Файл пустой или не удалось извлечь текст: {path}")

    logger.info(f"read_file: {p.name} ({len(text)} символов)")
    return title, text
