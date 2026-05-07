"""
Wiki Extractor — читает URL и извлекает структурированные инсайты для wiki.
Использует Claude Haiku для дешёвого извлечения.
"""
import logging
import re

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
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    truncated = text[:8000] if len(text) > 8000 else text

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": EXTRACT_PROMPT.format(url=url, date=date, text=truncated),
        }],
    )

    return response.content[0].text.strip()


def sanitize_topic_name(topic: str) -> str:
    """
    Нормализовать имя темы в snake_case для имени файла.
    Пример: "RAG Best Practices 2025" → "rag_best_practices_2025"
    """
    name = topic.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s-]+", "_", name)
    name = name.strip("_")
    return name or "misc"
