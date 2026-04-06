"""
Chunker — разбивает текст на чанки для индексации.

Стратегия: recursive text splitting
- Пытается разбить по параграфам → строкам → предложениям → словам
- Размер: 500 токенов, overlap: 50 токенов
- Точный подсчёт токенов через tiktoken (cl100k_base — модель OpenAI)
"""
import re
from dataclasses import dataclass

import tiktoken

from config import config

# cl100k_base — кодировка для text-embedding-3-small и GPT-4
_tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    token_count: int
    chunk_index: int
    chunk_total: int  # будет установлен после разбивки всего документа


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def split_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """
    Recursive text splitting.
    Разбиваем по убыванию приоритета разделителей:
    параграфы → строки → точки → пробелы
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, separators: list[str]) -> list[str]:
        if not text.strip():
            return []

        # Если текст уже влезает — возвращаем как есть
        if count_tokens(text) <= chunk_size:
            return [text.strip()]

        sep = separators[0]
        remaining_seps = separators[1:]

        # Разбиваем по текущему разделителю
        parts = text.split(sep) if sep else list(text)
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for part in parts:
            part_tokens = count_tokens(part + sep)

            # Если один фрагмент сам по себе больше chunk_size —
            # рекурсивно разбиваем его дальше
            if part_tokens > chunk_size and remaining_seps:
                if current_parts:
                    chunks.append(sep.join(current_parts).strip())
                    current_parts = []
                    current_tokens = 0
                chunks.extend(_split(part, remaining_seps))
                continue

            if current_tokens + part_tokens > chunk_size and current_parts:
                # Сохраняем текущий чанк
                chunks.append(sep.join(current_parts).strip())

                # Overlap: берём последние N токенов в следующий чанк
                overlap_parts: list[str] = []
                overlap_tokens = 0
                for p in reversed(current_parts):
                    p_tok = count_tokens(p + sep)
                    if overlap_tokens + p_tok <= chunk_overlap:
                        overlap_parts.insert(0, p)
                        overlap_tokens += p_tok
                    else:
                        break

                current_parts = overlap_parts
                current_tokens = overlap_tokens

            current_parts.append(part)
            current_tokens += part_tokens

        if current_parts:
            chunks.append(sep.join(current_parts).strip())

        return [c for c in chunks if c.strip()]

    return _split(text, separators)


def chunk_document(text: str, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """
    Разбить документ на чанки с метаданными.
    Возвращает список Chunk с заполненными chunk_index и chunk_total.
    """
    # Нормализуем пробелы и переносы
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    if not text:
        return []

    raw_chunks = split_text(text, chunk_size, chunk_overlap)
    total = len(raw_chunks)

    return [
        Chunk(
            text=chunk,
            token_count=count_tokens(chunk),
            chunk_index=i,
            chunk_total=total,
        )
        for i, chunk in enumerate(raw_chunks)
    ]
