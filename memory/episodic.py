"""
Episodic Memory — факты о пользователе в PostgreSQL.

Хранит постоянные факты которые Claude должен знать всегда:
- Технические предпочтения (языки, стек, стиль)
- Контекст проектов
- Личные предпочтения (стиль общения, часовой пояс и т.д.)

Всегда читается при старте MCP сервера и передаётся в каждый ответ.
"""
import logging
from dataclasses import dataclass

from core.db import get_conn, get_cursor

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    id: int
    category: str
    key: str
    value: str
    confidence: float


# ──────────────────────────────────────────────
# CRUD операции
# ──────────────────────────────────────────────

def get_all_facts() -> list[Fact]:
    """Получить все факты, отсортированные по категории."""
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                "SELECT id, category, key, value, confidence FROM user_facts ORDER BY category, key"
            )
            rows = cur.fetchall()
    return [Fact(**row) for row in rows]


def upsert_fact(key: str, value: str, category: str = "general", confidence: float = 1.0) -> Fact:
    """Создать или обновить факт."""
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                """
                INSERT INTO user_facts (category, key, value, confidence)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (key)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    category = EXCLUDED.category,
                    confidence = EXCLUDED.confidence,
                    updated_at = NOW()
                RETURNING id, category, key, value, confidence
                """,
                (category, key, value, confidence),
            )
            row = cur.fetchone()
    return Fact(**row)


def delete_fact(key: str) -> bool:
    """Удалить факт по ключу."""
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute("DELETE FROM user_facts WHERE key = %s", (key,))
            return cur.rowcount > 0


def format_facts_for_context(facts: list[Fact]) -> str:
    """
    Форматировать факты для системного промпта Claude.
    Компактно, ~300-500 токенов.
    """
    if not facts:
        return ""

    # Группируем по категории
    by_category: dict[str, list[Fact]] = {}
    for fact in facts:
        by_category.setdefault(fact.category, []).append(fact)

    lines = ["## Личный контекст пользователя\n"]
    for category, cat_facts in sorted(by_category.items()):
        lines.append(f"**{category.capitalize()}:**")
        for fact in cat_facts:
            lines.append(f"- {fact.key}: {fact.value}")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Предустановленные факты (заполняются один раз)
# ──────────────────────────────────────────────

DEFAULT_FACTS = [
    ("skill", "languages", "Go, Python, Dart/Flutter", "personal"),
    ("skill", "stack", "Go + Flutter + PostgreSQL + Docker", "personal"),
    ("preference", "response_style", "Короткие ответы без воды, примеры кода приветствуются", "personal"),
    ("preference", "timezone", "Asia/Almaty (UTC+5)", "personal"),
    ("preference", "language", "Русский язык в общении", "personal"),
    ("personal", "name", "Давлат", "personal"),
]


def seed_default_facts() -> None:
    """Заполнить базовые факты если они ещё не существуют."""
    for key, value, category, _ in DEFAULT_FACTS:
        try:
            # Только если ещё нет
            with get_conn() as conn:
                with get_cursor(conn) as cur:
                    cur.execute("SELECT 1 FROM user_facts WHERE key = %s", (key,))
                    if not cur.fetchone():
                        upsert_fact(key, value, category)
                        logger.info(f"Seeded fact: {key} = {value}")
        except Exception as e:
            logger.error(f"Failed to seed fact {key}: {e}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    seed_default_facts()
    facts = get_all_facts()
    print(format_facts_for_context(facts))
