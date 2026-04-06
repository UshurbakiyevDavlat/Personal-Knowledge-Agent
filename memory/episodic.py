"""
Episodic Memory — факты о пользователе в PostgreSQL.

Хранит постоянные факты которые Claude должен знать всегда:
- Личный контекст (имя, локация, стек)
- Опыт и навыки
- Активные проекты
- Предпочтения в общении

Всегда читается MCP сервером и передаётся в каждый ответ.
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
# Базовые факты — собраны из LinkedIn, GitHub, Notion
# ──────────────────────────────────────────────

DEFAULT_FACTS: list[tuple[str, str, str]] = [
    # ── Личное ──────────────────────────────────────────────
    ("name",            "Давлат (Davlatbek Ushurbakiyev)",                           "personal"),
    ("location",        "Алматы, Казахстан",                                         "personal"),
    ("timezone",        "Asia/Almaty (UTC+5)",                                       "personal"),
    ("github",          "https://github.com/UshurbakiyevDavlat",                     "personal"),
    ("linkedin",        "https://www.linkedin.com/in/davlatbeku/",                   "personal"),

    # ── Опыт ────────────────────────────────────────────────
    ("experience_years","~4 года backend-разработки",                                "experience"),
    ("education",       "Бакалавр Computer Science, IITU Алматы, GPA 3.5 (2019-2023)", "experience"),
    ("last_position",   "Software Developer, Mercury Solutions / Pinemelon.com (Go, PHP, Vue, Docker), до фев 2025", "experience"),
    ("prev_companies",  "Freedom Broker (Vue, trading), Alma Telecom (PHP), Tredo/Sxodim.com (Go, Laravel, Elasticsearch)", "experience"),
    ("certification",   "Go: The Complete Developer's Guide — Udemy (сент 2025)",    "experience"),

    # ── Технический стек ────────────────────────────────────
    ("primary_languages","Go, PHP/Laravel, Python, TypeScript",                      "skill"),
    ("frontend",        "Vue.js (Vue 2/3)",                                          "skill"),
    ("databases",       "PostgreSQL, MySQL, ClickHouse, Redis, Elasticsearch",       "skill"),
    ("devops",          "Docker, Docker Swarm, Nginx, CI/CD",                        "skill"),
    ("current_focus",   "AI Engineering: MCP серверы, RAG системы, Python агенты",  "skill"),
    ("architecture",    "Clean Architecture, DDD, микросервисы",                    "skill"),

    # ── Активные AI проекты ─────────────────────────────────
    ("project_knowledge_agent", "Personal Knowledge Agent — RAG на pgvector + Voyage AI, В разработке (апрель 2026)", "project"),
    ("project_linkedin_mcp",    "LinkedIn MCP Server — MCP для LinkedIn (JavaScript, готов)",                         "project"),
    ("project_postgres_mcp",    "PostgresMCP — MCP сервер для PostgreSQL (TypeScript, готов)",                        "project"),
    ("project_telegram_agent",  "Telegram Agent — личный AI ассистент в Telegram (в планах)",                        "project"),
    ("project_morning_digest",  "Morning Digest Agent — утренний AI дайджест (в планах)",                            "project"),
    ("project_operator_agent",  "OperatorAgent — AI агент для операторов, актуальная бизнес-идея (в планах)",        "project"),
    ("project_mycode_cli",      "MyCode CLI — аналог Claude Code, свой CLI (в планах)",                              "project"),

    # ── Рабочий контекст ────────────────────────────────────
    ("work_project",    "mercuryx (MPS) — основной рабочий проект на Go",            "work"),
    ("work_stack",      "Go, Kafka, Redis, PostgreSQL, Docker в рамках MPS проекта", "work"),
    ("mcp_ideas",       "Планируемые MCP: Redis, Weather, Docker, HTTP/REST, Kafka, Grafana, Habit Tracker", "work"),

    # ── Предпочтения ────────────────────────────────────────
    ("language",        "Русский язык в общении",                                    "preference"),
    ("response_style",  "Короткие ответы без воды, примеры кода приветствуются",     "preference"),
    ("tools",           "Cowork + Claude Code как основные AI инструменты",          "preference"),
]


def seed_default_facts() -> None:
    """Заполнить базовые факты если они ещё не существуют."""
    seeded = 0
    for key, value, category in DEFAULT_FACTS:
        try:
            with get_conn() as conn:
                with get_cursor(conn) as cur:
                    cur.execute("SELECT 1 FROM user_facts WHERE key = %s", (key,))
                    if not cur.fetchone():
                        upsert_fact(key, value, category)
                        seeded += 1
        except Exception as e:
            logger.error(f"Failed to seed fact '{key}': {e}")

    if seeded:
        logger.info(f"Seeded {seeded} new facts")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    seed_default_facts()
    facts = get_all_facts()
    print(format_facts_for_context(facts))
