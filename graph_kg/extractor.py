"""
Entity and relation extraction using Claude Haiku.
Сохраняет результаты в plain PostgreSQL таблицы kg_entities / kg_relations.
"""
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Из текста ниже извлеки все важные сущности и связи между ними.

Сущности (entities): люди, проекты, технологии, концепции, организации, методы.
Связи (relations): uses, implements, depends_on, created_by, part_of, related_to, contrasts_with.

Верни ТОЛЬКО JSON без markdown:
{
  "entities": [
    {"name": "FastAPI", "type": "technology", "description": "Python web framework"}
  ],
  "relations": [
    {"source": "AdashAI", "relation": "uses", "target": "FastAPI"}
  ]
}

Текст:
{text}"""


@dataclass
class Entity:
    name: str
    type: str
    description: str


@dataclass
class Relation:
    source: str
    relation: str
    target: str


def extract_entities_relations(text: str, title: str) -> tuple[list[Entity], list[Relation]]:
    """
    Извлечь сущности и связи из текста через Claude Haiku.
    Возвращает пустые списки при ошибке — никогда не бросает исключение.
    """
    if len(text) < 100:
        return [], []

    truncated = text[:2000]

    try:
        import anthropic
        from config import config as cfg

        client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=truncated)}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)

        entities = [
            Entity(
                name=e["name"],
                type=e.get("type", "concept"),
                description=e.get("description", ""),
            )
            for e in data.get("entities", [])
            if e.get("name")
        ]
        relations = [
            Relation(source=r["source"], relation=r["relation"], target=r["target"])
            for r in data.get("relations", [])
            if r.get("source") and r.get("target")
        ]

        logger.info(f"Extracted {len(entities)} entities, {len(relations)} relations from '{title}'")
        return entities, relations

    except Exception as e:
        logger.warning(f"Entity extraction failed for '{title}': {e}")
        return [], []


def save_to_db(entities: list[Entity], relations: list[Relation], doc_id: str, conn) -> None:
    """Сохранить entities и relations в plain PostgreSQL таблицы."""
    if not entities:
        return

    from core.db import get_cursor

    with get_cursor(conn) as cur:
        for entity in entities:
            cur.execute(
                """
                INSERT INTO kg_entities (name, type, description, doc_id)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    type = EXCLUDED.type,
                    description = EXCLUDED.description,
                    doc_id = EXCLUDED.doc_id
                """,
                (entity.name, entity.type, entity.description, doc_id),
            )

        for rel in relations:
            cur.execute(
                """
                INSERT INTO kg_relations (source_name, relation, target_name, doc_id)
                VALUES (%s, %s, %s, %s)
                """,
                (rel.source, rel.relation, rel.target, doc_id),
            )
