"""
Community detection и генерация summary для GraphRAG (LightRAG-style global retrieval).

Алгоритм:
1. Загружаем граф из kg_entities + kg_relations
2. Запускаем Louvain community detection
3. Для каждого community (>= 3 nodes) генерируем summary через Claude Haiku
4. Эмбеддим summary и сохраняем в community_summaries
"""
import json
import logging

from core.db import get_conn, get_cursor
from core.embedder import embed_texts

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """Ты аналитик знаний. Ниже список сущностей и связей между ними из базы знаний.

Сущности:
{entities}

Связи:
{relations}

Напиши краткое аналитическое резюме (3-5 предложений) этой группы концепций:
- Что объединяет эти сущности?
- Какова главная тема или область знаний?
- Какие ключевые связи наиболее важны?

Дай также короткое название (до 5 слов) для этой группы.

Верни JSON без markdown:
{{"title": "...", "summary": "..."}}"""


def _load_graph():
    """Загрузить граф из PostgreSQL в NetworkX."""
    import networkx as nx

    G = nx.Graph()

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute("SELECT name, type, description FROM kg_entities")
            for row in cur.fetchall():
                G.add_node(row["name"], type=row["type"], description=row["description"])

            cur.execute("SELECT source_name, relation, target_name FROM kg_relations")
            for row in cur.fetchall():
                src, rel, tgt = row["source_name"], row["relation"], row["target_name"]
                if G.has_node(src) and G.has_node(tgt):
                    G.add_edge(src, tgt, relation=rel)

    logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def _detect_communities(G) -> dict[str, int]:
    """Запустить Louvain community detection. Возвращает {node: community_id}."""
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
    except ImportError:
        logger.warning("python-louvain not installed, falling back to connected components")
        partition = {}
        for i, component in enumerate(G.__class__(G).connected_components() if hasattr(G.__class__, 'connected_components') else []):
            for node in component:
                partition[node] = i
        import networkx as nx
        partition = {}
        for i, component in enumerate(nx.connected_components(G)):
            for node in component:
                partition[node] = i
    return partition


def _generate_summary(community_id: int, nodes: list[str], G) -> tuple[str, str] | None:
    """Сгенерировать title + summary для community через Claude Haiku."""
    entity_lines = []
    for node in nodes[:30]:
        data = G.nodes[node]
        desc = data.get("description", "")
        entity_lines.append(f"- {node} ({data.get('type', 'concept')})" + (f": {desc}" if desc else ""))

    relation_lines = []
    for u, v, data in G.edges(nodes, data=True):
        if u in set(nodes) and v in set(nodes):
            relation_lines.append(f"- {u} --[{data.get('relation', 'related')}]--> {v}")
    relation_lines = relation_lines[:30]

    prompt = _SUMMARY_PROMPT.format(
        entities="\n".join(entity_lines) or "(нет сущностей)",
        relations="\n".join(relation_lines) or "(нет связей)",
    )

    try:
        import anthropic
        from config import config as cfg

        client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        title = data.get("title", f"Community {community_id}")
        summary = data.get("summary", "")
        return title, summary

    except Exception as e:
        logger.warning(f"Summary generation failed for community {community_id}: {e}")
        node_sample = ", ".join(nodes[:5])
        return f"Community {community_id}", f"Группа концепций: {node_sample}..."


def _upsert_community(community_id: str, title: str, summary: str, entity_count: int, embedding: list[float], conn) -> None:
    with get_cursor(conn) as cur:
        cur.execute(
            """
            INSERT INTO community_summaries (community_id, title, summary, entity_count, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            ON CONFLICT (community_id) DO UPDATE SET
                title = EXCLUDED.title,
                summary = EXCLUDED.summary,
                entity_count = EXCLUDED.entity_count,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
            """,
            (community_id, title, summary, entity_count, embedding),
        )


def generate_community_summaries(min_size: int = 3) -> int:
    """
    Полный цикл: загрузка графа → community detection → summary → embedding → upsert.

    Args:
        min_size: минимальный размер community (меньше — игнорируем)

    Returns:
        Количество созданных/обновлённых community summaries
    """
    G = _load_graph()

    if G.number_of_nodes() == 0:
        logger.info("Graph is empty, skipping community detection")
        return 0

    partition = _detect_communities(G)

    communities: dict[int, list[str]] = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)

    large_communities = {cid: nodes for cid, nodes in communities.items() if len(nodes) >= min_size}
    logger.info(f"Found {len(communities)} communities, {len(large_communities)} with >= {min_size} nodes")

    if not large_communities:
        return 0

    summaries_data = []
    for cid, nodes in large_communities.items():
        title, summary = _generate_summary(cid, nodes, G)
        summaries_data.append((cid, nodes, title, summary))

    texts = [f"{title}\n{summary}" for _, _, title, summary in summaries_data]
    try:
        embeddings = embed_texts(texts)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return 0

    count = 0
    with get_conn() as conn:
        for (cid, nodes, title, summary), embedding in zip(summaries_data, embeddings):
            community_id = f"c{cid}"
            _upsert_community(community_id, title, summary, len(nodes), embedding, conn)
            count += 1
            logger.debug(f"Upserted community {community_id}: '{title}' ({len(nodes)} nodes)")

    logger.info(f"Community summaries upserted: {count}")
    return count
