═══════════════════════════════════════════
ТЗ: Итерация 3 — Knowledge Graph Visualization (Web UI)
Проект: Personal-Knowledge-Agent · Python 3.12 + PostgreSQL 17
Дата: 2026-05-07
═══════════════════════════════════════════

## Цель
Добавить интерактивный веб-интерфейс для визуализации знаний — граф связей между документами, кластеры тем (communities из GraphRAG), wiki-темы и факты. Открывается в браузере, рендерит граф Obsidian-style с нодами и рёбрами. Без отдельного сервера — статический HTML + fetch к существующему MCP-серверу через REST API bridge.

## Контекст
Сейчас всё взаимодействие с KB идёт через текстовые MCP-tools. Нет никакого способа "увидеть" базу знаний визуально: что в ней есть, как документы связаны, где кластеры тем. Obsidian даёт это через граф заметок — мы хотим то же самое для KB. Реализация: самодостаточный `visualization/graph.html` + Flask micro-API (`visualization/api.py`) который читает данные из PostgreSQL и отдаёт их. Граф рендерит Cosmograph (WebGL, тысячи нод без лагов) или D3.js force-directed (если Cosmograph недоступен через CDN). Ноды — документы/wiki/факты, рёбра — cosine similarity > threshold.

## Файлы

**Создать:**
- `visualization/` — папка
- `visualization/api.py` — Flask micro-API для данных графа
- `visualization/graph.html` — self-contained HTML визуализация
- `visualization/graph_builder.py` — логика построения рёбер из pgvector cosine similarity

**Трогать:**
- `agent_server/server.py` — добавить MCP tool `kb_open_graph`

**Не трогать:**
- `schema.sql` — читаем из существующих таблиц documents, user_facts
- `retriever/search.py` — не затронут
- `core/` — не затронут

## Реализация

### 1. visualization/graph_builder.py — построение графа из PostgreSQL

```python
"""
Graph Builder — читает documents из PostgreSQL и строит граф на основе
cosine similarity между embedding векторами.

Ноды: документы, wiki-файлы, user_facts
Рёбра: cosine_similarity > EDGE_THRESHOLD (по умолчанию 0.75)
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

EDGE_THRESHOLD = 0.75   # минимальное сходство для создания ребра
MAX_NODES = 500         # ограничение для производительности браузера
MAX_EDGES_PER_NODE = 5  # максимум рёбер на ноду (top-5 ближайших)


def build_graph_data(
    source_filter: Optional[str] = None,
    limit: int = MAX_NODES,
) -> dict:
    """
    Построить данные графа для визуализации.
    
    Returns:
        dict с 'nodes': list, 'edges': list, 'stats': dict
        
        nodes: [{id, label, source_type, source_url, chunk_index, chunk_total, color}]
        edges: [{source, target, weight}]
    """
    from core.db import get_conn, get_cursor
    
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # Получить уникальные документы (один представитель на source_id = chunk_index=0)
            base_query = """
                SELECT DISTINCT ON (source_id, source_type)
                    id::text,
                    source_type,
                    source_id,
                    source_url,
                    title,
                    chunk_index,
                    chunk_total,
                    embedding::text
                FROM documents
                WHERE chunk_index = 0
                  AND embedding IS NOT NULL
            """
            
            params = []
            if source_filter:
                base_query += " AND source_type = %s"
                params.append(source_filter)
            
            base_query += f" LIMIT {limit}"
            cur.execute(base_query, params)
            docs = cur.fetchall()
            
            if not docs:
                return {"nodes": [], "edges": [], "stats": {"node_count": 0, "edge_count": 0}}
            
            # Построить ноды
            COLOR_MAP = {
                "notion": "#5B8AF5",
                "url": "#F5A623",
                "file": "#7ED321",
                "wiki": "#9013FE",
                "manual": "#D0021B",
            }
            
            nodes = []
            doc_ids = []
            for doc in docs:
                color = COLOR_MAP.get(doc["source_type"], "#AAAAAA")
                nodes.append({
                    "id": doc["id"],
                    "label": (doc["title"] or "Untitled")[:50],
                    "source_type": doc["source_type"],
                    "source_url": doc["source_url"],
                    "color": color,
                    "size": min(10 + doc["chunk_total"] * 2, 30),  # размер ∝ количеству чанков
                })
                doc_ids.append(doc["id"])
            
            # Добавить user_facts как отдельные ноды
            cur.execute("SELECT key, value, category FROM user_facts LIMIT 50")
            facts = cur.fetchall()
            for fact in facts:
                nodes.append({
                    "id": f"fact_{fact['key']}",
                    "label": f"📌 {fact['key']}: {fact['value'][:30]}",
                    "source_type": "fact",
                    "source_url": None,
                    "color": "#FF6B6B",
                    "size": 12,
                })
            
            # Рёбра: cosine similarity через KNN LATERAL join (O(n*k), не O(n²))
            # ВАЖНО: CROSS JOIN на 300-500 нодах = 90k-250k пар при каждом открытии графа.
            # Правильно: LATERAL + ORDER BY embedding для каждого нода — использует HNSW индекс.
            edges = []
            if len(doc_ids) > 1:
                cur.execute("""
                    WITH base AS (
                        SELECT id, embedding
                        FROM documents
                        WHERE id::text = ANY(%s) AND chunk_index = 0
                    )
                    SELECT DISTINCT
                        LEAST(b.id::text, nn.id::text) AS source,
                        GREATEST(b.id::text, nn.id::text) AS target,
                        1 - (b.embedding <=> nn.embedding) AS similarity
                    FROM base b
                    JOIN LATERAL (
                        SELECT id, embedding
                        FROM documents d_inner
                        WHERE chunk_index = 0
                          AND d_inner.id != b.id
                        ORDER BY b.embedding <=> d_inner.embedding
                        LIMIT %s
                    ) nn ON true
                    WHERE 1 - (b.embedding <=> nn.embedding) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (
                    doc_ids,
                    MAX_EDGES_PER_NODE,
                    EDGE_THRESHOLD,
                    len(doc_ids) * MAX_EDGES_PER_NODE,
                ))
                
                for row in cur.fetchall():
                    edges.append({
                        "source": row["source"],
                        "target": row["target"],
                        "weight": round(row["similarity"], 3),
                    })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "sources": list(set(n["source_type"] for n in nodes)),
        }
    }
```

### 2. visualization/api.py — Flask micro-API

```python
"""
Flask API для данных графа. Запускается на порту 7331.
Используется visualization/graph.html для загрузки данных.
"""
import json
import logging
from flask import Flask, jsonify, request, send_from_directory
from pathlib import Path

from visualization.graph_builder import build_graph_data

logger = logging.getLogger(__name__)
app = Flask(__name__, static_folder=str(Path(__file__).parent))


@app.route("/")
def index():
    """Отдать graph.html."""
    return send_from_directory(str(Path(__file__).parent), "graph.html")


@app.route("/api/graph")
def get_graph():
    """
    GET /api/graph?source=notion&limit=200
    Вернуть данные графа в формате {nodes, edges, stats}.
    """
    source_filter = request.args.get("source")  # optional
    limit = min(int(request.args.get("limit", 300)), 500)
    
    try:
        data = build_graph_data(source_filter=source_filter, limit=limit)
        return jsonify(data)
    except Exception as e:
        logger.error(f"graph API error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def get_stats():
    """GET /api/stats — краткая статистика KB."""
    from core.db import get_conn, get_cursor
    try:
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                cur.execute("""
                    SELECT source_type, COUNT(DISTINCT source_id) as doc_count, COUNT(*) as chunk_count
                    FROM documents
                    GROUP BY source_type
                """)
                sources = cur.fetchall()
                
                cur.execute("SELECT COUNT(*) as cnt FROM user_facts")
                facts_count = cur.fetchone()["cnt"]
        
        return jsonify({
            "sources": [dict(s) for s in sources],
            "facts_count": facts_count,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_viz_server(port: int = 7331):
    """Запустить Flask сервер. Вызывается из kb_open_graph."""
    import threading
    import webbrowser
    
    def _run():
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    
    import time
    time.sleep(1.0)  # дать серверу стартовать
    webbrowser.open(f"http://127.0.0.1:{port}/")
    
    return f"http://127.0.0.1:{port}/"
```

### 3. visualization/graph.html — интерактивный граф

Self-contained HTML с D3.js force-directed graph (CDN):

```html
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Knowledge Graph</title>
<style>
  body { margin: 0; background: #1a1a2e; font-family: 'Segoe UI', sans-serif; color: #eee; }
  #controls { position: fixed; top: 12px; left: 12px; z-index: 10; display: flex; gap: 8px; flex-wrap: wrap; }
  .btn { background: #16213e; border: 1px solid #0f3460; color: #eee; padding: 6px 14px;
         border-radius: 6px; cursor: pointer; font-size: 13px; }
  .btn:hover { background: #0f3460; }
  .btn.active { background: #e94560; border-color: #e94560; }
  #stats { position: fixed; top: 12px; right: 12px; background: #16213e; 
           padding: 12px; border-radius: 8px; font-size: 13px; min-width: 180px; }
  #tooltip { position: fixed; background: rgba(22,33,62,0.95); border: 1px solid #0f3460;
             padding: 10px; border-radius: 6px; font-size: 12px; pointer-events: none;
             display: none; max-width: 280px; z-index: 20; }
  #search { background: #16213e; border: 1px solid #0f3460; color: #eee; 
            padding: 6px 12px; border-radius: 6px; font-size: 13px; width: 200px; }
  svg { width: 100vw; height: 100vh; }
  .node { cursor: pointer; }
  .node circle { stroke-width: 1.5px; stroke: rgba(255,255,255,0.3); transition: r 0.2s; }
  .node:hover circle { stroke: #e94560; stroke-width: 2.5px; }
  .node.highlighted circle { stroke: #FFD700; stroke-width: 3px; }
  .link { stroke: rgba(255,255,255,0.12); stroke-width: 1px; }
  .link.strong { stroke: rgba(255,255,255,0.35); stroke-width: 2px; }
  #legend { position: fixed; bottom: 12px; left: 12px; background: #16213e;
            padding: 10px; border-radius: 8px; font-size: 12px; }
  .legend-item { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
  #loading { position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
             font-size: 20px; color: #e94560; }
</style>
</head>
<body>

<div id="loading">🔄 Загрузка графа...</div>

<div id="controls" style="display:none">
  <input id="search" type="text" placeholder="Поиск по заголовку...">
  <button class="btn active" onclick="filterSource('all')">Все</button>
  <button class="btn" onclick="filterSource('notion')">Notion</button>
  <button class="btn" onclick="filterSource('wiki')">Wiki</button>
  <button class="btn" onclick="filterSource('url')">URL</button>
  <button class="btn" onclick="filterSource('file')">Файлы</button>
  <button class="btn" onclick="filterSource('fact')">Факты</button>
  <button class="btn" onclick="resetZoom()">Reset</button>
</div>

<div id="stats"></div>
<div id="tooltip"></div>
<div id="legend" style="display:none"></div>
<svg id="graph"></svg>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
let allNodes = [], allEdges = [], simulation, svg, g;
let currentFilter = 'all';

const COLOR_MAP = {
  notion: '#5B8AF5', url: '#F5A623', file: '#7ED321',
  wiki: '#9013FE', manual: '#D0021B', fact: '#FF6B6B'
};

async function loadGraph() {
  try {
    const [graphRes, statsRes] = await Promise.all([
      fetch('/api/graph?limit=300'),
      fetch('/api/stats')
    ]);
    const graphData = await graphRes.json();
    const statsData = await statsRes.json();
    
    allNodes = graphData.nodes || [];
    allEdges = graphData.edges || [];
    
    document.getElementById('loading').style.display = 'none';
    document.getElementById('controls').style.display = 'flex';
    document.getElementById('legend').style.display = 'block';
    
    // Stats panel
    let statsHtml = '<b>📊 KB Stats</b><br>';
    statsHtml += `Нод: ${allNodes.length} | Рёбер: ${allEdges.length}<br><br>`;
    if (statsData.sources) {
      statsData.sources.forEach(s => {
        statsHtml += `${s.source_type}: ${s.doc_count} doc / ${s.chunk_count} chunks<br>`;
      });
    }
    statsHtml += `<br>Фактов: ${statsData.facts_count || 0}`;
    document.getElementById('stats').innerHTML = statsHtml;
    
    // Legend
    let legendHtml = '<b>Легенда:</b>';
    Object.entries(COLOR_MAP).forEach(([type, color]) => {
      legendHtml += `<div class="legend-item"><div class="legend-dot" style="background:${color}"></div>${type}</div>`;
    });
    document.getElementById('legend').innerHTML = legendHtml;
    
    renderGraph(allNodes, allEdges);
  } catch(e) {
    document.getElementById('loading').textContent = '❌ Ошибка загрузки: ' + e.message;
  }
}

function renderGraph(nodes, edges) {
  d3.select('#graph').selectAll('*').remove();
  
  const width = window.innerWidth, height = window.innerHeight;
  svg = d3.select('#graph');
  
  const zoom = d3.zoom()
    .scaleExtent([0.1, 8])
    .on('zoom', (e) => g.attr('transform', e.transform));
  svg.call(zoom);
  
  g = svg.append('g');
  window._zoom = zoom;
  window._svg = svg;
  
  // Edges
  const link = g.append('g').selectAll('line')
    .data(edges).join('line')
    .attr('class', e => e.weight > 0.85 ? 'link strong' : 'link');
  
  // Nodes
  const node = g.append('g').selectAll('.node')
    .data(nodes).join('g')
    .attr('class', 'node')
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );
  
  node.append('circle')
    .attr('r', d => d.size || 8)
    .attr('fill', d => COLOR_MAP[d.source_type] || '#888');
  
  node.append('title').text(d => d.label);
  
  // Tooltip
  const tooltip = document.getElementById('tooltip');
  node.on('mouseover', (e, d) => {
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 12) + 'px';
    tooltip.style.top = (e.clientY - 10) + 'px';
    tooltip.innerHTML = `<b>${d.label}</b><br>Тип: ${d.source_type}` +
      (d.source_url ? `<br><a href="${d.source_url}" target="_blank" style="color:#5B8AF5">Открыть →</a>` : '');
  }).on('mouseout', () => { tooltip.style.display = 'none'; });
  
  simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(edges).id(d => d.id).distance(80).strength(e => e.weight * 0.5))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(width/2, height/2))
    .force('collision', d3.forceCollide().radius(d => (d.size || 8) + 4))
    .on('tick', () => {
      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
}

function filterSource(type) {
  currentFilter = type;
  document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  
  const filteredNodes = type === 'all' ? allNodes : allNodes.filter(n => n.source_type === type);
  const nodeIds = new Set(filteredNodes.map(n => n.id));
  const filteredEdges = allEdges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));
  
  renderGraph(filteredNodes, filteredEdges);
}

function resetZoom() {
  window._svg.transition().duration(500).call(window._zoom.transform, d3.zoomIdentity);
}

document.getElementById('search').addEventListener('input', function(e) {
  const query = e.target.value.toLowerCase();
  if (!query) {
    d3.selectAll('.node').classed('highlighted', false);
    return;
  }
  d3.selectAll('.node').classed('highlighted', d => d.label.toLowerCase().includes(query));
});

window.addEventListener('resize', () => {
  if (allNodes.length > 0) renderGraph(allNodes, allEdges);
});

loadGraph();
</script>
</body>
</html>
```

### 4. agent_server/server.py — добавить `kb_open_graph`

```python
@mcp.tool(
    annotations={"readOnlyHint": True, "openWorldHint": False},
)
def kb_open_graph(
    port: int = 7331,
) -> str:
    """
    Открыть интерактивный граф базы знаний в браузере.
    
    Запускает локальный веб-сервер и открывает visualization/graph.html
    с D3.js force-directed графом: ноды = документы/wiki/факты,
    рёбра = cosine similarity > 0.75.
    
    Args:
        port: Порт для веб-сервера (по умолчанию 7331)
    
    Returns:
        URL открытого интерфейса
    """
    try:
        from visualization.api import run_viz_server
        url = run_viz_server(port=port)
        return f"🕸️ Knowledge Graph открыт: {url}\n\nФункции:\n  🔍 Поиск по заголовку документа\n  🎨 Фильтр по типу источника (Notion / Wiki / URL / Файлы / Факты)\n  🖱️ Drag & zoom, hover для просмотра деталей\n  🔗 Клик по ноде → открыть источник"
    except Exception as e:
        logger.error(f"kb_open_graph error: {e}", exc_info=True)
        return f"❌ Ошибка запуска граф-интерфейса: {e}"
```

### 5. requirements.txt — добавить зависимости

```
flask>=3.0.0
```

(D3.js грузится с CDN, flask — единственная новая зависимость)

## Стандарты
- **Karpathy**: Simple — HTML + Flask + D3, без Electron, без Webpack. Grapher as a side-daemon, не блокирует. Surgical — только visualization/ модуль и один tool в server.py.
- **Dev**: `dev-standards:python-api` — Flask в daemon-потоке, не блокирует MCP сервер. Graceful если порт занят.
- **Проект**: синхронный код, cosine distance через pgvector оператор `<=>` — не тащим всё в Python.

## Что НЕ делать
- **НЕ использовать Cosmograph** — требует регистрации, D3 достаточно для 300-500 нод
- **НЕ блокировать MCP сервер** при запуске Flask — только threading.Thread(daemon=True)
- **НЕ рендерить все чанки** — только chunk_index=0 (один представитель документа)
- **НЕ считать рёбра в Python** — использовать pgvector `<=>` в SQL для скорости
- **НЕ делать аутентификацию** — localhost only (127.0.0.1), не биндить 0.0.0.0
- **НЕ падать если Flask не установлен** — возвращать "❌ pip install flask" инструкцию

## Критерий готовности
- [ ] `kb_open_graph()` — открывает браузер с графом, возвращает URL
- [ ] В графе видны ноды всех типов: notion/url/wiki/file/fact с разными цветами
- [ ] Рёбра видны между семантически похожими документами
- [ ] Поиск по заголовку — подсвечивает matching ноды
- [ ] Фильтр "Wiki" — показывает только wiki-ноды
- [ ] Hover на ноде — показывает заголовок и ссылку на источник
- [ ] При повторном вызове `kb_open_graph()` — не падает, не открывает второй сервер
- [ ] MCP сервер продолжает работать пока открыт граф
- [ ] Karpathy review: код thoughtful, simple, surgical, goal-driven
