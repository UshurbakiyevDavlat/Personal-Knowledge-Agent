"""
Flask micro-API для данных графа. Запускается на порту 7331.
Используется visualization/graph.html для загрузки данных.
"""
import logging
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from visualization.graph_builder import build_graph_data

logger = logging.getLogger(__name__)
app = Flask(__name__, static_folder=str(Path(__file__).parent))

_server_started = False


@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).parent), "graph.html")


@app.route("/api/graph")
def get_graph():
    """GET /api/graph?source=notion&limit=200"""
    source_filter = request.args.get("source") or None
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
                cur.execute(
                    """
                    SELECT source_type,
                           COUNT(DISTINCT source_id) AS doc_count,
                           COUNT(*) AS chunk_count
                    FROM documents
                    GROUP BY source_type
                    """
                )
                sources = [dict(r) for r in cur.fetchall()]

                cur.execute("SELECT COUNT(*) AS cnt FROM user_facts")
                facts_count = cur.fetchone()["cnt"]

        return jsonify({"sources": sources, "facts_count": facts_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_viz_server(port: int = 7331) -> str:
    """Запустить Flask сервер в daemon-потоке. Идемпотентно."""
    global _server_started
    import threading
    import time
    import webbrowser

    url = f"http://127.0.0.1:{port}/"

    if not _server_started:
        def _run():
            import logging as _log
            _log.getLogger("werkzeug").setLevel(logging.WARNING)
            app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(1.0)
        _server_started = True

    webbrowser.open(url)
    return url
