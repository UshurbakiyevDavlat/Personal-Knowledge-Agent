# 🧠 Personal Knowledge Agent

Персональная RAG система — личная база знаний которую Claude использует как память. Индексирует твои Notion страницы, статьи и заметки, позволяя Claude находить нужный контекст без передачи всего содержимого в каждом запросе.

**Экономия токенов: 94% · В 16× дешевле · Работает в любом разговоре с Claude**

## Как это работает

```
Индексация (один раз):
Notion / URL / файл → Chunking → Voyage AI Embeddings → pgvector

Поиск (каждый запрос):
Твой вопрос → Embedding → Cosine similarity → Топ-5 чанков → Claude
```

Claude получает доступ к базе знаний через MCP сервер — инструменты `kb_search`, `kb_add_document` и другие доступны в любом разговоре автоматически.

## Стек

- **Python 3.11+** — основной язык
- **PostgreSQL 17 + pgvector** — хранение и поиск векторов (HNSW индекс)
- **Voyage AI** (`voyage-3`) — эмбеддинги, рекомендованы Anthropic для Claude
- **FastMCP** — MCP сервер для интеграции с Claude Code
- **Notion API** — индексация твоих страниц
- **Docker** — изолированная БД, не зависит от локального PostgreSQL

---

## Быстрый старт

### Требования

- Python 3.11+
- Docker Desktop
- Аккаунты: [Voyage AI](https://www.voyageai.com) · [Notion Integrations](https://www.notion.so/my-integrations)

### 1. Клонируй репозиторий

```bash
git clone https://github.com/твой-username/knowledge-agent.git
cd knowledge-agent
```

### 2. Запусти PostgreSQL с pgvector через Docker

```bash
docker run -d \
  --name knowledge-agent-db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=knowledge_agent \
  -p 5434:5432 \
  --restart unless-stopped \
  pgvector/pgvector:pg17
```

> Порт `5434` — чтобы не конфликтовать с локальным PostgreSQL если он есть.

Проверь что контейнер запустился:
```bash
docker ps | grep knowledge-agent-db
```

### 3. Накати схему БД

```bash
docker cp schema.sql knowledge-agent-db:/schema.sql
docker exec -it knowledge-agent-db psql -U postgres -d knowledge_agent -f /schema.sql
```

Проверь таблицы:
```bash
docker exec -it knowledge-agent-db psql -U postgres -d knowledge_agent -c "\dt"
```

Должно появиться: `documents`, `user_facts`, `index_log`.

### 4. Настрой окружение

```bash
# Виртуальное окружение
python -m venv venv

# Активация
source venv/bin/activate        # Linux / macOS
.\venv\Scripts\Activate.ps1    # Windows PowerShell

# Зависимости
pip install -r requirements.txt
```

Создай `.env` из шаблона:
```bash
cp .env.example .env
```

Открой `.env` и заполни:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5434/knowledge_agent
VOYAGE_API_KEY=pa-...        # https://www.voyageai.com
NOTION_API_KEY=secret_...    # https://www.notion.so/my-integrations
NOTION_ROOT_PAGE_IDS=...     # ID корневой страницы Notion (из URL)
```

**Как найти Notion Page ID:** открой страницу в браузере, скопируй UUID из URL:
```
https://notion.so/workspace/Моя-страница-32e262630981...
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          это и есть Page ID
```

### 5. Первая индексация Notion

```bash
# Заполнить базовые факты о пользователе
python -m memory.episodic

# Запустить индексацию (первый раз ~5-10 мин в зависимости от объёма)
python -m indexer.notion_indexer

# Проверить поиск
python -m retriever.search "как работает auth"
```

### 6. Подключи к Claude Code

MCP сервер добавляется через CLI:

```bash
# macOS / Linux
claude mcp add knowledge-agent \
  /полный/путь/до/venv/bin/python \
  /полный/путь/до/knowledge-agent/run_mcp_server.py

# Windows (PowerShell)
claude mcp add knowledge-agent `
  "C:\путь\до\knowledge-agent\venv\Scripts\python.exe" `
  "C:\путь\до\knowledge-agent\run_mcp_server.py"
```

**Windows пример (реальный путь):**
```powershell
claude mcp add knowledge-agent `
  "C:\Users\dushu\bestForLearning\knowledge-agent\venv\Scripts\python.exe" `
  "C:\Users\dushu\bestForLearning\knowledge-agent\run_mcp_server.py"
```

Проверь что подключился:
```bash
claude mcp list
# knowledge-agent: ... ✔ connected
```

> **Важно:** `run_mcp_server.py` — враппер-скрипт который автоматически устанавливает рабочую директорию и путь к модулям. Не используй `-m agent_server.server` напрямую — `claude mcp add` не поддерживает флаги начинающиеся с `-`.

### 7. Cowork (облачный интерфейс)

> ⚠️ **Текущий статус:** В разработке. Планируется деплой на VPS.

Cowork работает в облаке и не имеет доступа к локальному stdio серверу. Для работы в Cowork нужен HTTP/SSE транспорт с публичным URL:

```python
# run_mcp_server.py (HTTP режим для Cowork)
mcp.run(transport="sse", host="0.0.0.0", port=8000)
```

**Варианты деплоя:**
- **ngrok** — быстро и временно: `ngrok http 8000`, добавить URL в Claude.ai → Settings → Integrations
- **VPS** — постоянно: задеплоить на сервер с HTTPS, добавить как постоянный коннектор

---

## MCP Инструменты

После подключения Claude может использовать:

| Инструмент | Описание |
|---|---|
| `kb_search` | Семантический поиск по базе знаний |
| `kb_add_document` | Добавить текст вручную |
| `kb_add_url` | Проиндексировать веб-страницу |
| `kb_index_notion` | Переиндексировать Notion (всё или конкретные страницы) |
| `kb_list_sources` | Список проиндексированных источников |
| `kb_delete` | Удалить документ из базы |
| `kb_get_facts` | Получить эпизодическую память (факты о пользователе) |
| `kb_update_fact` | Обновить факт в эпизодической памяти |

---

## Структура проекта

```
knowledge-agent/
├── schema.sql              # SQL схема БД (запускается один раз)
├── requirements.txt
├── .env.example            # Шаблон настроек
├── config.py               # Конфигурация из .env
├── run_mcp_server.py       # Враппер для запуска MCP (используется claude mcp add)
├── core/
│   ├── db.py               # Connection pool PostgreSQL
│   ├── chunker.py          # Recursive text splitting (500 токенов)
│   └── embedder.py         # Voyage AI embeddings
├── indexer/
│   └── notion_indexer.py   # Индексация Notion (инкрементальная, параллельная)
├── retriever/
│   └── search.py           # Гибридный поиск (vector + full-text + RRF)
├── memory/
│   └── episodic.py         # Факты о пользователе
├── agent_server/
│   └── server.py           # MCP сервер (FastMCP, stdio транспорт)
└── scheduler/
    └── reindex.py          # Ночная переиндексация (APScheduler, 03:00 Almaty)
```

> **Примечание:** папка называется `agent_server/` (не `mcp/`) — чтобы избежать конфликта имён с установленным пакетом `mcp`.

---

## Ночная переиндексация (опционально)

Автоматически обновляет базу каждую ночь в 03:00 — только изменённые страницы:

```bash
# Запустить в фоне
nohup python -m scheduler.reindex > /tmp/ka-scheduler.log 2>&1 &

# Или через cron (Linux/macOS):
# 0 3 * * * cd /путь/до/knowledge-agent && venv/bin/python -m indexer.notion_indexer
```

---

## Стоимость

| | Без RAG | С RAG |
|---|---|---|
| Токенов на вопрос | ~50 000 | ~3 000 |
| 20 вопросов в день | ~$3.00 | ~$0.18 |
| В месяц | ~$90 | ~$5.40 |

Индексация 500 Notion страниц через Voyage AI: **~$0.01** (первые 200M токенов бесплатно).

---

## Docker команды — шпаргалка

```bash
# Запустить контейнер
docker start knowledge-agent-db

# Остановить
docker stop knowledge-agent-db

# Подключиться к psql
docker exec -it knowledge-agent-db psql -U postgres -d knowledge_agent

# Посмотреть логи
docker logs knowledge-agent-db

# Удалить контейнер (данные сохранятся в volume если настроен)
docker rm -f knowledge-agent-db
```

---

## .gitignore

Не забудь добавить в `.gitignore`:
```
.env
venv/
__pycache__/
*.pyc
```
