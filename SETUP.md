# Knowledge Agent — Инструкция по запуску

## 1. База данных

```bash
# Подключись к PostgreSQL и создай базу
psql -U postgres
```

```sql
CREATE DATABASE knowledge_agent;
\c knowledge_agent
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
\q
```

```bash
# Накати схему
psql -U postgres -d knowledge_agent -f schema.sql
```

---

## 2. Окружение

```bash
# Клонируй / скопируй проект в ~/projects/knowledge-agent
cd ~/projects/knowledge-agent

# Виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Зависимости
pip install -r requirements.txt

# Настройки
cp .env.example .env
# Открой .env и заполни:
#   DATABASE_URL=postgresql://postgres:пароль@localhost:5432/knowledge_agent
#   OPENAI_API_KEY=sk-...
#   NOTION_API_KEY=secret_...
#   NOTION_ROOT_PAGE_IDS=<ID страницы Обучение>
```

---

## 3. Первый запуск — индексация Notion

```bash
# Из папки knowledge-agent с активным venv:

# Заполнить базовые факты о пользователе
python -m memory.episodic

# Запустить индексацию Notion (первый раз может занять 5-10 мин)
python -m indexer.notion_indexer

# Проверить поиск
python -m retriever.search "как работает RAG"
```

---

## 4. Подключение к Claude Code / Cowork

Найди файл конфигурации Claude Code:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/claude/claude_desktop_config.json`

Добавь в секцию `mcpServers`:

```json
{
  "mcpServers": {
    "knowledge-agent": {
      "command": "/Users/ИМЯ/projects/knowledge-agent/venv/bin/python",
      "args": ["-m", "mcp.server"],
      "cwd": "/Users/ИМЯ/projects/knowledge-agent",
      "env": {
        "PYTHONPATH": "/Users/ИМЯ/projects/knowledge-agent"
      }
    }
  }
}
```

**Перезапусти Claude Code / Cowork.** Инструменты `kb_search`, `kb_add_document` и др. появятся автоматически.

---

## 5. Ночная переиндексация (опционально)

```bash
# Запустить планировщик в фоне
nohup python -m scheduler.reindex > /tmp/ka-scheduler.log 2>&1 &
echo $! > /tmp/ka-scheduler.pid

# Остановить
kill $(cat /tmp/ka-scheduler.pid)
```

Или добавь в cron:
```bash
crontab -e
# Добавь строку (каждую ночь в 03:00):
0 3 * * * cd /Users/ИМЯ/projects/knowledge-agent && venv/bin/python -m indexer.notion_indexer >> /tmp/ka-reindex.log 2>&1
```

---

## 6. Структура проекта

```
knowledge-agent/
├── schema.sql              ← SQL схема (запускается один раз)
├── requirements.txt
├── .env.example            ← Скопируй в .env и заполни
├── config.py               ← Конфигурация из .env
├── core/
│   ├── db.py               ← Пул соединений PostgreSQL
│   ├── chunker.py          ← Разбивка текста на чанки (500 токенов)
│   └── embedder.py         ← OpenAI text-embedding-3-small
├── indexer/
│   └── notion_indexer.py   ← Индексация Notion (инкрементальная)
├── retriever/
│   └── search.py           ← Гибридный поиск (vector + full-text)
├── memory/
│   └── episodic.py         ← Факты о пользователе (PostgreSQL)
├── mcp/
│   └── server.py           ← MCP сервер (stdio) для Claude
└── scheduler/
    └── reindex.py          ← Ночная переиндексация
```
