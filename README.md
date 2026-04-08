# 🧠 Personal Knowledge Agent

Персональная RAG система — личная база знаний которую Claude использует как память. Индексирует Notion страницы, статьи и заметки, позволяя Claude находить нужный контекст без передачи всего содержимого в каждом запросе.

**Экономия токенов: 94% · В 16× дешевле · Работает в Claude Code и Cowork**

## Как это работает

```
Индексация (инкрементально, каждые 2 часа):
Notion / URL / файл → Chunking → Voyage AI Embeddings → pgvector

Поиск (каждый запрос):
Твой вопрос → Embedding → Hybrid Search (vector + BM25 + RRF) → Топ-5 чанков → Claude
```

Claude получает доступ к базе знаний через MCP сервер — инструменты `kb_search`, `kb_add_document` и другие доступны в любом разговоре автоматически.

## Стек

- **Python 3.12** — основной язык
- **MCP SDK 1.27.0** (официальный от Anthropic) — MCP сервер
- **PostgreSQL 17 + pgvector** — хранение и поиск векторов (HNSW индекс)
- **Voyage AI** (`voyage-3`) — эмбеддинги, рекомендованы Anthropic для Claude
- **Notion API** — индексация страниц
- **Docker / Docker Compose** — изолированная БД
- **nginx + Let's Encrypt** — HTTPS для Cowork интеграции (VPS)
- **systemd** — автозапуск сервисов на VPS

---

## Быстрый старт (локально, Claude Code)

### Требования

- Python 3.12+
- Docker Desktop
- Аккаунты: [Voyage AI](https://www.voyageai.com) · [Notion Integrations](https://www.notion.so/my-integrations)

### 1. Клонируй репозиторий

```bash
git clone https://github.com/UshurbakiyevDavlat/Personal-Knowledge-Agent.git
cd Personal-Knowledge-Agent
```

### 2. Запусти PostgreSQL + pgvector через Docker Compose

```bash
docker compose up -d
```

`docker-compose.yml` поднимает `pgvector/pgvector:pg17` на порту `5432` (localhost only).

Проверь:
```bash
docker compose ps
```

### 3. Накати схему БД

```bash
docker exec -i pgvector psql -U agent -d knowledge < schema.sql
```

Должно вывести `CREATE TABLE`, `CREATE INDEX` — без ошибок.

### 4. Настрой окружение

```bash
python3 -m venv venv

# Активация
source venv/bin/activate        # Linux / macOS
.\venv\Scripts\Activate.ps1    # Windows PowerShell

pip install -r requirements.txt
```

Создай `.env`:

```env
DATABASE_URL=postgresql://agent:agentpass@localhost:5432/knowledge
VOYAGE_API_KEY=pa-...
NOTION_API_KEY=secret_...
NOTION_ROOT_PAGE_IDS=...
```

> ⚠️ Не добавляй inline-комментарии к числовым значениям — `int()` не умеет их парсить.

**Как найти Notion Page ID:** открой страницу в браузере, скопируй UUID из URL:
```
https://notion.so/workspace/Моя-страница-32e262630981...
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          это и есть Page ID
```

### 5. Первая индексация и поиск

```bash
# Запустить индексацию Notion
python -m indexer.notion_indexer

# Проверить поиск
python -m retriever.search "как работает auth"
```

### 6. Подключи к Claude Code (stdio)

```powershell
# Windows
claude mcp add knowledge-agent `
  "C:\путь\до\knowledge-agent\venv\Scripts\python.exe" `
  "C:\путь\до\knowledge-agent\run_mcp_server.py"

# Linux / macOS
claude mcp add knowledge-agent \
  /путь/до/knowledge-agent/venv/bin/python \
  /путь/до/knowledge-agent/run_mcp_server.py
```

Проверь:
```bash
claude mcp list
# knowledge-agent: ... ✔ connected
```

---

## Cowork / Claude.ai (VPS деплой)

После деплоя на VPS подключи свой MCP endpoint в Cowork через "Customize connectors".
Endpoint формируется автоматически на основе твоего домена: `https://<твой-домен>/sse`

> ⚠️ Не публикуй свой endpoint публично — добавь токен-защиту через nginx (см. раздел безопасности ниже).

Подробная инструкция по деплою ниже.

---

## MCP Инструменты

| Инструмент | Описание |
|---|---|
| `kb_search` | Гибридный поиск по базе знаний (vector + BM25 + RRF) |
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
├── schema.sql              # SQL схема БД
├── requirements.txt
├── config.py               # Конфигурация из .env
├── run_mcp_server.py       # Враппер запуска MCP (SSE транспорт)
├── core/
│   ├── db.py               # Connection pool PostgreSQL
│   ├── chunker.py          # Recursive text splitting (500 токенов)
│   └── embedder.py         # Voyage AI embeddings
├── indexer/
│   └── notion_indexer.py   # Индексация Notion (инкрементальная, параллельная)
├── retriever/
│   └── search.py           # Гибридный поиск (vector + full-text + RRF)
├── memory/
│   └── episodic.py         # Факты о пользователе (эпизодическая память)
├── agent_server/           # Не mcp/ — конфликт имён с пакетом mcp
│   └── server.py           # FastMCP SSE, host=0.0.0.0, port=8000
└── scheduler/
    └── reindex.py          # Переиндексация каждые 2 часа (APScheduler)
```

---

## VPS Деплой (Hetzner + HTTPS)

### Инфраструктура

- **Провайдер:** Hetzner Cloud, CAX11 (2 vCPU ARM64, 4GB RAM) — ~$5.49/мес
- **OS:** Ubuntu 24.04
- **Домен:** DuckDNS (бесплатно) или любой другой
- **HTTPS:** Let's Encrypt + Certbot (автообновление)

### Установка на сервере

```bash
# Docker
curl -fsSL https://get.docker.com | sh

# Python зависимости
apt install -y python3-venv python3-dev python3-pip

# nginx + Certbot
apt install -y nginx certbot python3-certbot-nginx

# Firewall
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw enable
```

### PostgreSQL (Docker Compose)

```bash
mkdir -p /opt/knowledge-agent && cd /opt/knowledge-agent
# положить docker-compose.yml (см. файл в репо)
docker compose up -d
```

### Деплой кода

```bash
cd /opt/knowledge-agent
git clone git@github.com:UshurbakiyevDavlat/Personal-Knowledge-Agent.git app
cd app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Создать .env с DATABASE_URL, VOYAGE_API_KEY, NOTION_API_KEY, NOTION_ROOT_PAGE_IDS
# Накатить схему
docker exec -i pgvector psql -U agent -d knowledge < schema.sql
```

### systemd — MCP сервер

Файл `/etc/systemd/system/knowledge-agent.service`:

```ini
[Unit]
Description=Knowledge Agent MCP Server
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/knowledge-agent/app
EnvironmentFile=/opt/knowledge-agent/app/.env
ExecStart=/opt/knowledge-agent/app/venv/bin/python run_mcp_server.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### systemd — Scheduler (переиндексация каждые 2 часа)

Файл `/etc/systemd/system/knowledge-agent-scheduler.service`:

```ini
[Unit]
Description=Knowledge Agent Notion Scheduler
After=network.target knowledge-agent.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/knowledge-agent/app
EnvironmentFile=/opt/knowledge-agent/app/.env
ExecStart=/opt/knowledge-agent/app/venv/bin/python scheduler/reindex.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable knowledge-agent knowledge-agent-scheduler
systemctl start knowledge-agent knowledge-agent-scheduler
```

### nginx конфиг

Файл `/etc/nginx/sites-available/knowledge-agent`:

```nginx
server {
    listen 80;
    server_name <твой-домен>;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/knowledge-agent /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx
certbot --nginx -d <твой-домен> --non-interactive --agree-tos -m your@email.com
```

### Деплой обновлений

```bash
# Локально
git push

# На сервере
cd /opt/knowledge-agent/app && git pull
systemctl restart knowledge-agent knowledge-agent-scheduler
```

### Управление сервисами

```bash
systemctl status knowledge-agent knowledge-agent-scheduler
journalctl -u knowledge-agent -f
journalctl -u knowledge-agent-scheduler -f
```

---

## Безопасность

MCP сервер содержит личные данные — закрой его токеном через nginx.

Замени единый `location /` на два блока:

```nginx
# /sse — требует токен
location /sse {
    if ($arg_token != "ВАШ_СЕКРЕТНЫЙ_ТОКЕН") {
        return 401;
    }
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 86400;
}

# /messages/ — пропускаем (session_id = неявная авторизация)
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 86400;
}
```

Сгенерировать токен:
```bash
openssl rand -hex 32
```

Подключение в Cowork — указывай URL с токеном:
```
https://<твой-домен>/sse?token=ВАШ_СЕКРЕТНЫЙ_ТОКЕН
```

---

## Стоимость

| | Без RAG | С RAG |
|---|---|---|
| Токенов на вопрос | ~50 000 | ~3 000 |
| 20 вопросов в день | ~$3.00 | ~$0.18 |
| В месяц | ~$90 | ~$5.40 |

- Индексация 500 Notion страниц: **~$0.01** (Voyage AI, первые 200M токенов бесплатно)
- VPS Hetzner CAX11: **~$5.49/мес**

**Итого: ~$10.89/мес с RAG + VPS** vs ~$90/мес без RAG — **экономия 88%**

---

## .gitignore

```
.env
venv/
__pycache__/
*.pyc
```
