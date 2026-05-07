═══════════════════════════════════════════
ТЗ: Reranker — Voyage rerank-2 в kb_search
Проект: Personal-Knowledge-Agent · Python 3.12 + voyageai SDK
Дата: 2026-05-07
═══════════════════════════════════════════

## Цель
Добавить cross-encoder reranking (Voyage rerank-2) после RRF-слияния в kb_search — повысить precision@5 на 15-25% без переиндексации.

## Контекст
Текущий пайплайн: embed query → vector search (top-10) + FTS (top-10) → RRF fusion → top-5. RRF хорош для объединения, но не переоценивает релевантность — это делает reranker. Voyage AI SDK, который уже используется для эмбеддингов (voyageai), поддерживает `client.rerank()` нативно. Стоимость: rerank-2 дешевле чем ещё один round-trip к vector DB.

## Файлы

**Трогать:**
- `retriever/search.py` — добавить функцию `_rerank()` и вызов в `search()`
- `config.py` — добавить `RERANK_MODEL` и `RERANK_ENABLED`
- `agent_server/server.py` — расширить параметр `kb_search` флагом `rerank`

**Не трогать:**
- `core/embedder.py` — не нужен для reranking
- `core/db.py` — не нужен
- `indexer/` — не нужен

## Реализация

### 1. config.py — добавить конфигурацию reranker

После строки `DEFAULT_TOP_K`:
```python
# Reranking
RERANK_ENABLED: bool = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "rerank-2")
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))   # финальный top_k после rerank
RERANK_CANDIDATES: int = int(os.getenv("RERANK_CANDIDATES", "20"))  # сколько отдать на rerank
```

### 2. retriever/search.py — добавить `_rerank()`

После функции `_reciprocal_rank_fusion` добавить:

```python
def _rerank(
    query: str,
    candidates: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """
    Voyage AI rerank-2 — cross-encoder переранжирование.
    Принимает N кандидатов, возвращает top_k в новом порядке.
    """
    if not candidates:
        return candidates

    try:
        import voyageai
        client = voyageai.Client(api_key=config.VOYAGE_API_KEY)

        documents = [r.content for r in candidates]
        result = client.rerank(
            query=query,
            documents=documents,
            model=config.RERANK_MODEL,
            top_k=top_k,
        )

        # result.results — список объектов с .index и .relevance_score
        reranked = []
        for item in result.results:
            candidate = candidates[item.index]
            candidate.score = round(item.relevance_score, 4)
            reranked.append(candidate)

        logger.info(f"Reranked {len(candidates)} → {len(reranked)} results")
        return reranked

    except Exception as e:
        logger.warning(f"Reranker failed, using RRF results: {e}")
        return candidates[:top_k]  # fallback — RRF результаты
```

### 3. retriever/search.py — интегрировать в `search()`

**ВАЖНО — сначала изменить сигнатуру `search()`** в retriever/search.py:

```python
def search(
    query: str,
    top_k: int = 5,
    source_filter: str | None = None,
    hybrid: bool = True,
    date_from: str | None = None,
    rerank: bool | None = None,   # <-- новый параметр
) -> list[SearchResult]:
    # use_rerank: явный параметр приоритетнее конфига
    use_rerank = rerank if rerank is not None else config.RERANK_ENABLED
    ...
```

Затем изменить финальную часть `search()` (после `ranked = _reciprocal_rank_fusion(...)`):

```python
    # Берём RERANK_CANDIDATES кандидатов (больше чем top_k) для reranker
    candidates_count = config.RERANK_CANDIDATES if use_rerank else top_k
    results = []
    for doc, score in ranked[:candidates_count]:
        meta = doc.get("metadata") or {}
        doc_date = meta.get("doc_date")
        if not doc_date and doc.get("indexed_at"):
            doc_date = doc["indexed_at"].date().isoformat()
        results.append(SearchResult(
            id=doc["id"],
            title=doc["title"] or "Untitled",
            content=doc["content"],
            source_type=doc["source_type"],
            source_url=doc.get("source_url"),
            chunk_index=doc["chunk_index"],
            chunk_total=doc["chunk_total"],
            score=round(score, 4),
            metadata=meta,
            doc_date=doc_date,
        ))

    # Reranking через use_rerank (не глобальный конфиг)
    if use_rerank and len(results) > 1:
        results = _rerank(query=query, candidates=results, top_k=top_k)
    else:
        results = results[:top_k]

    logger.info(f"Found {len(results)} results (rerank={'on' if use_rerank else 'off'})")
    return results
```

### 4. agent_server/server.py — добавить параметр `rerank` в kb_search

Изменить сигнатуру `kb_search`:
```python
def kb_search(
    query: str,
    top_k: int = 5,
    source_filter: str | None = None,
    date_from: str | None = None,
    rerank: bool = True,   # <-- новый параметр
) -> str:
```

Передать напрямую в search() — НЕ мутировать глобальный config:
```python
# ПРАВИЛЬНО: передаём rerank как параметр в search()
results = search(
    query=query,
    top_k=top_k,
    source_filter=source_filter,
    hybrid=True,
    date_from=date_from,
    rerank=rerank,   # thread-safe, нет глобальных мутаций
)
return format_results_for_claude(results)
```

**Edge case**: если Voyage API недоступен или reranker вернул ошибку — fallback к RRF результатам уже реализован в `_rerank()` через try/except.

**Edge case**: если результатов меньше 2 — reranker не вызывается (нет смысла).

## .env — добавить переменные (не менять код, только документация)

В `.env.example` добавить:
```
RERANK_ENABLED=true
RERANK_MODEL=rerank-2
RERANK_CANDIDATES=20
RERANK_TOP_K=5
```

## Стандарты
- **Karpathy**: Simple — используем уже установленный voyageai SDK, не добавляем зависимостей. Surgical — меняем только retriever/search.py и config.py. Thoughtful — fallback к RRF при ошибке reranker.
- **Dev**: `dev-standards:ai-llm` — reranker как отдельная функция с чётким интерфейсом, graceful degradation.
- **Проект**: синхронный код, логи через logger, config через `config.RERANK_*`.

## Что НЕ делать
- **НЕ менять HNSW индекс** или схему БД — reranker работает над уже найденными результатами
- **НЕ делать reranker обязательным** — если Voyage API лежит, поиск должен работать
- **НЕ увеличивать** `RERANK_CANDIDATES` выше 50 — это ограничение Voyage rerank API на один вызов
- **НЕ вызывать reranker** для запросов с одним результатом или пустым query

## Критерий готовности
- [ ] `kb_search("как работает auth в AdashAI")` — возвращает результаты (rerank по умолчанию включён)
- [ ] `kb_search("тест", rerank=False)` — работает без reranker (RRF fallback)
- [ ] При `RERANK_ENABLED=false` в .env — reranker не вызывается
- [ ] При недоступном Voyage API — search не падает, возвращает RRF результаты
- [ ] В логах виден: `Reranked 20 → 5 results`
- [ ] Karpathy review: код thoughtful, simple, surgical, goal-driven
