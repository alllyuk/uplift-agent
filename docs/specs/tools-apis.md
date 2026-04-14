# Спецификация: Tools / APIs

## 1. Обзор

4 инструмента, вызываемых из шагов Pipeline (`sme_causal/orchestrator/pipeline.py`). Все работают in-process (Python), кроме OpenAI API (HTTPS).

## 2. OpenAI LLM API

### 2.1 Контракт

```python
llm = ChatOpenAI(
    model=cfg.llm.model_name,
    temperature=cfg.llm.temperature,
    api_key=cfg.effective_openai_api_key,
    seed=cfg.data_generation.seed,
    model_kwargs={"response_format": {"type": "json_object"}}
)
response: AIMessage = llm.invoke(messages)
```

### 2.2 Параметры

| Параметр | Источник | Default |
|----------|----------|---------|
| model | `LLM_MODEL` env | Из конфигурации |
| temperature | `LLM_TEMPERATURE` env | Из конфигурации |
| seed | `DataGenerationConfig.seed` | 42 |
| response_format | Программно | `{"type": "json_object"}` |
| timeout | Конфигурация клиента | 120с на один LLM-вызов |

### 2.3 Fallback-стратегия

1. Вызов с `response_format={"type": "json_object"}`
2. При ошибке JSON mode → вызов без response_format (text mode)
3. Regex-извлечение `{...}` блока из текстового ответа

### 2.4 Ошибки

| Ошибка | Обработка |
|--------|-----------|
| LLM-исключение в synthesize | Ловится в `Pipeline.run`, кейс abort с `abort_reason = "pipeline_error: …"` |
| LLM-исключение в critic L2 / rag_refine | Fail-open: L2 возвращает пустой список issues, rag_refine делает skip |
| Rate limit (429) | LangChain встроенный retry с backoff |
| Invalid API key | Проверка наличия `OPENAI_API_KEY` при старте entry-point |
| JSON parse error | Fallback на text mode |

### 2.5 Cost

~$0.002/кейс: суммарно около `~10K` input tokens и `~2K` output tokens на кейс (при 1-2 LLM-вызовах, GPT-4o-mini rates).

### 2.6 Время выполнения

- Ожидаемая latency одного LLM-вызова: обычно `~10-30с`, в зависимости от модели и размера контекста.
- Явный per-call timeout в v1 не задаётся — используется дефолт langchain-openai.

### 2.7 Injection protection

- System prompt отделён от user content
- RAG-документы в user message, не в system
- Инструкция: "Используйте ТОЛЬКО предложенную информацию"

## 3. PSM Tool (Causal Estimation)

### 3.1 Контракт

```python
analyzer = CausalInferenceAnalyzer(
    target="Revenue_Growth_Rate",
    treatment_variable="Credit_Limit_Change",  # определяется из delta
    covariates=None,                            # auto-detect
    caliper=0.05,
)
result: PSMResult = analyzer.run(df)
# → PSMResult(ate=0.023, att=0.018, n_pairs=412)
```

### 3.2 Параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| caliper | 0.05 | Макс. разница propensity score при матчинге (default в `Pipeline`, `inference/psm_runner.py`) |
| replacement | False | 1:1 greedy matching без замены |
| logreg_max_iter | 1000 | Итерации логистической регрессии |
| auto-detect covariates | Да | Все числовые столбцы кроме target/treatment |
| auto-threshold | 75-й перцентиль | Для не-бинарного treatment |

### 3.3 Ошибки и надёжность

| Ситуация | Возврат | Значение `psm_reliable` / `psm_reason` |
|----------|---------|----------------------------------------|
| Отсутствие outcome- или treatment-столбца | `{ok: False, error: "..."}` | — (расчёт не выполнен) |
| `ATT` невалидный (`NaN`/`inf`) | `{ok: True, ..., psm_reliable: False}` | `"ATT is unavailable; naive ATE must not be used as the primary personal effect."` |
| `n_treated` или `n_control` недоступны в `matched_df` | `{ok: True, ..., psm_reliable: False}` | `"Matched sample sizes are unavailable."` |
| `n_treated < 100` или `n_control < 100` | `{ok: True, ..., psm_reliable: False}` | `"Matched sample is too small: n_treated=..., n_control=..., required>=100."` |
| Выборки достаточны | `{ok: True, ..., psm_reliable: True}` | `"Matched sample is large enough: n_treated=..., n_control=..."` |

Порог `PSM_MIN_GROUP_SIZE = 100` и функция `_psm_reliability` — в `inference/psm_runner.py`. Во всех случаях с `ok=True` числа (ATE/ATT/n_pairs) возвращаются и попадают в synthesize-промпт, но `psm_reliable=False` служит сигналом LLM хеджировать и отдавать приоритет RAG/Graph.

Latency: CPU-bound, обычно < 10с на 3000 клиентов.

## 4. Graph DSL Loader

### 4.1 Контракт

```python
graph_dsl: str = load_graph_dsl(method="llm", min_conf=0.45)
# → "Industry -> Avg_Monthly_Inflow | sign:+ | conf:0.85 | note:\"...\"\n..."
```

### 4.2 Выбор файла

| graph_method | Артефакт |
|--------------|----------|
| `llm` | `llm_edges.json` |
| `hybrid` | `hybrid_edges.json` |
| `algo` | `graph_consensus.json` |
| `algo_llm` | `algo_llm_edges.json` |

Рёбра с `confidence < min_conf` отбрасываются. In-memory кэш на время жизни процесса.

### 4.3 Ошибки

Файл не найден или невалидный JSON → `graph_dsl = ""` (пустая строка). Latency: обычно < 1с.

### 4.4 Правила использования графа в synthesize-промпте (`GRAPH_RESPONSE_RULES`)

Блок `[GRAPH_DSL]` передаётся в synthesize-промпт вместе с жёсткими инструкциями по его использованию (константа `GRAPH_RESPONSE_RULES` в `sme_causal/agent/agent_service.py`). Цель — не дать LLM придумать новые рёбра и не протечь во внешний ответ техническим маркерам DSL (`sign`, `conf`, стрелкам `->`).

Ключевые правила:
- Использовать **только** рёбра из блока `[GRAPH_DSL]`; новые рёбра не выдумывать.
- Имена узлов цитировать **точно** как в DSL (например `Revenue_Growth_Rate`, не «выручка» / «Profit»). Аналитик сопоставляет ответ с графом — перефразирование недопустимо.
- Поля `sign` и `conf` использовать только внутренне (sign задаёт направление, conf — доверие). В пользовательском тексте их не показывать; шаблоны вида `(sign:+, conf=0.7)` запрещены.
- Пользовательская формулировка: `«По причинному графу: A положительно/отрицательно влияет на B; доверие высокое/среднее/низкое: …»`. Маппинг `conf` → метка: `≥0.75` — высокое, `≥0.5` — среднее, иначе низкое.
- Если нет ребра или короткой цепочки от интервенции к целевой метрике — писать «В графе нет подтверждённого пути от A к B». Запрещено начинать с «На основании ребра A → B» при отсутствии такого ребра.
- Формулировка «связь между A и B» запрещена (скрывает направление каузальности).
- Если граф **не передан**, на рёбра не ссылаться и формулу `A -> B` не употреблять.

**Post-processing:** `CausalAgent._clean_public_graph_terms` дополнительно пост-обрабатывает ответ LLM регулярками — переводит остаточные шаблоны `На основании ребра A -> B (sign:+, conf=0.85):` в user-friendly формулировку и вырезает технические хвосты `(conf=0.7)` / `(sign:+, conf=0.85)`, если LLM всё-таки их прислала. Это защита от регрессий, а не замена prompt rules.

## 5. RAG Query Tool

### 5.1 Контракт

```python
rag = RAG(cfg)  # загружает chunks, embeddings, FAISS index
chunks: list[str] = rag.perform_query(query="Предложение эквайринга", top_k=3)
```

### 5.2 Зависимости

Pre-built артефакты: `chunks.parquet`, `embeddings.parquet`, `index.faiss` (в директории `rag_data/`).

### 5.3 Ошибки

Отсутствие индекса или embeddings → `rag_chunks = []` в шаге `estimation`. Latency: обычно < 5с.
