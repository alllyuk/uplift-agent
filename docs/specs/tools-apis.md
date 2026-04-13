# Спецификация: Tools / APIs

## 1. Обзор

4 инструмента, вызываемых из LangGraph nodes. Все работают in-process (Python), кроме OpenAI API (HTTPS).

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

### 2.4 Token truncation

Модуль truncation: tiktoken `gpt2`, лимит 2000 tokens/сообщение, обрезка с конца.

### 2.5 Ошибки

| Ошибка | Обработка |
|--------|-----------|
| API timeout (120с) | 1 retry, затем abort |
| Rate limit (429) | LangChain встроенный retry с backoff |
| Invalid API key | Проверка при старте |
| JSON parse error | Fallback на text mode |

### 2.6 Cost

~$0.002/кейс: суммарно около `~10K` input tokens и `~2K` output tokens на кейс (при 1-2 LLM-вызовах, GPT-4o-mini rates).

### 2.7 Время выполнения

- Ожидаемая latency одного LLM-вызова: обычно `~10-30с`, в зависимости от модели и размера контекста.
- Защитный timeout: `120с` на вызов, чтобы не обрывать длинные ответы слишком рано.

### 2.8 Injection protection

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
    caliper=0.1,
)
result: PSMResult = analyzer.run(df)
# → PSMResult(ate=0.023, att=0.018, n_pairs=412)
```

### 3.2 Параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| caliper | 0.1 | Макс. разница propensity score при матчинге |
| replacement | False | 1:1 greedy matching без замены |
| logreg_max_iter | 1000 | Итерации логистической регрессии |
| auto-detect covariates | Да | Все числовые столбцы кроме target/treatment |
| auto-threshold | 75-й перцентиль | Для не-бинарного treatment |

### 3.3 Ошибки

| Ошибка | Обработка |
|--------|-----------|
| Отсутствие столбцов | `{ok: False, error: str}` |
| Мало matched pairs (`n_pairs < 50`) | `{ok: False, ate, att, n_pairs}` — числа возвращаются, но `ok=False` сигнализирует ненадёжность |

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

## 5. RAG Query Tool

### 5.1 Контракт

```python
rag = RAG(cfg)  # загружает chunks, embeddings, FAISS index
chunks: list[str] = rag.perform_query(query="Предложение эквайринга", top_k=3)
```

### 5.2 Зависимости

Pre-built артефакты: `chunks.parquet`, `embeddings.parquet`, `index.faiss` (в директории `rag_data/`).

### 5.3 Ошибки

Отсутствие индекса или embeddings → `rag_chunks = []` в LangGraph node. Latency: обычно < 5с.
