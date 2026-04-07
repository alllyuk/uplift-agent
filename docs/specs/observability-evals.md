# Спецификация: Observability / Evals

## 1. Метрики

### 1.1 Продуктовые и агентские

| Метрика | Target (PoC) | Как измерять |
|---------|-------------|--------------|
| Время на рекомендацию (p95) | SLO ≤ 5 мин, target ≤ 3 мин | `latency_ms` из SQLite |
| % полезных рекомендаций | > 70% | Ручная экспертная оценка |
| % объяснимых ответов | > 80% | Critic: наличие ссылок на граф/RAG |
| % отправки на ручной анализ | < 20% | `SELECT count(*) FROM cases WHERE status != 'done'` |
| Корректность выбора tool | > 85% | Eval на тестовом наборе |
| Корректность отказа | > 80% | Eval: кейсы с заведомо плохими данными |
| % галлюцинаций | < 10% | Critic + RAGAS faithfulness |
| Консистентность текст/числа | > 95% | Critic check #1 |
| Полнота рекомендаций | > 95% | Critic check #5 |

### 1.2 Технические

| Метрика | Target (PoC) | Как измерять |
|---------|-------------|--------------|
| p95 latency | SLO ≤ 5 мин, target ≤ 3 мин | Расчёт в коде: сортировка `latency_ms`, 95-й перцентиль |
| Cost per case | < $1 | LangSmith: token usage × pricing |
| % успешных tool calls | > 90% | Loguru: подсчёт ошибок по module |
| Error rate | < 5% | `SELECT count(*) FROM cases WHERE status = 'aborted'` |

## 2. Логирование (Loguru)

### 2.1 Конфигурация

```python
logger.add(
    sink="artifacts/pipeline.log",
    rotation="10 MB",
    retention="90 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
)
```

### 2.2 Пример

```
2026-04-06 14:30:01 | INFO | orchestrator:intake:45 | case_id=abc123 client_id=C000005
2026-04-06 14:30:02 | INFO | orchestrator:policy_check:78 | case_id=abc123 blocked=false
2026-04-06 14:30:05 | INFO | psm_tool:compute:32 | case_id=abc123 ate=0.023 n_pairs=412
2026-04-06 14:30:07 | INFO | rag_tool:query:18 | case_id=abc123 top_k=3 chunks_found=3
2026-04-06 14:30:15 | INFO | synthesizer:generate:56 | case_id=abc123 tokens_in=4850 tokens_out=720
2026-04-06 14:30:15 | INFO | critic:check:92 | case_id=abc123 passed=true issues=[]
2026-04-06 14:30:15 | INFO | persister:save:12 | case_id=abc123 status=done latency_ms=14200
```

### 2.3 Что логируется

| Событие | Уровень | Данные |
|---------|---------|--------|
| Запуск кейса | INFO | case_id, client_id |
| Policy decision | INFO | case_id, blocked, reasons count |
| PSM result | INFO | case_id, ok, ate, att, n_pairs |
| RAG query | INFO | case_id, query (truncated), chunks_found |
| LLM call | INFO | case_id, tokens_in, tokens_out, latency_ms |
| Critic verdict | INFO | case_id, passed, issues |
| Case saved | INFO | case_id, status, total_latency_ms |
| Ошибки | ERROR | case_id, module, error message, traceback |

**Не логируется** (governance.md §2): сырые PII, полные тексты промптов/RAG-документов, API-ключи.

## 3. Трейсинг (LangSmith)

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=uplift-agent-poc
```

Один trace = один кейс. LangGraph автоматически создаёт spans для каждого node. `trace_id` сохраняется в `cases.trace_id` для навигации SQLite → LangSmith. Free tier: 5000 traces/мес. При превышении — трейсинг отключается, Loguru продолжает работать.

## 4. Evaluation Framework

### 4.1 RAGAS

| Метрика | Описание | Порог |
|---------|----------|-------|
| Faithfulness | Соответствие ответа контексту (RAG) | > 0.7 |
| Answer Relevancy | Релевантность ответа вопросу | > 0.7 |

Инфраструктура: `AgentLLMAdapter` (async-совместимость), `LocalE5Embeddings` (e5-small), вручную подготовленные тестовые пары.

### 4.2 Pollux (LLM-as-judge)

| Критерий | Описание | Шкала |
|----------|----------|-------|
| Depth of Analysis | Интеграция данных из графа, RAG, PSM | 0/1/2 |
| Applicability | Практическая полезность рекомендаций | 0/1/2 |
| Consistency | Внутренняя непротиворечивость | 0/1/2 |
| Evidence Usage | Использование предоставленных данных | 0/1/2 |
| Risk Awareness | Упоминание рисков и ограничений | 0/1/2 |

### 4.3 Critic Checks (rule-based)

Critic выполняется после каждой генерации LLM. Все проверки детерминированные, без LLM.

**Check 1 — Числовая консистентность:** если в тексте ответа явно упоминаются ATT/ATE (паттерн `ATT =`, `ATE ≈` и т.п.), значения сравниваются с `psm_result` (tolerance 10%). Несовпадение → issue.

**Check 2 — Атрибуция источников:** если цитируется `doc_id` (паттерн `doc_\d+`), проверяется его наличие в retrieved chunks. Цитирование отсутствующего документа → issue.

**Check 3 — Валидность рёбер графа:** если упоминается связь "A → B" (паттерн `\w+ -> \w+`), проверяется наличие ребра в `graph_dsl`. Несуществующее ребро → issue.

**Check 4 — Хеджирование уверенности:** если PSM-данные слабые (`psm_result` отсутствует, `|ATT| < 0.001`, или `n_pairs < 50`), текст проверяется на категоричные маркеры ("однозначно", "гарантированно", "точно приведёт", "100%"). Категоричность при слабых данных → issue.

**Check 5 — Полнота ответа:** обязательные поля Explanation (`drivers_pos`, `drivers_neg`, `expected_effect`) заполнены и содержат >= 10 символов.

### 4.4 Стратегия при fail Critic

1. Собрать все `issues` из 5 проверок
2. Если пусто → `{passed: True, issues: []}`
3. Если не пусто и `retry_count == 0` → инъекция issues в промпт, повторный synthesize
4. Если не пусто и `retry_count >= 1` → `{passed: False, issues: [...]}`, статус `degraded`

## 5. A/B тестирование промптов и моделей

### 5.1 Что сравнивается

- **Версии промптов** — `prompt_version_base`, `prompt_version_whatif` (см. `agent-orchestrator.md` §6.2)
- **LLM-модели** — через `LLM_MODEL_NAME` (например `gpt-4o-mini` vs `gpt-4o`)
- **Параметры** — `temperature`, `top_k` для RAG

### 5.2 Дизайн эксперимента

- Конфигурация в `experiments.yaml`: `experiment_id`, `variant_a` (control), `variant_b` (treatment), `traffic_split`, `start_at`, `end_at`, `success_metric`
- **Routing:** детерминированный split по `case_id` — `variant = "A" if hash(case_id) % 100 < split else "B"`. Без липкости по client (PoC обрабатывает кейсы независимо)
- В `CaseState` пишутся `experiment_id` и `variant`; в SQLite — столбец `experiment_variant` (см. `memory-context.md` §3.1)

### 5.3 Метрики сравнения

Переиспользуются метрики из §1: % полезных рекомендаций, % галлюцинаций, RAGAS faithfulness, p95 latency, cost per case, critic pass rate.

### 5.4 Анализ

- SQL-сравнение `cases` по `experiment_variant` для фиксированного `experiment_id`
- Минимальный размер выборки для PoC: `n ≥ 100` на вариант, простое сравнение средних
- Решающий критерий: вариант B принимается, если `lift ≥ 5%` по основной метрике
- Production: z-test для пропорций / t-test для непрерывных, power analysis — out of scope для PoC

### 5.5 Жизненный цикл эксперимента

`draft → running → analyzed → promoted | rolled-back`. Promotion = смена default-версии в `LLMConfig` + рестарт. Rollback = возврат предыдущей версии без удаления артефактов.

### 5.6 PoC scope

Фреймворк описан, реализация ограничена ручным запуском двух конфигураций сервиса и сравнением метрик из SQLite. Автоматический experiment runner — out of scope.

---

## 6. Health Checks

| Компонент | Проверка при старте | При недоступности в runtime |
|-----------|--------------------|-----------------------------|
| LLM API | Ping (простой вызов) | Abort с `llm_unavailable` |
| FAISS Index | Файл существует | RAG в degraded mode (`rag_chunks = []`) |
| Embeddings | Файл существует + загрузка | RAG в degraded mode (`rag_chunks = []`) |
| SQLite | CREATE TABLE / SELECT 1 | Cooldown пропускается (fail-open, `requires_human_review = True`). Persist невозможен — только Loguru. |
