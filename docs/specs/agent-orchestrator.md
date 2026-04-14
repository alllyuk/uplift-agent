# Спецификация: Agent / Orchestrator

## 1. Обзор

Pipeline Orchestrator — центральный компонент v1, реализованный как plain-Python класс `Pipeline` в `sme_causal/orchestrator/pipeline.py`. Управляет выполнением кейса через последовательность методов, работающих на shared `CaseState`: приём запроса → координация модулей (Policy, PSM, RAG, LLM) → проверка через Critic → сохранение в SQLite.

Готовые фреймворки агентных графов (LangGraph и т.п.) в v1 намеренно не используются — см. ADR-1 в `system-design.md`.

## 2. CaseState

```python
class CaseState(TypedDict):
    # Идентификация
    case_id: str                                    # UUID
    client_id: str                                  # "C000005"
    raw_query: Optional[str]                        # NL-запрос (если есть)

    # Контекст
    client_context: dict                            # 25 полей CONTEXT_FIELDS
    intervention_delta: dict                        # {"New_Product_Offer": 1, ...}

    # Policy
    policy_result: dict                             # {blocked, reasons, notes}
    cooldown_previous_case: Optional[dict]          # {case_id, created_at, explanation, raw_query} — если cooldown сработал

    # Estimation
    psm_result: Optional[dict]                      # {ok, ate, att, n_pairs} или None
    rag_chunks: list[str]                           # top-k чанков (накопительный список после rag_refine)
    rag_iterations: int                             # сколько раз вызывался RAG (1 — initial, ≤ 2 после rag_refine)
    rag_query_history: list[str]                    # все формулировки RAG-запросов по итерациям
    graph_dsl: str                                  # DSL-строка рёбер

    # Synthesis
    explanation: dict                               # {diagnosis, drivers_pos, drivers_neg, ...}

    # Critic
    critic_result: dict                             # {passed, rule_issues, llm_issues, issues} — см. §3.6
    retry_count: int                                # 0 или 1

    # Human review
    requires_human_review: bool
    review_reason: Optional[str]

    # Метаданные
    status: str                                     # intake|context|policy|estimation|synthesis|critic|rag_refine|done|aborted|degraded
    trace_id: Optional[str]
    latency_ms: Optional[int]
    abort_reason: Optional[str]

    # Версии промптов и A/B
    prompt_versions: dict                           # {"base": "v1.0", "whatif": "v2.1"} — фактически использованные шаблоны
    experiment_id: Optional[str]                    # ID A/B-эксперимента, если кейс попал под него
    variant: Optional[str]                          # "A" | "B" внутри эксперимента
```

## 3. Узлы (nodes)

### 3.0 Публичный API Pipeline

```python
Pipeline(
    df,
    *,
    case_store=None,
    graph_method="llm",        # llm | algo | hybrid | algo_llm
    use_rag=True,
    use_graph=True,
    use_psm=True,
    outcome_col="Revenue_Growth_Rate",
    covariates=None,
    caliper=0.05,
    min_conf=0.45,
    model=None,
    temperature=None,
).run(
    client_id,
    intervention_delta,
    *,
    raw_query=None,
    target_metric=None,        # переопределяет outcome_col для конкретного кейса
    match_info=None,
) -> CaseState
```

- **`target_metric`** в `run()` позволяет переопределить outcome-колонку для PSM и synthesize-промпта в рамках одного кейса (используется QueryParser для NL-запросов с явным указанием метрики). Если `target_metric is None`, используется `outcome_col`, заданный в конструкторе (`Revenue_Growth_Rate` по умолчанию).
- **Флаги `use_psm` / `use_rag` / `use_graph`** выключают соответствующую под-задачу `estimation`. Если все три выключены одновременно, guard `sources_requested` в `Pipeline.run` пропускает проверку `no_evidence` — кейс продолжается только с профилем клиента и delta (применимо для целей отладки / smoke-тестов без боковых зависимостей).
- **`match_info`** — структура, возвращаемая `QueryParser` (см. §7), прокидывается в synthesize-промпт, чтобы LLM могла хеджировать формулировку при неточном совпадении интервенции по NL-запросу.

### 3.1 intake

NL-парсинг выполняется **до** `Pipeline.run` — в entry-points (`app/run.py`, `app/streamlit_app.py`) через `QueryParser` (few-shot LLM, §7) и `parse_client_id_and_intent` (regex-извлечение `client_id`). На вход `Pipeline.run` поступают уже структурированные `client_id` и `intervention_delta`. Ошибки парсинга NL-запроса (невалидный текст, неразрешимая интервенция) обрабатываются в entry-point и до pipeline не доходят.

Внутри `Pipeline._intake`:
- Генерируется `case_id` (UUID4), `retry_count = 0` (через `create_case_state`)
- Проверяется наличие `client_id` в загруженном DataFrame — при отсутствии abort с `abort_reason = "client_not_found"`

### 3.2 load_context

Загрузка строки из CSV по `client_id` через `CausalAgent.build_context_for_client` → dict из 25 полей профиля (`CONTEXT_FIELDS`). Не найден → abort с `abort_reason = "client_not_found"`.

### 3.3 policy_check

Rule-based проверки допустимости:

1. Предложение продукта, который у клиента уже есть
2. Кредитный лимит ниже текущего
3. Скидка выше текущей
4. Комбинация: продукт + изменение лимита одновременно
5. Нулевая или отрицательная скидка
6. **Cooldown:** проверка по SQLite — если для `client_id` есть завершённый кейс (`status = 'done'`) с тем же типом интервенции за последние 30 дней → блокировка. При недоступности SQLite: cooldown пропускается, `requires_human_review = True`, `review_reason = "cooldown не проверен: SQLite недоступен"`

Результат: `{blocked: bool, reasons: [str], notes: dict}`

**Cooldown UX:** при срабатывании cooldown `_policy_check` прикрепляет к `CaseState.cooldown_previous_case` поля предыдущего завершённого кейса — `{case_id, created_at, explanation, raw_query}` (JSON-поле `result_json` парсится вспомогательным `_safe_json`). Entry-points (`app/run.py`, `app/streamlit_app.py`) отрисовывают этот объект как подсказку «эта интервенция уже оценивалась …, показан предыдущий результат» вместо пустого ответа на abort.

**Переходы:** `blocked` → abort | иначе → estimation

### 3.4 estimation (параллельный)

Три sub-task параллельно через `ThreadPoolExecutor` (по одному worker на источник):

| Sub-task | Input | Output | При ошибке |
|----------|-------|--------|------------|
| **PSM** | DataFrame, treatment из delta | `{ok, ate, att, n_pairs}` | `psm_result = None` |
| **RAG** | query из delta или raw_query, top_k=3 | `list[str]` чанков | `rag_chunks = []` |
| **Graph DSL** | graph_method, min_conf | DSL-строка рёбер | `graph_dsl = ""` |

**Precondition перед synthesize:** если все 3 источника недоступны (`psm_result is None` и `rag_chunks == []` и `graph_dsl == ""`), кейс прерывается — abort с `abort_reason = "no_evidence"`. Синтез без единого источника данных не имеет смысла.

### 3.5 synthesize

1. Выбор шаблона промпта для оценки заданной интервенции
2. Заполнение: контекст клиента, delta, PSM-summary, RAG-чанки, Graph DSL
3. При retry: добавление "ИСПРАВЬТЕ СЛЕДУЮЩИЕ ПРОБЛЕМЫ: {issues}"
4. LLM вызов: JSON mode → text fallback → regex-парсинг
5. Парсинг в Explanation: `{diagnosis, drivers_pos, drivers_neg, expected_effect, recommendations, raw_text}`

Ошибка: любое LLM-исключение ловится в `Pipeline.run` внешним `try/except` и приводит к abort с `abort_reason = "pipeline_error: …"`.

### 3.6 critic_check

Двухуровневая проверка: rule-based safety floor + LLM-augmented смысловые проверки.

**Уровень 1 — rule-based (структурные проверки, подробнее в `observability-evals.md` §4.3):**
1. Атрибуция источников (doc_id в rag_chunks)
2. Полнота обязательных полей (drivers_pos, drivers_neg, expected_effect)

**Уровень 2 — LLM-augmented (см. `observability-evals.md` §4.4):** запускается **только** если уровень 1 не нашёл блокирующих проблем. LLM проверяет смысловую консистентность объяснения с числами/документами, адекватность хеджирования и полноту recommendations.

**Результат:**
```python
{
  "passed": bool,                # True если оба уровня прошли
  "rule_issues": list[str],      # hard issues (блокирующие)
  "llm_issues": list[str],       # soft issues (смысловые)
  "issues": list[str],           # объединённый список для prompt retry
}
```

**Переходы:**
- pass → persist
- fail + `retry_count == 0` → **rag_refine** → synthesize (`retry_count := 1`)
- fail + `retry_count >= 1` → persist_with_warning

### 3.7 rag_refine

Адаптивная переформулировка RAG-запроса на основе critic issues. Запускается между `critic_check` (fail) и retry-`synthesize`. Это первый узел графа, где **LLM реально управляет вызовом инструмента** (см. ADR-8 в `system-design.md`).

**Шаги:**
1. На вход: `critic_result.issues`, `rag_query_history`, текущий `explanation`, `intervention_delta`
2. LLM-prompt: «По issues ниже сформулируй уточнённый русскоязычный RAG-запрос, который дозапросит недостающий контекст. Не повторяй формулировки из истории.» → новый `query_v2`
3. Вызов RAG с `query_v2`, top_k=3
4. **Append**, не replace: новые чанки добавляются к `rag_chunks` (с дедупликацией по chunk_id)
5. `rag_iterations += 1`, `rag_query_history.append(query_v2)`

**Stop conditions:**
- Максимум `rag_iterations = 2` (1 initial в estimation + 1 refine). Больше итераций не делаем — это PoC, бесконечные циклы запрещены.
- При недоступности RAG: rag_refine skip, переход сразу в synthesize (degraded).
- При недоступности LLM для формулировки query: rag_refine skip, переход в synthesize.

### 3.8 persist

1. Вычисление `latency_ms`
2. Определение `status`: done / aborted / degraded
3. Определение `requires_human_review`:
   - Critic fail после retry (v1)
   - SQLite был недоступен при cooldown-проверке (v1)
   - `status == "degraded"` из-за tool fails — **planned v2**
   - `|ATT| < 0.001` и мало matched pairs — **planned v2**
4. INSERT в SQLite (если доступен), логирование через Loguru

## 4. Stop conditions

- Максимум retry: 1. Бесконечных циклов нет.
- Граф завершается в persist (done/degraded/aborted)

## 5. Retry / Fallback

| Сценарий | Стратегия |
|----------|-----------|
| LLM: невалидный JSON | JSON mode → text mode fallback |
| Critic fail (rule или LLM) + `retry_count == 0` | **rag_refine → synthesize** (max 1 retry) |
| Critic fail + `retry_count >= 1` | persist_with_warning, `requires_human_review = True` |
| rag_refine: LLM не смог сформулировать query | Skip rag_refine → synthesize напрямую |
| rag_refine: RAG недоступен | Skip rag_refine → synthesize напрямую |
| `rag_iterations >= 2` | rag_refine больше не запускается |
| PSM / RAG / Graph fail (частично) | Skip, degraded mode |
| Все 3 источника недоступны | Abort с `no_evidence` |
| PSM: `n_treated < 100` или `n_control < 100` | `ok=True` (расчёт прошёл), но `psm_reliable=False` с пояснением в `psm_reason`. Порог — `PSM_MIN_GROUP_SIZE = 100` в `inference/psm_runner.py`. |
| LLM-исключение в synthesize | Ловится внешним `try/except` в `Pipeline.run` → abort с `abort_reason = "pipeline_error: …"` |
| SQLite недоступен | Cooldown skip (fail-open), persist skip, Loguru only |

## 6. Промпт-менеджмент

### 6.1 Шаблоны

- **prompt_base**: базовый анализ. System: роль + инструкции + JSON-схема. User: профиль + признаки + граф.
- **prompt_whatif**: оценка заданной интервенции. Дополнительно: delta, PSM-summary, RAG-чанки, match_info.
- Шаблоны: `ChatPromptTemplate.from_messages()` с переменными `{context}`, `{features_description}`, `{graph_block}`, `{what_if_block}`, `{psm_block}`, `{rag_context}`.

### 6.2 Версионирование

- **Storage:** метаданные версий лежат в `prompts/{name}/{version}.yaml` (например `prompts/whatif/v1.0.yaml`). YAML содержит `version`, `created_at`, `parent_version`, `notes`, список `variables` и указатель `source: {module, attribute}` на inline-шаблон в коде (`CausalAgent._prompt_base` / `_prompt_whatif`). Полный перенос текста промпта в YAML с runtime-загрузкой — задача v2.
- **Naming:** семантическое `vMAJOR.MINOR`. MAJOR — несовместимое изменение схемы переменных, MINOR — текстовая правка.
- **Активация:** активная версия выбирается через `LLMConfig` (`specs/serving-config.md` §2.2) — поля `prompt_version_base`, `prompt_version_whatif`. По умолчанию — `v1.0`.
- **Runtime-проверка:** `sme_causal/agent/prompt_registry.py::ensure_versions()` вызывается в `CausalAgent.__init__`. Если активная версия из config отсутствует в `prompts/<name>/`, поднимается `PromptVersionError` — опечатки в `LLM_PROMPT_VERSION_*` ловятся на старте, а не на первом synthesize.
- **Публикация активных версий:** `CausalAgent.active_prompt_versions()` возвращает актуальный dict `{base, whatif}`; `Pipeline._synthesize` копирует его в `CaseState.prompt_versions` до первого LLM-вызова — результат уходит в SQLite (`cases.prompt_versions_json`).
- **Rollback:** смена версии в config + рестарт сервиса. Старые версии не удаляются — остаются в репозитории и могут быть включены обратно.
- **PoC scope:** 2 шаблона (`base`, `whatif`), смена редкая, без автоматической runtime-загрузки текста из YAML. Production-grade registry — out of scope.

## 7. QueryParser и извлечение client_id

**`parse_client_id_and_intent`** (`sme_causal/core/utils.py`) — regex-извлечение идентификатора клиента из NL-запроса. Паттерн `\b[CС]\d{6}\b` сопоставляет **и латинскую `C`, и кириллическую `С`** (визуально идентичны — пользователи часто переключают раскладку и не замечают). В возвращаемом `client_id` префикс нормализуется в латинскую `C`, чтобы дальнейший lookup по CSV (где ID генерируются с латинской `C`) не промахивался. Возвращает кортеж `(client_id | None, cleaned_text)` — текст запроса без ID идёт в `QueryParser` как intent.

**`QueryParser`** (`sme_causal/agent/agent_service.py`) — few-shot LLM prompt для парсинга intent-части NL-запроса → `ParsedQuery`:

| Поле | Назначение |
|------|------------|
| `action_type` | `"what_if"` (оценка заданной интервенции, запускает Pipeline) или `"optimize"` (диагностика профиля без delta, запускает `CausalAgent.explain_client` напрямую) |
| `delta` | `intervention_delta` для Pipeline (`{"New_Product_Offer": 1, ...}`) |
| `target_metric` | Обнаруженная в запросе целевая метрика — прокидывается в `Pipeline.run(..., target_metric=...)` (см. §3.0) |
| `label` | Человекочитаемая метка типа запроса — отображается в UI/CLI |
| `info_text` | Пояснение по парсингу для отображения в UI |
| `match_info` | Маркер `confident` / `similar` по каждому ключу `delta` — сигнал для synthesize-промпта о неточном совпадении |

Если интервенция не извлекается надёжно (`delta = {}` при `action_type = what_if`), кейс не запускается — entry-point показывает fallback-сообщение.
