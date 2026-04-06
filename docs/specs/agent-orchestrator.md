# Спецификация: Agent / Orchestrator

## 1. Обзор

LangGraph Orchestrator — центральный компонент. Управляет выполнением кейса через граф состояний: приём запроса → координация модулей (Policy, PSM, RAG, LLM) → проверка через Critic → сохранение в SQLite.

## 2. CaseState

```python
class CaseState(TypedDict):
    # Идентификация
    case_id: str                                    # UUID
    mode: Literal["evaluate", "recommend"]
    client_id: str                                  # "C000005"
    raw_query: Optional[str]                        # NL-запрос (если есть)

    # Контекст
    client_context: dict                            # 25 полей CONTEXT_FIELDS
    intervention_delta: dict                        # {"New_Product_Offer": 1, ...}

    # Policy
    policy_result: dict                             # {blocked, reasons, notes}

    # Estimation
    psm_result: Optional[dict]                      # {ok, ate, att, n_pairs} или None
    rag_chunks: list[str]                           # top-k чанков
    graph_dsl: str                                  # DSL-строка рёбер

    # Synthesis
    explanation: dict                               # {diagnosis, drivers_pos, drivers_neg, ...}

    # Critic
    critic_result: dict                             # {passed, issues}
    retry_count: int                                # 0 или 1

    # Recommend mode
    candidate_results: Optional[list[dict]]         # [{delta, psm_result, explanation, critic_result}, ...]
    selected_candidate: Optional[dict]              # лучший кандидат

    # Human review
    requires_human_review: bool
    review_reason: Optional[str]

    # Метаданные
    status: str                                     # intake|context|policy|estimation|synthesis|critic|done|aborted|degraded
    trace_id: Optional[str]
    latency_ms: Optional[int]
    abort_reason: Optional[str]
```

## 3. Узлы (nodes)

### 3.1 intake

- **Structured input** (Streamlit) → прямое заполнение `mode`, `client_id`, `intervention_delta`
- **NL-запрос** → few-shot LLM prompt → `ParsedQuery` (client_id, intent, delta, match_info)
- Генерация `case_id` (UUID4), `retry_count = 0`
- Ошибка парсинга → abort с `parse_error`

### 3.2 load_context

Загрузка строки из CSV по `client_id` → dict из 25 полей профиля. Не найден → abort с `not_found`.

### 3.3 policy_check

Rule-based проверки допустимости:

1. Предложение продукта, который у клиента уже есть
2. Кредитный лимит ниже текущего
3. Скидка выше текущей
4. Комбинация: продукт + изменение лимита одновременно
5. Нулевая или отрицательная скидка
6. **Cooldown:** проверка по SQLite — если для `client_id` есть завершённый кейс (`status = 'done'`) с тем же типом интервенции за последние 30 дней → блокировка. При недоступности SQLite: cooldown пропускается, `requires_human_review = True`, `review_reason = "cooldown не проверен: SQLite недоступен"`

Результат: `{blocked: bool, reasons: [str], notes: dict}`

**Переходы:** `blocked` → abort | `mode == "recommend"` → generate_candidates | `mode == "evaluate"` → estimation

### 3.4 generate_candidates (только recommend)

1. Получение списка допустимых интервенций для клиента на основе его профиля и policy-правил
2. Для каждого кандидата: estimation → synthesize → critic
3. Результаты в `candidate_results[]`: `{delta, psm_result, explanation, critic_result}`
4. Ранжирование: по `psm_result.att` (desc), приоритет `critic_result.passed == True`
5. Кандидаты с неустранёнными issues отбраковываются (причина в `rejection_reason`)
6. Победитель → `selected_candidate`, его данные копируются в основные поля CaseState

### 3.5 estimation (параллельный)

Три sub-task через LangGraph parallel branching:

| Sub-task | Input | Output | При ошибке |
|----------|-------|--------|------------|
| **PSM** | DataFrame, treatment из delta | `{ok, ate, att, n_pairs}` | `psm_result = None` |
| **RAG** | query из delta или raw_query, top_k=3 | `list[str]` чанков | `rag_chunks = []` |
| **Graph DSL** | graph_method, min_conf | DSL-строка рёбер | `graph_dsl = ""` |

### 3.6 synthesize

1. Выбор шаблона промпта (evaluate / recommend)
2. Заполнение: контекст клиента, delta, PSM-summary, RAG-чанки, Graph DSL
3. При retry: добавление "ИСПРАВЬТЕ СЛЕДУЮЩИЕ ПРОБЛЕМЫ: {issues}"
4. LLM вызов: JSON mode → text fallback → regex-парсинг
5. Парсинг в Explanation: `{diagnosis, drivers_pos, drivers_neg, expected_effect, recommendations, raw_text}`

Ошибка: LLM timeout после retry → abort с `llm_timeout`.

### 3.7 critic_check

5 rule-based проверок (подробнее в `observability-evals.md`):
1. Числовая консистентность ATT/ATE
2. Атрибуция источников (doc_id в rag_chunks)
3. Валидность рёбер графа
4. Хеджирование при слабых данных
5. Полнота обязательных полей

**Переходы:** pass → persist | fail + `retry_count == 0` → `retry_count := 1`, synthesize | fail + `retry_count >= 1` → persist_with_warning

### 3.8 persist

1. Вычисление `latency_ms`
2. Определение `status`: done / aborted / degraded
3. Определение `requires_human_review`:
   - `status == "degraded"` (tool fails)
   - Critic fail после retry
   - `|ATT| < 0.001` и мало matched pairs
   - Все кандидаты отбракованы (recommend mode)
   - SQLite был недоступен при cooldown-проверке
4. INSERT в SQLite (если доступен), логирование через Loguru

## 4. Stop conditions

- Максимум retry: 1. Бесконечных циклов нет.
- Граф завершается в persist (done/degraded/aborted)

## 5. Retry / Fallback

| Сценарий | Стратегия |
|----------|-----------|
| LLM: невалидный JSON | JSON mode → text mode fallback |
| Critic fail | Повторный synthesize с issues (макс. 1 retry) |
| PSM / RAG / Graph fail | Skip, degraded mode |
| LLM timeout | 1 retry, затем abort |
| SQLite недоступен | Cooldown skip (fail-open), persist skip, Loguru only |

## 6. Промпт-менеджмент

- **prompt_base**: базовый анализ. System: роль + инструкции + JSON-схема. User: профиль + признаки + граф.
- **prompt_whatif**: what-if. Дополнительно: delta, PSM-summary, RAG-чанки, match_info.
- Шаблоны: `ChatPromptTemplate.from_messages()` с переменными `{context}`, `{features_description}`, `{graph_block}`, `{what_if_block}`, `{psm_block}`, `{rag_context}`.

## 7. QueryParser

Few-shot LLM prompt для парсинга NL-запроса → `ParsedQuery`: client_id, intent (evaluate/recommend), delta, match_info (маркер "similar" для неточных совпадений).
