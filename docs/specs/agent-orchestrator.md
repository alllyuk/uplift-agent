# Спецификация: Agent / Orchestrator

## 1. Обзор

LangGraph Orchestrator — центральный компонент. Управляет выполнением кейса через граф состояний: приём запроса → координация модулей (Policy, PSM, RAG, LLM) → проверка через Critic → сохранение в SQLite.

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

### 3.1 intake

- **Structured input** (Streamlit) → прямое заполнение `client_id`, `intervention_delta`
- **NL-запрос** → few-shot LLM prompt → `ParsedQuery` (client_id, delta, match_info)
- Генерация `case_id` (UUID4), `retry_count = 0`
- Ошибка парсинга или отсутствие интервенции → abort с `parse_error` / `missing_intervention`

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

**Переходы:** `blocked` → abort | иначе → estimation

### 3.4 estimation (параллельный)

Три sub-task через LangGraph parallel branching:

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

Ошибка: LLM timeout после retry → abort с `llm_timeout`.

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
   - `status == "degraded"` (tool fails)
   - Critic fail после retry
   - `|ATT| < 0.001` и мало matched pairs
   - SQLite был недоступен при cooldown-проверке
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
| PSM: `n_pairs < 50` | `ok=False`, числа доступны но ненадёжны |
| LLM timeout | 1 retry, затем abort |
| SQLite недоступен | Cooldown skip (fail-open), persist skip, Loguru only |

## 6. Промпт-менеджмент

### 6.1 Шаблоны

- **prompt_base**: базовый анализ. System: роль + инструкции + JSON-схема. User: профиль + признаки + граф.
- **prompt_whatif**: оценка заданной интервенции. Дополнительно: delta, PSM-summary, RAG-чанки, match_info.
- Шаблоны: `ChatPromptTemplate.from_messages()` с переменными `{context}`, `{features_description}`, `{graph_block}`, `{what_if_block}`, `{psm_block}`, `{rag_context}`.

### 6.2 Версионирование

- **Storage:** шаблоны лежат как файлы в `prompts/{name}/{version}.yaml` (например `prompts/whatif/v2.1.yaml`). YAML содержит `system`, `user`, и метаданные `version`, `created_at`, `parent_version`, `notes`.
- **Naming:** семантическое `vMAJOR.MINOR`. MAJOR — несовместимое изменение схемы переменных, MINOR — текстовая правка.
- **Активация:** активная версия выбирается через существующий `LLMConfig` (`specs/serving-config.md` §2.2) — поля `prompt_version_base`, `prompt_version_whatif`. По умолчанию — последний MAJOR.
- **Логирование:** фактически использованные версии записываются в `CaseState.prompt_versions` и в SQLite (`cases.prompt_versions_json`) для аудита и воспроизводимости.
- **Rollback:** смена версии в config + рестарт сервиса. Старые версии не удаляются — остаются в репозитории и могут быть включены обратно.
- **PoC scope:** 2 шаблона (`base`, `whatif`), смена редкая, без автоматического prompt registry. Production-grade registry — out of scope.

## 7. QueryParser

Few-shot LLM prompt для парсинга NL-запроса → `ParsedQuery`: client_id, delta, match_info (маркер "similar" для неточных совпадений). Если интервенция не извлекается надёжно, кейс не запускается.
