# Системный дизайн: Uplift Agent

## 1. Ключевые архитектурные решения

### ADR-1: LangGraph для оркестрации

Оркестрация на **LangGraph** — библиотека графов состояний поверх LangChain. Каждый шаг (intake, policy, estimation, synthesis, critic) — отдельный node с conditional edges. Нативная поддержка параллельных вызовов (PSM + RAG + Graph), интеграция с LangSmith, retry и routing «из коробки».

### ADR-2: PSM как детерминированный модуль

PSM-модуль (`CausalInferenceAnalyzer`, `PSMResult`) — самостоятельный компонент для количественной оценки эффекта (ATE/ATT, greedy 1:1 матчинг, caliper, auto-detect ковариат). Оборачивается как LangGraph tool node. Каузальная оценка — детерминированная задача, не требует LLM.

### ADR-3: Причинно-следственный граф как источник структурного контекста

DAG причинно-следственных связей между признаками клиентов (например, `Industry -> Revenue | sign:+ | conf:0.85`). Строится offline тремя методами: LLM-анализ, алгоритмический (PC/FCI), гибрид. Хранится как JSON с рёбрами `{source, target, sign, confidence, note}`. При запросе фильтруется по `confidence ≥ min_conf` и передаётся в промпт как DSL-строка. Даёт LLM структурное понимание каузальных механизмов — какие признаки влияют на outcome и через какие пути, что дополняет числовую оценку PSM и документальный контекст RAG.

### ADR-4: Rule-based Policy Checker и Critic

**Policy Checker** — rule-based проверки допустимости интервенций. **Critic** — rule-based проверки консистентности ответов LLM (5 проверок). Оба на Python if/else: safety-слой не должен зависеть от галлюцинаций LLM.

### ADR-5: SQLite для хранения кейсов

Одна таблица `cases`: case_id, mode, client_id, request/context/result как JSON, status, trace_id, timestamps. PoC-масштаб, минимум инфраструктуры. Используется также для cooldown-проверки в policy_check.

### ADR-6: LangSmith (free tier) + Loguru

LangSmith free tier (5000 traces/мес) для трейсинга LangGraph. Loguru для локальных структурированных логов и audit. Подробнее: `specs/observability-evals.md`.

### ADR-7: FAISS + e5-small для RAG

RAG: чанкинг → embedding (`multilingual-e5-small`, 384-dim) → FAISS `IndexFlatIP`. Для 50 документов brute-force оптимален. Подробнее: `specs/retriever.md`.

---

## 2. Модули

| # | Модуль | Роль | Реализация |
|---|--------|------|------------|
| 1 | Case Intake & Router | Приём запроса, определение mode/client_id | LangGraph node. Few-shot LLM для NL, прямое заполнение для Streamlit. |
| 2 | Client Context Retriever | Загрузка профиля из CSV | LangGraph node. 25 полей по client_id. |
| 3 | Policy & Eligibility | Блокировка невалидных интервенций | LangGraph node. Rule-based: дубликаты, лимиты, cooldown. |
| 4 | Candidate Generator | Список кандидатов (recommend mode) | LangGraph node. Получает список допустимых интервенций для клиента. |
| 5 | Causal Estimation | PSM: ATE/ATT | LangGraph tool node. Greedy 1:1 matching, caliper, auto-detect. |
| 6 | Causal Graph | Причинно-следственные связи между признаками | LangGraph tool node. Загрузка DAG из JSON, фильтрация по confidence → DSL для промпта. |
| 7 | Evidence Retrieval | RAG: поиск банковских документов | LangGraph tool node. FAISS + e5-small, top_k. |
| 8 | Critic / Guardrail | Пост-генерационные проверки | LangGraph node. 5 rule-based проверок. |
| 9 | Recommendation Synthesizer | Генерация Explanation через LLM | LangGraph node. Промпт-шаблоны + JSON/text fallback. |
| 10 | Case State & Memory | Персистентность кейсов | SQLite `cases`. Audit + cooldown. |

---

## 3. Workflow

### 3.1 CaseState

Все данные между узлами LangGraph через единый TypedDict:

```python
class CaseState(TypedDict):
    case_id: str                          # UUID
    mode: Literal["evaluate", "recommend"]
    client_id: str
    raw_query: str | None

    client_context: dict                  # 25 полей
    intervention_delta: dict              # {"New_Product_Offer": 1, ...}

    policy_result: dict                   # {blocked, reasons, notes}

    psm_result: dict | None               # {ok, ate, att, n_pairs}
    rag_chunks: list[str]
    graph_dsl: str

    explanation: dict                     # {diagnosis, drivers_pos, drivers_neg, ...}
    critic_result: dict                   # {passed, issues}
    retry_count: int

    # Recommend mode
    candidate_results: list[dict] | None  # [{delta, psm_result, explanation, critic_result}, ...]
    selected_candidate: dict | None

    # Human review
    requires_human_review: bool
    review_reason: str | None

    # Метаданные
    status: str                           # intake|...|done|aborted|degraded
    abort_reason: str | None
    trace_id: str | None
    latency_ms: int | None
```

### 3.2 Граф состояний

```
START
  │
  ▼
┌──────────────┐
│ intake       │  Парсинг запроса: mode, client_id, delta
└────┬─────────┘
     ▼
┌──────────────┐
│ load_context │  Профиль клиента из CSV
└────┬─────────┘
     ▼
┌──────────────┐
│ policy_check │  Eligibility + cooldown
└────┬─────────┘
     ├── blocked ─────────────▶ abort ─▶ persist ─▶ END
     │
     ├── mode = evaluate ─────▶ estimation ─▶ synthesize ─▶ critic_check
     │                          PSM + RAG + Graph            │
     │                                                       ├── pass ─▶ persist ─▶ END
     │                                                       ├── retry_count = 0
     │                                                       │   └── retry_count := 1 ─▶ synthesize
     │                                                       └── retry_count ≥ 1
     │                                                           └── persist_with_warning ─▶ END
     │
     └── mode = recommend ───▶ generate_candidates
                                │
                                ├── for each candidate:
                                │   estimation ─▶ synthesize ─▶ critic_check
                                │   save result to candidate_results[]
                                │
                                └── after all candidates:
                                    rank_and_select ─▶ persist ─▶ END
```

### 3.3 Описание узлов

Подробное описание каждого узла: `specs/agent-orchestrator.md`.

**Ключевые моменты:**
- **policy_check** включает cooldown через SQLite (30 дней). При недоступности SQLite — fail-open с `requires_human_review = True`.
- **generate_candidates** (recommend mode) — получает список допустимых интервенций для клиента, для каждого кандидата полный цикл estimation → synthesize → critic. Ранжирование по `att` desc.
- **estimation** — параллельный запуск PSM + RAG + Graph. Каждый может fail независимо → degraded mode.
- **critic_check** — 5 rule-based проверок. Макс. 1 retry.

---

## 4. State / Memory

### 4.1 Сессионное состояние

`CaseState` живёт в памяти LangGraph на время одного кейса. Агент **не использует** результаты прошлых кейсов для генерации рекомендаций — каждый запрос независимый. SQLite audit log используется только для cooldown-проверки в policy_check (safety-механизм, не агентская память).

### 4.2 SQLite

```sql
CREATE TABLE cases (
    case_id       TEXT PRIMARY KEY,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mode          TEXT NOT NULL,
    client_id     TEXT NOT NULL,
    raw_query     TEXT,
    request_json  TEXT NOT NULL,
    context_json  TEXT NOT NULL,
    result_json   TEXT,
    status        TEXT NOT NULL CHECK (status IN ('done', 'aborted', 'degraded')),
    abort_reason  TEXT,
    requires_human_review BOOLEAN DEFAULT FALSE,
    review_reason TEXT,
    trace_id      TEXT,
    latency_ms    INTEGER,
    updated_at    TIMESTAMP
);
```

### 4.3 Бюджет контекста

~5000 tokens input на вызов LLM. Защита: модуль truncation (tiktoken, 2000 tokens/msg). При нехватке бюджета приоритет сокращения: RAG → Graph DSL → PSM-summary. System prompt и профиль не сокращаются.

Подробнее: `specs/memory-context.md`.

### 4.4 Retention

- SQLite `cases`: 365 дней
- Loguru: 90 дней

---

## 5. Источники данных для estimation

Три параллельных источника обогащают контекст рекомендации:

### 5.1 PSM (количественная оценка эффекта)

Propensity Score Matching по синтетическому датасету (3000 клиентов). Результат: ATE, ATT, n_pairs — числовая оценка ожидаемого эффекта интервенции. Подробнее: `specs/tools-apis.md` §3.

### 5.2 Причинно-следственный граф

DAG причинно-следственных связей между признаками клиентов. Строится offline (LLM-анализ, алгоритмические методы или гибрид). Хранится как JSON с рёбрами `{source, target, sign, confidence, note}`. При запросе: фильтрация по `confidence ≥ min_conf` → DSL-строка для промпта вида `"Industry -> Revenue | sign:+ | conf:0.85"`. Даёт LLM структурное понимание каузальных механизмов. Подробнее: `specs/tools-apis.md` §4.

### 5.3 RAG (документальный контекст)

FAISS `IndexFlatIP` + `multilingual-e5-small` (384-dim). 50 банковских документов (рус + англ). Чанкинг: `RecursiveCharacterTextSplitter` (1500/120/1000). Поиск: top_k=3 для what-if. Даёт LLM контекст из банковской аналитики. Подробнее: `specs/retriever.md`.

---

## 6. Tool/API интеграции

| Tool | Контракт | Latency | При ошибке |
|------|----------|---------|------------|
| **OpenAI LLM** | `ChatOpenAI.invoke(messages) → AIMessage`. JSON mode + text fallback. | ~3-7с | Retry (1), затем abort |
| **PSM** | `CausalInferenceAnalyzer.run(df) → PSMResult(ate, att, n_pairs)` | < 5с | `psm_result = None`, degraded |
| **Graph DSL** | `load_graph_dsl(method, min_conf) → str`. In-memory кэш. | < 100мс | `graph_dsl = ""`, degraded |
| **RAG** | `RAG.perform_query(query, top_k) → list[str]`. FAISS search. | < 2с | `rag_chunks = []`, degraded |

Подробнее: `specs/tools-apis.md`.

---

## 7. Failure modes и guardrails

### 7.1 Failure modes

| Сценарий | Защита | Остаточный риск |
|----------|--------|-----------------|
| LLM: невалидный JSON | JSON mode → text fallback → regex | Низкий |
| LLM галлюцинирует рёбра | Critic check: ребро есть в graph DSL | Средний |
| PSM fail (мало данных) | Degraded mode, промпт без PSM | Средний |
| Policy блокирует | Ранний return с причиной | Низкий |
| RAG: нерелевантные чанки | Critic: doc_id есть в retrieved chunks | Средний |
| Overconfident при слабых данных | Critic: маркеры категоричности | Средний |
| OpenAI API недоступен | Retry (1), затем abort | Средний |
| SQLite недоступен | Cooldown skip (fail-open) + human review flag. Persist skip, только Loguru. | Средний |

### 7.2 Critic: 5 проверок

1. **Числовая консистентность**: ATT/ATE в тексте ↔ `psm_result`
2. **Атрибуция**: цитируемый doc_id ∈ `rag_chunks`
3. **Валидность рёбер**: "A → B" ∈ `graph_dsl`
4. **Хеджирование**: при слабых PSM-данных нет категоричных формулировок
5. **Полнота**: все обязательные поля Explanation заполнены

Retry: инъекция issues в промпт (макс. 1 retry). Подробнее: `specs/observability-evals.md`.

### 7.3 Safe failure policy

При неустранимом сбое или низкой уверенности: `requires_human_review = True`, disclaimer в Explanation, явная причина ограничения, trace для диагностики.

**Триггеры human review:**
- `status == "degraded"` (tool fails)
- `|ATT| < 0.001` с малым числом пар
- Critic fail после retry
- Все кандидаты отбракованы (recommend mode)
- SQLite недоступен при cooldown

---

## 8. Ограничения

| Метрика | Target (PoC) | Обоснование |
|---------|-------------|-------------|
| p95 latency (SLO) | ≤ 5 минут | Формальное обязательство |
| p95 latency (target) | ≤ 3 минуты | 1–2 LLM-вызова + PSM + RAG + retry |
| Cost на кейс | < $1 | ~$0.002 при GPT-4o-mini |
| Availability | ≥ 95% | Основной риск: OpenAI API |
| Error rate | < 5% | Fallback-стратегии |
| Concurrency | Single-user | PoC |
| Embedding model | `multilingual-e5-small` | Фиксирован |
| Синтетических клиентов | 3000 | Достаточно для PSM |
| RAG-корпус | 50 документов | Основные банковские темы |
