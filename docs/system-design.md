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

Одна таблица `cases`: case_id, client_id, request/context/result как JSON, status, trace_id, timestamps. PoC-масштаб, минимум инфраструктуры. Используется также для cooldown-проверки в policy_check.

### ADR-6: LangSmith (free tier) + Loguru

LangSmith free tier (5000 traces/мес) для трейсинга LangGraph. Loguru для локальных структурированных логов и audit. Подробнее: `specs/observability-evals.md`.

### ADR-7: FAISS + e5-small для RAG

RAG: чанкинг → embedding (`multilingual-e5-small`, 384-dim) → FAISS `IndexFlatIP`. Для 50 документов brute-force оптимален. Подробнее: `specs/retriever.md`.

---

## 2. Модули

| # | Модуль | Роль | Реализация |
|---|--------|------|------------|
| 1 | Case Intake & Router | Приём запроса, определение client_id и intervention_delta | LangGraph node. Few-shot LLM для NL, прямое заполнение для Streamlit. |
| 2 | Client Context Retriever | Загрузка профиля из CSV | LangGraph node. 25 полей по client_id. |
| 3 | Policy & Eligibility | Блокировка невалидных интервенций | LangGraph node. Rule-based: дубликаты, лимиты, cooldown. |
| 4 | Causal Estimation | PSM: ATE/ATT | LangGraph tool node. Greedy 1:1 matching, caliper, auto-detect. |
| 5 | Causal Graph | Причинно-следственные связи между признаками | LangGraph tool node. Загрузка DAG из JSON, фильтрация по confidence → DSL для промпта. |
| 6 | Evidence Retrieval | RAG: поиск банковских документов | LangGraph tool node. FAISS + e5-small, top_k. |
| 7 | Critic / Guardrail | Пост-генерационные проверки | LangGraph node. 5 rule-based проверок. |
| 8 | Intervention Synthesizer | Генерация Explanation через LLM | LangGraph node. Промпт-шаблоны + JSON/text fallback. |
| 9 | Case State & Memory | Персистентность кейсов | SQLite `cases`. Audit + cooldown. |

---

## 3. Workflow

### 3.1 CaseState

Все данные между узлами LangGraph передаются через единый `CaseState`. На уровне системного дизайна важны 5 групп полей:

- **Идентификация:** `case_id`, `client_id`, `raw_query`
- **Контекст кейса:** `client_context`, `intervention_delta`
- **Промежуточные результаты:** `policy_result`, `psm_result`, `rag_chunks`, `graph_dsl`
- **Финальный ответ и контроль качества:** `explanation`, `critic_result`, `retry_count`
- **Служебные метаданные:** `status`, `requires_human_review`, `review_reason`, `trace_id`, `latency_ms`, `abort_reason`

Полное определение TypedDict и семантика каждого поля вынесены в `specs/agent-orchestrator.md`.

### 3.2 Граф состояний

```
START
  │
  ▼
┌──────────────┐
│ intake       │  Парсинг запроса: client_id, delta
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
     └── allowed ─────────────▶ estimation ─▶ synthesize ─▶ critic_check
                                PSM + RAG + Graph            │
                                                              ├── pass ─▶ persist ─▶ END
                                                              ├── retry_count = 0
                                                              │   └── retry_count := 1 ─▶ synthesize
                                                              └── retry_count ≥ 1
                                                                  └── persist_with_warning ─▶ END
```

### 3.3 Описание узлов

Подробное описание каждого узла: `specs/agent-orchestrator.md`.

**Ключевые моменты:**
- **policy_check** включает cooldown через SQLite (30 дней). При недоступности SQLite — fail-open с `requires_human_review = True`.
- **estimation** — параллельный запуск PSM + RAG + Graph. Каждый может fail независимо → degraded mode. Если все 3 источника недоступны — abort с `no_evidence` (синтез без данных не имеет смысла).
- **PSM** — при `n_pairs < 50` возвращает `ok=False` (числа доступны, но ненадёжны).
- **critic_check** — 5 rule-based проверок. Макс. 1 retry.

---

## 4. State / Memory

`CaseState` живёт в памяти LangGraph только на время одного кейса. Агент не использует прошлые кейсы как memory для новых оценок; SQLite нужен для audit trail и cooldown safety-check.

Ключевые решения:
- **Персистентность:** одна таблица `cases` в SQLite для результата кейса, статуса, review-маркеров и `trace_id`
- **Контекстный бюджет:** около `~5000` input tokens на LLM-вызов; порядок сокращения при переполнении: `RAG → Graph DSL → PSM-summary`
- **Retention:** `SQLite cases = 365 дней`, `Loguru = 90 дней`, `LangSmith` — по политике внешнего сервиса

Полная схема SQLite, cooldown-query, TTL и PII-политика вынесены в `specs/memory-context.md`.

---

## 5. Источники данных для estimation

Три параллельных источника обогащают pipeline оценки интервенции:

- **PSM:** числовая оценка эффекта интервенции (`ATE`, `ATT`, `n_pairs`)
- **Причинно-следственный граф:** структурный контекст о связях между признаками
- **RAG:** документальный контекст из банковского корпуса

Подробные контракты и параметры вынесены в `specs/tools-apis.md` и `specs/retriever.md`.

---

## 6. Tool/API интеграции

Интеграции делятся на две группы:

- **Внешний runtime dependency:** OpenAI API для LLM-вызовов. Используется JSON mode с text fallback; защитный timeout — `120с`.
- **Локальные инструменты:** PSM, Graph DSL loader и RAG. Все три могут деградировать независимо, не останавливая весь кейс, если это не ломает safety-политику.

Подробные контракты, параметры, latency и side effects вынесены в `specs/tools-apis.md`.

---

## 7. Failure modes и guardrails

Система опирается на три слоя защиты:

- **Rule-based policy-check:** блокирует недопустимые интервенции до вызова synthesis
- **Degraded execution:** PSM, RAG и Graph могут падать независимо; кейс продолжается с пониженной уверенностью. Если все 3 недоступны — abort (`no_evidence`)
- **Critic / guardrail:** проверяет числовую консистентность, атрибуцию источников, валидность рёбер графа, хеджирование и полноту ответа

Safe failure policy:
- максимум `1` retry на этапе synthesis после critic fail;
- при неустранимой ошибке или слабой уверенности выставляется `requires_human_review = True`;
- типовые триггеры: `status == degraded`, `|ATT| < 0.001` при слабых данных, critic fail после retry, недоступность SQLite при cooldown.

Полный набор failure modes, critic-checks и observability-метрик вынесен в `specs/observability-evals.md` и `governance.md`.

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
