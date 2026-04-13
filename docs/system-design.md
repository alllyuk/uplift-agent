# Системный дизайн: Uplift Agent

## 1. Ключевые архитектурные решения

### ADR-1: LangGraph для оркестрации

Оркестрация на **LangGraph** — библиотека графов состояний поверх LangChain. Каждый шаг (intake, policy, estimation, synthesis, critic) — отдельный node с conditional edges. Нативная поддержка параллельных вызовов (PSM + RAG + Graph), интеграция с LangSmith, retry и routing «из коробки».

### ADR-2: PSM как детерминированный модуль

PSM-модуль (`CausalInferenceAnalyzer`, `PSMResult`) — самостоятельный компонент для количественной оценки эффекта (ATE/ATT, greedy 1:1 матчинг, caliper, auto-detect ковариат). Оборачивается как LangGraph tool node. Каузальная оценка — детерминированная задача, не требует LLM.

### ADR-3: Причинно-следственный граф как источник структурного контекста

DAG причинно-следственных связей между признаками клиентов (например, `Industry -> Revenue | sign:+ | conf:0.85`). Строится offline тремя методами: LLM-анализ, алгоритмический (PC/FCI), гибрид. Хранится как JSON с рёбрами `{source, target, sign, confidence, note}`. При запросе фильтруется по `confidence ≥ min_conf` и передаётся в промпт как DSL-строка. Даёт LLM структурное понимание каузальных механизмов — какие признаки влияют на outcome и через какие пути, что дополняет числовую оценку PSM и документальный контекст RAG.

### ADR-4: Rule-based Policy Checker и Critic

**Policy Checker** — rule-based проверки допустимости интервенций. **Critic** — двухуровневая проверка: Level 1 (rule-based структурные проверки: атрибуция источников, полнота ответа) + Level 2 (LLM-augmented семантические проверки). Safety-слой (L1) не зависит от галлюцинаций LLM.

### ADR-5: SQLite для хранения кейсов

Одна таблица `cases`: case_id, client_id, request/context/result как JSON, status, trace_id, timestamps. PoC-масштаб, минимум инфраструктуры. Используется также для cooldown-проверки в policy_check.

### ADR-6: LangSmith (free tier) + Loguru

LangSmith free tier (5000 traces/мес) для трейсинга LangGraph. Loguru для локальных структурированных логов и audit. Подробнее: `specs/observability-evals.md`.

### ADR-7: FAISS + e5-small для RAG

RAG: чанкинг → embedding (`multilingual-e5-small`, 384-dim) → FAISS `IndexFlatIP`. Для 50 документов brute-force оптимален. Подробнее: `specs/retriever.md`.

### ADR-8: Hybrid agentность — rule-based safety floor + LLM-driven refinement

Чтобы система соответствовала позиционированию «агентная», но не теряла предсказуемости и safety, выбран гибридный дизайн:

- **Rule-based safety floor:** Policy, Critic level 1 (2 структурные проверки: атрибуция источников, полнота ответа), determination статуса в Persister, conditional edges по результатам — всё детерминированное. Эти слои защищают от галлюцинаций LLM и делают систему тестируемой.
- **LLM-driven refinement loops поверх floor:** два места, где LLM реально управляет выполнением, а не только парсит/синтезирует:
  1. **Adaptive RAG** (`rag_refine`, см. `specs/agent-orchestrator.md` §3.7) — LLM формулирует уточнённый RAG-запрос на основе critic issues и инициирует повторный retrieval. Bounded: max 2 итерации.
  2. **LLM-augmented critic** (level 2, см. `specs/observability-evals.md` §4.4) — LLM проверяет смысловую консистентность объяснения с числами/документами, дополняя rule-based проверки.

Эта гибридность даёт реальный LLM-in-the-loop pattern (synthesize → critic → rag_refine → synthesize), но с жёсткими safety-границами: rule-based проверки всегда выполняются и блокируют некорректный вывод независимо от того, что предложил LLM.

**Что НЕ делается** (осознанно): LLM-driven planner, conditional tool selection через LLM перед estimation, ReAct цикл с произвольным числом итераций. Эти расширения возможны позже, но в PoC они увеличили бы поверхность атаки prompt injection и нарушили принцип «safety не зависит от LLM».

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
| 7 | Critic / Guardrail | Пост-генерационные проверки | LangGraph node. **2 уровня:** 5 rule-based + LLM-augmented смысловые проверки. |
| 8 | Intervention Synthesizer | Генерация Explanation через LLM | LangGraph node. Промпт-шаблоны + JSON/text fallback. |
| 9 | RAG Refiner | LLM-driven переформулировка RAG-запроса | LangGraph node. Запускается между critic fail и retry-synthesize, max 1 итерация. |
| 10 | Case State & Memory | Персистентность кейсов | SQLite `cases`. Audit + cooldown. |

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
                                PSM + RAG + Graph            │  (rule + LLM)
                                                              │
                                                              ├── pass ─▶ persist ─▶ END
                                                              │
                                                              ├── fail + retry_count = 0
                                                              │   └─▶ rag_refine ─▶ synthesize  (retry_count := 1)
                                                              │       (LLM формулирует уточнённый RAG query)
                                                              │
                                                              └── fail + retry_count ≥ 1
                                                                  └─▶ persist_with_warning ─▶ END
```

### 3.3 Описание узлов

Подробное описание каждого узла: `specs/agent-orchestrator.md`.

**Ключевые моменты:**
- **policy_check** включает cooldown через SQLite (30 дней). При недоступности SQLite — fail-open с `requires_human_review = True`.
- **estimation** — параллельный запуск PSM + RAG + Graph. Каждый может fail независимо → degraded mode. Если все 3 источника недоступны — abort с `no_evidence` (синтез без данных не имеет смысла).
- **PSM** — при `n_pairs < 50` возвращает `ok=False` (числа доступны, но ненадёжны).
- **critic_check** — 2 уровня: 5 rule-based проверок + LLM-augmented смысловые проверки. Макс. 1 retry, перед retry запускается `rag_refine`.
- **rag_refine** — LLM формулирует уточнённый RAG-запрос на основе critic issues, делает повторный retrieval (max `rag_iterations = 2`). Это первая точка реального LLM-driven control flow в системе (см. ADR-8).

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
- **Critic / guardrail:** Level 1 (rule-based) проверяет атрибуцию источников и полноту ответа; Level 2 (LLM-augmented) проверяет логическую консистентность, соответствие фактам, адекватность хеджирования и полноту рекомендаций

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

---

## 9. Масштабирование за пределы PoC

Раздел описывает ориентиры миграции PoC → production. Цель — показать, что текущая архитектура не блокирует рост, а ограничения §8 — сознательный выбор для PoC, а не архитектурный долг.

### 9.1 Текущий PoC

Single-user, in-process Streamlit, SQLite (single-writer), FAISS in-memory, один LLM API key, 3000 синтетических клиентов, 50 RAG-документов.

### 9.2 Узкие места и решения

| Компонент | PoC | Production |
|-----------|-----|------------|
| Agent Service | in-process | Stateless + N воркеров за load balancer (FastAPI/uvicorn) |
| Кейсы | sync HTTP | Async через очередь (Celery / RQ + Redis broker) |
| Client Data | CSV | PostgreSQL + индексы по сегментам |
| Case Store | SQLite | PostgreSQL (multi-writer, partitioning по `created_at`) |
| Retrieval | FAISS in-memory | Milvus / Weaviate (distributed, HNSW) |
| LLM | один API key | Multi-key + provider abstraction (OpenAI / Anthropic / собственный inference) с fallback |
| Observability | Loguru файлы | Loki / ELK + Prometheus + Grafana |
| Secrets | `.env` | Vault / AWS Secrets Manager |

### 9.3 Что НЕ масштабируется тривиально

- **PSM:** greedy 1:1 matching — `O(n²)`, дорого при росте популяции до миллионов клиентов. Решение — approximate matching или предвычисленный propensity index.
- **Cooldown-проверка:** требует индекс `(client_id, intervention_type, created_at)` и, при росте кейсов, денормализованную таблицу последних интервенций.

### 9.4 Этапы миграции PoC → production

1. Streamlit → REST API (см. `specs/rest-api.md`)
2. SQLite → PostgreSQL (миграционный скрипт)
3. CSV → PostgreSQL для клиентов
4. Sync → async через очередь
5. FAISS → Milvus / Weaviate
6. Multi-worker deployment (Kubernetes / docker-compose)

Каждый шаг — отдельная итерация; LangGraph и rule-based слои (Policy, Critic) переносятся без изменений.
