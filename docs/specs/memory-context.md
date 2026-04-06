# Спецификация: Memory / Context

## 1. Обзор

Модуль управления состоянием и памятью:
- Передача данных между узлами LangGraph (CaseState)
- Персистентное хранение завершённых кейсов (SQLite)
- Контроль бюджета контекста для LLM-вызовов

## 2. Session State (CaseState)

```
Создание (intake) → Обогащение (load_context, estimation) → Синтез → Проверка → Персистентность
```

- `CaseState` (TypedDict) создаётся в `intake`, обогащается каждым узлом
- Передаётся по ссылке через LangGraph — без сериализации между узлами
- Время жизни: от создания до завершения кейса (секунды–минуты)

### 2.1 Кросс-кейсовая память

Агент **не использует** результаты прошлых кейсов для генерации новых рекомендаций. Каждый кейс обрабатывается независимо — прошлые Explanation не влияют на новые.

**Исключение:** SQLite audit log используется для операционной проверки cooldown в policy_check (§3.3) — запрет повторной интервенции того же типа для клиента в течение 30 дней. Это safety-механизм, а не агентская память.

Просмотр истории кейсов по client_id доступен через UI/API, но не влияет на решения агента.

## 3. Персистентное хранилище (SQLite)

### 3.1 Схема

```sql
CREATE TABLE IF NOT EXISTS cases (
    case_id       TEXT PRIMARY KEY,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mode          TEXT NOT NULL,
    client_id     TEXT NOT NULL,
    raw_query     TEXT,
    request_json  TEXT NOT NULL,          -- JSON: intervention_delta
    context_json  TEXT NOT NULL,          -- JSON: client_context
    result_json   TEXT,                   -- JSON: Explanation (nullable при abort)
    status        TEXT NOT NULL CHECK (status IN ('done', 'aborted', 'degraded')),
    abort_reason  TEXT,
    requires_human_review BOOLEAN DEFAULT FALSE,
    review_reason TEXT,
    trace_id      TEXT,                   -- LangSmith trace ID
    latency_ms    INTEGER,
    updated_at    TIMESTAMP
);

CREATE INDEX idx_cases_client_id ON cases(client_id);
CREATE INDEX idx_cases_status ON cases(status);
CREATE INDEX idx_cases_created_at ON cases(created_at);
```

### 3.2 Операции

| Операция | Когда | Кто |
|----------|-------|-----|
| INSERT | После завершения кейса | persist node |
| SELECT by case_id | Просмотр результата | UI / API |
| SELECT by client_id + status | Cooldown-проверка (policy_check) и просмотр истории | policy_check node / UI |
| DELETE (по retention) | TTL 365 дней | Cron / startup cleanup |

### 3.3 Cooldown-запрос

```sql
SELECT 1 FROM cases
WHERE client_id = ? AND status = 'done'
  AND request_json LIKE ? -- тип интервенции
  AND created_at > datetime('now', '-30 days')
LIMIT 1;
```

При недоступности SQLite: cooldown пропускается (fail-open), кейс продолжается с `requires_human_review = True`.

Concurrency: single-writer (PoC, один пользователь). Default journal mode.

## 4. Context Budget

| Компонент | ~Tokens | Обязательный |
|-----------|---------|--------------|
| System prompt + JSON schema | ~2000 | Да |
| Описание признаков | ~500 | Да |
| Профиль клиента | ~300 | Да |
| What-if delta | ~50 | Да (evaluate) |
| Graph DSL | ~500 | Нет (degraded ok) |
| RAG-чанки (top_k=3) | ~1500 | Нет (degraded ok) |
| PSM-summary | ~100 | Нет (degraded ok) |
| **Итого input** | **~4950** | |
| **Output** | **~500–1000** | |

**Защита:** модуль truncation — hard limit 2000 tokens/сообщение (tiktoken gpt2). При нехватке бюджета: сначала уменьшить RAG (top_k), затем убрать Graph DSL, затем PSM-summary. System prompt и профиль не сокращаются.

## 5. Retention и PII

| Тип данных | Срок | Примечание |
|------------|------|------------|
| SQLite `cases` | 365 дней | Audit trail по кейсу: запрос, контекст, результат, статус, `trace_id`, причины abort/review |
| Loguru файлы | 90 дней | Технические логи: ошибки, latency, вызовы модулей, служебные статусы |
| LangSmith traces | По политике free tier | Внешний сервис, не управляется локальным TTL |
| CaseState (in-memory) | Время кейса | Освобождается после persist |

Пояснение по retention:
- `SQLite cases` хранится дольше, потому что используется как audit trail и как источник для cooldown-проверки.
- `Loguru` хранится меньше, потому что это операционные логи для отладки и postmortem.
- По истечении срока хранения SQLite-записи удаляются cleanup-процедурой, а Loguru-файлы — политикой `retention` в конфигурации.

Пояснение по PII:
- прямые идентификаторы маскируются перед передачей в LLM;
- в PoC используются синтетические `client_id`;
- в логах пишется `case_id`, а не открытые идентификаторы клиента;
- полные промпты и полный retrieved-контент не сохраняются в технических логах.
