# Спецификация: REST API

## 1. Обзор и статус

> **Статус в v1: не реализовано.** REST API описан как планируемый контракт v2 для внешних интеграций (CRM, batch-вызовы, тестовые скрипты). В v1 единственные entry points — Streamlit UI и CLI (`sme_causal.app.run`), оба вызывают `Pipeline` in-process. Каталога `sme_causal/api/` и FastAPI-зависимостей в v1 нет.

Когда REST будет реализован в v2, это будет тонкая обёртка поверх существующего `Pipeline` orchestrator на FastAPI; никаких изменений в ядре оркестратора не потребуется.

## 2. Endpoints

Все эндпоинты ниже — **planned для v2**, в v1 не доступны.

| Метод | Путь | Назначение | Статус |
|-------|------|-----------|--------|
| POST | `/v1/cases` | Создать кейс (sync или async) | planned v2 |
| GET | `/v1/cases/{case_id}` | Получить результат кейса | planned v2 |
| GET | `/v1/cases?client_id=...&status=...` | Список кейсов клиента | planned v2 |
| GET | `/v1/clients/{client_id}` | Профиль клиента (read-only) | planned v2 |
| GET | `/v1/health` | Health-check компонентов | planned v2 |

### 2.1 POST /v1/cases

**Request:**

```json
{
  "client_id": "C000005",
  "intervention_delta": {"New_Product_Offer": 1, "New_Product_Offer_Type": "acquiring"}
}
```

Альтернативно:

```json
{"raw_query": "Стоит ли предложить эквайринг клиенту C000005?"}
```

**Response 201** (sync, default):

```json
{
  "case_id": "uuid",
  "status": "done",
  "explanation": {...},
  "psm_result": {...},
  "requires_human_review": false
}
```

**Response 202** (async, `?async=true`):

```json
{"case_id": "uuid", "status": "running"}
```

### 2.2 GET /v1/cases/{case_id}

Возвращает поля из таблицы `cases` (см. `memory-context.md` §3.1) + parsed `explanation`.

### 2.3 GET /v1/health

```json
{
  "status": "ok",
  "components": {
    "llm": "ok",
    "faiss": "ok",
    "sqlite": "ok"
  }
}
```

## 3. Schemas

Все схемы базируются на `CaseState` (`agent-orchestrator.md` §2). Pydantic-модели генерируются автоматически и публикуются через OpenAPI (см. §6). Дублирование схем здесь не делаем.

## 4. Auth

| Среда | Механизм |
|-------|----------|
| PoC | API key через заголовок `X-API-Key` |
| Production | OAuth2 / JWT с scopes (маппинг на роли из `governance.md` §5: analyst / reviewer / admin) |

API key хранится в `.env` (`API_KEY`), не логируется (см. `governance.md` §2).

## 5. Errors

Единый формат:

```json
{"error": {"code": "policy_blocked", "message": "...", "case_id": "uuid"}}
```

| HTTP | Code | Когда |
|------|------|-------|
| 400 | `invalid_request` | Невалидный JSON, отсутствует обязательное поле |
| 404 | `client_not_found` / `case_not_found` | Нет такого ресурса |
| 409 | `cooldown_blocked` | Policy: cooldown активен |
| 422 | `policy_blocked` | Policy: интервенция недопустима |
| 429 | `rate_limited` | Превышен лимит запросов |
| 500 | `internal_error` | Непредвиденная ошибка |
| 503 | `llm_unavailable` | OpenAI / inference недоступен |


## 6. Rate limiting

- PoC: фиксированный лимит 10 req/min на API key, in-memory token bucket
- Production: per-role лимиты, Redis-backed bucket

