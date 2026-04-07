# Sequence: Happy Path

Sequence diagram для основного сценария «оценка заданной интервенции через Streamlit». Все 3 источника данных доступны, policy не блокирует, critic проходит с первой попытки.

Альтернативные пути (blocked policy, degraded mode, retry, abort) — см. `workflow.md`.

```mermaid
sequenceDiagram
    actor Analyst as 👤 Аналитик
    participant Agent as Agent Service<br/>(LangGraph)
    participant Tools as Local Tools<br/>(PSM, RAG, Graph)
    participant DB as SQLite
    participant LLM as OpenAI API

    Analyst->>Agent: client_id + intervention_delta
    Agent->>Agent: intake → load_context → policy_check
    Agent->>DB: cooldown lookup
    DB-->>Agent: allowed

    par estimation параллельно
        Agent->>Tools: PSM
    and
        Agent->>Tools: RAG
    and
        Agent->>Tools: Graph DSL
    end
    Tools-->>Agent: evidence bundle

    Agent->>LLM: synthesize prompt
    LLM-->>Agent: explanation JSON
    Agent->>Agent: critic check (passed)

    Agent->>DB: persist case
    Agent-->>Analyst: explanation + sources
```

## Что покрывает диаграмма

- Один полный успешный кейс от запроса аналитика до возврата объяснения
- Параллельное выполнение PSM, RAG и Graph DSL внутри estimation
- Один LLM-вызов в Synthesizer без retry
- Сохранение в SQLite через Persister

## Что НЕ покрывает

- Блокировку policy (см. `workflow.md` ветка `blocked`)
- Degraded mode при недоступности одного из источников
- Retry в Critic / повторный synthesize
- Abort на любом этапе

Полный набор переходов — в `workflow.md` (state machine).
