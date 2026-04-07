# Sequence: Happy Path

Sequence diagram для основного сценария «оценка заданной интервенции через Streamlit». Все 3 источника данных доступны, policy не блокирует, critic проходит с первой попытки.

Альтернативные пути (blocked policy, degraded mode, retry, abort) — см. `workflow.md`.

```mermaid
sequenceDiagram
    actor Analyst as 👤 Аналитик
    participant UI as Streamlit UI
    participant Orch as LangGraph Orchestrator
    participant CSV as Client Data (CSV)
    participant Policy as Policy Checker
    participant DB as SQLite (cases)
    participant PSM as PSM Tool
    participant RAG as RAG (FAISS)
    participant Graph as Graph DSL Loader
    participant Synth as Synthesizer
    participant LLM as OpenAI API
    participant Critic as Critic / Guardrail
    participant Persist as Persister

    Analyst->>UI: client_id + intervention_delta
    UI->>Orch: запуск кейса
    Orch->>Orch: intake (генерация case_id)
    Orch->>CSV: load_context(client_id)
    CSV-->>Orch: client_context (25 полей)

    Orch->>Policy: check(context, delta)
    Policy->>DB: cooldown lookup (client_id, type, 30d)
    DB-->>Policy: no recent intervention
    Policy-->>Orch: allowed

    par estimation (параллельно)
        Orch->>PSM: run(df, treatment from delta)
        PSM-->>Orch: {ok, ate, att, n_pairs}
    and
        Orch->>RAG: query(top_k=3)
        RAG-->>Orch: chunks[]
    and
        Orch->>Graph: load_dsl(method, min_conf)
        Graph-->>Orch: graph_dsl
    end

    Orch->>Synth: synthesize(context, delta, evidence)
    Synth->>LLM: prompt (JSON mode)
    LLM-->>Synth: explanation JSON
    Synth-->>Orch: explanation

    Orch->>Critic: check 5 rules
    Critic-->>Orch: passed=true

    Orch->>Persist: save(state)
    Persist->>DB: INSERT case
    Persist-->>Orch: status=done, latency_ms

    Orch-->>UI: финальный ответ
    UI-->>Analyst: explanation + PSM + sources
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
