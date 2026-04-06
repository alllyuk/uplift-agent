# Workflow: LangGraph State Machine

Пошаговое выполнение запроса через граф состояний LangGraph.

## Основной workflow

```mermaid
stateDiagram-v2
    [*] --> intake

    intake --> load_context : request parsed
    intake --> persist_aborted : parse error / missing intervention

    load_context --> policy_check : context loaded
    load_context --> persist_aborted : client not found

    policy_check --> estimation : allowed case
    policy_check --> persist_aborted : blocked by policy

    state estimation {
        [*] --> psm_compute
        [*] --> rag_query
        [*] --> graph_load
        psm_compute --> [*]
        rag_query --> [*]
        graph_load --> [*]
    }

    estimation --> synthesize : evidence ready
    estimation --> persist_aborted : all 3 sources failed

    synthesize --> critic_check : draft answer
    synthesize --> persist_aborted : LLM unavailable

    critic_check --> persist_done : passed
    critic_check --> synthesize : retry once
    critic_check --> persist_warning : still failing

    persist_done --> [*]
    persist_warning --> [*]
    persist_aborted --> [*]
```

## Failure Handling

```mermaid
flowchart LR
    PSM_FAIL[PSM unavailable] --> DEGRADED[Continue in degraded mode]
    RAG_FAIL[RAG unavailable] --> DEGRADED
    GRAPH_FAIL[Graph artifacts unavailable] --> DEGRADED
    ALL_FAIL[All 3 sources failed] --> ABORT[abort: no_evidence]
    CRITIC_FAIL[Critic still failing after retry] --> REVIEW[Mark for human review]
    SQLITE_FAIL[SQLite unavailable] --> SKIP[Cooldown skip + persist skip]

    DEGRADED --> REVIEW
    REVIEW --> PERSIST[persist_warning<br/>requires_human_review = true]
    SKIP --> LOG[Результат только в Loguru<br/>requires_human_review = true]
```
