# Workflow: Pipeline State Machine

Пошаговое выполнение запроса через `Pipeline` orchestrator (`sme_causal/orchestrator/pipeline.py`). В v1 оркестратор — plain Python с ThreadPoolExecutor для параллельного estimation; готовые фреймворки графов состояний (LangGraph и т.п.) не используются — см. ADR-1 в `../system-design.md`.

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

    critic_check --> persist_done : passed (rule + LLM)
    critic_check --> rag_refine : fail, retry_count = 0
    critic_check --> persist_warning : fail, retry_count >= 1

    rag_refine --> synthesize : new RAG query (LLM-driven)

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
