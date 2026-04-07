# C4 Component: Agent Service

Внутреннее устройство Agent Service — компоненты исполнения кейса.

```mermaid
flowchart TD
    subgraph agent ["Agent Service"]
        intake["Intake & Router"]
        context["Context Loader"]
        policy["Policy Checker"]
        estimation["Estimation Layer\nPSM + RAG + Graph"]
        synth["Intervention Synthesizer"]
        critic["Critic / Guardrail\nrule + LLM"]
        rag_refine["RAG Refiner\nLLM-driven"]
        persist["Case Persister"]
    end

    intake --> context
    context --> policy
    policy -- "allowed" --> estimation
    policy -- "blocked" --> persist
    estimation --> synth
    synth --> critic
    critic -- "pass" --> persist
    critic -- "fail, retry=0" --> rag_refine
    critic -- "fail, retry>=1" --> persist
    rag_refine --> synth

    style intake fill:#438DD5,color:#fff
    style context fill:#438DD5,color:#fff
    style policy fill:#438DD5,color:#fff
    style estimation fill:#438DD5,color:#fff
    style synth fill:#438DD5,color:#fff
    style critic fill:#438DD5,color:#fff
    style rag_refine fill:#438DD5,color:#fff
    style persist fill:#438DD5,color:#fff
```

## Интерфейсы компонентов

| Компонент | Входы из CaseState | Выходы в CaseState |
|-----------|-------------------|---------------------|
| Intake & Router | `raw_query` или structured input | `case_id`, `client_id`, `intervention_delta` |
| Context Loader | `client_id` | `client_context` |
| Policy Checker | `client_context`, `intervention_delta` | `policy_result` |
| Estimation Layer | `client_context`, `intervention_delta` | `psm_result`, `rag_chunks`, `graph_dsl` |
| Intervention Synthesizer | Контекст кейса и evidence | `explanation` |
| Critic / Guardrail | `explanation` и evidence | `critic_result` (rule_issues + llm_issues), `retry_count` |
| RAG Refiner | `critic_result.issues`, `rag_query_history` | Дополненный `rag_chunks`, `rag_iterations`, `rag_query_history` |
| Case Persister | Полное состояние кейса | `status`, `latency_ms`, запись в SQLite |

> `CaseState` — общий транспорт между узлами, не показан как отдельный компонент чтобы не перегружать схему.
