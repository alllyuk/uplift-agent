# C4 Component: Agent Service

Внутреннее устройство `Agent Service` — основные компоненты исполнения кейса и их связи.

```mermaid
C4Component
    title Компонентная диаграмма — Agent Service

    Container_Boundary(agent, "Agent Service") {
        Component(intake, "Intake & Router", "LangGraph node", "Разбирает вход и инициализирует кейс.")

        Component(context, "Context Loader", "LangGraph node", "Загружает профиль клиента.")

        Component(policy, "Policy Checker", "Rule-based node", "Проверяет eligibility и cooldown.")

        Component(candidates, "Candidate Generator", "LangGraph node", "Готовит кандидатов для recommend-режима.")

        Component(estimation, "Estimation Layer", "Tool coordinator", "Собирает effect и evidence:<br/>PSM, retrieval и graph lookup.")

        Component(synth, "Recommendation Synthesizer", "LLM node", "Формирует итоговый ответ.")

        Component(critic, "Critic / Guardrail", "Rule-based node", "Проверяет числа, источники<br/>и уровень уверенности.")

        Component(persist, "Case Persister", "Persistence node", "Сохраняет результат и финальный статус.")
    }

    Rel(intake, context, "client_id + mode")
    Rel(context, policy, "client context")
    Rel(policy, candidates, "recommend path")
    Rel(policy, estimation, "evaluate path")
    Rel(candidates, estimation, "candidate delta")
    Rel(estimation, synth, "effect + evidence")
    Rel(synth, critic, "draft answer")
    Rel(critic, synth, "retry with issues")
    Rel(critic, persist, "approved / degraded")
    Rel(policy, persist, "blocked case")

    UpdateRelStyle(critic, synth, $offsetX="40", $offsetY="-20")
    UpdateRelStyle(policy, persist, $offsetX="-20", $offsetY="20")
```

## Интерфейсы компонентов

| Компонент | Входные поля из CaseState | Выходные поля в CaseState |
|-----------|--------------------------|---------------------------|
| Intake & Router | `raw_query` или structured input | `case_id`, `mode`, `client_id`, `intervention_delta` |
| Context Loader | `client_id` | `client_context` |
| Policy Checker | `client_context`, `intervention_delta` | `policy_result` |
| Candidate Generator | `client_context`, `policy_result` | candidate deltas для recommend-режима |
| Estimation Layer | `client_context`, `intervention_delta` | `psm_result`, `rag_chunks`, `graph_dsl` |
| Recommendation Synthesizer | Контекст кейса и evidence | `explanation` |
| Critic / Guardrail | `explanation` и evidence | `critic_result`, `retry_count` |
| Case Persister | Полное состояние кейса | финальный `status`, `latency_ms`, запись в SQLite |

> `CaseState` используется как общий транспорт состояния между узлами, но не выделяется в отдельный компонент на диаграмме, чтобы не перегружать схему.
