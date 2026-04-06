# Data Flow: Uplift Agent

Как данные проходят через систему, что хранится, что логируется.

## Основной поток данных

```mermaid
flowchart TD
    subgraph INPUT ["Input"]
        USER["Пользователь"] --> UI["Web UI / CLI"]
        UI --> INTAKE["Intake & Router"]
    end

    subgraph SOURCES ["Local Data Sources"]
        CLIENTS[("Client Data<br/>CSV / Parquet")]
        RETRIEVAL[("Retrieval Store<br/>FAISS + chunks + embeddings")]
        GRAPH[("Graph Artifacts<br/>JSON / DSL")]
        CASES[("Case Store<br/>SQLite")]
    end

    subgraph PROCESS ["Case Processing"]
        CONTEXT["Context Loader"]
        POLICY["Policy Check"]
        EVIDENCE["Estimation Layer"]
        SYNTH["Recommendation Synthesizer"]
        CRITIC["Critic / Guardrail"]
        RESULT["Final Result"]
    end

    subgraph EXTERNAL ["External Services"]
        OPENAI["OpenAI API"]
        LANGSMITH["LangSmith"]
        LOGS["Loguru Logs"]
    end

    INTAKE --> CONTEXT
    CONTEXT --> POLICY
    POLICY -->|allowed case| EVIDENCE
    POLICY -->|blocked / abstain| RESULT

    CONTEXT -->|client_id| CLIENTS
    CLIENTS -->|client context| CONTEXT

    POLICY -->|cooldown lookup| CASES

    CLIENTS -->|population data| EVIDENCE
    RETRIEVAL -->|top-k chunks| EVIDENCE
    GRAPH -->|graph DSL| EVIDENCE

    EVIDENCE -->|effect + evidence bundle| SYNTH
    SYNTH -->|messages| OPENAI
    OPENAI -->|draft answer| SYNTH
    SYNTH --> CRITIC
    CRITIC -->|retry once| SYNTH
    CRITIC -->|approved / degraded| RESULT

    RESULT --> UI
    UI --> USER
    RESULT -->|persist case| CASES

    INTAKE -.-> LOGS
    POLICY -.-> LOGS
    EVIDENCE -.-> LOGS
    CRITIC -.-> LOGS
    RESULT -.-> LOGS

    INTAKE -.-> LANGSMITH
    SYNTH -.-> LANGSMITH
    CRITIC -.-> LANGSMITH
    RESULT -.-> LANGSMITH
```

## Формат данных на каждом этапе

| Этап | Данные | Формат | Размер |
|------|--------|--------|--------|
| Input | `{client_id, mode, intervention_delta}` или NL-запрос | JSON / text | ~0.1-0.2 KB |
| Client Context | Профиль клиента и признаки | Dict -> JSON | ~1 KB |
| Policy Result | `{blocked, reasons, notes}` | Dict | ~0.2 KB |
| Evidence Bundle | `psm_result + rag_chunks + graph_dsl` | Mixed | ~6-7 KB |
| LLM Prompt | System + user messages | List[Message] | ~5K tokens |
| Explanation | Структурированный ответ и raw_text | Dict -> JSON | ~2 KB |
| Case Record | Запрос, контекст, результат, статус | SQLite row | ~5-10 KB |

## Что НЕ сохраняется и НЕ логируется

| Данные | Причина |
|--------|---------|
| Полные промпты с данными клиентов | PII, governance.md §2 |
| Полное содержимое RAG-документов | Объём, нерелевантно для audit |
| Сырые Open identifiers клиентов | Замена на псевдонимы, governance.md §3 |
| Matched DataFrame из PSM | Объём; сохраняются только агрегаты (ATE, ATT, n_pairs) |
