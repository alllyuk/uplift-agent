# C4 Container: Uplift Agent

Внутренняя структура системы: контейнеры, хранилища и внешние зависимости.

```mermaid
flowchart TB
    analyst["👤 Аналитик"]

    subgraph uplift ["Uplift Agent"]
        ui["Web UI\nStreamlit"]
        agent["Agent Service\nLangGraph"]
        clientdata[("Client Data\nCSV / Parquet")]
        retrieval[("Retrieval Store\nFAISS / Parquet")]
        cases[("Case Store\nSQLite")]
    end

    openai["☁️ OpenAI API"]
    langsmith["☁️ LangSmith"]

    analyst -- "HTTP/Browser" --> ui
    ui -- "In-process" --> agent
    agent -- "File I/O" --> clientdata
    agent -- "Local retrieval" --> retrieval
    agent -- "SQLite" --> cases
    agent -- "HTTPS" --> openai
    agent -- "HTTPS" --> langsmith

    style ui fill:#438DD5,color:#fff
    style agent fill:#438DD5,color:#fff
    style clientdata fill:#438DD5,color:#fff
    style retrieval fill:#438DD5,color:#fff
    style cases fill:#438DD5,color:#fff
    style openai fill:#999,color:#fff
    style langsmith fill:#999,color:#fff
```

## Потоки данных

| От | К | Данные | Протокол |
|----|---|--------|----------|
| Web UI | Agent Service | Запрос пользователя и итоговый результат | In-process (planned: REST API, см. `specs/rest-api.md`) |
| Agent Service | Client Data | `client_id` → профиль клиента | File I/O |
| Agent Service | Retrieval Store | `query` → top-k чанки | Local retrieval |
| Agent Service | Case Store | Чтение cooldown-history и сохранение кейса | SQLite |
| Agent Service | OpenAI API | LLM request/response | HTTPS |
| Agent Service | LangSmith | Trace spans и метрики | HTTPS |
