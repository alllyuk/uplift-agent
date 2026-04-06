# C4 Container: Uplift Agent

Внутренняя структура системы: контейнеры (процессы, хранилища, UI) и их взаимодействие.

```mermaid
C4Container
    title Контейнерная диаграмма — Uplift Agent

    Person(analyst, "Банковский аналитик")

    System_Boundary(uplift, "Uplift Agent") {
        Container(ui, "Web UI", "Streamlit", "Запуск кейсов и просмотр результатов.")

        Container(agent, "Agent Service", "Python / LangGraph", "Оркестрация кейса, policy-check,<br/>tool calls, synthesis и guardrails.")

        ContainerDb(clientdata, "Client Data", "CSV / Parquet", "Профили клиентов и признаки<br/>для оценки интервенций.")

        ContainerDb(retrieval, "Retrieval Store", "FAISS / Parquet", "Индекс, embeddings и chunks<br/>банковского корпуса.")

        ContainerDb(cases, "Case Store", "SQLite", "Кейсы, статусы, audit<br/>и cooldown-history.")
    }

    System_Ext(openai, "OpenAI API")
    System_Ext(langsmith, "LangSmith")

    Rel(analyst, ui, "Запускает кейсы", "HTTP/Browser")
    Rel(ui, agent, "Передаёт запрос и получает результат", "In-process")
    Rel(agent, clientdata, "Читает профиль клиента", "File I/O")
    Rel(agent, retrieval, "Ищет evidence", "Local retrieval")
    Rel(agent, cases, "Читает и сохраняет кейсы", "SQLite")
    Rel(agent, openai, "Вызовы LLM", "HTTPS")
    Rel(agent, langsmith, "Трейсы", "HTTPS")
```

## Потоки данных между контейнерами

| От | К | Данные | Протокол |
|----|---|--------|----------|
| Web UI | Agent Service | Запрос пользователя и итоговый результат | In-process |
| Agent Service | Client Data | `client_id` → профиль клиента | File I/O |
| Agent Service | Retrieval Store | `query` → top-k чанки | Local retrieval |
| Agent Service | Case Store | Чтение cooldown-history и сохранение кейса | SQLite |
| Agent Service | OpenAI API | LLM request/response | HTTPS |
| Agent Service | LangSmith | Trace spans и метрики | HTTPS |
