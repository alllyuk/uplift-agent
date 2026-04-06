# C4 Context: Uplift Agent

Диаграмма верхнего уровня — система, пользователь, внешние сервисы и границы.

```mermaid
C4Context
    title Контекстная диаграмма — Uplift Agent

    Person(analyst, "Банковский аналитик", "Запускает кейсы и читает рекомендации.")

    System(uplift, "Uplift Agent", "PoC-система поддержки решений<br/>по клиентским интервенциям.")

    System_Ext(openai, "OpenAI API", "LLM для парсинга запросов<br/>и синтеза ответа.")
    System_Ext(langsmith, "LangSmith", "Трейсы и мониторинг выполнения.")

    Rel(analyst, uplift, "Запускает анализ и получает отчёт", "Streamlit UI / CLI")
    Rel(uplift, openai, "Вызовы LLM", "HTTPS")
    Rel(uplift, langsmith, "Отправляет трейсы", "HTTPS")
```

## Описание участников

| Участник | Тип | Описание |
|----------|-----|----------|
| Банковский аналитик | Person | Пользователь PoC, который запускает кейсы и читает рекомендации. |
| Uplift Agent | System | Ядро системы: orchestration, tool calls, guardrails и финальный отчёт. |
| OpenAI API | External System | Внешний LLM-провайдер для разбора запроса и синтеза ответа. |
| LangSmith | External System | Трейсинг и наблюдаемость выполнения пайплайна. |

> **Примечание:** Локальные данные и хранилища (CSV клиентов, retrieval-артефакты, SQLite) намеренно опущены на context-уровне и раскрываются в [C4 Container](c4-container.md).
