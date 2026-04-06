# C4 Context: Uplift Agent

Диаграмма верхнего уровня — система, пользователь, внешние сервисы.

```mermaid
flowchart TB
    analyst["👤 Банковский аналитик"]

    uplift["🟦 Uplift Agent\nPoC-система поддержки решений\nпо клиентским интервенциям"]

    openai["☁️ OpenAI API"]
    langsmith["☁️ LangSmith"]

    analyst -- "Streamlit UI / CLI" --> uplift
    uplift -- "HTTPS" --> openai
    uplift -- "HTTPS" --> langsmith

    style uplift fill:#438DD5,color:#fff
    style openai fill:#999,color:#fff
    style langsmith fill:#999,color:#fff
```

## Участники

| Участник | Тип | Описание |
|----------|-----|----------|
| Банковский аналитик | Person | Пользователь PoC. Запускает кейсы и читает рекомендации. |
| Uplift Agent | System | Ядро: orchestration, tool calls, guardrails, финальный отчёт. Включает локальные данные (CSV клиентов, FAISS-индекс, SQLite). |
| OpenAI API | External | LLM для парсинга запросов и синтеза ответа. |
| LangSmith | External | Трейсинг и наблюдаемость выполнения пайплайна. |

> Локальные хранилища (CSV, FAISS, SQLite) — часть системы. Детализация — в [C4 Container](c4-container.md).
