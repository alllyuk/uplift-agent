# Спецификация: Serving / Config

## 1. Startup Sequence

```
1. Загрузка .env → os.environ (python-dotenv)
2. Инициализация AppConfig (Pydantic BaseSettings)
3. Инициализация LLM client singleton
4. Загрузка FAISS index + embeddings + chunks для RAG
5. (Опционально) Pre-load graph DSL
6. Инициализация SQLite (CREATE TABLE IF NOT EXISTS)
7. Запуск Streamlit / CLI
```

**Проверки при старте:** наличие `LLM_API_KEY`; существование `artifacts/`; наличие RAG-артефактов (опционально); наличие CSV с данными клиентов (опционально — можно сгенерировать).

## 2. Система конфигурации

### 2.1 AppConfig (Pydantic)

```python
class AppConfig(BaseSettings):
    paths: PathsConfig
    data_generation: DataGenerationConfig
    llm: LLMConfig
    api: APIConfig
    logging: LoggingConfig
    streamlit: StreamlitConfig
    hybrid_graph: HybridGraphConfig
```

### 2.2 Sub-configs

| Config | Ключевые параметры |
|--------|--------------------|
| **PathsConfig** | `project_root`, `artifacts_dir`, `rag_data_dir`, имена файлов (CSV, JSON, Parquet, FAISS) |
| **DataGenerationConfig** | `n_clients` (3000), `seed` (42) |
| **LLMConfig** | `model_name`, `temperature`, `confidence_threshold` (0.45), `bootstrap_rounds`, `sample_rows` |
| **APIConfig** | `api_key`, `base_url`, `provider` (openai/local) |
| **LoggingConfig** | `level`, `file_rotation` (10MB), `file_retention` (90 days) |
| **StreamlitConfig** | `theme`, `page_title`, `page_icon` |

### 2.3 Приоритет

1. Environment variables (высший)
2. `.env` файл
3. Default values в Pydantic models

## 3. Секреты

| Секрет | Переменная | Хранение |
|--------|-----------|----------|
| OpenAI API Key | `LLM_API_KEY` | `.env` (gitignored) или env var |
| LangSmith API Key | `LANGCHAIN_API_KEY` | `.env` или env var |

Доступ: `os.getenv("LLM_API_KEY")` с fallback на `api.api_key` из Pydantic. `.env.example` — шаблон без реальных значений. Секреты не логируются.

## 4. Версии моделей

| Компонент | Настройка |
|-----------|-----------|
| LLM | `LLM_MODEL_NAME`, `LLM_PROVIDER` (openai/local), `LLM_BASE_URL` (для local) |
| Embedding | `intfloat/multilingual-e5-small` (384-dim), фиксирован в конфигурации. Смена → полный rebuild RAG-артефактов |

## 5. Entry points

| Режим | Описание |
|-------|----------|
| **Web UI** | Streamlit-приложение — интерактивный интерфейс |
| **CLI What-If** | Единичный what-if анализ: client_id + delta + флаги PSM/Graph |
| **CLI NL-Query** | Анализ по NL-запросу |
| **Full Pipeline** | Генерация данных → граф → RAG → eval |
| **RAG Build** | Создание chunks, embeddings, FAISS |
| **Eval** | RAGAS и Pollux оценка качества |

## 6. Деплой

Docker: `python:3.11-slim`, порт 8501 (Streamlit), volume `artifacts/`. CI/CD: Docker build → push → deploy.
