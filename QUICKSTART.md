# Quickstart & Deployment

Как поднять Uplift Agent локально и на удалённом сервере — через Docker (рекомендуется) или через локальный Python-venv.

- [Что нужно иметь помимо репозитория](#что-нужно-иметь-помимо-репозитория)
- [Деплой через Docker](#деплой-через-docker) — end-to-end без локального venv
- [Локальный запуск через venv](#локальный-запуск-через-venv) — если Docker использовать не хочется
- [Artifacts](#artifacts) — где и что пишется
- [RAG data](#rag-data)
- [Конфигурация](#конфигурация)
- [Тесты](#тесты)

---

## Что нужно иметь помимо репозитория

В git исключены артефакты рантайма и секреты (`.gitignore`: `artifacts/`, `causal_outputs/`, `reports/`, `.env`, производные RAG-индексы). Исходный корпус документов (`rag_data/document_corpus/`) лежит в репозитории — RAG поднимается без внешних данных.

**Обязательно:**

- **`.env`** с `OPENAI_API_KEY` и `LLM_MODEL` — без этого не будет работать ни LLM-инференс, ни эмбеддинги запросов.

**Для RAG (одноразовая подготовка после клона):**

- **FAISS-индекс и эмбеддинги** — не в git, генерируются из корпуса одной командой (`python -m sme_causal.app.build_rag` или эквивалент внутри контейнера — см. Docker-раздел).
  Результат: `rag_data/chunks.parquet`, `embeddings.parquet`, `index.faiss`. Пересобирать нужно при смене модели эмбеддингов или обновлении корпуса.
- **HuggingFace-модель эмбеддингов** (`intfloat/multilingual-e5-small`) — скачивается автоматически при первом запросе, ~100–500 МБ. Для оффлайн-среды предзагрузите её в `~/.cache/huggingface` / примонтированный `hf_cache` volume.

**Синтетика и графовые артефакты** (`synthetic_clients.csv`, `ground_truth_edges.json`, графы) — генерируются кодом из сида и не требуют внешних данных.

---

## Деплой через Docker

В репозитории есть готовые `Dockerfile` и `docker-compose.yml`. Всё ниже работает без локального venv — нужен только установленный Docker и Compose v2.

### End-to-end через Docker (локально)

```bash
# 1. Секреты: скопировать шаблон и подставить реальный OPENAI_API_KEY.
cp .env.example .env
# открыть .env и заменить your-actual-api-key на свой ключ

# 2. Собрать образ (первая сборка 5–10 мин: torch + faiss + sentence-transformers).
docker compose build

# 3. Bootstrap подложки (синтетика + причинный граф + RAG-индекс). Одноразово.
#    При первом запуске скачает модель эмбеддингов ~100–500 МБ и построит FAISS-индекс —
#    ещё 3–5 минут "тишины" в логах, это нормально.
docker compose run --rm uplift-agent \
    python -m sme_causal.app.main --graph-method llm

# 4. Поднять UI в фоне
docker compose up -d
# → http://localhost:8501

# 5. Агентный кейс из CLI (внутри уже запущенного контейнера)
docker compose exec uplift-agent \
    python -m sme_causal.app.run --client-id "C000005" \
        --what-if "New_Product_Offer=1,New_Product_Offer_Type=acquiring"

# 6. Пересобрать только RAG-индекс (когда обновили корпус документов)
docker compose exec uplift-agent python -m sme_causal.app.build_rag

# 7. Логи / остановка
docker compose logs -f
docker compose down          # тома сохраняются; `down -v` удалит их вместе с данными
```

**Права на `rag_data/`:** это bind-mount, контейнер пишет под uid=1000. Если клонировали под uid≠1000 (root на VPS, часть Mac-setups) — один раз требуется запустить `chown -R 1000:1000 ./rag_data`. Named volumes (`artifacts`, `causal_outputs`, `reports`, `hf_cache`) Docker поднимает с правами из образа.

### Что монтируется

| Точка в контейнере  | Тип volume                        | Назначение                                              |
|---------------------|-----------------------------------|---------------------------------------------------------|
| `/app/rag_data`     | bind-mount в `./rag_data/`        | Корпус документов (в git) + FAISS/chunks/embeddings (генерируются) |
| `/app/artifacts`    | named volume `artifacts`          | SQLite `cases.db`, LLM-графы, edge-reports, логи        |
| `/app/causal_outputs` | named volume `causal_outputs`   | Артефакты алгоритмических графов                        |
| `/app/reports`      | named volume `reports`            | Evaluation-отчёты по графу                              |
| `/home/app/.cache/huggingface` | named volume `hf_cache` | Кэш HuggingFace-моделей (sentence-transformers)        |

Пути совпадают с `PathsConfig` в `sme_causal/core/config.py` — код пишет ровно туда.

Посмотреть содержимое named-тома можно двумя способами:

```bash
docker compose exec uplift-agent ls /app/artifacts           # заглянуть изнутри
docker cp uplift-agent:/app/artifacts ./artifacts_snapshot   # выгрузить на хост
```

### Удалённый сервер

Минимум для production-деплоя на одиночный VPS:

- **Ресурсы:** ≥ 2 ГБ RAM (sentence-transformers + torch + faiss). Лучше 4 ГБ.
- **Docker Compose v2**
- **`.env` на сервере**
- **Права на `rag_data/` при клоне под root:** контейнер пишет в `/app/rag_data` под uid=1000, а bind-mount отдаёт каталог с правами хоста. После клона под root — один раз:
  ```bash
  chown -R 1000:1000 ./rag_data
  ```
  Иначе первый запуск может упасть с `PermissionError: /app/rag_data/chunks.parquet`.
- **Bootstrap на сервере:** тот же `docker compose run --rm uplift-agent python -m sme_causal.app.main`. Без этого в `artifacts/` не будет графов, и кейсы с `--graph-method` упадут.
- **Бэкапы SQLite:**
  ```bash
  docker run --rm -v "$(pwd)/artifacts:/data" -v "$(pwd):/backup" alpine \
      tar czf /backup/artifacts-$(date +%F).tgz -C /data .
  ```
- **Секреты:** `OPENAI_API_KEY` и прочее — через `.env` или secret-менеджер хостера, в образ не встраивать.
- **TLS/прокси:** Streamlit отдаёт голый HTTP на 8501, без built-in auth. Дефолтный `docker-compose.yml` биндит порт на `0.0.0.0` (удобно для локальной разработки) — на remote-хосте перед первым запуском поправьте в `docker-compose.yml`:
  ```yaml
  ports:
    - "127.0.0.1:8501:8501"
  ```
  Затем перед ним — Caddy/nginx/traefik с TLS (Let's Encrypt или self-signed для демо) и basic auth, проксирующий на `127.0.0.1:8501`. Без этого Streamlit торчит в интернет без аутентификации.

---

## Локальный запуск через venv

Если Docker использовать не хочется.

1. Создать виртуальное окружение и установить зависимости.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Отдельно: для работы графов (`pydot`/`pyvis`) нужен системный `graphviz` (`sudo apt install graphviz` / `brew install graphviz`). В Docker-образе он уже установлен.

2. Создать `.env` в корне проекта (подхватывается автоматически через `python-dotenv`):

```bash
cp .env.example .env
# открыть .env и подставить реальный OPENAI_API_KEY
```

Дополнительные переменные (LLM/data/paths/logging) — см. defaults в `sme_causal/core/config.py`. `OPENAI_API_KEY` обязателен для любых LLM-вызовов.

3. Подготовить подложку (синтетические данные + причинный граф + RAG-индекс) — одноразово или при смене конфигурации:

```bash
python -m sme_causal.app.main --graph-method llm   # llm | algo | algo_llm | hybrid
```

Что делает `main.py`:
- генерирует `synthetic_clients.csv` и `ground_truth_edges.json` из сида;
- строит причинный граф выбранным методом (`llm` / `algo` / `algo_llm` / `hybrid`) и экспортирует его в JSON/GEXF/GraphML;
- считает evaluation-метрики графа против ground truth → пишет в `reports/`;
- собирает RAG-индекс из `rag_data/document_corpus/` (эквивалент отдельного `python -m sme_causal.app.build_rag`).

После этого в `artifacts/` и `causal_outputs/` (в корне репо) лежит всё, что нужно `run.py` для pipeline.

4. Запустить агентный pipeline из CLI по конкретному кейсу:

```bash
python -m sme_causal.app.run --client-id "C000005" --what-if "New_Product_Offer=1,New_Product_Offer_Type=acquiring"

python -m sme_causal.app.run --json --what-if "Credit_Limit_Change=15.0,Tariff_Discount=1"

python -m sme_causal.app.run --what-if "Credit_Limit_Change=25.0"
```

**Поток pipeline:** `intake → context → policy_check → estimation (PSM + RAG + Graph параллельно) → synthesize → critic (L1 rule-based + L2 LLM) → [retry при необходимости] → persist в SQLite`.

Все три источника доказательств (PSM, Graph, RAG) включены по умолчанию. Отключить отдельно — `--no-psm`, `--no-graph`, `--no-rag`. Если все запрошенные источники упали, кейс завершается как `no_evidence`. Итоговые кейсы пишутся в `artifacts/cases.db`.

**Cooldown:** если для того же клиента и типа интервенции уже есть завершённый кейс за последние 30 дней, повторный запуск блокируется статусом `policy_blocked`. Сброс: `sqlite3 artifacts/cases.db "DELETE FROM cases WHERE status='done'"`.

Запрос в свободной форме (естественный язык):

```bash
python -m sme_causal.app.run -q "Оцените эффект от предложения зарплатного проекта клиенту C000005"
python -m sme_causal.app.run --client-id "C000005" -q "Что если поднять кредитный лимит клиенту на 20%" --outcome-col "Avg_Monthly_Inflow"
```

5. Интерактивный UI через Streamlit:

```bash
streamlit run sme_causal/app/streamlit_app.py
```

`OPENAI_API_KEY` можно задать через `.env` или в сайдбаре UI.

---

## Artifacts

По умолчанию пишутся в `artifacts/` в корне репо (переопределяется через `PATHS_ARTIFACTS_DIR`). Ключевые выходы:

- `cases.db` — SQLite с историей всех завершённых кейсов (см. `docs/specs/memory-context.md`).
- `synthetic_clients.csv` — сгенерированный датасет.
- `ground_truth_edges.json` — эталонные рёбра DAG от генератора.
- `llm_edges.json`, `graph_consensus.json`, `hybrid_edges.json` — рёбра, выведенные LLM / алгоритмическими методами / гибридным подходом.
- `edge_report.csv` — сравнение рёбер с ground truth (для синтетики).
- `graph_merged.{json,gexf,graphml,html}` — экспортированные графы + интерактивный PyVis.
- `pipeline.log`, `streamlit.log` — структурированные логи (Loguru).

Алгоритмические графы отдельно пишутся в `causal_outputs/` (настраивается в `config.py`). Отчёты каждого запуска с evaluation — в `reports/`.

---

## RAG data

Под `rag_data/` в корне репо:

- `document_corpus/` — txt-корпус и `metadata.csv` (в git);
- `chunks.parquet` — чанки с `chunk_id`, `doc_id`, `text` (генерируется);
- `embeddings.parquet` — эмбеддинги и `chunk_id` (генерируется);
- `index.faiss` — бинарный FAISS-индекс (генерируется).

---

## Конфигурация

Все дефолты — в `sme_causal/core/config.py` (Pydantic Settings). Переопределяется через env / `.env`: `LLM_MODEL`, `LLM_TEMPERATURE`, `DATA_N_CLIENTS`, `PATHS_ARTIFACTS_DIR`, logging-опции и др. `OPENAI_API_KEY` — обязателен для LLM-вызовов.

---

## Тесты

```bash
pytest -q
```

Запускаются из локального venv (папка `tests/` исключена из Docker-образа через `.dockerignore`).
