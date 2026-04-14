# Спецификация: Retriever (Evidence Retrieval Layer)

## 1. Обзор

Семантический поиск релевантных документов из банковского корпуса для обогащения контекста рекомендации. Класс `RAG`: чанкинг → embedding → индексация → поиск.

## 2. Источники данных

**Корпус:** ~50 текстовых документов (русский + английский) в директории `rag_data/document_corpus/`. Тематика: банковский сектор, проектное финансирование, кредитные рынки, цифровой банкинг, поведение клиентов МСБ. Каждый документ имеет метаданные: doc_id, title, source.

**Pre-built артефакты:**

| Файл | Описание |
|------|----------|
| `rag_data/chunks.parquet` | Чанки с метаданными |
| `rag_data/embeddings.parquet` | Векторные представления (384-dim float32) |
| `rag_data/index.faiss` | FAISS-индекс |

## 3. Индекс

- **Embedding:** `intfloat/multilingual-e5-small` (384-dim). E5-протокол: `"passage: {text}"` при индексации, `"query: {text}"` при поиске.
- **Нормализация:** L2 → cosine similarity через inner product.
- **FAISS:** `IndexFlatIP` (exact search). Для ~сотен чанков brute-force за < 50мс.

## 4. Чанкинг

| Параметр | Значение |
|----------|----------|
| Стратегия | `RecursiveCharacterTextSplitter` (LangChain) |
| chunk_size / overlap / min | 1500 / 120 / 1000 символов |
| Разделители | `\n\n`, `\n`, `.`, `!`, `?`, ` ` |
| Метаданные | `[TITLE] {title} \| [DOC_ID] {doc_id}` — префикс чанка |
| Chunk ID | `hash(f"{doc_id}::{local_idx}")` |

## 5. Параметры поиска

| Контекст | top_k | Обоснование |
|----------|-------|-------------|
| What-if (в составе Pipeline) | 3 | Бюджет ~1500 tokens |
| Standalone RAG-запрос | 5 | Больше контекста без ограничений промпта |

Reranking не используется — при ~сотнях чанков FAISS similarity достаточна.

## 6. Построение запроса

1. **Structured input:** `create_query(delta)` формирует русскоязычный запрос из `intervention_delta`. Пример: `{"New_Product_Offer": 1, "New_Product_Offer_Type": "acquiring"}` → `"Предложение нового продукта: acquiring"`.
2. **NL-запрос:** raw-текст пользователя напрямую.

## 7. Контракт

```python
rag = RAG(cfg)  # загружает chunks, embeddings, FAISS index при инициализации
chunks: list[str] = rag.perform_query(query="...", top_k=3)
# Список текстовых чанков, отсортированных по similarity (desc)
```

## 8. Ограничения

- Макс. контекст: 3 чанка × ~1500 символов ≈ 1500 tokens (на одну итерацию; при rag_refine суммарно до 6 чанков с дедупликацией)
- RAG-контент — untrusted input, не может менять системные инструкции (governance.md §7)
- Статичный корпус: rebuild при изменении документов или embedding model

## 9. Adaptive RAG (rag_refine)

RAG может вызываться повторно через узел `rag_refine` (см. `agent-orchestrator.md` §3.7) с переформулированным LLM-запросом. Это часть гибридной агентности (ADR-8 в `system-design.md`):

- **Trigger:** critic fail на первой попытке synthesize
- **Источник новой формулировки:** LLM получает critic issues + историю запросов из `rag_query_history` и формулирует уточнённый query, не повторяющий предыдущие
- **Stop:** `rag_iterations >= 2` (1 initial + 1 refine)
- **Накопление:** новые чанки **append** к `rag_chunks` с дедупликацией по chunk_id, а не replace
