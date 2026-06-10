from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
from loguru import logger  # type: ignore

from sme_causal.core.config import get_config

import hashlib
import os
import json
import math
import re
import uuid
import numpy as np
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

cfg = get_config()


class RAG:
    def __init__(
        self,
        cfg=get_config(),
        model_name: str = "intfloat/multilingual-e5-small",
        chunk_target: Tuple[int, int] = (1000, 1500),
        chunk_overlap: int = 120,
    ):
        """
        chunk_target: (min_chars, max_chars) – целевые размеры чанка в символах
        chunk_overlap: перекрытие между чанками в символах
        """
        self.cfg = cfg or get_config()
        self.model_name = model_name
        self.chunk_min, self.chunk_max = chunk_target
        self.chunk_overlap = chunk_overlap

        self._model = None

    # ===================== Публичные методы =====================

    def build_chunks(
        self, use_metadata: bool = True, write_to_disk: bool = True,
    ) -> pd.DataFrame:
        """
        Читает все .txt из cfg.document_corpus_dir, режет по абзацам в ~1000–1500 символов
        с overlap ~120, при use_metadata=True — добавляет префикс "title / doc_id" в текст чанка,
        и при write_to_disk=True сохраняет parquet в cfg.chunks_path.
        Возвращает DataFrame: [chunk_id(int64), doc_id(str), text(str)].
        """
        corpus_dir = self.cfg.document_corpus_dir
        assert corpus_dir.exists(), f"Корпус не найден: {corpus_dir}"

        meta = None
        if use_metadata:
            try:
                meta = self._load_metadata(self.cfg.documents_metadata_path)
            except Exception as e:
                print(
                    f"[RAG] WARN: metadata load failed: {e}. Continue without metadata."
                )
                meta = None

        rows = []

        for txt_path in sorted(corpus_dir.glob("*.txt")):
            doc_id, title = self._infer_doc_meta(txt_path, meta)
            raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
            norm_text = self._normalize_newlines(raw_text)

            # собираем чанки целевого размера
            chunks = self._assemble_chunks_from_paragraphs(norm_text)

            for i, chunk in enumerate(chunks):
                # перекрытие на уровне символов
                # (реализовано в _assemble_chunks_from_paragraphs)
                text = chunk
                if use_metadata and (title or doc_id):
                    prefix = f"[TITLE] {title} | [DOC_ID] {doc_id}\n\n"
                    text = prefix + text
                rows.append(
                    {
                        "chunk_id": np.int64(self._stable_chunk_id(doc_id, i)),
                        "doc_id": str(doc_id),
                        "text": text,
                    }
                )

        df = pd.DataFrame(rows, columns=["chunk_id", "doc_id", "text"])
        df["chunk_id"] = df["chunk_id"].astype("int64")

        if write_to_disk:
            self._ensure_parent_dir(self.cfg.chunks_path)
            df.to_parquet(self.cfg.chunks_path, index=False)
        return df

    def build_embeddings(self, chunks: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Строит эмбеддинги (768d) для чанков через multilingual-e5-base.
        Берет df из аргумента или из cfg.chunks_path. Если его нет, сперва build_chunks().
        Сохраняет parquet в cfg.embeddings_path со столбцами:
          - chunk_id (int64)
          - embedding (list[float], float32, длина 384)
        Возвращает тот же df.
        """
        if chunks is None:
            if not Path(self.cfg.chunks_path).exists():
                chunks = self.build_chunks()
            else:
                chunks = pd.read_parquet(self.cfg.chunks_path)

        model = self._load_model()

        # E5: добавляем префикс "passage: "
        texts = [f"passage: {t}" for t in chunks["text"].tolist()]
        emb = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True if len(texts) >= 64 else False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # для cosine (IP) в FAISS
        ).astype("float32")

        # Укладываем в DataFrame
        emb_df = pd.DataFrame(
            {"chunk_id": chunks["chunk_id"].astype("int64"), "embedding": list(emb)}
        )

        self._ensure_parent_dir(self.cfg.embeddings_path)
        emb_df.to_parquet(self.cfg.embeddings_path, index=False)
        return emb_df

    def build_faiss_index(self, embeddings: Optional[pd.DataFrame] = None) -> None:
        """
        Строит FAISS IndexFlatIP (cosine через нормализацию) и сохраняет в cfg.faiss_index_path.
        Если embeddings не переданы — берет из cfg.embeddings_path, иначе вызывает build_embeddings().
        """
        assert faiss is not None, "faiss не установлен. Установи: pip install faiss-cpu"

        if embeddings is None:
            if Path(self.cfg.embeddings_path).exists():
                embeddings = pd.read_parquet(self.cfg.embeddings_path)
            else:
                embeddings = self.build_embeddings()

        # Приводим к np.array [N, D]
        embs = np.stack(embeddings["embedding"].to_numpy(), axis=0).astype("float32")
        # На всякий случай нормализуем (encode выше уже нормализует)
        faiss.normalize_L2(embs)

        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        self._ensure_parent_dir(self.cfg.faiss_index_path)
        faiss.write_index(index, str(self.cfg.faiss_index_path))

    def perform_query(self, query: str, top_k: int = 5) -> List[str]:
        """
        Кодирует запрос той же моделью (E5, префикс "query: "), ищет top_k релевантных чанков
        через сохранённый FAISS-индекс, и возвращает список текстов чанков.
        """
        assert Path(
            self.cfg.faiss_index_path
        ).exists(), "FAISS индекс не найден. Сначала запусти build_faiss_index()."
        assert Path(
            self.cfg.embeddings_path
        ).exists(), "embeddings.parquet не найден. Сначала запусти build_embeddings()."
        assert Path(
            self.cfg.chunks_path
        ).exists(), "chunks.parquet не найден. Сначала запусти build_chunks()."

        # Грузим индекс и данные для сопоставления
        index = faiss.read_index(str(self.cfg.faiss_index_path))
        emb_df = pd.read_parquet(
            self.cfg.embeddings_path
        )  # порядок соответствует индексу
        chunks_df = pd.read_parquet(self.cfg.chunks_path)

        # Кодируем запрос
        model = self._load_model()
        q = model.encode(
            [f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        # Поиск
        sims, idxs = index.search(q, top_k)
        idxs = idxs[0].tolist()

        # Находим chunk_id по порядковому индексу, затем достаём тексты
        sel_chunk_ids = emb_df.iloc[idxs]["chunk_id"].tolist()
        # Быстрый join
        sel = chunks_df.set_index("chunk_id").loc[sel_chunk_ids]
        logger.info(f"[RAG] Retrieved {len(sel)} chunks for query: {query}...")
        previews = []
        for chunk_id, text in sel["text"].items():
            doc_match = re.search(r"\[DOC_ID\]\s*([^\s]+)", str(text))
            previews.append(
                {
                    "chunk_id": int(chunk_id),
                    "doc_id": doc_match.group(1) if doc_match else None,
                    "preview": re.sub(r"\s+", " ", str(text)).strip()[:200],
                }
            )
        logger.info(f"[RAG] Retrieved chunks: {previews}")
        return sel["text"].tolist()

    def run_rag_pipeline(self, use_metadata: bool = True, force_rebuild: bool = False) -> None:
        """
        Полный прогон: build_chunks(use_metadata) -> build_embeddings() -> build_faiss_index()
        Пропускает пересборку, если все артефакты актуальны (корпус не менялся).
        force_rebuild=True принудительно пересобирает всё.
        """
        if not force_rebuild and self._rag_cache_is_fresh():
            logger.info("RAG cache is up-to-date, skipping rebuild")
            return

        logger.info("Start building chunks")
        chunks = self.build_chunks(use_metadata=use_metadata)
        logger.info("start building embedds")
        emb = self.build_embeddings(chunks)
        logger.info("start building index")
        self.build_faiss_index(emb)
        logger.info("finished creating RAG")

    def _rag_cache_is_fresh(self) -> bool:
        """
        Проверяет, что все три RAG-артефакта существуют, консистентны между собой
        и корпус документов не менялся с момента последней сборки.
        """
        chunks_path = Path(self.cfg.chunks_path)
        embeddings_path = Path(self.cfg.embeddings_path)
        index_path = Path(self.cfg.faiss_index_path)

        if not all(p.exists() for p in (chunks_path, embeddings_path, index_path)):
            return False

        chunks_mtime = chunks_path.stat().st_mtime
        embeddings_mtime = embeddings_path.stat().st_mtime

        # chunks пересобраны без embeddings — неконсистентное состояние
        if chunks_mtime > embeddings_mtime:
            logger.warning("RAG cache inconsistent: chunks newer than embeddings, will rebuild")
            return False

        corpus_dir = self.cfg.document_corpus_dir
        if not corpus_dir.exists():
            return False

        for f in corpus_dir.rglob("*"):
            if f.is_file() and f.stat().st_mtime > chunks_mtime:
                return False

        return True

    # ===================== Внутренние утилиты =====================

    def _load_model(self):  # -> SentenceTransformer
        assert (
            SentenceTransformer is not None
        ), "sentence-transformers не установлен. Установи: pip install sentence-transformers"
        if self._model is None:
            # Авто-выбор устройства
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load_metadata(self, csv_path: Optional[Path]) -> Optional[pd.DataFrame]:
        """
        Надёжная загрузка metadata.csv:
        - пробуем несколько кодировок (utf-8, utf-8-sig, cp1251, latin1)
        - авто-детект разделителя (engine='python')
        - чистим неразрывные пробелы (NBSP) и хвосты
        Ожидаемые колонки (как минимум одна из пар):
        - doc_id (str), title (str), filename (str, опционально)
        """
        if not csv_path or not csv_path.exists():
            print(f"[RAG] metadata not found at: {csv_path}")
            return None

        encodings = ["utf-8", "utf-8-sig", "cp1251", "latin1"]
        last_err = None
        for enc in encodings:
            try:
                df = pd.read_csv(
                    csv_path,
                    encoding=enc,
                    engine="python",  # лучше переносит «грязные» CSV и авто-определяет sep
                    sep=None,  # let pandas sniff delimiter (comma/semicolon/tab)
                )
                # нормализуем строки: заменяем NBSP на обычный пробел и трим
                df = df.apply(
                    lambda col: col.map(
                        lambda x: (
                            x.replace("\u00a0", " ").strip()
                            if isinstance(x, str)
                            else x
                        )
                    )
                )
                # нормализуем имена колонок
                df.columns = [c.replace("\u00a0", " ").strip() for c in df.columns]
                return df
            except Exception as e:
                last_err = e
                continue

        print(
            f"[RAG] WARN: failed to read metadata '{csv_path}' with tried encodings {encodings}: {last_err}"
        )
        return None

    def _infer_doc_meta(
        self, txt_path: Path, meta: Optional[pd.DataFrame]
    ) -> Tuple[str, str]:
        """
        Возвращает (doc_id, title).
        1) doc_id извлекаем из имени файла по шаблону doc_XX (doc-XX тоже ок).
        Если не нашли — используем stem целиком.
        2) Если есть метадата и в ней есть строка с таким doc_id — title берём из неё
        (колонки 'title' / 'Title' и т.п.); иначе title = красивый stem файла.
        """
        stem = txt_path.stem
        # doc_id только из имени файла (или весь stem, если не нашли шаблон)
        doc_id = self._extract_doc_id_from_name(stem) or stem

        # title по умолчанию: из имени файла
        fallback_title = stem.replace("_", " ").strip()
        title = fallback_title

        if meta is not None and not meta.empty:
            # нормализуем имена колонок к lower()
            lowmap = {c.lower(): c for c in meta.columns}
            if "doc_id" in lowmap:
                # сравниваем как строки
                m = meta[meta[lowmap["doc_id"]].astype(str) == str(doc_id)]
                if not m.empty:
                    row = m.iloc[0]
                    # пробуем вытащить title из разных вариантов колонок
                    for cand in ("title", "document_title", "name"):
                        if (
                            cand in lowmap
                            and pd.notna(row[lowmap[cand]])
                            and str(row[lowmap[cand]]).strip()
                        ):
                            title = str(row[lowmap[cand]]).strip()
                            break
                    # если подходящей колонки с заголовком нет — останется fallback_title

        return str(doc_id), str(title)

    @staticmethod
    def _extract_doc_id_from_name(stem: str) -> Optional[str]:
        # поддержим doc_12, doc-12, DOC12
        m = re.search(r"doc[_\-]?(\d+)", stem, re.IGNORECASE)
        return f"doc_{m.group(1)}" if m else None

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        # нормализуем \r\n, убираем лишние пробельные хвосты строк
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        t = re.sub(r"[ \t]+\n", "\n", t)
        # убираем экзотические управляющие символы
        t = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", t)
        return t.strip()

    def _assemble_chunks_from_paragraphs(self, text: str) -> List[str]:
        """
        Разбивает объединённый текст на чанки с помощью RecursiveCharacterTextSplitter.
        Использует гибкое разбиение по абзацам, предложениям и пробелам.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_max,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " "],
        )

        chunks = splitter.split_text(text)

        # удалим совсем короткие чанки (меньше chunk_min символов)
        chunks = [c.strip() for c in chunks if len(c.strip()) >= self.chunk_min]

        return chunks

    @staticmethod
    def _stable_chunk_id(doc_id: str, local_idx: int) -> int:
        """
        Детерминированный int64 по (doc_id, local_idx).
        Использует SHA-256 вместо hash() для стабильности между запусками Python.
        """
        seed = f"{doc_id}::{local_idx}"
        h = hashlib.sha256(seed.encode()).hexdigest()
        return np.int64(int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF)

    @staticmethod
    def _ensure_parent_dir(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)


def clean_txt(text: str) -> str:
    """
    Очищает текст от мусорных символов и делает его пригодным для RAG-пайплайна.

    Удаляет:
    - неалфавитные и нечисловые символы (кроме базовых знаков пунктуации);
    - повторяющиеся пробелы;
    - избыточные пустые строки.
    """
    cleaned_text = text
    cleaned_text = re.sub(r"[^\S\r\n]+", " ", cleaned_text)  # убрать лишние пробелы
    cleaned_text = re.sub(
        r"[^\x00-\x7Fа-яА-ЯёЁ0-9,.;:!?«»“”\"'()\[\]{}\-–—%/\\\r\n ]+", "", cleaned_text
    )  # удалить мусорные символы
    cleaned_text = re.sub(
        r"\n{3,}", "\n\n", cleaned_text
    )  # убрать лишние пустые строки
    cleaned_text = re.sub(r"(\s)+", r"\1", cleaned_text)  # убрать дублирующиеся пробелы
    return cleaned_text.strip()
