"""Retrieval-quality evaluation for the RAG component (no manual labels).

Auto-eval methodology: an LLM produces a natural Russian question for each
sampled chunk; the source chunk is the gold passage. Two query styles are
generated per chunk:

- "literal"     — keeps the chunk's terminology (upper bound, easy).
- "paraphrase"  — formulated as an analyst's question without the chunk's
                  vocabulary (lower bound, hard; closer to real usage).

Metrics:
- Recall@k (chunk-level): the source chunk lies in top-k.
- Recall@k (doc-level):   any chunk from the source document lies in top-k.
- MRR (chunk-level):      mean reciprocal rank of the source chunk.

The script supports an ablation sweep across multiple embedding models.
LLM-generated queries are cached on disk, so adding a new model only
costs corpus re-encoding + dot-product retrieval.

Single-model run (uses the production E5-small encoder by default):
    python -m sme_causal.evaluation.rag_retrieval \\
        --n-chunks 30 --top-k 10 --seed 42

Sweep across embedding models:
    python -m sme_causal.evaluation.rag_retrieval \\
        --n-chunks 30 --top-k 10 --seed 42 \\
        --sweep-models "intfloat/multilingual-e5-small,intfloat/multilingual-e5-base,sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer

from sme_causal.core.config import get_config
from sme_causal.core.llm import invoke_with_fallback


_SYSTEM = (
    "Ты помогаешь оценивать качество поискового индекса по банковским и "
    "финансовым документам. Отвечай строго в формате, который попросили."
)


_QUERY_PROMPTS: Dict[str, str] = {
    "literal": (
        "Прочитай фрагмент и сформулируй ОДИН короткий русскоязычный вопрос, "
        "прямой ответ на который содержится в этом фрагменте. "
        "Используй ключевые термины из текста. Не упоминай слова "
        "«фрагмент», «документ», «согласно тексту». "
        "Верни ровно одну строку — сам вопрос, без кавычек и нумерации.\n\n"
        "ФРАГМЕНТ:\n{chunk}"
    ),
    "paraphrase": (
        "Прочитай фрагмент и сформулируй ОДИН короткий русскоязычный вопрос, "
        "который мог бы задать аналитик малого/среднего бизнеса, НЕ имея "
        "перед глазами этого фрагмента, но ответ на который содержится в нём. "
        "Перефразируй ключевые термины: не повторяй их дословно. "
        "Не упоминай документ. Один вопрос, одной строкой, без кавычек."
        "\n\nФРАГМЕНТ:\n{chunk}"
    ),
}


def _strip_metadata_prefix(text: str) -> str:
    """Drop the '[TITLE] ... | [DOC_ID] ...\\n\\n' header from the chunk."""
    if text.startswith("[TITLE]"):
        idx = text.find("\n\n")
        if idx > 0:
            return text[idx + 2 :]
    return text


def _clean_query(raw: str) -> str:
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    line = re.sub(r"^\s*[\d\.\)]+\s*", "", line)
    return line.strip().strip('«»"\'').strip()


def _generate_query(
    chunk_text: str, kind: str, *, model: str, api_key: str, seed: int
) -> str:
    user_prompt = _QUERY_PROMPTS[kind].format(
        chunk=_strip_metadata_prefix(chunk_text)[:3000]
    )
    msgs = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    text, _raw, _used_json = invoke_with_fallback(
        msgs, model=model, temperature=0.3, api_key=api_key, seed=seed,
    )
    return _clean_query(text)


def _safe_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", model_name)


def _corpus_fingerprint(chunks_df: pd.DataFrame) -> str:
    """Stable 12-hex digest over (chunk_id, text) pairs.

    Embedded in cache filenames so that any change to corpus contents,
    metadata, or splitter logic invalidates stale embeddings even when
    the chunk count happens to coincide.
    """
    s = chunks_df.sort_values("chunk_id")
    payload = "\n".join(
        f"{int(cid)}|{txt}" for cid, txt in zip(s["chunk_id"], s["text"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ChunkConfig:
    min_chars: int
    max_chars: int
    overlap: int

    @property
    def tag(self) -> str:
        return f"chunks-{self.min_chars}-{self.max_chars}-{self.overlap}"


_DEFAULT_CHUNK_CONFIG = ChunkConfig(1000, 1500, 120)


def build_chunks_for_config(cc: ChunkConfig) -> pd.DataFrame:
    """In-memory chunking with a custom config; does not touch the production cache."""
    from sme_causal.rag.rag_pipeline import RAG

    rag = RAG(
        chunk_target=(cc.min_chars, cc.max_chars), chunk_overlap=cc.overlap,
    )
    df = rag.build_chunks(use_metadata=True, write_to_disk=False)
    return df


def load_or_generate_queries(
    *,
    chunks_df: pd.DataFrame,
    chunk_config: ChunkConfig,
    corpus_fingerprint: str,
    n_chunks: int,
    seed: int,
    kinds: Sequence[str],
    cache_dir: Path,
    llm_model: str,
    api_key: str,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        cache_dir
        / f"queries_seed{seed}_n{n_chunks}_kinds-{'-'.join(kinds)}"
          f"_{chunk_config.tag}_fp{corpus_fingerprint}.parquet"
    )
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        logger.info(f"loaded {len(df)} cached queries from {cache_path}")
        return df

    sampled = chunks_df.sample(n=n_chunks, random_state=seed).reset_index(drop=True)
    rows: List[Dict] = []
    for i, row in sampled.iterrows():
        for kind in kinds:
            try:
                q = _generate_query(
                    row["text"], kind,
                    model=llm_model, api_key=api_key, seed=seed + int(i),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"query generation failed (chunk={row['chunk_id']}, kind={kind}): {e}"
                )
                continue
            if not q:
                continue
            rows.append({
                "chunk_id": int(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "kind": kind,
                "query": q,
            })
            logger.info(f"[{i+1}/{len(sampled)} | {kind}] q={q[:90]}")
    df = pd.DataFrame(rows)
    df.to_parquet(cache_path, index=False)
    logger.success(f"cached {len(df)} queries → {cache_path}")
    return df


def load_or_compute_corpus_embeddings(
    model_name: str,
    chunks_df: pd.DataFrame,
    cache_dir: Path,
    chunk_config: ChunkConfig,
    corpus_fingerprint: str,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (
        f"corpus_emb_{_safe_name(model_name)}_{chunk_config.tag}"
        f"_fp{corpus_fingerprint}.npy"
    )
    if path.exists():
        emb = np.load(path)
        if emb.shape[0] == len(chunks_df):
            logger.info(f"loaded cached corpus embeddings: {path} shape={emb.shape}")
            return emb
        logger.warning(f"cached embeddings size mismatch at {path}, re-encoding")

    logger.info(f"encoding {len(chunks_df)} chunks with {model_name}")
    encoder = SentenceTransformer(model_name)
    passages = [f"passage: {t}" for t in chunks_df["text"].tolist()]
    emb = encoder.encode(
        passages, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")
    np.save(path, emb)
    logger.success(f"saved corpus embeddings: {path} shape={emb.shape}")
    return emb


def evaluate_with_model(
    *,
    queries_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    model_name: str,
    top_k: int,
    cache_dir: Path,
    chunk_config: ChunkConfig,
    corpus_fingerprint: str,
) -> pd.DataFrame:
    chunk_emb = load_or_compute_corpus_embeddings(
        model_name, chunks_df, cache_dir, chunk_config, corpus_fingerprint,
    )
    encoder = SentenceTransformer(model_name)
    q_texts = [f"query: {q}" for q in queries_df["query"].tolist()]
    q_emb = encoder.encode(
        q_texts, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype("float32")
    sims = q_emb @ chunk_emb.T  # cosine, since both normalised
    top_idx = np.argsort(-sims, axis=1)[:, :top_k]

    chunk_ids = chunks_df["chunk_id"].to_numpy()
    cid2doc = dict(zip(chunks_df["chunk_id"], chunks_df["doc_id"]))

    ks = tuple(k for k in (1, 3, 5, 10) if k <= top_k)

    rows: List[Dict] = []
    for r in range(len(queries_df)):
        row = queries_df.iloc[r]
        ranked = chunk_ids[top_idx[r]].tolist()
        try:
            rank = ranked.index(int(row["chunk_id"])) + 1
        except ValueError:
            rank = 0
        doc_hits: Dict[str, int] = {}
        for k in ks:
            top_docs = {cid2doc[c] for c in ranked[:k]}
            doc_hits[f"doc_hit@{k}"] = int(row["doc_id"] in top_docs)
        rows.append({
            "embedding_model": model_name,
            "chunk_config": chunk_config.tag,
            "chunk_id": int(row["chunk_id"]),
            "doc_id": str(row["doc_id"]),
            "kind": str(row["kind"]),
            "query": str(row["query"]),
            "rank": int(rank),
            "top_sim": round(float(sims[r, top_idx[r, 0]]), 4),
            **doc_hits,
        })
    return pd.DataFrame(rows)


def aggregate_metrics(
    rows_df: pd.DataFrame, kinds: Sequence[str], ks: Sequence[int]
) -> pd.DataFrame:
    has_chunk_cfg = "chunk_config" in rows_df.columns
    out: List[Dict] = []
    if has_chunk_cfg:
        groups = rows_df.groupby(["embedding_model", "chunk_config"])
    else:
        groups = ((m, rows_df[rows_df["embedding_model"] == m]) for m in rows_df["embedding_model"].unique())
    for key, sub_m in groups:
        if has_chunk_cfg:
            model_name, cfg_tag = key
        else:
            model_name, cfg_tag = key, None
        for kind in list(kinds) + ["all"]:
            s = sub_m if kind == "all" else sub_m[sub_m["kind"] == kind]
            if s.empty:
                continue
            ranks = s["rank"].to_numpy()
            recall = {f"recall@{k}": float(((ranks > 0) & (ranks <= k)).mean()) for k in ks}
            doc_recall = {
                f"doc_recall@{k}": float(s[f"doc_hit@{k}"].mean())
                for k in ks if f"doc_hit@{k}" in s.columns
            }
            mrr = float(np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0).mean())
            row = {
                "embedding_model": model_name,
                "split": kind,
                "n": int(len(s)),
                "mrr": round(mrr, 4),
                **{k: round(v, 4) for k, v in recall.items()},
                **{k: round(v, 4) for k, v in doc_recall.items()},
            }
            if cfg_tag is not None:
                row = {"embedding_model": model_name, "chunk_config": cfg_tag, **{k: v for k, v in row.items() if k != "embedding_model"}}
            out.append(row)
    return pd.DataFrame(out)


def run(
    *,
    n_chunks: int,
    top_k: int,
    seed: int,
    kinds: Sequence[str],
    models: Sequence[str],
    chunk_configs: Sequence[ChunkConfig],
    out_dir: Path,
    cache_dir: Path,
) -> pd.DataFrame:
    cfg = get_config()

    parts = []
    queries_total = 0
    corpus_stats: List[Dict] = []
    for cc in chunk_configs:
        logger.info(f"== chunk config: {cc.tag} ==")
        if cc == _DEFAULT_CHUNK_CONFIG and Path(cfg.chunks_path).exists():
            chunks_df = pd.read_parquet(cfg.chunks_path)
            logger.info(
                f"using production chunks at {cfg.chunks_path}: {len(chunks_df)} chunks"
            )
        else:
            chunks_df = build_chunks_for_config(cc)
            logger.info(f"built in-memory chunks: {len(chunks_df)}")

        fp = _corpus_fingerprint(chunks_df)
        logger.info(f"corpus fingerprint for {cc.tag}: {fp}")

        # Sample diagnostics — recorded so the cross-config "non-paired"
        # comparison caveat in the writeup is auditable.
        sampled_for_log = chunks_df.sample(n=min(n_chunks, len(chunks_df)), random_state=seed)
        n_unique_docs_in_sample = int(sampled_for_log["doc_id"].nunique())

        corpus_stats.append({
            "chunk_config": cc.tag,
            "n_chunks": int(len(chunks_df)),
            "n_docs": int(chunks_df["doc_id"].nunique()),
            "avg_chunk_len": float(chunks_df["text"].str.len().mean()),
            "corpus_fingerprint": fp,
            "n_unique_docs_in_sample": n_unique_docs_in_sample,
        })

        queries_df = load_or_generate_queries(
            chunks_df=chunks_df,
            chunk_config=cc,
            corpus_fingerprint=fp,
            n_chunks=n_chunks, seed=seed, kinds=kinds, cache_dir=cache_dir,
            llm_model=cfg.llm.model_name, api_key=cfg.effective_openai_api_key,
        )
        queries_total += len(queries_df)

        for m in models:
            logger.info(f"== eval: model={m}, chunks={cc.tag} ==")
            parts.append(evaluate_with_model(
                queries_df=queries_df, chunks_df=chunks_df,
                model_name=m, top_k=top_k, cache_dir=cache_dir,
                chunk_config=cc, corpus_fingerprint=fp,
            ))
    full_df = pd.concat(parts, ignore_index=True)

    ks = tuple(k for k in (1, 3, 5, 10) if k <= top_k)
    summary_df = aggregate_metrics(full_df, kinds, ks)

    out_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_dir / "retrievals.csv", index=False, encoding="utf-8")
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")
    (out_dir / "details.json").write_text(
        json.dumps({
            "n_chunks": n_chunks,
            "kinds": list(kinds),
            "top_k": top_k,
            "ks_reported": list(ks),
            "seed": seed,
            "embedding_models": list(models),
            "chunk_configs": [{"min": c.min_chars, "max": c.max_chars, "overlap": c.overlap} for c in chunk_configs],
            "llm_model": cfg.llm.model_name,
            "corpus_stats_per_config": corpus_stats,
            "queries_total": int(queries_total),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.success(f"saved RAG retrieval eval to {out_dir}")
    print(summary_df.to_string(index=False))
    return summary_df


def _parse_chunk_configs(spec: str) -> List[ChunkConfig]:
    """Parse 'minXmaxXov,minXmaxXov,...' into a list of ChunkConfig."""
    out: List[ChunkConfig] = []
    for token in spec.split(","):
        parts = re.split(r"[xX*]", token.strip())
        if len(parts) != 3:
            raise SystemExit(
                f"chunk config must look like '500x800x60', got {token!r}"
            )
        out.append(ChunkConfig(int(parts[0]), int(parts[1]), int(parts[2])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval evaluation.")
    parser.add_argument("--n-chunks", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--query-kinds", type=str, default="literal,paraphrase",
        help="comma-separated; available: literal, paraphrase",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="intfloat/multilingual-e5-base",
        help="single embedding model (ignored when --sweep-models is given)",
    )
    parser.add_argument(
        "--sweep-models", type=str, default=None,
        help="comma-separated list of embedding models; overrides --embedding-model",
    )
    parser.add_argument(
        "--chunks-sweep", type=str, default=None,
        help="comma-separated list of chunk configs in 'minXmaxXoverlap' form, "
             "e.g. '500x800x60,1000x1500x120,1500x2500x180'; default uses (1000,1500,120)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_config()
    kinds = [k.strip() for k in args.query_kinds.split(",") if k.strip()]
    for k in kinds:
        if k not in _QUERY_PROMPTS:
            raise SystemExit(f"unknown query kind: {k}")

    models = (
        [m.strip() for m in args.sweep_models.split(",") if m.strip()]
        if args.sweep_models
        else [args.embedding_model]
    )
    chunk_configs = (
        _parse_chunk_configs(args.chunks_sweep)
        if args.chunks_sweep
        else [_DEFAULT_CHUNK_CONFIG]
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_dir or (cfg.full_artifacts_dir.parent / "reports" / f"rag_eval_{ts}")
    cache = args.cache_dir or (cfg.full_artifacts_dir.parent / "reports" / ".rag_eval_cache")
    run(
        n_chunks=args.n_chunks, top_k=args.top_k, seed=args.seed,
        kinds=kinds, models=models, chunk_configs=chunk_configs,
        out_dir=out, cache_dir=cache,
    )


if __name__ == "__main__":
    main()
