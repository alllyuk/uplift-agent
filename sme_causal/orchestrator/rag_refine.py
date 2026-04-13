"""RAG Refine: reformulate RAG query based on critic issues and re-query.

Spec reference: docs/specs/agent-orchestrator.md §3.7.
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from sme_causal.orchestrator.state import CaseState

_MAX_RAG_ITERATIONS = 2

_REFINE_PROMPT = """\
Ты — помощник аналитика. Предыдущий RAG-запрос не дал достаточно контекста, и критик нашёл проблемы.

Интервенция: {delta}
Предыдущие RAG-запросы: {history}
Проблемы критика: {issues}

Сформулируй ОДИН уточнённый поисковый запрос (1–2 предложения на русском), чтобы найти информацию,
которая поможет устранить проблемы критика. Верни только текст запроса, без пояснений.
"""


def rag_refine(state: CaseState) -> None:
    """Mutate *state* in-place: reformulate RAG query and append new chunks.

    Guards:
    - rag_iterations >= MAX → no-op
    - LLM failure → no-op
    - RAG failure → no-op
    """
    if state.get("rag_iterations", 0) >= _MAX_RAG_ITERATIONS:
        logger.info("rag_refine: max iterations reached, skipping")
        return

    from sme_causal.core.llm import invoke_with_fallback
    from sme_causal.core.config import get_config

    cfg = get_config()
    critic_result = state.get("critic_result", {})
    issues = critic_result.get("issues", [])
    if not issues:
        return

    # Step 1: LLM formulates a refined query
    prompt = _REFINE_PROMPT.format(
        delta=state.get("intervention_delta", {}),
        history=state.get("rag_query_history", []),
        issues=issues,
    )
    messages = [
        {"role": "system", "content": "Ты формулируешь поисковые запросы для RAG-системы."},
        {"role": "user", "content": prompt},
    ]

    try:
        content, _, _ = invoke_with_fallback(
            messages,
            model=cfg.llm.model_name,
            temperature=0.1,
            api_key=cfg.effective_llm_api_key,
        )
        new_query = content.strip()
        if not new_query:
            logger.warning("rag_refine: LLM returned empty query")
            return
    except Exception:
        logger.warning("rag_refine: LLM failed, skipping")
        return

    # Step 2: Query RAG
    try:
        from sme_causal.rag.rag_pipeline import RAG

        rag = RAG(cfg)
        new_chunks = rag.perform_query(new_query, top_k=3)
    except Exception:
        logger.warning("rag_refine: RAG query failed, skipping")
        return

    # Step 3: Deduplicate and append
    existing = set(state.get("rag_chunks", []))
    added = 0
    for chunk in new_chunks:
        if chunk not in existing:
            state.setdefault("rag_chunks", []).append(chunk)
            existing.add(chunk)
            added += 1

    state["rag_iterations"] = state.get("rag_iterations", 0) + 1
    state.setdefault("rag_query_history", []).append(new_query)

    logger.info(
        "rag_refine: query='{}' added {} new chunks (total {})",
        new_query[:80],
        added,
        len(state.get("rag_chunks", [])),
    )
