"""
LLM-based causal edge inference from tabular data (LangChain + OpenAI).

This module provides functionality to infer causal relationships between variables
using Large Language Models. It leverages feature descriptions and domain knowledge
to suggest causal edges while respecting DAG constraints.
"""

# llm_graph.py
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from sme_causal.core.config import get_config
from sme_causal.core.utils import extract_edges_from_text
from sme_causal.core.constants import LAYER_INDEX, ALLOWED_VARS
from sme_causal.core.llm import invoke_with_fallback
from sme_causal.core.columns import CLIENT_ID

SYSTEM_PROMPT = """\
Вы — ассистент по причинно-следственному анализу. Вам даны описание признаков
(деловые SME клиенты банка, интервенции, итоговые метрики) и бизнес-контекст задачи.

Нужно предложить направления рёбер причинно-следственного графа между доступными переменными.
Требования:
- Используйте ТОЛЬКО перечисленные переменные.
- Соблюдайте причинный порядок слоёв: Macro → Relationship → Transactional → Interventions → Outcomes.
- Рёбра «назад» по слоям запрещены. Рёбра между переменными одного слоя допустимы только если это правдоподобно и без циклов (предпочтительно избегать).
- Каждое ребро: {source, target, relation="causal", polarity in ["+","-","non-monotonic"], confidence in [0,1], rationale}.
- Опираться на предметные знания, избегайте циклов и явных логических противоречий.
- Выведите от 12 до 30 рёбер. Без циклов. Без дубликатов.
Отвечайте строго в JSON: {"edges": [...]}.
"""

EDGE_SCHEMA_EXAMPLE = {
    "edges": [
        {
            "source": "Avg_Monthly_Inflow",
            "target": "Avg_Monthly_Outflow",
            "relation": "causal",
            "polarity": "+",
            "confidence": 0.78,
            "rationale": "Рост выручки требует роста закупок и затрат.",
        }
    ]
}


def strip_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df without non-feature identifier columns.

    Currently removes `Client_ID` if present.
    """
    cols = [c for c in [CLIENT_ID] if c in df.columns]
    return df.drop(columns=cols) if cols else df


def _apply_layer_constraint(edges: List[Dict]) -> List[Dict]:
    """Apply DAG layer constraints to filter invalid causal edges.

    Args:
        edges: List of edge dictionaries with 'source' and 'target' keys.

    Returns:
        Filtered list of edges that respect the causal layer ordering.
    """
    out = []
    seen = set()
    for e in edges:
        src, dst = e.get("source"), e.get("target")
        if src not in ALLOWED_VARS or dst not in ALLOWED_VARS:
            continue
        if LAYER_INDEX[src] > LAYER_INDEX[dst]:
            continue  # запрещаем «назад»
        key = (src, dst)
        if key in seen or src == dst:
            continue
        seen.add(key)
        out.append(e)
    return out


def infer_edges_with_llm(
    df: pd.DataFrame,
    field_docs_ru: Dict[str, str],
    model_name: str | None = None,
    temperature: float | None = None,
    bootstrap_rounds: int | None = None,
    sample_rows: int | None = None,
) -> List[Dict]:
    """Infer causal edges using LLM based on feature descriptions and configuration.

    Args:
        df: Input DataFrame with SME client data.
        field_docs_ru: Dictionary mapping column names to their Russian descriptions.
        model_name: OpenAI model name to use for inference.
        temperature: Sampling temperature for LLM (0.0 to 1.0).
        bootstrap_rounds: Number of LLM calls to perform for consensus.
        sample_rows: Maximum number of rows to sample from DataFrame.

    Returns:
        List of inferred causal edges with confidence scores and rationales.
    """
    # Load configuration and set defaults
    cfg = get_config()

    # Use configuration defaults if not provided
    model_name = model_name or cfg.llm.model_name
    temperature = temperature if temperature is not None else cfg.llm.temperature
    bootstrap_rounds = (
        bootstrap_rounds if bootstrap_rounds is not None else cfg.llm.bootstrap_rounds
    )
    sample_rows = sample_rows if sample_rows is not None else cfg.llm.sample_rows

    logger.info("Starting LLM-based causal edge inference")
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(
        f"LLM config: model={model_name}, temp={temperature}, rounds={bootstrap_rounds}"
    )

    # Сэмпл таблицы для экономии токенов
    if len(df) > sample_rows:
        df_in = df.sample(sample_rows, random_state=17)
        logger.info(f"Sampled {sample_rows} rows from {len(df)} total for efficiency")
    else:
        df_in = df
        logger.info(f"Using full dataset with {len(df)} rows")

    var_list = [v for v in df_in.columns if v in ALLOWED_VARS]

    var_docs = "\n".join([f"- {v}: {field_docs_ru.get(v, '')}" for v in var_list])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            (
                "user",
                "Описание признаков:\n{var_docs}\n\n"
                "Ограничения по слоям (DAG): Macro → Relationship → Transactional → Interventions → Outcomes.\n"
                "Выведите JSON со списком рёбер (пример структуры ниже):\n{schema}\n",
            ),
        ]
    )

    schema_str = json.dumps(EDGE_SCHEMA_EXAMPLE, ensure_ascii=False)
    merged: Dict[Tuple[str, str], Dict] = {}

    for _ in range(bootstrap_rounds):
        msg = prompt.format_messages(
            system_prompt=SYSTEM_PROMPT,
            var_docs=var_docs,
            schema=schema_str,
        )
        text, _raw, used_json = invoke_with_fallback(
            msg,
            model=model_name,
            temperature=float(temperature),
            api_key=cfg.effective_openai_api_key or None,
            seed=None,
            top_p=None,
        )
        if not used_json:
            logger.debug("LLM fallback to non-JSON response_mode during edge inference")

        edges = extract_edges_from_text(text)

        edges = _apply_layer_constraint(edges)
        for e in edges:
            src, dst = e.get("source"), e.get("target")
            conf = float(e.get("confidence", 0.5))
            pol = e.get("polarity", "+")
            rat = e.get("rationale", "")
            key = (src, dst)
            if key not in merged:
                merged[key] = {
                    "source": src,
                    "target": dst,
                    "polarity": pol,
                    "confidence": conf,
                    "votes": 1,
                    "rationales": [rat],
                }
            else:
                m = merged[key]
                # усредняем confidence, копим рационали
                m["confidence"] = float(
                    (m["confidence"] * m["votes"] + conf) / (m["votes"] + 1)
                )
                # если знак расходится — пометим как non-monotonic
                if m["polarity"] != pol:
                    m["polarity"] = "non-monotonic"
                m["votes"] += 1
                if rat:
                    m["rationales"].append(rat)

    # порог по уверенности/голосам
    result = []
    for (src, dst), m in merged.items():
        if (
            m["confidence"] >= cfg.llm.confidence_threshold
            and m["votes"] >= cfg.llm.votes_threshold
        ):
            result.append(
                {
                    "source": src,
                    "target": dst,
                    "relation": "causal",
                    "polarity": m["polarity"],
                    "confidence": round(m["confidence"], 3),
                    "rationale": " | ".join(m["rationales"][-3:]),
                }
            )

    logger.success(f"LLM inference completed: {len(result)} causal edges found")
    logger.debug(
        f"Edge statistics: {len(merged)} total candidates, {len(result)} passed threshold"
    )
    return result
