from __future__ import annotations

"""
Shared data loading and inference helpers used by CLI scripts.

This module centralizes small utilities that were duplicated between
entrypoints so they can be reused without importing Streamlit or other
UI-specific code.
"""

from pathlib import Path
from typing import Dict, List
import json
import sys

import pandas as pd
from loguru import logger

from sme_causal.core.config import get_config
from sme_causal.inference.llm_graph import infer_edges_with_llm, strip_id_columns
from sme_causal.data.synth_data import FIELD_DOCS_RU, SynthConfig, generate_sme_data


def ensure_dataset(csv_path: Path, n_clients: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Load dataset from CSV or generate and save a new one if missing.

    Args:
        csv_path: Path to the synthetic clients CSV file.
        n_clients: Number of clients to generate if file is missing.
        seed: Random seed for generation.

    Returns:
        DataFrame with SME client data.
    """
    if csv_path.exists():
        logger.info(f"Loading dataset from {csv_path}")
        return pd.read_csv(csv_path)

    logger.warning(f"Dataset not found at {csv_path}. Generating synthetic data...")
    df = generate_sme_data(SynthConfig(n_clients=n_clients, seed=seed))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.success(f"Synthetic dataset saved to {csv_path}")
    return df


def load_or_infer_edges(df: pd.DataFrame, edges_path: Path) -> List[Dict]:
    """Load edges from file, or infer via LLM if API key is present.

    Requires an API key to infer edges when no saved edges are available.

    Args:
        df: DataFrame with SME client data.
        edges_path: Path to JSON file with previously inferred edges.

    Returns:
        List of edge dictionaries.
    """
    if edges_path.exists():
        logger.info(f"Loading edges from {edges_path}")
        return json.loads(edges_path.read_text(encoding="utf-8"))

    # Accept API key from env or config settings
    if get_config().effective_openai_api_key:
        logger.info("OPENAI_API_KEY detected. Running LLM-based edge inference…")
        edges = infer_edges_with_llm(
            df=strip_id_columns(df),
            field_docs_ru=FIELD_DOCS_RU,
        )
        edges_path.write_text(json.dumps(edges, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.success(f"Saved {len(edges)} inferred edges to {edges_path}")
        return edges

    logger.error(
        "No saved edges found and no OpenAI API key configured.\n"
        f"Provide OPENAI_API_KEY or ensure that '{edges_path}' exists with inferred edges."
    )
    sys.exit(2)
