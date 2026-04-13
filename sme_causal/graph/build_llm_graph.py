"""
LLM-based causal graph construction module.
Can be run independently or imported for use in other modules.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import networkx as nx

from sme_causal.inference.llm_graph import infer_edges_with_llm, strip_id_columns
from sme_causal.data.synth_data import generate_sme_data, SynthConfig, FIELD_DOCS_RU
from sme_causal.graph.graph_utils import edges_to_digraph, export_graph
from sme_causal.core.config import get_config


def build_llm_graph(
    df: pd.DataFrame,
    model_name: str | None = None,
    temperature: float | None = None,
    bootstrap_rounds: int | None = None,
    sample_rows: int | None = None,
    output_dir: Optional[Path] = None,
    return_graph: bool = False,
) -> Tuple[List[Dict], Optional[nx.DiGraph]]:
    """
    Build causal graph using LLM inference.

    Args:
        df: Input DataFrame
        model_name: LLM model to use
        temperature: Sampling temperature
        bootstrap_rounds: Number of bootstrap rounds
        sample_rows: Maximum rows to sample
        output_dir: Directory to save results (None for default)
        return_graph: Whether to return graph object

    Returns:
        Tuple of (edges_list, graph_object)
    """
    cfg = get_config()

    if output_dir is None:
        output_dir = cfg.full_artifacts_dir

    output_dir.mkdir(exist_ok=True)

    # Run LLM inference
    llm_edges = infer_edges_with_llm(
        df=strip_id_columns(df),
        field_docs_ru=FIELD_DOCS_RU,
        model_name=model_name or cfg.llm.model_name,
        temperature=temperature or cfg.llm.temperature,
        bootstrap_rounds=bootstrap_rounds or cfg.llm.bootstrap_rounds,
        sample_rows=sample_rows or cfg.llm.sample_rows,
    )

    # Save edges
    edges_path = output_dir / "llm_edges.json"
    with open(edges_path, "w", encoding="utf-8") as f:
        json.dump(llm_edges, f, ensure_ascii=False, indent=2)

    # Build and export graph
    G = edges_to_digraph(llm_edges)
    export_graph(G, out_prefix=str(output_dir / "graph_llm"))

    # Save CSV summary
    edges_df = pd.DataFrame(
        [
            {
                "source": e["source"],
                "target": e["target"],
                "polarity": e.get("polarity", "+"),
                "confidence": e.get("confidence", 0.5),
                "rationale": e.get("rationale", ""),
            }
            for e in llm_edges
        ]
    )
    edges_df.to_csv(output_dir / "llm_edges.csv", index=False)

    if return_graph:
        return llm_edges, G
    return llm_edges, None


def main():
    """Standalone execution for LLM graph building"""
    SEED = 42
    cfg = SynthConfig(n_clients=3000, seed=SEED)
    df = generate_sme_data(cfg)

    build_llm_graph(df=df)
    print("LLM graph construction completed!")


if __name__ == "__main__":
    main()
