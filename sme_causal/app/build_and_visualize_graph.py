"""
Example: build, save, and visualize a causal graph from Python.

This script reuses existing project utilities to:
- load/generate data
- load existing inferred edges or infer via LLM when API key is available
- build a NetworkX DiGraph
- export it to JSON/GEXF/GraphML
- create an interactive HTML visualization (PyVis)

Run:
    python build_and_visualize_graph.py

Optional env vars to override defaults (see config):
- LLM_MODEL, LLM_TEMPERATURE, LLM_BOOTSTRAP_ROUNDS, LLM_SAMPLE_ROWS
- OPENAI_API_KEY (required to infer edges if no saved edges exist)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

# Load environment variables from .env if present
from sme_causal.core import env  # noqa: F401
from sme_causal.core.config import AppConfig, get_config
from sme_causal.core.data_io import ensure_dataset, load_or_infer_edges
from sme_causal.core.utils import configure_logging
from sme_causal.graph.graph_utils import edges_to_digraph, export_graph
from sme_causal.graph.graph_viz import build_pyvis_html


def main() -> None:
    """Build, save, and visualize a graph from SME data using project utilities."""
    parser = argparse.ArgumentParser(
        description="Build, save, and visualize causal graph"
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=None,
        help="Minimum confidence to include an edge (defaults to config)",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=None,
        help="Output prefix for graph files (JSON/GEXF/GraphML/HTML). Defaults to config",
    )
    parser.add_argument(
        "--regen_data",
        action="store_true",
        help="Force regenerate synthetic data instead of loading existing CSV",
    )
    args = parser.parse_args()

    # Load configuration after parsing
    cfg: AppConfig = get_config()
    # Configure logging similar to other entrypoints
    configure_logging(
        cfg.pipeline_log_path,
        cfg.logging,
        add_stdout=True,
        stdout_format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    # use getattr to avoid static analysis issues with pydantic settings
    default_min_conf = float(getattr(cfg.llm, "confidence_threshold", 0.45))
    default_graph_prefix = str(
        getattr(cfg.paths, "graph_prefix", "graph_merged")
    )
    min_conf = (
        float(args.min_conf) if args.min_conf is not None else default_min_conf
    )
    out_prefix = (
        args.out_prefix
        if args.out_prefix is not None
        else str(cfg.full_artifacts_dir / default_graph_prefix)
    )

    # Prepare data
    if args.regen_data and cfg.synthetic_clients_path.exists():
        logger.info("--regen_data set: recreating synthetic dataset")
        cfg.synthetic_clients_path.unlink()
    df = ensure_dataset(cfg.synthetic_clients_path)

    # Load or infer edges
    edges = load_or_infer_edges(df, cfg.llm_edges_path)

    # Filter by confidence
    filtered = [e for e in edges if float(e.get("confidence", 0.0)) >= min_conf]
    logger.info(
        f"Confidence threshold {min_conf:.2f}: kept {len(filtered)} of {len(edges)} edges"
    )

    # Build graph and export artifacts
    G = edges_to_digraph(filtered)
    export_graph(G, out_prefix=out_prefix)
    logger.success(
        f"Graph exported: {out_prefix}.json | {out_prefix}.gexf | {out_prefix}.graphml"
    )

    # Create interactive HTML visualization
    html = build_pyvis_html(G, height_px=650, directed=True, physics=True)
    html_path = Path(f"{out_prefix}.html")
    html_path.write_text(html, encoding="utf-8")
    logger.success(f"Visualization saved to {html_path}")


if __name__ == "__main__":
    main()
