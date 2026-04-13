"""
Main orchestrator script for the SME causal graph inference pipeline.

This script coordinates the entire workflow:
1. Generate synthetic SME client data
2. Use LLM to infer causal relationships from data
3. Build and export causal graphs
4. Compare LLM results against ground truth for validation
"""

# main.py
from __future__ import annotations

import json
import sys
from typing import Dict, List
import argparse
import pandas as pd

from loguru import logger  # type: ignore

# Load environment variables from .env if present
from sme_causal.core import env  # noqa: F401

from sme_causal.core.config import get_config
from sme_causal.rag.rag_pipeline import RAG
from sme_causal.inference.llm_graph import infer_edges_with_llm, strip_id_columns
from sme_causal.data.synth_data import (
    FIELD_DOCS_RU,
    SynthConfig,
    generate_sme_data,
    ground_truth_edges,
)
from sme_causal.core.utils import configure_logging
from sme_causal.graph.build_algo_graph import build_algo_graph
from sme_causal.graph.build_hybrid_graph import build_hybrid_graph
from sme_causal.graph.build_algo_llm_graph import build_algo_llm_graph
from sme_causal.graph.graph_utils import (
    edges_to_digraph,
    export_graph,
    create_algo_edges,
)
from sme_causal.graph.evaluate_graphs import (
    evaluate_graph,
    create_report_directory,
    create_edge_comparison_table,
    save_metrics_and_edges,
)


# Load configuration
cfg = get_config()

# Configure logging
configure_logging(
    cfg.pipeline_log_path,
    cfg.logging,
    add_stdout=True,
    stdout_format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


def main() -> None:
    """Execute the complete SME causal graph inference pipeline.

    Orchestrates the full workflow from synthetic data generation through
    LLM-based causal inference to graph construction and validation.
    """
    parser = argparse.ArgumentParser(description="SME Causal Pipeline")
    parser.add_argument(
        "--graph-method",
        choices=["llm", "algo", "algo_llm", "hybrid"],
        default="llm",
        help="Graph construction method",
    )

    parser.add_argument("--min-conf", type=float, default=0.6)

    args = parser.parse_args()

    logger.info(
        f"Starting SME causal graph inference pipeline with method: {args.graph_method}"
    )

    # 1) Generate synthetic SME data
    logger.info("Step 1: Generating synthetic SME client data")
    synth_config = SynthConfig(
        n_clients=cfg.data_generation.n_clients, seed=cfg.data_generation.seed
    )
    logger.debug(
        f"Configuration: {synth_config.n_clients} clients, seed={synth_config.seed}"
    )

    df = generate_sme_data(synth_config)
    df.to_csv(cfg.synthetic_clients_path, index=False, encoding="utf-8")
    logger.success(f"Saved dataset → {cfg.synthetic_clients_path} (rows={len(df)})")

    # 2) Generate ground truth edges
    logger.info("Step 2: Generating ground truth causal edges")
    gt: List[Dict] = ground_truth_edges()
    with open(cfg.ground_truth_edges_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    logger.success(
        f"Saved ground truth → {cfg.ground_truth_edges_path} (edges={len(gt)})"
    )

    # 3) LLM / Algo inference of causal edges
    # Ensure API key is available before attempting LLM inference
    # Accept either env var or value loaded via pydantic settings
    if args.graph_method in ["llm", "hybrid", "algo_llm"]:
        if not cfg.effective_llm_api_key:
            logger.error(
                "No OpenAI API key found. Set OPENAI_API_KEY in your environment or .env file before running."
            )
            sys.exit(2)

    if args.graph_method == "hybrid":
        logger.info("Step 3: Running hybrid (LLM + Algorithmic) causal edge inference")

        hybrid_edges = build_hybrid_graph(
            df=df,
            use_llm_judge=True,
            confidence_threshold=args.min_conf,
        )

        hybrid_path = cfg.full_artifacts_dir / "hybrid_edges.json"
        with open(hybrid_path, "w", encoding="utf-8") as f:
            json.dump(hybrid_edges, f, ensure_ascii=False, indent=2)
        logger.success(
            f"Saved Hybrid edges → {hybrid_path} (edges={len(hybrid_edges)})"
        )
        edges = hybrid_edges

    elif args.graph_method == "llm":
        logger.info("Step 3: Running LLM-based causal edge inference")
        logger.debug(
            f"LLM config: model={cfg.llm.model_name}, temp={cfg.llm.temperature}, "
            f"rounds={cfg.llm.bootstrap_rounds}, samples={cfg.llm.sample_rows}"
        )

        llm_edges: List[Dict] = infer_edges_with_llm(
            df=strip_id_columns(df),  # ID не нужен
            field_docs_ru=FIELD_DOCS_RU,
            model_name=cfg.llm.model_name,
            temperature=cfg.llm.temperature,
            bootstrap_rounds=cfg.llm.bootstrap_rounds,
            sample_rows=cfg.llm.sample_rows,
        )
        with open(cfg.llm_edges_path, "w", encoding="utf-8") as f:
            json.dump(llm_edges, f, ensure_ascii=False, indent=2)
        logger.success(
            f"Saved LLM edges → {cfg.llm_edges_path} (edges={len(llm_edges)})"
        )
        edges = llm_edges

    elif args.graph_method == "algo":
        logger.info("Step 3: Running algorithmic causal inference")
        consensus, _ = build_algo_graph(csv_path=cfg.synthetic_clients_path)

        with open(cfg.algo_edges_path, "w", encoding="utf-8") as f:
            json.dump(consensus, f, ensure_ascii=False, indent=2)
        logger.success(
            f"Saved Algo edges → {cfg.algo_edges_path} (edges={len(consensus)})"
        )
        edges = create_algo_edges(consensus)

    elif args.graph_method == "algo_llm":
        logger.info(
            "Step 3: Running Algorithmic with LLM-based validation causal edge inference"
        )
        algo_llm_edges = build_algo_llm_graph(csv_path=cfg.synthetic_clients_path)

        algo_llm_path = cfg.full_algorithmic_dir / "algo_llm_edges.json"
        with open(algo_llm_path, "w", encoding="utf-8") as f:
            json.dump(algo_llm_edges, f, ensure_ascii=False, indent=2)
        logger.success(
            f"Saved Algo-LLM edges → {algo_llm_path} (edges={len(algo_llm_edges)})"
        )
        edges = create_algo_edges(algo_llm_edges)

    # 4) Build and export causal graph
    logger.info("Step 4: Building and exporting causal graph")
    G = edges_to_digraph(edges)
    if args.graph_method == "llm":
        dir = cfg.full_artifacts_dir
    elif args.graph_method == "algo":
        dir = cfg.full_algorithmic_dir
    elif args.graph_method == "hybrid":
        dir = cfg.full_artifacts_dir  # or create a separate dir
    elif args.graph_method == "algo_llm":
        dir = cfg.full_algorithmic_dir  # or create a separate dir
    export_graph(G, out_prefix=str(dir / cfg.paths.graph_prefix))
    logger.success(
        f"Exported graph → {dir / cfg.paths.graph_prefix}.(json|gexf|graphml) "
        f"(|E|={G.number_of_edges()})"
    )

    # 5) Validate against ground truth (for synthetic data/prompt adequacy control)
    logger.info("Step 5: Validating against ground truth")

    report_dir = create_report_directory(
        args.graph_method, bootstrap=False
    )  # single run without bootstrap evaluation

    metrics = evaluate_graph(edges, gt)
    edges_table = create_edge_comparison_table(edges, gt)
    save_metrics_and_edges(metrics, edges_table, report_dir)
    logger.success(f"Saved evaluation report → {report_dir}")

    # 6) Optional: Build or refresh RAG index
    logger.info("Step 6: Building or refreshing RAG index for document corpus")

    rag = RAG()
    rag.run_rag_pipeline(use_metadata=True)
    logger.success("RAG index built successfully!")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
