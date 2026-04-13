"""
Graph construction with LLM validation feedback loop.
Builds initial graph, validates with LLM, then rebuilds with constraints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate

from sme_causal.core.constants import LAYER_INDEX
from sme_causal.core.config import get_config
from sme_causal.graph.build_algo_graph import (
    prepare_dataset,
    build_knowledge,
    run_structure_learning,
    run_bootstrap,
    build_and_export_consensus,
    PipelineConfig,
    DatasetBundle,
    KnowledgeBundle,
    Edge,
    OUTPUT_DIR,
)

from sme_causal.core.llm import invoke_with_fallback

logger = logging.getLogger(__name__)


@dataclass
class LLMFeedback:
    """Container for LLM validation feedback"""

    impossible_edges: Set[Edge]
    missing_edges: Set[Edge]
    validation_rationale: str


VALIDATION_SYSTEM_PROMPT = """You are an expert in causal graph validation for SME banking data.
You understand business relationships and causal mechanisms in banking context.

Your task is to validate a causal graph by identifying:
1. IMPOSSIBLE edges - connections that violate domain logic or causal relationships, or strong unlikely to exist in real life
2. MISSING edges - IMPORTANT connections that should exist based on domain knowledge.

Consider temporal constraints: earlier layers cannot be caused by later ones.
Layer order: Macro → Relationship → Transactional → Interventions → Outcomes

Be conservative - only flag clear errors or important missing relationships."""

VALIDATION_USER_PROMPT = """Please validate this causal graph for SME banking clients:

Current Edges:
{edges_description}

Variables in dataset:
{variables_description}

Temporal layers:
- Macro: {macro_vars}
- Relationship: {relationship_vars}
- Transactional: {transactional_vars}
- Interventions: {interventions}
- Outcomes: {outcomes}

Return a JSON with:
{{
  "impossible_edges": [
    {{"source": "X", "target": "Y", "reason": "explanation"}}
  ],
  "missing_edges": [
    {{"source": "A", "target": "B", "confidence": 0.8, "reason": "explanation"}}
  ],
  "overall_assessment": "Brief summary of graph quality"
}}
"""


def prepare_graph_for_validation(
    graph: nx.DiGraph,
    bundle: DatasetBundle,
    knowledge: KnowledgeBundle,
) -> Dict:
    """Prepare graph information for LLM validation"""

    edges_info = []
    for u, v, data in graph.edges(data=True):
        edge_desc = {
            "source": u,
            "target": v,
            "confidence": data.get("mean_freq", 0.0),
            "robustness": data.get("robustness_label", "UNKNOWN"),
        }
        edges_info.append(edge_desc)

    layer_groups = {}
    for key, value in LAYER_INDEX.items():
        if value not in layer_groups:
            layer_groups[value] = []
        layer_groups[value].append(key)

    layer_vars = {
        "macro": layer_groups.get(0, []),
        "relationship": layer_groups.get(1, []),
        "transactional": layer_groups.get(2, []),
        "interventions": layer_groups.get(3, []),
        "outcomes": layer_groups.get(4, []),
    }

    return {
        "edges": edges_info,
        "variables": bundle.cols_for_dag,
        "layers": layer_vars,
    }


def validate_graph_with_llm(
    graph: nx.DiGraph,
    bundle: DatasetBundle,
    knowledge: KnowledgeBundle,
) -> LLMFeedback:
    """Send graph to LLM for validation and get feedback"""

    cfg = get_config()

    logger.info("Starting LLM validation of causal graph")

    graph_data = prepare_graph_for_validation(graph, bundle, knowledge)

    edges_desc = "\n".join(
        [
            f"- {e['source']} → {e['target']} (confidence: {e['confidence']:.2f}, robustness: {e['robustness']})"
            for e in graph_data["edges"]
        ]
    )

    vars_desc = "\n".join([f"- {var}" for var in graph_data["variables"]])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VALIDATION_SYSTEM_PROMPT),
            ("user", VALIDATION_USER_PROMPT),
        ]
    )

    message = prompt.format_messages(
        edges_description=edges_desc,
        variables_description=vars_desc,
        macro_vars=", ".join(graph_data["layers"]["macro"]),
        relationship_vars=", ".join(graph_data["layers"]["relationship"]),
        transactional_vars=", ".join(graph_data["layers"]["transactional"]),
        interventions=", ".join(graph_data["layers"]["interventions"]),
        outcomes=", ".join(graph_data["layers"]["outcomes"]),
    )

    response, _, _ = invoke_with_fallback(
        message,
        model=cfg.llm.model_name,
        temperature=cfg.llm.temperature,
        api_key=cfg.effective_openai_api_key or None,
    )

    feedback = parse_llm_validation_response(response)

    logger.info(
        f"LLM validation complete: {len(feedback.impossible_edges)} impossible edges, "
        f"{len(feedback.missing_edges)} missing edges suggested"
    )

    return feedback


def parse_llm_validation_response(response: str) -> LLMFeedback:
    """Parse LLM validation response into structured feedback"""
    cfg = get_config()
    try:
        if "{" in response and "}" in response:
            start = response.index("{")
            end = response.rindex("}") + 1
            json_str = response[start:end]
            data = json.loads(json_str)
        else:
            data = json.loads(response)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        return LLMFeedback(
            impossible_edges=set(),
            missing_edges=set(),
            validation_rationale="Failed to parse LLM response",
        )

    impossible = set()
    for edge_info in data.get("impossible_edges", []):
        if "source" in edge_info and "target" in edge_info:
            impossible.add((edge_info["source"], edge_info["target"]))

    missing = set()
    for edge_info in data.get("missing_edges", []):
        if "source" in edge_info and "target" in edge_info:
            # Only add if confidence is high enough
            if edge_info.get("confidence", 0) >= cfg.llm.confidence_threshold:
                missing.add((edge_info["source"], edge_info["target"]))

    return LLMFeedback(
        impossible_edges=impossible,
        missing_edges=missing,
        validation_rationale=data.get("overall_assessment", ""),
    )


def apply_llm_constraints_to_knowledge(
    knowledge: KnowledgeBundle,
    feedback: LLMFeedback,
    bundle: DatasetBundle,
) -> KnowledgeBundle:
    """Update knowledge bundle with LLM feedback constraints"""

    new_blacklist = set(knowledge.dir_blacklist)

    for edge in feedback.impossible_edges:
        if edge[0] in bundle.cols_for_dag and edge[1] in bundle.cols_for_dag:
            new_blacklist.add(edge)
            logger.debug(f"Adding impossible edge to blacklist: {edge}")

    return KnowledgeBundle(
        dir_blacklist=new_blacklist,
        temporal_tiers=knowledge.temporal_tiers,
        cols_mmhc=knowledge.cols_mmhc,
    )


def rebuild_graph_with_constraints(
    bundle: DatasetBundle,
    knowledge_updated: KnowledgeBundle,
    config: PipelineConfig,
    feedback: LLMFeedback,
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """Rebuild graph with LLM constraints applied"""

    logger.info("Rebuilding graph with LLM constraints")

    # If we have missing edges suggested, we can bias the bootstrap
    # by adjusting sampling or scoring parameters
    if feedback.missing_edges:
        # Increase bootstrap rounds for better coverage
        config.bootstrap.ges_runs = min(config.bootstrap.ges_runs * 2, 6)
        config.bootstrap.hc_runs = min(config.bootstrap.hc_runs * 2, 6)
        config.bootstrap.mmhc_runs = min(config.bootstrap.mmhc_runs * 2, 6)

    ges_boot, hc_boot, mmhc_boot = run_bootstrap(bundle, knowledge_updated, config)

    if feedback.missing_edges:
        ges_boot = boost_suggested_edges(
            ges_boot, feedback.missing_edges, boost_factor=0.6
        )
        hc_boot = boost_suggested_edges(
            hc_boot, feedback.missing_edges, boost_factor=0.6
        )
        mmhc_boot = boost_suggested_edges(
            mmhc_boot, feedback.missing_edges, boost_factor=0.6
        )

    consensus_full, consensus_subset, consensus_graph = build_and_export_consensus(
        bundle, knowledge_updated, config, ges_boot, hc_boot, mmhc_boot
    )

    return consensus_full, consensus_graph


def boost_suggested_edges(
    boot_df: pd.DataFrame,
    suggested_edges: Set[Edge],
    boost_factor: float = 0.6,
) -> pd.DataFrame:
    """Boost frequency of edges suggested by LLM"""

    df = boot_df.copy()

    for source, target in suggested_edges:
        # Check if edge exists in bootstrap results
        mask = (df["u"] == source) & (df["v"] == target)
        if mask.any():
            df.loc[mask, "freq"] = df.loc[mask, "freq"] + boost_factor
        else:
            new_row = pd.DataFrame(
                {
                    "u": [source],
                    "v": [target],
                    "freq": [boost_factor],
                }
            )
            df = pd.concat([df, new_row], ignore_index=True)

    # Ensure frequencies don't exceed 1.0
    df["freq"] = df["freq"].clip(upper=1.0)

    return df


def build_algo_llm_graph(
    csv_path: Path,
    output_dir: Optional[Path] = OUTPUT_DIR,
    max_iterations: int = 2,
) -> Tuple[List[Dict]]:
    """
    Main function for iterative graph refinement with LLM feedback.

    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save results
        max_iterations: Maximum number of refinement iterations

    Returns:
        List[Dict]: consensus_edges_list
    """

    config = PipelineConfig(csv_path=csv_path, output_dir=output_dir)
    cfg = get_config()
    output_dir.mkdir(exist_ok=True, parents=True)

    bundle = prepare_dataset(config)
    knowledge = build_knowledge(bundle)

    iteration_graphs = []
    best_graph = None
    best_consensus = None

    for iteration in range(max_iterations):
        logger.info(f"=== Iteration {iteration + 1}/{max_iterations} ===")

        if iteration == 0:
            logger.info("Building initial graph with algorithms")

            structure_results = run_structure_learning(bundle, knowledge, config)
            ges_boot, hc_boot, mmhc_boot = run_bootstrap(bundle, knowledge, config)

            consensus_full, consensus_subset, consensus_graph = (
                build_and_export_consensus(
                    bundle, knowledge, config, ges_boot, hc_boot, mmhc_boot
                )
            )
            best_graph = consensus_graph

        else:
            logger.info(f"Validating graph from iteration {iteration}")

            feedback = validate_graph_with_llm(
                best_graph,
                bundle,
                knowledge,
            )

            logger.info(f"LLM Feedback: {feedback.validation_rationale}")

            # If no issues found, we can stop early
            if not feedback.impossible_edges and not feedback.missing_edges:
                logger.info("LLM found no issues - stopping iteration")
                break

            knowledge = apply_llm_constraints_to_knowledge(knowledge, feedback, bundle)

            consensus_full, consensus_graph = rebuild_graph_with_constraints(
                bundle,
                knowledge,
                config,
                feedback,
            )

        best_graph = consensus_graph
        best_consensus = consensus_full
        iteration_graphs.append(consensus_graph)

    logger.info("Saving final refined graph")
    best_consensus.to_csv(cfg.full_algorithmic_dir / "algo_llm_edges.csv", index=False)

    consensus_list = best_consensus.to_dict("records")

    return consensus_list


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    cfg = get_config()
    artifacts_dir = cfg.full_artifacts_dir
    csv_path = artifacts_dir / "synthetic_clients.csv"

    logger.info("Starting iterative algo-graph refinement with LLM validation")

    final_edges = build_algo_llm_graph(
        csv_path=csv_path,
        output_dir=OUTPUT_DIR,
        max_iterations=3,
    )

    logger.info(f"Final graph has {len(final_edges)} edges")
    logger.info("Iterative refinement completed!")


if __name__ == "__main__":
    main()
