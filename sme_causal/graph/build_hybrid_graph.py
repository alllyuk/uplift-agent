"""
Hybrid graph builder that combines LLM and Algorithmic approaches.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd
import networkx as nx
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
import re

from sme_causal.graph.build_llm_graph import build_llm_graph
from sme_causal.graph.build_algo_graph import build_algo_graph
from sme_causal.graph.graph_utils import create_algo_edges
from sme_causal.core.config import get_config
from sme_causal.core.llm import invoke_with_fallback
from sme_causal.core.constants import ALLOWED_VARS
from sme_causal.data.synth_data import FIELD_DOCS_RU


LLM_JUDGE_SYSTEM_PROMPT = """\
You are an expert in causal analysis for SME banking clients. Your task is to judge and combine
causal edges from two different sources: LLM-based inference and algorithmic/statistical analysis.

You have access to field descriptions that explain what each variable represents in the context
of SME (Small and Medium Enterprise) banking clients.

For each edge or edge pair, you need to:
1. Evaluate the plausibility of the causal relationship based on domain knowledge
2. Consider the confidence scores from both sources (LLM and algorithmic)
3. Decide on the final confidence score and whether to include the edge
4. Provide brief reasoning for your decision

Important considerations:
- Domain knowledge about business metrics, banking relationships, and SME operations
- Statistical relationships suggested by algorithmic methods (correlation, mutual information, etc.)
- Semantic/logical relationships identified by LLM analysis
- Potential confounders, mediators, or spurious correlations
- Business logic and real-world causal mechanisms

Remember:
- Higher confidence when both methods agree
- Consider the strength of domain rationale
- Be skeptical of purely correlational relationships without causal mechanism
- Account for the temporal and logical ordering of variables

Output strictly as JSON array with no additional text.
"""

JUDGE_BATCH_PROMPT = """\
Evaluate the following causal edges for SME banking clients. Both LLM and algorithmic
methods may have suggested each edge with different confidence scores.

Field Descriptions:
{field_descriptions}

Edges to evaluate:
{edges_description}

Provide your judgment for each edge as a JSON array. Each judgment should include:
- edge_id: the edge number
- decision: "include" or "exclude"
- confidence: float between 0.0 and 1.0
- reasoning: brief explanation of your decision

Example format:
[
  {{
    "edge_id": 0,
    "decision": "include",
    "confidence": 0.85,
    "reasoning": "Strong causal mechanism: revenue directly drives transaction volume"
  }},
  {{
    "edge_id": 1,
    "decision": "exclude",
    "confidence": 0.2,
    "reasoning": "Likely spurious correlation without direct causal pathway"
  }}
]

Output only the JSON array, no additional text.
"""


def _combine_with_llm_judge(
    llm_edges: List[Dict],
    algo_edges: List[Dict],
    confidence_threshold: float,
    field_docs: Dict[str, str] = None,
    model_name: str = None,
    temperature: float = 0.2,
    batch_size: int = 10,
) -> List[Dict]:
    """
    Combine edges using LLM as a judge to evaluate and merge suggestions.

    Args:
        llm_edges: Edges from LLM graph
        algo_edges: Edges from Algorithmic graph
        field_docs: Dictionary of field descriptions
        confidence_threshold: Minimum confidence for edge inclusion
        model_name: LLM model to use for judging
        temperature: Temperature for LLM sampling
        batch_size: Number of edges to process in one LLM call

    Returns:
        Combined list of edges judged by LLM
    """
    cfg = get_config()
    model_name = model_name or cfg.llm.model_name

    if field_docs is None:
        field_docs = FIELD_DOCS_RU

    edge_map = defaultdict(lambda: {"llm": None, "algo": None})

    for edge in llm_edges:
        key = (edge["source"], edge["target"], edge.get("polarity", "+"))
        edge_map[key]["llm"] = edge

    for edge in algo_edges:
        key = (edge["source"], edge["target"], edge.get("polarity", "+"))
        edge_map[key]["algo"] = edge

    combined_edges = []

    # Process edges in batches for efficiency
    edge_items = list(edge_map.items())
    total_batches = (len(edge_items) + batch_size - 1) // batch_size

    logger.info(
        f"Processing {len(edge_items)} edge candidates in {total_batches} batches"
    )

    for batch_idx in range(0, len(edge_items), batch_size):
        batch = edge_items[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        logger.debug(
            f"Processing batch {batch_num}/{total_batches} with {len(batch)} edges"
        )

        batch_results = _judge_edge_batch(
            batch,
            field_docs=field_docs,
            model_name=model_name,
            temperature=temperature,
            api_key=cfg.effective_openai_api_key,
        )

        for (source, target, polarity), judgment in batch_results:
            if judgment.get("decision") == "include":
                confidence = float(judgment.get("confidence", 0.5))
                if confidence >= confidence_threshold:
                    edge_data = edge_map[(source, target, polarity)]
                    sources = []
                    rationales = []

                    if edge_data["llm"]:
                        sources.append("llm")
                        if edge_data["llm"].get("rationale"):
                            rationales.append(f"LLM: {edge_data['llm']['rationale']}")

                    if edge_data["algo"]:
                        sources.append("algo")
                        if edge_data["algo"].get("rationale"):
                            rationales.append(f"Algo: {edge_data['algo']['rationale']}")

                    combined_edge = {
                        "source": source,
                        "target": target,
                        "polarity": polarity,
                        "confidence": min(confidence, 1.0),
                        "method": "hybrid_llm_judge",
                        "sources": sources,
                        "judge_reasoning": judgment.get("reasoning", ""),
                    }

                    # Include original rationales if available
                    if rationales:
                        combined_edge["original_rationales"] = " | ".join(rationales)

                    combined_edges.append(combined_edge)

    logger.success(
        f"LLM judge completed: {len(combined_edges)} edges selected from {len(edge_items)} candidates"
    )

    return combined_edges


def _judge_edge_batch(
    edge_batch: List[Tuple],
    field_docs: Dict[str, str],
    model_name: str,
    temperature: float,
    api_key: str = None,
) -> List[Tuple]:
    """
    Judge a batch of edges using LLM with field descriptions for context.

    Args:
        edge_batch: List of ((source, target, polarity), edge_data) tuples
        field_docs: Dictionary mapping field names to descriptions
        model_name: LLM model name
        temperature: Sampling temperature
        api_key: API key for LLM service

    Returns:
        List of ((source, target, polarity), judgment) tuples
    """
    # Prepare field descriptions for relevant variables
    relevant_vars = set()
    for (source, target, _), _ in edge_batch:
        relevant_vars.add(source)
        relevant_vars.add(target)

    # Filter to only allowed variables
    relevant_vars = {v for v in relevant_vars if v in ALLOWED_VARS}

    field_descriptions = []
    for var in sorted(relevant_vars):
        desc = field_docs.get(var, "No description available")
        field_descriptions.append(f"- {var}: {desc}")

    field_desc_text = "\n".join(field_descriptions)

    edges_description = []
    for idx, ((source, target, polarity), edge_data) in enumerate(edge_batch):
        source_desc = field_docs.get(source, "")[:100]  # Truncate for brevity
        target_desc = field_docs.get(target, "")[:100]

        desc_parts = [
            f"Edge {idx}:",
            f"  {source} → {target}",
            f"  Polarity: {polarity}",
            f"  Source field: {source_desc}",
            f"  Target field: {target_desc}",
        ]

        if edge_data["llm"]:
            llm_conf = edge_data["llm"].get("confidence", 0.5)
            llm_rationale = edge_data["llm"].get("rationale", "")
            desc_parts.append(f"  LLM analysis:")
            desc_parts.append(f"    - Confidence: {llm_conf:.3f}")
            if llm_rationale:
                desc_parts.append(f"    - Rationale: {llm_rationale[:150]}")

        if edge_data["algo"]:
            algo_conf = edge_data["algo"].get("confidence", 0.5)
            algo_rationale = edge_data["algo"].get("rationale", "")
            desc_parts.append(f"  Algorithmic analysis:")
            desc_parts.append(f"    - Confidence: {algo_conf:.3f}")
            if algo_rationale:
                desc_parts.append(f"    - Rationale: {algo_rationale}")

        # Add consensus information
        if edge_data["llm"] and edge_data["algo"]:
            desc_parts.append(f"  Note: Suggested by both methods")
        elif edge_data["llm"]:
            desc_parts.append(f"  Note: Suggested only by LLM")
        elif edge_data["algo"]:
            desc_parts.append(f"  Note: Suggested only by algorithmic method")

        edges_description.append("\n".join(desc_parts))

    if not edges_description:
        return []

    edges_text = "\n\n".join(edges_description)

    prompt = ChatPromptTemplate.from_messages(
        [("system", LLM_JUDGE_SYSTEM_PROMPT), ("user", JUDGE_BATCH_PROMPT)]
    )

    msg = prompt.format_messages(
        field_descriptions=field_desc_text, edges_description=edges_text
    )
    try:
        logger.debug(f"Calling LLM judge for batch of {len(edge_batch)} edges")

        response_text, _, _ = invoke_with_fallback(
            msg,
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

        judgments = _parse_judgments(response_text)

        if len(judgments) < len(edge_batch):
            logger.warning("Not found some judgments in response_text")
            logger.warning(f"Response text:\n{response_text}")

        # Map judgments back to edges
        results = []
        for idx, ((source, target, polarity), edge_data) in enumerate(edge_batch):
            judgment = judgments.get(idx, None)
            if judgment is None:
                # Fallback if no judgment provided
                judgment = _create_fallback_judgment(edge_data)
                logger.warning(
                    f"No judgment for edge {idx} ({source}→{target}), using fallback"
                )
            results.append(((source, target, polarity), judgment))

        logger.debug(f"Successfully judged {len(results)} edges")
        return results

    except Exception as e:
        logger.error(f"Error in LLM edge judgment: {e}")
        # Fallback to simple heuristics for this batch
        return _fallback_batch_judgment(edge_batch)


def _parse_judgments(response_text: str) -> Dict[int, Dict]:
    """
    Parse LLM judgments from response text.

    Args:
        response_text: Raw LLM response

    Returns:
        Dictionary mapping edge index to judgment
    """
    try:
        # Look for JSON array pattern
        json_match = re.search(r"\[\s*\{.*?\}\s*\]", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            # Clean up potential formatting issues
            json_str = re.sub(r"[\n\r\t]+", " ", json_str)
            judgments_list = json.loads(json_str)

            # Convert to dict by edge_id
            judgments = {}
            for item in judgments_list:
                edge_id = item.get("edge_id")
                if edge_id is not None:
                    judgments[edge_id] = {
                        "decision": item.get("decision", "exclude"),
                        "confidence": float(item.get("confidence", 0.5)),
                        "reasoning": item.get("reasoning", ""),
                    }

            logger.debug(f"Successfully parsed {len(judgments)} judgments")
            return judgments

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM judgments as JSON: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error parsing LLM judgments: {e}")

    return {}


def _create_fallback_judgment(edge_data: Dict) -> Dict:
    """
    Create a fallback judgment based on simple heuristics.

    Args:
        edge_data: Dictionary with 'llm' and 'algo' edge information

    Returns:
        Judgment dictionary
    """
    cfg = get_config()

    llm_edge = edge_data.get("llm")
    algo_edge = edge_data.get("algo")

    if llm_edge and algo_edge:
        # Both methods suggest the edge
        llm_conf = llm_edge.get("confidence", 0.5)
        algo_conf = algo_edge.get("confidence", 0.5)
        avg_conf = (llm_conf + algo_conf) / 2

        # Boost confidence if both agree strongly
        if llm_conf > cfg.llm.confidence_threshold and algo_conf > 0.5:
            final_conf = min(avg_conf * 1.1, 0.95)
        else:
            final_conf = avg_conf

        return {
            "decision": "include" if final_conf >= 0.5 else "exclude",
            "confidence": final_conf,
            "reasoning": "Both methods agree on this edge (fallback judgment)",
        }

    elif llm_edge:
        # Only LLM suggests
        conf = llm_edge.get("confidence", 0.5)
        return {
            "decision": (
                "include" if conf >= cfg.llm.confidence_threshold else "exclude"
            ),
            "confidence": conf * 0.9,  # Slightly reduce confidence
            "reasoning": "LLM-only suggestion (fallback judgment)",
        }

    elif algo_edge:
        # Only algorithmic suggests
        conf = algo_edge.get("confidence", 0.4)
        return {
            "decision": "include" if conf >= 0.5 else "exclude",
            "confidence": conf * 0.9,  # Slightly reduce confidence
            "reasoning": "Algorithm-only suggestion (fallback judgment)",
        }

    else:
        return {
            "decision": "exclude",
            "confidence": 0.0,
            "reasoning": "No evidence from either method",
        }


def _fallback_batch_judgment(edge_batch: List[Tuple]) -> List[Tuple]:
    """
    Fallback judgment for an entire batch when LLM fails.

    Args:
        edge_batch: List of ((source, target, polarity), edge_data) tuples

    Returns:
        List of ((source, target, polarity), judgment) tuples
    """
    results = []
    for (source, target, polarity), edge_data in edge_batch:
        judgment = _create_fallback_judgment(edge_data)
        results.append(((source, target, polarity), judgment))

    logger.info(f"Applied fallback judgment to {len(results)} edges")
    return results


def _combine_weighted_average(
    confidence_threshold: float,
    llm_edges: List[Dict],
    algo_edges: List[Dict],
    llm_weight: float = 0.5,
) -> List[Dict]:
    """Combine using weighted average of confidences."""
    edge_map = defaultdict(lambda: {"confidence": 0, "sources": []})

    for edge in llm_edges:
        key = (edge["source"], edge["target"], edge.get("polarity", "+"))
        confidence = float(edge.get("confidence", 0.5))
        edge_map[key]["confidence"] += confidence * llm_weight
        edge_map[key]["sources"].append("llm")
        edge_map[key]["polarity"] = edge.get("polarity", "+")

    for edge in algo_edges:
        key = (edge["source"], edge["target"], edge.get("polarity", "+"))
        confidence = float(edge.get("confidence", 0.5))
        algo_weight = 1 - llm_weight
        edge_map[key]["confidence"] += confidence * algo_weight
        edge_map[key]["sources"].append("algo")
        edge_map[key]["polarity"] = edge.get("polarity", "+")

    combined_edges = []
    for (source, target, polarity), data in edge_map.items():
        sources = list(set(data["sources"]))
        if len(sources) == 2:
            final_confidence = data["confidence"]  # Already weighted
        else:
            # Single source - adjust for missing weight
            if "llm" in data["sources"]:
                final_confidence = data["confidence"] / llm_weight
            else:
                final_confidence = data["confidence"] / (1 - llm_weight)

        if len(sources) == 2 or final_confidence >= confidence_threshold:
            combined_edges.append(
                {
                    "source": source,
                    "target": target,
                    "polarity": polarity,
                    "confidence": min(final_confidence, 1.0),
                    "method": "hybrid",
                    "sources": list(set(data["sources"])),
                }
            )
    return combined_edges


def combine_edges(
    llm_edges: List[Dict],
    algo_edges: List[Dict],
    confidence_threshold: float,
    use_llm_judge: bool = True,
    llm_weight: float = 0.5,
    algo_weight: float = 0.5,
    model_name: str = None,
    temperature: float = 0.3,
    batch_size: int = 10,
) -> List[Dict]:
    """
    Combine edges from LLM and Algorithmic graphs.

    Args:
        llm_edges: Edges from LLM graph
        algo_edges: Edges from Algorithmic graph
        confidence_threshold: Minimum confidence for edge inclusion
        use_llm_judge: If True, use LLM as judge; if False, use weighted average
        llm_weight: Weight for LLM edges (1-llm_weight for algo) - used only if use_llm_judge=False
        algo_weight: Weight for algo edges - used only if use_llm_judge=False
        model_name: LLM model name for judge
        temperature: Temperature for LLM judge
        batch_size: Number of edges to process in one LLM call

    Returns:
        Combined list of edges
    """
    if use_llm_judge:
        logger.info("Using LLM as judge to combine graphs")
        # Get field descriptions
        field_docs = FIELD_DOCS_RU

        return _combine_with_llm_judge(
            llm_edges,
            algo_edges,
            field_docs=field_docs,
            confidence_threshold=confidence_threshold,
            model_name=model_name,
            temperature=temperature,
            batch_size=batch_size,
        )
    else:
        logger.info("Using weighted average to combine graphs")
        weights_sum = llm_weight + algo_weight
        if weights_sum != 1.0:
            llm_weight /= weights_sum
        return _combine_weighted_average(
            llm_edges, algo_edges, llm_weight, confidence_threshold=confidence_threshold
        )


def build_hybrid_graph(
    df: Optional[pd.DataFrame] = None,
    csv_path: Optional[Path] = None,
    use_llm_judge: bool = True,
    llm_weight: float = 0.5,
    confidence_threshold: float | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    bootstrap_rounds: int | None = None,
    sample_rows: int | None = None,
) -> List[Dict]:
    """
    Build hybrid graph combining LLM and Algorithmic approaches.

    Args:
        df: DataFrame with data (if provided, csv_path is ignored)
        csv_path: Path to CSV file with data
        use_llm_judge: If True, use LLM as judge to combine graphs (default)
        llm_weight: Weight for LLM edges. Only applies if use_llm_judge=False
        confidence_threshold: Minimum confidence for edge inclusion
        model_name: Model name for LLM graph
        temperature: Temperature for LLM
        bootstrap_rounds: Bootstrap rounds for LLM
        sample_rows: Sample rows for LLM

    Returns:
        List of edges
    """
    cfg = get_config()

    # Get data
    if df is None:
        if csv_path is None:
            csv_path = cfg.synthetic_clients_path
        df = pd.read_csv(csv_path)

    logger.info("Building hybrid graph: starting LLM graph construction...")

    # Build LLM graph
    llm_edges, _ = build_llm_graph(
        df=df,
        model_name=model_name or cfg.llm.model_name,
        temperature=temperature or cfg.llm.temperature,
        bootstrap_rounds=bootstrap_rounds or cfg.llm.bootstrap_rounds,
        sample_rows=sample_rows or cfg.llm.sample_rows,
    )

    logger.info(f"LLM graph built: {len(llm_edges)} edges")
    logger.info("Building hybrid graph: starting Algorithmic graph construction...")

    # Build Algo graph
    # Save df to temp file for algo graph (it needs a path)
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        algo_consensus, _ = build_algo_graph(csv_path=Path(tmp.name))

    # Convert algo consensus to edges format
    algo_edges = create_algo_edges(algo_consensus)

    logger.info(f"Algorithmic graph built: {len(algo_edges)} edges")

    # Combine edges using LLM judge by default

    confidence_threshold = confidence_threshold or cfg.hybrid_graph.confidence_threshold

    combined_edges = combine_edges(
        llm_edges,
        algo_edges,
        llm_weight=llm_weight,
        confidence_threshold=confidence_threshold,
        use_llm_judge=use_llm_judge,
        model_name=model_name or cfg.llm.model_name,
        temperature=0.2,
        batch_size=15,
    )

    logger.info(f"Hybrid graph complete: {len(combined_edges)} combined edges")

    # Save hybrid edges
    hybrid_edges_path = cfg.full_artifacts_dir / "hybrid_edges.json"
    with open(hybrid_edges_path, "w", encoding="utf-8") as f:
        json.dump(combined_edges, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved hybrid edges to: {hybrid_edges_path}")

    return combined_edges
