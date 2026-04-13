from __future__ import annotations
from typing import Dict, Any, List
import networkx as nx
from pathlib import Path
import json
import pandas as pd


def graph_to_dict(G: nx.DiGraph) -> Dict[str, Any]:
    """Convert a directed NetworkX graph to a serializable dict.

    Structure:
      {"nodes": [...], "edges": [{source, target, ...attrs}]}
    """
    return {
        "nodes": list(G.nodes()),
        "edges": [{"source": u, "target": v, **G[u][v]} for u, v in G.edges()],
    }


def edges_to_digraph(edges: List[Dict]) -> nx.DiGraph:
    """Convert a list of edge dicts to a NetworkX DiGraph.

    Each edge dict may contain keys: source, target, relation, polarity,
    confidence, rationale. Missing fields fall back to sensible defaults.
    """
    G = nx.DiGraph()
    for e in edges:
        src, dst = e["source"], e["target"]
        G.add_edge(
            src,
            dst,
            relation=e.get("relation", "causal"),
            polarity=e.get("polarity", "+"),
            confidence=float(e.get("confidence", 0.5)),
            rationale=e.get("rationale", ""),
        )
    return G


def export_graph(G: nx.DiGraph, out_prefix: str) -> None:
    """Export graph to JSON, GEXF, and GraphML using a provided prefix.

    Files produced: {out_prefix}.json, {out_prefix}.gexf, {out_prefix}.graphml
    """
    data = graph_to_dict(G)
    with open(out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    nx.write_gexf(G, out_prefix + ".gexf")
    nx.write_graphml(G, out_prefix + ".graphml")


def adjust_robustness(row: Any[Dict, pd.Series]) -> str:
    score = row.get("robustness_score", 0.5)
    support_ge_tau = row.get("support_ge_tau", 0)
    if support_ge_tau == 3:
        score *= 1.2
    elif support_ge_tau == 2:
        score *= 1.0
    elif support_ge_tau == 1:
        score *= 0.8
    score = min(score, 1.0)
    row["robustness_score"] = score
    return row


def create_algo_edges(edges_src: List[Dict]) -> List[Dict]:
    """Convert algorithmic format to edge list."""
    edges = []
    for row in edges_src:
        # Возможны два формата ключей, учитываем оба варианта
        source = row.get("source") or row.get("u")
        target = row.get("target") or row.get("v")

        # Меняем robustness_score в зависимости от уверенности алгоритмов, если нужно
        row = adjust_robustness(row)

        edges.append(
            {
                "source": source,
                "target": target,
                "polarity": row.get("sign") or row.get("polarity") or "?",
                "confidence": row.get("robustness_score", 0.5),
                "rationale": f"Algorithmic consensus: {row.get('robustness_label', '')}",
            }
        )

    return edges
