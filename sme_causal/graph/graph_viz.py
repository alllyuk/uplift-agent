"""
Reusable graph visualization utilities (no Streamlit side effects).

This module provides helpers to render a NetworkX directed graph with PyVis
and to serialize it to common formats. It mirrors functionality previously
implemented inside the Streamlit app, but without importing Streamlit, so it
can be safely used from plain Python scripts and tests.
"""

from __future__ import annotations

import io
import json
import tempfile
from typing import Tuple
import os

import networkx as nx
from pyvis.network import Network

from sme_causal.core.constants import LAYER_INDEX
from sme_causal.graph.graph_utils import graph_to_dict

# PyVis options JSON for consistent layout/appearance
_PYVIS_OPTIONS = """
{
  "nodes": {"font": {"size": 40, "face": "arial"}},
  "edges": {"font": {"size": 12}},
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -80000,
      "centralGravity": 0.3,
      "springLength": 95,
      "springConstant": 0.04,
      "damping": 0.09,
      "avoidOverlap": 0
    },
    "minVelocity": 0.75
  }
}
"""

_LARGE_NODE_SIZE = 36
_SMALL_NODE_SIZE = 28


def _layer_name(idx: int) -> str:
    """Get human-readable layer name for the given index.

    Args:
        idx: Causal layer index (0-4).

    Returns:
        Short layer name.
    """
    mapping = {
        0: "Macro",
        1: "Relationship",
        2: "Transactional",
        3: "Interventions",
        4: "Outcomes",
    }
    return mapping.get(idx, f"Layer{idx}")


def _color_by_layer(idx: int) -> str:
    """Color for a node layer.

    Args:
        idx: Causal layer index.

    Returns:
        Hex color code.
    """
    return {
        0: "#3949ab",  # Macro - indigo
        1: "#00897b",  # Relationship - teal
        2: "#6d4c41",  # Transactional - brown
        3: "#f9a825",  # Interventions - amber
        4: "#c62828",  # Outcomes - red
    }.get(idx, "#546e7a")


def _color_by_polarity(pol: str) -> str:
    """Color for edge polarity.

    Args:
        pol: "+" for positive, "-" for negative, other for neutral.

    Returns:
        Hex color code.
    """
    pol = (pol or "+").strip().lower()
    mapping = {
        "+": "#2e7d32",  # green
        "-": "#c62828",  # red
    }
    return mapping.get(pol, "#616161")  # gray


def build_pyvis_html(
    G: nx.DiGraph,
    height_px: int = 650,
    directed: bool = True,
    physics: bool = True,
) -> str:
    """Generate interactive HTML visualization of a directed graph using PyVis.

    Args:
        G: NetworkX directed graph to visualize.
        height_px: Height of the canvas in pixels.
        directed: Whether to show directed arrows on edges.
        physics: Enable physics-based layout.

    Returns:
        HTML string with an embedded interactive visualization.
    """
    nt = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=directed,
        notebook=True,
        cdn_resources="in_line",
    )

    nt.set_options(_PYVIS_OPTIONS)

    # Nodes with layer-specific style
    for node in G.nodes():
        idx = LAYER_INDEX.get(node, 0)
        nt.add_node(
            node,
            label=f"{node} [{_layer_name(idx)}]",
            color=_color_by_layer(idx),
            shape="dot",
            size=_LARGE_NODE_SIZE if idx in (0, 4) else _SMALL_NODE_SIZE,
            title=node,
        )

    # Edges with polarity and confidence styling
    for u, v, data in G.edges(data=True):
        pol = data.get("polarity", "+")
        conf = float(data.get("confidence", 0.5))
        title = f"{u} → {v}<br/>polarity: {pol}, confidence: {conf:.2f}"
        rationale = (data.get("rationale") or "").strip()
        if rationale:
            title += f"<br/>{rationale}"
        nt.add_edge(
            u,
            v,
            arrows="to",
            color=_color_by_polarity(pol),
            width=10 + 3 * max(0.0, min(conf, 1.0)),
            title=title,
            physics=physics,
        )

    # Save to a temporary file and return contents
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        try:
            nt.save_graph(tmp.name)
        except UnicodeEncodeError:
            # Fallback: write with explicit UTF-8 encoding
            with open(tmp.name, "w", encoding="utf-8") as f:
                f.write(nt.html)
        tmp_path = tmp.name
    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()
    # Clean up the temporary file to avoid leaving artifacts on disk
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    return html


def stringify_graph(G: nx.DiGraph) -> Tuple[bytes, bytes]:
    """Serialize graph to JSON and GEXF binary payloads.

    Args:
        G: NetworkX directed graph to serialize.

    Returns:
        Tuple of (json_bytes, gexf_bytes).
    """
    data = graph_to_dict(G)
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    gexf_buf = io.BytesIO()
    nx.write_gexf(G, gexf_buf)
    gexf_bytes = gexf_buf.getvalue()
    return json_bytes, gexf_bytes
