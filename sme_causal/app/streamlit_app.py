"""
Streamlit web application with chat-based agent interface.

Provides:
1. Chat interface for natural language queries and what-if analysis
2. Quick action buttons for common interventions
3. Causal graph visualization (collapsed in expander)
4. Pipeline status and critic feedback in responses
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger

from sme_causal.agent.agent_service import CausalAgent, QueryParser
from sme_causal.core.columns import (
    CONTEXT_FIELDS as CORE_CONTEXT_FIELDS,
    CLIENT_ID,
    INDUSTRY,
    REGION,
    NEW_PRODUCT_OFFER,
    NEW_PRODUCT_OFFER_TYPE,
    CREDIT_LIMIT_CHANGE,
    TARIFF_DISCOUNT,
)
from sme_causal.core.config import AppConfig, get_config
from sme_causal.core.llm_clients.factory import reset_llm_client
from sme_causal.core.utils import (
    configure_logging,
    parse_client_id_and_intent,
)
from sme_causal.graph.build_llm_graph import build_llm_graph
from sme_causal.graph.build_algo_graph import build_algo_graph
from sme_causal.graph.build_hybrid_graph import build_hybrid_graph
from sme_causal.graph.build_algo_llm_graph import build_algo_llm_graph
from sme_causal.graph.graph_utils import edges_to_digraph, export_graph, create_algo_edges
from sme_causal.graph.graph_viz import build_pyvis_html
from sme_causal.orchestrator.pipeline import Pipeline
from sme_causal.orchestrator.persistence import CaseStore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
cfg: AppConfig = get_config()
configure_logging(cfg.streamlit_log_path, cfg.logging, add_stdout=False)

API_KEY_ERROR = "Не задан OpenAI API key — задайте OPENAI_API_KEY в окружении или .env."
CONTEXT_FIELDS: List[str] = CORE_CONTEXT_FIELDS
DISPLAY_COLUMNS: List[str] = CONTEXT_FIELDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_api_key() -> bool:
    if not os.getenv("OPENAI_API_KEY") and cfg.effective_llm_api_key:
        os.environ["OPENAI_API_KEY"] = str(cfg.effective_llm_api_key)
    base = cfg.effective_llm_base_url
    if base:
        os.environ["OPENAI_BASE_URL"] = base
    return bool(os.getenv("OPENAI_API_KEY"))


@st.cache_data(show_spinner=False)
def load_existing_data(version_ts: float) -> pd.DataFrame:
    path = cfg.synthetic_clients_path
    if not path.exists():
        raise FileNotFoundError(f"Synthetic data file not found: {path}")
    _ = version_ts
    return pd.read_csv(path)


def render_explanation(expl: Any, title: Optional[str] = None) -> None:
    """Render explanation in readable form with fallback to JSON."""
    if hasattr(expl, "__dict__"):
        data = {
            "diagnosis": getattr(expl, "diagnosis", "") or "",
            "drivers_pos": getattr(expl, "drivers_pos", []) or [],
            "drivers_neg": getattr(expl, "drivers_neg", []) or [],
            "recommendations": getattr(expl, "recommendations", []) or [],
            "expected_effect": getattr(expl, "expected_effect", "") or "",
            "raw_text": getattr(expl, "raw_text", "") or "",
        }
    elif isinstance(expl, dict):
        data = {
            "diagnosis": str(expl.get("diagnosis", "")),
            "drivers_pos": list(expl.get("drivers_pos", []) or []),
            "drivers_neg": list(expl.get("drivers_neg", []) or []),
            "recommendations": list(expl.get("recommendations", []) or []),
            "expected_effect": str(expl.get("expected_effect", "")),
            "raw_text": str(expl.get("raw_text", "")),
        }
    else:
        st.write(expl)
        return

    if title:
        st.markdown(f"#### {title}")

    has_struct = any([
        data["diagnosis"],
        data["drivers_pos"],
        data["drivers_neg"],
        data["recommendations"],
        data["expected_effect"],
    ])

    if has_struct:
        col_p, col_n = st.columns(2)
        with col_p:
            st.markdown("**Drivers (+)**")
            if data["drivers_pos"]:
                st.markdown("\n".join([f"- {item}" for item in data["drivers_pos"]]))
            else:
                st.markdown("---")
        with col_n:
            st.markdown("**Drivers (-)**")
            if data["drivers_neg"]:
                st.markdown("\n".join([f"- {item}" for item in data["drivers_neg"]]))
            else:
                st.markdown("---")

        if data["recommendations"]:
            st.markdown("**Recommendations**")
            st.markdown("\n".join([
                f"{i + 1}. {rec}" for i, rec in enumerate(data["recommendations"])
            ]))

        if data["expected_effect"]:
            st.info(f"**Expected effect:** {data['expected_effect']}")
    else:
        try:
            st.json(json.loads(data["raw_text"]))
        except Exception:
            st.write(data["raw_text"])


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=cfg.streamlit.page_title,
    page_icon=cfg.streamlit.page_icon,
    layout=cfg.streamlit.layout,
)

st.title(cfg.streamlit.page_title or "SME Causal Agent")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    # LLM Provider
    current_provider = cfg.effective_llm_provider
    default_idx = 0 if current_provider == "openai" else 1
    provider = st.selectbox("LLM Provider", ["openai", "local"], index=default_idx)
    if provider != current_provider:
        os.environ["LLM_PROVIDER"] = provider
        reset_llm_client()
        st.success(f"Provider switched to: {provider}")
        st.rerun()

    model_name = st.text_input("Model", value=cfg.llm.model_name)
    temperature = st.slider("Temperature", 0.0, 1.0, float(np.clip(cfg.llm.temperature, 0.0, 1.0)), 0.05)
    min_conf = st.slider("Edge confidence threshold", 0.0, 1.0, float(cfg.hybrid_graph.confidence_threshold), 0.05)

    st.markdown("---")
    st.header("Client")

    # Client filters
    try:
        df = load_existing_data(cfg.synthetic_clients_path.stat().st_mtime)
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.stop()

    industry_filter = st.selectbox(INDUSTRY, options=["(all)"] + sorted(df[INDUSTRY].unique().tolist()))
    region_filter = st.selectbox(REGION, options=["(all)"] + sorted(df[REGION].unique().tolist()))

    dff = df.copy()
    if industry_filter != "(all)":
        dff = dff[dff[INDUSTRY] == industry_filter]
    if region_filter != "(all)":
        dff = dff[dff[REGION] == region_filter]

    ids = dff[CLIENT_ID].head(500).tolist()
    if not ids:
        st.warning("No clients match filters.")
        st.stop()

    client_id = st.selectbox("Client_ID (top-500)", options=ids)

    st.markdown("---")
    st.header("Analysis options")
    use_psm_flag = st.checkbox("PSM estimation", value=True)
    use_graph_flag = st.checkbox("Causal graph", value=True)
    use_rag_flag = st.checkbox("RAG context", value=True)

    graph_method = st.selectbox(
        "Graph method",
        ["llm", "algo", "algo_llm", "hybrid"],
        index=0,
    )

# ---------------------------------------------------------------------------
# Client profile (compact)
# ---------------------------------------------------------------------------
with st.expander("Client profile", expanded=False):
    cols_show = [c for c in DISPLAY_COLUMNS if c in dff.columns]
    st.dataframe(
        dff[dff[CLIENT_ID] == client_id][[CLIENT_ID] + cols_show],
        use_container_width=True,
        height=220,
    )

# ---------------------------------------------------------------------------
# Graph visualization (collapsed)
# ---------------------------------------------------------------------------
with st.expander("Causal graph", expanded=False):
    method_ui = st.radio(
        "Build method:",
        ["LLM", "Algo", "Algo_LLM_validation", "Hybrid"],
        horizontal=True,
    )
    run_graph_btn = st.button("Build graph")

    edges: Optional[List[Dict]] = None

    if run_graph_btn:
        if method_ui != "Algo" and not ensure_api_key():
            st.error(API_KEY_ERROR)
        else:
            with st.spinner("Building graph..."):
                try:
                    if method_ui == "Hybrid":
                        edges = build_hybrid_graph(
                            df=df, llm_weight=0.5,
                            confidence_threshold=float(min_conf),
                            model_name=model_name,
                            temperature=float(temperature),
                        )
                    elif method_ui == "LLM":
                        edges, _ = build_llm_graph(
                            df=df, model_name=model_name,
                            temperature=float(temperature),
                        )
                    elif method_ui == "Algo":
                        consensus, _ = build_algo_graph(csv_path=cfg.synthetic_clients_path)
                        edges = create_algo_edges(consensus)
                    elif method_ui == "Algo_LLM_validation":
                        consensus = build_algo_llm_graph(csv_path=cfg.synthetic_clients_path)
                        edges = create_algo_edges(consensus)
                    if edges:
                        st.success(f"Done: {len(edges)} edges")
                except Exception as e:
                    st.exception(e)

    # Load from artifacts if not just built
    if edges is None:
        artifact_map = {
            "Hybrid": cfg.full_artifacts_dir / "hybrid_edges.json",
            "LLM": cfg.llm_edges_path,
            "Algo": cfg.algo_edges_path,
            "Algo_LLM_validation": cfg.full_algorithmic_dir / "algo_llm_edges.json",
        }
        f = artifact_map.get(method_ui)
        if f and f.exists():
            raw = json.loads(f.read_text(encoding="utf-8"))
            if method_ui in ("Algo", "Algo_LLM_validation"):
                raw_edges = raw.get("edges", raw) if isinstance(raw, dict) else raw
                edges = create_algo_edges(raw_edges) if not isinstance(raw_edges[0] if raw_edges else {}, dict) or "source" not in (raw_edges[0] if raw_edges else {}) else raw_edges
            else:
                edges = raw
            st.info(f"Loaded {len(edges)} edges from artifacts")

    if edges:
        filtered = [e for e in edges if float(e.get("confidence", 0.0)) >= float(min_conf)]
        st.caption(f"Threshold: {min_conf:.2f}. Showing {len(filtered)}/{len(edges)} edges")
        G = edges_to_digraph(filtered)
        export_graph(G, out_prefix=str(cfg.full_artifacts_dir / cfg.paths.graph_prefix))
        html = build_pyvis_html(G, height_px=500, directed=True, physics=True)
        st.components.v1.html(html, height=520, scrolling=True)

# ---------------------------------------------------------------------------
# Initialize Pipeline + session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "pipeline" not in st.session_state or st.session_state.get("_pipe_cfg") != (
    model_name, temperature, graph_method, use_psm_flag, use_graph_flag, use_rag_flag
):
    case_store = None
    try:
        case_store = CaseStore(cfg.cases_db_path)
    except Exception:
        pass
    st.session_state["pipeline"] = Pipeline(
        df,
        case_store=case_store,
        graph_method=graph_method,
        use_rag=use_rag_flag,
        use_graph=use_graph_flag,
        use_psm=use_psm_flag,
        min_conf=min_conf,
        model=model_name,
        temperature=float(temperature),
    )
    st.session_state["_pipe_cfg"] = (
        model_name, temperature, graph_method, use_psm_flag, use_graph_flag, use_rag_flag
    )

pipeline: Pipeline = st.session_state["pipeline"]


# ---------------------------------------------------------------------------
# Quick action buttons
# ---------------------------------------------------------------------------
st.markdown("#### Quick actions")
col_b1, col_b2, col_b3, col_b4 = st.columns(4)
quick_query: Optional[str] = None

with col_b1:
    if st.button("Explain client"):
        quick_query = f"Explain the current situation of client {client_id}"
with col_b2:
    if st.button("Offer acquiring"):
        quick_query = f"Offer acquiring product to client {client_id}"
with col_b3:
    if st.button("Credit limit +15%"):
        quick_query = f"Increase credit limit by 15% for client {client_id}"
with col_b4:
    if st.button("Tariff discount"):
        quick_query = f"Apply tariff discount for client {client_id}"


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------
def _render_assistant_message(msg_data: Dict[str, Any]) -> None:
    """Render a structured assistant message inside st.chat_message."""
    state = msg_data.get("case_state", {})
    status = state.get("status", "unknown")

    # Pipeline steps status
    with st.status(
        "Analysis complete" if status in ("done", "degraded") else f"Status: {status}",
        expanded=False,
        state="complete" if status == "done" else ("error" if status == "aborted" else "running"),
    ):
        # Policy
        policy = state.get("policy_result", {})
        if policy.get("blocked"):
            st.write(f"Policy check: blocked --- {', '.join(policy.get('reasons', []))}")
        else:
            st.write("Policy check: passed")

        # PSM
        psm = state.get("psm_result")
        if psm and psm.get("ok"):
            st.write(f"PSM: ATT={psm.get('att')}, n_pairs={psm.get('n_pairs')}")
        elif psm:
            st.write(f"PSM: error --- {psm.get('error', 'unknown')}")
        else:
            st.write("PSM: skipped")

        # RAG
        chunks = state.get("rag_chunks", [])
        st.write(f"RAG: {len(chunks)} chunks" if chunks else "RAG: skipped")

        # Graph
        graph_dsl = state.get("graph_dsl", "")
        if graph_dsl:
            n_edges = graph_dsl.count("\n") + 1
            st.write(f"Graph DSL: loaded, ~{n_edges} edges")
        else:
            st.write("Graph: skipped")

        # Critic
        critic = state.get("critic_result", {})
        if critic:
            if critic.get("passed"):
                st.write("Critic: passed")
            else:
                retry = state.get("retry_count", 0)
                issues_str = "; ".join(critic.get("issues", [])[:3])
                st.write(f"Critic: failed (retry={retry}) --- {issues_str}")

    # Abort case
    if status == "aborted":
        reason = state.get("abort_reason", "unknown")
        if reason == "policy_blocked":
            policy = state.get("policy_result", {})
            st.warning("Intervention blocked by policy checks")
            for r in policy.get("reasons", []):
                st.markdown(f"- {r}")
            st.info("**Expected effect:** 0.0")
        else:
            st.error(f"Case aborted: {reason}")
        return

    # Explanation
    explanation = state.get("explanation", {})
    if explanation:
        render_explanation(explanation)

    # Metadata footer
    latency = state.get("latency_ms")
    case_id = state.get("case_id", "")[:8]
    st.caption(f"Status: {status} | Latency: {latency}ms | Case: {case_id}")

    if state.get("requires_human_review"):
        st.warning(f"Requires human review: {state.get('review_reason', '')}")


# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant_message(msg["content"])
        else:
            st.write(msg["content"])


# ---------------------------------------------------------------------------
# Process query
# ---------------------------------------------------------------------------
def _process_query(query_text: str) -> None:
    """Parse query, run pipeline, add result to chat."""
    if not ensure_api_key():
        st.error(API_KEY_ERROR)
        return

    # Add user message
    st.session_state["messages"].append({"role": "user", "content": query_text})

    # Parse client ID from query
    explicit_cid, cleaned_query = parse_client_id_and_intent(query_text)
    target_cid = explicit_cid or client_id

    # Parse intent via LLM
    parser = QueryParser(model=model_name, temperature=0.0)
    parsed_data = parser.parse(cleaned_query)

    delta: Dict[str, Any] = {}
    match_info: Optional[Dict] = None
    target_metric: Optional[str] = None
    action_type = "what_if"

    if parsed_data:
        action_type = parsed_data.action_type
        delta = parsed_data.delta
        match_info = parsed_data.match_info
        target_metric = parsed_data.target_metric

    # Run pipeline
    if action_type == "optimize":
        # For optimize, run pipeline with empty delta
        case_state = pipeline.run(
            target_cid, {},
            raw_query=cleaned_query,
            target_metric=target_metric,
        )
    else:
        case_state = pipeline.run(
            target_cid, delta,
            raw_query=cleaned_query,
            target_metric=target_metric,
            match_info=match_info,
        )

    # Add assistant message
    st.session_state["messages"].append({
        "role": "assistant",
        "content": {"case_state": dict(case_state)},
    })


# Handle quick action buttons
if quick_query:
    _process_query(quick_query)
    st.rerun()

# Handle chat input
chat_input = st.chat_input("Ask about interventions or client analysis...")
if chat_input:
    _process_query(chat_input)
    st.rerun()
