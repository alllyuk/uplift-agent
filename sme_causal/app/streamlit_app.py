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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path so `sme_causal` is importable
# without requiring PYTHONPATH=.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import streamlit as st
from loguru import logger

from sme_causal.agent.agent_service import CausalAgent, QueryParser
from sme_causal.core.columns import (
    CONTEXT_FIELDS as CORE_CONTEXT_FIELDS,
    CLIENT_ID,
    INDUSTRY,
    REGION,
)
from sme_causal.core.config import AppConfig, get_config
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
            st.markdown("**Положительные факторы**")
            if data["drivers_pos"]:
                st.markdown("\n".join([f"- {item}" for item in data["drivers_pos"]]))
            else:
                st.markdown("---")
        with col_n:
            st.markdown("**Отрицательные факторы**")
            if data["drivers_neg"]:
                st.markdown("\n".join([f"- {item}" for item in data["drivers_neg"]]))
            else:
                st.markdown("---")

        if data["recommendations"]:
            st.markdown("**Рекомендации**")
            st.markdown("\n".join([
                f"{i + 1}. {rec}" for i, rec in enumerate(data["recommendations"])
            ]))

        if data["expected_effect"]:
            st.info(f"**Ожидаемый эффект:** {data['expected_effect']}")
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
# Load data
# ---------------------------------------------------------------------------
try:
    df = load_existing_data(cfg.synthetic_clients_path.stat().st_mtime)
except FileNotFoundError as e:
    st.error(f"Датасет не найден: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — analysis options only
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Параметры анализа")
    use_psm_flag = st.checkbox("PSM-оценка", value=True)
    use_graph_flag = st.checkbox("Каузальный граф", value=True)
    use_rag_flag = st.checkbox("RAG-контекст", value=True)

    graph_method = st.selectbox(
        "Метод построения графа",
        ["llm", "algo", "algo_llm", "hybrid"],
        index=0,
    )
    min_conf = st.slider(
        "Порог уверенности рёбер", 0.0, 1.0,
        float(cfg.hybrid_graph.confidence_threshold), 0.05,
    )

# Use config defaults for model params (not exposed in UI)
model_name = cfg.llm.model_name
temperature = float(cfg.llm.temperature)

# ---------------------------------------------------------------------------
# Initialize Pipeline + session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Include mtime of orchestrator source files in cache key so the Pipeline is
# rebuilt (with re-imported modules) whenever orchestrator code changes.
import importlib as _importlib
from sme_causal.orchestrator import pipeline as _pipeline_mod
from sme_causal.orchestrator import persistence as _persistence_mod

def _file_mtime(mod) -> float:
    try:
        return Path(mod.__file__).stat().st_mtime
    except Exception:
        return 0.0

_src_fingerprint = (_file_mtime(_pipeline_mod), _file_mtime(_persistence_mod))
_pipe_key = (graph_method, use_psm_flag, use_graph_flag, use_rag_flag, min_conf, _src_fingerprint)
if "pipeline" not in st.session_state or st.session_state.get("_pipe_cfg") != _pipe_key:
    # Force reload of orchestrator modules so any class changes take effect
    _importlib.reload(_persistence_mod)
    _importlib.reload(_pipeline_mod)
    _CaseStore = _persistence_mod.CaseStore
    _Pipeline = _pipeline_mod.Pipeline
    case_store = None
    try:
        case_store = _CaseStore(cfg.cases_db_path)
    except Exception:
        pass
    st.session_state["pipeline"] = _Pipeline(
        df,
        case_store=case_store,
        graph_method=graph_method,
        use_rag=use_rag_flag,
        use_graph=use_graph_flag,
        use_psm=use_psm_flag,
        min_conf=min_conf,
    )
    st.session_state["_pipe_cfg"] = _pipe_key

pipeline = st.session_state["pipeline"]


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------
def _render_assistant_message(msg_data: Dict[str, Any]) -> None:
    """Render a structured assistant message inside st.chat_message."""
    # Baseline explanation (from "Объяснить клиента")
    if "baseline" in msg_data:
        render_explanation(msg_data["baseline"], title="Базовая диагностика клиента")
        return

    state = msg_data.get("case_state", {})
    status = state.get("status", "unknown")

    status_labels = {
        "done": "Анализ завершён",
        "degraded": "Анализ завершён (пониженная уверенность)",
        "aborted": "Анализ прерван",
    }

    # Pipeline steps status
    with st.status(
        status_labels.get(status, f"Статус: {status}"),
        expanded=False,
        state="complete" if status == "done" else ("error" if status == "aborted" else "running"),
    ):
        policy = state.get("policy_result", {})
        if policy.get("blocked"):
            st.write(f"Policy-проверка: заблокировано — {', '.join(policy.get('reasons', []))}")
        else:
            st.write("Policy-проверка: пройдена")

        psm = state.get("psm_result")
        if psm and psm.get("ok"):
            st.write(f"PSM: ATT={psm.get('att')}, n_pairs={psm.get('n_pairs')}")
        elif psm:
            st.write(f"PSM: ошибка — {psm.get('error', 'unknown')}")
        else:
            st.write("PSM: пропущен")

        chunks = state.get("rag_chunks", [])
        st.write(f"RAG: найдено {len(chunks)} фрагментов" if chunks else "RAG: пропущен")

        graph_dsl = state.get("graph_dsl", "")
        if graph_dsl:
            n_edges = graph_dsl.count("\n") + 1
            st.write(f"Каузальный граф: загружен, ~{n_edges} рёбер")
        else:
            st.write("Каузальный граф: пропущен")

        critic = state.get("critic_result", {})
        if critic:
            if critic.get("passed"):
                st.write("Critic: проверка пройдена")
            else:
                retry = state.get("retry_count", 0)
                issues_str = "; ".join(critic.get("issues", [])[:3])
                st.write(f"Critic: не пройдена (повтор={retry}) — {issues_str}")

    # Abort case
    if status == "aborted":
        reason = state.get("abort_reason", "unknown")
        if reason == "policy_blocked":
            policy = state.get("policy_result", {})
            prev = state.get("cooldown_previous_case")
            if prev:
                st.info(
                    f"♻️ Эта интервенция уже оценивалась для клиента "
                    f"**{state.get('client_id')}** {prev.get('created_at', '?')}. "
                    f"Показан предыдущий результат (кейс `{(prev.get('case_id') or '')[:8]}`)."
                )
                render_explanation(prev.get("explanation") or {})
            else:
                st.warning("Интервенция заблокирована policy-проверками")
                for r in policy.get("reasons", []):
                    st.markdown(f"- {r}")
                st.info("**Ожидаемый эффект:** 0.0")
        else:
            reason_labels = {
                "no_evidence": "нет данных ни от одного источника",
                "client_not_found": "клиент не найден в датасете",
            }
            st.error(f"Кейс прерван: {reason_labels.get(reason, reason)}")
        return

    # Explanation
    explanation = state.get("explanation", {})
    if explanation:
        render_explanation(explanation)

    # Metadata footer
    latency = state.get("latency_ms")
    case_id = state.get("case_id", "")[:8]
    st.caption(f"Статус: {status} | Задержка: {latency} мс | Кейс: {case_id}")

    if state.get("requires_human_review"):
        st.warning(f"Требуется ручная проверка: {state.get('review_reason', '')}")


# ---------------------------------------------------------------------------
# Process query
# ---------------------------------------------------------------------------
def _explain_client(target_cid: str) -> None:
    """Run baseline diagnosis via agent.explain_client() and add to chat."""
    agent = pipeline.agent
    ctx = agent.build_context_for_client(df, target_cid)
    expl = agent.explain_client(
        ctx,
        use_graph=use_graph_flag,
        min_conf=min_conf,
    )
    expl_dict = {
        "diagnosis": getattr(expl, "diagnosis", "") or "",
        "drivers_pos": getattr(expl, "drivers_pos", []) or [],
        "drivers_neg": getattr(expl, "drivers_neg", []) or [],
        "recommendations": getattr(expl, "recommendations", []) or [],
        "expected_effect": getattr(expl, "expected_effect", "") or "",
        "raw_text": getattr(expl, "raw_text", "") or "",
    }
    st.session_state["messages"].append({
        "role": "assistant",
        "content": {"baseline": expl_dict},
    })


def _enqueue_query(
    query_text: str, *, is_explain: bool = False, fallback_cid: Optional[str] = None
) -> None:
    """Append user message + mark query pending. Actual work happens on next rerun."""
    if not ensure_api_key():
        st.error(API_KEY_ERROR)
        return
    st.session_state["messages"].append({"role": "user", "content": query_text})
    st.session_state["pending_query"] = {
        "text": query_text,
        "is_explain": is_explain,
        "fallback_cid": fallback_cid,
    }


def _execute_pending_query() -> None:
    """Run pipeline for queued query and append assistant message."""
    pq = st.session_state.pop("pending_query", None)
    if not pq:
        return
    query_text = pq["text"]
    is_explain = pq["is_explain"]
    fallback_cid = pq["fallback_cid"]

    explicit_cid, cleaned_query = parse_client_id_and_intent(query_text)
    target_cid = explicit_cid or fallback_cid

    # Remember the client actually analyzed so the profile expander
    # can reflect it (even when extracted from a free-form query).
    if target_cid:
        st.session_state["active_client_id"] = target_cid

    if is_explain:
        _explain_client(target_cid)
        return

    parser = QueryParser(model=model_name, temperature=0.0)
    parsed_data = parser.parse(cleaned_query)

    delta: Dict[str, Any] = {}
    match_info: Optional[Dict] = None
    target_metric: Optional[str] = None
    if parsed_data:
        delta = parsed_data.delta
        match_info = parsed_data.match_info
        target_metric = parsed_data.target_metric

    if not delta:
        _explain_client(target_cid)
        return

    case_state = pipeline.run(
        target_cid, delta,
        raw_query=cleaned_query,
        target_metric=target_metric,
        match_info=match_info,
    )
    st.session_state["messages"].append({
        "role": "assistant",
        "content": {"case_state": dict(case_state)},
    })


# ---------------------------------------------------------------------------
# Main area: two side-by-side options — free query (left) | quick actions (right)
# ---------------------------------------------------------------------------
if "query_key_counter" not in st.session_state:
    st.session_state["query_key_counter"] = 0

left_col, or_col, right_col = st.columns([10, 1, 10])

# --- Right column (executed first so client_id is known) ---
with right_col:
    st.markdown("##### 🎯 Вариант 2 — выбрать клиента и быстрое действие")
    st.caption("Кнопка сразу запускает анализ для выбранного клиента.")
    sel_c1, sel_c2, sel_c3 = st.columns(3)
    with sel_c1:
        industries = ["(любая)"] + sorted(df[INDUSTRY].dropna().unique().tolist())
        industry_filter = st.selectbox("Отрасль", industries, index=0)
    with sel_c2:
        regions = ["(любой)"] + sorted(df[REGION].dropna().unique().tolist())
        region_filter = st.selectbox("Регион", regions, index=0)

    dff = df.copy()
    if industry_filter != "(любая)":
        dff = dff[dff[INDUSTRY] == industry_filter]
    if region_filter != "(любой)":
        dff = dff[dff[REGION] == region_filter]

    with sel_c3:
        ids = dff[CLIENT_ID].astype(str).tolist()
        if not ids:
            st.warning("Под выбранные фильтры не подходит ни один клиент.")
            st.stop()
        client_id = st.selectbox("ID клиента", ids, index=0)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("Объяснить клиента", use_container_width=True):
            _enqueue_query(
                f"Объясни текущую ситуацию клиента {client_id}",
                is_explain=True,
                fallback_cid=client_id,
            )
            st.rerun()
        if st.button("Кредитный лимит +15%", use_container_width=True):
            _enqueue_query(
                f"Что если поднять кредитный лимит клиенту {client_id} на 15%",
                fallback_cid=client_id,
            )
            st.rerun()
    with col_b2:
        if st.button("Предложить эквайринг", use_container_width=True):
            _enqueue_query(
                f"Предложить клиенту {client_id} эквайринг",
                fallback_cid=client_id,
            )
            st.rerun()
        if st.button("Скидка на тариф", use_container_width=True):
            _enqueue_query(
                f"Применить скидку на тариф клиенту {client_id}",
                fallback_cid=client_id,
            )
            st.rerun()

# --- Left column (uses client_id as fallback) ---
with left_col:
    st.markdown("##### 📝 Вариант 1 — описать запрос своими словами")
    st.caption(
        "Можно указать ID клиента прямо в тексте (например `C000005`), "
        "либо запрос применится к клиенту, выбранному справа."
    )
    with st.form(
        key=f"query_form_{st.session_state['query_key_counter']}",
        clear_on_submit=True,
    ):
        chat_input = st.text_input(
            "Запрос",
            placeholder="Предложить клиенту C000005 эквайринг / Что если поднять кредитный лимит на 20%?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Отправить")
    if submitted and chat_input.strip():
        _enqueue_query(chat_input.strip(), fallback_cid=client_id)
        st.session_state["query_key_counter"] += 1
        st.rerun()

# --- Middle column: vertical "OR" divider ---
with or_col:
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.9em;"
        " letter-spacing:2px; padding-top:60px;'>или</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Client profile (collapsed) — reflects the client actually analyzed,
# which may come from a free-form query rather than the selectbox.
# ---------------------------------------------------------------------------
profile_cid = st.session_state.get("active_client_id") or client_id
# Use the full df (not the industry/region-filtered dff) so that a client
# extracted from a free-form query is always found.
profile_row = df[df[CLIENT_ID].astype(str) == str(profile_cid)]
with st.expander(f"Профиль клиента {profile_cid}", expanded=False):
    if profile_row.empty:
        st.info(f"Клиент `{profile_cid}` не найден в датасете.")
    else:
        cols_show = [c for c in DISPLAY_COLUMNS if c in profile_row.columns]
        st.dataframe(
            profile_row[[CLIENT_ID] + cols_show].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# Chat history (including pending query execution)
# ---------------------------------------------------------------------------
hist_header_col, hist_clear_col = st.columns([5, 1])
with hist_header_col:
    st.markdown("##### 💬 История диалога")
with hist_clear_col:
    if st.session_state.get("messages") and st.button(
        "Очистить чат", use_container_width=True
    ):
        st.session_state["messages"] = []
        st.session_state.pop("pending_query", None)
        st.rerun()

# Group messages into (user, assistant) pairs. Pairs render newest-first,
# but within each pair: user question on top, assistant answer below.
_pairs: List[List[Dict[str, Any]]] = []
_current: List[Dict[str, Any]] = []
for _m in st.session_state["messages"]:
    if _m["role"] == "user":
        if _current:
            _pairs.append(_current)
        _current = [_m]
    else:
        _current.append(_m)
if _current:
    _pairs.append(_current)

_has_pending = st.session_state.get("pending_query") is not None

for _idx, _pair in enumerate(reversed(_pairs)):
    _is_newest = _idx == 0
    for _m in _pair:
        with st.chat_message(_m["role"]):
            if _m["role"] == "assistant":
                _render_assistant_message(_m["content"])
            else:
                st.write(_m["content"])
    # If this is the newest pair and a query is pending, render spinner
    # directly under the user message (before any older pairs).
    if _is_newest and _has_pending:
        with st.chat_message("assistant"):
            with st.spinner("Выполняю анализ..."):
                _execute_pending_query()
        st.rerun()

# ---------------------------------------------------------------------------
# Graph visualization (collapsed)
# ---------------------------------------------------------------------------
with st.expander("Каузальный граф", expanded=False):
    method_ui = st.radio(
        "Метод построения:",
        ["LLM", "Algo", "Algo_LLM_validation", "Hybrid"],
        horizontal=True,
    )
    run_graph_btn = st.button("Построить граф")

    edges: Optional[List[Dict]] = None

    if run_graph_btn:
        if method_ui != "Algo" and not ensure_api_key():
            st.error(API_KEY_ERROR)
        else:
            with st.spinner("Строю граф..."):
                try:
                    if method_ui == "Hybrid":
                        edges = build_hybrid_graph(
                            df=df, llm_weight=0.5,
                            confidence_threshold=float(min_conf),
                            model_name=model_name,
                            temperature=temperature,
                        )
                    elif method_ui == "LLM":
                        edges, _ = build_llm_graph(
                            df=df, model_name=model_name,
                            temperature=temperature,
                        )
                    elif method_ui == "Algo":
                        consensus, _ = build_algo_graph(csv_path=cfg.synthetic_clients_path)
                        edges = create_algo_edges(consensus)
                    elif method_ui == "Algo_LLM_validation":
                        consensus = build_algo_llm_graph(csv_path=cfg.synthetic_clients_path)
                        edges = create_algo_edges(consensus)
                    if edges:
                        st.success(f"Готово: {len(edges)} рёбер")
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
            st.info(f"Загружено {len(edges)} рёбер из артефактов")

    if edges:
        filtered = [e for e in edges if float(e.get("confidence", 0.0)) >= float(min_conf)]
        st.caption(f"Порог: {min_conf:.2f}. Показано {len(filtered)} из {len(edges)} рёбер")
        G = edges_to_digraph(filtered)
        export_graph(G, out_prefix=str(cfg.full_artifacts_dir / cfg.paths.graph_prefix))
        html = build_pyvis_html(G, height_px=500, directed=True, physics=True)
        st.components.v1.html(html, height=520, scrolling=True)
