"""
Streamlit web application for SME causal graph inference and visualization.

This module provides an interactive web interface for:
1. Generating synthetic SME client data
2. Running LLM-based causal inference
3. Visualizing causal graphs with PyVis
4. Comparing results against ground truth
5. Explaining individual client scenarios

The application integrates all components of the SME causal analysis pipeline
into a user-friendly web interface.
"""

# streamlit_app.py

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger

from sme_causal.inference.psm import CausalInferenceAnalyzer

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

from sme_causal.agent.agent_service import CausalAgent, QueryParser, ParsedQuery

# Load environment variables from .env if present
from sme_causal.core.config import AppConfig, get_config
from sme_causal.core.utils import (
    configure_logging,
    sanity_checks,
    parse_client_id_and_intent,
)

# To reset LLM client if provider changes in UI
from sme_causal.core.llm_clients.factory import reset_llm_client

# Локальные модули из вашего проекта
from sme_causal.data.synth_data import (
    FIELD_DOCS_RU,
)
from sme_causal.graph.graph_utils import (
    edges_to_digraph,
    export_graph,
)
from sme_causal.graph.graph_viz import build_pyvis_html
from sme_causal.graph.build_llm_graph import build_llm_graph
from sme_causal.graph.build_algo_graph import build_algo_graph
from sme_causal.graph.graph_utils import create_algo_edges
from sme_causal.graph.build_hybrid_graph import build_hybrid_graph
from sme_causal.graph.build_algo_llm_graph import build_algo_llm_graph


# Load configuration
cfg: AppConfig = get_config()

# Configure logging for Streamlit (file sink only to avoid UI duplicates)
configure_logging(cfg.streamlit_log_path, cfg.logging, add_stdout=False)


# -----------------------------
# CONSTANTS
# -----------------------------
API_KEY_ERROR = "Не задан OpenAI API key — задайте OPENAI_API_KEY в окружении или .env."

# Fields used to build per-client context for LLM explanations
# Берём из централизованных констант
CONTEXT_FIELDS: List[str] = CORE_CONTEXT_FIELDS

# Columns to display for selected client
DISPLAY_COLUMNS: List[str] = CONTEXT_FIELDS

# Node size/colors are handled in graph_viz utilities


# -----------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -----------------------------
def ensure_api_key() -> bool:
    """Check if OpenAI API key is available in environment or config.

    Returns:
        True if an API key is available in environment or config.
    """
    if not os.getenv("OPENAI_API_KEY") and cfg.effective_llm_api_key:
        # Fall back to config-provided key if env is empty
        os.environ["OPENAI_API_KEY"] = str(cfg.effective_llm_api_key)

    # base_url (для local обычно задан; для openai оставим None -> дефолт api.openai.com)
    base = cfg.effective_llm_base_url
    if base:
        os.environ["OPENAI_BASE_URL"] = base

    return bool(os.getenv("OPENAI_API_KEY"))


def run_if_api_key(work: Callable[[], None]) -> None:
    """Run a block only if API key is available; otherwise show error.

    Args:
        work: Zero-arg callable to execute if key is present.
    """
    if not ensure_api_key():
        st.error(API_KEY_ERROR)
        return
    try:
        work()
    except Exception as e:  # UI-friendly error display
        st.exception(e)


@st.cache_data(show_spinner=False)
def load_existing_data(version_ts: float) -> pd.DataFrame:
    path = cfg.synthetic_clients_path
    if not path.exists():
        raise FileNotFoundError(f"Synthetic data file not found: {path}")
    _ = version_ts  # touch cache key
    return pd.read_csv(path)


def render_explanation(expl, title: str | None = None) -> None:
    """Отрисовка объяснения в читаемом виде с fallback на JSON.
    expl — объект Explanation из agent_service (или совместимый словарь).
    """
    # Приведём к словарю
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
        # незнакомый формат — просто покажем как есть
        st.write(expl)
        return

    if title:
        st.markdown(f"#### {title}")

    has_struct = any(
        [
            data["diagnosis"],
            data["drivers_pos"],
            data["drivers_neg"],
            data["recommendations"],
            data["expected_effect"],
        ]
    )

    if has_struct:

        # Драйверы + / −
        col_p, col_n = st.columns(2)
        with col_p:
            st.markdown("**Драйверы роста ( + )**")
            if data["drivers_pos"]:
                st.markdown("\n".join([f"- ✅ {item}" for item in data["drivers_pos"]]))
            else:
                st.markdown("—")
        with col_n:
            st.markdown("**Сдерживающие факторы ( − )**")
            if data["drivers_neg"]:
                st.markdown("\n".join([f"- ⚠️ {item}" for item in data["drivers_neg"]]))
            else:
                st.markdown("—")

        # Рекомендации
        if data["recommendations"]:
            st.markdown("**Рекомендации**")
            st.markdown(
                "\n".join(
                    [
                        f"1. {rec}" if i == 0 else f"{i + 1}. {rec}"
                        for i, rec in enumerate(data["recommendations"])
                    ]
                )
            )
        else:
            st.markdown(" ")

        # Ожидаемый эффект
        if data["expected_effect"]:
            st.info(f"**Ожидаемый эффект:** {data['expected_effect']}")
    else:
        # Структуры нет — попробуем показать JSON красиво
        try:
            st.json(json.loads(data["raw_text"]))
        except Exception:
            st.write(data["raw_text"])


# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title=cfg.streamlit.page_title,
    page_icon=cfg.streamlit.page_icon,
    layout=cfg.streamlit.layout,
)

st.title(cfg.streamlit.page_title or "Demo")

with st.sidebar:
    st.header("Настройки")

    # --- Переключатель провайдера (openai | local)
    current_provider = cfg.effective_llm_provider
    default_idx = 0 if current_provider == "openai" else 1
    provider = st.selectbox("LLM Provider", ["openai", "local"], index=default_idx)

    if provider != current_provider:
        # переключаем провайдера «на лету»
        os.environ["LLM_PROVIDER"] = provider
        reset_llm_client()
        st.success(f"Provider switched to: {provider}")
        st.rerun()  # чтобы UI и конфиг пересобрались под новый провайдер

    model_name = st.text_input("Модель", value=cfg.llm.model_name)
    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        float(np.clip(cfg.llm.temperature, 0.0, 1.0)),
        0.05,
    )
    bootstrap_rounds = st.slider(
        "Bootstrap rounds", 1, 10, int(cfg.llm.bootstrap_rounds), 1
    )
    sample_rows = st.slider(
        "Сэмпл строк (для LLM)",
        200,
        2000,
        int(np.clip(cfg.llm.sample_rows, 200, 2000)),
        100,
    )
    min_conf = st.slider(
        "Порог доверия рёбер для визуализации и предсказаний",
        0.0,
        1.0,
        float(cfg.hybrid_graph.confidence_threshold),
        0.05,
    )


logger.info("Start loading data")
# --- Шаг 1: Данные
st.markdown("### 1) Датасет")
try:
    with st.spinner("Загрузка датасета..."):
        df = load_existing_data(cfg.synthetic_clients_path.stat().st_mtime)
    st.success(f"Датасет загружен: {len(df)} строк")
except FileNotFoundError as e:
    st.error(f"Файл датасета не найден: {e}")
    st.stop()

# --- Шаг 2: LLM → рёбра
st.markdown("### 2) Построение причинно‑следственного графа")
method = st.radio(
    "Метод построения графа:",
    ["LLM", "Algo", "Algo_LLM_validation", "Hybrid"],
    horizontal=True,
    help="LLM: использует языковую модель. "
    "Algo: использует статистические алгоритмы. "
    "Algo_LLM_validation: использует LLM валидацию для алго-графа. "
    "Hybrid: соединяет результаты двух подходов вместе с помощью LLM-as-a-judge",
)
run_button = st.button("Запустить построение графа")

edges: List[Dict] | None = None
if run_button:
    if method != "Algo" and not ensure_api_key():
        st.error(API_KEY_ERROR)
    else:
        with st.spinner("Построение графа, подождите..."):
            try:
                if method == "Hybrid":
                    edges = build_hybrid_graph(
                        df=df,
                        llm_weight=0.5,
                        confidence_threshold=float(min_conf),
                        model_name=model_name,
                        temperature=float(temperature),
                        bootstrap_rounds=int(bootstrap_rounds),
                        sample_rows=int(sample_rows),
                    )
                    st.success(f"Готово: гибридный граф содержит {len(edges)} рёбер")
                elif method == "LLM":
                    edges, graph = build_llm_graph(
                        df=df,
                        model_name=model_name,
                        temperature=float(temperature),
                        bootstrap_rounds=int(bootstrap_rounds),
                        sample_rows=int(sample_rows),
                    )
                    st.success(f"Готово: конструктор графа вернул {len(edges)} рёбер")
                elif method == "Algo":
                    consensus, _ = build_algo_graph(csv_path=cfg.synthetic_clients_path)
                    edges = create_algo_edges(consensus)
                    st.success(f"Готово: Алгоритм вернул {len(edges)} рёбер")

                elif method == "Algo_LLM_validation":
                    consensus = build_algo_llm_graph(
                        csv_path=cfg.synthetic_clients_path
                    )
                    edges = create_algo_edges(consensus)
                    st.success(f"Готово: Алгоритм вернул {len(edges)} рёбер")

            except Exception as e:
                st.exception(e)

# Если уже есть артефакт, предложим подгрузить
if edges is None:
    if method == "Hybrid":
        f = cfg.full_artifacts_dir / "hybrid_edges.json"
        if f.exists():
            edges = json.loads(f.read_text(encoding="utf-8"))
            st.info(f"Загружены сохранённые гибридные рёбра: {len(edges)}")
    elif method == "LLM":
        f = cfg.llm_edges_path
        if f.exists():
            edges = json.loads(f.read_text(encoding="utf-8"))
            st.info(f"Загружены сохранённые LLM рёбра: {len(edges)}")
    elif method == "Algo":
        f = cfg.algo_edges_path
        if f.exists():
            consensus = json.loads(f.read_text(encoding="utf-8"))["edges"]
            edges = create_algo_edges(consensus)
            st.info(f"Загружены сохранённые Algo-рёбра: {len(edges)}")
    elif method == "Algo_LLM_validation":
        f = cfg.full_algorithmic_dir / "algo_llm_edges.json"
        if f.exists():
            consensus = json.loads(f.read_text(encoding="utf-8"))
            edges = create_algo_edges(consensus)
            st.info(f"Загружены сохранённые Algo_LLM рёбра: {len(edges)}")

# --- Шаг 3: Визуализация графа
if edges:
    st.markdown("### 3) Визуализация графа")
    # фильтр по порогу доверия
    filtered = [e for e in edges if float(e.get("confidence", 0.0)) >= float(min_conf)]
    st.caption(
        f"Порог: {min_conf:.2f}. Отфильтровано: {len(filtered)} рёбер из {len(edges)}"
    )

    G = edges_to_digraph(filtered)
    # Сохраним артефакты (не обязательно, но удобно)
    default_graph_prefix = getattr(cfg.paths, "graph_prefix", "graph_merged")
    export_graph(G, out_prefix=str(cfg.full_artifacts_dir / default_graph_prefix))

    html = build_pyvis_html(G, height_px=650, directed=True, physics=True)
    st.components.v1.html(html, height=670, scrolling=True)


# --- Шаг 4: Мини‑объяснение по клиенту (опционально)
st.markdown("### 4) Объяснение для выбранного клиента (LLM)")
col_c1, col_c2 = st.columns([1.2, 1.8])

# Готовим агент (если есть ключ); держим в сессии чтобы не пересоздавать
agent: CausalAgent | None = None
check_api_key_if_need = 1 if method == "Algo" else ensure_api_key()

method_mapping = {
    "LLM": "llm",
    "Algorithmic": "algo",
    "Algo_LLM_validation": "algo_llm",
    "Hybrid": "hybrid",
}

if check_api_key_if_need:
    graph_method_key = method_mapping.get(method, "llm")

    desired_agent_cfg = (model_name, float(temperature), graph_method_key)
    if (
        "agent" not in st.session_state
        or st.session_state.get("agent_cfg") != desired_agent_cfg
    ):
        st.session_state["agent"] = CausalAgent(
            model=model_name,
            temperature=float(temperature),
            graph_method=graph_method_key,
        )
        st.session_state["agent_cfg"] = desired_agent_cfg
    agent = st.session_state["agent"]


# Initialize client_id to avoid scope issues
client_id = None


# Helper to run intervention scenarios and render results
def _run_intervention(
    delta: Dict,
    label: str,
    info_text: str | None = None,
    *,
    use_psm: bool = False,
    use_graph: bool = False,
    use_rag: bool = False,
    rag_query_text: Optional[str] = None,
    match_info: Optional[Dict] = None,
    target_metric: Optional[str] = None,
) -> None:
    """Запуск интервенции: LLM-объяснение + (опционально) PSM-оценка ATT/ATE."""

    def _work() -> None:
        ctx = agent.build_context_for_client(df, client_id)

        # Проверки запускаем только если delta не пустая
        if delta:
            checks = sanity_checks(ctx, delta)
            if checks["blocked"]:
                # UI: кратко описываем сценарий и причины блокировки
                st.success(f"{label} Analysis Complete")
                st.markdown("**Intervention Scenario:**")
                st.info(
                    info_text or "\n".join([f"• {k} → {v}" for k, v in delta.items()])
                )

                st.markdown("**Rule-based decision (до LLM/PSM):**")
                st.warning("Интервенция не даёт аплифта по детерминированным правилам:")
                st.markdown("\n".join([f"- {r}" for r in checks["reasons"]]))
                st.info("**Ожидаемый эффект:** 0.0")

                # (Опционально) показать структурированный «псевдо-ответ», чтобы UI был единообразен
                expl_dict = {
                    "diagnosis": "Интервенция конфликтует с текущими условиями клиента.",
                    "drivers_pos": [],
                    "drivers_neg": checks["reasons"],
                    "recommendations": [
                        "Изменить параметры предложения или выбрать другую интервенцию."
                    ],
                    "expected_effect": "0.0",
                    "raw_text": json.dumps({"checks": checks}, ensure_ascii=False),
                }
                st.markdown("**Analysis Results (rule-based):**")
                render_explanation(expl_dict)

                # PSM и LLM не запускаем
                return

        psm_metrics_for_llm: Dict[str, object] | None = None
        psm_warning: str | None = None
        psm_exception: Exception | None = None
        psm_caption: str | None = None
        psm_display: Dict[str, object] | None = None

        if use_psm and delta:
            treatment_col = next((k for k in delta.keys() if k in df.columns), None)
            outcome_col = (
                target_metric
                if target_metric and target_metric in df.columns
                else "Revenue_Growth_Rate"
            )
            if target_metric and target_metric not in df.columns:
                st.warning(
                    f"Целевая метрика '{target_metric}' не найдена в датасете. Используется метрика по умолчанию: 'Revenue_Growth_Rate'."
                )
            covariates = [
                "Industry",
                "Region",
                "Business_Size",
                "Avg_Account_Balance",
                "Avg_Monthly_Inflow",
                "Avg_Monthly_Outflow",
                "Num_Products",
            ]
            # берутся автоматически в зависимости от treatment/outcome

            if treatment_col is None:
                psm_warning = "PSM: не смог определить treatment-колонку из delta (ключи отсутствуют в df)."
            elif outcome_col not in df.columns:
                psm_warning = f"PSM: нет outcome-колонки '{outcome_col}' в df."
            else:
                covariates = [c for c in covariates if c in df.columns]
                if not covariates:
                    psm_warning = "PSM: список ковариат пуст (ни одной из ожидаемых колонок нет в df)."
                else:
                    thr = None
                    s = df[treatment_col]
                    if not s.dropna().isin([0, 1]).all():
                        if pd.api.types.is_numeric_dtype(s):
                            try:
                                # thr = float(
                                #     np.nanpercentile(
                                #         pd.to_numeric(s, errors="coerce"), 75
                                #     )
                                # )
                                thr = 0.01
                                # psm_caption = (
                                #     f"PSM: авто-порог для бинаризации '{treatment_col}' = {thr:.4f}"
                                #     " (75-й перцентиль)"
                                # )
                            except Exception:
                                thr = None
                        if thr is None:
                            psm_warning = (
                                f"PSM: колонка '{treatment_col}' не бинарная и не числовая — "
                                "задай явный threshold/бинаризацию."
                            )

                    if psm_warning is None:
                        try:
                            caliper = 0.1 if "Product" not in treatment_col else 1
                            analyzer = CausalInferenceAnalyzer(
                                # covariates=covariates,
                                target=outcome_col,
                                treatment_variable=treatment_col,
                                threshold=thr,
                                replacement=False,
                                caliper=caliper,
                            )
                            res = analyzer.run(df)

                            att = getattr(res, "ate", None)
                            ate = getattr(res, "ate_naive", None)
                            n_pairs = getattr(res, "n_pairs", None)
                            matched_df = getattr(res, "matched_df", None)

                            n_treated = n_control = None
                            treated_col_name = getattr(
                                analyzer, "_treated", "__treated__"
                            )
                            if (
                                isinstance(matched_df, pd.DataFrame)
                                and not matched_df.empty
                                and treated_col_name in matched_df.columns
                            ):
                                n_treated = int(
                                    (matched_df[treated_col_name] == 1).sum()
                                )
                                n_control = int(
                                    (matched_df[treated_col_name] == 0).sum()
                                )

                            psm_metrics_for_llm = {
                                "att": att,
                                "ate": ate,
                                "n_pairs": n_pairs,
                                "n_treated": n_treated,
                                "n_control": n_control,
                                "treatment_col": treatment_col,
                                "outcome_col": outcome_col,
                                "threshold": thr,
                                "caliper": caliper,
                                "covariates": covariates,
                            }
                            psm_display = {
                                "treatment_col": treatment_col,
                                "outcome_col": outcome_col,
                                "threshold": thr,
                                "att": att,
                                "ate": ate,
                                "n_pairs": n_pairs,
                                "n_treated": n_treated,
                                "n_control": n_control,
                            }
                        except Exception as exc:  # noqa: BLE001
                            psm_exception = exc

        expl = agent.explain_what_if(
            ctx,
            delta,
            psm_metrics=psm_metrics_for_llm,
            use_graph=use_graph,
            use_rag=use_rag,
            min_conf=min_conf,
            rag_query_text=rag_query_text,
            match_info=match_info,
            target_metric=target_metric,
        )
        st.success(f"{label} Analysis Complete")
        st.markdown("**Intervention Scenario:**")
        if info_text:
            st.info(info_text)
        elif delta:
            st.info("\n".join([f"• {k} → {v}" for k, v in delta.items()]))
        else:
            st.info(f"Анализ по общему запросу: '{rag_query_text}'")

        if match_info:
            similar_matches_ui = []
            for col, info in match_info.items():
                if info.get("status") == "similar":
                    similar_matches_ui.append(
                        f"`{col}` (из запроса «{info.get('query_phrase', '...')}»)"
                    )
            if similar_matches_ui:
                st.warning(
                    f"**Внимание:** Следующие параметры интервенции были подобраны по схожести, а не по точному совпадению: {', '.join(similar_matches_ui)}. Результаты могут быть менее точными."
                )

        st.markdown("**Analysis Results (LLM):**")
        render_explanation(expl)

        # ---- 2) (опц.) PSM — количественная оценка эффекта ----
        if not use_psm:
            return
        st.markdown("**PSM Result:**")
        if psm_caption:
            st.caption(psm_caption)

        if psm_exception is not None:
            st.exception(psm_exception)
            return

        if psm_warning:
            st.warning(psm_warning)
            return

        if not psm_display:
            st.info("PSM: результаты недоступны.")
            return

        def _fmt(val):
            if val is None:
                return "—"
            try:
                num = float(val)
            except (TypeError, ValueError):
                return str(val)
            if np.isnan(num):
                return "NaN"
            return f"{num:.4f}"

        att_val = psm_display.get("att")
        ate_val = psm_display.get("ate")

        def _is_nan(val) -> bool:
            try:
                return np.isnan(float(val))
            except (TypeError, ValueError):
                return False

        if att_val is None or _is_nan(att_val):
            st.info(
                "Не удалось сформировать пары под ограничениями caliper — "
                f"ATT (матчинг) = NaN. ATE (наивный) = {_fmt(ate_val)}."
            )
            return

        st.info(
            f"Treatment: {psm_display.get('treatment_col')} | "
            f"Outcome: {psm_display.get('outcome_col')} | "
            f"ATT = {_fmt(att_val)} | "
            f"ATE = {_fmt(ate_val)} | "
            f"matched pairs = {psm_display.get('n_pairs') if psm_display.get('n_pairs') is not None else '—'} | "
            f"treated={psm_display.get('n_treated') if psm_display.get('n_treated') is not None else '—'} "
            f"control={psm_display.get('n_control') if psm_display.get('n_control') is not None else '—'}"
        )

    run_if_api_key(_work)


with col_c1:
    # фильтры для удобства
    industry_filter = st.selectbox(
        INDUSTRY, options=["(all)"] + sorted(df[INDUSTRY].unique().tolist())
    )
    region_filter = st.selectbox(
        REGION, options=["(all)"] + sorted(df[REGION].unique().tolist())
    )
    dff = df.copy()
    if industry_filter != "(all)":
        dff = dff[dff[INDUSTRY] == industry_filter]
    if region_filter != "(all)":
        dff = dff[dff[REGION] == region_filter]
    # список клиентов (не перегружаем UI)
    ids = dff[CLIENT_ID].head(500).tolist()
    if not ids:
        st.warning("Нет клиентов под выбранные фильтры.")
    else:
        client_id = st.selectbox("Client_ID (top‑500)", options=ids)

with col_c2:
    if ids:
        cols_show = DISPLAY_COLUMNS
        st.dataframe(
            dff[dff[CLIENT_ID] == client_id][
                [CLIENT_ID] + [c for c in cols_show if c in dff.columns]
            ],
            width="stretch",
            height=220,
        )

st.markdown("---")
st.markdown("#### Настройки анализа:")

use_psm_flag = st.checkbox("Включить PSM-оценку эффекта", value=False)
use_graph_flag = st.checkbox("Использовать причинно-следственный граф", value=False)
use_rag_flag = st.checkbox(
    "Включить использование rag-контекста для модели", value=False
)
st.markdown("---")

col_e1, col_e2, col_e3 = st.columns([1, 1, 1])
with col_e1:
    run_explain = st.button("Объяснение ситуации клиента")
with col_e2:
    pass
with col_e3:
    pass

st.markdown("---")

# Second row for the third button
col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
with col_f1:
    run_product_offer = st.button("📦 Offer acquiring product")
with col_f2:
    run_credit_limit = st.button("💳 Increase credit limit by +15%")
with col_f3:
    run_discount = st.button("📢 Apply discounted tariff")

# st.markdown("Или сформулируйте ваш запрос в свободной форме:")
open_query_text = st.text_area(
    "Или сформулируйте ваш запрос в свободной форме:",
    placeholder="Например: 'Как отразится введение эквайринга на выручку клиента C000482?' "
    "или 'Подбери лучшую интервенцию для роста среднего баланса клиента C000501.'",
    height=100,
    # label_visibility="collapsed",
)
run_open_query = st.button("🚀 Выполнить открытый запрос")

st.markdown("---")
st.markdown("#### Результаты анализа:")

# --- Baseline explanation
if agent and CLIENT_ID in df.columns and run_explain and client_id is not None:

    def _work_explain() -> None:
        ctx = agent.build_context_for_client(df, client_id)
        expl = agent.explain_client(ctx, use_graph=use_graph_flag, min_conf=min_conf)
        st.success("Готово")
        render_explanation(expl, title="Analysis Results")

    run_if_api_key(_work_explain)

# --- Product Offer Intervention
if agent and CLIENT_ID in df.columns and run_product_offer and client_id is not None:
    _run_intervention(
        {NEW_PRODUCT_OFFER: 1, NEW_PRODUCT_OFFER_TYPE: "acquiring"},
        label="Offer acquiring product",
        info_text="• Enable acquiring product offer",
        use_psm=use_psm_flag,
        use_graph=use_graph_flag,
        use_rag=use_rag_flag,
    )

# --- Credit Limit Intervention
if agent and CLIENT_ID in df.columns and run_credit_limit and client_id is not None:

    def _credit_delta() -> Dict:
        ctx = agent.build_context_for_client(df, client_id)
        new_limit = max(float(ctx.get(CREDIT_LIMIT_CHANGE, 0.0)), 15.0)
        return {CREDIT_LIMIT_CHANGE: new_limit}

    _run_intervention(
        _credit_delta(),
        label="Increase credit limit by +15%",
        info_text="• Raise credit limit by +15%",
        use_psm=use_psm_flag,
        use_graph=use_graph_flag,
        use_rag=use_rag_flag,
    )

# --- Discount Intervention
if agent and CLIENT_ID in df.columns and run_discount and client_id is not None:
    _run_intervention(
        {TARIFF_DISCOUNT: 1},
        label="Apply discounted tariff",
        info_text="• Apply discounted tariff",
        use_psm=use_psm_flag,
        use_graph=use_graph_flag,
        use_rag=use_rag_flag,
    )


if run_open_query and open_query_text:
    if not agent:
        st.error("Агент не инициализирован. Проверьте API ключ и настройки выше.")
    else:
        # Парсинг client_id ожидает формат CXXXXXX, где XXXXXX - шестизначное число-индентификатор
        explicit_client_id, cleaned_query_text = parse_client_id_and_intent(
            open_query_text
        )

        final_client_id_to_use = None
        if explicit_client_id:
            if explicit_client_id not in df[CLIENT_ID].values:
                st.error(
                    f"Клиент с ID '{explicit_client_id}', указанный в запросе, не найден в датасете."
                )
                st.stop()
            final_client_id_to_use = explicit_client_id
            st.info(
                f"Обнаружен ID клиента в запросе: {final_client_id_to_use}. Анализ будет проведен для него."
            )
        elif client_id:
            final_client_id_to_use = client_id
            st.info(
                f"ID клиента не найден в запросе. Используется выбранный в выпадающем списке клиент: {final_client_id_to_use}."
            )

        if not final_client_id_to_use:
            st.error(
                "Не удалось определить ID клиента. Пожалуйста, выберите клиента из списка или укажите его в запросе (например, C123456)."
            )
            st.stop()

        client_id = final_client_id_to_use

        with st.spinner("Анализирую ваш запрос..."):
            parser = QueryParser(model=model_name, temperature=0.0)
            parsed_data = parser.parse(cleaned_query_text)

        if not parsed_data:
            st.error(
                "Не удалось разобрать запрос. Пожалуйста, попробуйте переформулировать его."
            )
        else:
            action_type = parsed_data.action_type
            delta = parsed_data.delta
            label = parsed_data.label
            info_text = parsed_data.info_text
            match_info = parsed_data.match_info
            target_metric = parsed_data.target_metric

            if action_type == "what_if":
                if not delta:
                    st.info(
                        "Не удалось определить в запросе конкретную интервенцию из диапазона возможных. Выполняю общий анализ по вашему запросу..."
                    )
                    label = "Общий анализ по запросу"
                    info_text = f"Анализ по общему запросу: '{cleaned_query_text}'"

                _run_intervention(
                    delta=delta,
                    label=label,
                    info_text=info_text,
                    use_psm=use_psm_flag,
                    use_graph=use_graph_flag,
                    use_rag=use_rag_flag,
                    rag_query_text=cleaned_query_text,
                    match_info=match_info,
                    target_metric=target_metric,
                )

            elif action_type == "optimize":
                st.info(
                    f"Запрос на оптимизацию для клиента {client_id}. Выполняю базовый анализ с рекомендациями..."
                )

                def _work_optimize():
                    ctx = agent.build_context_for_client(df, client_id)
                    expl = agent.explain_client(
                        ctx,
                        use_graph=use_graph_flag,
                        min_conf=min_conf,
                        target_metric=target_metric,
                    )
                    st.success("Готово")
                    render_explanation(
                        expl, title=f"Анализ и рекомендации для клиента {client_id}"
                    )

                run_if_api_key(_work_optimize)

            else:
                st.error(
                    f"Неизвестный тип действия '{action_type}'. Пожалуйста, переформулируйте запрос."
                )

st.markdown("---")
st.caption("Демо UI.")
