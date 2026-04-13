import sys
import argparse
import json as _json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from loguru import logger

# Load environment variables from .env if present
# from sme_causal.core import env  # noqa: F401
from sme_causal.agent.agent_service import CausalAgent, Explanation, QueryParser, ParsedQuery
from sme_causal.core.config import get_config
from sme_causal.core.utils import configure_logging, sanity_checks, parse_client_id_and_intent
from sme_causal.core.columns import (
    CLIENT_ID,
    NEW_PRODUCT_OFFER,
    NEW_PRODUCT_OFFER_TYPE,
)
from sme_causal.inference.psm import CausalInferenceAnalyzer

PSM_MIN_GROUP_SIZE = 100


def _parse_kv_pairs(text: str) -> Dict[str, object]:
    """Parse simple comma-separated key=value pairs into a dict.

    Examples:
        "New_Product_Offer=1,New_Product_Offer_Type=acquiring"
        "Credit_Limit_Change=12.5,Targeted_Communication=true"
    """

    def _coerce(val: str):
        v = val.strip()
        low = v.lower()
        if low in {"true", "false"}:
            return low == "true"
        try:
            if "." in v:
                return float(v)
            return int(v)
        except Exception:
            return v

    out: Dict[str, object] = {}
    if not text:
        return out
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            logger.warning(f"Ignoring malformed pair: '{part}' (expected key=value)")
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = _coerce(v)
    return out


def _is_finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _psm_reliability(
    *,
    att: object,
    n_treated: Optional[int],
    n_control: Optional[int],
    min_group_size: int = PSM_MIN_GROUP_SIZE,
) -> Dict[str, object]:
    if not _is_finite_number(att):
        return {
            "psm_reliable": False,
            "psm_reason": "ATT is unavailable; naive ATE must not be used as the primary personal effect.",
        }

    if n_treated is None or n_control is None:
        return {
            "psm_reliable": False,
            "psm_reason": "Matched sample sizes are unavailable.",
        }

    if n_treated < min_group_size or n_control < min_group_size:
        return {
            "psm_reliable": False,
            "psm_reason": (
                f"Matched sample is too small: n_treated={n_treated}, "
                f"n_control={n_control}, required>={min_group_size}."
            ),
        }

    return {
        "psm_reliable": True,
        "psm_reason": (
            f"Matched sample is large enough: n_treated={n_treated}, "
            f"n_control={n_control}."
        ),
    }


def _run_psm(
    df: pd.DataFrame,
    intervention_delta: Dict[str, object],
    *,
    outcome_col: str = "Revenue_Growth_Rate",
    treatment_col: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    caliper: float = 0.05,
    match_ratio: int = 1,
    threshold: Optional[float] = None,
) -> Dict[str, object]:
    """
    Выполняет расчет ATT (matched) и ATE (наивный) через CausalInferenceAnalyzer.run(df).
    Возвращает dict с метриками/диагностикой.

    Примечание:
    - В твоём классе нет аргумента match_ratio. Здесь мы трактуем:
      match_ratio == 1  -> replacement = False (1:1 без возврата)
      match_ratio != 1  -> replacement = True  (с возвратом)
    """
    # 1) Input checks
    if outcome_col not in df.columns:
        return {"ok": False, "error": f"Outcome column '{outcome_col}' not found"}

    # If treatment not set, take first key from delta
    if treatment_col is None:
        treatment_col = next(
            (k for k in intervention_delta.keys() if k in df.columns), None
        )
    if treatment_col is None:
        return {"ok": False, "error": "Cannot infer treatment column from delta keys"}

    # 2) Threshold for non-binary treatment
    s = df[treatment_col]
    if not s.dropna().isin([0, 1]).all():
        if pd.api.types.is_numeric_dtype(s):
            if threshold is None:
                threshold = float(
                    np.nanpercentile(pd.to_numeric(s, errors="coerce"), 75)
                )
                logger.info(
                    f"PSM: auto-threshold for '{treatment_col}' = {threshold:.4f}"
                )
        else:
            return {
                "ok": False,
                "error": f"Treatment '{treatment_col}' is non-binary and non-numeric. Provide explicit threshold or pre-binarize.",
            }

    # 3) Initialize and run analyzer
    try:
        analyzer = CausalInferenceAnalyzer(
            target=outcome_col,
            treatment_variable=treatment_col,
            covariates=covariates,
            threshold=threshold,  # None если бинарный; число если автобинаризация
            replacement=(match_ratio != 1),  # mapping см. примечание в докстроке
            caliper=caliper,
            # logreg_max_iter оставить по умолчанию (1000)
        )
        res = analyzer.run(df)
    except Exception as e:
        return {"ok": False, "error": f"PSM failed: {e}"}

    # 4) Collect metrics
    att = getattr(res, "ate", None)
    ate = getattr(res, "ate_naive", None)
    n_pairs = getattr(res, "n_pairs", None)
    matched_df = getattr(res, "matched_df", None)

    n_treated = n_control = None
    if isinstance(matched_df, pd.DataFrame) and not matched_df.empty:
        treated_col_name = "__treated__"  # внутреннее имя из класса
        if treated_col_name in matched_df.columns:
            n_treated = int((matched_df[treated_col_name] == 1).sum())
            n_control = int((matched_df[treated_col_name] == 0).sum())

    if n_pairs == 0:
        n_treated = n_treated or 0
        n_control = n_control or 0

    reliability = _psm_reliability(
        att=att,
        n_treated=n_treated,
        n_control=n_control,
    )

    return {
        "ok": True,
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "covariates": covariates,
        "threshold": threshold,
        "caliper": caliper,
        "match_ratio": match_ratio,  # для протокола (см. mapping)
        "replacement": (match_ratio != 1),
        "att": att,
        "ate": ate,
        "n_pairs": n_pairs,
        "n_treated": n_treated,
        "n_control": n_control,
        **reliability,
    }


def main() -> None:
    cfg = get_config()

    # CLI arguments
    parser = argparse.ArgumentParser(description="Explain SME client and run what-if or natural language queries")

    # Client identification
    parser.add_argument(
        "--client-id", type=str, default=None, help="Client_ID to analyze (overrides ID found in query)"
    )

    # Mode 1: Explicit Key-Value pairs
    parser.add_argument(
        "--what-if",
        type=str,
        default=None,
        help="Comma-separated key=value pairs for scenario changes (e.g. 'New_Product_Offer=1')",
    )

    # Mode 2: Natural Language Query
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Natural language query (e.g. 'Should we offer acquiring to C000123?').",
    )

    # Output format
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON to stdout (suppress info logs)",
    )
    parser.add_argument(
        "--debug-json",
        action="store_true",
        help="Print JSON with raw LLM responses, prompts, and source contexts",
    )

    # Graph & Method settings
    parser.add_argument(
        "--graph-method",
        choices=["llm", "algo", "hybrid", "algo_llm"],
        default="llm",
        help="Graph construction method: llm, algo, algo_llm or hybrid",
    )

    # Graph usage flag
    parser.add_argument(
        "--use_graph", action="store_true", help="Use causal graph in prompts"
    )

    # RAG flag
    parser.add_argument(
        "--use_rag",
        action="store_true",
        help="Use RAG to enrich context (documents)",
    )

    # PSM settings
    parser.add_argument(
        "--use-psm",
        action="store_true",
        help="Run PSM ATT/ATE estimation in addition to LLM what-if",
    )
    parser.add_argument(
        "--outcome-col",
        type=str,
        default="Revenue_Growth_Rate",
        help="Default outcome column for PSM (if not extracted from query)",
    )
    parser.add_argument(
        "--treatment-col",
        type=str,
        default=None,
        help="Treatment column for PSM (default: infer from delta)",
    )
    parser.add_argument(
        "--covariates",
        type=str,
        default="",
        help="Comma-separated covariates (default: heuristic set)",
    )
    parser.add_argument("--psm-caliper", type=float, default=0.05, help="PSM caliper")
    parser.add_argument(
        "--psm-match-ratio", type=int, default=1, help="PSM match ratio"
    )
    parser.add_argument(
        "--psm-threshold",
        type=float,
        default=None,
        help="Binarization threshold for non-binary treatment",
    )

    args = parser.parse_args()
    if args.debug_json:
        args.json = True

    # Configure logging for CLI run
    configure_logging(
        cfg.pipeline_log_path,
        cfg.logging,
        add_stdout=not args.json,  # keep stdout clean when JSON requested
        stdout_format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    # 1. Load Data
    csv_path: Path = cfg.synthetic_clients_path
    if not csv_path.exists():
        logger.error(
            f"Dataset not found at {csv_path}. Generate it via 'python main.py' or Streamlit UI."
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded synthetic clients data from {csv_path} (rows={len(df)})")

    # 2. Check OpenAI API Key
    if (args.graph_method != "algo" or args.query) and not cfg.effective_openai_api_key:
        logger.error(
            "No OpenAI API key found. Set OPENAI_API_KEY in your environment or .env file."
        )
        sys.exit(2)

    # 3. Initialize Agent
    agent = CausalAgent(
        model=cfg.llm.model_name,
        temperature=cfg.llm.temperature,
        seed=cfg.data_generation.seed,
        graph_method=args.graph_method,
    )
    logger.info(
        f"CausalAgent initialized: model={cfg.llm.model_name}, temp={cfg.llm.temperature}, seed={cfg.data_generation.seed}"
    )

    # 4. Parse Inputs (Query vs Explicit args)

    # Defaults
    delta: Dict[str, object] = {}
    match_info: Optional[Dict] = None
    target_metric: Optional[str] = args.outcome_col
    action_type: str = "what_if"
    rag_query_text: Optional[str] = None
    query_analysis_label: str = ""

    # Determine Client ID logic variables
    extracted_client_id = None
    cleaned_query = ""

    if args.query:
        logger.info(f"Processing natural language query: '{args.query}'")

        # A) Extract Client ID from text
        extracted_client_id, cleaned_query = parse_client_id_and_intent(args.query)

        # B) Parse Intent via LLM
        query_parser = QueryParser(model=cfg.llm.model_name, temperature=0.0)
        parsed_data: Optional[ParsedQuery] = query_parser.parse(cleaned_query)

        if parsed_data:
            action_type = parsed_data.action_type
            delta = parsed_data.delta
            match_info = parsed_data.match_info
            query_analysis_label = parsed_data.label
            rag_query_text = cleaned_query # Use text for RAG if delta is ambiguous

            # Update target metric if found in query, otherwise keep CLI default
            if parsed_data.target_metric:
                target_metric = parsed_data.target_metric
                logger.info(f"Target metric detected in query: {target_metric}")
        else:
            logger.warning("Failed to parse query intent. Falling back to basic context.")
            rag_query_text = cleaned_query

    elif args.what_if:
        # Explicit what-if overrides
        delta = _parse_kv_pairs(args.what_if)
        logger.info(f"Using explicit what-if parameters: {delta}")
    else:
        # Default fallback
        delta = {NEW_PRODUCT_OFFER: 1, NEW_PRODUCT_OFFER_TYPE: "acquiring"}
        logger.info("No query or --what-if provided, using default: new acquiring offer")

    # 5. Resolve Client ID
    # CLI arg takes precedence over extracted ID
    client_id = args.client_id or extracted_client_id

    if not client_id:
        # Fallback to first client if absolutely nothing is provided
        client_id = df[CLIENT_ID].iloc[0]
        logger.warning(f"No Client ID provided or found in query. Using first available: {client_id}")

    if client_id not in set(df[CLIENT_ID].tolist()):
        logger.error(f"Client_ID not found in dataset: {client_id}")
        sys.exit(3)

    # 6. Build Context & Run Sanity Checks
    ctx = agent.build_context_for_client(df, client_id)

    # Checks only apply if we have a concrete delta to check
    checks_result = {"blocked": False, "reasons": []}
    if delta:
        checks_result = sanity_checks(ctx, delta)

    # If blocked by sanity checks
    if checks_result["blocked"]:
        msg = "Intervention blocked by rule-based sanity checks. It has conflict with currect client features."
        if args.json:
            payload = {
                "client_id": client_id,
                "action_type": action_type,
                "context": ctx,
                "what_if": {
                    "delta": delta,
                    "explanation": {
                        "summary": msg,
                        "reasons": checks_result["reasons"],
                        "uplift": 0.0,
                    },
                },
            }
            print(_json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))
        else:
            logger.info("--------------------------------")
            logger.info(f"What-if scenario: {delta}")
            logger.warning(msg)
            for r in checks_result["reasons"]:
                logger.info("• %s", r)
            logger.info("=> Expected Uplift: 0.0")
        return

    # 7. Execution Logic based on Action Type

    # A) If 'optimize' -> Just explain client situation with specific target metric
    if action_type == "optimize":
        logger.info(f"Action is 'optimize' for target '{target_metric}'. Running client diagnosis...")
        expl: Explanation = agent.explain_client(
            ctx,
            use_graph=args.use_graph,
            target_metric=target_metric
        )

        if args.json:
            payload = {
                "client_id": client_id,
                "action_type": "optimize",
                "target_metric": target_metric,
                "context": ctx,
                "analysis": expl.to_dict(include_debug=args.debug_json)
            }
            print(_json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))
        else:
            logger.info(f"Optimization Analysis for {client_id} ({target_metric}):")
            logger.info(expl)
        return

    # B) If 'what_if' (default) -> Run Baseline + Scenario Analysis + PSM

    # Baseline Diagnosis
    expl_baseline: Explanation = agent.explain_client(
        ctx,
        use_graph=args.use_graph,
        target_metric=target_metric
    )

    # PSM Calculation (Optional)
    psm_result: Optional[Dict[str, object]] = None
    psm_metrics_for_llm: Optional[Dict[str, object]] = None

    if args.use_psm and delta:
        user_covariates = None
        if args.covariates.strip():
            user_covariates = [
                c.strip() for c in args.covariates.split(",") if c.strip()
            ]

        logger.info(f"Running PSM for target: {target_metric}")
        psm_result = _run_psm(
            df,
            delta,
            outcome_col=target_metric,  # Use parsed or CLI metric
            treatment_col=args.treatment_col,
            covariates=user_covariates,
            caliper=args.psm_caliper,
            match_ratio=args.psm_match_ratio,
            threshold=args.psm_threshold,
        )

        if psm_result.get("ok"):
            psm_metrics_for_llm = {
                "att": psm_result.get("att"),
                "ate": psm_result.get("ate"),
                "n_pairs": psm_result.get("n_pairs"),
                "n_treated": psm_result.get("n_treated"),
                "n_control": psm_result.get("n_control"),
                "treatment_col": psm_result.get("treatment_col"),
                "outcome_col": psm_result.get("outcome_col"),
                "threshold": psm_result.get("threshold"),
                "caliper": psm_result.get("caliper"),
                "covariates": psm_result.get("covariates"),
                "psm_reliable": psm_result.get("psm_reliable"),
                "psm_reason": psm_result.get("psm_reason"),
            }

    # What-If Explanation
    what_if_expl: Explanation = agent.explain_what_if(
        ctx,
        delta,
        psm_metrics=psm_metrics_for_llm,
        use_graph=args.use_graph,
        use_rag=args.use_rag,
        rag_query_text=rag_query_text,
        match_info=match_info,
        target_metric=target_metric
    )

    # 8. Output Results
    if args.json:
        payload = {
            "client_id": client_id,
            "action_type": "what_if",
            "query_parsed": {
                "label": query_analysis_label,
                "match_info": match_info,
                "detected_target": target_metric
            } if args.query else None,
            "context": ctx,
            "baseline_diagnosis": expl_baseline.to_dict(include_debug=args.debug_json),
            "what_if": {
                "delta": delta,
                "analysis": what_if_expl.to_dict(include_debug=args.debug_json),
                "psm_stats": psm_result if psm_result and psm_result.get("ok") else None
            },
        }
        print(_json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))
    else:
        logger.info(f"=== Analysis for {client_id} ===")
        if query_analysis_label:
            logger.info(f"Query Intent: {query_analysis_label}")
        if match_info:
             logger.info(f"Param matching info: {match_info}")

        logger.info("--- Baseline Diagnosis ---")
        logger.info(expl_baseline)

        logger.info("--- What-if Scenario ---")
        logger.info(f"Delta: {delta}")
        if not delta and rag_query_text:
            logger.info(f"General Context: {rag_query_text}")

        logger.info(what_if_expl)

        if psm_result is not None:
            if psm_result.get("ok"):
                logger.info("--- PSM Results ---")
                logger.info(
                    f"Target: {psm_result.get('outcome_col')} | "
                    f"ATT={psm_result.get('att')} | "
                    f"ATE={psm_result.get('ate')} | "
                    f"matched_pairs={psm_result.get('n_pairs')}"
                )
            else:
                logger.warning(f"PSM skipped/failed: {psm_result.get('error')}")


if __name__ == "__main__":
    main()
