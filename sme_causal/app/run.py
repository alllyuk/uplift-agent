import sys
import argparse
import json as _json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
from loguru import logger

from sme_causal.agent.agent_service import CausalAgent, Explanation, QueryParser, ParsedQuery
from sme_causal.core.config import get_config
from sme_causal.core.utils import configure_logging, parse_client_id_and_intent
from sme_causal.core.columns import (
    CLIENT_ID,
    NEW_PRODUCT_OFFER,
    NEW_PRODUCT_OFFER_TYPE,
)
from sme_causal.orchestrator.pipeline import Pipeline
from sme_causal.orchestrator.persistence import CaseStore


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

    # Evidence source flags (all enabled by default)
    parser.add_argument(
        "--no-graph", action="store_true",
        help="Disable causal graph in prompts (enabled by default)",
    )
    parser.add_argument(
        "--no-rag", action="store_true",
        help="Disable RAG context enrichment (enabled by default)",
    )
    parser.add_argument(
        "--no-psm", action="store_true",
        help="Disable PSM ATT/ATE estimation (enabled by default)",
    )
    parser.add_argument(
        "--outcome-col",
        type=str,
        default="Revenue_Growth_Rate",
        help="Default outcome column for PSM (if not extracted from query)",
    )
    parser.add_argument(
        "--covariates",
        type=str,
        default="",
        help="Comma-separated covariates (default: heuristic set)",
    )
    parser.add_argument("--psm-caliper", type=float, default=0.05, help="PSM caliper")

    args = parser.parse_args()
    if args.debug_json:
        args.json = True

    # Derive positive flags from --no-* args
    use_graph = not args.no_graph
    use_rag = not args.no_rag
    use_psm = not args.no_psm

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

    # Normalize Cyrillic 'С' to Latin 'C' so that --client-id "С000005"
    # (which looks identical to "C000005") also matches dataset rows.
    if isinstance(client_id, str) and client_id[:1] in ("С", "с"):
        client_id = "C" + client_id[1:]

    if not client_id:
        # Fallback to first client if absolutely nothing is provided
        client_id = df[CLIENT_ID].iloc[0]
        logger.warning(f"No Client ID provided or found in query. Using first available: {client_id}")

    if client_id not in set(df[CLIENT_ID].tolist()):
        logger.error(f"Client_ID not found in dataset: {client_id}")
        sys.exit(3)

    # 6. Build Context (policy checks are handled inside Pipeline._policy_check)
    ctx = agent.build_context_for_client(df, client_id)

    # 7. Execution Logic based on Action Type

    # A) If 'optimize' -> Just explain client situation with specific target metric
    if action_type == "optimize":
        logger.info(f"Action is 'optimize' for target '{target_metric}'. Running client diagnosis...")
        expl: Explanation = agent.explain_client(
            ctx,
            use_graph=use_graph,
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

    # B) If 'what_if' (default) -> Run via Pipeline

    # Initialize Pipeline with SQLite persistence
    user_covariates = None
    if args.covariates.strip():
        user_covariates = [c.strip() for c in args.covariates.split(",") if c.strip()]

    case_store: Optional[CaseStore] = None
    try:
        case_store = CaseStore(cfg.cases_db_path)
    except Exception:
        logger.warning("Could not initialize SQLite store (non-fatal)")

    pipeline = Pipeline(
        df,
        case_store=case_store,
        graph_method=args.graph_method,
        use_rag=use_rag,
        use_graph=use_graph,
        use_psm=use_psm,
        outcome_col=target_metric,
        covariates=user_covariates,
        caliper=args.psm_caliper,
        min_conf=cfg.llm.confidence_threshold,
        model=cfg.llm.model_name,
        temperature=cfg.llm.temperature,
    )

    # Run pipeline
    case_state = pipeline.run(
        client_id,
        delta,
        raw_query=rag_query_text,
        target_metric=target_metric,
        match_info=match_info,
    )

    # Extract results from CaseState
    psm_result = case_state.get("psm_result")
    explanation = case_state.get("explanation", {})
    status = case_state.get("status", "unknown")
    critic_result = case_state.get("critic_result", {})

    # 8. Output Results
    if args.json:
        payload = {
            "client_id": client_id,
            "action_type": "what_if",
            "case_id": case_state.get("case_id"),
            "status": status,
            "query_parsed": {
                "label": query_analysis_label,
                "match_info": match_info,
                "detected_target": target_metric,
            } if args.query else None,
            "context": ctx,
            "what_if": {
                "delta": delta,
                "analysis": explanation,
                "psm_stats": psm_result if psm_result and psm_result.get("ok") else None,
            },
            "critic": critic_result,
            "requires_human_review": case_state.get("requires_human_review", False),
            "latency_ms": case_state.get("latency_ms"),
        }
        if case_state.get("abort_reason"):
            payload["abort_reason"] = case_state["abort_reason"]
        if case_state.get("abort_reason") == "policy_blocked":
            payload["policy_result"] = case_state.get("policy_result", {})
        if case_state.get("cooldown_previous_case"):
            payload["cooldown_previous_case"] = case_state["cooldown_previous_case"]
        print(_json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))
    else:
        logger.info(f"=== Analysis for {client_id} (case={case_state.get('case_id', '?')[:8]}) ===")
        if query_analysis_label:
            logger.info(f"Query Intent: {query_analysis_label}")
        if match_info:
            logger.info(f"Param matching info: {match_info}")

        if status == "aborted":
            prev = case_state.get("cooldown_previous_case")
            abort_reason = case_state.get("abort_reason")
            if abort_reason == "policy_blocked" and prev:
                logger.info(
                    "♻️ Эта интервенция уже оценивалась {} (кейс {}). Показан предыдущий результат:",
                    prev.get("created_at", "?"),
                    (prev.get("case_id") or "")[:8],
                )
                prev_expl = prev.get("explanation") or {}
                for key in ("diagnosis", "drivers_pos", "drivers_neg", "recommendations", "expected_effect"):
                    val = prev_expl.get(key)
                    if val:
                        logger.info(f"  {key}: {val}")
            elif abort_reason == "policy_blocked":
                logger.warning("Intervention blocked by policy checks:")
                for r in case_state.get("policy_result", {}).get("reasons", []):
                    logger.info(f"  • {r}")
                logger.info("=> Expected Uplift: 0.0")
            else:
                logger.warning(f"Case aborted: {abort_reason}")
        else:
            logger.info("--- What-if Scenario ---")
            logger.info(f"Delta: {delta}")
            if not delta and rag_query_text:
                logger.info(f"General Context: {rag_query_text}")

            # Format explanation from CaseState dict
            for key in ("diagnosis", "drivers_pos", "drivers_neg", "recommendations", "expected_effect"):
                val = explanation.get(key)
                if val:
                    logger.info(f"  {key}: {val}")

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

        # Pipeline metadata
        logger.info(f"--- Pipeline Status: {status} | Latency: {case_state.get('latency_ms')}ms ---")
        if not critic_result.get("passed", True):
            logger.warning(f"Critic issues: {critic_result.get('issues', [])}")
        if case_state.get("requires_human_review"):
            logger.warning(f"Requires human review: {case_state.get('review_reason')}")

    if case_store:
        case_store.close()


if __name__ == "__main__":
    main()
