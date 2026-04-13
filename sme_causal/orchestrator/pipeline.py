"""Pipeline orchestrator: full case lifecycle management.

Implements the flow from docs/specs/agent-orchestrator.md:
  intake → load_context → policy_check → estimation(PSM+RAG+Graph)
    → synthesize → critic_check → [rag_refine → synthesize] → persist
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

from sme_causal.agent.agent_service import CausalAgent
from sme_causal.core.columns import CLIENT_ID, CONTEXT_FIELDS
from sme_causal.core.config import AppConfig, get_config
from sme_causal.core.utils import sanity_checks, create_query
from sme_causal.inference.psm_runner import run_psm
from sme_causal.orchestrator.critic import run_critic
from sme_causal.orchestrator.rag_refine import rag_refine
from sme_causal.orchestrator.state import CaseState, create_case_state


class Pipeline:
    """Orchestrates a single case through the full analysis pipeline."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        case_store: Optional[Any] = None,
        graph_method: str = "llm",
        use_rag: bool = True,
        use_graph: bool = True,
        use_psm: bool = True,
        outcome_col: str = "Revenue_Growth_Rate",
        covariates: Optional[List[str]] = None,
        caliper: float = 0.05,
        min_conf: float = 0.45,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.df = df
        self.case_store = case_store
        self.use_rag = use_rag
        self.use_graph = use_graph
        self.use_psm = use_psm
        self.outcome_col = outcome_col
        self.covariates = covariates or [
            "Industry", "Region", "Business_Size",
            "Avg_Account_Balance", "Avg_Monthly_Inflow",
            "Avg_Monthly_Outflow", "Num_Products",
        ]
        self.caliper = caliper
        self.min_conf = min_conf
        self.cfg: AppConfig = get_config()

        agent_kwargs: Dict[str, Any] = {"graph_method": graph_method}
        if model:
            agent_kwargs["model"] = model
        if temperature is not None:
            agent_kwargs["temperature"] = temperature
        self.agent = CausalAgent(**agent_kwargs)

        self._rag_instance: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        client_id: str,
        intervention_delta: Dict[str, Any],
        *,
        raw_query: Optional[str] = None,
        target_metric: Optional[str] = None,
        match_info: Optional[Dict] = None,
    ) -> CaseState:
        """Execute full pipeline and return CaseState."""
        state = create_case_state(client_id, intervention_delta, raw_query=raw_query)
        effective_target = target_metric or self.outcome_col

        try:
            self._intake(state)
            if state["status"] == "aborted":
                return self._finalize(state)

            self._load_context(state)
            if state["status"] == "aborted":
                return self._finalize(state)

            self._policy_check(state)
            if state.get("policy_result", {}).get("blocked"):
                state["status"] = "aborted"
                state["abort_reason"] = "policy_blocked"
                return self._finalize(state)

            self._estimation(state, effective_target)

            # No-evidence abort: only when at least one source was requested
            sources_requested = self.use_psm or self.use_rag or self.use_graph
            if sources_requested and (
                state.get("psm_result") is None
                and not state.get("rag_chunks")
                and not state.get("graph_dsl")
            ):
                state["status"] = "aborted"
                state["abort_reason"] = "no_evidence"
                return self._finalize(state)

            self._synthesize(state, effective_target, match_info=match_info)
            self._critic_check(state)

            # Retry loop (max 1): L2 issues serve as feedback for retry,
            # after retry only L1 rules are checked (L2 already did its job)
            if not state.get("critic_result", {}).get("passed", True):
                if state.get("retry_count", 0) == 0:
                    state["status"] = "rag_refine"
                    rag_refine(state)
                    state["retry_count"] = 1
                    self._synthesize(
                        state, effective_target,
                        match_info=match_info,
                        issues_context=self._format_issues(state),
                    )
                    self._critic_check(state, rules_only=True)

            # Determine final status
            critic = state.get("critic_result", {})
            if critic.get("passed", True):
                state["status"] = "done"
            else:
                state["status"] = "degraded"
                state["requires_human_review"] = True
                state["review_reason"] = (
                    "Critic failed after retry: "
                    + "; ".join(critic.get("issues", [])[:3])
                )

        except Exception as exc:
            logger.exception("Pipeline error for case {}", state.get("case_id", "?"))
            state["status"] = "aborted"
            state["abort_reason"] = f"pipeline_error: {exc}"

        return self._finalize(state)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------
    def _intake(self, state: CaseState) -> None:
        """Validate client_id exists in DataFrame."""
        state["status"] = "intake"
        if state["client_id"] not in self.df[CLIENT_ID].values:
            state["status"] = "aborted"
            state["abort_reason"] = "client_not_found"
            logger.warning("Client {} not found in data", state["client_id"])

    def _load_context(self, state: CaseState) -> None:
        """Build client context from DataFrame."""
        state["status"] = "context"
        try:
            ctx = self.agent.build_context_for_client(self.df, state["client_id"])
            state["client_context"] = ctx
        except ValueError:
            state["status"] = "aborted"
            state["abort_reason"] = "client_not_found"

    def _policy_check(self, state: CaseState) -> None:
        """Run sanity checks + cooldown check."""
        state["status"] = "policy"
        delta = state.get("intervention_delta", {})
        ctx = state.get("client_context", {})

        # Rule-based sanity checks
        if delta:
            checks = sanity_checks(ctx, delta)
        else:
            checks = {"blocked": False, "reasons": [], "notes": []}
        state["policy_result"] = checks

        # Cooldown check (fail-open on SQLite error)
        if self.case_store and not checks.get("blocked"):
            try:
                if self.case_store.check_cooldown(
                    state["client_id"], delta
                ):
                    checks["blocked"] = True
                    checks.setdefault("reasons", []).append(
                        "Cooldown: аналогичная интервенция выполнена менее 30 дней назад"
                    )
                    state["policy_result"] = checks
            except Exception:
                logger.warning("Cooldown check failed (fail-open)")
                state["requires_human_review"] = True
                state["review_reason"] = "cooldown не проверен: SQLite недоступен"

    def _estimation(self, state: CaseState, target_metric: str) -> None:
        """Run PSM, RAG, Graph in parallel. Each can fail independently."""
        state["status"] = "estimation"
        delta = state.get("intervention_delta", {})

        futures: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            if self.use_psm and delta:
                futures["psm"] = pool.submit(
                    self._run_psm_safe, delta, target_metric
                )
            if self.use_rag:
                futures["rag"] = pool.submit(
                    self._run_rag_safe, state
                )
            if self.use_graph:
                futures["graph"] = pool.submit(
                    self._run_graph_safe
                )

            for key, future in futures.items():
                try:
                    result = future.result(timeout=300)
                    if key == "psm":
                        state["psm_result"] = result
                    elif key == "rag":
                        chunks, query = result
                        state["rag_chunks"] = chunks
                        if query:
                            state["rag_query_history"] = [query]
                            state["rag_iterations"] = 1
                    elif key == "graph":
                        state["graph_dsl"] = result
                except Exception:
                    logger.warning("Estimation {} failed", key)

    def _synthesize(
        self,
        state: CaseState,
        target_metric: str,
        *,
        match_info: Optional[Dict] = None,
        issues_context: Optional[str] = None,
    ) -> None:
        """Call agent.explain_what_if with pre-fetched data."""
        state["status"] = "synthesis"
        ctx = state.get("client_context", {})
        delta = state.get("intervention_delta", {})
        psm = state.get("psm_result")

        # Prepare RAG context string from chunks
        rag_ctx_str: Optional[str] = None
        chunks = state.get("rag_chunks", [])
        if chunks:
            rag_ctx_str = "\n\n".join(chunks)

        expl = self.agent.explain_what_if(
            ctx,
            delta,
            psm_metrics=psm,
            use_graph=self.use_graph,
            use_rag=False,  # We pass pre-fetched rag_context instead
            min_conf=self.min_conf,
            rag_query_text=state.get("raw_query"),
            match_info=match_info,
            target_metric=target_metric,
            rag_context=rag_ctx_str,
            issues_context=issues_context,
        )
        # Convert Explanation to dict
        state["explanation"] = {
            "diagnosis": getattr(expl, "diagnosis", "") or "",
            "drivers_pos": getattr(expl, "drivers_pos", []) or [],
            "drivers_neg": getattr(expl, "drivers_neg", []) or [],
            "recommendations": getattr(expl, "recommendations", []) or [],
            "expected_effect": getattr(expl, "expected_effect", "") or "",
            "raw_text": getattr(expl, "raw_text", "") or "",
        }

    def _critic_check(self, state: CaseState, *, rules_only: bool = False) -> None:
        """Run critic. On retry pass, rules_only=True skips L2 LLM."""
        state["status"] = "critic"
        state["critic_result"] = run_critic(state, rules_only=rules_only)

    # ------------------------------------------------------------------
    # Estimation helpers (safe wrappers)
    # ------------------------------------------------------------------
    def _run_psm_safe(
        self, delta: Dict[str, Any], target_metric: str
    ) -> Optional[Dict[str, Any]]:
        try:
            result = run_psm(
                self.df,
                delta,
                outcome_col=target_metric,
                covariates=self.covariates,
                caliper=self.caliper,
            )
            if result.get("ok"):
                logger.info(
                    "PSM ok: ATT={} n_pairs={}",
                    result.get("att"), result.get("n_pairs"),
                )
                return result
            logger.warning("PSM not ok: {}", result.get("error"))
            return None
        except Exception:
            logger.warning("PSM exception, skipping")
            return None

    def _run_rag_safe(self, state: CaseState) -> tuple:
        """Return (chunks_list, query_text)."""
        try:
            from sme_causal.rag.rag_pipeline import RAG

            if self._rag_instance is None:
                self._rag_instance = RAG(self.cfg)
            delta = state.get("intervention_delta", {})
            query = state.get("raw_query") or create_query(delta)
            chunks = self._rag_instance.perform_query(query, top_k=3)
            logger.info("RAG: query='{}' found {} chunks", query[:60], len(chunks))
            return (chunks, query)
        except Exception:
            logger.warning("RAG exception, skipping")
            return ([], None)

    def _run_graph_safe(self) -> str:
        """Load graph DSL string."""
        try:
            dsl = self.agent._load_graph_dsl(min_conf=self.min_conf)
            if dsl:
                n_edges = dsl.count("\n") + 1
                logger.info("Graph DSL loaded: ~{} edges", n_edges)
            return dsl or ""
        except Exception:
            logger.warning("Graph DSL load exception, skipping")
            return ""

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    def _finalize(self, state: CaseState) -> CaseState:
        """Compute latency, persist, and return state."""
        state["latency_ms"] = int(
            (time.time() - state.get("started_at", time.time())) * 1000
        )
        if self.case_store:
            try:
                self.case_store.save_case(state)
            except Exception:
                logger.warning("Failed to persist case (non-fatal)")
        logger.info(
            "case_id={} status={} latency={}ms review={}",
            state["case_id"][:8],
            state.get("status"),
            state.get("latency_ms"),
            state.get("requires_human_review", False),
        )
        return state

    @staticmethod
    def _format_issues(state: CaseState) -> str:
        """Format critic issues for retry prompt."""
        critic = state.get("critic_result", {})
        issues = critic.get("issues", [])
        if not issues:
            return ""
        return "\n".join(f"- {issue}" for issue in issues)
