"""CaseState definition and factory.

Mirrors the spec in docs/specs/agent-orchestrator.md §2.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict


class CaseState(TypedDict, total=False):
    """Full state carried through the pipeline for a single case."""

    # Identification
    case_id: str  # UUID4
    client_id: str  # e.g. "C000005"
    raw_query: Optional[str]  # NL query if provided

    # Context
    client_context: Dict[str, Any]  # 25 CONTEXT_FIELDS
    intervention_delta: Dict[str, Any]  # e.g. {"New_Product_Offer": 1}

    # Policy
    policy_result: Dict[str, Any]  # {blocked, reasons, notes}

    # Estimation
    psm_result: Optional[Dict[str, Any]]  # {ok, ate, att, n_pairs} or None
    rag_chunks: List[str]  # accumulated RAG chunks
    rag_iterations: int  # how many RAG calls (initial=1, after refine ≤2)
    rag_query_history: List[str]  # all RAG query formulations
    graph_dsl: str  # DSL edge string

    # Synthesis
    explanation: Dict[str, Any]  # ExplanationDict-compatible

    # Critic
    critic_result: Dict[str, Any]  # {passed, rule_issues, llm_issues, issues}
    retry_count: int  # 0 or 1

    # Human review
    requires_human_review: bool
    review_reason: Optional[str]

    # Metadata
    status: str  # intake|context|policy|estimation|synthesis|critic|rag_refine|done|aborted|degraded
    trace_id: Optional[str]
    latency_ms: Optional[int]
    abort_reason: Optional[str]
    started_at: float  # time.time() at intake

    # Prompt versions & A/B
    prompt_versions: Dict[str, str]  # {"base": "v1.0", "whatif": "v2.1"}
    experiment_id: Optional[str]
    variant: Optional[str]  # "A" | "B"


def create_case_state(
    client_id: str,
    intervention_delta: Dict[str, Any],
    *,
    raw_query: Optional[str] = None,
) -> CaseState:
    """Create a new CaseState with sensible defaults."""
    return CaseState(
        case_id=uuid.uuid4().hex,
        client_id=client_id,
        raw_query=raw_query,
        client_context={},
        intervention_delta=intervention_delta,
        policy_result={},
        psm_result=None,
        rag_chunks=[],
        rag_iterations=0,
        rag_query_history=[],
        graph_dsl="",
        explanation={},
        critic_result={},
        retry_count=0,
        requires_human_review=False,
        review_reason=None,
        status="intake",
        trace_id=None,
        latency_ms=None,
        abort_reason=None,
        started_at=time.time(),
        prompt_versions={},
        experiment_id=None,
        variant=None,
    )
