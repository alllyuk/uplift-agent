"""Integration tests for the Pipeline orchestrator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sme_causal.orchestrator.pipeline import Pipeline
from sme_causal.orchestrator.persistence import CaseStore
from sme_causal.orchestrator.state import CaseState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame for pipeline testing."""
    return pd.DataFrame(
        {
            "Client_ID": ["C000001", "C000002", "C000005"],
            "Industry": ["Retail", "IT", "Retail"],
            "Region": ["Moscow", "SPB", "Moscow"],
            "Business_Size": ["Medium", "Small", "Large"],
            "Avg_Account_Balance": [100_000, 50_000, 200_000],
            "Avg_Monthly_Inflow": [30_000, 15_000, 60_000],
            "Avg_Monthly_Outflow": [25_000, 12_000, 50_000],
            "Num_Products": [3, 1, 5],
            "Revenue_Growth_Rate": [0.05, -0.02, 0.10],
            "Tariff_Discount": [0, 0, 1],
            "New_Product_Offer": [0, 1, 0],
            "New_Product_Offer_Type": ["none", "acquiring", "none"],
            "Credit_Limit_Change": [0.0, 5.0, 10.0],
            "Product_Types": ["loan", "acquiring", "loan,card"],
            "Months_Since_Last_Contact": [6, 2, 12],
            "Digital_Engagement_Score": [0.5, 0.8, 0.3],
            "Satisfaction_Score": [7, 8, 6],
            "Complaint_Count": [0, 1, 2],
            "Cross_Sell_Score": [0.6, 0.7, 0.4],
            "Churn_Risk_Score": [0.1, 0.3, 0.2],
            "Lifetime_Value": [50000, 20000, 100000],
            "Transaction_Count": [100, 50, 200],
            "Avg_Transaction_Amount": [300, 200, 500],
            "Account_Age_Months": [36, 12, 60],
            "Targeted_Communication": [0, 1, 0],
        }
    )


@pytest.fixture
def tmp_db(tmp_path) -> CaseStore:
    return CaseStore(tmp_path / "test_cases.db")


CANNED_EXPLANATION_JSON = json.dumps(
    {
        "diagnosis": "Скидка может стимулировать рост выручки клиента.",
        "drivers_pos": ["Увеличение лояльности клиента при снижении тарифа на обслуживание"],
        "drivers_neg": ["Снижение маржинальности на начальном этапе действия предложения"],
        "recommendations": ["Рассмотреть поэтапное внедрение тарифной скидки"],
        "expected_effect": "Рост выручки на 3-5% в течение квартала. Приоритет: PROFILE",
    },
    ensure_ascii=False,
)


def _mock_invoke(*args, **kwargs):
    """Mock for invoke_with_fallback returning canned JSON."""
    return (CANNED_EXPLANATION_JSON, None, True)


def _mock_rag_query(query, top_k=5):
    """Mock for RAG.perform_query."""
    return [
        "doc_01: Тарифные скидки повышают удержание клиентов на 5-10%.",
        "doc_02: Лояльность клиентов растёт при снижении стоимости обслуживания.",
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestPipelineHappyPath:
    """Happy path: all tools succeed, critic passes."""

    @patch("sme_causal.core.llm.invoke_with_fallback", _mock_invoke)
    @patch("sme_causal.agent.agent_service.CausalAgent._invoke_with_fallback")
    @patch("sme_causal.orchestrator.pipeline.Pipeline._run_rag_safe")
    def test_happy_path(self, mock_rag, mock_agent_invoke, sample_df, tmp_db):
        mock_agent_invoke.return_value = CANNED_EXPLANATION_JSON
        mock_rag.return_value = (
            ["doc_01: Скидки повышают удержание клиентов"],
            "эффект скидки на тариф",
        )

        pipeline = Pipeline(
            sample_df,
            case_store=tmp_db,
            use_psm=True,
            use_graph=False,
            use_rag=True,
        )
        state = pipeline.run("C000005", {"Tariff_Discount": 1})

        assert state["status"] in ("done", "degraded")
        assert state["client_id"] == "C000005"
        assert state["latency_ms"] is not None
        assert state["case_id"]

        # Should be persisted
        saved = tmp_db.get_case(state["case_id"])
        assert saved is not None
        assert saved["status"] == state["status"]


class TestPipelineClientNotFound:
    """Client not in DataFrame -> abort."""

    @patch("sme_causal.agent.agent_service.CausalAgent._invoke_with_fallback")
    def test_client_not_found(self, mock_invoke, sample_df):
        mock_invoke.return_value = CANNED_EXPLANATION_JSON
        pipeline = Pipeline(sample_df, use_psm=False, use_graph=False, use_rag=False)
        state = pipeline.run("C999999", {"Tariff_Discount": 1})

        assert state["status"] == "aborted"
        assert state["abort_reason"] == "client_not_found"


class TestPipelinePolicyBlock:
    """Cooldown blocks the intervention (same type within 30 days)."""

    @patch("sme_causal.agent.agent_service.CausalAgent._invoke_with_fallback")
    def test_cooldown_block(self, mock_invoke, sample_df, tmp_db):
        mock_invoke.return_value = CANNED_EXPLANATION_JSON

        # Pre-seed a recent "done" case for same intervention type
        from sme_causal.orchestrator.state import create_case_state

        old_case = create_case_state("C000001", {"Tariff_Discount": 1})
        old_case["status"] = "done"
        tmp_db.save_case(old_case)

        pipeline = Pipeline(
            sample_df, case_store=tmp_db,
            use_psm=False, use_graph=False, use_rag=False,
        )
        state = pipeline.run("C000001", {"Tariff_Discount": 1})

        assert state["status"] == "aborted"
        assert state["abort_reason"] == "policy_blocked"


class TestPipelineNoEvidence:
    """All 3 sources fail -> abort with no_evidence."""

    @patch("sme_causal.agent.agent_service.CausalAgent._invoke_with_fallback")
    @patch("sme_causal.orchestrator.pipeline.Pipeline._run_psm_safe", return_value=None)
    @patch("sme_causal.orchestrator.pipeline.Pipeline._run_rag_safe", return_value=([], None))
    @patch("sme_causal.orchestrator.pipeline.Pipeline._run_graph_safe", return_value="")
    def test_no_evidence(self, mock_graph, mock_rag, mock_psm, mock_invoke, sample_df):
        mock_invoke.return_value = CANNED_EXPLANATION_JSON
        pipeline = Pipeline(sample_df, use_psm=True, use_graph=True, use_rag=True)
        state = pipeline.run("C000001", {"Tariff_Discount": 1})

        assert state["status"] == "aborted"
        assert state["abort_reason"] == "no_evidence"


class TestPipelineCriticRetry:
    """Critic fails once, rag_refine + retry succeeds."""

    @patch("sme_causal.agent.agent_service.CausalAgent._invoke_with_fallback")
    @patch("sme_causal.orchestrator.pipeline.Pipeline._run_rag_safe")
    @patch("sme_causal.orchestrator.pipeline.Pipeline._run_graph_safe", return_value="")
    def test_critic_retry_then_pass(self, mock_graph, mock_rag, mock_invoke, sample_df, tmp_db):
        mock_rag.return_value = (["doc_01: test chunk"], "test query")

        # First call: incomplete explanation (will fail completeness check)
        # Second call (after retry): complete explanation
        call_count = {"n": 0}

        def _invoke_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Incomplete — missing drivers_neg and expected_effect
                return json.dumps({
                    "diagnosis": "Test",
                    "drivers_pos": ["factor"],
                    "drivers_neg": [],
                    "expected_effect": "",
                }, ensure_ascii=False)
            return CANNED_EXPLANATION_JSON

        mock_invoke.side_effect = _invoke_side_effect

        pipeline = Pipeline(
            sample_df,
            case_store=tmp_db,
            use_psm=False,
            use_graph=False,
            use_rag=True,
        )

        # Mock rag_refine to be a no-op (just update iterations)
        with patch("sme_causal.orchestrator.pipeline.rag_refine") as mock_rag_refine:
            state = pipeline.run("C000001", {"Tariff_Discount": 1})

        # Should have retried
        assert state["retry_count"] >= 1
        # Final status should be done or degraded (depending on second critic)
        assert state["status"] in ("done", "degraded")


class TestPipelineCriticFailTwice:
    """Critic fails twice -> degraded + human_review."""

    @patch("sme_causal.agent.agent_service.CausalAgent._invoke_with_fallback")
    @patch("sme_causal.orchestrator.pipeline.Pipeline._run_rag_safe")
    @patch("sme_causal.orchestrator.pipeline.Pipeline._run_graph_safe", return_value="")
    def test_critic_fail_twice(self, mock_graph, mock_rag, mock_invoke, sample_df, tmp_db):
        mock_rag.return_value = (["doc_01: test chunk"], "test query")

        # Always return incomplete explanation
        bad_explanation = json.dumps({
            "diagnosis": "x",
            "drivers_pos": [],
            "drivers_neg": [],
            "expected_effect": "",
        }, ensure_ascii=False)
        mock_invoke.return_value = bad_explanation

        pipeline = Pipeline(
            sample_df,
            case_store=tmp_db,
            use_psm=False,
            use_graph=False,
            use_rag=True,
        )

        with patch("sme_causal.orchestrator.pipeline.rag_refine"):
            state = pipeline.run("C000001", {"Tariff_Discount": 1})

        assert state["status"] == "degraded"
        assert state["requires_human_review"] is True
        assert state["review_reason"]


class TestCaseStorePersistence:
    """Verify SQLite persistence round-trip."""

    def test_save_and_read(self, tmp_db):
        from sme_causal.orchestrator.state import create_case_state

        s = create_case_state("C000005", {"Tariff_Discount": 1})
        s["status"] = "done"
        s["latency_ms"] = 5000
        s["explanation"] = {"diagnosis": "test"}
        tmp_db.save_case(s)

        loaded = tmp_db.get_case(s["case_id"])
        assert loaded is not None
        assert loaded["client_id"] == "C000005"
        assert loaded["status"] == "done"

    def test_cooldown(self, tmp_db):
        from sme_causal.orchestrator.state import create_case_state

        s = create_case_state("C000005", {"Tariff_Discount": 1})
        s["status"] = "done"
        tmp_db.save_case(s)

        assert tmp_db.check_cooldown("C000005", {"Tariff_Discount": 1}) is True
        assert tmp_db.check_cooldown("C000005", {"Credit_Limit_Change": 15}) is False
        assert tmp_db.check_cooldown("C000099", {"Tariff_Discount": 1}) is False
