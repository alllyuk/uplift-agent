"""Unit tests for orchestrator Critic (Level 1 structural + Level 2 LLM)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sme_causal.orchestrator.critic import (
    CriticResult,
    critic_check_rules,
    critic_check_llm,
    run_critic,
    _check_source_attribution,
    _check_completeness,
)
from sme_causal.orchestrator.state import create_case_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_state(**overrides):
    s = create_case_state("C000005", {"Tariff_Discount": 1})
    s["explanation"] = {
        "diagnosis": "Скидка может привести к росту выручки клиента.",
        "drivers_pos": ["Увеличение лояльности клиента при снижении тарифа"],
        "drivers_neg": ["Снижение маржинальности на начальном этапе работы"],
        "recommendations": ["Рассмотреть поэтапное внедрение скидки"],
        "expected_effect": "Рост выручки на 3-5% в течение квартала",
        "raw_text": "",
    }
    s["psm_result"] = {"att": 0.05, "ate": 0.03, "n_pairs": 200}
    s.update(overrides)
    return s


# ---------------------------------------------------------------------------
# Source attribution
# ---------------------------------------------------------------------------
class TestSourceAttribution:
    def test_valid_citation(self):
        expl = {"diagnosis": "Согласно doc_12, эффект положительный"}
        chunks = ["doc_12: содержимое документа..."]
        assert _check_source_attribution(expl, chunks) == []

    def test_invalid_citation(self):
        expl = {"diagnosis": "Согласно doc_99, эффект положительный"}
        chunks = ["doc_12: содержимое документа..."]
        issues = _check_source_attribution(expl, chunks)
        assert len(issues) == 1
        assert "doc_99" in issues[0]

    def test_no_chunks_no_issues(self):
        expl = {"diagnosis": "Согласно doc_12, ..."}
        assert _check_source_attribution(expl, []) == []

    def test_multiple_valid_citations(self):
        expl = {"diagnosis": "doc_01 и doc_02 подтверждают эффект"}
        chunks = ["doc_01: текст", "doc_02: текст"]
        assert _check_source_attribution(expl, chunks) == []

    def test_one_valid_one_invalid(self):
        expl = {"diagnosis": "doc_01 подтверждает, doc_99 тоже"}
        chunks = ["doc_01: текст"]
        issues = _check_source_attribution(expl, chunks)
        assert len(issues) == 1
        assert "doc_99" in issues[0]


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------
class TestCompleteness:
    def test_all_present(self):
        expl = {
            "drivers_pos": ["Увеличение лояльности клиента"],
            "drivers_neg": ["Снижение маржинальности"],
            "expected_effect": "Рост выручки 3-5%",
        }
        assert _check_completeness(expl) == []

    def test_missing_field(self):
        expl = {"drivers_pos": ["ok"], "drivers_neg": ["ok"]}
        issues = _check_completeness(expl)
        assert any("expected_effect" in i for i in issues)

    def test_too_short(self):
        expl = {
            "drivers_pos": ["ab"],
            "drivers_neg": ["Снижение маржинальности"],
            "expected_effect": "Рост выручки 3-5%",
        }
        issues = _check_completeness(expl)
        assert any("drivers_pos" in i for i in issues)

    def test_all_missing(self):
        issues = _check_completeness({})
        assert len(issues) == 3

    def test_empty_list(self):
        expl = {
            "drivers_pos": [],
            "drivers_neg": ["Снижение маржинальности"],
            "expected_effect": "Рост выручки 3-5%",
        }
        issues = _check_completeness(expl)
        assert any("drivers_pos" in i for i in issues)


# ---------------------------------------------------------------------------
# critic_check_rules integration
# ---------------------------------------------------------------------------
class TestCriticCheckRules:
    def test_clean_pass(self):
        s = _make_state()
        issues = critic_check_rules(s)
        assert issues == []

    def test_empty_explanation_fails(self):
        s = _make_state(explanation={})
        issues = critic_check_rules(s)
        assert len(issues) >= 3  # at least completeness failures

    def test_phantom_doc_fails(self):
        s = _make_state(
            rag_chunks=["doc_01: текст"],
        )
        s["explanation"]["diagnosis"] = "По данным doc_99 эффект очевиден"
        issues = critic_check_rules(s)
        assert any("doc_99" in i for i in issues)


# ---------------------------------------------------------------------------
# LLM critic (mocked)
# ---------------------------------------------------------------------------
class TestCriticCheckLLM:
    def test_clean_llm_response(self):
        s = _make_state()
        mock_return = ('{"llm_issues": []}', None, True)
        with patch("sme_causal.core.llm.invoke_with_fallback", return_value=mock_return):
            issues = critic_check_llm(s)
        assert issues == []

    def test_llm_finds_issues(self):
        s = _make_state()
        mock_return = (
            '{"llm_issues": [{"id": "L2", "field": "diagnosis", "note": "Необоснованный вывод"}]}',
            None,
            True,
        )
        with patch("sme_causal.core.llm.invoke_with_fallback", return_value=mock_return):
            issues = critic_check_llm(s)
        assert len(issues) == 1
        assert "L2" in issues[0]

    def test_llm_failure_failopen(self):
        s = _make_state()
        with patch("sme_causal.core.llm.invoke_with_fallback", side_effect=Exception("timeout")):
            issues = critic_check_llm(s)
        assert issues == []


# ---------------------------------------------------------------------------
# run_critic (combined)
# ---------------------------------------------------------------------------
class TestRunCritic:
    def test_rules_fail_skips_llm(self):
        s = _make_state(explanation={})
        with patch("sme_causal.orchestrator.critic.critic_check_llm") as mock_llm:
            result = run_critic(s)
        mock_llm.assert_not_called()
        assert result["passed"] is False
        assert len(result["rule_issues"]) > 0

    def test_rules_pass_calls_llm(self):
        s = _make_state()
        mock_return = ('{"llm_issues": []}', None, True)
        with patch("sme_causal.core.llm.invoke_with_fallback", return_value=mock_return):
            result = run_critic(s)
        assert result["passed"] is True

    def test_rules_only_skips_llm(self):
        s = _make_state()
        with patch("sme_causal.orchestrator.critic.critic_check_llm") as mock_llm:
            result = run_critic(s, rules_only=True)
        mock_llm.assert_not_called()
        assert result["passed"] is True

    def test_result_structure(self):
        s = _make_state(explanation={})
        result = run_critic(s)
        assert "passed" in result
        assert "rule_issues" in result
        assert "llm_issues" in result
        assert "issues" in result
        assert isinstance(result["issues"], list)
