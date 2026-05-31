"""Critic checks — rule-based (Level 1) and LLM-augmented (Level 2).

Level 1 — lightweight structural checks (no NLU):
  - Source attribution: cited doc_ids must exist in RAG chunks
  - Completeness: required explanation fields must be non-empty

Level 2 — LLM-augmented semantic checks (numeric consistency, graph edge
validity, hedging adequacy, recommendation completeness). Runs only when
L1 passes. Fail-open on LLM errors.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TypedDict

from loguru import logger

from sme_causal.orchestrator.state import CaseState


# -----------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------
class CriticResult(TypedDict):
    passed: bool
    rule_issues: List[str]
    llm_issues: List[str]
    issues: List[str]  # union of rule_issues + llm_issues


# -----------------------------------------------------------------------
# Level 1 — rule-based checks (structural only)
# -----------------------------------------------------------------------

def critic_check_rules(state: CaseState) -> List[str]:
    """Run structural checks. Return list of issue strings (empty = pass)."""
    explanation = state.get("explanation", {})
    rag_chunks = state.get("rag_chunks", [])

    issues: List[str] = []
    issues.extend(_check_source_attribution(explanation, rag_chunks))
    issues.extend(_check_completeness(explanation))
    return issues


def _check_source_attribution(
    explanation: Dict[str, Any],
    rag_chunks: List[str],
) -> List[str]:
    """Any doc_id cited in text must exist in retrieved chunks."""
    text = _explanation_full_text(explanation)
    if not text or not rag_chunks:
        return []
    issues: List[str] = []
    cited_ids = set(re.findall(r"doc_\d+", text, re.IGNORECASE))
    chunks_text = " ".join(rag_chunks)
    for doc_id in cited_ids:
        if doc_id not in chunks_text:
            issues.append(f"Ссылка на несуществующий документ: {doc_id}")
    return issues


def _check_completeness(explanation: Dict[str, Any]) -> List[str]:
    """Required Explanation fields are non-empty (>=10 chars)."""
    issues: List[str] = []
    for field in ("drivers_pos", "drivers_neg", "expected_effect"):
        val = explanation.get(field)
        if val is None:
            issues.append(f"Поле '{field}' отсутствует в объяснении")
            continue
        if isinstance(val, list):
            combined = " ".join(str(v) for v in val)
            if len(combined.strip()) < 10:
                issues.append(f"Поле '{field}' содержит менее 10 символов")
        elif isinstance(val, str):
            if len(val.strip()) < 10:
                issues.append(f"Поле '{field}' содержит менее 10 символов")
    return issues


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _explanation_full_text(explanation: Dict[str, Any]) -> str:
    """Concatenate all textual fields of explanation into a single string."""
    parts: List[str] = []
    for key in ("diagnosis", "expected_effect", "raw_text"):
        val = explanation.get(key)
        if val:
            parts.append(str(val))
    for key in ("drivers_pos", "drivers_neg", "recommendations"):
        val = explanation.get(key)
        if isinstance(val, list):
            parts.extend(str(v) for v in val)
    return " ".join(parts)


# -----------------------------------------------------------------------
# Level 2 — LLM-augmented checks
# -----------------------------------------------------------------------

_LLM_CRITIC_PROMPT = """\
Ты — критик качества аналитических объяснений. Проверь ответ аналитического агента по 4 категориям.

Данные кейса:
- Intervention delta: {delta}
- PSM result: {psm}
- RAG chunks: {rag_summary}
- Graph DSL: {graph_summary}

Ответ агента:
{explanation_text}

Проверь по следующим критериям:
L1 — Логическая консистентность: drivers_pos и drivers_neg не противоречат друг другу и совпадают по знаку с PSM ATT.
L2 — Соответствие фактам: утверждения в diagnosis и expected_effect подкреплены evidence (PSM, RAG, Graph), нет необоснованных выводов.
L3 — Адекватность хеджирования: при слабых данных текст содержит явные оговорки о неопределённости.
L4 — Полнота recommendations: рекомендации адресуют именно тот intervention из delta, а не общие пожелания.

Для каждой найденной проблемы укажи severity:
- "high" — содержательная ошибка, искажающая смысл или вводящая читателя в заблуждение:
    • ссылка на doc_id, отсутствующий в RAG chunks;
    • прямое противоречие со знаком PSM ATT (ATT положительный, а в ответе категоричный отрицательный эффект, или наоборот);
    • утверждения о фактах клиента, которых нет ни в одном из источников (PSM/RAG/Graph);
    • категоричный вывод при заведомо ненадёжных данных без хеджирования (psm_ok=False, формулировки «гарантирован», «точно»);
    • полное отсутствие рекомендаций по адресуемой интервенции.
- "low" — стилистическое замечание, не меняющее содержательную картину:
    • хеджирование присутствует, но могло бы быть сильнее;
    • рекомендация даёт направление, но не предельно конкретна;
    • формулировка корректна по смыслу, но избыточна или общна;
    • drivers_pos и drivers_neg частично дублируются по идее.

Верни ТОЛЬКО JSON (без markdown):
{{"llm_issues": [{{"id": "L2", "field": "diagnosis", "severity": "high", "note": "..."}}, ...]}}
Если проблем нет, верни: {{"llm_issues": []}}
"""


def critic_check_llm(state: CaseState) -> List[str]:
    """Level 2: LLM-augmented semantic checks. Returns list of issue strings.

    Fail-open: on LLM error returns empty list (safety guaranteed by L1 rules).
    """
    from sme_causal.core.llm import invoke_with_fallback
    from sme_causal.core.utils import parse_json_obj_from_text
    from sme_causal.core.config import get_config

    cfg = get_config()
    explanation = state.get("explanation", {})
    psm = state.get("psm_result")
    rag_chunks = state.get("rag_chunks", [])
    graph_dsl = state.get("graph_dsl", "")

    rag_summary = "\n\n".join(rag_chunks) if rag_chunks else "нет данных"
    graph_summary = graph_dsl if graph_dsl else "нет данных"
    psm_str = str(psm) if psm else "нет данных"

    prompt = _LLM_CRITIC_PROMPT.format(
        delta=state.get("intervention_delta", {}),
        psm=psm_str,
        rag_summary=rag_summary,
        graph_summary=graph_summary,
        explanation_text=_explanation_full_text(explanation),
    )

    messages = [
        {"role": "system", "content": "Ты — критик качества аналитических ответов."},
        {"role": "user", "content": prompt},
    ]

    try:
        content, _, _ = invoke_with_fallback(
            messages,
            model=cfg.llm.model_name,
            temperature=0.0,
            api_key=cfg.effective_llm_api_key,
        )
        obj = parse_json_obj_from_text(content)
        raw_issues = obj.get("llm_issues", [])
        if not isinstance(raw_issues, list):
            return []
        items = [it for it in raw_issues if isinstance(it, dict)]
        high = [it for it in items if it.get("severity", "high") == "high"]
        low = [it for it in items if it.get("severity") == "low"]
        if low:
            logger.info(
                "L2 critic low-severity issues (logged, not blocking): {} item(s): {}",
                len(low), low,
            )
        if high:
            logger.warning(
                "L2 critic high-severity issues (blocking, will trigger retry): {} item(s): {}",
                len(high), high,
            )
        return [
            f"[{item.get('id', '?')}] {item.get('field', '?')}: {item.get('note', '?')}"
            for item in high
        ]
    except Exception:
        logger.warning("LLM critic failed (fail-open), returning no issues")
        return []


# -----------------------------------------------------------------------
# Combined critic
# -----------------------------------------------------------------------

def run_critic(state: CaseState, *, rules_only: bool = False) -> CriticResult:
    """Run Level 1 rules; if clean and not rules_only, run Level 2 LLM.

    Args:
        rules_only: If True, skip L2 LLM checks (used on retry pass).
    """
    rule_issues = critic_check_rules(state)

    llm_issues: List[str] = []
    if not rule_issues and not rules_only:
        llm_issues = critic_check_llm(state)

    all_issues = rule_issues + llm_issues
    return CriticResult(
        passed=len(all_issues) == 0,
        rule_issues=rule_issues,
        llm_issues=llm_issues,
        issues=all_issues,
    )
