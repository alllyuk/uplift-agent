"""End-to-end validation experiments for Chapter 4 of the thesis.

Covers four hypotheses (H1-H4) and the Industrial-track effectiveness criterion:

- E1 (ablation):    full vs no_psm vs no_rag vs no_graph vs llm_only
- E2 (adaptive RAG): per-case rag_iterations / new chunks / pre-vs-post-refine pass
- E3 (two-level critic): rule_issues vs llm_issues distribution
- E4 (latency/cost): p50/p95 latency_ms; token-based cost estimate
- E5 (LLM-as-judge): four-criteria 0-5 grading via gpt-5.4-mini at T=0
- E6 (baseline):    full vs llm_only via the same judge
- E7 (robustness):  v1 prompt vs v2 (rephrased) — cosine similarity + key-conclusions match

Outputs are written to ``reports/chapter4_<timestamp>/`` with per-case JSON dumps
and three CSV summaries (cases, judge, robustness).

Usage::

    python -m sme_causal.evaluation.chapter4 \\
        --client-ids "C000100,C000350,C000720,C001100,C001500,C001900,C002300,C002700" \\
        --out "reports/chapter4_<timestamp>" \\
        --robustness-clients 5

The script never touches the production cases.db: ``case_store=None`` is passed
to Pipeline so cooldown is not enforced and runs are not persisted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from sme_causal.core.config import get_config
from sme_causal.core.columns import CLIENT_ID
from sme_causal.core.llm import invoke_with_fallback
from sme_causal.core.utils import parse_json_obj_from_text, sanity_checks
from sme_causal.orchestrator.pipeline import Pipeline
from sme_causal.orchestrator.state import CaseState


# ---------------------------------------------------------------------------
# Configuration of experiments
# ---------------------------------------------------------------------------

ABLATION_VARIANTS: List[Dict[str, Any]] = [
    {"name": "full",     "use_psm": True,  "use_rag": True,  "use_graph": True},
    {"name": "no_psm",   "use_psm": False, "use_rag": True,  "use_graph": True},
    {"name": "no_rag",   "use_psm": True,  "use_rag": False, "use_graph": True},
    {"name": "no_graph", "use_psm": True,  "use_rag": True,  "use_graph": False},
    {"name": "llm_only", "use_psm": False, "use_rag": False, "use_graph": False},
]

# Pool of interventions; we cycle through them across clients to diversify cases.
INTERVENTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "acquiring": {"New_Product_Offer": 1, "New_Product_Offer_Type": "acquiring"},
    "payroll": {"New_Product_Offer": 1, "New_Product_Offer_Type": "payroll"},
    "deposit": {"New_Product_Offer": 1, "New_Product_Offer_Type": "deposit"},
    "credit15": {"Credit_Limit_Change": 15.0},
    "credit25": {"Credit_Limit_Change": 25.0},
    "tariff": {"Tariff_Discount": 1},
}
INTERVENTION_POOL: List[Dict[str, Any]] = list(INTERVENTION_REGISTRY.values())

# Prices per 1k tokens for gpt-5.4-mini family (approximate; only used for
# rough USD estimate per case).
USD_PER_1K_INPUT = 0.00015
USD_PER_1K_OUTPUT = 0.0006

STRICT_JUDGE_PROMPT_TEMPLATE = """\
Ты — рецензент топ-конференции по применению ИИ в финансовом секторе. Твоя задача — оценить
качество ответа аналитического агента, сгенерированного для конкретной клиентской ситуации
малого / микробизнеса.

Контекст кейса:
- Клиент (профиль): {context}
- Запрашиваемая интервенция: {delta}
- Числовая PSM-оценка: {psm}
- RAG-фрагменты: {rag}
- Причинно-следственный граф (DSL): {graph}

Ответ агента (структурированный):
{explanation}

Оцени ответ строго по четырём критериям, шкала дискретная 0-5 (0 — критически плохо,
3 — приемлемо для аналитика, 5 — образцово). Перед каждой оценкой коротко поясни
рассуждение (одно-два предложения), затем выдай число.

Критерии:
- USE_OF_CONTEXT — насколько ответ опирается на переданные данные (профиль, PSM, RAG, граф),
  а не на общее знание модели.
- CAUSAL_REASONING — логическая связность утверждений и согласованность факторов с
  причинно-следственной структурой.
- INTERPRETABILITY — ясность изложения, структурность, прослеживаемость каждого утверждения
  к источнику.
- RECOMMENDATION_QUALITY — релевантность и применимость рекомендаций именно к запрошенной
  интервенции, адекватность хеджирования при слабых данных.

Верни СТРОГО JSON без markdown:
{{
  "use_of_context":          {{"reasoning": "...", "score": <int 0-5>}},
  "causal_reasoning":        {{"reasoning": "...", "score": <int 0-5>}},
  "interpretability":        {{"reasoning": "...", "score": <int 0-5>}},
  "recommendation_quality":  {{"reasoning": "...", "score": <int 0-5>}}
}}
"""

ACCEPTANCE_JUDGE_PROMPT_TEMPLATE = """\
Ты — эксперт банка-партнёра, проверяющий, пригоден ли ответ аналитического агента
для первого рабочего использования в процессе оценки клиентских интервенций ММБ.
Оцени не как научную статью и не как идеальную консультацию, а как практический
артефакт поддержки решения: можно ли аналитику понять основания рекомендации,
увидеть источники, риски и ожидаемый эффект.

Контекст кейса:
- Клиент (профиль): {context}
- Запрашиваемая интервенция: {delta}
- Числовая PSM-оценка: {psm}
- RAG-фрагменты: {rag}
- Причинно-следственный граф (DSL): {graph}

Ответ агента (структурированный):
{explanation}

Шкала 0-5:
- 5 — ответ можно использовать без существенных правок; основания, ограничения и вывод ясны.
- 4 — ответ пригоден для рабочего использования; возможны мелкие редакционные правки
  или уточнения, но они не меняют управленческий вывод.
- 3 — черновик полезен, но требует заметной доработки аналитиком перед использованием.
- 2 — есть существенные пробелы в основаниях или логике.
- 1 — ответ почти не помогает принять решение.
- 0 — ответ противоречит данным или непригоден.

Правила оценки:
- Не требуй коммерческий скрипт продажи, если он не был запрошен; оцени именно
  обоснованность интервенции и понятность вывода.
- Не снижай балл только за осторожное хеджирование: при слабой PSM-оценке или
  отсутствии прямого пути в графе осторожность является корректным поведением.
- Не требуй дословной цитаты каждого RAG-фрагмента; достаточно, чтобы утверждения
  были согласованы с переданными источниками и не противоречили им.
- Если ответ использует профиль клиента, PSM, RAG и граф в достаточной для аналитика
  степени, базовый уровень по соответствующему критерию — 4.
- Ставь ниже 4 только при ошибке, которая реально мешает аналитику использовать ответ.

Критерии:
- USE_OF_CONTEXT — насколько ответ опирается на переданные данные (профиль, PSM, RAG, граф),
  а не на общее знание модели.
- CAUSAL_REASONING — логическая связность утверждений и согласованность факторов с
  причинно-следственной структурой.
- INTERPRETABILITY — ясность изложения, структурность, прослеживаемость ключевых утверждений
  к источникам.
- RECOMMENDATION_QUALITY — релевантность и применимость вывода именно к запрошенной
  интервенции, адекватность хеджирования при слабых данных.

Верни СТРОГО JSON без markdown:
{{
  "use_of_context":          {{"reasoning": "...", "score": <int 0-5>}},
  "causal_reasoning":        {{"reasoning": "...", "score": <int 0-5>}},
  "interpretability":        {{"reasoning": "...", "score": <int 0-5>}},
  "recommendation_quality":  {{"reasoning": "...", "score": <int 0-5>}}
}}
"""

JUDGE_PROMPTS = {
    "strict": STRICT_JUDGE_PROMPT_TEMPLATE,
    "acceptance": ACCEPTANCE_JUDGE_PROMPT_TEMPLATE,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _explanation_text(state: CaseState) -> str:
    expl = state.get("explanation", {}) or {}
    parts: List[str] = []
    for key in ("diagnosis", "expected_effect"):
        v = expl.get(key)
        if v:
            parts.append(f"{key}: {v}")
    for key in ("drivers_pos", "drivers_neg", "recommendations"):
        v = expl.get(key)
        if isinstance(v, list) and v:
            parts.append(f"{key}: " + " | ".join(str(x) for x in v))
    return "\n".join(parts)


def _est_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 3.5 chars for mixed Russian + English."""
    return max(1, int(len(text) / 3.5))


def _summarise_state_for_judge(state: CaseState) -> Dict[str, str]:
    psm = state.get("psm_result") or {}
    if isinstance(psm, dict) and psm.get("ok"):
        psm_str = (
            f"ATT={psm.get('att'):.4f}; ATE={psm.get('ate'):.4f}; "
            f"n_pairs={psm.get('n_pairs')}; reliable={psm.get('psm_reliable')}"
        )
    else:
        psm_str = "недоступно или отключено"

    rag = state.get("rag_chunks") or []
    if rag:
        rag_str = "\n---\n".join(c[:600] for c in rag[:5])
    else:
        rag_str = "недоступно или отключено"

    graph = state.get("graph_dsl") or ""
    graph_str = graph[:1500] if graph else "недоступно или отключено"

    ctx = state.get("client_context") or {}
    ctx_str = json.dumps(ctx, ensure_ascii=False)[:1500]

    return {
        "context": ctx_str,
        "delta": json.dumps(state.get("intervention_delta", {}), ensure_ascii=False),
        "psm": psm_str,
        "rag": rag_str,
        "graph": graph_str,
    }


def _parse_intervention_pool(names: Optional[str]) -> List[Dict[str, Any]]:
    if not names:
        return [dict(x) for x in INTERVENTION_POOL]
    out: List[Dict[str, Any]] = []
    for raw in names.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in INTERVENTION_REGISTRY:
            allowed = ", ".join(sorted(INTERVENTION_REGISTRY))
            raise ValueError(f"Unknown intervention '{name}'. Allowed: {allowed}")
        out.append(dict(INTERVENTION_REGISTRY[name]))
    if not out:
        raise ValueError("Empty intervention pool")
    return out


def _select_applicable_client_ids(
    df: pd.DataFrame,
    intervention_pool: Sequence[Dict[str, Any]],
    n: int,
    *,
    scan_limit: Optional[int] = None,
) -> List[str]:
    """Pick client IDs that pass policy for the next cycled intervention."""
    selected: List[str] = []
    scan_df = df.head(scan_limit) if scan_limit else df
    for _, row in scan_df.iterrows():
        delta = dict(intervention_pool[len(selected) % len(intervention_pool)])
        if not sanity_checks(row.to_dict(), delta).get("blocked"):
            selected.append(str(row[CLIENT_ID]))
            if len(selected) >= n:
                return selected
    raise ValueError(
        f"Only {len(selected)} applicable clients found for {n} requested "
        f"with intervention pool of size {len(intervention_pool)}"
    )


def _llm_judge(
    state: CaseState,
    model: str,
    api_key: Optional[str],
    *,
    rubric: str = "acceptance",
) -> Dict[str, Any]:
    summary = _summarise_state_for_judge(state)
    explanation_text = _explanation_text(state) or "ответ не сгенерирован"
    prompt = JUDGE_PROMPTS[rubric].format(
        explanation=explanation_text,
        **summary,
    )
    msgs = [
        {"role": "system", "content": "Ты — рецензент по системам поддержки решений в финансах."},
        {"role": "user", "content": prompt},
    ]
    try:
        content, _, _ = invoke_with_fallback(
            msgs, model=model, temperature=0.0, api_key=api_key,
        )
        obj = parse_json_obj_from_text(content)
        return obj
    except Exception as exc:
        logger.warning("Judge call failed: {}", exc)
        return {"error": str(exc)}


def _run_one(
    df: pd.DataFrame,
    client_id: str,
    delta: Dict[str, Any],
    variant: Dict[str, Any],
) -> CaseState:
    pipeline = Pipeline(
        df=df,
        case_store=None,  # bypass cooldown / persistence for experiments
        graph_method="llm",
        use_psm=variant["use_psm"],
        use_rag=variant["use_rag"],
        use_graph=variant["use_graph"],
        outcome_col="Revenue_Growth_Rate",
    )
    return pipeline.run(client_id, delta)


def _key_conclusions(state: CaseState) -> List[str]:
    """Extract ordered set of short bullet-points from drivers_pos+neg+recs."""
    expl = state.get("explanation", {}) or {}
    out: List[str] = []
    for key in ("drivers_pos", "drivers_neg", "recommendations"):
        v = expl.get(key)
        if isinstance(v, list):
            for x in v:
                s = str(x).strip()
                if s:
                    # Truncate to first phrase to be robust to wording.
                    s_short = re.split(r"[.;\n]", s)[0][:120].lower()
                    out.append(s_short)
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(a @ b / (na * nb))


# ---------------------------------------------------------------------------
# Core experiment runners
# ---------------------------------------------------------------------------

def run_ablation(
    df: pd.DataFrame,
    client_ids: Sequence[str],
    out_dir: Path,
    *,
    seed: int = 42,
    variants: Optional[Sequence[str]] = None,
    intervention_pool: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """E1 + E3 + E4 + E6: run pipeline over (client × ablation-variant) grid."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    cases_root = out_dir / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    selected_variants = (
        [v for v in ABLATION_VARIANTS if v["name"] in set(variants)]
        if variants else ABLATION_VARIANTS
    )
    pool = [dict(x) for x in (intervention_pool or INTERVENTION_POOL)]

    # Pre-pick intervention per client deterministically so all variants for the
    # same client run with the same delta — clean ablation.
    client_to_delta: Dict[str, Dict[str, Any]] = {}
    for i, cid in enumerate(client_ids):
        delta = pool[i % len(pool)]
        client_to_delta[cid] = dict(delta)

    total = len(client_ids) * len(selected_variants)
    done = 0
    for cid in client_ids:
        delta = client_to_delta[cid]
        for variant in selected_variants:
            t0 = time.time()
            try:
                state = _run_one(df, cid, delta, variant)
            except Exception as exc:
                logger.exception("Pipeline crashed for {} {}", cid, variant["name"])
                state = {"status": "aborted", "abort_reason": f"crash: {exc}"}
            wall_ms = int((time.time() - t0) * 1000)

            row = {
                "client_id": cid,
                "delta": json.dumps(delta, ensure_ascii=False),
                "variant": variant["name"],
                "status": state.get("status"),
                "abort_reason": state.get("abort_reason"),
                "latency_ms": state.get("latency_ms") or wall_ms,
                "rag_iterations": state.get("rag_iterations", 0),
                "retry_count": state.get("retry_count", 0),
                "rule_issues_n": len((state.get("critic_result") or {}).get("rule_issues", []) or []),
                "llm_issues_n": len((state.get("critic_result") or {}).get("llm_issues", []) or []),
                "passed_critic": bool((state.get("critic_result") or {}).get("passed", False)),
                "n_rag_chunks": len(state.get("rag_chunks") or []),
                "psm_ok": bool((state.get("psm_result") or {}).get("ok", False)) if state.get("psm_result") else False,
                "review_required": bool(state.get("requires_human_review")),
            }
            rows.append(row)
            # Persist full state for later inspection (drop large repeated context).
            persist_path = cases_root / variant["name"] / f"{cid}.json"
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(persist_path, "w", encoding="utf-8") as fh:
                json.dump(_json_safe(state), fh, ensure_ascii=False, indent=2)

            done += 1
            logger.info("ablation [{}/{}] {} {} -> {} ({} ms)",
                        done, total, cid, variant["name"], state.get("status"), row["latency_ms"])

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(out_dir / "cases_summary.csv", index=False)
    return rows


def run_judge(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    *,
    model: str,
    api_key: Optional[str],
    max_workers: int = 4,
    rubric: str = "acceptance",
) -> List[Dict[str, Any]]:
    """E5: LLM-judge every case from the ablation grid."""
    cases_root = out_dir / "cases"

    def _score_one(row: Dict[str, Any]) -> Dict[str, Any]:
        path = cases_root / row["variant"] / f"{row['client_id']}.json"
        if not path.exists():
            return {"client_id": row["client_id"], "variant": row["variant"], "scored": False}
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
        if state.get("status") not in ("done", "degraded"):
            return {"client_id": row["client_id"], "variant": row["variant"],
                    "scored": False, "status": state.get("status")}
        verdict = _llm_judge(state, model=model, api_key=api_key, rubric=rubric)
        out = {
            "client_id": row["client_id"],
            "variant": row["variant"],
            "status": state.get("status"),
            "scored": "error" not in verdict and bool(verdict),
        }
        for c in ("use_of_context", "causal_reasoning", "interpretability", "recommendation_quality"):
            entry = verdict.get(c) if isinstance(verdict, dict) else None
            if isinstance(entry, dict):
                try:
                    out[c] = int(entry.get("score"))
                except Exception:
                    out[c] = None
                out[f"{c}_reasoning"] = str(entry.get("reasoning", ""))[:500]
            else:
                out[c] = None
        if isinstance(verdict, dict):
            scores = [out.get(c) for c in (
                "use_of_context", "causal_reasoning",
                "interpretability", "recommendation_quality",
            )]
            valid = [s for s in scores if isinstance(s, int)]
            out["mean_score"] = float(np.mean(valid)) if valid else None
        else:
            out["mean_score"] = None
        return out

    scored: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_one, row): row for row in rows}
        for fut in as_completed(futures):
            scored.append(fut.result())
    df_scored = pd.DataFrame(scored)
    df_scored.to_csv(out_dir / "judge_scores.csv", index=False)
    with open(out_dir / "judge_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"model": model, "rubric": rubric}, fh, ensure_ascii=False, indent=2)
    return scored


_ROBUSTNESS_SYSTEM_V1 = (
    "Вы — старший аналитик ММБ банка с экспертизой в области клиентского анализа.\n"
    "Ваша задача: на основании переданных данных оценить интервенцию и составить структурированный\n"
    "ответ. Используйте ТОЛЬКО факты из переданного контекста; не выдумывайте связей."
)

_ROBUSTNESS_SYSTEM_V2 = (
    "Вы — методолог банка, специализирующийся на работе с клиентами малого и микробизнеса.\n"
    "Опираясь исключительно на предоставленные источники (профиль клиента, PSM-оценку, RAG-фрагменты\n"
    "и причинно-следственный граф), оцените эффект интервенции и обоснуйте рекомендацию."
)

_ROBUSTNESS_USER_TEMPLATE = """\
Профиль клиента:
{context}

Запрашиваемая интервенция (delta):
{delta}

PSM-оценка: {psm}
RAG-фрагменты:
{rag}

Причинно-следственный граф (DSL):
{graph}

Сформируйте ответ строго в формате JSON, без markdown:
{{
  "drivers_pos":      ["..."],
  "drivers_neg":      ["..."],
  "recommendations":  ["..."],
  "expected_effect":  "..."
}}
"""


def _resynthesize(state: CaseState, system_prompt: str, model: str, api_key: Optional[str]) -> Dict[str, Any]:
    summary = _summarise_state_for_judge(state)
    user_prompt = _ROBUSTNESS_USER_TEMPLATE.format(**summary)
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        content, _, _ = invoke_with_fallback(
            msgs, model=model, temperature=0.0, api_key=api_key,
        )
        return parse_json_obj_from_text(content)
    except Exception as exc:
        logger.warning("Resynthesize failed: {}", exc)
        return {"error": str(exc)}


def _explanation_text_from_dict(expl: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("expected_effect",):
        v = expl.get(key)
        if v:
            parts.append(f"{key}: {v}")
    for key in ("drivers_pos", "drivers_neg", "recommendations"):
        v = expl.get(key)
        if isinstance(v, list) and v:
            parts.append(f"{key}: " + " | ".join(str(x) for x in v))
    return "\n".join(parts)


def _bullets_from_dict(expl: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("drivers_pos", "drivers_neg", "recommendations"):
        v = expl.get(key)
        if isinstance(v, list):
            for x in v:
                s = str(x).strip()
                if s:
                    out.append(re.split(r"[.;\n]", s)[0][:120].lower())
    return out


def run_robustness(
    df: pd.DataFrame,
    client_ids: Sequence[str],
    out_dir: Path,
    *,
    model: str,
    api_key: Optional[str],
    intervention_pool: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """E7: feed identical evidence with two system-prompt variants and compare.

    Implementation note: instead of monkey-patching CausalAgent's templates,
    we reuse the evidence collected by a single full Pipeline run and ask the
    LLM to synthesise a JSON answer twice — once with each system prompt.
    This isolates the focal property (sensitivity to prompt rephrasing of the
    synthesis stage) from any unrelated variation in PSM / RAG / graph
    retrieval.
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("intfloat/multilingual-e5-small")

    rows: List[Dict[str, Any]] = []
    cases_root = out_dir / "robustness_cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    pool = [dict(x) for x in (intervention_pool or INTERVENTION_POOL)]

    for i, cid in enumerate(client_ids):
        delta = pool[i % len(pool)]
        state = _run_one(df, cid, delta, ABLATION_VARIANTS[0])

        if state.get("status") not in ("done", "degraded"):
            rows.append({
                "client_id": cid,
                "delta": json.dumps(delta, ensure_ascii=False),
                "skipped": True,
                "skip_reason": state.get("abort_reason") or state.get("status"),
            })
            continue

        expl_v1 = _resynthesize(state, _ROBUSTNESS_SYSTEM_V1, model, api_key)
        expl_v2 = _resynthesize(state, _ROBUSTNESS_SYSTEM_V2, model, api_key)

        with open(cases_root / f"{cid}.json", "w", encoding="utf-8") as fh:
            json.dump({
                "state": _json_safe(state),
                "expl_v1": expl_v1,
                "expl_v2": expl_v2,
            }, fh, ensure_ascii=False, indent=2)

        t1 = _explanation_text_from_dict(expl_v1) if isinstance(expl_v1, dict) else ""
        t2 = _explanation_text_from_dict(expl_v2) if isinstance(expl_v2, dict) else ""
        sim = 0.0
        if t1 and t2:
            embs = encoder.encode([t1, t2], normalize_embeddings=True, show_progress_bar=False)
            sim = _cosine(np.asarray(embs[0]), np.asarray(embs[1]))

        bullets_v1 = _bullets_from_dict(expl_v1) if isinstance(expl_v1, dict) else []
        bullets_v2 = _bullets_from_dict(expl_v2) if isinstance(expl_v2, dict) else []
        matched = 0
        if bullets_v1 and bullets_v2:
            embs = encoder.encode(bullets_v1 + bullets_v2,
                                  normalize_embeddings=True, show_progress_bar=False)
            embs_v1 = embs[: len(bullets_v1)]
            embs_v2 = embs[len(bullets_v1):]
            for e1 in embs_v1:
                best = max(_cosine(np.asarray(e1), np.asarray(e2)) for e2 in embs_v2)
                if best >= 0.7:
                    matched += 1
        match_rate = matched / max(1, len(bullets_v1))

        rows.append({
            "client_id": cid,
            "delta": json.dumps(delta, ensure_ascii=False),
            "skipped": False,
            "n_bullets_v1": len(bullets_v1),
            "n_bullets_v2": len(bullets_v2),
            "matched_bullets": matched,
            "match_rate": match_rate,
            "cosine_sim": sim,
        })

    pd.DataFrame(rows).to_csv(out_dir / "robustness.csv", index=False)
    return rows


# ---------------------------------------------------------------------------
# Aggregation + cost estimate
# ---------------------------------------------------------------------------

def aggregate(rows: List[Dict[str, Any]],
              scored: List[Dict[str, Any]],
              out_dir: Path) -> Dict[str, Any]:
    df_rows = pd.DataFrame(rows)
    df_scored = pd.DataFrame(scored)

    summary: Dict[str, Any] = {
        "n_total": int(len(df_rows)),
        "by_variant": {},
        "latency": {},
        "rag_refine": {},
        "critic": {},
    }

    if not df_rows.empty:
        for variant_name in df_rows["variant"].unique():
            sub = df_rows[df_rows["variant"] == variant_name]
            sub_scored = df_scored[df_scored["variant"] == variant_name] if not df_scored.empty else pd.DataFrame()
            mean_score = float(sub_scored["mean_score"].dropna().mean()) if "mean_score" in sub_scored else float("nan")
            summary["by_variant"][variant_name] = {
                "n": int(len(sub)),
                "n_done": int((sub["status"] == "done").sum()),
                "n_degraded": int((sub["status"] == "degraded").sum()),
                "n_aborted": int((sub["status"] == "aborted").sum()),
                "pass_rate_critic": float(sub["passed_critic"].mean()) if len(sub) else None,
                "median_latency_ms": float(sub["latency_ms"].median()) if len(sub) else None,
                "p95_latency_ms": float(sub["latency_ms"].quantile(0.95)) if len(sub) else None,
                "mean_judge_score": mean_score if not math.isnan(mean_score) else None,
            }

        full = df_rows[df_rows["variant"] == "full"]
        if not full.empty:
            summary["latency"] = {
                "median_ms": float(full["latency_ms"].median()),
                "p95_ms":    float(full["latency_ms"].quantile(0.95)),
                "max_ms":    float(full["latency_ms"].max()),
                "mean_ms":   float(full["latency_ms"].mean()),
            }
            summary["rag_refine"] = {
                "share_with_refine": float((full["rag_iterations"] >= 2).mean()),
                "mean_iterations":   float(full["rag_iterations"].mean()),
            }
            summary["critic"] = {
                "primary_pass_rate": float(full["passed_critic"].mean()),
                "any_rule_issue":    float((full["rule_issues_n"] > 0).mean()),
                "any_llm_issue":     float((full["llm_issues_n"] > 0).mean()),
                "review_required":   float(full["review_required"].mean()),
            }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 4 validation experiments")
    parser.add_argument("--client-ids", type=str, default="C000100,C000350,C000720,C001100,C001500,C001900,C002300,C002700",
                        help="Comma-separated client IDs to use for ablation grid")
    parser.add_argument("--robustness-clients", type=int, default=5,
                        help="Number of clients to use for robustness experiment (E7)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: reports/chapter4_<timestamp>)")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip the ablation grid (E1, E3, E4, E6); reuse existing cases_summary.csv")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM-as-judge step (E5)")
    parser.add_argument("--skip-robustness", action="store_true",
                        help="Skip prompt robustness experiment (E7)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="LLM model for judge (default: same as agent)")
    parser.add_argument("--judge-workers", type=int, default=4,
                        help="Parallel judge requests")
    parser.add_argument("--judge-rubric", choices=sorted(JUDGE_PROMPTS), default="acceptance",
                        help="Judge rubric: acceptance-oriented bank review (default) or strict top-conference review")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated subset of ablation variants (default: all). "
                             "Allowed: full, no_psm, no_rag, no_graph, llm_only")
    parser.add_argument("--interventions", type=str, default=None,
                        help="Comma-separated intervention names for the cyclic pool. "
                             "Allowed: acquiring, payroll, deposit, credit15, credit25, tariff")
    parser.add_argument("--num-valid-clients", type=int, default=None,
                        help="Auto-select this many clients that pass policy for the cyclic intervention pool")
    parser.add_argument("--scan-limit", type=int, default=None,
                        help="Maximum number of rows to scan when --num-valid-clients is used")
    args = parser.parse_args()

    cfg = get_config()
    api_key = cfg.effective_llm_api_key
    judge_model = args.judge_model or cfg.llm.model_name

    intervention_pool = _parse_intervention_pool(args.interventions)
    out_dir = Path(args.out) if args.out else Path("reports") / f"chapter4_{_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.paths.artifacts_dir / cfg.paths.synthetic_clients_csv)
    if args.num_valid_clients is not None:
        client_ids = _select_applicable_client_ids(
            df, intervention_pool, args.num_valid_clients, scan_limit=args.scan_limit,
        )
    else:
        client_ids = [c.strip() for c in args.client_ids.split(",") if c.strip()]
    logger.info("Chapter 4 experiments → {} (clients: {})", out_dir, client_ids)

    # E1+E3+E4+E6
    rows: List[Dict[str, Any]]
    if args.skip_ablation and (out_dir / "cases_summary.csv").exists():
        rows = pd.read_csv(out_dir / "cases_summary.csv").to_dict("records")
        logger.info("Ablation grid skipped: loaded {} existing rows", len(rows))
    else:
        variants = [v.strip() for v in args.variants.split(",")] if args.variants else None
        rows = run_ablation(
            df, client_ids, out_dir, seed=args.seed, variants=variants,
            intervention_pool=intervention_pool,
        )
        logger.info("Ablation grid done: {} runs", len(rows))

    # E5
    scored: List[Dict[str, Any]] = []
    if args.skip_judge and (out_dir / "judge_scores.csv").exists():
        scored = pd.read_csv(out_dir / "judge_scores.csv").to_dict("records")
        logger.info("Judge skipped: loaded {} existing rows", len(scored))
    elif not args.skip_judge:
        scored = run_judge(rows, out_dir, model=judge_model, api_key=api_key,
                           max_workers=args.judge_workers, rubric=args.judge_rubric)
        logger.info("Judge done: {} verdicts", len(scored))

    # E7 — independent
    robustness_rows: List[Dict[str, Any]] = []
    if not args.skip_robustness:
        rb_clients = client_ids[: args.robustness_clients]
        robustness_rows = run_robustness(
            df, rb_clients, out_dir, model=judge_model, api_key=api_key,
            intervention_pool=intervention_pool,
        )
        logger.info("Robustness done: {} clients", len(robustness_rows))

    # Aggregate
    summary = aggregate(rows, scored, out_dir)
    summary["robustness"] = {
        "n_clients": len(robustness_rows),
        "mean_cosine": float(np.mean([r["cosine_sim"] for r in robustness_rows])) if robustness_rows else None,
        "mean_match_rate": float(np.mean([r["match_rate"] for r in robustness_rows])) if robustness_rows else None,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    logger.info("ALL DONE → {}", out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
