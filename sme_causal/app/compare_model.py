from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd
import numpy as np
from loguru import logger

from sme_causal.agent.agent_service import CausalAgent, Explanation
from sme_causal.core.columns import (
    CLIENT_ID,
    NEW_PRODUCT_OFFER,
    NEW_PRODUCT_OFFER_TYPE,
)
from sme_causal.core.config import get_config
from sme_causal.inference.psm import CausalInferenceAnalyzer
from sme_causal.core.utils import sanity_checks


# ============================ ВСПОМОГАТЕЛЬНЫЕ ТИПЫ ============================


@dataclass
class ScenarioResult:
    name: str
    explanation: Explanation
    predicted_effect: Optional[float]  # предсказанный revenue_growth_rate
    psm_effect: Optional[float]        # что взяли из PSM (если было)
    meta: Dict[str, object]            # здесь лежит client_id и прочее


# ============================ УТИЛИТЫ ============================

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
                "error": (
                    f"Treatment '{treatment_col}' is non-binary and non-numeric. "
                    "Provide explicit threshold or pre-binarize."
                ),
            }

    # 3) Initialize and run analyzer
    try:
        analyzer = CausalInferenceAnalyzer(
            target=outcome_col,
            treatment_variable=treatment_col,
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
    }


def choose_primary_psm_effect(psm_metrics: Dict[str, object]) -> Optional[float]:
    """
    Выбираем primary_effect_pp из PSM (ATT vs ATE).
    Здесь реализована простая версия правил: если есть ATT — берём его,
    иначе ATE (с учётом разных ключей).
    """
    if not psm_metrics:
        return None

    att = psm_metrics.get("att")
    if att is None:
        att = psm_metrics.get("ate_matched")

    # несколько вариантов имени ATE на всякий случай
    ate = psm_metrics.get("ate")
    if ate is None:
        ate = psm_metrics.get("ate_naive")
    if ate is None:
        ate = psm_metrics.get("naive_ate")

    def _to_float(v) -> Optional[float]:
        try:
            if v is None:
                return None
            x = float(v)
            if math.isnan(x):
                return None
            return x
        except Exception:
            return None

    att_f = _to_float(att)
    ate_f = _to_float(ate)

    # простое правило: если есть ATT — используем его, иначе ATE
    if att_f is not None:
        return att_f
    return ate_f


_NUM_RE = re.compile(r"[-+]?\d+(\.\d+)?")


def parse_effect_from_text(text: str) -> Optional[float]:
    """
    Наивно вытаскиваем первое число из строки expected_effect.
    Работает, если ты научишь LLM писать что-то вроде:
    "Ожидаемый эффект: +0.004 (п.п.) ..."
    """
    if not text:
        return None
    m = _NUM_RE.search(text.replace(",", "."))
    if not m:
        return None
    try:
        val = float(m.group(0))
        if math.isnan(val):
            return None
        return val
    except Exception:
        return None


def extract_effect(
    explanation: Explanation,
    psm_metrics: Optional[Dict[str, object]],
    allow_psm: bool
) -> Tuple[Optional[float], Optional[float]]:
    """
    Возвращает (predicted_effect, psm_effect).

    Новая логика:
    1. Если allow_psm=True:
        1.1. Пытаемся вытащить эффект из PSM (primary ATT/ATE).
        1.2. Если PSM есть — используем его как predicted_effect.
        1.3. Если PSM нет — fallback: пытаемся вытащить число из LLM-текста.
    2. Если allow_psm=False:
        - используем только эффект из текста LLM.
    """

    # 1) PSM
    psm_effect = choose_primary_psm_effect(psm_metrics) if psm_metrics else None

    # allow_psm=True → сначала пытаемся использовать PSM
    if allow_psm and psm_effect is not None:
        return psm_effect, psm_effect

    # 2) fallback → LLM текст
    effect_from_text = parse_effect_from_text(explanation.expected_effect)

    if effect_from_text is not None:
        return effect_from_text, psm_effect

    # 3) Ни текста, ни PSM
    return None, psm_effect


# ============================ ТРИ СЦЕНАРИЯ МОДЕЛИ ============================


def run_scenario_full(
    agent: CausalAgent,
    df: pd.DataFrame,
    client_id: str,
    delta: Dict[str, object],
    psm_metrics: Optional[Dict[str, object]] = None,
    use_graph: bool = True,
    use_rag: bool = True,
) -> ScenarioResult:
    """
    Сценарий 1:
    - клиентский контекст
    - интервенция
    - PSM
    - граф
    - RAG
    """
    new_df = df.drop("Revenue_Growth_Rate", axis=1, errors="ignore")
    base_ctx = agent.build_context_for_client(new_df, client_id)

    checks_result = {"blocked": False}
    if delta:
        checks_result = sanity_checks(base_ctx, delta)

    if checks_result.get("blocked"):
        expl = None  # type: ignore
        return ScenarioResult(
            name="full_info",
            explanation=expl,
            predicted_effect=0.0,
            psm_effect=0.0,
            meta={"client_id": client_id, "use_graph": use_graph, "use_rag": use_rag},
        )

    psm_result = _run_psm(
        df,
        delta
    )

    if psm_result.get("ok"):
        psm_metrics_for_llm = {
            "att": psm_result.get("att"),
            "ate": psm_result.get("ate"),
        }
    else:
        psm_metrics_for_llm = psm_metrics

    expl = agent.explain_what_if(
        base_ctx=base_ctx,
        delta_changes=delta,
        psm_metrics=psm_metrics_for_llm,
        use_graph=use_graph,
        use_rag=use_rag,
        predict_concrete_target=True,
    )

    predicted_effect, psm_effect = extract_effect(
        explanation=expl, psm_metrics=psm_metrics_for_llm, allow_psm=True
    )

    return ScenarioResult(
        name="full_info",
        explanation=expl,
        predicted_effect=predicted_effect,
        psm_effect=psm_effect,
        meta={"client_id": client_id, "use_graph": use_graph, "use_rag": use_rag},
    )


def run_scenario_client_only(
    agent: CausalAgent,
    df: pd.DataFrame,
    client_id: str,
    delta: Dict[str, object],
) -> ScenarioResult:
    """
    Сценарий 2:
    - только клиентский контекст + интервенция
    - без PSM / графа / RAG
    """
    new_df = df.drop("Revenue_Growth_Rate", axis=1, errors="ignore")
    base_ctx = agent.build_context_for_client(new_df, client_id)

    expl = agent.explain_what_if(
        base_ctx=base_ctx,
        delta_changes=delta,
        psm_metrics=None,
        use_graph=False,
        use_rag=False,
        predict_concrete_target=True,
    )

    predicted_effect, psm_effect = extract_effect(
        explanation=expl, psm_metrics=None, allow_psm=False
    )

    return ScenarioResult(
        name="client_only",
        explanation=expl,
        predicted_effect=predicted_effect,
        psm_effect=psm_effect,
        meta={"client_id": client_id, "use_graph": False, "use_rag": False},
    )


def run_scenario_no_client(
    agent: CausalAgent,
    delta: Dict[str, object],
) -> ScenarioResult:
    """
    Сценарий 3:
    - нет профиля клиента
    - есть только описанная интервенция
    - без PSM / графа / RAG
    """
    base_ctx: Dict[str, object] = {}

    expl = agent.explain_what_if(
        base_ctx=base_ctx,
        delta_changes=delta,
        psm_metrics=None,
        use_graph=False,
        use_rag=False,
        predict_concrete_target=True,
    )

    predicted_effect, psm_effect = extract_effect(
        explanation=expl, psm_metrics=None, allow_psm=False
    )

    return ScenarioResult(
        name="no_client",
        explanation=expl,
        predicted_effect=predicted_effect,
        psm_effect=psm_effect,
        meta={"client_id": None, "use_graph": False, "use_rag": False},
    )


# ============================ ОБЕРТКА ДЛЯ ОДНОГО КЛИЕНТА ============================


def run_all_scenarios_for_client(
    agent: CausalAgent,
    df: pd.DataFrame,
    client_id: str,
    delta: Dict[str, object],
    outcome_col: str = "Revenue_Growth_Rate",
    use_graph: bool = True,
    use_rag: bool = True,
    show_res: bool = False,
) -> Tuple[List[ScenarioResult], Optional[float]]:
    """
    Прогон трёх сценариев (full_info, client_only, no_client) для одного клиента.

    Возвращает:
        - список ScenarioResult
        - true_value (истинный Revenue_Growth_Rate клиента или None)

    Печать результатов управляется флагом show_res.
    """

    # true value из df (если есть)
    true_value: Optional[float] = None
    if outcome_col in df.columns:
        row = df[df[CLIENT_ID] == client_id]
        if not row.empty:
            try:
                val = float(row[outcome_col].iloc[0])
                if not math.isnan(val):
                    true_value = val
            except Exception:
                true_value = None

    # Запуск сценариев
    s1 = run_scenario_full(
        agent=agent,
        df=df,
        client_id=client_id,
        delta=delta,
        use_graph=use_graph,
        use_rag=use_rag,
    )

    s2 = run_scenario_client_only(
        agent=agent,
        df=df,
        client_id=client_id,
        delta=delta,
    )

    s3 = run_scenario_no_client(
        agent=agent,
        delta=delta,
    )

    scenario_results = [s1, s2, s3]

    if show_res:
        print("\n=== Подробные результаты по клиенту ===")
        print("client_id\tscenario\ttrue_value\tpredicted_effect\tpsm_effect\tabs_error")

        for r in scenario_results:
            pred = r.predicted_effect
            psm_eff = r.psm_effect
            abs_err_display = "NA"

            if true_value is not None and pred is not None and not math.isnan(pred):
                err = pred - true_value
                abs_err = abs(err)
                abs_err_display = f"{abs_err:.6f}"

            print(
                f"{client_id}\t"
                f"{r.name}\t"
                f"{true_value if true_value is not None else 'NA'}\t"
                f"{pred if pred is not None else 'NA'}\t"
                f"{psm_eff if psm_eff is not None else 'NA'}\t"
                f"{abs_err_display}"
            )

    return scenario_results, true_value


# ============================ MAIN ============================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Сравнение одной и той же LLM-модели в трёх режимах знания "
            "(full_info / client_only / no_client) для заданной интервенции."
        )
    )

    parser.add_argument(
        "--client-id",
        required=False,
        default=None,
        help="ID клиента (значение колонки CLIENT_ID в датасете). "
             "Если не указан, берутся первые 5 клиентов из датасета.",
    )

    parser.add_argument(
        "--what-if",
        type=str,
        default=None,
        help="Comma-separated key=value pairs for scenario changes (e.g. 'New_Product_Offer=1')",
    )

    parser.add_argument(
        "--psm-json",
        type=Path,
        default=None,
        help=(
            "Опциональный JSON-файл с PSM-метриками "
            '(ключи: "att", "ate" / "ate_naive" и т.п.). '
            "Если задан, используется в сценарии full_info."
        ),
    )
    parser.add_argument(
        "--graph-method",
        choices=["hybrid", "algo", "llm", "algo_llm"],
        default=None,
        help="Какой вариант графа подключать в CausalAgent (опционально).",
    )

    args = parser.parse_args()
    cfg = get_config()

    # Загружаем датасет
    csv_path: Path = cfg.synthetic_clients_path

    if not csv_path.exists():
        logger.error(f"Dataset not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset from {csv_path} (rows={len(df)})")

    if CLIENT_ID not in df.columns:
        logger.error(f"Column '{CLIENT_ID}' not found in dataset.")
        return

    # Определяем список клиентов для анализа
    if args.client_id is not None:
        if args.client_id not in set(df[CLIENT_ID].tolist()):
            logger.error(f"Client_ID not found in dataset: {args.client_id}")
            return
        client_ids = [args.client_id]
        logger.info(f"Using single client_id: {args.client_id}")
    else:
        unique_ids = df[CLIENT_ID].dropna().unique().tolist()
        if not unique_ids:
            logger.error("No client IDs found in dataset.")
            return
        client_ids = unique_ids[:5]
        logger.info(f"No --client-id provided. Using first 5 clients: {client_ids}")

    # Разбираем what-if
    if args.what_if:
        delta = _parse_kv_pairs(args.what_if)
        logger.info(f"Using explicit what-if parameters: {delta}")
    else:
        delta = {NEW_PRODUCT_OFFER: 1, NEW_PRODUCT_OFFER_TYPE: "acquiring"}
        logger.info(
            "No --what-if provided, using default: "
            "New_Product_Offer=1, New_Product_Offer_Type=acquiring"
        )

    # Инициализируем агента
    agent = CausalAgent(graph_method=args.graph_method)

    # ------------------ ЗАПУСК СЦЕНАРИЕВ ДЛЯ ВСЕХ КЛИЕНТОВ ------------------

    all_results: List[ScenarioResult] = []

    outcome_col = "Revenue_Growth_Rate"
    if outcome_col not in df.columns:
        logger.warning(
            f"Column '{outcome_col}' not found in dataset. "
            "MSE/MAE по этой метрике посчитать не получится."
        )

    # Для расчёта метрик: собираем ошибки по каждому сценарию
    errors_by_scenario: Dict[str, List[float]] = {
        "full_info": [],
        "client_only": [],
        "no_client": [],
    }

    for cid in client_ids:
        logger.info(f"Running scenarios for client_id={cid!r}")

        scenario_results, true_value = run_all_scenarios_for_client(
            agent=agent,
            df=df,
            client_id=cid,
            delta=delta,
            outcome_col=outcome_col,
            use_graph=(args.graph_method is not None),
            use_rag=True,
            show_res=True,  # можно поставить False, если не нужен подробный вывод
        )

        all_results.extend(scenario_results)

        # накапливаем ошибки для MSE/MAE
        if true_value is not None:
            for r in scenario_results:
                pred = r.predicted_effect
                if pred is None or math.isnan(pred):
                    continue
                err = pred - true_value
                errors_by_scenario[r.name].append(err)

    # ------------------ СВОДКА MSE / MAE ПО СЦЕНАРИЯМ ------------------

    print("\n=== MSE / MAE по сценариям (по колонке Revenue_Growth_Rate) ===")
    for scenario_name, errs in errors_by_scenario.items():
        if errs:
            mse = sum(e * e for e in errs) / len(errs)
            mae = sum(abs(e) for e in errs) / len(errs)
            print(f"{scenario_name}\tMSE={mse:.8f}\tMAE={mae:.8f}\t(n={len(errs)})")
        else:
            print(f"{scenario_name}\tMSE=NA\tMAE=NA\t(n=0)")

    # При необходимости можно раскомментировать сохранение объяснений:
    # with open("compare_explanations.json", "w", encoding="utf-8") as f:
    #     json.dump(
    #         {
    #             r.name + f"_{r.meta.get('client_id')}": {
    #                 "client_id": r.meta.get("client_id"),
    #                 "predicted_effect": r.predicted_effect,
    #                 "psm_effect": r.psm_effect,
    #                 "explanation": r.explanation.to_dict(),
    #             }
    #             for r in all_results
    #         },
    #         f,
    #         ensure_ascii=False,
    #         indent=2,
    #     )


if __name__ == "__main__":
    main()
