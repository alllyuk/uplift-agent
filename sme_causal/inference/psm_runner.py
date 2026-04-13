"""PSM runner: thin wrapper around CausalInferenceAnalyzer for pipeline use.

Extracted from app/run.py to allow reuse in the orchestrator pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from sme_causal.inference.psm import CausalInferenceAnalyzer

PSM_MIN_GROUP_SIZE = 100


def _is_finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


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


def run_psm(
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
    """Run PSM analysis via CausalInferenceAnalyzer.

    Returns a dict with ok=True/False and metrics (att, ate, n_pairs, etc.).
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
            threshold=threshold,
            replacement=(match_ratio != 1),
            caliper=caliper,
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
        treated_col_name = "__treated__"
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
        "match_ratio": match_ratio,
        "replacement": (match_ratio != 1),
        "att": att,
        "ate": ate,
        "n_pairs": n_pairs,
        "n_treated": n_treated,
        "n_control": n_control,
        **reliability,
    }
