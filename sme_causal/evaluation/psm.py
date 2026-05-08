"""Empirical evaluation of PSM against ground-truth treatment effects.

Validates the propensity-score-matching module on synthetic data where the
true per-client uplift is *empirical*: drawn at generation time and stored in
``_uplift_*_factual`` columns of the DataFrame produced by
:func:`generate_with_counterfactuals`. This removes any distance between the
data and the ground truth used as the benchmark.

Each evaluated intervention gets three groups of metrics:

1.  **Bias of PSM ATT.**
    For every bootstrap replication we compute both the PSM estimate of ATT
    and the *empirical* true ATT on the same subsample (``mean of factual
    uplift on actually-treated rows in that subsample``). The bias is taken
    pairwise: ``bias_k = psm_att_k − true_att_subsample_k``. This isolates
    matching error from mere subsample variation.

2.  **Covariate balance (standardised mean differences).**
    For every numeric covariate we compute the SMD between treated and control
    groups before and after matching. Cohen's threshold is ``|SMD| < 0.1`` for
    good balance, ``< 0.25`` for acceptable balance.

3.  **Bootstrap distribution of the PSM estimator.**
    Mean estimate, 95 %-bootstrap CI, mean absolute error, fraction of
    replications whose CI covers the *fixed* (pre-bootstrap) true ATT.

Outputs:
    reports/psm_eval_<timestamp>/
        summary.csv    — aggregated metrics per intervention
        bootstrap.csv  — per-replication ATT, true ATT on subsample, balance
        details.json   — full numeric snapshot, including config

Run:
    python -m sme_causal.evaluation.psm --n-bootstrap 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from sme_causal.core.config import get_config
from sme_causal.data.synth_data import SynthConfig, generate_with_counterfactuals
from sme_causal.inference.psm_runner import run_psm

# Covariates passed to PSM (matches the production Pipeline default set).
DEFAULT_COVARIATES: List[str] = [
    "Industry",
    "Region",
    "Business_Size",
    "Avg_Account_Balance",
    "Avg_Monthly_Inflow",
    "Avg_Monthly_Outflow",
    "Num_Products",
]

# Subset of numeric covariates for which SMD is meaningful directly. For
# categorical covariates SMD is reported per one-hot level inside `_balance`.
NUMERIC_COVARIATES: List[str] = [
    "Avg_Account_Balance",
    "Avg_Monthly_Inflow",
    "Avg_Monthly_Outflow",
    "Num_Products",
]
CATEGORICAL_COVARIATES: List[str] = ["Industry", "Region", "Business_Size"]


# Each entry: (intervention key in `true_effects`, treatment column in df,
# factual-uplift column in df, human-readable label).
INTERVENTIONS: List[Tuple[str, str, str, str]] = [
    ("New_Product_Offer",            "New_Product_Offer", "_uplift_offer_factual",    "Предложение нового продукта"),
    ("Credit_Limit_Change_positive", "T_credit_positive", "_uplift_credit_factual",   "Положительное изменение кредитного лимита"),
    ("Tariff_Discount",              "Tariff_Discount",   "_uplift_discount_factual", "Тарифная скидка"),
]


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------

def _smd(treated_vals: np.ndarray, control_vals: np.ndarray) -> float:
    """Standardised mean difference: (μ_T − μ_C) / sqrt((σ²_T + σ²_C) / 2)."""
    if len(treated_vals) == 0 or len(control_vals) == 0:
        return float("nan")
    mean_t = float(np.mean(treated_vals))
    mean_c = float(np.mean(control_vals))
    var_t = float(np.var(treated_vals, ddof=1)) if len(treated_vals) > 1 else 0.0
    var_c = float(np.var(control_vals, ddof=1)) if len(control_vals) > 1 else 0.0
    pooled = math.sqrt((var_t + var_c) / 2.0) if (var_t + var_c) > 0 else 0.0
    if pooled == 0.0:
        return float("nan")
    return (mean_t - mean_c) / pooled


def _expand_categorical(
    df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode the listed categorical columns. Returns a frame
    containing only the encoded indicators and the list of indicator names."""
    if not columns:
        return df.iloc[:, 0:0], []
    encoded = pd.get_dummies(df[columns], drop_first=False)
    return encoded, list(encoded.columns)


def _balance_diagnostics(
    full_df: pd.DataFrame,
    matched_df: Optional[pd.DataFrame],
    treatment_col: str,
) -> Dict[str, float]:
    """Aggregate SMDs across covariates before and after matching.

    Returns means and maxima of |SMD| over all numeric covariates and one-hot
    indicators of categorical covariates."""
    full_treated_mask = full_df[treatment_col].to_numpy() == 1

    # Pre-match SMDs.
    pre_smds: List[float] = []
    for col in NUMERIC_COVARIATES:
        s_t = full_df.loc[full_treated_mask, col].to_numpy()
        s_c = full_df.loc[~full_treated_mask, col].to_numpy()
        pre_smds.append(_smd(s_t, s_c))
    cat_full, cat_levels = _expand_categorical(full_df, CATEGORICAL_COVARIATES)
    for level in cat_levels:
        s_t = cat_full.loc[full_treated_mask, level].to_numpy(dtype=float)
        s_c = cat_full.loc[~full_treated_mask, level].to_numpy(dtype=float)
        pre_smds.append(_smd(s_t, s_c))
    pre_smds_arr = np.array([v for v in pre_smds if not math.isnan(v)])

    # Post-match SMDs (matched_df may be missing if PSM failed).
    post_smds_arr: np.ndarray = np.array([], dtype=float)
    if matched_df is not None and len(matched_df) > 0 and "__treated__" in matched_df.columns:
        m_treated_mask = matched_df["__treated__"].to_numpy() == 1
        post_smds: List[float] = []
        for col in NUMERIC_COVARIATES:
            if col in matched_df.columns:
                s_t = matched_df.loc[m_treated_mask, col].to_numpy()
                s_c = matched_df.loc[~m_treated_mask, col].to_numpy()
                post_smds.append(_smd(s_t, s_c))
        cat_matched_avail = [c for c in CATEGORICAL_COVARIATES if c in matched_df.columns]
        if cat_matched_avail:
            cat_matched, cat_levels_m = _expand_categorical(matched_df, cat_matched_avail)
            for level in cat_levels_m:
                s_t = cat_matched.loc[m_treated_mask, level].to_numpy(dtype=float)
                s_c = cat_matched.loc[~m_treated_mask, level].to_numpy(dtype=float)
                post_smds.append(_smd(s_t, s_c))
        post_smds_arr = np.array([v for v in post_smds if not math.isnan(v)])

    def _stat(arr: np.ndarray, fn) -> float:
        return float(fn(np.abs(arr))) if arr.size > 0 else float("nan")

    return {
        "pre_smd_mean_abs": _stat(pre_smds_arr, np.mean),
        "pre_smd_max_abs": _stat(pre_smds_arr, np.max),
        "post_smd_mean_abs": _stat(post_smds_arr, np.mean),
        "post_smd_max_abs": _stat(post_smds_arr, np.max),
    }


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class BootstrapSample:
    intervention: str
    iteration: int
    psm_att: Optional[float]
    naive_att: Optional[float]  # mean(Y | treated) − mean(Y | control), no matching
    true_att_subsample: Optional[float]
    pairwise_bias_psm: Optional[float]    # psm_att − true_att_subsample
    pairwise_bias_naive: Optional[float]  # naive_att − true_att_subsample
    n_pairs: Optional[int]
    n_treated: Optional[int]
    n_control: Optional[int]
    psm_reliable: bool
    pre_smd_mean_abs: float
    pre_smd_max_abs: float
    post_smd_mean_abs: float
    post_smd_max_abs: float


@dataclass
class InterventionSummary:
    intervention: str
    label: str

    # Ground truth on the full dataset (does not change between bootstraps).
    true_ate_empirical: float
    true_att_empirical: float
    true_ate_analytical: float
    true_att_analytical: float
    n_treated_factual: int

    # PSM estimator: bootstrap distribution.
    psm_att_mean: float
    psm_att_ci_low: float
    psm_att_ci_high: float
    psm_att_covers_fixed_true: bool

    # Naive estimator (difference of means without matching) — biased under
    # confounding, included as a sanity baseline.
    naive_att_mean: float
    naive_att_ci_low: float
    naive_att_ci_high: float

    # Pairwise bias (correct measure of matching error).
    pairwise_bias_mean: float
    pairwise_bias_abs_mean: float
    pairwise_bias_ci_low: float
    pairwise_bias_ci_high: float
    pairwise_bias_ci_contains_zero: bool

    # Pairwise bias of the naive estimator — quantifies confounding strength.
    naive_pairwise_bias_mean: float
    naive_pairwise_bias_ci_low: float
    naive_pairwise_bias_ci_high: float

    # Balance diagnostics, averaged over bootstrap.
    pre_smd_mean_abs_avg: float
    post_smd_mean_abs_avg: float
    post_smd_max_abs_avg: float
    post_smd_max_abs_p95: float

    n_bootstrap: int


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------

def _bootstrap_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=n, size=n)


def evaluate_psm(
    cfg: SynthConfig,
    *,
    n_bootstrap: int,
    caliper: float = 0.05,
    covariates: Optional[List[str]] = None,
) -> Tuple[List[InterventionSummary], List[BootstrapSample]]:
    covariates = covariates or DEFAULT_COVARIATES

    df, true_effects = generate_with_counterfactuals(cfg)
    n = len(df)
    logger.info("Synthetic dataset: n={}, covariates={}", n, covariates)

    summaries: List[InterventionSummary] = []
    samples: List[BootstrapSample] = []

    for intervention_key, treatment_col, factual_col, label in INTERVENTIONS:
        eff = true_effects[intervention_key]
        true_ate_emp = eff["ate"]
        true_att_emp = eff["att"]
        true_ate_an = eff["ate_analytical"]
        true_att_an = eff["att_analytical"]
        n_treated_factual = int(eff["n_treated_factual"])

        logger.info(
            "Intervention '{}': empirical true ATT={:.4f} (analytical {:.4f}); "
            "n_treated_factual={}",
            intervention_key, true_att_emp, true_att_an, n_treated_factual,
        )

        psm_atts: List[float] = []
        naive_atts: List[float] = []
        pairwise_biases: List[float] = []
        naive_pairwise_biases: List[float] = []
        post_smd_means: List[float] = []
        post_smd_maxes: List[float] = []
        pre_smd_means: List[float] = []

        for k in range(n_bootstrap):
            idx = _bootstrap_indices(n, seed=cfg.seed + 1 + k)
            df_b = df.iloc[idx].reset_index(drop=True)

            # PSM estimate on subsample. `return_matched_df=True` is required
            # for the post-match SMD diagnostics below; production code paths
            # leave it disabled to keep `psm_result` light.
            res = run_psm(
                df_b, {treatment_col: 1},
                outcome_col="Revenue_Growth_Rate",
                treatment_col=treatment_col,
                covariates=covariates,
                caliper=caliper,
                return_matched_df=True,
            )

            psm_att = res.get("att") if res.get("ok") else None
            n_pairs = res.get("n_pairs")
            n_treated = res.get("n_treated")
            n_control = res.get("n_control")
            reliable = bool(res.get("psm_reliable", False))
            matched_df = res.get("matched_df")

            # Empirical true ATT on the *same* subsample.
            treated_mask_b = df_b[treatment_col].to_numpy() == 1
            outcome_b = df_b["Revenue_Growth_Rate"].to_numpy()
            true_att_b = (
                float(df_b.loc[treated_mask_b, factual_col].mean())
                if treated_mask_b.any()
                else float("nan")
            )

            # Naive estimator: difference of mean outcome between treated and
            # control without matching. Biased under confounding, useful as a
            # baseline that shows how much PSM corrects.
            naive_att_b: Optional[float]
            if treated_mask_b.any() and (~treated_mask_b).any():
                naive_att_b = float(
                    outcome_b[treated_mask_b].mean()
                    - outcome_b[~treated_mask_b].mean()
                )
            else:
                naive_att_b = None

            psm_att_f = (
                float(psm_att)
                if psm_att is not None and np.isfinite(psm_att)
                else None
            )
            pairwise_bias_psm = (
                psm_att_f - true_att_b
                if psm_att_f is not None and not math.isnan(true_att_b)
                else None
            )
            pairwise_bias_naive = (
                naive_att_b - true_att_b
                if naive_att_b is not None and not math.isnan(true_att_b)
                else None
            )

            balance = _balance_diagnostics(df_b, matched_df, treatment_col)

            sample = BootstrapSample(
                intervention=intervention_key,
                iteration=k,
                psm_att=psm_att_f,
                naive_att=naive_att_b,
                true_att_subsample=None if math.isnan(true_att_b) else true_att_b,
                pairwise_bias_psm=pairwise_bias_psm,
                pairwise_bias_naive=pairwise_bias_naive,
                n_pairs=int(n_pairs) if n_pairs is not None else None,
                n_treated=int(n_treated) if n_treated is not None else None,
                n_control=int(n_control) if n_control is not None else None,
                psm_reliable=reliable,
                pre_smd_mean_abs=balance["pre_smd_mean_abs"],
                pre_smd_max_abs=balance["pre_smd_max_abs"],
                post_smd_mean_abs=balance["post_smd_mean_abs"],
                post_smd_max_abs=balance["post_smd_max_abs"],
            )
            samples.append(sample)

            if psm_att_f is not None:
                psm_atts.append(psm_att_f)
            if naive_att_b is not None:
                naive_atts.append(naive_att_b)
            if pairwise_bias_psm is not None:
                pairwise_biases.append(pairwise_bias_psm)
            if pairwise_bias_naive is not None:
                naive_pairwise_biases.append(pairwise_bias_naive)
            if not math.isnan(balance["pre_smd_mean_abs"]):
                pre_smd_means.append(balance["pre_smd_mean_abs"])
            if not math.isnan(balance["post_smd_mean_abs"]):
                post_smd_means.append(balance["post_smd_mean_abs"])
            if not math.isnan(balance["post_smd_max_abs"]):
                post_smd_maxes.append(balance["post_smd_max_abs"])

        if not psm_atts:
            logger.warning("No successful PSM runs for '{}'", intervention_key)
            continue

        psm_att_arr = np.asarray(psm_atts)
        naive_att_arr = np.asarray(naive_atts) if naive_atts else np.asarray([np.nan])
        pb_arr = np.asarray(pairwise_biases) if pairwise_biases else np.asarray([np.nan])
        npb_arr = (
            np.asarray(naive_pairwise_biases)
            if naive_pairwise_biases
            else np.asarray([np.nan])
        )

        att_ci_low, att_ci_high = np.quantile(psm_att_arr, [0.025, 0.975])
        naive_ci_low, naive_ci_high = (
            np.quantile(naive_att_arr, [0.025, 0.975])
            if naive_atts
            else (float("nan"), float("nan"))
        )
        pb_ci_low, pb_ci_high = (
            np.quantile(pb_arr, [0.025, 0.975])
            if pairwise_biases
            else (float("nan"), float("nan"))
        )
        npb_ci_low, npb_ci_high = (
            np.quantile(npb_arr, [0.025, 0.975])
            if naive_pairwise_biases
            else (float("nan"), float("nan"))
        )

        post_smd_max_arr = np.asarray(post_smd_maxes) if post_smd_maxes else np.asarray([np.nan])

        summary = InterventionSummary(
            intervention=intervention_key,
            label=label,
            true_ate_empirical=true_ate_emp,
            true_att_empirical=true_att_emp,
            true_ate_analytical=true_ate_an,
            true_att_analytical=true_att_an,
            n_treated_factual=n_treated_factual,
            psm_att_mean=float(psm_att_arr.mean()),
            psm_att_ci_low=float(att_ci_low),
            psm_att_ci_high=float(att_ci_high),
            psm_att_covers_fixed_true=bool(att_ci_low <= true_att_emp <= att_ci_high),
            naive_att_mean=float(naive_att_arr.mean()) if naive_atts else float("nan"),
            naive_att_ci_low=float(naive_ci_low),
            naive_att_ci_high=float(naive_ci_high),
            pairwise_bias_mean=float(pb_arr.mean()) if pairwise_biases else float("nan"),
            pairwise_bias_abs_mean=float(np.mean(np.abs(pb_arr))) if pairwise_biases else float("nan"),
            pairwise_bias_ci_low=float(pb_ci_low),
            pairwise_bias_ci_high=float(pb_ci_high),
            pairwise_bias_ci_contains_zero=bool(
                pb_ci_low <= 0.0 <= pb_ci_high
            ) if pairwise_biases else False,
            naive_pairwise_bias_mean=float(npb_arr.mean()) if naive_pairwise_biases else float("nan"),
            naive_pairwise_bias_ci_low=float(npb_ci_low),
            naive_pairwise_bias_ci_high=float(npb_ci_high),
            pre_smd_mean_abs_avg=float(np.mean(pre_smd_means)) if pre_smd_means else float("nan"),
            post_smd_mean_abs_avg=float(np.mean(post_smd_means)) if post_smd_means else float("nan"),
            post_smd_max_abs_avg=float(np.mean(post_smd_max_arr)),
            post_smd_max_abs_p95=float(np.quantile(post_smd_max_arr, 0.95)),
            n_bootstrap=len(psm_atts),
        )
        summaries.append(summary)

        logger.info(
            "  -> PSM ATT={:.4f} (bias {:+.4f}, CI contains 0: {}); "
            "naive ATT={:.4f} (bias {:+.4f}); "
            "balance: pre |SMD|≈{:.3f} → post |SMD|≈{:.3f}",
            summary.psm_att_mean,
            summary.pairwise_bias_mean,
            summary.pairwise_bias_ci_contains_zero,
            summary.naive_att_mean,
            summary.naive_pairwise_bias_mean,
            summary.pre_smd_mean_abs_avg,
            summary.post_smd_mean_abs_avg,
        )

    return summaries, samples


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------

def save_results(
    summaries: List[InterventionSummary],
    samples: List[BootstrapSample],
    output_dir: Path,
    *,
    cfg: SynthConfig,
    n_bootstrap: int,
    caliper: float,
    covariates: List[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([asdict(s) for s in summaries])
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8")

    samples_df = pd.DataFrame([asdict(s) for s in samples])
    samples_df.to_csv(output_dir / "bootstrap.csv", index=False, encoding="utf-8")

    details = {
        "config": {
            "n_clients": cfg.n_clients,
            "seed": cfg.seed,
            "confounded": cfg.confounded,
            "n_bootstrap": n_bootstrap,
            "caliper": caliper,
            "covariates": covariates,
            "numeric_covariates": NUMERIC_COVARIATES,
            "categorical_covariates": CATEGORICAL_COVARIATES,
        },
        "summaries": [asdict(s) for s in summaries],
    }
    with (output_dir / "details.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    logger.success("Results saved to {}", output_dir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empirical PSM ground-truth evaluation on synthetic data.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=30)
    parser.add_argument("--n-clients", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--caliper", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--confounded",
        action="store_true",
        help="Use the confounded data-generating mode where intervention "
             "probabilities depend on covariates (industry, business size, "
             "profit, liquidity pressure, price sensitivity). Without this "
             "flag the assignment is almost-randomized and PSM cannot show "
             "its strength.",
    )
    args = parser.parse_args()

    cfg = SynthConfig(
        n_clients=args.n_clients,
        seed=args.seed,
        confounded=args.confounded,
    )
    summaries, samples = evaluate_psm(
        cfg, n_bootstrap=args.n_bootstrap, caliper=args.caliper,
        covariates=DEFAULT_COVARIATES,
    )

    if args.output_dir:
        out = Path(args.output_dir)
    else:
        app_cfg = get_config()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = app_cfg.paths.artifacts_dir.parent / "reports" / f"psm_eval_{timestamp}"

    save_results(
        summaries, samples, out,
        cfg=cfg, n_bootstrap=args.n_bootstrap,
        caliper=args.caliper, covariates=DEFAULT_COVARIATES,
    )


if __name__ == "__main__":
    main()
