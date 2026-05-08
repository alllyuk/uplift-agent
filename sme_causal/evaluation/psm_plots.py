"""Visualisations for PSM ground-truth evaluation.

Produces the forest plot used in the thesis: for each intervention and
generation regime (randomized / confounded), shows the PSM and naive ATT
estimates with their 95 %-bootstrap intervals next to the empirical true
ATT marker.

Run (после `evaluation.psm` в обоих режимах):
    python -m sme_causal.evaluation.psm_plots \
        --randomized reports/psm_eval_randomized \
        --confounded reports/psm_eval_confounded \
        --out reports/psm_forest.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INTERVENTION_ORDER = [
    ("New_Product_Offer",            "Предложение продукта"),
    ("Credit_Limit_Change_positive", "Положительное изм. лимита"),
    ("Tariff_Discount",              "Тарифная скидка"),
]


def _draw_panel(ax: plt.Axes, summary: pd.DataFrame, title: str) -> None:
    """Render one panel of the forest plot (randomized OR confounded)."""
    rows = []
    for key, label in INTERVENTION_ORDER:
        r = summary.loc[summary["intervention"] == key]
        if r.empty:
            continue
        rec = r.iloc[0]
        rows.append((label, rec))

    n_rows = len(rows)
    # Each intervention occupies a vertical slot. Two estimators per slot
    # (PSM, naive) drawn slightly above/below the slot centre, plus the true
    # ATT marker exactly at the centre.
    ax.set_xlabel("ATT")
    ax.set_title(title, fontsize=12)
    ax.set_yticks([])
    ax.set_xlim(0.0, max(0.05, summary["psm_att_ci_high"].max() * 1.15))
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    psm_color = "#1f77b4"
    naive_color = "#7f7f7f"
    true_color = "#d62728"

    psm_y_offset = 0.18
    naive_y_offset = -0.18

    for i, (label, rec) in enumerate(rows):
        y = (n_rows - 1 - i)
        # PSM
        ax.errorbar(
            rec["psm_att_mean"], y + psm_y_offset,
            xerr=[[rec["psm_att_mean"] - rec["psm_att_ci_low"]],
                  [rec["psm_att_ci_high"] - rec["psm_att_mean"]]],
            fmt="o", color=psm_color, markersize=7, capsize=4,
            label="PSM (95 % CI)" if i == 0 else None,
        )
        # Naive
        ax.errorbar(
            rec["naive_att_mean"], y + naive_y_offset,
            xerr=[[rec["naive_att_mean"] - rec["naive_att_ci_low"]],
                  [rec["naive_att_ci_high"] - rec["naive_att_mean"]]],
            fmt="s", color=naive_color, markersize=6, capsize=4,
            label="Наивная разность (95 % CI)" if i == 0 else None,
        )
        # True empirical ATT marker
        ax.plot(
            rec["true_att_empirical"], y,
            marker="*", color=true_color, markersize=14,
            label="Истинный ATT (эмпирический)" if i == 0 else None,
        )
        # Intervention label on the left
        ax.text(
            -0.02, y, label, ha="right", va="center",
            transform=ax.get_yaxis_transform(), fontsize=10,
        )

    ax.set_ylim(-0.6, n_rows - 0.4)


def make_forest(
    randomized_dir: Path, confounded_dir: Path, out_path: Path
) -> None:
    rand_summary = pd.read_csv(randomized_dir / "summary.csv")
    conf_summary = pd.read_csv(confounded_dir / "summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), sharey=True)
    _draw_panel(axes[0], rand_summary, "Случайное назначение интервенций")
    _draw_panel(axes[1], conf_summary, "Конфаундированное назначение")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.tight_layout(rect=(0.04, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forest plot for PSM evaluation.")
    parser.add_argument("--randomized", required=True, type=str)
    parser.add_argument("--confounded", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    args = parser.parse_args()

    make_forest(
        Path(args.randomized), Path(args.confounded), Path(args.out),
    )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
