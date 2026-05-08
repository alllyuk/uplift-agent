"""Plots for RAG retrieval evaluation.

Produces a single figure with two panels:
- chunk-level Recall@k (strict gold = source chunk),
- document-level Recall@k (any chunk from the source document).

Both panels overlay literal and paraphrase splits to make the
robustness gap visible.

Run (after `evaluation.rag_retrieval`):
    python -m sme_causal.evaluation.rag_plots \\
        --in-dir reports/rag_eval_<timestamp> \\
        --out reports/rag_recall_at_k.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


_CHUNK_RE = re.compile(r"^recall@(\d+)$")
_DOC_RE = re.compile(r"^doc_recall@(\d+)$")


def _extract_curve(row: pd.Series, pattern: re.Pattern) -> list[tuple[int, float]]:
    pairs = []
    for col, value in row.items():
        m = pattern.match(str(col))
        if m:
            pairs.append((int(m.group(1)), float(value)))
    pairs.sort(key=lambda p: p[0])
    return pairs


_KIND_LABEL = {
    "literal": "Дословный запрос",
    "paraphrase": "Перефразированный запрос",
}
_KIND_COLOR = {
    "literal": "#1f77b4",
    "paraphrase": "#d62728",
}


def make_plot(
    in_dir: Path,
    out_path: Path,
    embedding_model: str | None = None,
) -> None:
    summary = pd.read_csv(in_dir / "summary.csv")
    if "embedding_model" in summary.columns and embedding_model is not None:
        summary = summary[summary["embedding_model"] == embedding_model]
        if summary.empty:
            raise SystemExit(
                f"no rows for embedding_model={embedding_model!r} in {in_dir/'summary.csv'}"
            )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = [
        (axes[0], _CHUNK_RE, "Chunk-level Recall@k"),
        (axes[1], _DOC_RE, "Document-level Recall@k"),
    ]

    for ax, pattern, title in panels:
        for _, row in summary.iterrows():
            kind = str(row["split"])
            if kind not in _KIND_LABEL:
                continue  # skip 'all'
            curve = _extract_curve(row, pattern)
            if not curve:
                continue
            xs = [k for k, _ in curve]
            ys = [v for _, v in curve]
            ax.plot(
                xs, ys,
                marker="o", linewidth=2, markersize=7,
                color=_KIND_COLOR[kind],
                label=_KIND_LABEL[kind],
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("k")
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks([1, 3, 5, 10])
        ax.grid(linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Recall@k")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RAG retrieval recall@k.")
    parser.add_argument("--in-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--embedding-model", type=str, default=None,
        help="when sweep summary.csv has an embedding_model column, filter to this one",
    )
    args = parser.parse_args()
    make_plot(args.in_dir, args.out, embedding_model=args.embedding_model)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
