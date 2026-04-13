"""
Graph evaluation module - compares graphs against ground truth and runs bootstrap analysis.
Works with both LLM and algorithmic graphs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
from pathlib import Path
import json
from collections import Counter, defaultdict
import argparse
import tempfile
from datetime import datetime

from sme_causal.data.synth_data import ground_truth_edges
from sme_causal.graph.graph_utils import create_algo_edges
from sme_causal.core.config import get_config
from sme_causal.graph.build_llm_graph import build_llm_graph
from sme_causal.graph.build_algo_graph import build_algo_graph
from sme_causal.graph.build_hybrid_graph import build_hybrid_graph
from sme_causal.graph.build_algo_llm_graph import build_algo_llm_graph


def load_graph_edges(graph_source: Union[str, Path]) -> List[Dict]:
    """Load edges from various json formats"""
    if isinstance(graph_source, (str, Path)):
        path = Path(graph_source)
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                graph = json.load(f)
                if isinstance(graph, Dict):
                    graph = graph["edges"]
                return graph
        raise ValueError(f"Unsupported graph filetype: .json is expected")
    raise ValueError(f"Unsupported graph source: {type(graph_source)}")


def to_key(edge: Dict) -> Tuple[str, str, str]:
    """Convert edge to unique key (source, target, polarity)."""
    return (edge["source"], edge["target"], edge.get("sign") or edge.get("polarity"))


def to_structural_key(edge: Dict) -> Tuple[str, str]:
    """Convert edge to structural key (source, target), ignoring polarity."""
    return (edge["source"], edge["target"])


def build_adjacency_list(edges: List[Dict]) -> Dict[str, set]:
    """Build adjacency list for fast path search."""
    adj = defaultdict(set)
    for edge in edges:
        adj[edge["source"]].add(edge["target"])
    return adj


def evaluate_graph(graph_edges: List[Dict], gt_edges: List[Dict]) -> Dict[str, float]:
    """
    Evaluate graph using standard metrics.
    Returns dictionary with metrics.
    """
    gt_keys = {to_key(e) for e in gt_edges}
    llm_keys = {to_key(e) for e in graph_edges}

    gt_struct_keys = {to_structural_key(e) for e in gt_edges}
    llm_struct_keys = {to_structural_key(e) for e in graph_edges}

    # Exact match with polarity
    TP_set = gt_keys & llm_keys
    FP_set = llm_keys - gt_keys
    FN_set = gt_keys - llm_keys

    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)

    # Reversed Positive (RP): edge exists structurally but direction is opposite
    fp_struct_set = llm_struct_keys - gt_struct_keys
    RP_set = {(s, t) for s, t in fp_struct_set if (t, s) in gt_struct_keys}
    RP = len(RP_set)

    # Adjusted metrics (considering RP)
    total_pred = TP + FP + RP
    total_true = TP + FN + RP

    precision_adj = TP / total_pred if total_pred > 0 else 0
    recall_adj = TP / total_true if total_true > 0 else 0
    f1_adj = (
        2 * (precision_adj * recall_adj) / (precision_adj + recall_adj)
        if (precision_adj + recall_adj) > 0
        else 0
    )

    # Causal Path Fidelity (CPF): A->B->C
    gt_adj = build_adjacency_list(gt_edges)
    llm_adj = build_adjacency_list(graph_edges)

    gt_paths = set()
    for src in gt_adj:
        for mid in gt_adj[src]:
            if mid in gt_adj:
                for end in gt_adj[mid]:
                    gt_paths.add((src, mid, end))

    reproduced_paths = 0
    for src, mid, end in gt_paths:
        if mid in llm_adj.get(src, set()) and end in llm_adj.get(mid, set()):
            reproduced_paths += 1

    cpf = reproduced_paths / len(gt_paths) if gt_paths else 1.0

    return {
        "precision": precision_adj,
        "recall": recall_adj,
        "f1": f1_adj,
        "cpf": cpf,
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "rp": RP,
        "total_gt_paths": len(gt_paths),
        "reproduced_paths": reproduced_paths,
    }


def calculate_ci(
    data: List[float], confidence: float = 0.95
) -> Tuple[float, float, Tuple[float, float]]:
    """Calculate confidence interval for data."""
    n = len(data)
    if n == 0:
        return 0, 0, (0, 0)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    margin = std * 1.96 / np.sqrt(n)  # z=1.96 for 95% CI
    return mean, std, (mean - margin, mean + margin)


def create_report_directory(graph_method: str, bootstrap: bool) -> Path:
    """Create reports directory with descriptive name."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bootstrap_flag = "bootstrap" if bootstrap else "single"
    run_name = f"{graph_method}_{bootstrap_flag}_{timestamp}"

    run_dir = reports_dir / run_name
    run_dir.mkdir(exist_ok=True)

    return run_dir


def save_metrics_and_edges(metrics: Dict, edges_table: pd.DataFrame, report_dir: Path):
    """Save metrics and edge comparison table to report directory."""
    metrics_file = report_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    edges_file = report_dir / "edge_comparison.csv"
    edges_table.to_csv(edges_file, index=False, float_format="%.4f")

    print(f"Results saved to: {report_dir}")
    print(f"   - Metrics: {metrics_file}")
    print(f"   - Edge comparison: {edges_file}")


def create_edge_comparison_table(
    graph_edges: List[Dict], gt_edges: List[Dict]
) -> pd.DataFrame:
    """Create table comparing graph edges with ground truth."""
    gt_keys = {to_key(e) for e in gt_edges}
    graph_keys = {to_key(e) for e in graph_edges}

    table_rows = []

    # Add ground truth edges
    for gt_edge in gt_edges:
        source, target, sign = (
            gt_edge["source"],
            gt_edge["target"],
            gt_edge.get("sign", "+"),
        )
        key = (source, target, sign)

        table_rows.append(
            {
                "From": source,
                "To": target,
                "Sign": sign,
                "In_Ground_Truth": "Yes",
                "Found_In_Graph": "Yes" if key in graph_keys else "No",
            }
        )

    # Add false positive edges
    for edge in graph_edges:
        key = to_key(edge)
        if key not in gt_keys:
            table_rows.append(
                {
                    "From": edge["source"],
                    "To": edge["target"],
                    "Sign": edge.get("sign") or edge.get("polarity", "+"),
                    "In_Ground_Truth": "No",
                    "Found_In_Graph": "Yes",
                }
            )

    return pd.DataFrame(table_rows)


def run_single_evaluation(args, clients_df: pd.DataFrame, gt_edges: List[Dict]) -> Dict:
    cfg = get_config()

    if args.graph_method == "hybrid":
        edges = build_hybrid_graph(df=clients_df)
        print(f"Built LLM graph: {len(edges)} edges")
    elif args.graph_method == "llm":
        edges = build_llm_graph(df=clients_df)
        print(f"Built LLM graph: {len(edges)} edges")
    elif args.graph_method == "algo":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            clients_df.to_csv(tmp.name, index=False)
            edges, _ = build_algo_graph(csv_path=Path(tmp.name))
        edges = create_algo_edges(edges)
        print(f"Built algorithmic graph: {len(edges)} edges")
    elif args.graph_method == "algo_llm":
        edges = build_algo_llm_graph(csv_path=cfg.synthetic_clients_path)
        edges = create_algo_edges(edges)
        print(f"Built algorithmic-LLM graph: {len(edges)} edges")

    metrics = evaluate_graph(edges, gt_edges)
    return edges, metrics


def run_bootstrap_evaluation(
    args, clients_df: pd.DataFrame, gt_edges: List[Dict]
) -> Dict:
    """Run bootstrap evaluation and return results."""

    print(
        f"Running bootstrap evaluation: ({args.bootstrap_reps} reps, {args.sample_frac*100}% samples)"
    )

    bootstrap_results_list = []
    edge_frequency_counter = Counter()
    rng = np.random.default_rng(42)

    for i in range(args.bootstrap_reps):
        print(f"Bootstrap iteration {i+1}/{args.bootstrap_reps}", end="\r")

        # Create bootstrap sample
        sample_idx = rng.choice(
            len(clients_df), size=int(len(clients_df) * args.sample_frac), replace=True
        )
        df_sample = clients_df.iloc[sample_idx].copy()

        edges, metrics = run_single_evaluation(args, df_sample, gt_edges)
        bootstrap_results_list.append(metrics)

        # Count edge frequencies
        for edge in edges:
            key = (edge["source"], edge["target"], edge.get("polarity", "+"))
            edge_frequency_counter[key] += 1

    print("\n" + "=" * 80)
    print(
        f"Bootstrap Analysis ({args.graph_method.upper()} Graph, N={args.bootstrap_reps})"
    )
    print("=" * 80)

    # Calculate bootstrap statistics
    results_df = pd.DataFrame(bootstrap_results_list)
    metrics_summary = {}

    for col in results_df.columns:
        mean, std, ci_ = calculate_ci(results_df[col])
        metrics_summary[col] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_lower": round(ci_[0], 4),
            "ci_upper": round(ci_[1], 4),
        }

    # Print bootstrap statistics
    print("\n--- Bootstrap Statistics ---")
    for metric, vals in metrics_summary.items():
        print(
            f"{metric:<20} | "
            f"Mean: {vals['mean']:.4f} | "
            f"Std: {vals['std']:.4f} | "
            f"95% CI: [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}]"
        )

    bootstrap_edges = []
    frequency_map = {}

    for (source, target, sign), count in edge_frequency_counter.items():
        edge = {"source": source, "target": target, "sign": sign}
        bootstrap_edges.append(edge)
        # Store frequency for later merging
        frequency_map[(source, target, sign)] = round(count / args.bootstrap_reps, 4)

    # Use create_edge_comparison_table
    edges_table = create_edge_comparison_table(bootstrap_edges, gt_edges)

    # Add frequency column
    edges_table["Frequency"] = edges_table.apply(
        lambda row: frequency_map.get((row["From"], row["To"], row["Sign"]), 0.0),
        axis=1,
    )

    # Rename columns to match bootstrap context
    edges_table = edges_table.rename(columns={"Found_In_Graph": "Found_In_Bootstrap"})

    # Sort by frequency
    edges_table = edges_table.sort_values(by=["Frequency"], ascending=False)

    # Print edge table summary
    print(
        f"\n--- Edge Frequency Table (showing top 20 of {len(edges_table)} edges) ---"
    )
    print("-" * 120)
    print(
        f"{'From':<15} {'To':<15} {'Sign':<5} {'Frequency':<8} {'In_GT':<6} {'Found':<6}"
    )
    print("-" * 120)

    for _, row in edges_table.head(20).iterrows():
        color = (
            "🔴"
            if row["In_Ground_Truth"] == "Yes" and row["Found_In_Bootstrap"] == "No"
            else "🟢"
        )
        print(
            f"{color} {row['From']:<13} → {row['To']:<13} {row['Sign']:<5} "
            f"{row['Frequency']:<8} "
            f"{row['In_Ground_Truth']:<6} {row['Found_In_Bootstrap']:<6}"
        )

    return {
        "metrics": metrics_summary,
        "edges_table": edges_table,
        "edge_frequencies": dict(edge_frequency_counter),
        "total_iterations": args.bootstrap_reps,
    }


def main():
    """Standalone evaluation with improved structure"""

    parser = argparse.ArgumentParser(description="Evaluate causal graphs")
    parser.add_argument(
        "--graph-method",
        choices=["llm", "algo", "hybrid", "algo_llm"],
        required=True,
        help="Which graph to evaluate (required)",
    )
    parser.add_argument(
        "--bootstrap", action="store_true", help="Run bootstrap evaluation (optional)"
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10,
        help="Number of bootstrap repetitions (default: 10)",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.8,
        help="Sample fraction for bootstrap (default: 0.8)",
    )

    args = parser.parse_args()
    gt_edges = ground_truth_edges()

    print(f"Evaluating {args.graph_method.upper()} graph against ground truth")
    print("=" * 50)

    # Load graph edges
    cfg = get_config()
    clients_df = pd.read_csv(cfg.synthetic_clients_path)
    print(f"Using synthetic clients data from: {cfg.synthetic_clients_path}")

    # Create report directory
    report_dir = create_report_directory(args.graph_method, args.bootstrap)

    if args.bootstrap:
        # Run bootstrap evaluation
        bootstrap_results = run_bootstrap_evaluation(args, clients_df, gt_edges)
        save_metrics_and_edges(
            bootstrap_results["metrics"], bootstrap_results["edges_table"], report_dir
        )
    else:
        # Run single evaluation
        edges, metrics = run_single_evaluation(args, clients_df, gt_edges)
        edges_table = create_edge_comparison_table(edges, gt_edges)
        save_metrics_and_edges(metrics, edges_table, report_dir)

        # Print basic metrics
        print("\n--- Basic Evaluation Results ---")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric:<20}: {value:.4f}")
            else:
                print(f"{metric:<20}: {value}")


if __name__ == "__main__":
    main()
