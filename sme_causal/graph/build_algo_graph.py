"""
Algorithmic causal graph construction module.
Combines multiple causal discovery algorithms with bootstrap consensus.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import networkx as nx

import os
import time
import inspect
import itertools
from dataclasses import dataclass, field

from collections import Counter
from sklearn.preprocessing import KBinsDiscretizer
from tqdm.auto import trange

from pgmpy.estimators import ExpertKnowledge, GES, HillClimbSearch, PC
from pgmpy.estimators.CITests import chi_square

from sme_causal.data.synth_data import LAYER_ORDER
from sme_causal.graph.graph_utils import export_graph, adjust_robustness
from sme_causal.graph.graph_viz import build_pyvis_html
from sme_causal.core.config import get_config

import logging
import warnings

# Отключаем логи pgmpy
logging.getLogger("pgmpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

# Отключаем лишние логи sklearn
logging.getLogger("sklearn").setLevel(logging.WARNING)


OUTPUT_DIR = Path("causal_outputs")
# ====== ОПИСАНИЕ СТОЛБЦОВ ПО ПРОЕКТУ ======
# Интервенции (управленческие воздействия)
INTERVENTIONS = [
    "New_Product_Offer",  # 0/1
    "New_Product_Offer_Type",  # Категориальный
    "Credit_Limit_Change",  # непрерывный
    "Tariff_Discount",  # 0/1
]

# Целевые метрики (outcomes)
OUTCOMES = [
    "Revenue_Growth_Rate",  # непрерывный
    "Revenue_Trend",  # категориальный (up/flat/down)
]

# Базовые "квази-экзогенные" атрибуты
EXOGENOUS = [
    "Industry",
    "Region",
    "Business_Size",
    "Years_in_Operation",
    "Client_Tenure",
]

NUMERIC_CANDIDATES = [
    "Years_in_Operation",
    "Client_Tenure",
    "Num_Products",
    "Product_Usage_History",
    "Total_Bank_Profit",
    "Avg_Monthly_Inflow",
    "Avg_Monthly_Outflow",
    "Monthly_Transaction_Count",
    "Avg_Account_Balance",
    "New_Product_Offer",
    "Credit_Limit_Change",
    "Targeted_Communication",
    "Revenue_Growth_Rate",
]

CATEGORICAL_CANDIDATES = [
    "Industry",
    "Region",
    "Business_Size",
    "New_Product_Offer_Type",
    "Revenue_Trend",
]

MULTI_VALUE_COLS = ["Product_Types"]
ID_COLS = ["Client_ID"]

_LAYER_ITEMS = list(LAYER_ORDER.items())
_BASE_LAYER_INDEX: Dict[str, int] = {
    var: idx
    for idx, (_, vars_in_layer) in enumerate(_LAYER_ITEMS)
    for var in vars_in_layer
}
_BASE_KEYS_BY_LENGTH = sorted(_BASE_LAYER_INDEX.keys(), key=len, reverse=True)

Edge = Tuple[str, str]

EDGE_ATTR_SPECS = {
    "freq_GES": (float, 0.0),
    "freq_HC": (float, 0.0),
    "freq_MMHC": (float, 0.0),
    "support_ge_tau": (int, 0),
    "mean_freq": (float, 0.0),
    "std_freq": (float, 0.0),
    "dir_agree": (float, 0.0),
    "robustness_score": (float, 0.0),
    "robustness_label": (str, ""),
    "I2Y": (int, 0),
}

CONSENSUS_ALGOS = ["GES", "HC", "MMHC"]
GT_ALGOS = CONSENSUS_ALGOS + ["CONS"]
CONSENSUS_FILTER_LABELS = {"HIGH", "MEDIUM"}


@dataclass
class BootstrapConfig:
    ges_runs: int = 4
    hc_runs: int = 4
    mmhc_runs: int = 4
    sample_frac: float = 0.8
    seed: int = 1
    scoring_method: str = "bic-d"
    hc_max_indegree: int = 4
    mmhc_max_indegree: int = 3
    pc_alpha: float = 0.05
    pc_max_cond_vars: int = 2


@dataclass
class PipelineConfig:
    csv_path: Path
    output_dir: Path = OUTPUT_DIR
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)


@dataclass
class DatasetBundle:
    df_raw: pd.DataFrame
    df_proc: pd.DataFrame
    dtypes_map: Dict[str, str]
    disc_df: pd.DataFrame
    cols_for_dag: List[str]
    present_interventions: List[str]
    present_outcomes: List[str]


@dataclass
class KnowledgeBundle:
    dir_blacklist: Set[Edge]
    temporal_tiers: List[List[str]]
    cols_mmhc: List[str]


@dataclass
class StructureResult:
    edges: List[Edge]
    dag: nx.DiGraph


# ====== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======


def _canonical_base(col: str) -> Optional[str]:
    if col in _BASE_LAYER_INDEX:
        return col
    for base in _BASE_KEYS_BY_LENGTH:
        if col.startswith(f"{base}__") or col.startswith(f"{base}_"):
            return base
    return None


def _layer_index(col: str) -> Optional[int]:
    base = _canonical_base(col)
    if base is None:
        return None
    return _BASE_LAYER_INDEX.get(base)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def split_multi_values(series: pd.Series) -> Set[str]:
    vals: Set[str] = set()
    for x in series.dropna().astype(str):
        for t in (s.strip() for s in x.split(",")):
            if t:
                vals.add(t)
    return vals


def multi_hot_encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    values = sorted(split_multi_values(df[col]))
    for v in values:
        new_col = f"{col}__{v}"
        df[new_col] = (
            df[col]
            .fillna("")
            .astype(str)
            .apply(lambda s: int(v in [x.strip() for x in s.split(",") if x.strip()]))
        )
    return df.drop(columns=[col])


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()

    for col in [c for c in MULTI_VALUE_COLS if c in df.columns]:
        df = multi_hot_encode(df, col)

    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    categoricals = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    if categoricals:
        df = pd.get_dummies(df, columns=categoricals, drop_first=False, dtype=int)

    for c in ID_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    dtypes_map: Dict[str, str] = {}
    for col in df.columns:
        dtypes_map[col] = (
            "numerical" if df[col].dtype.kind in "biufc" else "categorical"
        )

    return df, dtypes_map


def ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def save_graph_png(graph: nx.DiGraph, path_png: Path) -> None:
    try:
        from networkx.drawing.nx_pydot import to_pydot

        pdot = to_pydot(graph)
        pdot.write_png(path_png)
    except Exception as exc:
        print(
            f"[warn] Не удалось сохранить PNG ({exc}). Установите pydot/graphviz или сохраните граф в DOT вручную."
        )


def graph_from_edges(nodes: Iterable[str], edges: Iterable[Edge]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for u, v in edges:
        if u in graph and v in graph and u != v:
            graph.add_edge(u, v)
    return graph


def make_discrete_frame(
    df_proc: pd.DataFrame,
    dtypes_map: Dict[str, str],
    cols: Optional[Sequence[str]] = None,
    n_bins: int = 4,
) -> pd.DataFrame:
    cols = list(cols or df_proc.columns)
    disc = df_proc.copy()
    for col in cols:
        if dtypes_map.get(col) != "numerical":
            continue
        vals = disc[col].values
        uniq = np.unique(vals[~pd.isna(vals)])
        if len(uniq) <= max(5, n_bins) or np.all(np.isin(uniq, [0, 1])):
            continue
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        disc[col] = kb.fit_transform(disc[[col]]).astype(int)
    return disc


def build_direction_blacklist(all_cols: Sequence[str]) -> Set[Edge]:
    forbidden: Set[Edge] = set()
    for src in all_cols:
        src_layer = _layer_index(src)
        if src_layer is None:
            continue
        for dst in all_cols:
            if src == dst:
                continue
            dst_layer = _layer_index(dst)
            if dst_layer is None:
                continue
            if src_layer > dst_layer:
                forbidden.add((src, dst))
    return forbidden


def build_temporal_tiers(cols: Sequence[str]) -> List[List[str]]:
    tiers: List[List[str]] = []
    assigned: Set[str] = set()
    for _, base_vars in _LAYER_ITEMS:
        base_set = set(base_vars)
        tier = [col for col in cols if _canonical_base(col) in base_set]
        if tier:
            tiers.append(tier)
            assigned.update(tier)
    leftovers = [c for c in cols if c not in assigned]
    if leftovers:
        tiers.append(leftovers)
    flat = [x for tier in tiers for x in tier]
    assert len(flat) == len(set(flat)), "Дубли переменных в ярусах"
    return tiers


def make_cols_for_skeleton(cols: Sequence[str]) -> List[str]:
    drop_prefixes = (
        "Industry_",
        "Region_",
        "Business_Size_",
        "New_Product_Offer_Type_",
        "Revenue_Trend_",
    )
    keep: List[str] = []
    for col in cols:
        if col.startswith(drop_prefixes):
            continue
        keep.append(col)
    for col in INTERVENTIONS + ["Revenue_Growth_Rate"]:
        if col in cols and col not in keep:
            keep.append(col)
    return keep


def chi2_indep(
    X: str,
    Y: str,
    Z: Optional[Sequence[str]],
    *,
    data: Optional[pd.DataFrame] = None,
    significance_level: float = 0.05,
    **kwargs,
) -> bool:
    if Z is None:
        Z_list: List[str] = []
    elif isinstance(Z, (list, tuple)):
        Z_list = list(Z)
    else:
        Z_list = list(Z)

    out = chi_square(
        X,
        Y,
        Z=Z_list,
        data=data,
        boolean=True,
        significance_level=significance_level,
    )
    if isinstance(out, (bool, np.bool_)):
        return bool(out)
    chi_val, p_val, *_ = chi_square(X, Y, Z=Z_list, data=data, boolean=False)
    return p_val > significance_level


def _check_temporal_tiers_ok(tiers: Sequence[Sequence[str]]) -> None:
    flat = [x for tier in tiers for x in tier]
    assert len(flat) == len(set(flat)), "Дубли переменных в ярусах"
    assert all(isinstance(x, str) and x for x in flat)


def _to_freq_df(counter: Counter, n_runs: int) -> pd.DataFrame:
    df = pd.DataFrame(
        [(u, v, counter[(u, v)] / n_runs) for (u, v) in counter.keys()],
        columns=["u", "v", "freq"],
    )
    return df.sort_values("freq", ascending=False)


def _save_freq_df(df: pd.DataFrame, path: Path) -> None:
    df.sort_values("freq", ascending=False).to_csv(path, index=False)
    print(f"[BOOT] saved: {path}")


def _filter_tiers(tiers: Sequence[Sequence[str]], keep: Set[str]) -> List[List[str]]:
    filtered = [[node for node in tier if node in keep] for tier in tiers]
    return [tier for tier in filtered if tier]


def _forbidden_in_subset(forbidden_edges: Set[Edge], keep: Set[str]) -> Set[Edge]:
    return {(u, v) for (u, v) in forbidden_edges if u in keep and v in keep}


def _bootstrap_edges(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    n_runs: int,
    frac: float,
    seed: int,
    desc: str,
    learn_edges: Callable[[pd.DataFrame], Iterable[Edge]],
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    counter: Counter[Edge] = Counter()
    for _ in trange(n_runs, desc=desc):
        sample = df[cols].sample(
            frac=frac,
            replace=True,
            random_state=rng.randint(1, 10**9),
        )
        counter.update(list(learn_edges(sample)))
    return _to_freq_df(counter, n_runs)


def bootstrap_ges(
    df_disc: pd.DataFrame,
    cols: Sequence[str],
    dir_blacklist: Set[Edge],
    temporal_tiers: Sequence[Sequence[str]],
    *,
    n_runs: int,
    frac: float,
    seed: int,
    scoring_method: str,
) -> pd.DataFrame:
    keep = set(cols)
    forbidden = _forbidden_in_subset(dir_blacklist, keep)
    tiers = _filter_tiers(temporal_tiers, keep)

    def learn(sample: pd.DataFrame) -> Iterable[Edge]:
        ek = ExpertKnowledge(forbidden_edges=forbidden, temporal_order=tiers)
        dag = GES(sample).estimate(
            scoring_method=scoring_method,
            expert_knowledge=ek,
        )
        return dag.edges()

    return _bootstrap_edges(
        df_disc,
        cols,
        n_runs=n_runs,
        frac=frac,
        seed=seed,
        desc="GES bootstrap",
        learn_edges=learn,
    )


def bootstrap_hc(
    df_disc: pd.DataFrame,
    cols: Sequence[str],
    dir_blacklist: Set[Edge],
    temporal_tiers: Sequence[Sequence[str]],
    *,
    n_runs: int,
    frac: float,
    seed: int,
    scoring_method: str,
    max_indegree: int,
) -> pd.DataFrame:
    keep = set(cols)
    forbidden = _forbidden_in_subset(dir_blacklist, keep)
    tiers = _filter_tiers(temporal_tiers, keep)

    def learn(sample: pd.DataFrame) -> Iterable[Edge]:
        ek = ExpertKnowledge(forbidden_edges=forbidden, temporal_order=tiers)
        dag = HillClimbSearch(sample).estimate(
            scoring_method=scoring_method,
            max_indegree=max_indegree,
            expert_knowledge=ek,
            show_progress=False,
        )
        return dag.edges()

    return _bootstrap_edges(
        df_disc,
        cols,
        n_runs=n_runs,
        frac=frac,
        seed=seed,
        desc="HC bootstrap",
        learn_edges=learn,
    )


def bootstrap_fast_mmhc(
    df_disc: pd.DataFrame,
    cols_light: Sequence[str],
    dir_blacklist: Set[Edge],
    temporal_tiers: Sequence[Sequence[str]],
    *,
    n_runs: int,
    frac: float,
    seed: int,
    pc_alpha: float,
    pc_max_cond_vars: int,
    scoring_method: str,
    max_indegree: int,
) -> pd.DataFrame:
    keep = set(cols_light)
    tiers = _filter_tiers(temporal_tiers, keep)
    forbidden_domain = _forbidden_in_subset(dir_blacklist, keep)

    def learn(sample: pd.DataFrame) -> Iterable[Edge]:
        pc_est = PC(sample)
        kwargs = dict(
            variant="stable",
            ci_test=chi2_indep,
            significance_level=pc_alpha,
            max_cond_vars=pc_max_cond_vars,
            n_jobs=-1,
            show_progress=False,
        )
        sig = inspect.signature(pc_est.build_skeleton)
        pc_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        skeleton_graph, _sep_sets = pc_est.build_skeleton(**pc_kwargs)
        skeleton_edges = list(skeleton_graph.edges())

        all_pairs = set(itertools.permutations(cols_light, 2))
        skeleton_directed = set(
            sum(([(u, v), (v, u)] for (u, v) in skeleton_edges), [])
        )
        forbidden_outside = all_pairs - skeleton_directed
        forbidden_all = forbidden_domain | forbidden_outside

        ek = ExpertKnowledge(forbidden_edges=forbidden_all, temporal_order=tiers)
        dag = HillClimbSearch(sample).estimate(
            scoring_method=scoring_method,
            max_indegree=max_indegree,
            expert_knowledge=ek,
            show_progress=False,
        )
        return dag.edges()

    return _bootstrap_edges(
        df_disc,
        cols_light,
        n_runs=n_runs,
        frac=frac,
        seed=seed,
        desc="FAST-MMHC bootstrap",
        learn_edges=learn,
    )


def _merge_freqs(named_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for name, df in named_dfs.items():
        tmp = df.rename(columns={"freq": f"freq_{name}"})
        if merged is None:
            merged = tmp
        else:
            merged = pd.merge(merged, tmp, on=["u", "v"], how="outer")
    if merged is None:
        return pd.DataFrame(columns=["u", "v"])
    freq_cols = [c for c in merged.columns if c.startswith("freq_")]
    for col in freq_cols:
        merged[col] = merged[col].fillna(0.0)
    return merged


def _direction_agreement(df_freqs: pd.DataFrame, algos: Sequence[str]) -> pd.DataFrame:
    df = df_freqs.copy()
    freq_cols = [f"freq_{algo}" for algo in algos]
    df["sum_freq"] = df[freq_cols].sum(axis=1)
    rev = df[["u", "v", "sum_freq"]].rename(
        columns={"u": "v", "v": "u", "sum_freq": "sum_freq_rev"}
    )
    out = df.merge(rev, on=["u", "v"], how="left")
    out["sum_freq_rev"] = out["sum_freq_rev"].fillna(0.0)
    out["dir_total"] = out["sum_freq"] + out["sum_freq_rev"]
    out["dir_agree"] = np.where(
        out["dir_total"] > 0,
        np.maximum(out["sum_freq"], out["sum_freq_rev"]) / out["dir_total"],
        np.nan,
    )
    return out


def build_consensus(
    ges_boot: pd.DataFrame,
    hc_boot: pd.DataFrame,
    mmhc_boot: pd.DataFrame,
    *,
    tau: float = 0.5,
) -> pd.DataFrame:
    boot_map = {"GES": ges_boot, "HC": hc_boot, "MMHC": mmhc_boot}
    union = _merge_freqs(boot_map)
    freq_cols = [f"freq_{algo}" for algo in CONSENSUS_ALGOS]

    union["support_ge_tau"] = (union[freq_cols] >= tau).sum(axis=1)
    union["mean_freq"] = union[freq_cols].mean(axis=1)
    union["std_freq"] = union[freq_cols].std(axis=1, ddof=0).fillna(0.0)

    dir_df = _direction_agreement(union, CONSENSUS_ALGOS)
    dir_df["robustness_score"] = dir_df["mean_freq"] * dir_df["dir_agree"].fillna(0.0)

    def label(row: pd.Series) -> str:
        score = row["robustness_score"]
        support = row["support_ge_tau"]
        if support >= 3 and score >= 0.6:
            return "HIGH"
        if support >= 2 and score >= 0.5:
            return "MEDIUM"
        if score >= 0.3:
            return "LOW"
        return "WEAK"

    dir_df["robustness_label"] = dir_df.apply(label, axis=1)
    dir_df["I2Y"] = dir_df.apply(
        lambda row: int(
            _canonical_base(row["u"]) in INTERVENTIONS
            and _canonical_base(row["v"]) in OUTCOMES
        ),
        axis=1,
    )
    return dir_df


def consensus_to_graph(df: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    for _, row in df.iterrows():
        u, v = row["u"], row["v"]
        graph.add_node(u)
        graph.add_node(v)
        attrs = {}
        for col, (caster, default) in EDGE_ATTR_SPECS.items():
            val = row.get(col, default)
            attrs[col] = default if pd.isna(val) else caster(val)
        graph.add_edge(u, v, **attrs)
    return graph


def prepare_dataset(config: PipelineConfig) -> DatasetBundle:
    df_raw = load_data(config.csv_path)
    df_proc, dtypes_map = preprocess(df_raw)
    print("\nРазмер после предобработки:", df_proc.shape)
    print("Типы признаков (после кодировок):", dict(Counter(dtypes_map.values())))

    all_cols = list(df_proc.columns)
    present_interventions = ensure_columns(df_proc, INTERVENTIONS)
    present_outcomes = ensure_columns(df_proc, OUTCOMES)
    print("Интервенции в данных:", present_interventions)
    print("Целевые метрики в данных:", present_outcomes)

    clean_csv = config.output_dir / "cleaned_dataset.csv"
    df_proc.to_csv(clean_csv, index=False)
    print(f"Сохранён очищенный датасет: {clean_csv}")

    disc_df = make_discrete_frame(df_proc, dtypes_map, all_cols, n_bins=4)

    return DatasetBundle(
        df_raw=df_raw,
        df_proc=df_proc,
        dtypes_map=dtypes_map,
        disc_df=disc_df,
        cols_for_dag=all_cols,
        present_interventions=present_interventions,
        present_outcomes=present_outcomes,
    )


def build_knowledge(bundle: DatasetBundle) -> KnowledgeBundle:
    dir_blacklist = build_direction_blacklist(bundle.cols_for_dag)
    temporal_tiers = build_temporal_tiers(bundle.cols_for_dag)
    _check_temporal_tiers_ok(temporal_tiers)
    cols_mmhc = make_cols_for_skeleton(bundle.cols_for_dag)
    print(
        f"[FAST-MMHC] skeleton columns: {len(cols_mmhc)} / {len(bundle.cols_for_dag)}"
    )
    return KnowledgeBundle(
        dir_blacklist=dir_blacklist,
        temporal_tiers=temporal_tiers,
        cols_mmhc=cols_mmhc,
    )


def _write_edges(edges: Sequence[Edge], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(edges), handle, ensure_ascii=False, indent=2)
    print(f"Сохранены рёбра: {path}")


def run_structure_learning(
    bundle: DatasetBundle,
    knowledge: KnowledgeBundle,
    config: PipelineConfig,
) -> Dict[str, StructureResult]:
    results: Dict[str, StructureResult] = {}

    ek = ExpertKnowledge(
        forbidden_edges=set(knowledge.dir_blacklist),
        required_edges=set(),
        temporal_order=knowledge.temporal_tiers,
    )

    ges = GES(bundle.disc_df[bundle.cols_for_dag])
    dag_ges = ges.estimate(
        scoring_method=config.bootstrap.scoring_method, expert_knowledge=ek
    )
    ges_edges = list(dag_ges.edges())
    print("[GES] кол-во рёбер:", len(ges_edges))
    print("[GES] пример рёбер:", ges_edges[:12])
    results["GES"] = StructureResult(edges=ges_edges, dag=dag_ges)
    _write_edges(ges_edges, config.output_dir / "ges_edges.json")
    save_graph_png(
        graph_from_edges(dag_ges.nodes(), ges_edges),
        config.output_dir / "ges_graph.png",
    )

    ek_hc = ExpertKnowledge(
        forbidden_edges=set(knowledge.dir_blacklist),
        temporal_order=knowledge.temporal_tiers,
    )
    hc = HillClimbSearch(bundle.disc_df[bundle.cols_for_dag])
    dag_hc = hc.estimate(
        scoring_method=config.bootstrap.scoring_method,
        max_indegree=config.bootstrap.hc_max_indegree,
        expert_knowledge=ek_hc,
        show_progress=True,
    )
    hc_edges = list(dag_hc.edges())
    print("[HC] edges:", len(hc_edges))
    results["HC"] = StructureResult(edges=hc_edges, dag=dag_hc)
    _write_edges(hc_edges, config.output_dir / "hc_edges.json")
    save_graph_png(
        graph_from_edges(dag_hc.nodes(), hc_edges),
        config.output_dir / "hc_graph.png",
    )

    pc = PC(bundle.disc_df[knowledge.cols_mmhc])
    kwargs = dict(
        variant="stable",
        ci_test=chi2_indep,
        significance_level=config.bootstrap.pc_alpha,
        max_cond_vars=config.bootstrap.pc_max_cond_vars,
        n_jobs=-1,
        show_progress=False,
    )
    sig = inspect.signature(pc.build_skeleton)
    pc_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    t0 = time.time()
    skeleton_graph, _ = pc.build_skeleton(**pc_kwargs)
    print(
        f"[FAST-MMHC] PC skeleton in {time.time() - t0:.2f}s: "
        f"{skeleton_graph.number_of_nodes()} nodes, "
        f"{skeleton_graph.number_of_edges()} undirected edges"
    )

    skeleton_edges = list(skeleton_graph.edges())
    all_pairs = set(itertools.permutations(knowledge.cols_mmhc, 2))
    skeleton_directed = set(sum(([(u, v), (v, u)] for (u, v) in skeleton_edges), []))
    forbidden_outside = all_pairs - skeleton_directed
    forbidden_edges = set(knowledge.dir_blacklist) | forbidden_outside

    ek_mmhc = ExpertKnowledge(
        forbidden_edges=forbidden_edges,
        temporal_order=knowledge.temporal_tiers,
    )

    hc_mmhc = HillClimbSearch(bundle.disc_df[knowledge.cols_mmhc])
    dag_mmhc = hc_mmhc.estimate(
        scoring_method=config.bootstrap.scoring_method,
        max_indegree=config.bootstrap.mmhc_max_indegree,
        expert_knowledge=ek_mmhc,
        show_progress=False,
    )
    mmhc_edges = list(dag_mmhc.edges())
    print("[FAST-MMHC] edges:", len(mmhc_edges))
    results["MMHC"] = StructureResult(edges=mmhc_edges, dag=dag_mmhc)
    _write_edges(mmhc_edges, config.output_dir / "mmhc_fast_edges.json")
    save_graph_png(
        graph_from_edges(dag_mmhc.nodes(), mmhc_edges),
        config.output_dir / "mmhc_fast_graph.png",
    )

    for name, result in results.items():
        bad = [(u, v) for (u, v) in result.edges if (u, v) in knowledge.dir_blacklist]
        if bad:
            raise ValueError(f"Запрещённые рёбра найдены для {name}: {bad}")

    return results


def run_bootstrap(
    bundle: DatasetBundle,
    knowledge: KnowledgeBundle,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    boot_cfg = config.bootstrap
    ges_boot = bootstrap_ges(
        bundle.disc_df,
        bundle.cols_for_dag,
        set(knowledge.dir_blacklist),
        knowledge.temporal_tiers,
        n_runs=boot_cfg.ges_runs,
        frac=boot_cfg.sample_frac,
        seed=boot_cfg.seed,
        scoring_method=boot_cfg.scoring_method,
    )
    _save_freq_df(ges_boot, config.output_dir / "bootstrap_ges_edges.csv")

    hc_boot = bootstrap_hc(
        bundle.disc_df,
        bundle.cols_for_dag,
        set(knowledge.dir_blacklist),
        knowledge.temporal_tiers,
        n_runs=boot_cfg.hc_runs,
        frac=boot_cfg.sample_frac,
        seed=boot_cfg.seed,
        scoring_method=boot_cfg.scoring_method,
        max_indegree=boot_cfg.hc_max_indegree,
    )
    _save_freq_df(hc_boot, config.output_dir / "bootstrap_hc_edges.csv")

    mmhc_boot = bootstrap_fast_mmhc(
        bundle.disc_df,
        knowledge.cols_mmhc,
        set(knowledge.dir_blacklist),
        knowledge.temporal_tiers,
        n_runs=boot_cfg.mmhc_runs,
        frac=boot_cfg.sample_frac,
        seed=boot_cfg.seed,
        pc_alpha=boot_cfg.pc_alpha,
        pc_max_cond_vars=boot_cfg.pc_max_cond_vars,
        scoring_method=boot_cfg.scoring_method,
        max_indegree=boot_cfg.mmhc_max_indegree,
    )
    _save_freq_df(mmhc_boot, config.output_dir / "bootstrap_mmhc_fast_edges.csv")

    return ges_boot, hc_boot, mmhc_boot


def aggregate_one_hot_nodes(df_edges: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует one-hot вершины обратно к исходным категориальным переменным.
    К примеру:
    - Industry_Construction, Industry_IT -> Industry
    - Business_Size_medium, Business_Size_small -> Business_Size
    - Region_RU-MOW, Region_RU-SPE -> Region
    """

    def get_base_node_name(node_name: str) -> str:
        """Возвращает базовое имя для one-hot вершины"""
        for pattern in CATEGORICAL_CANDIDATES:
            if node_name.startswith(pattern):
                return pattern
        return node_name

    # Создаем копию DataFrame
    df_agg = df_edges.copy()

    # Агрегируем имена вершин
    df_agg["u_base"] = df_agg["u"].apply(get_base_node_name)
    df_agg["v_base"] = df_agg["v"].apply(get_base_node_name)

    # Удаляем петли (ребра из вершины в саму себя после агрегации)
    df_agg = df_agg[df_agg["u_base"] != df_agg["v_base"]]

    # Группируем по агрегированным вершинам и объединяем метрики
    grouped = (
        df_agg.groupby(["u_base", "v_base"])
        .agg(
            {
                "freq_GES": "max",  # берем максимальную частоту
                "freq_HC": "max",
                "freq_MMHC": "max",
                "support_ge_tau": "max",  # максимальная поддержка
                "mean_freq": "max",  # максимальное среднее
                "std_freq": "mean",  # среднее std (более стабильно)
                "dir_agree": "max",  # максимальное согласие по направлению
                "robustness_score": "max",  # максимальный robustness score
                "I2Y": "max",  # если хоть одно ребро I2Y, то 1
            }
        )
        .reset_index()
    )

    # Переименовываем столбцы обратно
    grouped = grouped.rename(columns={"u_base": "u", "v_base": "v"})

    # Пересчитываем производные метрики
    freq_cols = ["freq_GES", "freq_HC", "freq_MMHC"]
    grouped["support_ge_tau"] = (grouped[freq_cols] >= 0.5).sum(axis=1)
    grouped["mean_freq"] = grouped[freq_cols].mean(axis=1)
    grouped["std_freq"] = grouped[freq_cols].std(axis=1, ddof=0).fillna(0.0)

    # Пересчитываем robustness_label на основе новых метрик
    def get_robustness_label(row: pd.Series) -> str:
        score = row["robustness_score"]
        support = row["support_ge_tau"]
        if support >= 3 and score >= 0.6:
            return "HIGH"
        if support >= 2 and score >= 0.5:
            return "MEDIUM"
        if score >= 0.3:
            return "LOW"
        return "WEAK"

    grouped["robustness_label"] = grouped.apply(get_robustness_label, axis=1)

    return grouped


def build_and_export_consensus(
    bundle: DatasetBundle,
    knowledge: KnowledgeBundle,
    config: PipelineConfig,
    ges_boot: pd.DataFrame,
    hc_boot: pd.DataFrame,
    mmhc_boot: pd.DataFrame,
    aggregate_one_hot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, nx.DiGraph]:
    consensus = build_consensus(ges_boot, hc_boot, mmhc_boot, tau=0.5)

    if aggregate_one_hot:
        print("Агрегируем one-hot вершины обратно к исходным именам...")
        consensus = aggregate_one_hot_nodes(consensus)

    consensus_csv = config.output_dir / "consensus_edges.csv"
    consensus.to_csv(consensus_csv, index=False)
    print(f"Consensus saved: {consensus_csv}")

    df_for_graph = consensus.apply(adjust_robustness, axis=1)
    graph = consensus_to_graph(df_for_graph)

    out_prefix = config.output_dir / "graph_consensus"
    export_graph(graph, out_prefix=str(out_prefix))
    print(f"Saved: {out_prefix}.json | {out_prefix}.gexf | {out_prefix}.graphml")

    html = build_pyvis_html(graph, height_px=650, directed=True, physics=True)
    (out_prefix.with_suffix(".html")).write_text(html, encoding="utf-8")
    print(f"Saved: {out_prefix}.html")

    return consensus, df_for_graph, graph


def build_algo_graph(
    csv_path: Path,
    output_dir: Optional[Path] = OUTPUT_DIR,
    return_graph: bool = False,
) -> Tuple[List[Dict], Optional[nx.DiGraph]]:
    """
    Build causal graph using algorithmic approaches.

    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save results (None for default)
        return_graph: Whether to return graph object

    Returns:
        Tuple of (consensus_edges_df, graph_object)
    """
    config = PipelineConfig(csv_path=csv_path, output_dir=output_dir)

    OUTPUT_DIR.mkdir(exist_ok=True)

    bundle = prepare_dataset(config)
    knowledge = build_knowledge(bundle)
    structure_results = run_structure_learning(bundle, knowledge, config)
    ges_boot, hc_boot, mmhc_boot = run_bootstrap(bundle, knowledge, config)
    consensus_full, consensus_subset, consensus_graph = build_and_export_consensus(
        bundle, knowledge, config, ges_boot, hc_boot, mmhc_boot
    )

    consensus_full.to_csv(config.output_dir / "algo_consensus_edges.csv", index=False)

    # convert pd.DataFrame to List[Dict] as other functions expected
    # similar output to biuld_llm_graph
    consensus_list = consensus_full.to_dict("records")

    if return_graph:
        return consensus_list, consensus_graph
    return consensus_list, None


def main():
    """Standalone execution for algorithmic graph building"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    cfg = get_config()
    artifacts_dir = cfg.full_artifacts_dir
    csv_path = artifacts_dir / "synthetic_clients.csv"

    build_algo_graph(
        csv_path=csv_path,
        output_dir=OUTPUT_DIR,
    )
    print("Algorithmic graph construction completed!")


if __name__ == "__main__":
    main()
