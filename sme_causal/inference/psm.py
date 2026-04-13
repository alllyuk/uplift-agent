import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

@dataclass
class PSMResult:
    ate: float
    ate_naive: float
    n_pairs: int
    matched_df: pd.DataFrame

class CausalInferenceAnalyzer:
    """
    PSM: загрузка → бинаризация treatment → пропенсити → матчинг → ATE.
    """

    def __init__(
        self,
        target: str,
        treatment_variable: str,
        covariates: Optional[List[str]] = None,
        threshold: Optional[float] = 0.01,   # если treatment не бинарный
        replacement: bool = False,           # без replacement по умолчанию (честный 1:1)
        caliper: Optional[float] = 0.1,     # максимальная разница propensity
        logreg_max_iter: int = 1000,
    ):
        self.covariates = covariates
        self.target = target
        self.treatment_variable = treatment_variable
        self.threshold = threshold
        self.replacement = replacement
        self.caliper = caliper
        self.logreg_max_iter = logreg_max_iter

        self.data: pd.DataFrame | None = None
        self.matched_data: pd.DataFrame | None = None

        # служебные поля
        self._treated = "__treated__"
        self._score = "__propensity__"
        self._pair_id = "__pair_id__"

    def autodetect_covariates(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        treatment: Optional[str] = None,
    ) -> list[str]:
        """
        Автоматически выбирает ковариаты: все столбцы df, кроме target/treatment и служебных.
        target/treatment можно не передавать — тогда берём из self.
        """
        y = target or self.target
        t = treatment or self.treatment_variable

        ignore = {
            y, t,
            "Client_ID",                 
            self._treated,               # служебные
            self._score,
            self._pair_id,
        }
        # Сохраняем порядок колонок как в df
        return [c for c in df.columns if c not in ignore]

    # ---------- загрузка ----------
    def load_data(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Файл не найден: {path}")
        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        self.preprocess_data(df)

    def preprocess_data(self, df: pd.DataFrame) -> None:
        if not self.covariates:
            self.covariates = self.autodetect_covariates(df)
        # 1. Проверяем, что все нужные колонки есть в df
        need = [*self.covariates, self.target, self.treatment_variable]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют колонки: {missing}")

        # 2. Берём только нужные столбцы, остальное выкидываем
        self.data = df[need].copy()

        # 3. Приводим колонку target (outcome) к числовому типу
        #    (если там строки, даты или NaN — превратятся в NaN)
        self.data[self.target] = pd.to_numeric(self.data[self.target], errors="coerce")

        # 4. Убираем строки, где нет данных по outcome или treatment
        self.data.dropna(subset=[self.target, self.treatment_variable], inplace=True)

        # 5. Перенумеровываем индексы (удобно для последующей работы)
        self.data.reset_index(drop=True, inplace=True)

    # ---------- бинаризация treatment ----------
    def categorize_treatment(self) -> None:
        assert self.data is not None
        s = self.data[self.treatment_variable]
        # если уже бинарный
        if s.dropna().isin([0, 1]).all():
            self.data[self._treated] = s.astype(int)
        else:
            if self.threshold is None:
                raise ValueError("treatment не бинарный: укажи threshold")
            self.data[self._treated] = (pd.to_numeric(s, errors="coerce") >= float(self.threshold)).astype(int)

    # ---------- пропенсити ----------
    def estimate_propensity_scores(self) -> None:
        assert self.data is not None
        if self._treated not in self.data.columns:
            raise RuntimeError("сначала вызови categorize_treatment()")

        # One-hot для категориальных ковариат
        X = pd.get_dummies(self.data[self.covariates], drop_first=True)
        if X.shape[1] == 0:
            raise ValueError("После one-hot нет признаков")
        

        # Масштабирование не обязательно (стабильно для разных скейлов)
        scaler = StandardScaler(with_mean=False)  # sparse-friendly
        X_scaled = scaler.fit_transform(X)

        y = self.data[self._treated].astype(int).values
        lr = LogisticRegression(max_iter=self.logreg_max_iter, solver="lbfgs")
        lr.fit(X_scaled, y)
        self.data[self._score] = lr.predict_proba(X_scaled)[:, 1]

    # ---------- матчинг ----------
    def find_matched_pairs(self) -> pd.DataFrame:
        assert self.data is not None
        if self._score not in self.data.columns:
            raise RuntimeError("сначала вызови estimate_propensity_scores()")

        df = self.data
        treated = df[df[self._treated] == 1].copy()
        control = df[df[self._treated] == 0].copy()
        if treated.empty or control.empty:
            raise ValueError("Недостаточно treated/control наблюдений")

        # матрица расстояний по |p_t - p_c|
        D = pairwise_distances(
            treated[[self._score]].values,
            control[[self._score]].values,
            metric="euclidean",
        )
        dist = pd.DataFrame(D, index=treated.index, columns=control.index)

        pairs: list[tuple[int, int]] = []

        if self.replacement:
            # с replacement: для каждого treated берём ближайшего control
            min_idx = dist.idxmin(axis=1)
            for it, ic in min_idx.items():
                if self.caliper is not None:
                    if abs(df.loc[it, self._score] - df.loc[ic, self._score]) > float(self.caliper):
                        continue
                pairs.append((it, ic))
        else:
            # без replacement: жадный выбор, «вычёркиваем» control
            dist_copy = dist.copy()
            used_c: set[int] = set()
            for it in dist_copy.index:
                row = dist_copy.loc[it].copy()
                # вычеркнуть уже использованных контролей
                if used_c:
                    row.loc[list(used_c)] = np.inf
                ic = row.idxmin()
                if not np.isfinite(row.loc[ic]):
                    continue
                if self.caliper is not None:
                    if abs(df.loc[it, self._score] - df.loc[ic, self._score]) > float(self.caliper):
                        continue
                pairs.append((it, ic))
                used_c.add(int(ic))

        if not pairs:
            self.matched_data = df.iloc[0:0].copy()
            return self.matched_data

        rows = []
        for k, (it, ic) in enumerate(pairs, start=1):
            trow = df.loc[it].to_dict()
            crow = df.loc[ic].to_dict()
            trow[self._pair_id] = k
            crow[self._pair_id] = k
            rows.append(trow)
            rows.append(crow)

        matched = pd.DataFrame(rows).reset_index(drop=True)
        keep = [self.target, self.treatment_variable, self._treated, self._score, self._pair_id] + \
               [c for c in self.covariates if c in matched.columns]
        self.matched_data = matched[keep]
        return self.matched_data

    def _compute_naive_ate(self) -> float:
        """ATE на исходных данных без матчинга (treated vs control)."""
        if self.data is None or self.data.empty:
            return float("nan")
        if self._treated not in self.data.columns:
            raise RuntimeError("сначала вызови categorize_treatment().")

        treated = self.data[self.data[self._treated] == 1][self.target].astype(float)
        control = self.data[self.data[self._treated] == 0][self.target].astype(float)

        if treated.empty or control.empty:
            return float("nan")

        return float(treated.mean() - control.mean())

    # ---------- ATE: на матчированных данных (разность средних) ----------
    def calculate_ate_matched(self) -> PSMResult:
        naive_ate = self._compute_naive_ate()

        if self.matched_data is None or self.matched_data.empty:
            return PSMResult(
                ate=float("nan"),
                ate_naive=naive_ate,
                n_pairs=0,
                matched_df=pd.DataFrame(),
            )

        # посчитаем число "валидных" пар (в которых есть и treated, и control, и не NaN по target)
        if self._pair_id in self.matched_data.columns:
            valid_pairs = (
                self.matched_data
                .dropna(subset=[self.target])
                .groupby(self._pair_id)[self._treated]
                .nunique()
            )
            n_pairs = int((valid_pairs == 2).sum())
        else:
            n_pairs = 0

        treated = self.matched_data[self.matched_data[self._treated] == 1][self.target].astype(float)
        control = self.matched_data[self.matched_data[self._treated] == 0][self.target].astype(float)

        if treated.empty or control.empty:
            return PSMResult(
                ate=float("nan"),
                ate_naive=naive_ate,
                n_pairs=n_pairs,
                matched_df=self.matched_data.copy(),
            )

        ate = float(treated.mean() - control.mean())
        return PSMResult(
            ate=ate,
            ate_naive=naive_ate,
            n_pairs=n_pairs,
            matched_df=self.matched_data.copy(),
        )

    # ---------- ATE: на исходных данных (наивный сплит T=1 vs T=0) ----------
    def calculate_ate_original(self) -> PSMResult:
        if self.data is None or self.data.empty:
            raise RuntimeError("Нет данных: сначала вызови run() или preprocess_data() и categorize_treatment().")
        if self._treated not in self.data.columns:
            raise RuntimeError("Сначала вызови categorize_treatment().")

        naive_ate = self._compute_naive_ate()
        return PSMResult(
            ate=naive_ate,
            ate_naive=naive_ate,
            n_pairs=0,
            matched_df=pd.DataFrame(),
        )
    
    # ---------- Гистограмма propensity score ДО матчинга ----------
    def plot_hist_original(self, bins: int | str = "auto") -> None:
        if self.data is None or self.data.empty:
            raise RuntimeError("Нет данных: сначала вызови run() или preprocess_data() + estimate_propensity_scores().")
        if self._score not in self.data.columns:
            raise RuntimeError("Сначала вычисли propensity score: estimate_propensity_scores().")
        if self._treated not in self.data.columns:
            raise RuntimeError("Сначала вызови categorize_treatment().")

        ps_t0 = self.data.loc[self.data[self._treated] == 0, self._score]
        ps_t1 = self.data.loc[self.data[self._treated] == 1, self._score]

        # объединяем данные для подсчета общих бинов
        all_scores = np.concatenate([ps_t0.to_numpy(), ps_t1.to_numpy()])
        bin_edges = np.histogram_bin_edges(all_scores, bins=bins)

        plt.hist(ps_t0, bins=bin_edges, density=True, alpha=0.5, label="Control (T=0)")
        plt.hist(ps_t1, bins=bin_edges, density=True, alpha=0.5, label="Treated (T=1)")
        plt.title("Propensity score (original data)")
        plt.xlabel("propensity")
        plt.ylabel("density")
        plt.legend()
        plt.show()

    # ---------- Гистограмма propensity score ПОСЛЕ матчинга ----------
    def plot_hist_matched(self, bins: int | str = "auto") -> None:
        if self.matched_data is None or self.matched_data.empty:
            raise RuntimeError("Нет matched данных: сначала вызови find_matched_pairs().")
        if self._score not in self.matched_data.columns or self._treated not in self.matched_data.columns:
            raise RuntimeError("В matched данных отсутствуют служебные колонки (__propensity__ / __treated__).")

        ps_t0 = self.matched_data.loc[self.matched_data[self._treated] == 0, self._score]
        ps_t1 = self.matched_data.loc[self.matched_data[self._treated] == 1, self._score]

        all_scores = np.concatenate([ps_t0.to_numpy(), ps_t1.to_numpy()])
        bin_edges = np.histogram_bin_edges(all_scores, bins=bins)

        plt.hist(ps_t0, bins=bin_edges, density=True, alpha=0.5, label="Control (T=0)")
        plt.hist(ps_t1, bins=bin_edges, density=True, alpha=0.5, label="Treated (T=1)")
        plt.title("Propensity score (matched data)")
        plt.xlabel("propensity")
        plt.ylabel("density")
        plt.legend()
        plt.show()
    
    # ---------- полный прогон ----------
    def run(self, df_or_path: str | pd.DataFrame) -> PSMResult:
        if isinstance(df_or_path, (str, os.PathLike)):
            self.load_data(str(df_or_path))
        else:
            self.preprocess_data(df_or_path)
        self.categorize_treatment()
        self.estimate_propensity_scores()
        self.find_matched_pairs()
        return self.calculate_ate_matched()
