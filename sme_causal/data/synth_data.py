"""
генерация синтетики + «истинный» граф
"""

# synth_data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

RNG = np.random.default_rng

INDUSTRIES = [
    "Retail",
    "Manufacturing",
    "IT_Services",
    "Construction",
    "Healthcare",
    "Hospitality",
]
REGIONS = ["RU-MOW", "RU-SPE", "RU-MOS", "RU-KDA", "RU-SVE", "RU-TA"]
SIZES = ["micro", "small", "medium"]

FIELD_DOCS_RU: Dict[str, str] = {
    # Macro
    "Industry": "Отрасль (ОКВЭД) бизнеса",
    "Region": "Регион регистрации/работы",
    "Business_Size": "Размер компании (micro/small/medium)",
    "Years_in_Operation": "Возраст бизнеса, лет",
    # Bank–Client
    "Client_Tenure": "Лет с банком",
    "Num_Products": "Кол-во продуктов банка",
    "Has_Deposit": "Есть депозитный счёт",
    "Has_Card": "Есть расчётная карта",
    "Has_Acquiring": "Есть эквайринг",
    "Has_Payroll": "Есть зарплатный проект",
    "Has_Loan": "Есть кредит",
    "Total_Bank_Profit": "Совокупная прибыль банка от клиента",
    # Transactional
    "Avg_Monthly_Inflow": "Средний месячный приток средств",
    "Avg_Monthly_Outflow": "Средний месячный отток средств",
    "Monthly_Transaction_Count": "Среднее число транзакций/мес",
    "Avg_Account_Balance": "Средний остаток на счёте",
    "Net_Cashflow": "Чистый денежный поток: Inflow – Outflow",
    "Activity_Bucket": "Категория активности по числу транзакций (низкая/средняя/высокая)",
    "Balance_Bucket": "Категория остатка (низкий/средний/высокий)",
    # Interventions
    "New_Product_Offer": "Флаг предложения нового продукта",
    "New_Product_Offer_Type": "Тип предложенного продукта",
    "Credit_Limit_Change": "Изменение кредитного лимита (проц.)",
    "Tariff_Discount": "Спецтариф/льготные условия (временные скидки/нулевые комиссии 1–3 мес)",    
    # Outcomes
    "Revenue_Growth_Rate": "Годовой рост выручки (притока), %",
    "Revenue_Trend": "Категория тренда (up/flat/down)",
}

# «Истинный» порядок слоёв для DAG-ограничения
LAYER_ORDER = {
    "Macro": ["Industry", "Region", "Business_Size", "Years_in_Operation"],
    "Relationship": [
        "Client_Tenure",
        "Num_Products",
        "Has_Deposit",
        "Has_Card",
        "Has_Acquiring",
        "Has_Payroll",
        "Has_Loan",
        "Total_Bank_Profit",
    ],
    "Transactional": [
        "Avg_Monthly_Inflow",
        "Avg_Monthly_Outflow",
        "Monthly_Transaction_Count",
        "Avg_Account_Balance",
    ],
    "Derived": [
        "Net_Cashflow",
        "Activity_Bucket",
        "Balance_Bucket",
    ], 
    "Interventions": [
        "New_Product_Offer",
        "New_Product_Offer_Type",
        "Credit_Limit_Change",
        "Tariff_Discount",
    ],
    "Outcomes": ["Revenue_Growth_Rate", "Revenue_Trend"],
}


def _sample_product_types(
    rng: np.random.Generator, industry: str, size: str
) -> List[str]:
    """Sample bank product types based on industry and business size.

    Args:
        rng: Random number generator for probabilistic sampling.
        industry: Business industry category.
        size: Business size category (micro/small/medium).

    Returns:
        List of applicable bank product types for the given industry and size.
    """
    base = ["deposit", "card"]
    if industry in {"Retail", "Hospitality"}:
        base.append("acquiring")
    if size in {"small", "medium"}:
        base.append("payroll")
    # кредит чаще у Construction/Manufacturing/SME
    if industry in {"Construction", "Manufacturing"} or size == "medium":
        base.append("loan")
    # случайная вариация
    base = list(sorted(set(base)))
    # вероятностное «отсечение» 0–1 типов
    if rng.random() < 0.3 and "payroll" in base:
        base.remove("payroll")
    return base


def _region_multiplier(region: str) -> float:
    """Get regional economic multiplier for business inflows.

    Args:
        region: Region code with the following meanings:
            - RU-MOW: Moscow city (Moscow Oblast)
            - RU-SPE: Saint Petersburg city
            - RU-MOS: Moscow Oblast (excluding Moscow city)
            - RU-KDA: Krasnodar region
            - RU-SVE: Sverdlovsk region
            - RU-TA: Tatarstan Republic

    Returns:
        Multiplier coefficient reflecting regional economic differences.
    """
    # Простая гео-гетерогенность
    return {
        "RU-MOW": 1.10,
        "RU-SPE": 1.05,
        "RU-MOS": 0.98,
        "RU-KDA": 1.02,
        "RU-SVE": 0.97,
        "RU-TA": 1.00,
    }[region]


def _industry_multiplier(industry: str) -> float:
    """Get industry-specific multiplier for business inflows.

    Args:
        industry: Business industry category.

    Returns:
        Multiplier coefficient reflecting industry-specific economic scale.
    """
    return {
        "Retail": 1.00,
        "Hospitality": 0.95,
        "Manufacturing": 1.10,
        "IT_Services": 1.20,
        "Construction": 1.05,
        "Healthcare": 1.15,
    }[industry]


def _size_multiplier(size: str) -> float:
    """Get business size multiplier for inflows.

    Args:
        size: Business size category (micro/small/medium).

    Returns:
        Multiplier coefficient reflecting business scale differences.
    """
    return {"micro": 0.35, "small": 0.75, "medium": 1.30}[size]


def _industry_txn_intensity(industry: str) -> Tuple[float, float]:
    """Get industry-specific transaction intensity coefficients.

    Args:
        industry: Business industry category.

    Returns:
        Tuple of (transaction_frequency_multiplier, outflow_ratio_multiplier).
    """
    # базовые коэффициенты для частоты транзакций и расходов
    if industry in {"Retail", "Hospitality"}:
        return (1.35, 0.82)
    if industry in {"IT_Services"}:
        return (0.85, 0.55)
    if industry in {"Manufacturing", "Construction"}:
        return (1.00, 0.92)
    return (1.05, 0.78)  # Healthcare


def _industry_trend(industry: str) -> float:
    """Get baseline industry-specific revenue growth rate.

    Args:
        industry: Business industry category.

    Returns:
        Baseline annual revenue growth rate without interventions.
    """
    # Средний базовый рост выручки (год к году) без интервенций
    return {
        "Retail": 0.02,
        "Hospitality": 0.015,
        "Manufacturing": 0.012,
        "IT_Services": 0.045,
        "Construction": 0.01,
        "Healthcare": 0.03,
    }[industry]


def _offer_type(rng: np.random.Generator, industry: str) -> str:
    """Sample product offer type weighted by industry preferences.

    Args:
        rng: Random number generator for probabilistic sampling.
        industry: Business industry category.

    Returns:
        Product offer type (loan/acquiring/card/payroll).
    """
    # если предложение есть, тип чаще соответствует отрасли
    pool = ["loan", "acquiring", "card", "payroll"]
    if industry in {"Retail", "Hospitality"}:
        weights = [0.15, 0.55, 0.2, 0.1]
    elif industry in {"Manufacturing", "Construction"}:
        weights = [0.55, 0.15, 0.15, 0.15]
    elif industry == "IT_Services":
        weights = [0.2, 0.15, 0.45, 0.2]
    else:  # Healthcare
        weights = [0.35, 0.2, 0.25, 0.2]
    return rng.choice(pool, p=np.array(weights))


# -----------------------------------------------------------------------------
# Single source of truth: modifiers and uplift sampling
# -----------------------------------------------------------------------------
#
# These helpers are shared between `generate_sme_data` (factual sampling) and
# `generate_with_counterfactuals` (potential-outcome sampling for non-treated).
# Any change here propagates to both paths automatically — this is what
# guarantees that the ground-truth ATT/ATE used for PSM evaluation match the
# per-client uplifts that were actually injected into Revenue_Growth_Rate.

def _compute_modifiers(
    *,
    avg_inflow: float,
    avg_outflow: float,
    avg_balance: float,
) -> Dict[str, float]:
    """Deterministic per-client modifiers used in uplift formulas."""
    inflow_safe = avg_inflow + 1e-9
    balance_to_inflow = avg_balance / inflow_safe
    return {
        "liquidity_pressure": 1.0 if balance_to_inflow < 0.45 else 0.0,
        "price_sensitivity": (
            0.6 * (avg_outflow / inflow_safe)
            + 0.4 * (1.0 - min(1.0, balance_to_inflow))
        ),
    }


def _sample_uplift_offer(
    rng: np.random.Generator,
    industry: str,
    offer_type: str,
    liquidity_pressure: float,
) -> float:
    """Sample revenue uplift conditional on the offer type and modifiers."""
    if industry in {"Retail", "Hospitality"} and offer_type == "acquiring":
        return float(rng.uniform(0.02, 0.06))
    if offer_type == "loan" and liquidity_pressure > 0.0:
        return float(rng.uniform(0.015, 0.05))
    return float(rng.uniform(0.005, 0.02))


def _sample_uplift_credit_positive(
    rng: np.random.Generator,
    liquidity_pressure: float,
) -> float:
    """Sample revenue uplift for a positive credit-limit change."""
    return float(rng.uniform(0.01, 0.035) * (1.0 + 0.6 * liquidity_pressure))


def _sample_uplift_credit_negative(rng: np.random.Generator) -> float:
    """Sample revenue change for a negative credit-limit change."""
    return float(rng.uniform(-0.02, -0.005))


def _sample_uplift_discount(
    rng: np.random.Generator,
    price_sensitivity: float,
) -> float:
    """Sample revenue uplift for an applied tariff discount."""
    return float(rng.uniform(0.008, 0.03) * price_sensitivity)


@dataclass
class SynthConfig:
    """Configuration parameters for synthetic data generation.

    Attributes:
        n_clients: Number of synthetic SME clients to generate.
        seed: Random seed for reproducible generation.
        confounded: If False (default), interventions are assigned almost at
            random — independently of client covariates. This is the historical
            mode and is suitable as a sanity baseline for PSM. If True,
            intervention probabilities depend on the same covariates that drive
            the outcome (industry, business size, total bank profit, liquidity
            pressure, price sensitivity), creating observable confounding that
            PSM is supposed to correct. The two modes use the same RNG-call
            sequence — only the thresholds change — so reproducibility under
            ``seed`` is preserved per mode.
    """

    n_clients: int = 3000
    seed: int = 42
    confounded: bool = False


def generate_sme_data(
    cfg: SynthConfig,
    *,
    include_audit_columns: bool = False,
) -> pd.DataFrame:
    """Generate synthetic SME client data with realistic causal relationships.

    Creates synthetic data for SME (Small and Medium Enterprises) clients with
    correlated features following a predefined causal DAG structure. Includes
    industry-specific patterns, regional variations, and intervention effects.

    Args:
        cfg: Configuration object containing generation parameters.
        include_audit_columns: If True, the returned DataFrame additionally
            contains internal sampling components used for ground-truth
            evaluation: ``_uplift_offer_factual``, ``_uplift_credit_factual``,
            ``_uplift_discount_factual``, ``_growth_base``, ``_noise_eps``,
            ``_liquidity_pressure``, ``_price_sensitivity``. These columns
            constitute oracle information about the true treatment effect and
            **must not** leak into causal-discovery training, prompts to the
            LLM, or any production data flow — set this flag only inside
            evaluation scripts.

    Returns:
        DataFrame with synthetic client data. The shape and column set are
        identical to the previous behaviour unless ``include_audit_columns``
        is enabled.
    """
    logger.info(
        f"Starting synthetic data generation for {cfg.n_clients} clients"
    )
    logger.debug(f"Random seed: {cfg.seed}")

    rng = RNG(cfg.seed)
    rows = []

    # Progress logging for large datasets
    progress_interval = max(1, cfg.n_clients // 10)

    for i in range(cfg.n_clients):
        if i % progress_interval == 0 and i > 0:
            logger.debug(
                f"Generated {i}/{cfg.n_clients} clients ({i / cfg.n_clients * 100:.1f}%)"
            )
        industry = rng.choice(INDUSTRIES)
        region = rng.choice(REGIONS)
        size = rng.choice(SIZES, p=[0.45, 0.4, 0.15])
        years = max(0, int(rng.normal(7, 5)))
        tenure = max(0, min(years, int(abs(rng.normal(years * 0.7, 2)))))

        # Доходность и объёмы
        base_inflow = (
            1_000_000
            * _industry_multiplier(industry)
            * _size_multiplier(size)
            * _region_multiplier(region)
            * rng.uniform(0.7, 1.3)
        )
        txn_intensity_k, outflow_ratio_k = _industry_txn_intensity(industry)
        avg_outflow = base_inflow * rng.uniform(0.65, 0.95) * outflow_ratio_k
        txn_count = int(np.clip(rng.normal(120 * txn_intensity_k, 45), 10, 800))
        avg_balance = max(
            0.0, (base_inflow - avg_outflow) * rng.uniform(0.5, 1.8)
        )

        prod_types = _sample_product_types(rng, industry, size)
        num_products = len(prod_types) + int(rng.uniform(0, 2))

        has_deposit = int("deposit" in prod_types)
        has_card = int("card" in prod_types)
        has_acquiring = int("acquiring" in prod_types)
        has_payroll = int("payroll" in prod_types)
        has_loan = int("loan" in prod_types)

        # прибыль банка растёт с продуктами и объёмами
        total_profit = (
            0.004 * base_inflow
            + 0.01 * avg_outflow
            + 500 * txn_count  # небольшая комиссия за транзакции
            + 30_000 * ("loan" in prod_types)
            + 15_000 * ("acquiring" in prod_types)
        ) * rng.uniform(0.6, 1.4)

        # Модификаторы (без rng) — вычисляются здесь, потому что в
        # confounded-режиме они нужны для назначения интервенций. Их же
        # используют ниже формулы uplift'а, что снимает любое расхождение
        # между условиями назначения и величины эффекта.
        _mods_now = _compute_modifiers(
            avg_inflow=base_inflow,
            avg_outflow=avg_outflow,
            avg_balance=avg_balance,
        )
        liquidity_pressure = _mods_now["liquidity_pressure"]
        price_sensitivity = _mods_now["price_sensitivity"]

        # Вероятности назначения интервенций.
        # В randomized-режиме (confounded=False) — фиксированные пороги, как
        # в исходном генераторе; PSM здесь подтверждает только корректность
        # реализации матчинга на почти случайных данных.
        # В confounded-режиме — вероятности зависят от ковариат, формирующих
        # одновременно и outcome (т. е. рисуется наблюдаемое смещение), что и
        # является целевой нагрузкой для метода propensity-score-matching.
        if cfg.confounded:
            # Используем total_profit напрямую: тогда ребро
            # `Total_Bank_Profit -> New_Product_Offer` в ground_truth_edges
            # отражает реальную зависимость в коде, а не прокси через
            # коррелирующий base_inflow. Делитель 50_000 нормирует значение
            # к разумному диапазону: для small/medium бизнеса total_profit
            # ~ 5_000–80_000, что после деления даёт 0.1–1.6.
            profit_intensity = min(total_profit / 50_000.0, 2.5)
            industry_offer_bonus = 0.05 if industry in {"Retail", "Hospitality"} else 0.0
            size_offer_bonus = (
                0.04 if size == "medium"
                else 0.02 if size == "small"
                else 0.0
            )
            p_offer = max(
                0.05,
                min(0.55,
                    0.08 + 0.10 * profit_intensity + industry_offer_bonus + size_offer_bonus),
            )
            p_credit_event = max(
                0.05,
                min(0.45, 0.08 + 0.20 * liquidity_pressure),
            )
            p_discount = max(
                0.05,
                min(0.50, 0.08 + 0.30 * price_sensitivity),
            )
        else:
            p_offer = 0.25
            p_credit_event = 0.18
            p_discount = 0.22

        new_offer_flag = int(rng.random() < p_offer)
        new_offer_type = (
            _offer_type(rng, industry) if new_offer_flag else "none"
        )
        credit_limit_change = 0.0
        if rng.random() < p_credit_event:
            if cfg.confounded and liquidity_pressure > 0.0:
                # Под confounding — клиенты с давлением на ликвидность
                # систематически получают увеличение лимита (положительный сдвиг)
                credit_limit_change = rng.normal(15, 6)
            else:
                credit_limit_change = rng.normal(12, 8)
        tariff_discount = int(rng.random() < p_discount)

        # Базовый рост; модификаторы (liquidity_pressure, price_sensitivity)
        # уже вычислены выше при назначении интервенций.
        growth_base = _industry_trend(industry) + rng.normal(0.0, 0.02)

        uplift_offer = (
            _sample_uplift_offer(
                rng, industry, new_offer_type, liquidity_pressure,
            )
            if new_offer_flag
            else 0.0
        )

        if credit_limit_change > 0:
            uplift_credit = _sample_uplift_credit_positive(rng, liquidity_pressure)
        elif credit_limit_change < 0:
            uplift_credit = _sample_uplift_credit_negative(rng)
        else:
            uplift_credit = 0.0

        uplift_discount = (
            _sample_uplift_discount(rng, price_sensitivity)
            if tariff_discount
            else 0.0
        )

        noise_eps = float(rng.normal(0.0, 0.01))

        revenue_growth_rate = float(
            growth_base + uplift_offer + uplift_credit + uplift_discount + noise_eps
        )
        if revenue_growth_rate > 0.02:
            trend = "up"
        elif revenue_growth_rate < -0.02:
            trend = "down"
        else:
            trend = "flat"

        rows.append(
            {
                "Client_ID": f"C{i:06d}",
                # Macro
                "Industry": industry,
                "Region": region,
                "Business_Size": size,
                "Years_in_Operation": years,
                # Relationship
                "Client_Tenure": tenure,
                "Num_Products": num_products,
                "Has_Deposit": has_deposit,
                "Has_Card": has_card,
                "Has_Acquiring": has_acquiring,
                "Has_Payroll": has_payroll,
                "Has_Loan": has_loan,
                "Total_Bank_Profit": round(total_profit, 2),
                # Transactional
                "Avg_Monthly_Inflow": round(base_inflow, 2),
                "Avg_Monthly_Outflow": round(avg_outflow, 2),
                "Monthly_Transaction_Count": txn_count,
                "Avg_Account_Balance": round(avg_balance, 2),
                # Interventions
                "New_Product_Offer": new_offer_flag,
                "New_Product_Offer_Type": new_offer_type,
                "Credit_Limit_Change": round(credit_limit_change, 2),
                "Tariff_Discount": int(tariff_discount),
                # Outcomes
                "Revenue_Growth_Rate": round(revenue_growth_rate, 4),
                "Revenue_Trend": trend,
                # Внутренние компоненты, использованные при генерации.
                # Сохраняются для аудита и для расчёта эмпирического эталона
                # ATT/ATE при оценке PSM (см. generate_with_counterfactuals).
                "_uplift_offer_factual": round(uplift_offer, 6),
                "_uplift_credit_factual": round(uplift_credit, 6),
                "_uplift_discount_factual": round(uplift_discount, 6),
                "_growth_base": round(growth_base, 6),
                "_noise_eps": round(noise_eps, 6),
                "_liquidity_pressure": liquidity_pressure,
                "_price_sensitivity": round(price_sensitivity, 6),
            }
        )

    df = pd.DataFrame(rows)

    df["Net_Cashflow"] = (df["Avg_Monthly_Inflow"] - df["Avg_Monthly_Outflow"]).round(2)

    # бинирование Activity_Bucket / Balance_Bucket после генерации всего df
    # Пороги выберем по тертилям распределений, чтобы были стабильные размеры классов
    activity_tertiles = df["Monthly_Transaction_Count"].quantile([0.33, 0.66]).values
    balance_tertiles = df["Avg_Account_Balance"].quantile([0.33, 0.66]).values

    def _to_bucket(x: float, lo: float, hi: float) -> str:
        if x <= lo:
            return "low"
        if x <= hi:
            return "medium"
        return "high"


    df["Activity_Bucket"] = df["Monthly_Transaction_Count"].apply(
        lambda v: _to_bucket(v, activity_tertiles[0], activity_tertiles[1])
    )
    df["Balance_Bucket"] = df["Avg_Account_Balance"].apply(
        lambda v: _to_bucket(v, balance_tertiles[0], balance_tertiles[1])
    )

    if not include_audit_columns:
        # Drop oracle columns to prevent ground-truth leakage into downstream
        # consumers (CSV exports, causal-discovery algorithms, RAG corpora,
        # LLM prompts). Evaluation scripts opt in via include_audit_columns=True.
        audit_cols = [c for c in df.columns if str(c).startswith("_")]
        if audit_cols:
            df = df.drop(columns=audit_cols)

    logger.success(
        f"Generated synthetic dataset with {len(df)} clients and {len(df.columns)} features"
    )
    logger.debug(f"DataFrame shape: {df.shape}")
    return df


def ground_truth_edges(*, confounded: bool = False) -> List[Dict]:
    """Ground-truth causal edges actually realised by the synthetic generator.

    The set of edges depends on the generation mode declared in
    :class:`SynthConfig`:

    - ``confounded=False`` (randomized intervention assignment) — only those
      edges that the generator actually produces. Treatment indicators
      (``New_Product_Offer``, ``Credit_Limit_Change``, ``Tariff_Discount``)
      are independent of covariates, so no covariate→intervention edges
      exist in the ground truth.
    - ``confounded=True`` (intervention assignment depends on covariates) —
      additionally includes the realised covariate→intervention edges:
      ``Total_Bank_Profit → New_Product_Offer``,
      ``Industry → New_Product_Offer``,
      ``Business_Size → New_Product_Offer``,
      ``Avg_Account_Balance → Credit_Limit_Change``,
      ``Avg_Monthly_Inflow → Credit_Limit_Change``,
      ``Avg_Monthly_Outflow → Tariff_Discount``,
      ``Avg_Account_Balance → Tariff_Discount``.

    The two graphs differ only in those mode-specific edges; everything else
    (industry → outcome, intervention → outcome, etc.) is identical.

    Args:
        confounded: Match the assignment regime of the data generator.

    Returns:
        List of edges with keys ``source``, ``target``, ``sign``, ``rationale``.
    """
    E = []

    def add(src: str, dst: str, sign: str = "+", note: str = "") -> None:
        E.append(
            {"source": src, "target": dst, "sign": sign, "rationale": note}
        )

    # Macro -> Relationship/Transactional
    add("Industry", 
        "Avg_Monthly_Inflow", 
        "+", 
        "Отраслевой масштаб спроса"
    )
    add(
        "Industry",
        "Monthly_Transaction_Count",
        "+",
        "Розница/HoReCa транзакционно активнее",
    )
    add(
        "Industry", 
        "Avg_Monthly_Outflow", 
        "+", 
        "Отраслевые затраты"
    )

    for f in ["Has_Deposit", "Has_Card", "Has_Acquiring", "Has_Payroll", "Has_Loan"]:
        add("Industry", f, "+", "Выбор типов продуктов зависит от отрасли")

    add("Region", 
        "Avg_Monthly_Inflow", 
        "+", 
        "Эффект мегаполисов"
    )
    add(
        "Business_Size",
        "Avg_Monthly_Inflow",
        "+",
        "Больше размер — больше оборот",
    )
    add("Business_Size", 
        "Num_Products", 
        "+", 
        "Крупнее — больше продуктов"
    )

    # Relationship
    add(
        "Client_Tenure", 
        "Num_Products", 
        "+", 
        "С ростом стажа растёт кросс‑селл"
    )

    add(
        "Client_Tenure", 
        "Revenue_Growth_Rate", 
        "+", 
        "Долгосрочные клиенты более стабильны",
        )

    for f in ["Has_Deposit", "Has_Card", "Has_Acquiring", "Has_Payroll", "Has_Loan"]:
        add(f, "Total_Bank_Profit", "+", "Тип продукта влияет на комиссии/проценты")
    
    add(
        "Num_Products",
        "Total_Bank_Profit",
        "+",
        "Больше продуктов — выше прибыль",
    )

    # Конфаундирующие рёбра — реализуются только в confounded-режиме
    # (см. SynthConfig.confounded). В randomized-режиме (по умолчанию)
    # назначение интервенций не зависит от ковариат, и эти рёбра отсутствуют.
    if confounded:
        add(
            "Total_Bank_Profit",
            "New_Product_Offer",
            "+",
            "Confounding: прибыльным клиентам чаще делают предложение",
        )
        add(
            "Industry",
            "New_Product_Offer",
            "+",
            "Confounding: розница/HoReCa получают предложения чаще",
        )
        add(
            "Business_Size",
            "New_Product_Offer",
            "+",
            "Confounding: средний и малый бизнес получают предложения чаще микро",
        )
        add(
            "Avg_Account_Balance",
            "Credit_Limit_Change",
            "-",
            "Confounding: низкий баланс → давление на ликвидность → выше "
            "вероятность изменения лимита (через liquidity_pressure)",
        )
        add(
            "Avg_Monthly_Inflow",
            "Credit_Limit_Change",
            "-",
            "Confounding: низкий приток → давление на ликвидность → выше "
            "вероятность изменения лимита (через liquidity_pressure)",
        )
        add(
            "Avg_Monthly_Outflow",
            "Tariff_Discount",
            "+",
            "Confounding: высокий отток → высокая ценовая чувствительность → "
            "выше вероятность скидки (через price_sensitivity)",
        )
        add(
            "Avg_Account_Balance",
            "Tariff_Discount",
            "-",
            "Confounding: низкий баланс → высокая ценовая чувствительность → "
            "выше вероятность скидки (через price_sensitivity)",
        )
        add(
            "Avg_Monthly_Inflow",
            "Tariff_Discount",
            "-",
            "Confounding: низкий приток → высокая ценовая чувствительность "
            "(price_sensitivity зависит от inflow в знаменателе обеих "
            "компонент: outflow/inflow и balance/inflow)",
        )

    # Transactional
    add(
        "Avg_Monthly_Inflow",
        "Avg_Monthly_Outflow",
        "+",
        "Размер выручки часто положительно коррелирует с затратами",
    )
    add(
        "Avg_Monthly_Inflow",
        "Monthly_Transaction_Count",
        "+",
        "Больше оборот — больше платежей",
    )
    add(
        "Net_Cashflow",
        "Avg_Account_Balance",
        "+",
        "Баланс зависит от чистого потока",
    )
    add(
        "Avg_Monthly_Outflow",
        "Avg_Account_Balance",
        "-",
        "Расходы «съедают» остаток",
    )

    # Transactional -> Derived
    add(
        "Avg_Monthly_Inflow", 
        "Net_Cashflow", 
        "+", 
        "Net = Inflow - Outflow"
    )
    add(
        "Avg_Monthly_Outflow", 
        "Net_Cashflow", 
        "-", 
        "Net = Inflow - Outflow"
    )

    add(
        "Monthly_Transaction_Count", 
        "Total_Bank_Profit", 
        "+", 
        "Комиссионный доход от транзакций",
        )

    # Interventions -> Outcomes
    add(
        "New_Product_Offer",
        "Revenue_Growth_Rate",
        "+",
        "Удачное предложение даёт аплифт",
    )
    add(
        "New_Product_Offer_Type",
        "Revenue_Growth_Rate",
        "+",
        "Матч типа оффера с отраслью",
    )
    add(
        "Credit_Limit_Change",
        "Revenue_Growth_Rate",
        "+",
        "Лимит помогает ограниченным в ликвидности",
    )
    add(
        "Tariff_Discount",
        "Revenue_Growth_Rate",
        "+",
        "Снижение комиссий ускоряет рост и удержание",
    )
    # Transactional -> Outcomes
    add(
        "Avg_Account_Balance", 
        "Revenue_Growth_Rate", 
        "+", 
        "Объем баланса позволяет инвестировать в рост",
        )

    add("Industry",
        "Revenue_Growth_Rate",
        "+",
        "Отраслевой тренд роста"
    )
    return E


# -----------------------------------------------------------------------------
# Ground-truth treatment effects for PSM evaluation
# -----------------------------------------------------------------------------
#
# The synthetic generator above samples each per-client uplift from a uniform
# distribution conditional on industry, liquidity pressure, etc. To validate
# PSM against a true effect, we need the *expected* uplift conditional on the
# observed covariates — i.e. the per-client CATE the matching procedure should
# recover. The functions below compute these expectations analytically from the
# same parameters used in `generate_sme_data`.

# Industry-conditional weights for product offer types (mirror `_offer_type`).
_OFFER_TYPE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Retail":       {"loan": 0.15, "acquiring": 0.55, "card": 0.20, "payroll": 0.10},
    "Hospitality":  {"loan": 0.15, "acquiring": 0.55, "card": 0.20, "payroll": 0.10},
    "Manufacturing":{"loan": 0.55, "acquiring": 0.15, "card": 0.15, "payroll": 0.15},
    "Construction": {"loan": 0.55, "acquiring": 0.15, "card": 0.15, "payroll": 0.15},
    "IT_Services":  {"loan": 0.20, "acquiring": 0.15, "card": 0.45, "payroll": 0.20},
    "Healthcare":   {"loan": 0.35, "acquiring": 0.20, "card": 0.25, "payroll": 0.20},
}


def _expected_uplift_offer(industry: str, liquidity_pressure: float) -> float:
    """Expected revenue uplift if `New_Product_Offer` is hypothetically applied.

    Mirrors the conditional logic of `generate_sme_data` but takes midpoints
    of the underlying uniform distributions instead of single samples.
    """
    weights = _OFFER_TYPE_WEIGHTS[industry]
    expected = 0.0
    for offer_type, w in weights.items():
        if industry in {"Retail", "Hospitality"} and offer_type == "acquiring":
            mean_uplift = 0.04          # midpoint of U(0.02, 0.06)
        elif offer_type == "loan" and liquidity_pressure > 0.0:
            mean_uplift = 0.0325        # midpoint of U(0.015, 0.05)
        else:
            mean_uplift = 0.0125        # midpoint of U(0.005, 0.02)
        expected += w * mean_uplift
    return expected


def _expected_uplift_credit_positive(liquidity_pressure: float) -> float:
    """Expected revenue uplift if `Credit_Limit_Change > 0` is hypothetically applied."""
    # Midpoint of U(0.01, 0.035) is 0.0225; multiplier (1 + 0.6 * liq_pressure).
    return 0.0225 * (1.0 + 0.6 * liquidity_pressure)


def _expected_uplift_discount(price_sensitivity: float) -> float:
    """Expected revenue uplift if `Tariff_Discount` is hypothetically applied."""
    # Midpoint of U(0.008, 0.03) is 0.019.
    return 0.019 * price_sensitivity


def generate_with_counterfactuals(
    cfg: SynthConfig,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Generate synthetic data and per-client *empirical* treatment effects.

    Per-client CATE is built from the actual uplift draws used by the
    generator: for treated clients it equals the factual uplift sample
    (`_uplift_*_factual`), for non-treated clients it is sampled with the same
    formulas using a *separate* RNG seeded as ``cfg.seed + 10_000_000``. This
    keeps `generate_sme_data` bit-for-bit reproducible and removes any
    distance between the data and the ground truth used for PSM evaluation.

    True ATE = mean of per-client CATE across the population.
    True ATT = mean of CATE on actually-treated clients
              (collapses to the mean of factual uplift on those clients).

    For each intervention we additionally return the analytical expectation
    (midpoints of the underlying uniforms, weighted by industry probabilities
    where applicable). It serves as a convergence check: at large N it should
    match the empirical figure up to ``σ / √n`` noise.
    """
    df = generate_sme_data(cfg, include_audit_columns=True)

    rng_cf = np.random.default_rng(cfg.seed + 10_000_000)
    n = len(df)

    industry_arr = df["Industry"].to_numpy()
    liq_arr = df["_liquidity_pressure"].to_numpy()
    price_arr = df["_price_sensitivity"].to_numpy()

    factual_offer = df["_uplift_offer_factual"].to_numpy()
    factual_credit = df["_uplift_credit_factual"].to_numpy()
    factual_discount = df["_uplift_discount_factual"].to_numpy()

    treated_offer = (df["New_Product_Offer"].to_numpy() == 1)
    treated_credit_positive = (df["Credit_Limit_Change"].to_numpy() > 0)
    treated_discount = (df["Tariff_Discount"].to_numpy() == 1)

    # Per-client potential-outcome contrast (CATE).
    # Sequential loop because `_offer_type` and `_sample_uplift_*` consume the
    # counterfactual RNG one client at a time; the order is deterministic in
    # `cfg.seed`.
    cate_offer = np.empty(n, dtype=float)
    cate_credit_positive = np.empty(n, dtype=float)
    cate_discount = np.empty(n, dtype=float)

    for i in range(n):
        ind = str(industry_arr[i])
        lp = float(liq_arr[i])
        ps = float(price_arr[i])

        # Offer
        if treated_offer[i]:
            cate_offer[i] = factual_offer[i]
        else:
            cf_offer_type = _offer_type(rng_cf, ind)
            cate_offer[i] = _sample_uplift_offer(rng_cf, ind, cf_offer_type, lp)

        # Credit limit, binarised on the positive direction
        if treated_credit_positive[i]:
            cate_credit_positive[i] = factual_credit[i]  # already > 0
        else:
            cate_credit_positive[i] = _sample_uplift_credit_positive(rng_cf, lp)

        # Tariff discount
        if treated_discount[i]:
            cate_discount[i] = factual_discount[i]
        else:
            cate_discount[i] = _sample_uplift_discount(rng_cf, ps)

    # Analytical expectations (asymptotic check, do not enter the PSM benchmark).
    cate_offer_analytical = np.array([
        _expected_uplift_offer(ind, lp)
        for ind, lp in zip(industry_arr, liq_arr)
    ])
    cate_credit_pos_analytical = np.array([
        _expected_uplift_credit_positive(lp) for lp in liq_arr
    ])
    cate_discount_analytical = np.array([
        _expected_uplift_discount(ps) for ps in price_arr
    ])

    df = df.assign(
        cate_offer=cate_offer.round(6),
        cate_credit_positive=cate_credit_positive.round(6),
        cate_discount=cate_discount.round(6),
        T_credit_positive=treated_credit_positive.astype(int),
    )

    def _att(cate: np.ndarray, treated: np.ndarray) -> float:
        return float(cate[treated].mean()) if treated.any() else float("nan")

    true_effects: Dict[str, Dict[str, float]] = {
        "New_Product_Offer": {
            "ate": float(cate_offer.mean()),
            "att": _att(cate_offer, treated_offer),
            "n_treated_factual": int(treated_offer.sum()),
            "ate_analytical": float(cate_offer_analytical.mean()),
            "att_analytical": _att(cate_offer_analytical, treated_offer),
        },
        "Credit_Limit_Change_positive": {
            "ate": float(cate_credit_positive.mean()),
            "att": _att(cate_credit_positive, treated_credit_positive),
            "n_treated_factual": int(treated_credit_positive.sum()),
            "ate_analytical": float(cate_credit_pos_analytical.mean()),
            "att_analytical": _att(cate_credit_pos_analytical, treated_credit_positive),
        },
        "Tariff_Discount": {
            "ate": float(cate_discount.mean()),
            "att": _att(cate_discount, treated_discount),
            "n_treated_factual": int(treated_discount.sum()),
            "ate_analytical": float(cate_discount_analytical.mean()),
            "att_analytical": _att(cate_discount_analytical, treated_discount),
        },
    }

    logger.info(
        "Counterfactual generation: n={}; "
        "empirical ATT offer={:.4f}, credit+={:.4f}, discount={:.4f}",
        len(df),
        true_effects["New_Product_Offer"]["att"],
        true_effects["Credit_Limit_Change_positive"]["att"],
        true_effects["Tariff_Discount"]["att"],
    )
    return df, true_effects
