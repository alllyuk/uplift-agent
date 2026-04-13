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


@dataclass
class SynthConfig:
    """Configuration parameters for synthetic data generation.

    Attributes:
        n_clients: Number of synthetic SME clients to generate.
        seed: Random seed for reproducible generation.
    """

    n_clients: int = 3000
    seed: int = 42


def generate_sme_data(cfg: SynthConfig) -> pd.DataFrame:
    """Generate synthetic SME client data with realistic causal relationships.

    Creates synthetic data for SME (Small and Medium Enterprises) clients with
    correlated features following a predefined causal DAG structure. Includes
    industry-specific patterns, regional variations, and intervention effects.

    Args:
        cfg: Configuration object containing generation parameters.

    Returns:
        DataFrame with synthetic client data including all feature columns.
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

        # Интервенции
        new_offer_flag = int(rng.random() < 0.25)
        new_offer_type = (
            _offer_type(rng, industry) if new_offer_flag else "none"
        )
        credit_limit_change = 0.0
        if rng.random() < 0.18:
            credit_limit_change = rng.normal(
                12, 8
            )  # % изменение лимита, чаще +/-
        tariff_discount = int(rng.random() < 0.22)

        # Базовый рост + влияние интервенций и «ликвидности»
        growth_base = _industry_trend(industry) + rng.normal(0.0, 0.02)
        liquidity_pressure = (
            1.0 if (avg_balance / (base_inflow + 1e-9)) < 0.45 else 0.0
        )

        uplift_offer = 0.0
        if new_offer_flag:
            if (
                industry in {"Retail", "Hospitality"}
                and new_offer_type == "acquiring"
            ):
                uplift_offer = rng.uniform(0.02, 0.06)
            elif new_offer_type == "loan" and liquidity_pressure > 0.0:
                uplift_offer = rng.uniform(0.015, 0.05)
            else:
                uplift_offer = rng.uniform(0.005, 0.02)

        uplift_credit = 0.0
        if credit_limit_change > 0:
            uplift_credit = rng.uniform(0.01, 0.035) * (
                1.0 + 0.6 * liquidity_pressure
            )
        elif credit_limit_change < 0:
            uplift_credit = rng.uniform(-0.02, -0.005)

        price_sensitivity = 0.6 * (avg_outflow / (base_inflow + 1e-9)) + 0.4 * (1.0 - min(1.0, avg_balance / (base_inflow + 1e-9)))
        uplift_discount = rng.uniform(0.008, 0.03) * price_sensitivity if tariff_discount else 0.0

        revenue_growth_rate = float(
            growth_base
            + uplift_offer
            + uplift_credit
            + uplift_discount
            + rng.normal(0.0, 0.01)
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

    logger.success(
        f"Generated synthetic dataset with {len(df)} clients and {len(df.columns)} features"
    )
    logger.debug(f"DataFrame shape: {df.shape}")
    return df


def ground_truth_edges() -> List[Dict]:
    """Generate ground truth causal edges from the synthetic data generation process.

    Returns the actual causal relationships (DAG edges) that were used in the
    synthetic data generation. Each edge includes source, target, sign, and rationale.

    Returns:
        List of dictionaries representing causal edges with keys:
        - source: Source variable name
        - target: Target variable name
        - sign: Relationship direction (+ or -)
        - rationale: Human-readable explanation of the relationship
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

    add(
        "Total_Bank_Profit", 
        "New_Product_Offer", 
        "+", 
        "Прибыльным клиентам предлагают больше продуктов",
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
