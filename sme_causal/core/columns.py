from __future__ import annotations

"""Canonical column names and common feature groups.

Use these constants instead of repeating raw strings across modules.
"""

# Identifiers
CLIENT_ID = "Client_ID"

# Macro
INDUSTRY = "Industry"
REGION = "Region"
BUSINESS_SIZE = "Business_Size"
YEARS_IN_OPERATION = "Years_in_Operation"

# Relationship
CLIENT_TENURE = "Client_Tenure"
NUM_PRODUCTS = "Num_Products"
HAS_DEPOSIT = "Has_Deposit"
HAS_CARD = "Has_Card"
HAS_ACQUIRING = "Has_Acquiring"
HAS_PAYROLL = "Has_Payroll"
HAS_LOAN = "Has_Loan"
TOTAL_BANK_PROFIT = "Total_Bank_Profit"

# Derived (new)
NET_CASHFLOW = "Net_Cashflow"
ACTIVITY_BUCKET = "Activity_Bucket"
BALANCE_BUCKET = "Balance_Bucket"

# Transactional
AVG_MONTHLY_INFLOW = "Avg_Monthly_Inflow"
AVG_MONTHLY_OUTFLOW = "Avg_Monthly_Outflow"
MONTHLY_TRANSACTION_COUNT = "Monthly_Transaction_Count"
AVG_ACCOUNT_BALANCE = "Avg_Account_Balance"

# Interventions
NEW_PRODUCT_OFFER = "New_Product_Offer"
NEW_PRODUCT_OFFER_TYPE = "New_Product_Offer_Type"
CREDIT_LIMIT_CHANGE = "Credit_Limit_Change"
TARIFF_DISCOUNT = "Tariff_Discount"

# Outcomes
REVENUE_GROWTH_RATE = "Revenue_Growth_Rate"
REVENUE_TREND = "Revenue_Trend"


INTERVENTIONS_RANGES = {
    NEW_PRODUCT_OFFER: [0, 1],
    NEW_PRODUCT_OFFER_TYPE: ["acquiring", "loan", "payroll", "deposit", "card"],
    CREDIT_LIMIT_CHANGE: "greater than 0.0",
    TARIFF_DISCOUNT: "between 0.0 and 100.0",
}


# Context fields used by Agent/UI (order matters for display)
CONTEXT_FIELDS = [
    INDUSTRY,
    REGION,
    BUSINESS_SIZE,
    YEARS_IN_OPERATION,
    CLIENT_TENURE,
    NUM_PRODUCTS,
    HAS_DEPOSIT,
    HAS_CARD,
    HAS_ACQUIRING,
    HAS_PAYROLL,
    HAS_LOAN,
    TOTAL_BANK_PROFIT,
    AVG_MONTHLY_INFLOW,
    AVG_MONTHLY_OUTFLOW,
    MONTHLY_TRANSACTION_COUNT,
    AVG_ACCOUNT_BALANCE,
    NET_CASHFLOW,
    ACTIVITY_BUCKET,
    BALANCE_BUCKET,
    NEW_PRODUCT_OFFER,
    NEW_PRODUCT_OFFER_TYPE,
    CREDIT_LIMIT_CHANGE,
    TARIFF_DISCOUNT,
    REVENUE_GROWTH_RATE,
    REVENUE_TREND,
]


__all__ = [
    # identifiers
    "CLIENT_ID",
    # macro
    "INDUSTRY",
    "REGION",
    "BUSINESS_SIZE",
    "YEARS_IN_OPERATION",
    # relationship
    "CLIENT_TENURE",
    "NUM_PRODUCTS",
    "HAS_DEPOSIT",
    "HAS_CARD",
    "HAS_ACQUIRING",
    "HAS_PAYROLL",
    "HAS_LOAN",
    "TOTAL_BANK_PROFIT",
    # transactional
    "AVG_MONTHLY_INFLOW",
    "AVG_MONTHLY_OUTFLOW",
    "MONTHLY_TRANSACTION_COUNT",
    "AVG_ACCOUNT_BALANCE",
    # derived
    "NET_CASHFLOW",
    "ACTIVITY_BUCKET",
    "BALANCE_BUCKET",
    # interventions
    "NEW_PRODUCT_OFFER",
    "NEW_PRODUCT_OFFER_TYPE",
    "CREDIT_LIMIT_CHANGE",
    "TARIFF_DISCOUNT",
    # outcomes
    "REVENUE_GROWTH_RATE",
    "REVENUE_TREND",
    # groups
    "CONTEXT_FIELDS",
]
