"""Shared constants for DAG layers and allowed variables."""

from __future__ import annotations

# Layer indices for DAG constraints: prohibit backward edges
LAYER_INDEX = {
    "Industry": 0,
    "Region": 0,
    "Business_Size": 0,
    "Years_in_Operation": 0,
    "Client_Tenure": 1,
    "Num_Products": 1,
    "Has_Deposit": 1,
    "Has_Card": 1,
    "Has_Acquiring": 1,
    "Has_Payroll": 1,
    "Has_Loan": 1,
    "Total_Bank_Profit": 1,
    "Avg_Monthly_Inflow": 2,
    "Avg_Monthly_Outflow": 2,
    "Monthly_Transaction_Count": 2,
    "Avg_Account_Balance": 2,
    "Net_Cashflow": 2,
    "Activity_Bucket": 2,
    "Balance_Bucket": 2,
    "New_Product_Offer": 3,
    "New_Product_Offer_Type": 3,
    "Credit_Limit_Change": 3,
    "Tariff_Discount": 3,
    "Revenue_Growth_Rate": 4,
    "Revenue_Trend": 4,
}

ALLOWED_VARS = set(LAYER_INDEX.keys())

