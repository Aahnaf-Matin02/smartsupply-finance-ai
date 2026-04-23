"""
SmartSupply Finance AI — Shared Utilities
Feature encoding helpers used by both train_models.py and app.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Categorical columns and their fixed orderings ────────────────────────────
CAT_COLS = {
    "Product_Category": ["Electronics", "Apparel", "FMCG", "Pharma", "Industrial", "Food & Bev"],
    "Season":           ["Q1_Winter", "Q2_Spring", "Q3_Summer", "Q4_Holiday"],
    "Region":           ["Dhaka", "Chittagong", "Sylhet", "Rajshahi", "Khulna"],
}

# Pre-built int maps (deterministic, no fitted LabelEncoder needed at inference)
CAT_MAPS: dict[str, dict] = {
    col: {v: i for i, v in enumerate(vals)}
    for col, vals in CAT_COLS.items()
}

# ── Numeric feature columns ───────────────────────────────────────────────────
BASE_FEATURE_COLS = [
    "Historical_Sales",       "Promotion_Flag",         "Lead_Time_Days",
    "Supplier_Rating",        "Delivery_Delay_History", "Order_Quantity",
    "Inventory_On_Hand",      "Reorder_Point",          "Stockout_History",
    "Holding_Cost",           "Transport_Cost",         "Demand_Variability",
    "Unit_Cost",              "Selling_Price",           "Gross_Margin",
    "AP_Days",                "AR_Days",                "Inventory_Days",
    "Cash_On_Hand",           "Purchase_Value",         "Monthly_Operating_Cost",
    "Working_Capital",        "Financing_Cost",         "Discount_Rate",
    "Stock_Cover_Days",       "Reorder_Gap",            "Demand_Stock_Ratio",
    "Carrying_Cost_Ratio",    "Cash_Conversion_Cycle",
    # Encoded categoricals appended last
    "Product_Category_Enc",   "Season_Enc",             "Region_Enc",
]


def encode_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer-encoded columns for categorical features and return the copy."""
    df = df.copy()
    for col, mapping in CAT_MAPS.items():
        df[f"{col}_Enc"] = df[col].map(mapping).fillna(0).astype(int)
    return df


def get_X(df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature matrix (all BASE_FEATURE_COLS, NaN → 0)."""
    df_enc = encode_df(df)
    return df_enc[BASE_FEATURE_COLS].fillna(0)


def encode_single(input_dict: dict) -> pd.DataFrame:
    """
    Convert a user-supplied dict of raw values into a single-row feature DataFrame
    ready for model.predict().  All categoricals are encoded deterministically.
    """
    row = {col: input_dict.get(col, 0) for col in BASE_FEATURE_COLS
           if col not in ("Product_Category_Enc", "Season_Enc", "Region_Enc")}

    row["Product_Category_Enc"] = CAT_MAPS["Product_Category"].get(
        input_dict.get("Product_Category", "FMCG"), 2
    )
    row["Season_Enc"] = CAT_MAPS["Season"].get(
        input_dict.get("Season", "Q2_Spring"), 1
    )
    row["Region_Enc"] = CAT_MAPS["Region"].get(
        input_dict.get("Region", "Dhaka"), 0
    )

    return pd.DataFrame([row])[BASE_FEATURE_COLS]
