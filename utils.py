"""
SmartSupply Finance AI — Shared Utilities (Unilever Dataset)
"""
import pandas as pd
import numpy as np

CAT_COLS = {
    "Product_Type":      ["haircare", "skincare", "cosmetics"],
    "Customer_Segment":  ["Non-binary", "Female", "Male", "Unknown"],
    "Carrier":           ["Carrier A", "Carrier B", "Carrier C"],
    "Supplier_ID":       ["Supplier 1", "Supplier 2", "Supplier 3", "Supplier 4", "Supplier 5"],
    "Inspection_Result": ["Pass", "Pending", "Fail"],
    "Transport_Mode":    ["Road", "Air", "Rail", "Sea"],
    "Route":             ["Route A", "Route B", "Route C"],
}

CAT_MAPS = {col: {v: i for i, v in enumerate(vals)} for col, vals in CAT_COLS.items()}

BASE_FEATURE_COLS = [
    # Raw numeric SCM
    "Price", "Availability", "Units_Sold", "Stock_Levels",
    "Shipping_Lead_Time", "Order_Quantity", "Shipping_Time",
    "Shipping_Cost", "Supplier_Lead_Time", "Production_Volume",
    "Mfg_Lead_Time", "Mfg_Cost", "Defect_Rate", "Logistics_Cost",
    # Engineered
    "Gross_Margin_Pct", "Revenue_Per_Unit", "Daily_Demand",
    "Stock_Cover_Days", "Demand_Stock_Ratio", "Total_Lead_Time",
    "Cost_Per_Unit", "Reorder_Point", "Reorder_Gap",
    "Carrying_Cost_Ratio", "Supply_Efficiency", "Shipping_Cost_Ratio",
    # Encoded categoricals
    "Product_Type_Enc", "Customer_Segment_Enc", "Carrier_Enc",
    "Supplier_ID_Enc", "Inspection_Result_Enc", "Transport_Mode_Enc", "Route_Enc",
]


def encode_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in CAT_MAPS.items():
        df[f"{col}_Enc"] = df[col].map(mapping).fillna(0).astype(int)
    return df


def get_X(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = encode_df(df)
    return df_enc[BASE_FEATURE_COLS].fillna(0)


def encode_single(input_dict: dict) -> pd.DataFrame:
    row = {}
    raw_cols = [c for c in BASE_FEATURE_COLS
                if not c.endswith("_Enc")]
    for c in raw_cols:
        row[c] = input_dict.get(c, 0)
    for col, mapping in CAT_MAPS.items():
        row[f"{col}_Enc"] = mapping.get(input_dict.get(col, ""), 0)
    return pd.DataFrame([row])[BASE_FEATURE_COLS]
