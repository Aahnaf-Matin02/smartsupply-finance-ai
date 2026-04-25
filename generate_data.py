"""
SmartSupply Finance AI — Data Processor (Unilever Dataset)
Loads the real Supply_Chain_Analytics_Uniliver.xlsx, engineers features,
derives ML targets, and augments to 800 rows for robust training.
"""

import os
import numpy as np
import pandas as pd

COL_MAP = {
    "Product type":               "Product_Type",
    "SKU":                        "SKU_ID",
    "Price":                      "Price",
    "Availability":               "Availability",
    "Number of products sold":    "Units_Sold",
    "Revenue generated":          "Revenue",
    "Customer demographics":      "Customer_Segment",
    "Stock levels":               "Stock_Levels",
    "Lead times":                 "Shipping_Lead_Time",
    "Order quantities":           "Order_Quantity",
    "Shipping times":             "Shipping_Time",
    "Shipping carriers":          "Carrier",
    "Shipping costs":             "Shipping_Cost",
    "Supplier name":              "Supplier_ID",
    "Lead time":                  "Supplier_Lead_Time",
    "Production volumes":         "Production_Volume",
    "Manufacturing lead time":    "Mfg_Lead_Time",
    "Manufacturing costs":        "Mfg_Cost",
    "Inspection results":         "Inspection_Result",
    "Defect rates":               "Defect_Rate",
    "Transportation modes":       "Transport_Mode",
    "Routes":                     "Route",
    "Costs":                      "Logistics_Cost",
}

CAT_COLS = {
    "Product_Type":      ["haircare", "skincare", "cosmetics"],
    "Customer_Segment":  ["Non-binary", "Female", "Male", "Unknown"],
    "Carrier":           ["Carrier A", "Carrier B", "Carrier C"],
    "Supplier_ID":       ["Supplier 1", "Supplier 2", "Supplier 3", "Supplier 4", "Supplier 5"],
    "Inspection_Result": ["Pass", "Pending", "Fail"],
    "Transport_Mode":    ["Road", "Air", "Rail", "Sea"],
    "Route":             ["Route A", "Route B", "Route C"],
}


def load_and_engineer(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path)
    df  = raw.rename(columns=COL_MAP)
    df  = df.dropna(subset=["Units_Sold", "Price", "Stock_Levels"])
    df["Units_Sold"]   = df["Units_Sold"].clip(lower=1)
    df["Stock_Levels"] = df["Stock_Levels"].clip(lower=0)
    df["Defect_Rate"]  = df["Defect_Rate"].clip(lower=0)

    df["Gross_Margin_Pct"]   = ((df["Price"] - df["Mfg_Cost"]) / df["Price"] * 100).round(2)
    df["Revenue_Per_Unit"]   = (df["Revenue"] / df["Units_Sold"]).round(3)
    df["Daily_Demand"]       = (df["Units_Sold"] / 30.0).round(3)
    df["Stock_Cover_Days"]   = (df["Stock_Levels"] / (df["Daily_Demand"] + 0.01)).round(1)
    df["Demand_Stock_Ratio"] = (df["Units_Sold"] / (df["Stock_Levels"] + 1)).round(3)
    df["Total_Lead_Time"]    = df["Supplier_Lead_Time"] + df["Mfg_Lead_Time"]
    df["Cost_Per_Unit"]      = (df["Logistics_Cost"] / (df["Units_Sold"] + 1)).round(3)
    df["Reorder_Point"]      = (df["Daily_Demand"] * df["Total_Lead_Time"] * 1.2 + 5).astype(int)
    df["Reorder_Gap"]        = df["Stock_Levels"] - df["Reorder_Point"]
    df["Carrying_Cost_Ratio"]= ((df["Mfg_Cost"] * df["Stock_Levels"]) /
                                 (df["Price"] * df["Units_Sold"] + 1)).round(4)
    df["Supply_Efficiency"]  = (df["Production_Volume"] / (df["Order_Quantity"] + 1)).round(3)
    df["Shipping_Cost_Ratio"]= (df["Shipping_Cost"] / (df["Revenue"] + 1)).round(4)
    return df


def derive_targets(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    n = len(df)

    noise      = np.random.normal(1.0, 0.12, n)
    avail_mult = 1 + (df["Availability"].values - 50) / 200
    defect_drag= 1 - df["Defect_Rate"].values / 20
    df["Future_Sales"] = np.maximum(1,
        (df["Units_Sold"] * noise * avail_mult * defect_drag).round().astype(int))

    med_ship = df["Shipping_Time"].median()
    dscore = (
        (df["Shipping_Time"] > med_ship).astype(int) * 0.40
        + (df["Defect_Rate"] > 2.5).astype(int) * 0.30
        + (df["Inspection_Result"] == "Fail").astype(int) * 0.30
        + np.random.normal(0, 0.08, n)
    )
    df["Delay_Flag"] = (dscore > 0.35).astype(int)

    low_stock = (df["Stock_Levels"] < df["Reorder_Point"]).astype(int)
    rscore = (
        (1 / (df["Stock_Cover_Days"] + 1)) * 30
        + df["Defect_Rate"] / 5 * 25
        + low_stock * 35
        + df["Demand_Stock_Ratio"] / 5 * 10
        + np.random.normal(0, 3, n)
    )
    pct = np.percentile(rscore, [33, 67])
    df["Inventory_Risk_Class"] = np.where(
        rscore < pct[0], "Low", np.where(rscore < pct[1], "Medium", "High"))

    cost_to_rev    = df["Logistics_Cost"] / (df["Revenue"] + 1)
    mfg_margin_inv = df["Mfg_Cost"] / (df["Price"] + 0.01)
    ship_burden    = df["Shipping_Cost"] / (df["Revenue"] + 1)
    stock_burden   = (df["Reorder_Point"] - df["Stock_Levels"]).clip(lower=0) / (df["Units_Sold"] + 1)
    cash_raw = cost_to_rev * 35 + mfg_margin_inv * 30 + ship_burden * 20 + stock_burden * 15
    mn, mx = cash_raw.min(), cash_raw.max()
    df["Cash_Stress_Score"] = ((cash_raw - mn) / (mx - mn + 1e-9) * 100).round(2)
    return df


def augment(df: pd.DataFrame, target_rows: int = 800, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    if len(df) >= target_rows:
        return df.reset_index(drop=True)
    needed   = target_rows - len(df)
    numeric  = df.select_dtypes(include=[np.number]).columns.tolist()
    t_cols   = ["Future_Sales","Delay_Flag","Inventory_Risk_Class","Cash_Stress_Score"]
    j_cols   = [c for c in numeric if c not in t_cols]
    rows     = []
    for i in np.random.choice(len(df), needed, replace=True):
        row = df.iloc[i].copy()
        for c in j_cols:
            row[c] = max(0, row[c] * np.random.normal(1.0, 0.08))
        rows.append(row)
    combined = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    combined = derive_targets(combined, seed=seed + 1)
    return combined.reset_index(drop=True)


def process_dataset(
    excel_path: str = "data/Supply_Chain_Analytics_Uniliver.xlsx",
    out_csv:    str = "data/smartsupply_dataset.csv",
    target_rows: int = 800,
) -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)
    df = load_and_engineer(excel_path)
    df = derive_targets(df)
    df = augment(df, target_rows=target_rows)
    df.to_csv(out_csv, index=False)
    print(f"✅  Dataset saved → {out_csv}  [{len(df):,} rows × {len(df.columns)} cols]")
    return df


# keep old entry-point name so app.py import still works
def generate_dataset():
    return process_dataset()


if __name__ == "__main__":
    df = process_dataset()
    print(df[["SKU_ID","Product_Type","Units_Sold","Future_Sales",
              "Delay_Flag","Inventory_Risk_Class","Cash_Stress_Score"]].head(10))
