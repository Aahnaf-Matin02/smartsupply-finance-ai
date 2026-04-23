"""
SmartSupply Finance AI — Data Generation
Generates a realistic synthetic dataset combining SCM and Finance variables.
"""

import pandas as pd
import numpy as np
import os


def generate_dataset(n_records: int = 3000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    n = n_records

    # ── IDENTIFIERS ────────────────────────────────────────────────────────
    sku_pool     = [f"SKU-{i:04d}" for i in range(1, 201)]
    supplier_pool = [f"SUP-{i:02d}"  for i in range(1, 26)]

    sku_ids      = np.random.choice(sku_pool,      n)
    supplier_ids = np.random.choice(supplier_pool, n)

    # ── CATEGORICAL ─────────────────────────────────────────────────────────
    categories = ["Electronics", "Apparel", "FMCG", "Pharma", "Industrial", "Food & Bev"]
    seasons    = ["Q1_Winter",   "Q2_Spring", "Q3_Summer", "Q4_Holiday"]
    regions    = ["Dhaka",       "Chittagong", "Sylhet", "Rajshahi", "Khulna"]

    product_cat = np.random.choice(categories, n, p=[0.20, 0.15, 0.25, 0.15, 0.10, 0.15])
    season      = np.random.choice(seasons,    n)
    region      = np.random.choice(regions,    n)

    # ── SCM VARIABLES ────────────────────────────────────────────────────────
    historical_sales      = (np.random.gamma(4, 40, n) + 20).astype(int)
    promotion_flag        = np.random.binomial(1, 0.25, n)
    lead_time_days        = np.random.randint(3, 36, n)
    supplier_rating       = np.round(np.clip(np.random.normal(3.8, 0.6, n), 2.0, 5.0), 1)
    delivery_delay_history = np.random.poisson(2.5, n)
    order_quantity        = (historical_sales * np.random.uniform(0.7, 1.8, n)).astype(int)
    inventory_on_hand     = np.random.randint(5, 601, n)
    demand_variability    = np.round(np.random.beta(2, 5, n), 3)

    daily_demand  = historical_sales / 30.0
    reorder_point = (daily_demand * lead_time_days * 1.2 + 15).astype(int)

    # Stockout history correlated with low stock vs demand
    low_stock_flag    = (inventory_on_hand < reorder_point).astype(int)
    stockout_history  = np.maximum(0, np.random.poisson(1.5, n) + low_stock_flag * 2 - 1)

    holding_cost   = np.round(np.random.uniform(0.5, 12.0, n), 2)
    transport_cost = np.round(np.random.uniform(5.0, 80.0, n), 2)

    # ── FINANCE VARIABLES ────────────────────────────────────────────────────
    unit_cost     = np.round(np.random.gamma(5, 20, n) + 5,  2)
    selling_price = np.round(unit_cost * np.random.uniform(1.15, 2.8, n), 2)
    gross_margin  = np.round((selling_price - unit_cost) / selling_price * 100, 2)

    ap_days          = np.random.randint(15, 76, n)
    ar_days          = np.random.randint(7,  61, n)
    inventory_days   = np.random.randint(10, 121, n)
    cash_on_hand     = np.round(np.random.lognormal(10, 1.2, n), 2)
    purchase_value   = np.round(unit_cost * order_quantity, 2)
    monthly_op_cost  = np.round(np.random.lognormal(8, 0.8, n), 2)
    financing_cost   = np.round(np.random.uniform(0.06, 0.18, n), 4)
    discount_rate    = np.round(np.random.uniform(0.02, 0.12, n), 4)

    inventory_value  = inventory_on_hand * unit_cost
    working_capital  = np.round(cash_on_hand + inventory_value - purchase_value * (ap_days / 30.0), 2)

    loss_from_stockout  = np.round(stockout_history * selling_price * np.random.randint(1, 8, n), 2)
    excess_stock        = np.maximum(0, inventory_on_hand - reorder_point * 1.5)
    loss_from_overstock = np.round(excess_stock * holding_cost, 2)

    # ── ENGINEERED FEATURES (pre-computed) ───────────────────────────────────
    stock_cover_days  = np.round(inventory_on_hand / (daily_demand + 0.01), 1)
    reorder_gap       = inventory_on_hand - reorder_point
    demand_stock_ratio = np.round(historical_sales / (inventory_on_hand + 1), 3)
    carrying_cost_ratio = np.round((holding_cost * inventory_on_hand) / (purchase_value + 1), 4)
    ccc               = ar_days + inventory_days - ap_days

    # ── TARGET VARIABLES ─────────────────────────────────────────────────────

    # 1) Future Sales — regression
    season_mult = {
        "Q1_Winter": 0.88, "Q2_Spring": 1.00,
        "Q3_Summer": 1.08, "Q4_Holiday": 1.35
    }
    s_mult       = np.array([season_mult[s] for s in season])
    noise        = np.random.normal(0, demand_variability * historical_sales)
    future_sales = np.maximum(1,
        (historical_sales * s_mult * (1 + 0.35 * promotion_flag) + noise).astype(int)
    )

    # 2) Delay Flag — binary classification
    delay_prob = np.clip(
        0.15
        + (5 - supplier_rating) * 0.08
        + delivery_delay_history * 0.04
        + lead_time_days * 0.005
        + demand_variability * 0.20
        + np.random.normal(0, 0.05, n),
        0.05, 0.92
    )
    delay_flag = np.random.binomial(1, delay_prob, n)

    # 3) Inventory Risk Class — multiclass (Low / Medium / High)
    risk_raw = np.clip(
        demand_variability * 0.30
        + stockout_history / 10.0
        + low_stock_flag * 0.40
        + (demand_stock_ratio > 2).astype(int) * 0.30
        + np.random.normal(0, 0.05, n),
        0, 1
    )
    inv_risk_class = np.where(
        risk_raw < 0.35, "Low",
        np.where(risk_raw < 0.65, "Medium", "High")
    )

    # 4) Cash Stress Score — 0–100 regression target
    cash_stress = np.clip(
        ccc / 150.0 * 25
        + financing_cost * 150
        + np.where(cash_on_hand > 0, loss_from_stockout / (cash_on_hand + 1), 0) * 50
        + monthly_op_cost / (cash_on_hand + 1) * 30
        + purchase_value / (np.maximum(working_capital, 1) + 1) * 15
        + np.random.normal(0, 3, n),
        0, 100
    ).round(2)

    # ── ASSEMBLE DATAFRAME ───────────────────────────────────────────────────
    df = pd.DataFrame({
        "SKU_ID":                  sku_ids,
        "Supplier_ID":             supplier_ids,
        "Product_Category":        product_cat,
        "Season":                  season,
        "Region":                  region,
        "Historical_Sales":        historical_sales,
        "Promotion_Flag":          promotion_flag,
        "Lead_Time_Days":          lead_time_days,
        "Supplier_Rating":         supplier_rating,
        "Delivery_Delay_History":  delivery_delay_history,
        "Order_Quantity":          order_quantity,
        "Inventory_On_Hand":       inventory_on_hand,
        "Reorder_Point":           reorder_point,
        "Stockout_History":        stockout_history,
        "Holding_Cost":            holding_cost,
        "Transport_Cost":          transport_cost,
        "Demand_Variability":      demand_variability,
        "Unit_Cost":               unit_cost,
        "Selling_Price":           selling_price,
        "Gross_Margin":            gross_margin,
        "AP_Days":                 ap_days,
        "AR_Days":                 ar_days,
        "Inventory_Days":          inventory_days,
        "Cash_On_Hand":            cash_on_hand,
        "Purchase_Value":          purchase_value,
        "Monthly_Operating_Cost":  monthly_op_cost,
        "Working_Capital":         working_capital,
        "Financing_Cost":          financing_cost,
        "Discount_Rate":           discount_rate,
        "Loss_from_Stockout":      loss_from_stockout,
        "Loss_from_Overstock":     loss_from_overstock,
        "Stock_Cover_Days":        stock_cover_days,
        "Reorder_Gap":             reorder_gap,
        "Demand_Stock_Ratio":      demand_stock_ratio,
        "Carrying_Cost_Ratio":     carrying_cost_ratio,
        "Cash_Conversion_Cycle":   ccc,
        # Targets
        "Future_Sales":            future_sales,
        "Delay_Flag":              delay_flag,
        "Inventory_Risk_Class":    inv_risk_class,
        "Cash_Stress_Score":       cash_stress,
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/smartsupply_dataset.csv", index=False)
    print(f"✅  Dataset saved → data/smartsupply_dataset.csv  "
          f"[{n:,} rows × {len(df.columns)} cols]")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.describe().T[["mean", "std", "min", "max"]].round(2))
