# 🏭 SmartSupply Finance AI

### ML-Based Inventory, Supplier Risk & Cash-Flow Optimization System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Business Problem

> *"How can a company reduce stockouts and excess inventory while protecting cash flow and choosing safer suppliers?"*

Many firms today face supply chain volatility, trade disruptions, supplier uncertainty, and demand shocks — while finance teams struggle with fragmented data and forecasting pressure. **SmartSupply Finance AI** bridges the gap between SCM and Finance by using machine learning to make integrated procurement and cash-flow decisions.

---

## 🎯 What It Does

Instead of SCM and Finance working in silos, this system answers 4 critical questions simultaneously:

| Question | ML Task | Model |
|---|---|---|
| How much will we sell next period? | Demand Forecasting | XGBoost Regressor |
| Will this supplier deliver on time? | Delay Classification | XGBoost Classifier |
| Is this SKU at risk of stockout/overstock? | Inventory Risk Scoring | Random Forest Classifier |
| How will this order affect our cash flow? | Cash Stress Prediction | Random Forest Regressor |

---

## 🚀 Live Dashboard Features

| Page | What You See |
|---|---|
| 🏠 Overview | KPIs, alerts, portfolio risk summary |
| 📈 Demand Forecasting | Actual vs Predicted, seasonal trends, promotion uplift |
| 🚚 Supplier Risk | Delay leaderboard, rating vs risk scatter, category analysis |
| 📦 Inventory Risk | Risk heatmap by region & category, stockout loss analysis |
| 💰 Cash Flow Impact | CCC breakdown, working capital vs stress, financial summary |
| 🔍 SHAP Explainability | Feature importance, beeswarm plots, top drivers |
| ⚡ Live Prediction Engine | Real-time predictions + radar chart + reorder recommendations |

---

## 📁 Project Structure

```
smartsupply/
│
├── app.py                    # Streamlit dashboard (7 pages)
├── generate_data.py          # Synthetic dataset generator (3,000 rows × 40 cols)
├── train_models.py           # Full ML training pipeline (4 models)
├── utils.py                  # Shared feature encoding utilities
├── requirements.txt          # All Python dependencies
│
├── data/
│   └── smartsupply_dataset.csv   # Generated dataset
│
└── models/
    ├── demand_model.pkl           # XGBoost Regressor
    ├── supplier_model.pkl         # XGBoost Classifier
    ├── inventory_risk_model.pkl   # Random Forest Classifier
    ├── inventory_risk_encoder.pkl # Label encoder
    ├── cashflow_model.pkl         # Random Forest Regressor
    └── metrics.json               # All model evaluation scores
```

---

## 🧠 Dataset — 40 Features

**SCM Variables**
`SKU_ID` · `Supplier_ID` · `Product_Category` · `Season` · `Region` · `Historical_Sales` · `Promotion_Flag` · `Lead_Time_Days` · `Supplier_Rating` · `Delivery_Delay_History` · `Order_Quantity` · `Inventory_On_Hand` · `Reorder_Point` · `Stockout_History` · `Holding_Cost` · `Transport_Cost` · `Demand_Variability`

**Finance Variables**
`Unit_Cost` · `Selling_Price` · `Gross_Margin` · `AP_Days` · `AR_Days` · `Inventory_Days` · `Cash_On_Hand` · `Purchase_Value` · `Monthly_Operating_Cost` · `Working_Capital` · `Financing_Cost` · `Discount_Rate` · `Loss_from_Stockout` · `Loss_from_Overstock`

**Engineered Features**
`Stock_Cover_Days` · `Reorder_Gap` · `Demand_Stock_Ratio` · `Carrying_Cost_Ratio` · `Cash_Conversion_Cycle`

**Target Variables**
`Future_Sales` · `Delay_Flag` · `Inventory_Risk_Class` · `Cash_Stress_Score`

---

## 📊 Model Performance

| Model | Metric | Score |
|---|---|---|
| Demand Forecasting (XGBoost) | R² | 0.668 |
| Supplier Delay (XGBoost) | ROC-AUC | 0.573 |
| Inventory Risk (Random Forest) | Accuracy | **93.8%** |
| Cash Stress (Random Forest) | R² | **0.955** |

---

## ⚙️ Installation & Run

```bash
# 1. Clone the repo
git clone https://github.com/Aahnaf-Matin02/smartsupply-finance-ai.git
cd smartsupply-finance-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset (auto-runs on first launch too)
python generate_data.py

# 4. Train all models
python train_models.py

# 5. Launch the dashboard
streamlit run app.py
```

> ✅ The app auto-generates data and trains models on first launch if files are missing — fully self-contained.

---

## 🔬 Tech Stack

- **ML**: `scikit-learn`, `xgboost`, `shap`
- **Data**: `pandas`, `numpy`, `scipy`
- **Visualization**: `plotly`, `matplotlib`, `seaborn`
- **Dashboard**: `streamlit`
- **Persistence**: `joblib`

---

## 💡 Key Outputs

Given any SKU's inputs, the system instantly generates:
- ✅ Next-period demand forecast
- ✅ Supplier delay probability + risk flag
- ✅ Inventory risk class (Low / Medium / High)
- ✅ Cash stress score (0–100)
- ✅ Recommended reorder quantity (safety-stock method)
- ✅ Financial impact estimate (holding cost, stockout loss, cash %)
- ✅ Multi-dimensional radar risk chart

---

## 🌏 Context

Built with Bangladesh FMCG context in mind — currency in BDT (৳), regions include Dhaka, Chittagong, Sylhet, Rajshahi, Khulna; product categories cover Electronics, Apparel, FMCG, Pharma, Industrial, and Food & Beverage.

---

## 👤 Author

**Chowdhury Aahnaf Matin**
BBA (Supply Chain Management & Finance) — North South University, Dhaka
[GitHub](https://github.com/Aahnaf-Matin02)

---

## 📄 License

This project is licensed under the MIT License.
