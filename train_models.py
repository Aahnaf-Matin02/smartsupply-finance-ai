"""
SmartSupply Finance AI — Model Training Pipeline
Trains 4 models: Demand Forecast · Supplier Delay · Inventory Risk · Cash Stress
Saves all artifacts to models/ directory.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBRegressor, XGBClassifier

from utils import get_X, BASE_FEATURE_COLS

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# A) DEMAND FORECASTING  (XGBoost Regressor)
# ════════════════════════════════════════════════════════════════════════════
def train_demand_model(df: pd.DataFrame):
    print("\n📈  Training Demand Forecasting  (XGBoost Regressor) …")
    X = get_X(df)
    y = df["Future_Sales"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred = model.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    mape = float(np.mean(np.abs((y_te.values - y_pred) / (y_te.values + 1))) * 100)

    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2),
               "R2": round(r2, 4),  "MAPE": round(mape, 2)}
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return model, metrics


# ════════════════════════════════════════════════════════════════════════════
# B) SUPPLIER DELAY CLASSIFICATION  (XGBoost Classifier)
# ════════════════════════════════════════════════════════════════════════════
def train_supplier_model(df: pd.DataFrame):
    print("\n🚚  Training Supplier Delay Classifier  (XGBoost) …")
    X = get_X(df)
    y = df["Delay_Flag"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    metrics = {
        "Accuracy":  round(accuracy_score(y_te, y_pred),  4),
        "Precision": round(precision_score(y_te, y_pred), 4),
        "Recall":    round(recall_score(y_te, y_pred),    4),
        "F1":        round(f1_score(y_te, y_pred),        4),
        "ROC_AUC":   round(roc_auc_score(y_te, y_prob),   4),
    }
    print(f"   Acc={metrics['Accuracy']}  F1={metrics['F1']}  AUC={metrics['ROC_AUC']}")
    return model, metrics


# ════════════════════════════════════════════════════════════════════════════
# C) INVENTORY RISK SCORING  (Random Forest Classifier)
# ════════════════════════════════════════════════════════════════════════════
def train_inventory_risk_model(df: pd.DataFrame):
    print("\n📦  Training Inventory Risk Classifier  (Random Forest) …")
    X = get_X(df)

    le = LabelEncoder()
    y  = le.fit_transform(df["Inventory_Risk_Class"])  # High=0 Low=1 Medium=2

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=12,
        min_samples_leaf=5, random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    metrics = {
        "Accuracy":   round(accuracy_score(y_te, y_pred), 4),
        "F1_Weighted": round(f1_score(y_te, y_pred, average="weighted"), 4),
        "Classes":    list(le.classes_),
    }
    print(f"   Acc={metrics['Accuracy']}  F1(w)={metrics['F1_Weighted']}  "
          f"Classes={metrics['Classes']}")
    return model, le, metrics


# ════════════════════════════════════════════════════════════════════════════
# D) CASH STRESS PREDICTION  (Random Forest Regressor)
# ════════════════════════════════════════════════════════════════════════════
def train_cashflow_model(df: pd.DataFrame):
    print("\n💰  Training Cash Stress Predictor  (Random Forest Regressor) …")
    X = get_X(df)
    y = df["Cash_Stress_Score"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=300, max_depth=12,
        min_samples_leaf=5, random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)

    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return model, metrics


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def train_all_models():
    print("=" * 62)
    print("  SmartSupply Finance AI — Model Training Pipeline")
    print("=" * 62)

    # Load or generate dataset
    if not os.path.exists("data/smartsupply_dataset.csv"):
        from generate_data import generate_dataset
        df = generate_dataset()
    else:
        df = pd.read_csv("data/smartsupply_dataset.csv")
        print(f"✅  Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

    os.makedirs("models", exist_ok=True)

    demand_model,  demand_metrics            = train_demand_model(df)
    supplier_model, supplier_metrics         = train_supplier_model(df)
    inv_model, inv_le, inv_metrics           = train_inventory_risk_model(df)
    cash_model,    cash_metrics              = train_cashflow_model(df)

    # Persist
    joblib.dump(demand_model,   "models/demand_model.pkl")
    joblib.dump(supplier_model, "models/supplier_model.pkl")
    joblib.dump(inv_model,      "models/inventory_risk_model.pkl")
    joblib.dump(inv_le,         "models/inventory_risk_encoder.pkl")
    joblib.dump(cash_model,     "models/cashflow_model.pkl")

    all_metrics = {
        "demand":          demand_metrics,
        "supplier_delay":  supplier_metrics,
        "inventory_risk":  inv_metrics,
        "cash_stress":     cash_metrics,
    }
    with open("models/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n✅  All models saved to models/")
    print("=" * 62)
    return all_metrics


if __name__ == "__main__":
    m = train_all_models()
    print("\nSummary:")
    for name, vals in m.items():
        print(f"  {name}: {vals}")
