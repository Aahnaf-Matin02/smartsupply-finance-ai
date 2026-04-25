"""
SmartSupply Finance AI — Model Training Pipeline (Unilever Dataset)
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBRegressor, XGBClassifier
from utils import get_X, BASE_FEATURE_COLS

warnings.filterwarnings("ignore")


def train_demand_model(df):
    print("\n📈  Training Demand Forecasting (XGBoost Regressor)…")
    X, y = get_X(df), df["Future_Sales"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    y_pred = model.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    mape = float(np.mean(np.abs((y_te.values - y_pred) / (y_te.values + 1))) * 100)
    metrics = {"MAE": round(mae,2), "RMSE": round(rmse,2), "R2": round(r2,4), "MAPE": round(mape,2)}
    print(f"   MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return model, metrics


def train_supplier_model(df):
    print("\n🚚  Training Supplier Delay Classifier (XGBoost)…")
    X, y = get_X(df), df["Delay_Flag"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    metrics = {
        "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_te, y_pred, zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_te, y_prob), 4),
    }
    print(f"   Acc={metrics['Accuracy']}  F1={metrics['F1']}  AUC={metrics['ROC_AUC']}")
    return model, metrics


def train_inventory_risk_model(df):
    print("\n📦  Training Inventory Risk Classifier (Random Forest)…")
    X = get_X(df)
    le = LabelEncoder()
    y  = le.fit_transform(df["Inventory_Risk_Class"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        "Accuracy":    round(accuracy_score(y_te, y_pred), 4),
        "F1_Weighted": round(f1_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "Classes":     list(le.classes_),
    }
    print(f"   Acc={metrics['Accuracy']}  F1(w)={metrics['F1_Weighted']}  Classes={metrics['Classes']}")
    return model, le, metrics


def train_cashflow_model(df):
    print("\n💰  Training Cash Stress Predictor (Random Forest Regressor)…")
    X, y = get_X(df), df["Cash_Stress_Score"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=8,
                                   min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    metrics = {"MAE": round(mae,2), "RMSE": round(rmse,2), "R2": round(r2,4)}
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return model, metrics


def train_all_models():
    print("=" * 60)
    print("  SmartSupply Finance AI — Training on Unilever Dataset")
    print("=" * 60)

    csv_path = "data/smartsupply_dataset.csv"
    if not os.path.exists(csv_path):
        from generate_data import process_dataset
        df = process_dataset()
    else:
        df = pd.read_csv(csv_path)
        print(f"✅  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    os.makedirs("models", exist_ok=True)

    demand_model,  demand_m           = train_demand_model(df)
    supplier_model, supplier_m        = train_supplier_model(df)
    inv_model, inv_le, inv_m          = train_inventory_risk_model(df)
    cash_model,    cash_m             = train_cashflow_model(df)

    joblib.dump(demand_model,   "models/demand_model.pkl")
    joblib.dump(supplier_model, "models/supplier_model.pkl")
    joblib.dump(inv_model,      "models/inventory_risk_model.pkl")
    joblib.dump(inv_le,         "models/inventory_risk_encoder.pkl")
    joblib.dump(cash_model,     "models/cashflow_model.pkl")

    all_metrics = {
        "demand":         demand_m,
        "supplier_delay": supplier_m,
        "inventory_risk": inv_m,
        "cash_stress":    cash_m,
    }
    with open("models/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n✅  All models saved → models/")
    print("=" * 60)
    return all_metrics


if __name__ == "__main__":
    train_all_models()
