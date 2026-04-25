"""
SmartSupply Finance AI — Streamlit Dashboard
ML-Based Inventory, Supplier Risk & Cash-Flow Optimization
Run: streamlit run app.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Working-dir guard (run from project root) ────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# ── Auto-setup on first run ──────────────────────────────────────────────────
if not os.path.exists("data/smartsupply_dataset.csv"):
    with st.spinner("⚙️ Generating dataset for the first time …"):
        from generate_data import generate_dataset
        generate_dataset()

if not os.path.exists("models/demand_model.pkl"):
    with st.spinner("🤖 Training models for the first time (takes ~30 s) …"):
        from train_models import train_all_models
        train_all_models()
    st.rerun()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartSupply Finance AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ─── Hide default Streamlit header & footer only ─── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0b1a2e 0%, #0f2744 60%, #0b1a2e 100%) !important;
    top: 56px !important;
    height: calc(100vh - 56px) !important;
}
section[data-testid="stSidebar"] * { color: #c9dff0 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.85rem; }

/* ─── Sidebar collapse/expand toggle button — ALWAYS VISIBLE ─── */
[data-testid="collapsedControl"] {
    top: 64px !important;
    background: #0f2744 !important;
    border-radius: 0 10px 10px 0 !important;
    width: 28px !important;
    height: 52px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 3px 0 12px rgba(0,0,0,0.35) !important;
    border: 1px solid rgba(0,180,216,0.4) !important;
    border-left: none !important;
    z-index: 999 !important;
    opacity: 1 !important;
    visibility: visible !important;
    cursor: pointer !important;
}
[data-testid="collapsedControl"]:hover {
    background: #1a3f70 !important;
    width: 34px !important;
    transition: all 0.2s ease;
}
[data-testid="collapsedControl"] svg {
    color: #00d4ff !important;
    fill: #00d4ff !important;
    width: 18px !important;
    height: 18px !important;
}

/* ─── Sidebar's own collapse button (inside sidebar) ─── */
section[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"],
section[data-testid="stSidebar"] button[kind="header"] {
    color: #00d4ff !important;
    background: rgba(0,180,216,0.15) !important;
    border-radius: 8px !important;
}

/* ─── Fixed top navbar ─── */
.topbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 56px;
    background: linear-gradient(90deg, #0b1a2e 0%, #0f2744 50%, #0b1a2e 100%);
    border-bottom: 2px solid rgba(0,180,216,0.4);
    display: flex;
    align-items: center;
    padding: 0 20px;
    z-index: 1000;
    box-shadow: 0 2px 20px rgba(0,0,0,0.4);
    gap: 16px;
}
.topbar-logo {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: white;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 8px;
}
.topbar-logo span { color: #00d4ff; }
.topbar-divider {
    width: 1px; height: 28px;
    background: rgba(0,180,216,0.35);
    margin: 0 4px;
}
.topbar-nav {
    display: flex;
    gap: 4px;
    flex-wrap: nowrap;
    overflow-x: auto;
    scrollbar-width: none;
    flex: 1;
}
.topbar-nav::-webkit-scrollbar { display: none; }
.topbar-pill {
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    color: #7eb8d4;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,180,216,0.15);
    white-space: nowrap;
    cursor: pointer;
    text-decoration: none;
    transition: all 0.15s;
    letter-spacing: 0.3px;
}
.topbar-pill:hover {
    background: rgba(0,180,216,0.2);
    color: white;
    border-color: rgba(0,180,216,0.5);
}
.topbar-pill.active {
    background: linear-gradient(90deg, #0077b6, #00b4d8);
    color: white;
    border-color: #00b4d8;
    font-weight: 600;
}
.topbar-badge {
    margin-left: auto;
    background: rgba(0,180,216,0.15);
    border: 1px solid rgba(0,180,216,0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.68rem;
    color: #7eb8d4;
    white-space: nowrap;
    font-family: 'Space Mono', monospace;
}

/* ─── Push main content below topbar ─── */
.main .block-container {
    padding: 4.8rem 2.2rem 2rem !important;
    max-width: 1400px;
}

/* ─── KPI card ─── */
.kpi-card {
    background: linear-gradient(135deg, #0f2744 0%, #1a3f70 100%);
    border: 1px solid rgba(0,180,216,0.25);
    border-radius: 14px;
    padding: 20px 22px 16px;
    color: white;
    margin-bottom: 8px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    transition: transform .15s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-card .kv { font-size: 2rem; font-weight: 700; color: #00d4ff;
                 font-family: 'Space Mono', monospace; line-height: 1.1; }
.kpi-card .kl { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.5px;
                 color: #7eb8d4; margin-top: 6px; }
.kpi-card .ks { font-size: 0.78rem; color: #a0c4d8; margin-top: 2px; }

/* ─── Alert pills ─── */
.alert-red    { background:#e63946; border-radius:10px; padding:11px 16px;
                color:white; margin:6px 0; font-size:.85rem;
                box-shadow:0 3px 12px rgba(230,57,70,.35); }
.alert-amber  { background:#f4a261; border-radius:10px; padding:11px 16px;
                color:#1a1a1a; margin:6px 0; font-size:.85rem; }
.alert-green  { background:#2ec4b6; border-radius:10px; padding:11px 16px;
                color:#0a2a26; margin:6px 0; font-size:.85rem; }

/* ─── Section header ─── */
.sec-hdr {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.92rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #0077b6;
    border-bottom: 2px solid #00b4d8;
    padding-bottom: 5px;
    margin: 22px 0 14px;
}

/* ─── Prediction result box ─── */
.pred-box {
    background: white;
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 2px 14px rgba(0,0,0,.08);
    border-left: 5px solid #00b4d8;
    margin: 10px 0;
}
.pred-box h4 { margin: 0 0 10px; color: #0f2744; }

/* ─── Big prediction card ─── */
.big-pred {
    background: linear-gradient(135deg, #0f2744, #1a3f70);
    border-radius: 16px;
    padding: 24px 28px;
    color: white;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 0 8px 28px rgba(0,0,0,.3);
}
.big-pred .bv { font-size: 2.4rem; font-weight: 700;
                 font-family: 'Space Mono', monospace; }
.big-pred .bl { font-size: 0.72rem; text-transform: uppercase;
                 letter-spacing: 1.5px; color: #7eb8d4; margin-top: 4px; }

/* ─── Page title ─── */
.pg-title {
    font-size: 1.9rem; font-weight: 700; color: #0f2744;
    border-bottom: 3px solid #00b4d8;
    padding-bottom: 10px; margin-bottom: 22px;
}
</style>
""", unsafe_allow_html=True)


# ── Load assets ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/smartsupply_dataset.csv")

@st.cache_resource
def load_models():
    return {
        "demand":   joblib.load("models/demand_model.pkl"),
        "supplier": joblib.load("models/supplier_model.pkl"),
        "inv_risk": joblib.load("models/inventory_risk_model.pkl"),
        "inv_le":   joblib.load("models/inventory_risk_encoder.pkl"),
        "cashflow": joblib.load("models/cashflow_model.pkl"),
    }

@st.cache_data
def load_metrics():
    with open("models/metrics.json") as f:
        return json.load(f)

df      = load_data()
models  = load_models()
metrics = load_metrics()

from utils import get_X, encode_single, CAT_COLS

# ── Page state ───────────────────────────────────────────────────────────────
PAGES = [
    "🏠  Overview Dashboard",
    "📈  Demand Forecasting",
    "🚚  Supplier Risk",
    "📦  Inventory Risk",
    "💰  Cash Flow Impact",
    "🔍  SHAP Explainability",
    "⚡  Live Prediction Engine",
]
PAGES_SHORT = ["Overview", "Demand", "Supplier Risk",
               "Inv. Risk", "Cash Flow", "SHAP", "⚡ Predict"]

if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0

# ── Persistent branding bar (logo only — no clickable HTML) ──────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">🏭 <span>SmartSupply</span> Finance AI</div>
    <div class="topbar-divider"></div>
    <div style="font-size:0.72rem;color:#7eb8d4;letter-spacing:.5px;">
        ML-Based Inventory &amp; Cash-Flow Intelligence
    </div>
    <div class="topbar-badge" style="margin-left:auto;">3,000 rows · 4 models · Live</div>
</div>
""", unsafe_allow_html=True)

# ── Real functional navbar using st.button ───────────────────────────────────
# CSS: style these specific buttons to look like pill tabs
st.markdown("""
<style>
/* Target only the navbar button row */
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"].nav-btn) {
    gap: 4px !important;
}
/* All nav buttons */
div.nav-row > div[data-testid="column"] > div[data-testid="stButton"] > button {
    width: 100%;
    padding: 6px 4px !important;
    border-radius: 20px !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    border: 1px solid rgba(0,180,216,0.25) !important;
    background: rgba(255,255,255,0.04) !important;
    color: #4a7fa8 !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
    line-height: 1.2 !important;
}
div.nav-row > div[data-testid="column"] > div[data-testid="stButton"] > button:hover {
    background: rgba(0,180,216,0.18) !important;
    color: #0f2744 !important;
    border-color: #00b4d8 !important;
    transform: translateY(-1px);
}
/* Active page button */
div.nav-row > div[data-testid="column"] > div[data-testid="stButton"] > button[data-active="true"],
div.nav-row .active-nav-btn button {
    background: linear-gradient(90deg, #0077b6, #00b4d8) !important;
    color: white !important;
    font-weight: 700 !important;
    border-color: #00b4d8 !important;
    box-shadow: 0 2px 10px rgba(0,180,216,0.35) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="nav-row">', unsafe_allow_html=True)
nav_cols = st.columns(7)
for i, (col, label) in enumerate(zip(nav_cols, PAGES_SHORT)):
    with col:
        is_active = (st.session_state.page_idx == i)
        btn_style = "primary" if is_active else "secondary"
        if st.button(label, key=f"nav_{i}", type=btn_style, use_container_width=True):
            st.session_state.page_idx = i
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<hr style="margin:6px 0 16px;border:none;border-top:1px solid #e8edf2;">', unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 SmartSupply Finance AI")
    st.markdown("*Inventory & Cash-Flow Risk Intelligence*")
    st.markdown("---")
    st.markdown("**📍 Navigate**")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 SmartSupply Finance AI")
    st.markdown("*Inventory & Cash-Flow Risk Intelligence*")
    st.markdown("---")
    st.markdown("**📍 Navigate**")

    sidebar_idx = st.radio(
        "Navigate",
        options=list(range(len(PAGES))),
        format_func=lambda i: PAGES[i],
        index=st.session_state.page_idx,
        key="sidebar_radio",
        label_visibility="collapsed",
    )
    if sidebar_idx != st.session_state.page_idx:
        st.session_state.page_idx = sidebar_idx
        st.rerun()

    st.markdown("---")
    st.markdown("**📊 Dataset**")
    st.caption(f"Rows: **{len(df):,}**  ·  SKUs: **{df['SKU_ID'].nunique()}**  ·  Suppliers: **{df['Supplier_ID'].nunique()}**")

    st.markdown("**🤖 Model Scores**")
    st.caption(f"Demand R²: **{metrics['demand']['R2']}**")
    st.caption(f"Delay AUC: **{metrics['supplier_delay']['ROC_AUC']}**")
    st.caption(f"Inv Risk F1: **{metrics['inventory_risk']['F1_Weighted']}**")
    st.caption(f"Cash R²: **{metrics['cash_stress']['R2']}**")

    st.markdown("---")
    st.markdown("**ML Models Used**")
    st.caption("• XGBoost Regressor (Demand)")
    st.caption("• XGBoost Classifier (Delay)")
    st.caption("• Random Forest Clf (Inv Risk)")
    st.caption("• Random Forest Reg (Cash Stress)")
    st.markdown("---")
    st.caption("💡 Collapsed sidebar? Use the top nav buttons above.")

# Derive current page from session state (single source of truth)
page = PAGES[st.session_state.page_idx]


# ════════════════════════════════════════════════════════════════════════════
# PAGE 0  —  OVERVIEW DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview Dashboard":
    st.markdown('<div class="pg-title">🏠 SmartSupply Finance AI — Overview</div>', unsafe_allow_html=True)
    st.markdown("Integrated Supply Chain & Financial Risk Intelligence Platform · Bangladesh FMCG context")

    # ── KPI row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    high_risk   = int((df["Inventory_Risk_Class"] == "High").sum())
    delay_rate  = df["Delay_Flag"].mean() * 100
    avg_stress  = df["Cash_Stress_Score"].mean()
    total_loss  = df["Loss_from_Stockout"].sum()

    for col, val, lbl, sub in [
        (c1, df["SKU_ID"].nunique(),       "Unique SKUs",          "active in dataset"),
        (c2, f"{high_risk:,}",              "High-Risk SKUs",       f"{high_risk/len(df)*100:.1f}% of records"),
        (c3, f"{delay_rate:.1f}%",          "Supplier Delay Rate",  "across all transactions"),
        (c4, f"{avg_stress:.1f}/100",       "Avg Cash Stress",      "portfolio-wide"),
        (c5, f"৳{total_loss/1e6:.1f}M",    "Total Stockout Loss",  "estimated revenue"),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kv">{val}</div>
            <div class="kl">{lbl}</div>
            <div class="ks">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Alerts + Risk donut ───────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1.6])

    with col_a:
        st.markdown('<div class="sec-hdr">🚨 Active System Alerts</div>', unsafe_allow_html=True)
        hr_skus = df[df["Inventory_Risk_Class"] == "High"]["SKU_ID"].value_counts().head(3)
        for sku, cnt in hr_skus.items():
            st.markdown(f'<div class="alert-red">🔴 <b>{sku}</b> — High stockout risk ({cnt} records)</div>', unsafe_allow_html=True)
        risky_sups = df[df["Delay_Flag"] == 1]["Supplier_ID"].value_counts().head(2)
        for sup, cnt in risky_sups.items():
            st.markdown(f'<div class="alert-amber">⚠️ <b>{sup}</b> — Frequent delays ({cnt} incidents)</div>', unsafe_allow_html=True)
        hi_cash = int((df["Cash_Stress_Score"] > 75).sum())
        if hi_cash:
            st.markdown(f'<div class="alert-amber">💸 <b>{hi_cash} SKUs</b> — Cash stress score > 75</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="sec-hdr">📊 Inventory Risk Distribution</div>', unsafe_allow_html=True)
        rc = df["Inventory_Risk_Class"].value_counts().reset_index()
        rc.columns = ["Risk", "Count"]
        fig = px.pie(rc, values="Count", names="Risk", hole=0.45,
                     color="Risk",
                     color_discrete_map={"Low": "#2ec4b6", "Medium": "#ffd166", "High": "#e63946"})
        fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                          legend=dict(orientation="h", y=-0.08))
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="sec-hdr">📈 Historical vs Future Demand by Category</div>', unsafe_allow_html=True)
        cat_df = df.groupby("Product_Category")[["Historical_Sales", "Future_Sales"]].mean().reset_index()
        fig2 = px.bar(cat_df.melt("Product_Category"),
                      x="Product_Category", y="value", color="variable", barmode="group",
                      color_discrete_map={"Historical_Sales": "#1e3a5f", "Future_Sales": "#00b4d8"},
                      labels={"value": "Avg Units", "variable": ""})
        fig2.update_layout(height=310, margin=dict(t=10, b=10), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig2, use_container_width=True)

    with col_d:
        st.markdown('<div class="sec-hdr">💰 Cash Stress Score Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(df, x="Cash_Stress_Score", nbins=35, color_discrete_sequence=["#2a5298"])
        fig3.add_vline(x=df["Cash_Stress_Score"].mean(), line_dash="dash",
                       line_color="#e63946", annotation_text=f"Mean {df['Cash_Stress_Score'].mean():.1f}")
        fig3.add_vline(x=70, line_dash="dot",
                       line_color="#f77f00", annotation_text="High-stress threshold")
        fig3.update_layout(height=310, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">📋 Latest Records (Sample)</div>', unsafe_allow_html=True)
    show_cols = ["SKU_ID", "Supplier_ID", "Product_Category", "Region",
                 "Historical_Sales", "Inventory_On_Hand", "Delay_Flag",
                 "Inventory_Risk_Class", "Cash_Stress_Score"]
    st.dataframe(df[show_cols].head(25), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1  —  DEMAND FORECASTING
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈  Demand Forecasting":
    st.markdown('<div class="pg-title">📈 Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown("**Model**: XGBoost Regressor · **Target**: `Future_Sales` (next period unit demand per SKU)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE",    metrics["demand"]["MAE"])
    c2.metric("RMSE",   metrics["demand"]["RMSE"])
    c3.metric("R² Score", metrics["demand"]["R2"])
    c4.metric("MAPE",   f"{metrics['demand']['MAPE']}%")

    st.markdown("---")

    # Actual vs Predicted scatter
    st.markdown('<div class="sec-hdr">🎯 Actual vs Predicted Demand (test sample, n=400)</div>', unsafe_allow_html=True)
    samp = df.sample(400, random_state=7)
    y_pred_samp = models["demand"].predict(get_X(samp))
    fig = px.scatter(x=samp["Future_Sales"], y=y_pred_samp,
                     color=samp["Product_Category"],
                     opacity=0.65,
                     labels={"x": "Actual Future Sales", "y": "Predicted Future Sales",
                             "color": "Category"})
    max_v = max(float(samp["Future_Sales"].max()), float(y_pred_samp.max()))
    fig.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v,
                  line=dict(color="#e63946", dash="dash", width=2))
    fig.update_layout(height=400, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="sec-hdr">📅 Demand by Season</div>', unsafe_allow_html=True)
        order = ["Q1_Winter", "Q2_Spring", "Q3_Summer", "Q4_Holiday"]
        seas_df = df.groupby("Season")[["Historical_Sales", "Future_Sales"]].mean().reindex(order).reset_index()
        fig2 = px.line(seas_df.melt("Season"), x="Season", y="value", color="variable", markers=True,
                       color_discrete_map={"Historical_Sales": "#1e3a5f", "Future_Sales": "#00b4d8"})
        fig2.update_layout(height=300, margin=dict(t=10, b=10), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.markdown('<div class="sec-hdr">🎯 Promotion Uplift by Category</div>', unsafe_allow_html=True)
        pr = df.groupby(["Product_Category", "Promotion_Flag"])["Future_Sales"].mean().reset_index()
        pr["Promotion_Flag"] = pr["Promotion_Flag"].map({0: "No Promo", 1: "On Promo"})
        fig3 = px.bar(pr, x="Product_Category", y="Future_Sales", color="Promotion_Flag",
                      barmode="group",
                      color_discrete_map={"No Promo": "#1e3a5f", "On Promo": "#f77f00"})
        fig3.update_layout(height=300, margin=dict(t=10, b=10), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="sec-hdr">📊 Demand Variability by Category</div>', unsafe_allow_html=True)
        vdf = df.groupby("Product_Category")["Demand_Variability"].mean().reset_index().sort_values("Demand_Variability")
        fig4 = px.bar(vdf, x="Demand_Variability", y="Product_Category", orientation="h",
                      color="Demand_Variability", color_continuous_scale="Reds",
                      labels={"Demand_Variability": "Avg Variability (CV)"})
        fig4.update_layout(height=290, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col_r2:
        st.markdown('<div class="sec-hdr">🗺️ Regional Demand Heatmap</div>', unsafe_allow_html=True)
        heat = df.groupby(["Region", "Product_Category"])["Future_Sales"].mean().unstack()
        fig5 = px.imshow(heat, color_continuous_scale="Blues", aspect="auto",
                         labels={"color": "Avg Future Sales"})
        fig5.update_layout(height=290, margin=dict(t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2  —  SUPPLIER RISK
# ════════════════════════════════════════════════════════════════════════════
elif page == "🚚  Supplier Risk":
    st.markdown('<div class="pg-title">🚚 Supplier Delay Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown("**Model**: XGBoost Classifier · **Target**: `Delay_Flag` (0 = On-time, 1 = Delayed)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{metrics['supplier_delay']['Accuracy']*100:.1f}%")
    c2.metric("Precision", f"{metrics['supplier_delay']['Precision']*100:.1f}%")
    c3.metric("Recall",    f"{metrics['supplier_delay']['Recall']*100:.1f}%")
    c4.metric("F1 Score",  f"{metrics['supplier_delay']['F1']*100:.1f}%")
    c5.metric("ROC-AUC",   metrics["supplier_delay"]["ROC_AUC"])

    st.markdown("---")
    st.markdown('<div class="sec-hdr">🏭 Supplier Risk Leaderboard</div>', unsafe_allow_html=True)

    sup = df.groupby("Supplier_ID").agg(
        Avg_Rating        = ("Supplier_Rating",         "mean"),
        Delay_Rate        = ("Delay_Flag",               "mean"),
        Avg_Lead_Time     = ("Lead_Time_Days",            "mean"),
        Avg_Delay_History = ("Delivery_Delay_History",   "mean"),
        Total_Txns        = ("Order_Quantity",            "count"),
    ).round(2).reset_index()
    sup["Risk_Score"] = (
        (5 - sup["Avg_Rating"]) * 15
        + sup["Delay_Rate"] * 40
        + sup["Avg_Delay_History"] * 3
    ).round(1)
    sup = sup.sort_values("Risk_Score", ascending=False)
    sup["Risk"] = pd.cut(sup["Risk_Score"], bins=[-np.inf, 20, 40, np.inf],
                          labels=["🟢 Low", "🟡 Medium", "🔴 High"])
    st.dataframe(sup.head(20), use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="sec-hdr">⭐ Rating vs Delay Rate</div>', unsafe_allow_html=True)
        fig = px.scatter(sup, x="Avg_Rating", y="Delay_Rate",
                         size="Total_Txns", color="Risk_Score",
                         color_continuous_scale="RdYlGn_r",
                         hover_data=["Supplier_ID"],
                         labels={"Avg_Rating": "Avg Supplier Rating",
                                 "Delay_Rate": "Delay Rate"})
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="sec-hdr">📦 Delay Rate by Category</div>', unsafe_allow_html=True)
        cd = df.groupby("Product_Category")["Delay_Flag"].mean().reset_index().sort_values("Delay_Flag")
        fig2 = px.bar(cd, x="Delay_Flag", y="Product_Category", orientation="h",
                      color="Delay_Flag", color_continuous_scale="Reds",
                      labels={"Delay_Flag": "Proportion Delayed"})
        fig2.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col_s, col_r2 = st.columns(2)
    with col_s:
        st.markdown("**✅ Top 5 Safest Suppliers**")
        st.dataframe(sup.tail(5)[["Supplier_ID", "Avg_Rating", "Delay_Rate", "Risk_Score", "Risk"]],
                     use_container_width=True)
    with col_r2:
        st.markdown("**⚠️ Top 5 Riskiest Suppliers**")
        st.dataframe(sup.head(5)[["Supplier_ID", "Avg_Rating", "Delay_Rate", "Risk_Score", "Risk"]],
                     use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3  —  INVENTORY RISK
# ════════════════════════════════════════════════════════════════════════════
elif page == "📦  Inventory Risk":
    st.markdown('<div class="pg-title">📦 Inventory Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown("**Model**: Random Forest Classifier · **Target**: `Inventory_Risk_Class` (Low / Medium / High)")

    c1, c2 = st.columns(2)
    c1.metric("Accuracy",          f"{metrics['inventory_risk']['Accuracy']*100:.1f}%")
    c2.metric("F1 (Weighted)",     f"{metrics['inventory_risk']['F1_Weighted']*100:.1f}%")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="sec-hdr">📊 Risk Distribution by Category</div>', unsafe_allow_html=True)
        rc = df.groupby(["Product_Category", "Inventory_Risk_Class"]).size().reset_index(name="Count")
        fig = px.bar(rc, x="Product_Category", y="Count", color="Inventory_Risk_Class",
                     barmode="stack",
                     color_discrete_map={"Low": "#2ec4b6", "Medium": "#ffd166", "High": "#e63946"})
        fig.update_layout(height=340, margin=dict(t=10, b=10), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="sec-hdr">📦 Stock Cover Days by Risk Class</div>', unsafe_allow_html=True)
        fig2 = px.box(df, x="Inventory_Risk_Class", y="Stock_Cover_Days",
                      color="Inventory_Risk_Class",
                      color_discrete_map={"Low": "#2ec4b6", "Medium": "#ffd166", "High": "#e63946"},
                      category_orders={"Inventory_Risk_Class": ["Low", "Medium", "High"]})
        fig2.update_layout(height=340, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">🗺️ Risk Heatmap: Region × Category</div>', unsafe_allow_html=True)
    df2 = df.copy()
    df2["Risk_N"] = df2["Inventory_Risk_Class"].map({"Low": 1, "Medium": 2, "High": 3})
    heat = df2.groupby(["Region", "Product_Category"])["Risk_N"].mean().unstack()
    fig3 = px.imshow(heat, color_continuous_scale="RdYlGn_r", aspect="auto",
                     labels={"color": "Avg Risk (1=Low → 3=High)"})
    fig3.update_layout(height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown('<div class="sec-hdr">📉 Stockout Loss by Risk Class</div>', unsafe_allow_html=True)
        fig4 = px.violin(df, x="Inventory_Risk_Class", y="Loss_from_Stockout",
                         color="Inventory_Risk_Class", box=True,
                         color_discrete_map={"Low": "#2ec4b6", "Medium": "#ffd166", "High": "#e63946"},
                         category_orders={"Inventory_Risk_Class": ["Low", "Medium", "High"]})
        fig4.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col_r2:
        st.markdown('<div class="sec-hdr">⚡ High-Risk SKU Details</div>', unsafe_allow_html=True)
        hr = df[df["Inventory_Risk_Class"] == "High"][
            ["SKU_ID", "Product_Category", "Inventory_On_Hand",
             "Reorder_Point", "Stock_Cover_Days", "Stockout_History", "Loss_from_Stockout"]
        ].sort_values("Loss_from_Stockout", ascending=False).head(12)
        st.dataframe(hr, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4  —  CASH FLOW IMPACT
# ════════════════════════════════════════════════════════════════════════════
elif page == "💰  Cash Flow Impact":
    st.markdown('<div class="pg-title">💰 Cash Flow & Working Capital Impact</div>', unsafe_allow_html=True)
    st.markdown("**Model**: Random Forest Regressor · **Target**: `Cash_Stress_Score` (0–100)")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",      metrics["cash_stress"]["MAE"])
    c2.metric("RMSE",     metrics["cash_stress"]["RMSE"])
    c3.metric("R² Score", metrics["cash_stress"]["R2"])

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="sec-hdr">💸 Cash Stress by Category</div>', unsafe_allow_html=True)
        cd = df.groupby("Product_Category")["Cash_Stress_Score"].agg(["mean", "std"]).reset_index()
        cd.columns = ["Category", "Mean", "Std"]
        fig = px.bar(cd, x="Category", y="Mean", error_y="Std",
                     color="Mean", color_continuous_scale="Reds",
                     labels={"Mean": "Avg Cash Stress"})
        fig.add_hline(y=50, line_dash="dash", line_color="#f77f00",
                      annotation_text="Risk threshold 50")
        fig.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="sec-hdr">🔄 Cash Conversion Cycle by Category</div>', unsafe_allow_html=True)
        ccc = df.groupby("Product_Category")[["AR_Days", "Inventory_Days", "AP_Days"]].mean().reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="AR Days",       x=ccc["Product_Category"], y=ccc["AR_Days"],       marker_color="#e63946"))
        fig2.add_trace(go.Bar(name="Inventory Days", x=ccc["Product_Category"], y=ccc["Inventory_Days"], marker_color="#ffd166"))
        fig2.add_trace(go.Bar(name="AP Days (-))",   x=ccc["Product_Category"], y=-ccc["AP_Days"],       marker_color="#2ec4b6"))
        fig2.update_layout(barmode="relative", height=320, margin=dict(t=10, b=10),
                           legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="sec-hdr">🔗 Working Capital vs Cash Stress</div>', unsafe_allow_html=True)
        sp = df.sample(600, random_state=42)
        fig3 = px.scatter(sp, x="Working_Capital", y="Cash_Stress_Score",
                          color="Inventory_Risk_Class",
                          color_discrete_map={"Low": "#2ec4b6", "Medium": "#ffd166", "High": "#e63946"},
                          opacity=0.55,
                          labels={"Working_Capital": "Working Capital (৳)", "Cash_Stress_Score": "Cash Stress"})
        fig3.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.markdown('<div class="sec-hdr">📈 Financing Cost vs Cash Stress (trendline)</div>', unsafe_allow_html=True)
        fig4 = px.scatter(sp, x="Financing_Cost", y="Cash_Stress_Score",
                          color="Cash_Stress_Score", color_continuous_scale="Reds",
                          trendline="ols", opacity=0.55)
        fig4.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">💼 Financial Summary by Supplier</div>', unsafe_allow_html=True)
    fin = df.groupby("Supplier_ID").agg(
        Avg_PurchaseValue = ("Purchase_Value",      "mean"),
        Avg_WorkingCap    = ("Working_Capital",     "mean"),
        Avg_CashStress    = ("Cash_Stress_Score",   "mean"),
        Total_StockoutLoss= ("Loss_from_Stockout",  "sum"),
        Total_OverstockLoss=("Loss_from_Overstock", "sum"),
    ).round(2).reset_index().sort_values("Avg_CashStress", ascending=False).head(20)
    st.dataframe(fin, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5  —  SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍  SHAP Explainability":
    st.markdown('<div class="pg-title">🔍 Model Explainability — SHAP Analysis</div>', unsafe_allow_html=True)
    st.markdown("Understand *why* each prediction was made using SHAP (SHapley Additive exPlanations)")

    model_choice = st.selectbox("Select model to explain", [
        "📈 Demand Forecasting",
        "🚚 Supplier Delay",
        "📦 Inventory Risk",
        "💰 Cash Flow Stress",
    ])

    try:
        import shap
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

        model_map = {
            "📈 Demand Forecasting": models["demand"],
            "🚚 Supplier Delay":     models["supplier"],
            "📦 Inventory Risk":     models["inv_risk"],
            "💰 Cash Flow Stress":   models["cashflow"],
        }
        chosen = model_map[model_choice]
        X_shap = get_X(df.sample(250, random_state=42))

        with st.spinner("Computing SHAP values … (may take a few seconds)"):
            explainer   = shap.TreeExplainer(chosen)
            shap_values = explainer.shap_values(X_shap)

        # Handle multiclass (inventory risk)
        if isinstance(shap_values, list):
            sv = shap_values[-1]
            st.info("Multiclass model — showing SHAP for the **High-Risk** class")
        else:
            sv = shap_values

        col_desc = st.columns(1)[0]
        with col_desc:
            st.markdown('<div class="sec-hdr">📊 Global Feature Importance (mean |SHAP|)</div>', unsafe_allow_html=True)
            fig_bar, ax = plt.subplots(figsize=(9, 5))
            shap.summary_plot(sv, X_shap, plot_type="bar", show=False, color="#1e3a5f")
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close()

        st.markdown('<div class="sec-hdr">🐝 SHAP Beeswarm — Feature Impact Direction</div>', unsafe_allow_html=True)
        fig_bee, ax2 = plt.subplots(figsize=(9, 6))
        shap.summary_plot(sv, X_shap, show=False)
        plt.tight_layout()
        st.pyplot(fig_bee)
        plt.close()

        # Table
        st.markdown('<div class="sec-hdr">🏆 Top 15 Features by Mean |SHAP| Value</div>', unsafe_allow_html=True)
        feat_imp = pd.DataFrame({
            "Feature":      X_shap.columns,
            "Mean |SHAP|":  np.abs(sv).mean(axis=0),
        }).sort_values("Mean |SHAP|", ascending=False).head(15).reset_index(drop=True)
        feat_imp["Mean |SHAP|"] = feat_imp["Mean |SHAP|"].round(5)
        st.dataframe(feat_imp, use_container_width=True)

    except ImportError:
        st.warning("⚠️ SHAP not installed — showing built-in feature importance instead.")
        chosen_key = {
            "📈 Demand Forecasting": "demand",
            "🚚 Supplier Delay":     "supplier",
            "📦 Inventory Risk":     "inv_risk",
            "💰 Cash Flow Stress":   "cashflow",
        }[model_choice]
        chosen = models[chosen_key]
        X_fi = get_X(df.sample(100, random_state=1))
        imp_df = pd.DataFrame({
            "Feature":    X_fi.columns,
            "Importance": chosen.feature_importances_,
        }).sort_values("Importance", ascending=False).head(20)
        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues")
        fig.update_layout(height=560, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6  —  LIVE PREDICTION ENGINE
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚡  Live Prediction Engine":
    st.markdown('<div class="pg-title">⚡ Live Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown("Enter real values below — all 4 models run instantly in parallel.")

    st.markdown('<div class="sec-hdr">📝 Input Parameters</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 🏷️ Product & Context")
        product_cat  = st.selectbox("Product Category", CAT_COLS["Product_Category"])
        season       = st.selectbox("Season/Quarter",   CAT_COLS["Season"])
        region       = st.selectbox("Region",           CAT_COLS["Region"])
        promo        = st.selectbox("Promotion Active", [0, 1], format_func=lambda x: "Yes" if x else "No")
        hist_sales   = st.number_input("Historical Sales (units/month)", 10, 1000, 160)

    with col2:
        st.markdown("##### 🚚 Supplier & Inventory")
        sup_rating   = st.slider("Supplier Rating", 2.0, 5.0, 3.8, 0.1)
        lead_time    = st.number_input("Lead Time (days)", 3, 45, 14)
        delay_hist   = st.number_input("Delay History (last 12 months)", 0, 20, 3)
        inv_on_hand  = st.number_input("Inventory On Hand (units)", 0, 1000, 130)
        ord_qty      = st.number_input("Planned Order Quantity", 10, 2000, 210)
        dem_var      = st.slider("Demand Variability (CV)", 0.0, 1.0, 0.25, 0.01)

    with col3:
        st.markdown("##### 💰 Financials")
        unit_cost    = st.number_input("Unit Cost (৳)", 5.0, 500.0, 80.0, step=5.0)
        sell_price   = st.number_input("Selling Price (৳)", 10.0, 1200.0, 160.0, step=5.0)
        ap_days      = st.number_input("Accounts Payable Days", 10, 90, 30)
        ar_days      = st.number_input("Accounts Receivable Days", 7, 60, 25)
        inv_days     = st.number_input("Inventory Days", 10, 120, 45)
        cash_oh      = st.number_input("Cash On Hand (৳)", 5000.0, 500000.0, 25000.0, step=1000.0)
        fin_cost     = st.slider("Financing Cost Rate", 0.06, 0.20, 0.10, 0.01)

    # ── Derived features ─────────────────────────────────────────────────
    daily_demand    = hist_sales / 30.0
    reorder_pt      = int(daily_demand * lead_time * 1.2 + 15)
    stockout_h      = max(0, 2 if inv_on_hand < reorder_pt else 0)
    gross_margin    = (sell_price - unit_cost) / sell_price * 100
    purch_val       = unit_cost * ord_qty
    monthly_op      = 5000.0
    wk_cap          = cash_oh + inv_on_hand * unit_cost - purch_val * (ap_days / 30.0)
    stock_cover     = inv_on_hand / (daily_demand + 0.01)
    reorder_gap     = inv_on_hand - reorder_pt
    dsratio         = hist_sales / (inv_on_hand + 1)
    cc_ratio        = (2.5 * inv_on_hand) / (purch_val + 1)
    ccc             = ar_days + inv_days - ap_days

    input_dict = {
        "Historical_Sales": hist_sales,       "Promotion_Flag": promo,
        "Lead_Time_Days": lead_time,           "Supplier_Rating": sup_rating,
        "Delivery_Delay_History": delay_hist,  "Order_Quantity": ord_qty,
        "Inventory_On_Hand": inv_on_hand,      "Reorder_Point": reorder_pt,
        "Stockout_History": stockout_h,        "Holding_Cost": 2.5,
        "Transport_Cost": 25.0,                "Demand_Variability": dem_var,
        "Unit_Cost": unit_cost,                "Selling_Price": sell_price,
        "Gross_Margin": gross_margin,          "AP_Days": ap_days,
        "AR_Days": ar_days,                    "Inventory_Days": inv_days,
        "Cash_On_Hand": cash_oh,               "Purchase_Value": purch_val,
        "Monthly_Operating_Cost": monthly_op,  "Working_Capital": wk_cap,
        "Financing_Cost": fin_cost,            "Discount_Rate": 0.08,
        "Stock_Cover_Days": stock_cover,       "Reorder_Gap": reorder_gap,
        "Demand_Stock_Ratio": dsratio,         "Carrying_Cost_Ratio": cc_ratio,
        "Cash_Conversion_Cycle": ccc,
        "Product_Category": product_cat,       "Season": season,
        "Region": region,
    }

    st.markdown("---")

    if st.button("🚀 Run All 4 ML Predictions", type="primary", use_container_width=True):
        X_in = encode_single(input_dict)

        demand_pred   = float(models["demand"].predict(X_in)[0])
        delay_prob    = float(models["supplier"].predict_proba(X_in)[0][1])
        inv_risk_enc  = int(models["inv_risk"].predict(X_in)[0])
        inv_risk_cls  = models["inv_le"].inverse_transform([inv_risk_enc])[0]
        inv_risk_pbs  = models["inv_risk"].predict_proba(X_in)[0]
        cash_stress   = float(models["cashflow"].predict(X_in)[0])

        # Reorder recommendation (safety-stock method)
        safety_stock  = dem_var * daily_demand * np.sqrt(lead_time)
        rec_reorder   = max(0.0, daily_demand * lead_time + safety_stock - inv_on_hand)

        # ── Result cards ─────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">🎯 Prediction Results</div>', unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)

        delay_col = "#e63946" if delay_prob > 0.6 else "#f77f00" if delay_prob > 0.35 else "#2ec4b6"
        risk_col  = {"High": "#e63946", "Medium": "#ffd166", "Low": "#2ec4b6"}[inv_risk_cls]
        cash_col  = "#e63946" if cash_stress > 70 else "#f77f00" if cash_stress > 40 else "#2ec4b6"

        for col, val, lbl, sub in [
            (r1, f"{int(demand_pred)}", "📈 Predicted Demand", "units · next period"),
            (r2, f"{delay_prob*100:.1f}%", "🚚 Delay Probability",
             "⚠️ HIGH RISK" if delay_prob > 0.6 else "🟡 MODERATE" if delay_prob > 0.35 else "✅ LOW RISK"),
            (r3, inv_risk_cls, "📦 Inventory Risk", f"Low {inv_risk_pbs[1]:.0%} · Med {inv_risk_pbs[2]:.0%} · High {inv_risk_pbs[0]:.0%}"),
            (r4, f"{cash_stress:.1f}", "💰 Cash Stress Score",
             "🔴 Stressed" if cash_stress > 70 else "🟡 Moderate" if cash_stress > 40 else "🟢 Healthy"),
        ]:
            col.markdown(f"""
            <div class="big-pred">
                <div class="bl">{lbl}</div>
                <div class="bv">{val}</div>
                <div style="font-size:.75rem;color:#7eb8d4;margin-top:6px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Gauge charts ─────────────────────────────────────────────────
        g1, g2 = st.columns(2)
        with g1:
            fig_g1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(delay_prob * 100, 1),
                title={"text": "Supplier Delay Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": delay_col},
                    "steps": [
                        {"range": [0, 35],  "color": "#d4edda"},
                        {"range": [35, 60], "color": "#fff3cd"},
                        {"range": [60, 100],"color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": 60},
                },
            ))
            fig_g1.update_layout(height=280, margin=dict(t=50, b=10))
            st.plotly_chart(fig_g1, use_container_width=True)

        with g2:
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(cash_stress, 1),
                title={"text": "Cash Stress Score (0–100)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": cash_col},
                    "steps": [
                        {"range": [0, 40],  "color": "#d4edda"},
                        {"range": [40, 70], "color": "#fff3cd"},
                        {"range": [70, 100],"color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": 70},
                },
            ))
            fig_g2.update_layout(height=280, margin=dict(t=50, b=10))
            st.plotly_chart(fig_g2, use_container_width=True)

        st.markdown("---")

        # ── Smart recommendations ─────────────────────────────────────────
        st.markdown('<div class="sec-hdr">💡 Smart Recommendations</div>', unsafe_allow_html=True)
        rec1, rec2 = st.columns(2)

        with rec1:
            below_rp = inv_on_hand < reorder_pt
            st.markdown(f"""
            <div class="pred-box">
                <h4>📦 Inventory Action Plan</h4>
                <p><b>Recommended Reorder Qty:</b> {int(rec_reorder):,} units</p>
                <p><b>Current Stock Cover:</b> {stock_cover:.1f} days</p>
                <p><b>Reorder Point:</b> {reorder_pt:,} units</p>
                <p><b>Safety Stock:</b> {int(safety_stock):,} units</p>
                {'<p style="color:#e63946;font-weight:600;">⚠️ REORDER NOW — stock below reorder point!</p>'
                  if below_rp else
                '<p style="color:#2ec4b6;font-weight:600;">✅ Stock level is currently adequate.</p>'}
            </div>""", unsafe_allow_html=True)

        with rec2:
            hold_cost_est   = int(rec_reorder * 2.5)
            stockout_loss_e = int(max(0, demand_pred - inv_on_hand) * sell_price)
            cash_pct        = purch_val / (cash_oh + 1) * 100

            st.markdown(f"""
            <div class="pred-box">
                <h4>💰 Financial Impact Estimate</h4>
                <p><b>Holding Cost (reorder):</b> ৳{hold_cost_est:,}</p>
                <p><b>Potential Stockout Loss:</b> ৳{stockout_loss_e:,}</p>
                <p><b>Order Cash Impact:</b> {cash_pct:.1f}% of cash on hand</p>
                <p><b>Cash Conversion Cycle:</b> {int(ccc)} days</p>
                <p><b>Working Capital:</b> ৳{int(wk_cap):,}</p>
            </div>""", unsafe_allow_html=True)

        # ── Radar chart of risk dimensions ────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sec-hdr">🕸️ Multi-Dimensional Risk Radar</div>', unsafe_allow_html=True)

        radar_labels = ["Demand Risk", "Supplier Risk", "Inventory Risk",
                        "Cash Stress", "Lead Time Risk", "Variability Risk"]
        radar_vals = [
            min(100, abs(demand_pred - hist_sales) / (hist_sales + 1) * 200),
            delay_prob * 100,
            {"Low": 20, "Medium": 55, "High": 90}[inv_risk_cls],
            cash_stress,
            min(100, lead_time / 35 * 100),
            dem_var * 100,
        ]
        radar_vals_closed = radar_vals + [radar_vals[0]]
        angles = [i / len(radar_labels) * 360 for i in range(len(radar_labels))]
        angles_closed = angles + [angles[0]]

        fig_rad = go.Figure(go.Scatterpolar(
            r=radar_vals_closed,
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor="rgba(0,180,216,0.20)",
            line=dict(color="#00b4d8", width=2),
            name="Risk Profile",
        ))
        fig_rad.add_trace(go.Scatterpolar(
            r=[50] * (len(radar_labels) + 1),
            theta=radar_labels + [radar_labels[0]],
            line=dict(color="#f77f00", dash="dash", width=1),
            name="Threshold (50)",
        ))
        fig_rad.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=420, margin=dict(t=30, b=30),
            showlegend=True,
            legend=dict(orientation="h", y=-0.05),
        )
        st.plotly_chart(fig_rad, use_container_width=True)
