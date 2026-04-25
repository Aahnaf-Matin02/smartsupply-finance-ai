"""
SmartSupply Finance AI — Streamlit Dashboard (Unilever Dataset)
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

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# ── Auto-setup ───────────────────────────────────────────────────────────────
if not os.path.exists("data/smartsupply_dataset.csv"):
    with st.spinner("⚙️ Processing Unilever dataset…"):
        from generate_data import process_dataset
        process_dataset()

if not os.path.exists("models/demand_model.pkl"):
    with st.spinner("🤖 Training models (~20 s)…"):
        from train_models import train_all_models
        train_all_models()
    st.rerun()

st.set_page_config(
    page_title="SmartSupply Finance AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}

#MainMenu,footer,header{visibility:hidden;}

/* ── Fixed topbar ── */
.topbar{
    position:fixed;top:0;left:0;right:0;height:54px;
    background:linear-gradient(90deg,#0b1a2e 0%,#0f2744 50%,#0b1a2e 100%);
    border-bottom:2px solid rgba(0,180,216,.4);
    display:flex;align-items:center;padding:0 20px;
    z-index:1000;box-shadow:0 2px 20px rgba(0,0,0,.4);gap:14px;
}
.topbar-logo{font-weight:700;font-size:1rem;color:white;white-space:nowrap;}
.topbar-logo span{color:#00d4ff;}
.topbar-divider{width:1px;height:26px;background:rgba(0,180,216,.35);margin:0 4px;}
.topbar-badge{margin-left:auto;background:rgba(0,180,216,.15);
    border:1px solid rgba(0,180,216,.3);border-radius:20px;
    padding:4px 12px;font-size:.68rem;color:#7eb8d4;
    white-space:nowrap;font-family:'Space Mono',monospace;}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{
    background:linear-gradient(160deg,#0b1a2e 0%,#0f2744 60%,#0b1a2e 100%) !important;
    top:54px !important;height:calc(100vh - 54px) !important;
}
section[data-testid="stSidebar"] *{color:#c9dff0 !important;}

/* ── Sidebar toggle arrow ── */
[data-testid="collapsedControl"]{
    top:62px !important;
    background:#0f2744 !important;
    border-radius:0 10px 10px 0 !important;
    width:26px !important;height:50px !important;
    display:flex !important;align-items:center !important;justify-content:center !important;
    box-shadow:3px 0 12px rgba(0,0,0,.35) !important;
    border:1px solid rgba(0,180,216,.45) !important;border-left:none !important;
    z-index:999 !important;opacity:1 !important;visibility:visible !important;
}
[data-testid="collapsedControl"] svg{color:#00d4ff !important;fill:#00d4ff !important;}

/* ── Main content push ── */
.main .block-container{padding:4.6rem 2rem 2rem !important;max-width:1400px;}

/* ── Nav buttons ── */
div.nav-row > div[data-testid="column"] > div[data-testid="stButton"] > button{
    width:100%;padding:5px 4px !important;border-radius:20px !important;
    font-size:.71rem !important;font-weight:500 !important;
    border:1px solid rgba(0,180,216,.2) !important;
    background:rgba(255,255,255,.04) !important;color:#4a7fa8 !important;
    transition:all .15s ease !important;white-space:nowrap !important;
}
div.nav-row > div[data-testid="column"] > div[data-testid="stButton"] > button:hover{
    background:rgba(0,180,216,.18) !important;color:#0f2744 !important;
    border-color:#00b4d8 !important;
}

/* ── KPI card ── */
.kpi-card{
    background:linear-gradient(135deg,#0f2744 0%,#1a3f70 100%);
    border:1px solid rgba(0,180,216,.25);border-radius:14px;
    padding:18px 20px 14px;color:white;margin-bottom:8px;
    box-shadow:0 6px 24px rgba(0,0,0,.25);transition:transform .15s;
}
.kpi-card:hover{transform:translateY(-2px);}
.kpi-card .kv{font-size:1.85rem;font-weight:700;color:#00d4ff;
               font-family:'Space Mono',monospace;line-height:1.1;}
.kpi-card .kl{font-size:.7rem;text-transform:uppercase;letter-spacing:1.5px;
               color:#7eb8d4;margin-top:5px;}
.kpi-card .ks{font-size:.75rem;color:#a0c4d8;margin-top:2px;}

/* ── Alerts ── */
.alert-red  {background:#e63946;border-radius:10px;padding:10px 15px;
              color:white;margin:5px 0;font-size:.83rem;}
.alert-amber{background:#f4a261;border-radius:10px;padding:10px 15px;
              color:#1a1a1a;margin:5px 0;font-size:.83rem;}
.alert-green{background:#2ec4b6;border-radius:10px;padding:10px 15px;
              color:#0a2a26;margin:5px 0;font-size:.83rem;}

/* ── Section header ── */
.sec-hdr{
    font-weight:600;font-size:.88rem;text-transform:uppercase;
    letter-spacing:1.2px;color:#0077b6;
    border-bottom:2px solid #00b4d8;padding-bottom:4px;margin:18px 0 12px;
}

/* ── Prediction cards ── */
.pred-box{
    background:white;border-radius:12px;padding:16px 20px;
    box-shadow:0 2px 14px rgba(0,0,0,.08);
    border-left:5px solid #00b4d8;margin:8px 0;
}
.pred-box h4{margin:0 0 8px;color:#0f2744;}
.big-pred{
    background:linear-gradient(135deg,#0f2744,#1a3f70);
    border-radius:16px;padding:20px 22px;color:white;
    text-align:center;margin-bottom:8px;
    box-shadow:0 8px 28px rgba(0,0,0,.3);
}
.big-pred .bv{font-size:2.2rem;font-weight:700;font-family:'Space Mono',monospace;}
.big-pred .bl{font-size:.7rem;text-transform:uppercase;letter-spacing:1.5px;
               color:#7eb8d4;margin-top:4px;}
.big-pred .bs{font-size:.75rem;color:#a0c8dc;margin-top:5px;}

/* ── Page title ── */
.pg-title{
    font-size:1.75rem;font-weight:700;color:#0f2744;
    border-bottom:3px solid #00b4d8;padding-bottom:8px;margin-bottom:18px;
}
.pg-sub{font-size:.85rem;color:#5a7fa8;margin-top:-12px;margin-bottom:16px;}
</style>
""", unsafe_allow_html=True)

# ── Load ─────────────────────────────────────────────────────────────────────
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

# ── Pages ────────────────────────────────────────────────────────────────────
PAGES = [
    "🏠  Overview Dashboard",
    "📈  Demand Forecasting",
    "🚚  Supplier Risk",
    "📦  Inventory Risk",
    "📊  Finance & SHAP",
    "⚡  Live Prediction Engine",
]
PAGES_SHORT = ["🏠 Overview","📈 Demand","🚚 Supplier","📦 Inventory","📊 Finance & SHAP","⚡ Predict"]

if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0

# ── Fixed top brand bar ───────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">🏭 <span>SmartSupply</span> Finance AI</div>
  <div class="topbar-divider"></div>
  <div style="font-size:.7rem;color:#7eb8d4;">Unilever SCM Dataset · ML-Powered</div>
  <div class="topbar-badge">800 rows · 4 models · Live</div>
</div>
""", unsafe_allow_html=True)

# ── Functional nav bar (st.button) ───────────────────────────────────────────
st.markdown('<div class="nav-row">', unsafe_allow_html=True)
nav_cols = st.columns(len(PAGES))
for i, (col, label) in enumerate(zip(nav_cols, PAGES_SHORT)):
    with col:
        btn_type = "primary" if st.session_state.page_idx == i else "secondary"
        if st.button(label, key=f"nav_{i}", type=btn_type, use_container_width=True):
            st.session_state.page_idx = i
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<hr style="margin:5px 0 14px;border:none;border-top:1px solid #e2e8f0;">', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 SmartSupply Finance AI")
    st.markdown("*Unilever Supply Chain Analytics*")
    st.markdown("---")
    sidebar_idx = st.radio("Navigate", options=list(range(len(PAGES))),
                            format_func=lambda i: PAGES[i],
                            index=st.session_state.page_idx,
                            key="sidebar_radio", label_visibility="collapsed")
    if sidebar_idx != st.session_state.page_idx:
        st.session_state.page_idx = sidebar_idx
        st.rerun()

    st.markdown("---")
    st.markdown("**📊 Dataset**")
    st.caption(f"Source: **Unilever SCM Analytics**")
    st.caption(f"Rows: **{len(df):,}**  ·  SKUs: **{df['SKU_ID'].nunique()}**")
    st.caption(f"Suppliers: **{df['Supplier_ID'].nunique()}**  ·  Products: **{df['Product_Type'].nunique()}**")

    st.markdown("**🤖 Model Performance**")
    st.caption(f"Demand R²: **{metrics['demand']['R2']}**")
    st.caption(f"Delay AUC: **{metrics['supplier_delay']['ROC_AUC']}**")
    st.caption(f"Inv Risk Acc: **{metrics['inventory_risk']['Accuracy']}**")
    st.caption(f"Cash R²: **{metrics['cash_stress']['R2']}**")
    st.markdown("---")
    st.caption("💡 Sidebar collapsed? Use the nav buttons above.")

page = PAGES[st.session_state.page_idx]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview Dashboard":
    st.markdown('<div class="pg-title">🏠 SmartSupply Finance AI — Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Unilever Supply Chain Analytics · Haircare · Skincare · Cosmetics</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    high_risk  = int((df["Inventory_Risk_Class"]=="High").sum())
    delay_rate = df["Delay_Flag"].mean()*100
    avg_stress = df["Cash_Stress_Score"].mean()
    total_rev  = df["Revenue"].sum()
    avg_defect = df["Defect_Rate"].mean()

    for col, val, lbl, sub in [
        (c1, df["SKU_ID"].nunique(),        "Unique SKUs",         "across all products"),
        (c2, f"{high_risk:,}",              "High-Risk SKUs",      f"{high_risk/len(df)*100:.1f}% of records"),
        (c3, f"{delay_rate:.1f}%",          "Supplier Delay Rate", "shipping + defect delays"),
        (c4, f"{avg_stress:.1f}/100",       "Avg Cash Stress",     "portfolio-wide score"),
        (c5, f"{avg_defect:.2f}%",          "Avg Defect Rate",     "quality indicator"),
    ]:
        col.markdown(f"""<div class="kpi-card">
            <div class="kv">{val}</div><div class="kl">{lbl}</div><div class="ks">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    ca, cb = st.columns([1,1.6])

    with ca:
        st.markdown('<div class="sec-hdr">🚨 Active Alerts</div>', unsafe_allow_html=True)
        hr = df[df["Inventory_Risk_Class"]=="High"]["SKU_ID"].value_counts().head(3)
        for sku, cnt in hr.items():
            st.markdown(f'<div class="alert-red">🔴 <b>{sku}</b> — High inventory risk ({cnt} records)</div>', unsafe_allow_html=True)
        risky = df[df["Delay_Flag"]==1]["Supplier_ID"].value_counts().head(2)
        for sup, cnt in risky.items():
            st.markdown(f'<div class="alert-amber">⚠️ <b>{sup}</b> — Delay flag raised ({cnt} times)</div>', unsafe_allow_html=True)
        hi_cash = int((df["Cash_Stress_Score"]>70).sum())
        if hi_cash:
            st.markdown(f'<div class="alert-amber">💸 <b>{hi_cash} records</b> — Cash stress > 70</div>', unsafe_allow_html=True)
        fail_insp = int((df["Inspection_Result"]=="Fail").sum())
        if fail_insp:
            st.markdown(f'<div class="alert-red">🔬 <b>{fail_insp} SKUs</b> — Failed quality inspection</div>', unsafe_allow_html=True)

    with cb:
        st.markdown('<div class="sec-hdr">📊 Inventory Risk Distribution</div>', unsafe_allow_html=True)
        rc = df["Inventory_Risk_Class"].value_counts().reset_index()
        rc.columns = ["Risk","Count"]
        fig = px.pie(rc, values="Count", names="Risk", hole=0.45,
                     color="Risk", color_discrete_map={"Low":"#2ec4b6","Medium":"#ffd166","High":"#e63946"})
        fig.update_layout(height=270, margin=dict(t=5,b=5,l=5,r=5), legend=dict(orientation="h",y=-0.08))
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-hdr">📦 Revenue by Product Type</div>', unsafe_allow_html=True)
        rd = df.groupby("Product_Type")["Revenue"].sum().reset_index()
        fig2 = px.bar(rd, x="Product_Type", y="Revenue", color="Product_Type",
                      color_discrete_map={"haircare":"#0077b6","skincare":"#00b4d8","cosmetics":"#90e0ef"})
        fig2.update_layout(height=290, margin=dict(t=5,b=5), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-hdr">🏭 Defect Rate by Supplier</div>', unsafe_allow_html=True)
        sd = df.groupby("Supplier_ID")["Defect_Rate"].mean().reset_index().sort_values("Defect_Rate")
        fig3 = px.bar(sd, x="Defect_Rate", y="Supplier_ID", orientation="h",
                      color="Defect_Rate", color_continuous_scale="RdYlGn_r")
        fig3.update_layout(height=290, margin=dict(t=5,b=5), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">📋 Dataset Sample</div>', unsafe_allow_html=True)
    show = ["SKU_ID","Product_Type","Supplier_ID","Units_Sold","Stock_Levels","Defect_Rate",
            "Inspection_Result","Delay_Flag","Inventory_Risk_Class","Cash_Stress_Score"]
    st.dataframe(df[show].head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DEMAND FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Demand Forecasting":
    st.markdown('<div class="pg-title">📈 Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">XGBoost Regressor · Target: Future_Sales</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("MAE",    metrics["demand"]["MAE"])
    c2.metric("RMSE",   metrics["demand"]["RMSE"])
    c3.metric("R²",     metrics["demand"]["R2"])
    c4.metric("MAPE",   f"{metrics['demand']['MAPE']}%")

    st.markdown("---")
    st.markdown('<div class="sec-hdr">🎯 Actual vs Predicted (test sample)</div>', unsafe_allow_html=True)
    samp    = df.sample(min(300, len(df)), random_state=7)
    y_pred  = models["demand"].predict(get_X(samp))
    fig = px.scatter(x=samp["Future_Sales"], y=y_pred, color=samp["Product_Type"],
                     opacity=0.65,
                     color_discrete_map={"haircare":"#0077b6","skincare":"#00b4d8","cosmetics":"#90e0ef"},
                     labels={"x":"Actual Future Sales","y":"Predicted","color":"Product"})
    mv = max(float(samp["Future_Sales"].max()), float(y_pred.max()))
    fig.add_shape(type="line",x0=0,y0=0,x1=mv,y1=mv,line=dict(color="#e63946",dash="dash",width=2))
    fig.update_layout(height=380, margin=dict(t=5,b=5))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-hdr">📦 Avg Sales by Product Type</div>', unsafe_allow_html=True)
        pd2 = df.groupby("Product_Type")[["Units_Sold","Future_Sales"]].mean().reset_index()
        fig2 = px.bar(pd2.melt("Product_Type"), x="Product_Type", y="value", color="variable", barmode="group",
                      color_discrete_map={"Units_Sold":"#1e3a5f","Future_Sales":"#00b4d8"})
        fig2.update_layout(height=290, margin=dict(t=5,b=5), legend=dict(orientation="h",y=-0.1))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-hdr">📈 Demand vs Availability</div>', unsafe_allow_html=True)
        fig3 = px.scatter(df.sample(200, random_state=1), x="Availability", y="Future_Sales",
                          color="Product_Type", trendline="ols", opacity=0.6,
                          color_discrete_map={"haircare":"#0077b6","skincare":"#00b4d8","cosmetics":"#90e0ef"})
        fig3.update_layout(height=290, margin=dict(t=5,b=5))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="sec-hdr">🏭 Forecast by Supplier</div>', unsafe_allow_html=True)
        sp = df.groupby("Supplier_ID")[["Units_Sold","Future_Sales"]].mean().reset_index()
        fig4 = px.bar(sp.melt("Supplier_ID"), x="Supplier_ID", y="value", color="variable", barmode="group",
                      color_discrete_map={"Units_Sold":"#1e3a5f","Future_Sales":"#00b4d8"})
        fig4.update_layout(height=280, margin=dict(t=5,b=5), legend=dict(orientation="h",y=-0.12))
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        st.markdown('<div class="sec-hdr">🔬 Defect Rate Impact on Demand</div>', unsafe_allow_html=True)
        fig5 = px.scatter(df.sample(200, random_state=2), x="Defect_Rate", y="Future_Sales",
                          color="Inventory_Risk_Class", trendline="ols", opacity=0.6,
                          color_discrete_map={"Low":"#2ec4b6","Medium":"#ffd166","High":"#e63946"})
        fig5.update_layout(height=280, margin=dict(t=5,b=5))
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SUPPLIER RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚚  Supplier Risk":
    st.markdown('<div class="pg-title">🚚 Supplier Delay Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">XGBoost Classifier · Target: Delay_Flag (0=On-time / 1=Delayed)</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  f"{metrics['supplier_delay']['Accuracy']*100:.1f}%")
    c2.metric("Precision", f"{metrics['supplier_delay']['Precision']*100:.1f}%")
    c3.metric("Recall",    f"{metrics['supplier_delay']['Recall']*100:.1f}%")
    c4.metric("F1 Score",  f"{metrics['supplier_delay']['F1']*100:.1f}%")
    c5.metric("ROC-AUC",   metrics["supplier_delay"]["ROC_AUC"])

    st.markdown("---")
    st.markdown('<div class="sec-hdr">🏭 Supplier Risk Leaderboard</div>', unsafe_allow_html=True)
    sup = df.groupby("Supplier_ID").agg(
        Delay_Rate       =("Delay_Flag",        "mean"),
        Avg_Defect       =("Defect_Rate",        "mean"),
        Avg_Ship_Time    =("Shipping_Time",      "mean"),
        Avg_Lead_Time    =("Supplier_Lead_Time", "mean"),
        Fail_Inspections =("Inspection_Result",  lambda x: (x=="Fail").sum()),
        Total_Txns       =("Units_Sold",         "count"),
    ).round(3).reset_index()
    sup["Risk_Score"] = (sup["Delay_Rate"]*40 + sup["Avg_Defect"]*15
                         + sup["Avg_Ship_Time"]*2 + sup["Fail_Inspections"]*5).round(1)
    sup = sup.sort_values("Risk_Score", ascending=False)
    sup["Risk"] = pd.cut(sup["Risk_Score"], bins=[-np.inf,20,40,np.inf],
                          labels=["🟢 Low","🟡 Medium","🔴 High"])
    st.dataframe(sup, use_container_width=True)

    st.markdown("---")
    cl, cr = st.columns(2)
    with cl:
        st.markdown('<div class="sec-hdr">⚡ Delay Rate by Carrier</div>', unsafe_allow_html=True)
        cd = df.groupby("Carrier")["Delay_Flag"].mean().reset_index()
        fig = px.bar(cd, x="Carrier", y="Delay_Flag", color="Delay_Flag",
                     color_continuous_scale="Reds",
                     labels={"Delay_Flag":"Delay Rate"})
        fig.update_layout(height=290, margin=dict(t=5,b=5), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<div class="sec-hdr">🚢 Delay by Transport Mode</div>', unsafe_allow_html=True)
        tm = df.groupby("Transport_Mode")["Delay_Flag"].mean().reset_index().sort_values("Delay_Flag")
        fig2 = px.bar(tm, x="Delay_Flag", y="Transport_Mode", orientation="h",
                      color="Delay_Flag", color_continuous_scale="RdYlGn_r",
                      labels={"Delay_Flag":"Delay Proportion"})
        fig2.update_layout(height=290, margin=dict(t=5,b=5), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    cl2, cr2 = st.columns(2)
    with cl2:
        st.markdown('<div class="sec-hdr">🔬 Defect Rate vs Shipping Time</div>', unsafe_allow_html=True)
        fig3 = px.scatter(df.sample(200,random_state=3), x="Defect_Rate", y="Shipping_Time",
                          color="Delay_Flag", symbol="Product_Type", opacity=0.65,
                          color_discrete_map={0:"#2ec4b6",1:"#e63946"},
                          labels={"Delay_Flag":"Delayed"})
        fig3.update_layout(height=270, margin=dict(t=5,b=5))
        st.plotly_chart(fig3, use_container_width=True)

    with cr2:
        st.markdown('<div class="sec-hdr">🗺️ Delay Heatmap: Supplier × Route</div>', unsafe_allow_html=True)
        heat = df.groupby(["Supplier_ID","Route"])["Delay_Flag"].mean().unstack()
        fig4 = px.imshow(heat, color_continuous_scale="RdYlGn_r", aspect="auto",
                         labels={"color":"Delay Rate"})
        fig4.update_layout(height=270, margin=dict(t=5,b=5))
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — INVENTORY RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦  Inventory Risk":
    st.markdown('<div class="pg-title">📦 Inventory Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Random Forest Classifier · Target: Inventory_Risk_Class (Low / Medium / High)</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    c1.metric("Accuracy",     f"{metrics['inventory_risk']['Accuracy']*100:.1f}%")
    c2.metric("F1 (Weighted)",f"{metrics['inventory_risk']['F1_Weighted']*100:.1f}%")

    st.markdown("---")
    cl, cr = st.columns(2)
    with cl:
        st.markdown('<div class="sec-hdr">📊 Risk by Product Type</div>', unsafe_allow_html=True)
        rc = df.groupby(["Product_Type","Inventory_Risk_Class"]).size().reset_index(name="Count")
        fig = px.bar(rc, x="Product_Type", y="Count", color="Inventory_Risk_Class", barmode="stack",
                     color_discrete_map={"Low":"#2ec4b6","Medium":"#ffd166","High":"#e63946"})
        fig.update_layout(height=310, margin=dict(t=5,b=5), legend=dict(orientation="h",y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<div class="sec-hdr">📦 Stock Cover Days by Risk Class</div>', unsafe_allow_html=True)
        fig2 = px.box(df, x="Inventory_Risk_Class", y="Stock_Cover_Days",
                      color="Inventory_Risk_Class",
                      color_discrete_map={"Low":"#2ec4b6","Medium":"#ffd166","High":"#e63946"},
                      category_orders={"Inventory_Risk_Class":["Low","Medium","High"]})
        fig2.update_layout(height=310, margin=dict(t=5,b=5), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    cl2, cr2 = st.columns(2)
    with cl2:
        st.markdown('<div class="sec-hdr">🗺️ Risk Heatmap: Supplier × Product</div>', unsafe_allow_html=True)
        df2 = df.copy()
        df2["Risk_N"] = df2["Inventory_Risk_Class"].map({"Low":1,"Medium":2,"High":3})
        heat = df2.groupby(["Supplier_ID","Product_Type"])["Risk_N"].mean().unstack()
        fig3 = px.imshow(heat, color_continuous_scale="RdYlGn_r", aspect="auto",
                         labels={"color":"Avg Risk (1=Low→3=High)"})
        fig3.update_layout(height=290, margin=dict(t=5,b=5))
        st.plotly_chart(fig3, use_container_width=True)

    with cr2:
        st.markdown('<div class="sec-hdr">🔬 Defect Rate by Risk Class</div>', unsafe_allow_html=True)
        fig4 = px.violin(df, x="Inventory_Risk_Class", y="Defect_Rate",
                         color="Inventory_Risk_Class", box=True,
                         color_discrete_map={"Low":"#2ec4b6","Medium":"#ffd166","High":"#e63946"},
                         category_orders={"Inventory_Risk_Class":["Low","Medium","High"]})
        fig4.update_layout(height=290, margin=dict(t=5,b=5), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">⚠️ High-Risk SKU Details</div>', unsafe_allow_html=True)
    hr = df[df["Inventory_Risk_Class"]=="High"][
        ["SKU_ID","Product_Type","Supplier_ID","Stock_Levels","Reorder_Point",
         "Stock_Cover_Days","Defect_Rate","Inspection_Result"]
    ].sort_values("Defect_Rate", ascending=False).head(15)
    st.dataframe(hr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FINANCE & SHAP (COMBINED)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Finance & SHAP":
    st.markdown('<div class="pg-title">📊 Finance Impact & SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Cash Stress Model (RF Regressor) + SHAP feature importance — combined view</div>', unsafe_allow_html=True)

    # ── Cash Stress Metrics ───────────────────────────────────────────────
    c1,c2,c3 = st.columns(3)
    c1.metric("Cash Stress MAE",  metrics["cash_stress"]["MAE"])
    c2.metric("Cash Stress RMSE", metrics["cash_stress"]["RMSE"])
    c3.metric("Cash Stress R²",   metrics["cash_stress"]["R2"])

    st.markdown("---")

    # ── Cash Stress Charts ────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">💸 Cash Stress Analysis</div>', unsafe_allow_html=True)
    cf1, cf2 = st.columns(2)

    with cf1:
        cd = df.groupby("Product_Type")["Cash_Stress_Score"].agg(["mean","std"]).reset_index()
        cd.columns = ["Product","Mean","Std"]
        fig = px.bar(cd, x="Product", y="Mean", error_y="Std",
                     color="Mean", color_continuous_scale="Reds",
                     labels={"Mean":"Avg Cash Stress"})
        fig.add_hline(y=50, line_dash="dash", line_color="#f77f00", annotation_text="Risk threshold")
        fig.update_layout(height=290, margin=dict(t=5,b=5), coloraxis_showscale=False,
                          title="Cash Stress by Product Type")
        st.plotly_chart(fig, use_container_width=True)

    with cf2:
        fig2 = px.histogram(df, x="Cash_Stress_Score", nbins=30,
                             color="Product_Type",
                             color_discrete_map={"haircare":"#0077b6","skincare":"#00b4d8","cosmetics":"#90e0ef"},
                             barmode="overlay", opacity=0.7)
        fig2.add_vline(x=df["Cash_Stress_Score"].mean(), line_dash="dash",
                       line_color="#e63946", annotation_text=f"Mean {df['Cash_Stress_Score'].mean():.1f}")
        fig2.update_layout(height=290, margin=dict(t=5,b=5), title="Cash Stress Distribution",
                           legend=dict(orientation="h",y=-0.12))
        st.plotly_chart(fig2, use_container_width=True)

    cf3, cf4 = st.columns(2)
    with cf3:
        samp2 = df.sample(min(300,len(df)), random_state=5)
        fig3 = px.scatter(samp2, x="Logistics_Cost", y="Cash_Stress_Score",
                          color="Inventory_Risk_Class", size="Defect_Rate",
                          color_discrete_map={"Low":"#2ec4b6","Medium":"#ffd166","High":"#e63946"},
                          opacity=0.6, title="Logistics Cost vs Cash Stress")
        fig3.update_layout(height=280, margin=dict(t=30,b=5))
        st.plotly_chart(fig3, use_container_width=True)

    with cf4:
        fig4 = px.scatter(samp2, x="Mfg_Cost", y="Cash_Stress_Score",
                          color="Product_Type", trendline="ols", opacity=0.6,
                          color_discrete_map={"haircare":"#0077b6","skincare":"#00b4d8","cosmetics":"#90e0ef"},
                          title="Manufacturing Cost vs Cash Stress")
        fig4.update_layout(height=280, margin=dict(t=30,b=5))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ── SHAP Section ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">🔍 SHAP Explainability</div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Select model to explain", [
        "💰 Cash Stress (Recommended)",
        "📈 Demand Forecasting",
        "🚚 Supplier Delay",
        "📦 Inventory Risk",
    ])

    model_map = {
        "💰 Cash Stress (Recommended)": models["cashflow"],
        "📈 Demand Forecasting":        models["demand"],
        "🚚 Supplier Delay":            models["supplier"],
        "📦 Inventory Risk":            models["inv_risk"],
    }
    chosen = model_map[model_choice]
    X_shap = get_X(df.sample(min(300, len(df)), random_state=42))

    try:
        import shap, matplotlib.pyplot as plt, matplotlib
        matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

        with st.spinner("Computing SHAP values…"):
            explainer   = shap.TreeExplainer(chosen)
            shap_values = explainer.shap_values(X_shap)

        if isinstance(shap_values, list):
            sv = shap_values[-1]
            st.info("Multiclass model — showing SHAP for **High-Risk** class")
        else:
            sv = shap_values

        sh1, sh2 = st.columns(2)

        with sh1:
            st.markdown("**📊 Global Feature Importance**")
            fig_b, ax = plt.subplots(figsize=(7,5))
            shap.summary_plot(sv, X_shap, plot_type="bar", show=False, color="#1e3a5f", max_display=12)
            plt.tight_layout()
            st.pyplot(fig_b)
            plt.close()

        with sh2:
            st.markdown("**🐝 SHAP Beeswarm — Impact Direction**")
            fig_bw, ax2 = plt.subplots(figsize=(7,5))
            shap.summary_plot(sv, X_shap, show=False, max_display=12)
            plt.tight_layout()
            st.pyplot(fig_bw)
            plt.close()

        st.markdown('<div class="sec-hdr">🏆 Top 12 Features by Mean |SHAP|</div>', unsafe_allow_html=True)
        fi = pd.DataFrame({
            "Feature":     X_shap.columns,
            "Mean |SHAP|": np.abs(sv).mean(axis=0),
        }).sort_values("Mean |SHAP|", ascending=False).head(12).reset_index(drop=True)
        fi["Mean |SHAP|"] = fi["Mean |SHAP|"].round(5)
        st.dataframe(fi, use_container_width=True)

    except ImportError:
        st.warning("SHAP not installed — showing built-in feature importance.")
        fi = pd.DataFrame({
            "Feature":    X_shap.columns,
            "Importance": chosen.feature_importances_,
        }).sort_values("Importance", ascending=False).head(15)
        fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Blues")
        fig_fi.update_layout(height=500, margin=dict(t=5,b=5), coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">💼 Cash Stress by Supplier (Financial Summary)</div>', unsafe_allow_html=True)
    fin = df.groupby("Supplier_ID").agg(
        Avg_Cash_Stress  =("Cash_Stress_Score",   "mean"),
        Avg_Logistics    =("Logistics_Cost",       "mean"),
        Avg_Mfg_Cost     =("Mfg_Cost",             "mean"),
        Avg_Revenue      =("Revenue",              "mean"),
        Avg_Defect       =("Defect_Rate",           "mean"),
    ).round(2).reset_index().sort_values("Avg_Cash_Stress", ascending=False)
    st.dataframe(fin, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LIVE PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡  Live Prediction Engine":
    st.markdown('<div class="pg-title">⚡ Live Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Enter any SKU values — all 4 models predict instantly in parallel using the Unilever dataset</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">📝 Input Parameters</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### 🏷️ Product Details")
        product_type    = st.selectbox("Product Type",    CAT_COLS["Product_Type"])
        customer_seg    = st.selectbox("Customer Segment",CAT_COLS["Customer_Segment"])
        carrier         = st.selectbox("Carrier",         CAT_COLS["Carrier"])
        transport_mode  = st.selectbox("Transport Mode",  CAT_COLS["Transport_Mode"])
        route           = st.selectbox("Route",           CAT_COLS["Route"])

    with c2:
        st.markdown("##### 🚚 Supply Chain")
        supplier_id     = st.selectbox("Supplier",        CAT_COLS["Supplier_ID"])
        insp_result     = st.selectbox("Inspection Result",CAT_COLS["Inspection_Result"])
        units_sold      = st.number_input("Units Sold (last period)", 100, 1500, 650)
        stock_levels    = st.number_input("Current Stock Levels",     0,   500,  80)
        order_qty       = st.number_input("Order Quantity",          10,  1000, 200)
        availability    = st.slider("Availability (%)",               1,   100,  55)
        defect_rate     = st.slider("Defect Rate (%)",           0.0,   5.0,  1.5, 0.1)

    with c3:
        st.markdown("##### 💰 Costs & Logistics")
        price           = st.number_input("Selling Price ($)",      2.0,  100.0,  50.0, 2.0)
        mfg_cost        = st.number_input("Manufacturing Cost ($)", 1.0,   80.0,  25.0, 1.0)
        shipping_cost   = st.number_input("Shipping Cost ($)",      1.0,   30.0,   8.0, 0.5)
        logistics_cost  = st.number_input("Logistics Cost ($)",    50.0, 1000.0, 450.0, 10.0)
        ship_time       = st.number_input("Shipping Time (days)",    1,    15,    5)
        ship_lead_time  = st.number_input("Shipping Lead Time (days)",1,   30,   10)
        supplier_lead   = st.number_input("Supplier Lead Time (days)",5,   40,   20)
        mfg_lead        = st.number_input("Mfg Lead Time (days)",    5,   40,   22)
        prod_volume     = st.number_input("Production Volume",       50, 1000,  400)

    # ── Derived features ──────────────────────────────────────────────────
    revenue          = price * units_sold
    gross_margin_pct = (price - mfg_cost) / price * 100
    revenue_per_unit = revenue / units_sold
    daily_demand     = units_sold / 30.0
    stock_cover_days = stock_levels / (daily_demand + 0.01)
    demand_stock_rat = units_sold / (stock_levels + 1)
    total_lead_time  = supplier_lead + mfg_lead
    cost_per_unit    = logistics_cost / (units_sold + 1)
    reorder_point    = int(daily_demand * total_lead_time * 1.2 + 5)
    reorder_gap      = stock_levels - reorder_point
    carry_cost_ratio = (mfg_cost * stock_levels) / (price * units_sold + 1)
    supply_eff       = prod_volume / (order_qty + 1)
    shipping_cost_r  = shipping_cost / (revenue + 1)

    input_dict = {
        "Price": price, "Availability": availability, "Units_Sold": units_sold,
        "Stock_Levels": stock_levels, "Shipping_Lead_Time": ship_lead_time,
        "Order_Quantity": order_qty, "Shipping_Time": ship_time,
        "Shipping_Cost": shipping_cost, "Supplier_Lead_Time": supplier_lead,
        "Production_Volume": prod_volume, "Mfg_Lead_Time": mfg_lead,
        "Mfg_Cost": mfg_cost, "Defect_Rate": defect_rate, "Logistics_Cost": logistics_cost,
        "Gross_Margin_Pct": gross_margin_pct, "Revenue_Per_Unit": revenue_per_unit,
        "Daily_Demand": daily_demand, "Stock_Cover_Days": stock_cover_days,
        "Demand_Stock_Ratio": demand_stock_rat, "Total_Lead_Time": total_lead_time,
        "Cost_Per_Unit": cost_per_unit, "Reorder_Point": reorder_point,
        "Reorder_Gap": reorder_gap, "Carrying_Cost_Ratio": carry_cost_ratio,
        "Supply_Efficiency": supply_eff, "Shipping_Cost_Ratio": shipping_cost_r,
        "Product_Type": product_type, "Customer_Segment": customer_seg,
        "Carrier": carrier, "Supplier_ID": supplier_id,
        "Inspection_Result": insp_result, "Transport_Mode": transport_mode, "Route": route,
    }

    st.markdown("---")
    if st.button("🚀 Run All 4 ML Predictions", type="primary", use_container_width=True):

        X_in = encode_single(input_dict)

        demand_pred  = float(models["demand"].predict(X_in)[0])
        delay_prob   = float(models["supplier"].predict_proba(X_in)[0][1])
        inv_enc      = int(models["inv_risk"].predict(X_in)[0])
        inv_cls      = models["inv_le"].inverse_transform([inv_enc])[0]
        inv_probs    = models["inv_risk"].predict_proba(X_in)[0]
        inv_classes  = models["inv_le"].classes_
        cash_stress  = float(models["cashflow"].predict(X_in)[0])
        cash_stress  = float(np.clip(cash_stress, 0, 100))

        # Safety stock & reorder recommendation
        safety_stock   = 0.2 * daily_demand * np.sqrt(total_lead_time)
        rec_reorder    = max(0.0, daily_demand * total_lead_time + safety_stock - stock_levels)
        holding_cost_e = rec_reorder * mfg_cost * 0.2
        stockout_loss  = max(0, demand_pred - stock_levels) * price
        cash_pct       = logistics_cost / (revenue + 1) * 100

        # ── Result cards ──────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">🎯 Prediction Results</div>', unsafe_allow_html=True)
        r1,r2,r3,r4 = st.columns(4)

        risk_col  = {"High":"#e63946","Medium":"#ffd166","Low":"#2ec4b6"}[inv_cls]
        delay_col = "#e63946" if delay_prob>0.6 else "#f77f00" if delay_prob>0.35 else "#2ec4b6"
        cash_col  = "#e63946" if cash_stress>70 else "#f77f00" if cash_stress>40 else "#2ec4b6"

        delay_txt = "🔴 HIGH RISK" if delay_prob>0.6 else "🟡 MODERATE" if delay_prob>0.35 else "✅ LOW RISK"
        cash_txt  = "🔴 Stressed" if cash_stress>70 else "🟡 Moderate" if cash_stress>40 else "🟢 Healthy"

        # Build inv prob string using class labels
        inv_prob_str = "  ".join([f"{c}: {p:.0%}" for c,p in zip(inv_classes, inv_probs)])

        for col, val, lbl, sub in [
            (r1, f"{int(demand_pred)}", "📈 Predicted Demand", "units · next period"),
            (r2, f"{delay_prob*100:.1f}%", "🚚 Delay Probability", delay_txt),
            (r3, inv_cls, "📦 Inventory Risk", inv_prob_str),
            (r4, f"{cash_stress:.1f}", "💰 Cash Stress", cash_txt),
        ]:
            col.markdown(f"""<div class="big-pred">
                <div class="bl">{lbl}</div>
                <div class="bv">{val}</div>
                <div class="bs">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Gauges ────────────────────────────────────────────────────────
        g1, g2 = st.columns(2)
        with g1:
            fig_g1 = go.Figure(go.Indicator(
                mode="gauge+number", value=round(delay_prob*100,1),
                title={"text":"Delay Probability (%)"},
                gauge={"axis":{"range":[0,100]},"bar":{"color":delay_col},
                       "steps":[{"range":[0,35],"color":"#d4edda"},
                                 {"range":[35,60],"color":"#fff3cd"},
                                 {"range":[60,100],"color":"#f8d7da"}],
                       "threshold":{"line":{"color":"red","width":3},"thickness":.75,"value":60}},
            ))
            fig_g1.update_layout(height=260, margin=dict(t=50,b=10))
            st.plotly_chart(fig_g1, use_container_width=True)

        with g2:
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number", value=round(cash_stress,1),
                title={"text":"Cash Stress Score (0–100)"},
                gauge={"axis":{"range":[0,100]},"bar":{"color":cash_col},
                       "steps":[{"range":[0,40],"color":"#d4edda"},
                                 {"range":[40,70],"color":"#fff3cd"},
                                 {"range":[70,100],"color":"#f8d7da"}],
                       "threshold":{"line":{"color":"red","width":3},"thickness":.75,"value":70}},
            ))
            fig_g2.update_layout(height=260, margin=dict(t=50,b=10))
            st.plotly_chart(fig_g2, use_container_width=True)

        st.markdown("---")

        # ── Recommendations ────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">💡 Smart Recommendations</div>', unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)

        below_rp = stock_levels < reorder_point
        with rc1:
            st.markdown(f"""<div class="pred-box">
                <h4>📦 Inventory Action Plan</h4>
                <p><b>Recommended Reorder Qty:</b> {int(rec_reorder):,} units</p>
                <p><b>Stock Cover Days:</b> {stock_cover_days:.1f} days</p>
                <p><b>Reorder Point:</b> {reorder_point:,} units</p>
                <p><b>Safety Stock Buffer:</b> {int(safety_stock):,} units</p>
                {'<p style="color:#e63946;font-weight:600;">⚠️ REORDER NOW — below reorder point!</p>'
                  if below_rp else
                '<p style="color:#2ec4b6;font-weight:600;">✅ Stock level adequate.</p>'}
            </div>""", unsafe_allow_html=True)

        with rc2:
            st.markdown(f"""<div class="pred-box">
                <h4>💰 Financial Impact Estimate</h4>
                <p><b>Est. Holding Cost:</b> ${holding_cost_e:,.0f}</p>
                <p><b>Potential Stockout Loss:</b> ${stockout_loss:,.0f}</p>
                <p><b>Logistics as % of Revenue:</b> {cash_pct:.1f}%</p>
                <p><b>Gross Margin:</b> {gross_margin_pct:.1f}%</p>
                <p><b>Supply Efficiency:</b> {supply_eff:.2f}x</p>
            </div>""", unsafe_allow_html=True)

        # ── Radar ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sec-hdr">🕸️ Multi-Dimensional Risk Radar</div>', unsafe_allow_html=True)

        radar_labels = ["Demand Risk","Delay Risk","Inventory Risk",
                        "Cash Stress","Defect Risk","Lead Time Risk"]
        radar_vals = [
            min(100, abs(demand_pred - units_sold) / (units_sold + 1) * 200),
            delay_prob * 100,
            {"Low":20,"Medium":55,"High":90}[inv_cls],
            cash_stress,
            min(100, defect_rate / 5 * 100),
            min(100, total_lead_time / 60 * 100),
        ]
        fig_r = go.Figure(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself", fillcolor="rgba(0,180,216,.18)",
            line=dict(color="#00b4d8", width=2), name="Risk Profile",
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=[50]*(len(radar_labels)+1), theta=radar_labels+[radar_labels[0]],
            line=dict(color="#f77f00",dash="dash",width=1), name="Threshold (50)",
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            height=420, margin=dict(t=30,b=30),
            legend=dict(orientation="h",y=-0.05),
        )
        st.plotly_chart(fig_r, use_container_width=True)
