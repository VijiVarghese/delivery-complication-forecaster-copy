"""
Delivery Complication Forecaster — Streamlit Dashboard
Healthcare Data Analytics & Ethics — Capstone Project 4
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Delivery Complication Forecaster",
    page_icon="⚠️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load / Train Model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("complication_model.pkl") and os.path.exists("feature_columns.json"):
        with open("complication_model.pkl","rb") as f:
            mdl = pickle.load(f)
        with open("feature_columns.json") as f:
            cols = json.load(f)
        return mdl, cols

    df = pd.read_csv("maternity_master.csv")
    df = df[(df["Age"]>=18)&(df["Age"]<=45)&(df["LOS"]>=2)].copy().reset_index(drop=True)
    X  = pd.get_dummies(df[["Age","DeliveryType","LaborDuration","Location"]], drop_first=True)
    y  = (df["Complications"]=="Yes").astype(int)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    mdl = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    mdl.fit(X_train, y_train)
    return mdl, list(X.columns)

model, feature_cols = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚠️ Delivery Complication Forecaster")
st.markdown(
    "Enter patient details to estimate **delivery complication risk**. "
    "Cesarean deliveries carry ~3× higher complication rates. "
    "This tool is for clinical decision support only."
)

# Quick stats banner
col_a, col_b, col_c = st.columns(3)
col_a.metric("Overall Complication Rate", "21.0%", "in training data")
col_b.metric("Cesarean Risk",  "39.3%", "+25.8% vs Vaginal")
col_c.metric("Vaginal Risk",   "13.3%", "baseline")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Input Panel
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Patient Details")
    age      = st.slider("Patient Age (years)", min_value=18, max_value=45, value=30)
    location = st.selectbox("Location", ["Urban", "Rural"])

with c2:
    st.subheader("Delivery Information")
    delivery  = st.selectbox("Planned Delivery Type", ["Vaginal", "Cesarean"])
    labor_hrs = st.slider("Expected Labor Duration (hours)", 1.0, 16.0, 8.0, 0.5)

with c3:
    st.subheader("Prediction")
    # Build input vector
    row = pd.DataFrame({
        "Age":                    [age],
        "LaborDuration":          [labor_hrs],
        "DeliveryType_Vaginal":   [1 if delivery  == "Vaginal" else 0],
        "Location_Urban":         [1 if location  == "Urban"   else 0],
    })[feature_cols]

    risk_score = model.predict_proba(row)[0][1]
    risk_pct   = round(risk_score * 100, 1)

    if risk_score < 0.20:
        risk_level = "LOW"
        st.success(f"### ✅ LOW RISK")
    elif risk_score < 0.40:
        risk_level = "MODERATE"
        st.warning(f"### ⚠️ MODERATE RISK")
    else:
        risk_level = "HIGH"
        st.error(f"### 🚨 HIGH RISK")

    st.metric("Complication Probability", f"{risk_pct}%")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Recommendation + Risk Gauge
# ─────────────────────────────────────────────────────────────────────────────
rec_col, gauge_col = st.columns([1, 2])

with rec_col:
    st.markdown("#### Clinical Recommendation")
    recs = {
        "LOW":      "Standard care protocol. Routine monitoring. No additional preparation required.",
        "MODERATE": "Inform patient of elevated risk. Keep anaesthesiology on standby. Review prior records.",
        "HIGH":     "Prepare additional surgical team. Alert ICU. Priority pre-delivery counselling recommended."
    }
    if risk_level == "HIGH":
        st.error(recs[risk_level])
    elif risk_level == "MODERATE":
        st.warning(recs[risk_level])
    else:
        st.info(recs[risk_level])

with gauge_col:
    fig, ax = plt.subplots(figsize=(7, 1.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("Complication Probability (%)", fontsize=10)
    ax.barh(0.5, 20, left=0,  height=0.6, color="#2ECC71", alpha=0.35)
    ax.barh(0.5, 20, left=20, height=0.6, color="#F39C12", alpha=0.35)
    ax.barh(0.5, 60, left=40, height=0.6, color="#E74C3C", alpha=0.35)
    ax.plot(risk_pct, 0.5, 'v', color='black', markersize=12, zorder=5)
    ax.text(risk_pct, 0.85, f"{risk_pct}%", ha='center', fontsize=9, fontweight='bold')
    ax.legend(handles=[
        mpatches.Patch(color="#2ECC71", alpha=0.5, label="Low (0–20%)"),
        mpatches.Patch(color="#F39C12", alpha=0.5, label="Moderate (20–40%)"),
        mpatches.Patch(color="#E74C3C", alpha=0.5, label="High (40–100%)"),
    ], loc="upper right", fontsize=8)
    ax.spines[['top','right','left']].set_visible(False)
    st.pyplot(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Risk Matrix by Age Group (from training data)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Complication Risk by Age & Delivery Type")

try:
    import seaborn as sns
    df_vis = pd.read_csv("maternity_master.csv")
    df_vis = df_vis[(df_vis["Age"]>=18)&(df_vis["Age"]<=45)].copy()
    df_vis["AgeGroup"] = pd.cut(df_vis["Age"], bins=[17,24,29,34,39,45],
                                labels=["18-24","25-29","30-34","35-39","40-45"])
    matrix = df_vis.groupby(["AgeGroup","DeliveryType"])["Complications"].apply(
        lambda x: (x=="Yes").sum()/len(x)*100
    ).unstack().round(1)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax2, cbar_kws={"label": "%"},
                annot_kws={"size": 11, "weight": "bold"})
    ax2.set_title("Complication Rate (%) by Age Group & Delivery Type", fontweight="bold")
    ax2.set_xlabel("Delivery Type"); ax2.set_ylabel("Age Group")
    st.pyplot(fig2, use_container_width=True)
except Exception:
    st.info("Upload maternity_master.csv to the app folder to see the risk matrix.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ This tool is for educational and decision-support purposes only. "
    "Predictions must be reviewed by a qualified clinician. "
    "Model: Gradient Boosting Classifier trained on 490 anonymised maternity records. "
    "Statistical validation: T-test p < 0.001, Chi-square p < 0.001."
)
