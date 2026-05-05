import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import linprog

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Seralung Optimiz", layout="wide")

# -------------------------
# PREMIUM UI STYLE
# -------------------------
st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 20%, #111827, #020617);
    color: #E5E7EB;
}

/* Title */
h1 {
    font-weight: 700;
    letter-spacing: -1px;
}

/* Cards */
.card {
    background: rgba(17, 24, 39, 0.7);
    padding: 20px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 15px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #4F46E5);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
    font-weight: 600;
}

.stButton > button:hover {
    opacity: 0.9;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(17,24,39,0.7);
    padding: 15px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.title("Seralung Optimiz")
st.caption("AI-powered menu profit optimization for modern restaurants")

# -------------------------
# DATA
# -------------------------
df = pd.DataFrame({
    "Item": ["Burger", "Pasta", "Coffee", "Salad"],
    "Price": [12, 15, 5, 10],
    "Cost": [5, 7, 1.5, 4],
    "Labour (hrs)": [0.2, 0.25, 0.05, 0.1],
    "Max Demand": [80, 60, 200, 100]
})

st.markdown("### 📋 Menu Input")
df = st.data_editor(df, use_container_width=True)

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("Constraints")
labour = st.sidebar.number_input("Labour Hours", value=16.0)
budget = st.sidebar.number_input("Budget ($)", value=500.0)

# -------------------------
# RUN MODEL
# -------------------------
if st.button("Run Optimization"):

    df["Profit"] = df["Price"] - df["Cost"]

    c = -df["Profit"].values
    A = [df["Labour (hrs)"].values, df["Cost"].values]
    b = [labour, budget]
    bounds = [(0, x) for x in df["Max Demand"]]

    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    if result.success:

        df["Qty"] = result.x
        df["Total Profit"] = df["Qty"] * df["Profit"]
        df["Revenue"] = df["Qty"] * df["Price"]

        total_profit = df["Total Profit"].sum()
        total_revenue = df["Revenue"].sum()

        # -------------------------
        # KPI ROW
        # -------------------------
        st.markdown("### 📊 Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Profit", f"${total_profit:,.0f}")
        c2.metric("Revenue", f"${total_revenue:,.0f}")
        c3.metric("Items", len(df))

        # -------------------------
        # CHARTS
        # -------------------------
        st.markdown("### 📈 Performance")

        col1, col2 = st.columns(2)

        fig1 = px.bar(df, x="Item", y="Total Profit")
        fig2 = px.pie(df, names="Item", values="Revenue")

        for fig in [fig1, fig2]:
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)

        # -------------------------
        # INSIGHTS (CARD STYLE)
        # -------------------------
        st.markdown("### 🧠 Insights")

        best_item = df.loc[df["Total Profit"].idxmax(), "Item"]
        worst_item = df.loc[df["Total Profit"].idxmin(), "Item"]

        st.markdown(f"""
        <div class="card">
        <h4>Weekly Summary</h4>
        <p>💰 Profit: <b>${total_profit:,.0f}</b></p>
        <p>🔥 Best item: <b>{best_item}</b></p>
        <p>⚠️ Weak item: <b>{worst_item}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------
        # PRICING SUGGESTION
        # -------------------------
        st.markdown("### 💰 Pricing Strategy")

        df["Suggested Price"] = df["Price"] * 1.1
        best_price_item = df.loc[df["Suggested Price"].idxmax(), "Item"]

        st.markdown(f"""
        <div class="card">
        <p>👉 Increase price of <b>{best_price_item}</b> to improve profit</p>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------
        # TABLE
        # -------------------------
        st.markdown("### 📦 Detailed Data")
        st.dataframe(df, use_container_width=True)

    else:
        st.error("Optimization failed")
