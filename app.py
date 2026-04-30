import streamlit as st
import pandas as pd
from scipy.optimize import linprog

# -------------------------
# PAGE SETUP
# -------------------------
st.set_page_config(page_title="Restaurant Profit Tool", layout="centered")

st.title("🍽️ Restaurant Profit Calculator")
st.write("Optimize your menu profit using real constraints.")

# -------------------------
# CSV UPLOAD
# -------------------------
st.subheader("📂 Upload Data (optional)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        "Item": ["Coffee", "Sandwich", "Burger"],
        "Price": [5.0, 8.0, 12.0],
        "Cost": [1.5, 3.0, 5.0],
        "Labour (hrs)": [0.05, 0.1, 0.2],
        "Max Demand": [200, 80, 50]
    })

# Editable table
df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# -------------------------
# SETTINGS
# -------------------------
st.subheader("⚙️ Settings")

col1, col2 = st.columns(2)

with col1:
    labour_hours = st.number_input("Total Labour Hours", min_value=0.0, value=16.0)

with col2:
    period = st.selectbox("Planning Period", ["Daily", "Weekly"])

# Adjust for weekly
if period == "Weekly":
    labour_hours *= 7
    df["Max Demand"] = df["Max Demand"] * 7

# -------------------------
# BUTTON
# -------------------------
if st.button("🚀 Calculate Optimal Plan"):

    # Validation
    required_cols = ["Price", "Cost", "Labour (hrs)", "Max Demand"]
    if not all(col in df.columns for col in required_cols):
        st.error("Missing required columns.")
        st.stop()

    if df[required_cols].isnull().values.any():
        st.error("Please fill all values.")
        st.stop()

    if (df[required_cols] < 0).any().any():
        st.error("Values cannot be negative.")
        st.stop()

    # -------------------------
    # CALCULATION
    # -------------------------
    df["Profit"] = df["Price"] - df["Cost"]

    c = -df["Profit"].values
    A = [df["Labour (hrs)"].values]
    b = [labour_hours]
    bounds = [(0, x) for x in df["Max Demand"]]

    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    # -------------------------
    # RESULTS
    # -------------------------
    if result.success:

        df["Optimal Qty"] = result.x.round(2)
        df["Revenue"] = (df["Optimal Qty"] * df["Price"]).round(2)
        df["Total Cost"] = (df["Optimal Qty"] * df["Cost"]).round(2)
        df["Total Profit"] = (df["Optimal Qty"] * df["Profit"]).round(2)

        total_profit = df["Total Profit"].sum()
        total_revenue = df["Revenue"].sum()
        total_cost = df["Total Cost"].sum()
        labour_used = (df["Labour (hrs)"] * df["Optimal Qty"]).sum()

        # -------------------------
        # KPIs
        # -------------------------
        st.subheader("📊 Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Profit", f"${total_profit:.2f}")
        c2.metric("📈 Revenue", f"${total_revenue:.2f}")
        c3.metric("💸 Cost", f"${total_cost:.2f}")

        st.write(f"🕒 Labour Used: {labour_used:.2f} / {labour_hours}")

        # -------------------------
        # TABLE
        # -------------------------
        st.subheader("📦 Optimal Plan")
        st.dataframe(df, use_container_width=True)

        # -------------------------
        # INSIGHTS
        # -------------------------
        st.subheader("🧠 Insights")

        if labour_used > 0.9 * labour_hours:
            st.warning("⚠️ Labour is limiting your profit.")
        else:
            st.success("✅ Labour capacity is sufficient.")

        best_item = df.loc[df["Profit"].idxmax(), "Item"]
        st.write(f"🔥 Focus on: {best_item}")

        low_profit_items = df[df["Profit"] < df["Profit"].mean()]["Item"].tolist()
        if low_profit_items:
            st.write(f"📉 Low profit items: {', '.join(low_profit_items)}")

        # -------------------------
        # DOWNLOAD
        # -------------------------
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

    else:
        st.error("Optimization failed. Check your inputs.")