# ================================
# MenuProfit Pro (Improved Version)
# ================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog

st.set_page_config(page_title="MenuProfit Pro", layout="wide")

# -------------------------------
# DEFAULT DATA
# -------------------------------
DEFAULT = pd.DataFrame({
    "Dish Name": ["Burger","Pasta","Salad","Coffee"],
    "Category": ["Mains","Mains","Starters","Drinks"],
    "Ingredient Cost($)": [5,4,2,1],
    "Selling Price($)": [15,14,8,5],
    "Sold per Week": [100,80,120,200],
    "Cook Time (min)": [10,12,5,2],
    "Price Sensitivity": [-1.2,-1.0,-0.8,-0.5],
})

# -------------------------------
# SMART PRICE ROUNDING
# -------------------------------
def smart_price_round(price):
    if price < 10:
        return round(price * 2) / 2
    return round(price) - 0.01

# -------------------------------
# DEMAND MODEL (IMPROVED)
# -------------------------------
def demand_qty(new_price, base_price, base_qty, sensitivity):
    qty = base_qty * (new_price / base_price) ** sensitivity
    return np.clip(qty, base_qty*0.3, base_qty*1.5)

# -------------------------------
# PROFIT FUNCTIONS
# -------------------------------
def weekly_profit(price, row):
    qty = demand_qty(price, row["Selling Price($)"], row["Sold per Week"], row["Price Sensitivity"])
    return (price - row["Ingredient Cost($)"]) * qty

def total_profit(prices, rows, fixed):
    return sum(weekly_profit(prices[i], rows[i]) for i in range(len(rows))) - fixed

# -------------------------------
# OPTIMISER
# -------------------------------
def optimise(rows, fixed, mode):
    p0 = np.array([r["Selling Price($)"] for r in rows])
    bounds = [(r["Ingredient Cost($)"]*1.3, r["Selling Price($)"]*1.3) for r in rows]

    res1 = minimize(lambda p: -total_profit(p, rows, fixed), p0, bounds=bounds)

    if mode == "Fast":
        return res1.x

    res2 = differential_evolution(lambda p: -total_profit(p, rows, fixed), bounds)

    return res2.x if -res2.fun > -res1.fun else res1.x

# -------------------------------
# LP OPTIMISATION
# -------------------------------
def kitchen_lp(df, hours, budget):
    profit = (df["Selling Price($)"] - df["Ingredient Cost($)"]).values
    cook = df["Cook Time (min)"].values / 60
    cost = df["Ingredient Cost($)"].values

    return linprog(-profit,
                   A_ub=[cook, cost],
                   b_ub=[hours, budget],
                   bounds=[(0, q) for q in df["Sold per Week"]],
                   method="highs")

# -------------------------------
# AI INSIGHTS
# -------------------------------
def generate_insights(df, opt_prices):
    tips = []
    for i, row in df.iterrows():
        gain = weekly_profit(opt_prices[i], row) - weekly_profit(row["Selling Price($)"], row)

        if gain > 200:
            tips.append(f"🚀 Increase price of {row['Dish Name']} — strong profit upside")

        if row["Ingredient Cost($)"]/row["Selling Price($)"] > 0.4:
            tips.append(f"🔴 {row['Dish Name']} has high food cost — reduce cost or increase price")

    return tips

# -------------------------------
# UI
# -------------------------------
st.title("🍽️ MenuProfit Pro")

mode = st.radio("Optimisation Mode", ["Fast","Advanced"])
fixed = st.number_input("Fixed Costs ($/week)", value=2000)

menu = st.data_editor(DEFAULT, num_rows="dynamic")

if st.button("Analyse"):
    rows = menu.to_dict("records")

    # Optimisation
    opt_prices = optimise(rows, fixed, mode)
    opt_prices = [smart_price_round(p) for p in opt_prices]

    # LP
    lp = kitchen_lp(menu, 120, 3000)

    # Profit
    current = total_profit([r["Selling Price($)"] for r in rows], rows, fixed)
    new = total_profit(opt_prices, rows, fixed)

    st.subheader("💰 Profit")
    st.metric("Current", f"${current:,.0f}")
    st.metric("Optimised", f"${new:,.0f}")
    st.metric("Gain", f"${new-current:,.0f}")

    # Prices table
    st.subheader("💰 Suggested Prices")
    out = []
    for i, r in enumerate(rows):
        out.append({
            "Dish": r["Dish Name"],
            "Current": r["Selling Price($)"],
            "New": opt_prices[i]
        })
    st.dataframe(pd.DataFrame(out))

    # LP output
    if lp.success:
        st.subheader("🧠 Kitchen Focus")
        menu["Suggested Qty"] = lp.x.round(0)
        st.dataframe(menu[["Dish Name","Sold per Week","Suggested Qty"]])

    # Insights
    st.subheader("💡 Smart Insights")
    tips = generate_insights(menu, opt_prices)
    for t in tips:
        st.write(t)
