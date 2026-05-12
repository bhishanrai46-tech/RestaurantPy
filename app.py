# ─────────────────────────────────────────────────────────────
# ADD THESE NEW COLUMNS TO DEFAULT DATA
# ─────────────────────────────────────────────────────────────

DEFAULT = {
    "Dish Name": ["Wagyu Burger","Truffle Pasta","Caesar Salad"],
    "Category": ["Mains","Mains","Starters"],

    # Existing
    "Ingredient Cost($)": [8.50,6.20,2.80],
    "Selling Price($)": [24.00,22.00,14.00],
    "Sold per Week": [120,85,200],
    "Cook Time (min)": [12,15,5],
    "Price Sensitivity": [-1.2,-1.0,-0.8],

    # NEW REALISTIC VARIABLES
    "Prep Time (min)": [8,10,2],
    "Wastage %": [8,12,5],
    "Labour Cost/hr": [30,30,30],
    "Current Stock": [120,80,300],
    "Reorder Point": [40,30,100],
    "Customer Rating": [4.7,4.4,4.2],
    "Bundle Attach Rate %": [35,22,18],
}

# ─────────────────────────────────────────────────────────────
# TRUE COST CALCULATION
# (ingredient + wastage + labour)
# ─────────────────────────────────────────────────────────────

def calculate_true_cost(row):

    ingredient_cost = row["Ingredient Cost($)"]

    wastage_pct = row["Wastage %"] / 100

    labour_rate = row["Labour Cost/hr"]

    prep_time = row["Prep Time (min)"]

    cook_time = row["Cook Time (min)"]

    # Add wastage impact
    real_food_cost = ingredient_cost * (1 + wastage_pct)

    # Labour cost
    total_minutes = prep_time + cook_time

    labour_cost = (total_minutes / 60) * labour_rate

    # Final true cost
    true_cost = real_food_cost + labour_cost

    return round(true_cost, 2)

# ─────────────────────────────────────────────────────────────
# PROFIT PER DISH
# ─────────────────────────────────────────────────────────────

def true_profit_per_dish(row):

    true_cost = calculate_true_cost(row)

    selling_price = row["Selling Price($)"]

    return round(selling_price - true_cost, 2)

# ─────────────────────────────────────────────────────────────
# PROFIT PER MINUTE
# VERY IMPORTANT KPI
# ─────────────────────────────────────────────────────────────

def profit_per_minute(row):

    profit = true_profit_per_dish(row)

    total_time = (
        row["Prep Time (min)"]
        + row["Cook Time (min)"]
    )

    if total_time <= 0:
        return 0

    return round(profit / total_time, 2)

# ─────────────────────────────────────────────────────────────
# INVENTORY ALERTS
# ─────────────────────────────────────────────────────────────

def inventory_alert(row):

    if row["Current Stock"] <= row["Reorder Point"]:
        return "⚠️ Reorder Needed"

    return "✅ Stock Healthy"

# ─────────────────────────────────────────────────────────────
# CUSTOMER RATING SCORE
# ─────────────────────────────────────────────────────────────

def rating_label(rating):

    if rating >= 4.5:
        return "⭐ Excellent"

    elif rating >= 4.0:
        return "👍 Good"

    elif rating >= 3.0:
        return "⚠️ Average"

    else:
        return "❌ Poor"

# ─────────────────────────────────────────────────────────────
# UPSELL / BUNDLE ESTIMATION
# ─────────────────────────────────────────────────────────────

def estimated_bundle_sales(row):

    base_orders = row["Sold per Week"]

    attach_rate = row["Bundle Attach Rate %"] / 100

    return round(base_orders * attach_rate)

# ─────────────────────────────────────────────────────────────
# MAIN ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────

def enhanced_analysis(df):

    df = df.copy()

    # TRUE COST
    df["True Cost($)"] = df.apply(
        calculate_true_cost,
        axis=1
    )

    # TRUE PROFIT
    df["True Profit/Dish($)"] = df.apply(
        true_profit_per_dish,
        axis=1
    )

    # PROFIT PER MINUTE
    df["Profit Per Minute($)"] = df.apply(
        profit_per_minute,
        axis=1
    )

    # WEEKLY TRUE PROFIT
    df["Weekly True Profit($)"] = (
        df["True Profit/Dish($)"]
        * df["Sold per Week"]
    )

    # INVENTORY STATUS
    df["Inventory Alert"] = df.apply(
        inventory_alert,
        axis=1
    )

    # CUSTOMER SCORE
    df["Rating Status"] = df["Customer Rating"].apply(
        rating_label
    )

    # ESTIMATED BUNDLE SALES
    df["Extra Bundle Orders"] = df.apply(
        estimated_bundle_sales,
        axis=1
    )

    # FOOD COST %
    df["True Food Cost %"] = round(
        (df["True Cost($)"] / df["Selling Price($)"]) * 100,
        1
    )

    return df

# ─────────────────────────────────────────────────────────────
# STREAMLIT DISPLAY
# ─────────────────────────────────────────────────────────────

st.markdown("## 🍽️ Advanced Restaurant Intelligence")

df = pd.DataFrame(DEFAULT)

results = enhanced_analysis(df)

# KPI CARDS
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Total Weekly Profit",
        f"${results['Weekly True Profit($)'].sum():,.0f}"
    )

with k2:
    st.metric(
        "Average Profit / Minute",
        f"${results['Profit Per Minute($)'].mean():.2f}"
    )

with k3:
    st.metric(
        "Average Food Cost %",
        f"{results['True Food Cost %'].mean():.1f}%"
    )

with k4:
    low_stock = (
        results["Inventory Alert"]
        == "⚠️ Reorder Needed"
    ).sum()

    st.metric(
        "Low Stock Items",
        low_stock
    )

# ─────────────────────────────────────────────────────────────
# DISPLAY TABLE
# ─────────────────────────────────────────────────────────────

st.dataframe(
    results[
        [
            "Dish Name",
            "Selling Price($)",
            "True Cost($)",
            "True Profit/Dish($)",
            "Profit Per Minute($)",
            "Weekly True Profit($)",
            "True Food Cost %",
            "Inventory Alert",
            "Rating Status",
            "Extra Bundle Orders",
        ]
    ],
    use_container_width=True
)

# ─────────────────────────────────────────────────────────────
# PROFIT PER MINUTE CHART
# ─────────────────────────────────────────────────────────────

fig = px.bar(
    results.sort_values(
        "Profit Per Minute($)",
        ascending=False
    ),
    x="Dish Name",
    y="Profit Per Minute($)",
    color="Profit Per Minute($)",
    title="Most Efficient Dishes"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ─────────────────────────────────────────────────────────────
# SMART AI INSIGHTS
# ─────────────────────────────────────────────────────────────

st.markdown("## 💡 Smart Insights")

for _, row in results.iterrows():

    if row["Profit Per Minute($)"] < 0.4:

        st.warning(
            f"{row['Dish Name']} is slow and not very profitable. "
            f"Consider simplifying prep."
        )

    if row["True Food Cost %"] > 35:

        st.error(
            f"{row['Dish Name']} has very high food cost "
            f"({row['True Food Cost %']}%)."
        )

    if row["Customer Rating"] >= 4.5:

        st.success(
            f"{row['Dish Name']} is highly rated. "
            f"Promote this item more aggressively."
        )

    if row["Inventory Alert"] == "⚠️ Reorder Needed":

        st.warning(
            f"{row['Dish Name']} inventory is running low."
        )
