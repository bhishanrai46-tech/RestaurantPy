"""
MenuProfit Pro — Restaurant Pricing Intelligence Platform
=========================================================
Run:  pip install streamlit pandas numpy plotly scipy openpyxl
      streamlit run menuprofit_app.py
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import linprog, minimize, differential_evolution

st.set_page_config(
    page_title="MenuProfit Pro",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stAppViewContainer"] { background: #0a0a0f; color: #e2e4ed; }
[data-testid="stSidebar"] { background: #0d0d17 !important; border-right: 1px solid rgba(255,255,255,0.06); }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; letter-spacing: -0.5px; }
.section-title { font-family: 'DM Serif Display', serif; font-size: 1.6rem; color: #f0c060; margin: 2rem 0 0.5rem 0; padding-bottom: 6px; border-bottom: 1px solid rgba(240,192,96,0.2); }
.kpi-card { background: linear-gradient(135deg, rgba(15,15,30,0.9), rgba(20,20,40,0.7)); border: 1px solid rgba(240,192,96,0.15); border-radius: 14px; padding: 20px 22px; text-align: center; height: 100%; }
.kpi-label { font-size: 0.78rem; color: #8b8fa8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #f0c060; line-height: 1.1; }
.kpi-sub   { font-size: 0.82rem; color: #6b7280; margin-top: 4px; }
.insight-card { background: rgba(15,15,28,0.8); border-left: 3px solid #f0c060; border-radius: 0 10px 10px 0; padding: 14px 18px; margin-bottom: 10px; font-size: 0.9rem; }
.insight-card.warn   { border-color: #f59e0b; }
.insight-card.danger { border-color: #ef4444; }
.insight-card.good   { border-color: #22c55e; }
.rec-card { background: rgba(17,17,34,0.9); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 16px 20px; margin-bottom: 10px; font-size: 0.88rem; line-height: 1.6; }
[data-testid="stMetric"] { background: rgba(15,15,28,0.7); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); padding: 14px !important; }
.stTabs [data-baseweb="tab-list"] { background: rgba(0,0,0,0.3); border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8b8fa8 !important; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background: rgba(240,192,96,0.15) !important; color: #f0c060 !important; }
.stButton > button { background: linear-gradient(135deg, #d97706, #f59e0b); color: #0a0a0f; border: none; border-radius: 10px; padding: 10px 28px; font-weight: 700; font-size: 0.95rem; width: 100%; }
</style>
""", unsafe_allow_html=True)

MIN_MARGIN        = 0.28
MAX_PRICE_BUMP    = 0.35
SENSITIVITY_STEPS = (-20, -15, -10, -5, 0, +5, +10, +15, +20)
QUAD_COLOR  = {"Star": "#f0c060", "Puzzle": "#60a5fa", "Plow Horse": "#34d399", "Dog": "#f87171"}
QUAD_ICON   = {"Star": "⭐", "Puzzle": "🧩", "Plow Horse": "🐎", "Dog": "🐕"}
QUAD_ACTION = {
    "Star":       "PROTECT — Feature prominently, lock in quality. Never compromise on this item.",
    "Puzzle":     "PROMOTE — Upsell via staff training, add to daily specials. Margin is there, just needs volume.",
    "Plow Horse": "REPRICE or STREAMLINE — Either raise price carefully or reduce ingredient cost.",
    "Dog":        "REVIEW — Bundle with a Star, promote as limited-time, or consider removing.",
}
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#c9cde0"),
    margin=dict(l=20, r=20, t=40, b=20),
)

DEFAULT_DATA = {
    "Item":         ["Wagyu Burger", "Truffle Pasta", "Caesar Salad", "Espresso Martini",
                     "Lava Cake",   "House Wine",    "Grilled Salmon", "Garlic Bread"],
    "Category":     ["Mains", "Mains", "Starters", "Drinks", "Desserts", "Drinks", "Mains", "Starters"],
    "Cost ($)":     [8.50, 6.20, 2.80, 1.90, 2.40, 3.20, 9.80, 0.80],
    "Price ($)":    [24.00, 22.00, 14.00, 16.00, 12.00, 11.00, 28.00, 7.00],
    "Wkly Qty":     [120, 85, 200, 180, 90, 250, 70, 160],
    "Labour (min)": [12, 15, 5, 3, 8, 2, 18, 4],
    "Elasticity":   [-1.2, -1.0, -0.8, -0.6, -1.1, -0.9, -1.3, -0.7],
}

def demand(p, p0, q0, eps):
    if p <= 0: return 0.0
    return float(q0) * (p / float(p0)) ** float(eps)

def item_cm(p, item):
    q = demand(p, item["Price ($)"], item["Wkly Qty"], item["Elasticity"])
    return (p - item["Cost ($)"]) * q

def total_profit(prices, items, fixed):
    return sum(item_cm(prices[i], items[i]) for i in range(len(items))) - fixed

def run_lp(df, labour_hrs, budget_usd):
    profits  = (df["Price ($)"] - df["Cost ($)"]).values
    labour_h = df["Labour (min)"].values / 60.0
    costs    = df["Cost ($)"].values
    qtys     = df["Wkly Qty"].values
    return linprog(-profits, A_ub=[labour_h, costs], b_ub=[labour_hrs, budget_usd],
                   bounds=[(0, q) for q in qtys], method="highs")

def run_nlp(items, fixed):
    p0     = np.array([i["Price ($)"] for i in items])
    bounds = [(i["Cost ($)"] * (1 + MIN_MARGIN), i["Price ($)"] * (1 + MAX_PRICE_BUMP)) for i in items]
    r1 = minimize(lambda p: -total_profit(p, items, fixed), p0, bounds=bounds,
                  method="L-BFGS-B", options={"ftol": 1e-12, "maxiter": 2000})
    r2 = differential_evolution(lambda p: -total_profit(p, items, fixed),
                                bounds, seed=42, maxiter=300, popsize=12, tol=1e-9)
    if -r2.fun >= -r1.fun: return r2.x, -r2.fun
    return r1.x, -r1.fun

def sensitivity(item, steps=SENSITIVITY_STEPS):
    rows = []
    p0   = item["Price ($)"]
    base = item_cm(p0, item)
    for pct in steps:
        p  = p0 * (1 + pct / 100)
        q  = demand(p, p0, item["Wkly Qty"], item["Elasticity"])
        cm = (p - item["Cost ($)"]) * q
        rows.append({"Delta%": pct, "Price": round(p, 2), "Qty": round(q, 1),
                     "Revenue": round(p * q, 2), "Wkly CM": round(cm, 2),
                     "Delta CM": round(cm - base, 2),
                     "Margin%": round((p - item["Cost ($)"]) / p * 100, 1) if p > 0 else 0})
    return rows

def menu_engineering(df):
    df = df.copy()
    df["CM"]  = df["Price ($)"] - df["Cost ($)"]
    df["CM%"] = df["CM"] / df["Price ($)"] * 100
    avg_cm  = df["CM"].mean()
    avg_qty = df["Wkly Qty"].mean()
    def quad(row):
        if row["CM"] >= avg_cm and row["Wkly Qty"] >= avg_qty: return "Star"
        if row["CM"] >= avg_cm and row["Wkly Qty"] <  avg_qty: return "Puzzle"
        if row["CM"] <  avg_cm and row["Wkly Qty"] >= avg_qty: return "Plow Horse"
        return "Dog"
    df["Quadrant"] = df.apply(quad, axis=1)
    return df, avg_cm, avg_qty

def breakeven(df, fixed):
    total_cm     = ((df["Price ($)"] - df["Cost ($)"]) * df["Wkly Qty"]).sum()
    total_rev    = (df["Price ($)"] * df["Wkly Qty"]).sum()
    total_covers = df["Wkly Qty"].sum()
    net_profit   = total_cm - fixed
    avg_cm_cover = total_cm / total_covers if total_covers else 0
    be_covers    = fixed / avg_cm_cover if avg_cm_cover > 0 else float("inf")
    safety       = net_profit / total_cm * 100 if total_cm > 0 else 0
    fc_ratio     = (df["Cost ($)"] * df["Wkly Qty"]).sum() / total_rev * 100 if total_rev > 0 else 0
    return dict(total_cm=total_cm, total_rev=total_rev, net_profit=net_profit,
                monthly_profit=net_profit * 4.33, annual_profit=net_profit * 52,
                total_covers=total_covers, avg_cm_cover=avg_cm_cover,
                be_covers=be_covers, surplus=total_covers - be_covers,
                safety_pct=safety, fc_ratio=fc_ratio, fixed=fixed)

def build_recommendations(eng_df, be, opt_prices):
    recs = []
    emoji = "🟢" if be["safety_pct"] > 25 else ("🟡" if be["safety_pct"] > 10 else "🔴")
    label = "Healthy" if be["safety_pct"] > 25 else ("Caution" if be["safety_pct"] > 10 else "Critical")
    recs.append(("info", f"**Safety Margin: {be['safety_pct']:.1f}%** — {emoji} {label}. "
                 f"Need {be['be_covers']:.0f} covers/week to break even; currently {be['total_covers']:.0f}."))
    if be["fc_ratio"] > 35:
        recs.append(("warn", f"**Food Cost Ratio is {be['fc_ratio']:.1f}%** — above 35% danger zone. "
                     f"Target is 32%. Renegotiate supplier contracts or trim portions."))
    for i, row in eng_df.iterrows():
        p_opt  = opt_prices[i] if opt_prices is not None else row["Price ($)"]
        chg    = (p_opt - row["Price ($)"]) / row["Price ($)"] * 100
        quad   = row["Quadrant"]
        fc_pct = row["Cost ($)"] / row["Price ($)"] * 100
        if quad == "Dog":
            recs.append(("danger", f"**{row['Item']} ({QUAD_ICON[quad]} Dog)** — Low margin and low volume. "
                         f"Bundle with a Star, run limited-time promotion, or remove from menu."))
        elif quad == "Plow Horse" and chg > 3:
            recs.append(("warn", f"**{row['Item']} ({QUAD_ICON[quad]} Plow Horse)** — High volume but thin margin ({row['CM%']:.0f}%). "
                         f"Suggested price: **${p_opt:.2f}** (+{chg:.1f}%). Elasticity (e={row['Elasticity']}) can absorb this."))
        elif quad == "Puzzle":
            recs.append(("warn", f"**{row['Item']} ({QUAD_ICON[quad]} Puzzle)** — Great margin ({row['CM%']:.0f}%) but low volume. "
                         f"Add to specials board, train staff to upsell."))
        elif quad == "Star":
            recs.append(("good", f"**{row['Item']} ({QUAD_ICON[quad]} Star)** — Top performer. Protect quality. "
                         f"Consider a premium variant at a higher price point."))
        if fc_pct > 38:
            recs.append(("danger", f"**{row['Item']}** — Food cost is {fc_pct:.0f}% of price (target 32%). "
                         f"Raise price to ${row['Cost ($)'] / 0.32:.2f} or reduce ingredient cost."))
    return recs

def gauge_chart(value, title, min_val=0, max_val=100, danger=10, caution=25):
    color = "#22c55e" if value > caution else ("#f59e0b" if value > danger else "#ef4444")
    fig   = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"size": 13, "color": "#c9cde0"}},
        number={"suffix": "%", "font": {"size": 26, "color": color}},
        gauge=dict(axis=dict(range=[min_val, max_val], tickcolor="#444"),
                   bar=dict(color=color, thickness=0.25), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[{"range": [0, danger], "color": "rgba(239,68,68,0.15)"},
                           {"range": [danger, caution], "color": "rgba(245,158,11,0.12)"},
                           {"range": [caution, max_val], "color": "rgba(34,197,94,0.08)"}],
                   threshold=dict(line=dict(color=color, width=2), thickness=0.75, value=value))))
    fig.update_layout(**PLOTLY_DARK, height=200)
    return fig

def waterfall_chart(df, opt_prices):
    items    = df.to_dict("records")
    curr_cms = [(r["Price ($)"] - r["Cost ($)"]) * r["Wkly Qty"] for r in items]
    opt_cms  = [item_cm(opt_prices[i], items[i]) for i in range(len(items))]
    deltas   = [opt_cms[i] - curr_cms[i] for i in range(len(items))]
    fig = go.Figure(go.Waterfall(
        name="CM Impact", orientation="v", x=[r["Item"] for r in items], y=deltas,
        connector={"line": {"color": "rgba(255,255,255,0.1)"}},
        increasing={"marker": {"color": "#22c55e"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#f0c060"}}))
    fig.update_layout(**PLOTLY_DARK, title="Weekly CM Change: Current to Optimised ($)",
                      height=320, xaxis_tickangle=-30)
    return fig

def bcg_scatter(eng_df, avg_cm, avg_qty):
    fig = go.Figure()
    for quad, grp in eng_df.groupby("Quadrant"):
        fig.add_trace(go.Scatter(x=grp["Wkly Qty"], y=grp["CM"], mode="markers+text",
                                  name=f"{QUAD_ICON[quad]} {quad}", text=grp["Item"],
                                  textposition="top center", textfont=dict(size=10, color="#c9cde0"),
                                  marker=dict(size=16, color=QUAD_COLOR[quad], opacity=0.85,
                                              line=dict(width=1, color="rgba(255,255,255,0.2)"))))
    fig.add_hline(y=avg_cm,  line_dash="dash", line_color="rgba(255,255,255,0.18)", line_width=1)
    fig.add_vline(x=avg_qty, line_dash="dash", line_color="rgba(255,255,255,0.18)", line_width=1)
    x_min = eng_df["Wkly Qty"].min() * 0.85; x_max = eng_df["Wkly Qty"].max() * 1.05
    y_min = eng_df["CM"].min() * 0.85;        y_max = eng_df["CM"].max() * 1.05
    for label, ax, ay, xa, ya in [("STAR", x_max, y_max, "right", "top"),
                                    ("PUZZLE", x_min, y_max, "left", "top"),
                                    ("PLOW HORSE", x_max, y_min, "right", "bottom"),
                                    ("DOG", x_min, y_min, "left", "bottom")]:
        fig.add_annotation(x=ax, y=ay, text=label, showarrow=False,
                           font=dict(size=9, color="rgba(255,255,255,0.22)"),
                           xanchor=xa, yanchor=ya)
    fig.update_layout(**PLOTLY_DARK, height=420,
                      title="Menu Engineering Matrix — Popularity vs Contribution Margin",
                      xaxis_title="Weekly Quantity (Popularity)",
                      yaxis_title="Contribution Margin / Unit ($)")
    return fig

def sensitivity_heatmap(df_items):
    items  = df_items.to_dict("records")
    matrix = [[s["Delta CM"] for s in sensitivity(item)] for item in items]
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[f"{s:+d}%" for s in SENSITIVITY_STEPS],
        y=[i["Item"] for i in items],
        colorscale=[[0, "#ef4444"], [0.5, "#1f2937"], [1, "#22c55e"]],
        colorbar=dict(
            title=dict(text="Delta CM $", font=dict(color="#c9cde0")),
            tickfont=dict(color="#c9cde0"),
        ),
        text=[[f"${v:+,.0f}" for v in row] for row in matrix],
        texttemplate="%{text}", textfont=dict(size=9), zmid=0))
    fig.update_layout(**PLOTLY_DARK, height=320,
                      title="Sensitivity Heatmap — Delta Weekly CM per Item vs Price Change")
    return fig

def tornado_chart(df_items):
    items = df_items.to_dict("records")
    data  = []
    for item in items:
        sens = sensitivity(item)
        low  = next(s["Delta CM"] for s in sens if s["Delta%"] == -20)
        high = next(s["Delta CM"] for s in sens if s["Delta%"] == +20)
        data.append({"Item": item["Item"], "Low": low, "High": high, "Swing": high - low})
    data = sorted(data, key=lambda x: x["Swing"])
    fig  = go.Figure()
    for d in data:
        fig.add_trace(go.Bar(y=[d["Item"]], x=[d["Low"]],  orientation="h",
                             marker_color="#ef4444", showlegend=False))
        fig.add_trace(go.Bar(y=[d["Item"]], x=[d["High"]], orientation="h",
                             marker_color="#22c55e", showlegend=False))
    fig.update_layout(**PLOTLY_DARK, barmode="relative",
                      title="Tornado Chart — CM Swing Range (-20% to +20% price)",
                      height=320, xaxis_title="Delta Weekly CM ($)")
    return fig

def category_chart(eng_df):
    records = []
    for cat in eng_df["Category"].unique():
        sub = eng_df[eng_df["Category"] == cat]
        records.append({"Category": cat,
                         "Revenue": (sub["Price ($)"] * sub["Wkly Qty"]).sum(),
                         "CM":      (sub["CM"]         * sub["Wkly Qty"]).sum()})
    cat_df = pd.DataFrame(records)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cat_df["Category"], y=cat_df["Revenue"], name="Revenue",
                         marker_color="#60a5fa", opacity=0.75))
    fig.add_trace(go.Bar(x=cat_df["Category"], y=cat_df["CM"], name="Contribution Margin",
                         marker_color="#f0c060", opacity=0.9))
    fig.update_layout(**PLOTLY_DARK, barmode="group", height=320,
                      title="Revenue & CM by Category (weekly)", yaxis_title="$")
    return fig

# Session state
for key in ["opt_prices", "eng_df", "be", "recs"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "recs" else []

# Sidebar
with st.sidebar:
    st.markdown("## Business Constraints")
    weekly_fixed  = st.number_input("Weekly Fixed Costs ($)", value=3500, step=100)
    labour_hrs    = st.number_input("Total Labour Hours / Week", value=120, step=5)
    weekly_budget = st.number_input("Ingredient Budget / Week ($)", value=4000, step=100)
    st.divider()
    st.markdown("## Pricing Guardrails")
    min_margin_pct = st.slider("Minimum Gross Margin %", 20, 60, 30)
    max_bump_pct   = st.slider("Max Price Increase % (cap)", 5, 50, 35)
    st.divider()
    st.markdown("## Import / Export")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            imported_df = pd.read_csv(uploaded)
            missing = [c for c in DEFAULT_DATA.keys() if c not in imported_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}"); uploaded = None
            else:
                st.success("CSV loaded")
        except Exception as e:
            st.error(f"Error: {e}"); uploaded = None
    st.divider()
    st.caption("Elasticity guide:\n-0.5 = inelastic (luxury)\n-1.0 = unit elastic\n-1.5 = elastic")

# Header
st.markdown("""
<div style="padding:10px 0 24px 0;">
  <div style="font-size:0.8rem;letter-spacing:0.15em;color:#f0c060;text-transform:uppercase;margin-bottom:6px;">Restaurant Intelligence Platform</div>
  <h1 style="margin:0;font-size:2.6rem;color:#f5f5f0;">MenuProfit <span style="color:#f0c060;">Pro</span></h1>
  <p style="color:#6b7280;margin:6px 0 0 0;font-size:0.92rem;">Real-time pricing optimisation &middot; Sensitivity analysis &middot; Menu engineering &middot; Breakeven intelligence</p>
</div>
""", unsafe_allow_html=True)

# Menu input
st.markdown('<div class="section-title">Menu Input</div>', unsafe_allow_html=True)
base_df = pd.DataFrame(imported_df if uploaded else DEFAULT_DATA)
st.caption("Edit any cell directly. Use negative elasticity values (e.g. -0.8 = inelastic, -1.3 = elastic).")
menu_df = st.data_editor(base_df, use_container_width=True, num_rows="dynamic",
    column_config={
        "Cost ($)":     st.column_config.NumberColumn(format="$%.2f", min_value=0.0),
        "Price ($)":    st.column_config.NumberColumn(format="$%.2f", min_value=0.0),
        "Wkly Qty":     st.column_config.NumberColumn(min_value=0),
        "Labour (min)": st.column_config.NumberColumn(min_value=0),
        "Elasticity":   st.column_config.NumberColumn(format="%.2f", min_value=-3.0, max_value=-0.1),
    }, hide_index=True)

MIN_MARGIN     = min_margin_pct / 100
MAX_PRICE_BUMP = max_bump_pct   / 100

col_run, col_csv = st.columns([3, 1])
with col_run:
    run_btn = st.button("Run Full Analysis")
with col_csv:
    if st.session_state.eng_df is not None:
        buf = io.StringIO()
        st.session_state.eng_df.to_csv(buf, index=False)
        st.download_button("Export CSV", buf.getvalue(),
                           file_name="menuprofit_results.csv", mime="text/csv")

# Run analysis
if run_btn:
    if menu_df.empty:
        st.error("Please add at least one menu item."); st.stop()
    with st.spinner("Running optimisation engines..."):
        items_list = menu_df.to_dict("records")
        run_lp(menu_df, labour_hrs, weekly_budget)
        nlp_prices, _ = run_nlp(items_list, weekly_fixed)
        st.session_state.opt_prices = nlp_prices
        eng_df, _, _ = menu_engineering(menu_df)
        st.session_state.eng_df = eng_df
        be = breakeven(menu_df, weekly_fixed)
        st.session_state.be = be
        st.session_state.recs = build_recommendations(eng_df, be, nlp_prices)
    st.success("Analysis complete!")

# Results
if st.session_state.eng_df is not None:
    eng_df     = st.session_state.eng_df
    opt_prices = st.session_state.opt_prices
    be         = st.session_state.be
    recs       = st.session_state.recs
    items_list = menu_df.to_dict("records")

    curr_profit    = total_profit(np.array([r["Price ($)"] for r in items_list]), items_list, weekly_fixed)
    opt_profit_val = total_profit(opt_prices, items_list, weekly_fixed)
    weekly_gain    = opt_profit_val - curr_profit
    annual_gain    = weekly_gain * 52

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Optimisation", "Sensitivity", "Menu Engineering", "Breakeven", "Recommendations"
    ])

    with tab1:
        st.markdown('<div class="section-title">Current Performance Snapshot</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, label, value, sub in [
            (c1, "Weekly Revenue",    f"${be['total_rev']:,.0f}",  ""),
            (c2, "Weekly CM",         f"${be['total_cm']:,.0f}",   ""),
            (c3, "Weekly Net Profit", f"${be['net_profit']:,.0f}", ""),
            (c4, "Food Cost Ratio",   f"{be['fc_ratio']:.1f}%",    "Target 32%"),
        ]:
            with col:
                st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            fig_rev = px.bar(eng_df.assign(Revenue=eng_df["Price ($)"] * eng_df["Wkly Qty"]),
                             x="Item", y="Revenue", color="Category",
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             title="Weekly Revenue by Item")
            fig_rev.update_layout(**PLOTLY_DARK, height=320, xaxis_tickangle=-30)
            st.plotly_chart(fig_rev, use_container_width=True)
        with col_b:
            st.plotly_chart(category_chart(eng_df), use_container_width=True)
        fc_df = eng_df.copy()
        fc_df["FC%"] = fc_df["Cost ($)"] / fc_df["Price ($)"] * 100
        fig_fc = px.bar(fc_df.sort_values("FC%", ascending=False), x="Item", y="FC%",
                        color="FC%", color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                        range_color=[20, 50], title="Food Cost % per Item (Target 32%)")
        fig_fc.add_hline(y=32, line_dash="dash", line_color="#f0c060",
                          annotation_text="32% target", annotation_font_color="#f0c060")
        fig_fc.update_layout(**PLOTLY_DARK, height=280, xaxis_tickangle=-30, coloraxis_showscale=False)
        st.plotly_chart(fig_fc, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">Price Optimisation Engine</div>', unsafe_allow_html=True)
        st.caption("LP (linear) gives a fast baseline. NLP + Differential Evolution models real demand elasticity.")
        o1, o2, o3 = st.columns(3)
        with o1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Current Weekly Profit</div>'
                        f'<div class="kpi-value">${curr_profit:,.0f}</div></div>', unsafe_allow_html=True)
        with o2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Optimised Weekly Profit</div>'
                        f'<div class="kpi-value" style="color:#22c55e;">${opt_profit_val:,.0f}</div></div>',
                        unsafe_allow_html=True)
        with o3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Annual Uplift</div>'
                        f'<div class="kpi-value" style="color:#f0c060;">${annual_gain:,.0f}</div>'
                        f'<div class="kpi-sub">${weekly_gain:+,.0f} / week</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(waterfall_chart(menu_df, opt_prices), use_container_width=True)
        rows = []
        for i, item in enumerate(items_list):
            p_c = item["Price ($)"]; p_o = round(opt_prices[i] * 4) / 4
            delta = (p_o - p_c) / p_c * 100
            cm_c = item_cm(p_c, item); cm_o = item_cm(p_o, item)
            rows.append({"Item": item["Item"], "Current ($)": p_c, "Suggested ($)": p_o,
                         "Price Change": f"{delta:+.1f}%", "Current CM/wk": f"${cm_c:,.0f}",
                         "Optimised CM/wk": f"${cm_o:,.0f}", "CM Change": f"${cm_o - cm_c:+,.0f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown('<div class="section-title">Sensitivity Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(sensitivity_heatmap(menu_df), use_container_width=True)
        st.plotly_chart(tornado_chart(menu_df), use_container_width=True)
        st.markdown("#### Per-Item Deep Dive")
        sel_name = st.selectbox("Select item", menu_df["Item"].tolist(), key="sens_select")
        sel_item = menu_df[menu_df["Item"] == sel_name].to_dict("records")[0]
        sens_df  = pd.DataFrame(sensitivity(sel_item))
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=sens_df["Delta%"], y=sens_df["Wkly CM"], mode="lines+markers",
                                    name="Weekly CM", line=dict(color="#f0c060", width=2), marker=dict(size=8)))
        fig_s.add_trace(go.Scatter(x=sens_df["Delta%"], y=sens_df["Revenue"], mode="lines+markers",
                                    name="Revenue", line=dict(color="#60a5fa", width=2, dash="dot"), marker=dict(size=6)))
        fig_s.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.25)")
        fig_s.update_layout(**PLOTLY_DARK, title=f"Price Sensitivity — {sel_name}",
                             xaxis_title="Price Change %", yaxis_title="$ / week", height=320)
        st.plotly_chart(fig_s, use_container_width=True)
        disp = sens_df.copy()
        disp["Price"]    = disp["Price"].map("${:.2f}".format)
        disp["Revenue"]  = disp["Revenue"].map("${:,.0f}".format)
        disp["Wkly CM"]  = disp["Wkly CM"].map("${:,.0f}".format)
        disp["Delta CM"] = disp["Delta CM"].map("${:+,.0f}".format)
        st.dataframe(disp.rename(columns={"Delta%": "Price Delta%"}), use_container_width=True, hide_index=True)

    with tab4:
        st.markdown('<div class="section-title">Menu Engineering — BCG Matrix</div>', unsafe_allow_html=True)
        st.caption("Popularity (weekly qty) vs Profitability (CM per unit). Dashed lines = averages.")
        _, avg_cm, avg_qty = menu_engineering(menu_df)
        st.plotly_chart(bcg_scatter(eng_df, avg_cm, avg_qty), use_container_width=True)
        col_left, col_right = st.columns(2)
        for quad in ["Star", "Puzzle", "Plow Horse", "Dog"]:
            subset = eng_df[eng_df["Quadrant"] == quad]
            col    = col_left if quad in ("Star", "Plow Horse") else col_right
            with col:
                if subset.empty: continue
                items_str = ", ".join(subset["Item"].tolist())
                st.markdown(f'<div class="rec-card"><b style="color:{QUAD_COLOR[quad]};">'
                            f'{QUAD_ICON[quad]} {quad}</b> &middot; {items_str}<br>'
                            f'<span style="color:#6b7280;font-size:0.82rem;">{QUAD_ACTION[quad]}</span></div>',
                            unsafe_allow_html=True)
        fig_cm = px.bar(eng_df.sort_values("CM%", ascending=True), x="CM%", y="Item",
                        orientation="h", color="Quadrant", color_discrete_map=QUAD_COLOR,
                        title="Contribution Margin % per Item")
        fig_cm.add_vline(x=50, line_dash="dash", line_color="#f0c060",
                          annotation_text="50% target", annotation_font_color="#f0c060")
        fig_cm.update_layout(**PLOTLY_DARK, height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab5:
        st.markdown('<div class="section-title">Breakeven & Financial Health</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1: st.plotly_chart(gauge_chart(be["safety_pct"], "Operating Safety Margin", danger=10, caution=25), use_container_width=True)
        with b2: st.plotly_chart(gauge_chart(100 - be["fc_ratio"], "Gross Margin Health", danger=60, caution=68), use_container_width=True)
        with b3:
            surplus_pct = be["surplus"] / be["be_covers"] * 100 if be["be_covers"] > 0 else 0
            st.plotly_chart(gauge_chart(min(surplus_pct, 100), "Volume Buffer Above BEP", danger=20, caution=50), use_container_width=True)
        st.markdown("#### Breakeven Scenario Table")
        avg_cm_c = be["avg_cm_cover"]
        sce_rows = []
        for m in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
            fc = weekly_fixed * m
            sce_rows.append({"Fixed Costs/wk": f"${fc:,.0f}",
                              "BEP Covers":     f"{fc / avg_cm_c:.0f}" if avg_cm_c > 0 else "N/A",
                              "Current Surplus":f"{be['total_covers'] - fc / avg_cm_c:+.0f}" if avg_cm_c > 0 else "N/A",
                              "Net Profit/wk":  f"${be['total_cm'] - fc:+,.0f}",
                              "Annual Profit":  f"${(be['total_cm'] - fc) * 52:+,.0f}"})
        st.dataframe(pd.DataFrame(sce_rows), use_container_width=True, hide_index=True)
        wf = go.Figure(go.Waterfall(
            orientation="v",
            x=["Revenue", "Ingredient Cost", "Fixed Costs", "Net Profit"],
            y=[be["total_rev"], -(be["total_rev"] - be["total_cm"]), -weekly_fixed, be["net_profit"]],
            connector={"line": {"color": "rgba(255,255,255,0.1)"}},
            increasing={"marker": {"color": "#22c55e"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#f0c060"}},
            text=[f"${v:,.0f}" for v in [be["total_rev"], -(be["total_rev"] - be["total_cm"]),
                                          -weekly_fixed, be["net_profit"]]],
            textposition="outside"))
        wf.update_layout(**PLOTLY_DARK, title="Weekly P&L Waterfall", height=320)
        st.plotly_chart(wf, use_container_width=True)
        st.markdown("#### Summary Numbers")
        st.dataframe(pd.DataFrame({
            "Metric": ["Weekly CM", "Weekly Fixed Costs", "Weekly Net Profit", "Monthly Net Profit",
                       "Annual Net Profit", "Avg CM per Cover", "Breakeven Covers",
                       "Surplus Covers", "Safety Margin", "Food Cost Ratio"],
            "Value":  [f"${be['total_cm']:,.2f}", f"${be['fixed']:,.2f}", f"${be['net_profit']:,.2f}",
                       f"${be['monthly_profit']:,.2f}", f"${be['annual_profit']:,.0f}",
                       f"${be['avg_cm_cover']:.2f}", f"{be['be_covers']:.0f}/wk",
                       f"{be['surplus']:+.0f}/wk", f"{be['safety_pct']:.1f}%", f"{be['fc_ratio']:.1f}%"]
        }), use_container_width=True, hide_index=True)

    with tab6:
        st.markdown('<div class="section-title">Actionable Recommendations</div>', unsafe_allow_html=True)
        TYPE_CLASS = {"info": "", "warn": "warn", "danger": "danger", "good": "good"}
        TYPE_ICON  = {"info": "💡", "warn": "⚠️", "danger": "🔴", "good": "✅"}
        for kind, msg in recs:
            st.markdown(f'<div class="insight-card {TYPE_CLASS.get(kind, "")}">'
                        f'{TYPE_ICON.get(kind, "•")} {msg}</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### What-If Price Simulator")
        st.caption("Adjust a single item price and see real demand-adjusted impact.")
        wi_item  = st.selectbox("Item to adjust", menu_df["Item"].tolist(), key="wi_select")
        wi_row   = menu_df[menu_df["Item"] == wi_item].to_dict("records")[0]
        wi_price = st.slider(f"New price for {wi_item}",
                             min_value=float(wi_row["Cost ($)"] * 1.1),
                             max_value=float(wi_row["Price ($)"] * 2.0),
                             value=float(wi_row["Price ($)"]), step=0.25)
        wi_q      = demand(wi_price, wi_row["Price ($)"], wi_row["Wkly Qty"], wi_row["Elasticity"])
        wi_cm     = (wi_price - wi_row["Cost ($)"]) * wi_q
        base_cm_v = (wi_row["Price ($)"] - wi_row["Cost ($)"]) * wi_row["Wkly Qty"]
        delta     = wi_cm - base_cm_v
        wi1, wi2, wi3 = st.columns(3)
        wi1.metric("New Qty / Week",  f"{wi_q:.0f}",        f"{wi_q - wi_row['Wkly Qty']:+.0f}")
        wi2.metric("New CM / Week",   f"${wi_cm:,.0f}",     f"${delta:+,.0f}")
        wi3.metric("Annual Impact",   f"${delta * 52:+,.0f}", "vs current")

    st.markdown('<div style="text-align:center;color:#374151;font-size:0.78rem;padding:30px 0 10px 0;">'
                'MenuProfit Pro &middot; Built for independent restaurants &middot; Real maths, not guesswork</div>',
                unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#4b5563;">
      <div style="font-size:3rem;margin-bottom:16px;">🍽️</div>
      <div style="font-size:1.1rem;">Configure your menu above, set constraints in the sidebar,<br>
      then click <b style="color:#f0c060;">Run Full Analysis</b> to unlock all insights.</div>
    </div>
    """, unsafe_allow_html=True)
