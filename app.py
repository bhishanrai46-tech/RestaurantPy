"""
MenuProfit Pro v2  —  Restaurant Pricing & Profit Optimiser
===========================================================
OR techniques from Taha's Operations Research (10th Ed.):

  Ch. 2–3   Linear Programming      →  LP shadow price analysis
  Ch. 10    Nonlinear Optimisation   →  Price-elasticity pricing engine
  Ch. 15    Decision Analysis        →  Menu engineering classification
  Ch. 19    Monte Carlo Simulation   →  Weekly revenue forecasting

Run:
    pip install streamlit pandas numpy plotly scipy anthropic openpyxl
    streamlit run menuprofit_app.py

Set ANTHROPIC_API_KEY as an env variable or in .streamlit/secrets.toml:
    ANTHROPIC_API_KEY = "sk-ant-..."
"""

import io, os, datetime, textwrap
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize, differential_evolution, linprog

# ── Try importing Anthropic (optional) ──────────────────────────────────────
try:
    import anthropic as _anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MenuProfit Pro",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS  — refined luxury-dark aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

*, html, body { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; font-size:14px; }

[data-testid="stAppViewContainer"] { background:#080A10; color:#DDE1EE; }
[data-testid="stSidebar"]          { background:#0D0F18 !important; }
[data-testid="stHeader"]           { background:transparent !important; }
[data-testid="stToolbar"]          { display:none !important; }

/* Banner */
.banner {
    background: linear-gradient(120deg,#0F1220 0%,#0C0E19 60%,#0F1220 100%);
    border-bottom: 1px solid rgba(212,168,80,.2);
    padding: 20px 28px 16px;
    margin: -1rem -1rem 1.6rem;
    display: flex; align-items: center; gap: 16px;
}
.banner-title {
    font-family:'Cormorant Garamond',serif;
    font-size: 1.9rem; font-weight:700; color:#fff; line-height:1;
}
.banner-title span { color:#D4A840; }
.banner-sub { font-size:.78rem; color:#5A6280; margin-top:4px; letter-spacing:.03em; }
.banner-or-badge {
    margin-left:auto; background:rgba(212,168,80,.08);
    border:1px solid rgba(212,168,80,.22); border-radius:8px;
    padding:6px 14px; font-size:.72rem; color:#D4A840; letter-spacing:.1em;
    font-family:'DM Sans',sans-serif; font-weight:600;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(255,255,255,.03); border-radius:10px;
    padding:4px; gap:2px; border:1px solid rgba(255,255,255,.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius:7px; color:#5A6280 !important;
    font-size:.82rem !important; padding:8px 18px !important;
    font-weight:600 !important; letter-spacing:.02em;
}
.stTabs [aria-selected="true"] {
    background:rgba(212,168,80,.14) !important;
    color:#D4A840 !important;
}

/* Section heading */
.sh {
    font-family:'Cormorant Garamond',serif;
    font-size:1.25rem; font-weight:700; color:#D4A840;
    border-bottom:1px solid rgba(212,168,80,.15);
    padding-bottom:7px; margin:1.4rem 0 .9rem;
}

/* KPI card */
.kpi-wrap { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:1rem; }
.kpi {
    flex:1 1 120px; min-width:110px;
    background:linear-gradient(150deg,#10131E,#0C0E18);
    border:1px solid rgba(255,255,255,.07); border-radius:12px;
    padding:14px 16px; text-align:center;
}
.kpi-lbl { font-size:.68rem; color:#4A5270; text-transform:uppercase; letter-spacing:.09em; margin-bottom:5px; }
.kpi-val { font-size:1.55rem; font-weight:600; line-height:1.1; }
.kpi-sub { font-size:.68rem; color:#3A4260; margin-top:3px; }
.kpi-val.gold  { color:#D4A840; }
.kpi-val.green { color:#3DCC90; }
.kpi-val.red   { color:#E05454; }
.kpi-val.white { color:#C8CCDE; }
.kpi-val.blue  { color:#5B9EF0; }

/* Alert box */
.alert {
    border-radius:9px; padding:12px 15px; margin-bottom:9px;
    font-size:.83rem; line-height:1.65;
    border-left:3px solid #D4A840; background:rgba(212,168,80,.06);
}
.alert.red   { border-color:#E05454; background:rgba(224,84,84,.06); }
.alert.green { border-color:#3DCC90; background:rgba(61,204,144,.05); }
.alert.blue  { border-color:#5B9EF0; background:rgba(91,158,240,.05); }

/* OR insight box */
.or-box {
    background:#0D1022; border:1px solid rgba(91,158,240,.2);
    border-radius:10px; padding:14px 17px; margin:.5rem 0 1rem;
    font-size:.82rem; line-height:1.65; color:#8899CC;
}
.or-box strong { color:#5B9EF0; }

/* Tip bar */
.tip {
    background:rgba(255,255,255,.025); border-radius:8px;
    padding:9px 13px; font-size:.78rem; color:#4A5270;
    margin:.3rem 0 1rem; border:1px solid rgba(255,255,255,.05);
}

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#A07820,#D4A840) !important;
    color:#080A10 !important; font-weight:700 !important;
    border:none !important; border-radius:10px !important;
    padding:10px 22px !important; font-size:.88rem !important;
    width:100% !important; letter-spacing:.02em !important;
}
.stButton > button:hover { filter:brightness(1.1) !important; }

/* Metric */
[data-testid="stMetric"] {
    background:rgba(12,14,24,.9); border-radius:10px;
    border:1px solid rgba(255,255,255,.06); padding:10px !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"]  input { font-size:.88rem; }

/* AI response area */
.ai-box {
    background:#0D1022; border:1px solid rgba(212,168,80,.18);
    border-radius:12px; padding:20px 22px; margin-top:12px;
    font-size:.88rem; line-height:1.85; color:#B8BECE;
    white-space:pre-wrap;
}
.ai-box h3 { color:#D4A840; font-family:'Cormorant Garamond',serif; font-size:1.1rem; margin:12px 0 5px; }
.ai-box strong { color:#D4A840; }

@media (max-width:640px) {
    .banner-title { font-size:1.4rem; }
    .kpi-val { font-size:1.25rem; }
    .stTabs [data-baseweb="tab"] { font-size:.74rem !important; padding:6px 9px !important; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#8090B0"),
    margin=dict(l=16, r=16, t=40, b=16),
)
GOLD   = "#D4A840"
GREEN  = "#3DCC90"
RED    = "#E05454"
BLUE   = "#5B9EF0"
PURPLE = "#9B72D8"

QUAD_COLOR = {"Star": GOLD, "Puzzle": BLUE, "Plow Horse": GREEN, "Dog": RED}
QUAD_ICON  = {"Star": "⭐", "Puzzle": "🧩", "Plow Horse": "🐎", "Dog": "🐕"}

MIN_MARGIN   = 0.28
MAX_PRICE_UP = 0.35

# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULT MENU DATA
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT = {
    "Dish":             ["Wagyu Burger","Truffle Pasta","Caesar Salad","Espresso Martini",
                         "Lava Cake","House Wine","Grilled Salmon","Garlic Bread"],
    "Category":         ["Mains","Mains","Starters","Drinks","Desserts","Drinks","Mains","Starters"],
    "Cost ($)":         [8.50, 6.20, 2.80, 1.90, 2.40, 3.20, 9.80, 0.80],
    "Price ($)":        [24.00,22.00,14.00,16.00,12.00,11.00,28.00, 7.00],
    "Sold / Week":      [120,   85,  200,  180,   90,  250,   70, 160],
    "Cook Time (min)":  [ 12,   15,    5,    3,    8,    2,   18,   4],
    "Elasticity":       [-1.2, -1.0, -0.8, -0.6, -1.1, -0.9, -1.3,-0.7],
}

# ─────────────────────────────────────────────────────────────────────────────
#  OR MATH ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# ── Demand function (price elasticity of demand) ─────────────────────────────
def demand(new_price, base_price, base_qty, elasticity):
    """
    Power-law demand curve (Taha Ch. 10 — nonlinear models).
    qty = base_qty × (new_price / base_price)^elasticity
    """
    if new_price <= 0:
        return 0.0
    return float(base_qty) * (new_price / float(base_price)) ** float(elasticity)

def margin_contribution(price, cost):
    return price - cost

def weekly_profit_item(price, row):
    qty = demand(price, row["Price ($)"], row["Sold / Week"], row["Elasticity"])
    return margin_contribution(price, row["Cost ($)"]) * qty

def total_weekly_profit(prices, rows, fixed):
    return sum(weekly_profit_item(prices[i], rows[i]) for i in range(len(rows))) - fixed


# ── Linear Programming with shadow prices (Taha Ch. 2–3) ────────────────────
def run_lp(df, labour_hrs, budget):
    """
    Maximise contribution margin subject to:
      ∑ cook_time_h × qty ≤ labour_hrs    (capacity constraint)
      ∑ cost × qty        ≤ budget         (ingredient budget)
      0 ≤ qty ≤ base_qty

    Returns (result, shadow_labour, shadow_budget).
    Shadow prices (dual variables) answer: 'how much extra profit does one
    extra unit of this resource generate?' — a key LP insight from Ch. 3.
    """
    cm      = (df["Price ($)"] - df["Cost ($)"]).values
    cook_h  = df["Cook Time (min)"].values / 60.0
    costs   = df["Cost ($)"].values
    qtys    = df["Sold / Week"].values.astype(float)

    result = linprog(
        -cm,
        A_ub=[cook_h, costs],
        b_ub=[labour_hrs, budget],
        bounds=[(0.0, q) for q in qtys],
        method="highs",
    )

    shadow_labour = shadow_budget = 0.0
    if result.status == 0 and hasattr(result, "ineqlin") and result.ineqlin is not None:
        m = result.ineqlin.marginals
        if m is not None and len(m) >= 2:
            shadow_labour = float(abs(m[0]))
            shadow_budget = float(abs(m[1]))

    return result, shadow_labour, shadow_budget


# ── Nonlinear price optimiser (Taha Ch. 10 — scipy differential evolution) ──
def optimise_prices(rows, fixed, min_margin=MIN_MARGIN, max_up=MAX_PRICE_UP):
    """
    Global optimisation using differential evolution (Taha §10.4).
    Objective: maximise total weekly net profit.
    Bounds: [cost × (1+min_margin),  current_price × (1+max_up)]
    """
    bounds = [
        (r["Cost ($)"] * (1 + min_margin), r["Price ($)"] * (1 + max_up))
        for r in rows
    ]
    p0 = np.array([r["Price ($)"] for r in rows])

    # Fast local refinement first
    r1 = minimize(
        lambda p: -total_weekly_profit(p, rows, fixed),
        p0, bounds=bounds, method="L-BFGS-B",
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    # Global search (differential evolution)
    r2 = differential_evolution(
        lambda p: -total_weekly_profit(p, rows, fixed),
        bounds, seed=42, maxiter=300, popsize=12, tol=1e-9,
    )
    if -r2.fun >= -r1.fun:
        return r2.x, -r2.fun
    return r1.x, -r1.fun


# ── Menu classification — Decision Analysis (Taha Ch. 15) ───────────────────
def classify_menu(df):
    df = df.copy()
    df["Profit/Dish"] = df["Price ($)"] - df["Cost ($)"]
    df["Margin %"]    = df["Profit/Dish"] / df["Price ($)"] * 100
    df["CM/Min"]      = df["Profit/Dish"] / df["Cook Time (min)"].replace(0, 1)  # efficiency
    avg_p = df["Profit/Dish"].mean()
    avg_q = df["Sold / Week"].mean()

    def quad(r):
        hi_p = r["Profit/Dish"] >= avg_p
        hi_q = r["Sold / Week"] >= avg_q
        if hi_p and hi_q:  return "Star"
        if hi_p:           return "Puzzle"
        if hi_q:           return "Plow Horse"
        return "Dog"

    df["Type"] = df.apply(quad, axis=1)
    return df, avg_p, avg_q


# ── Monte Carlo simulation (Taha Ch. 19) ─────────────────────────────────────
def monte_carlo(df, n=8000, sigma=0.15, fixed=0):
    """
    Simulate weekly profit under uncertain demand (±15% default std dev).
    Applies Ch. 19 Monte Carlo method: draw N demand scenarios and compute
    the 90% confidence interval on net weekly profit.
    """
    cm   = (df["Price ($)"] - df["Cost ($)"]).values
    qtys = df["Sold / Week"].values.astype(float)
    rng  = np.random.default_rng(42)
    sim  = rng.normal(qtys, qtys * sigma, (n, len(qtys))).clip(0)
    profits = (sim * cm).sum(axis=1) - fixed
    return {
        "mean": profits.mean(),
        "p5":   np.percentile(profits, 5),
        "p50":  np.percentile(profits, 50),
        "p95":  np.percentile(profits, 95),
        "sims": profits,
    }


# ── Financial summary ─────────────────────────────────────────────────────────
def calc_fin(df, fixed):
    cm     = df["Price ($)"] - df["Cost ($)"]
    pool   = (cm * df["Sold / Week"]).sum()
    rev    = (df["Price ($)"] * df["Sold / Week"]).sum()
    covers = df["Sold / Week"].sum()
    net    = pool - fixed
    avg_cm = pool / covers if covers else 0
    bep    = fixed / avg_cm if avg_cm > 0 else float("inf")
    fc_pct = (df["Cost ($)"] * df["Sold / Week"]).sum() / rev * 100 if rev > 0 else 0
    return dict(
        pool=pool, rev=rev, net=net,
        monthly=net * 4.33, annual=net * 52,
        covers=covers, avg_cm=avg_cm,
        bep=bep, surplus=covers - bep,
        safety=net / pool * 100 if pool > 0 else 0,
        fc_pct=fc_pct, fixed=fixed,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_html(label, value, sub="", cls="gold"):
    return (f'<div class="kpi">'
            f'<div class="kpi-lbl">{label}</div>'
            f'<div class="kpi-val {cls}">{value}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>')


def fig_bar(names, values, title, h=290):
    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker=dict(
            color=values,
            colorscale=[[0, RED], [0.4, "#F0A030"], [1, GREEN]],
            showscale=False,
            line=dict(width=0),
        ),
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color="#8090B0"),
    ))
    fig.update_layout(**DARK, title=title, height=h, xaxis_tickangle=-28)
    return fig


def fig_waterfall(names, values, title, h=300):
    fig = go.Figure(go.Waterfall(
        orientation="v", x=names, y=values,
        connector={"line": {"color": "rgba(255,255,255,.06)"}},
        increasing={"marker": {"color": GREEN}},
        decreasing={"marker": {"color": RED}},
        totals={"marker": {"color": GOLD}},
        text=[f"${abs(v):+,.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color="#8090B0"),
    ))
    fig.update_layout(**DARK, title=title, height=h, xaxis_tickangle=-28)
    return fig


def fig_bcg(eng_df, avg_p, avg_q):
    fig = go.Figure()
    for t, grp in eng_df.groupby("Type"):
        fig.add_trace(go.Scatter(
            x=grp["Sold / Week"], y=grp["Profit/Dish"],
            mode="markers+text", name=f"{QUAD_ICON[t]} {t}",
            text=grp["Dish"], textposition="top center",
            textfont=dict(size=9, color="#8090B0"),
            marker=dict(
                size=16, color=QUAD_COLOR[t], opacity=.9,
                line=dict(width=1.5, color="rgba(255,255,255,.12)"),
            ),
        ))
    fig.add_hline(y=avg_p, line_dash="dot", line_color="rgba(255,255,255,.12)")
    fig.add_vline(x=avg_q, line_dash="dot", line_color="rgba(255,255,255,.12)")
    for lbl, ax, ay, xa, ya in [
        ("⭐ STARS",       eng_df["Sold / Week"].max()*1.02, eng_df["Profit/Dish"].max()*1.02, "right","top"),
        ("🧩 PUZZLES",     eng_df["Sold / Week"].min()*.92,  eng_df["Profit/Dish"].max()*1.02, "left","top"),
        ("🐎 PLOW HORSES", eng_df["Sold / Week"].max()*1.02, eng_df["Profit/Dish"].min()*.92,  "right","bottom"),
        ("🐕 DOGS",        eng_df["Sold / Week"].min()*.92,  eng_df["Profit/Dish"].min()*.92,  "left","bottom"),
    ]:
        fig.add_annotation(x=ax, y=ay, text=lbl, showarrow=False,
                           font=dict(size=8, color="rgba(255,255,255,.15)"),
                           xanchor=xa, yanchor=ya)
    fig.update_layout(
        **DARK, height=390,
        title="Menu Engineering Matrix — popularity vs. profit (Taha Ch. 15)",
        xaxis_title="Orders / Week  (popularity)", yaxis_title="Profit per Dish $",
    )
    return fig


def fig_mc(mc):
    fig = go.Figure()
    bins = np.linspace(mc["sims"].min(), mc["sims"].max(), 60)
    counts, edges = np.histogram(mc["sims"], bins=bins)
    mid = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=mid, y=counts,
        marker=dict(
            color=mid,
            colorscale=[[0, RED], [0.35, "#F0A030"], [0.65, BLUE], [1, GREEN]],
            showscale=False, line=dict(width=0),
        ),
        name="Simulated outcomes",
        opacity=.85,
    ))
    for val, lbl, col in [
        (mc["p5"],  "P5",   RED),
        (mc["p50"], "P50",  GOLD),
        (mc["p95"], "P95",  GREEN),
    ]:
        fig.add_vline(x=val, line_color=col, line_dash="dash", line_width=1.5,
                      annotation_text=f"{lbl} ${val:,.0f}",
                      annotation_font_color=col, annotation_font_size=10)
    fig.update_layout(
        **DARK, height=270,
        title=f"Monte Carlo revenue forecast — 8,000 simulations (Taha Ch. 19)",
        xaxis_title="Weekly Profit ($)", yaxis_title="Frequency",
    )
    return fig


def fig_fc_bar(df):
    fc_vals = (df["Cost ($)"] / df["Price ($)"] * 100).tolist()
    fig = go.Figure(go.Bar(
        x=df["Dish"].tolist(), y=fc_vals,
        marker=dict(
            color=fc_vals,
            colorscale=[[0, GREEN], [0.4, "#F0A030"], [1, RED]],
            showscale=False, line=dict(width=0),
        ),
        text=[f"{v:.0f}%" for v in fc_vals],
        textposition="outside", textfont=dict(size=10, color="#8090B0"),
    ))
    fig.add_hline(y=32, line_dash="dash", line_color=GOLD,
                  annotation_text="Target 32%", annotation_font_color=GOLD)
    fig.update_layout(**DARK, height=260,
                      title="Food cost % per dish  (lower = better)",
                      xaxis_tickangle=-28)
    return fig


def fig_price_comparison(rows, opt_prices):
    names = [r["Dish"] for r in rows]
    curr  = [r["Price ($)"] for r in rows]
    opt   = [round(p * 4) / 4 for p in opt_prices]
    fig   = go.Figure()
    fig.add_trace(go.Bar(name="Current price",   x=names, y=curr,
                         marker_color=BLUE, opacity=.65))
    fig.add_trace(go.Bar(name="Suggested price", x=names, y=opt,
                         marker_color=GOLD, opacity=.85))
    fig.update_layout(**DARK, barmode="group", height=280,
                      title="Current vs. suggested prices",
                      xaxis_tickangle=-28, yaxis_title="Price ($)")
    return fig


def fig_shadow_prices(sh_l, sh_b, labour_hrs, budget):
    fig = go.Figure(go.Bar(
        x=["Extra hour of kitchen time", "Extra $1 ingredient budget"],
        y=[sh_l, sh_b],
        marker=dict(color=[BLUE, GOLD], line=dict(width=0)),
        text=[f"${sh_l:.2f}", f"${sh_b:.3f}"],
        textposition="outside",
        textfont=dict(size=12, color="#8090B0"),
    ))
    fig.update_layout(**DARK, height=220,
                      title="LP Shadow Prices — value of relaxing each constraint (Taha Ch. 3)",
                      yaxis_title="Extra weekly profit ($)")
    return fig


def fig_pl(fin):
    fig = go.Figure(go.Waterfall(
        orientation="v",
        x=["Sales Revenue", "− Ingredient Cost", "− Fixed Costs", "Net Profit"],
        y=[fin["rev"],
           -(fin["rev"] - fin["pool"]),
           -fin["fixed"],
           fin["net"]],
        connector={"line": {"color": "rgba(255,255,255,.06)"}},
        increasing={"marker": {"color": GREEN}},
        decreasing={"marker": {"color": RED}},
        totals={"marker": {"color": GOLD}},
        text=[f"${abs(v):,.0f}" for v in
              [fin["rev"], -(fin["rev"]-fin["pool"]), -fin["fixed"], fin["net"]]],
        textposition="outside",
        textfont=dict(size=10, color="#8090B0"),
    ))
    fig.update_layout(**DARK, title="Weekly P&L waterfall", height=280)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  AI RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def build_ai_prompt(eng_df, fin, opt_prices, sh_l, sh_b, mc):
    dish_lines = []
    for i, row in eng_df.iterrows():
        p_opt  = round(float(opt_prices[i]) * 4) / 4
        uplift = (p_opt - row["Price ($)"]) / row["Price ($)"] * 100
        dish_lines.append(
            f"  • {row['Dish']} ({row['Category']}) | "
            f"Cost ${row['Cost ($)']:.2f} | Price ${row['Price ($)']:.2f} | "
            f"Margin {row['Margin %']:.1f}% | {row['Sold / Week']:.0f}/wk | "
            f"Type: {row['Type']} | Suggested: ${p_opt:.2f} ({uplift:+.1f}%) | "
            f"Efficiency: ${row['CM/Min']:.2f}/min cook-time"
        )
    binding = "kitchen time" if sh_l > sh_b * 10 else "ingredient budget"
    prompt = f"""You are a sharp restaurant profit consultant. Analyse this data and give specific, actionable advice.

WEEKLY FINANCIALS
  Revenue      : ${fin['rev']:,.0f}
  Profit Pool  : ${fin['pool']:,.0f}   (before fixed costs)
  Net Profit   : ${fin['net']:,.0f}   (after fixed costs ${fin['fixed']:,.0f})
  Food Cost    : {fin['fc_pct']:.1f}%  (industry target <32%)
  Safety Buffer: {fin['safety']:.1f}% above breakeven
  Breakeven    : {fin['bep']:.0f} orders/wk  (current: {fin['covers']:.0f})
  Annual Profit: ${fin['annual']:,.0f}

LP SHADOW PRICES (resource value — Taha Ch. 3)
  Each extra hour of kitchen time is worth ${sh_l:.2f}/wk in profit
  Each extra $1 of ingredient budget yields ${sh_b:.3f}/wk in profit
  → Binding constraint is {binding}

MONTE CARLO FORECAST (8,000 simulations — Taha Ch. 19)
  Expected weekly profit : ${mc['mean']:,.0f}
  90% confidence interval: ${mc['p5']:,.0f} – ${mc['p95']:,.0f}
  Risk (P5 scenario)     : ${mc['p5']:,.0f}

MENU (with OR-optimised suggested prices)
{chr(10).join(dish_lines)}

Respond with exactly these 4 sections. Be specific — use dollar amounts from the data above.

### 🎯 IMMEDIATE ACTIONS
3 specific things to do this week (reference dishes by name, prices, expected dollar impact).

### 💰 PRICING STRATEGY
Which prices to change, by how much, and why (reference elasticity, shadow prices, and the LP analysis).

### 🍽️ MENU CHANGES
What to keep, promote, remove, or bundle — with reasoning from the engineering matrix (Stars/Puzzles/Plow Horses/Dogs).

### 📈 REVENUE UPSIDE
Quantify the total untapped opportunity and 2 specific ways to capture it within 30 days."""
    return prompt


def stream_ai_response(prompt, api_key):
    """Stream Claude's response in Streamlit."""
    client = _anthropic.Anthropic(api_key=api_key)
    placeholder = st.empty()
    full = ""
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=(
            "You are a restaurant profitability consultant with expertise in "
            "menu engineering and operations research. Be concise, specific, "
            "and always reference dollar amounts from the provided data."
        ),
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            full += chunk
            placeholder.markdown(f'<div class="ai-box">{full}▌</div>', unsafe_allow_html=True)
    placeholder.markdown(f'<div class="ai-box">{full}</div>', unsafe_allow_html=True)
    return full


# ─────────────────────────────────────────────────────────────────────────────
#  HTML REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(df, eng_df, fin, opt_prices, mc, sh_l, sh_b, ai_text, period, rdate):
    factor  = 4.33 if period == "Monthly" else 1
    p_lbl   = "Month" if period == "Monthly" else "Week"
    rows    = df.to_dict("records")
    curr_p  = total_weekly_profit(np.array([r["Price ($)"] for r in rows]), rows, fin["fixed"])
    opt_p   = total_weekly_profit(opt_prices, rows, fin["fixed"])
    gain_wk = opt_p - curr_p

    dish_rows = ""
    for i, row in eng_df.iterrows():
        p_opt = round(opt_prices[i] * 4) / 4
        chg   = (p_opt - row["Price ($)"]) / row["Price ($)"] * 100
        fc_p  = row["Cost ($)"] / row["Price ($)"] * 100
        dish_rows += f"""<tr>
          <td>{row['Dish']}</td><td>{row['Category']}</td>
          <td>${row['Cost ($)']:.2f}</td><td>${row['Price ($)']:.2f}</td>
          <td>${row['Profit/Dish']:.2f} ({row['Margin %']:.0f}%)</td>
          <td>{int(row['Sold / Week'])}</td>
          <td>${row['Profit/Dish']*row['Sold / Week']:,.0f}</td>
          <td style="color:{'#F0A030' if p_opt!=row['Price ($)'] else '#666'}">${p_opt:.2f} ({chg:+.1f}%)</td>
          <td style="color:{'#E05454' if fc_p>35 else '#3DCC90'}">{fc_p:.0f}%</td>
          <td><span style="background:{QUAD_COLOR[row['Type']]}22;color:{QUAD_COLOR[row['Type']]};
                padding:2px 9px;border-radius:20px;font-size:11px;">{QUAD_ICON[row['Type']]} {row['Type']}</span></td>
        </tr>"""

    ai_section = ""
    if ai_text:
        ai_section = f"""
        <div class="sec">
          <h2>AI Recommendations</h2>
          <div style="white-space:pre-wrap;font-size:13.5px;line-height:1.8;color:#B8BECE;
                      background:#0D1022;border-radius:10px;padding:18px 20px;border:1px solid rgba(212,168,80,.15);">
            {ai_text}
          </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MenuProfit Pro — {period} Report — {rdate}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=DM+Sans:wght@300;400;600&display=swap');
*{{box-sizing:border-box;margin:0;padding:0;font-family:'DM Sans',sans-serif}}
body{{background:#080A10;color:#DDE1EE;padding:0}}
.page{{max-width:980px;margin:0 auto;padding:28px 20px}}
.header{{background:linear-gradient(120deg,#0F1220,#0C0E19);border-bottom:1px solid rgba(212,168,80,.2);
         padding:26px 30px;border-radius:14px;margin-bottom:26px}}
.header h1{{font-family:'Cormorant Garamond',serif;font-size:1.9rem;color:#fff}}
.header h1 span{{color:#D4A840}}
.header p{{color:#5A6280;font-size:.8rem;margin-top:5px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:26px}}
.kpi{{background:linear-gradient(145deg,#10131E,#0C0E18);border:1px solid rgba(255,255,255,.07);
      border-radius:11px;padding:16px;text-align:center}}
.kpi .lbl{{font-size:.68rem;color:#4A5270;text-transform:uppercase;letter-spacing:.09em;margin-bottom:5px}}
.kpi .val{{font-size:1.5rem;font-weight:600;color:#D4A840}}
.kpi .sub{{font-size:.68rem;color:#3A4260;margin-top:3px}}
.kpi.g .val{{color:#3DCC90}}.kpi.r .val{{color:#E05454}}.kpi.w .val{{color:#C8CCDE}}
.sec{{margin-bottom:26px}}
.sec h2{{font-family:'Cormorant Garamond',serif;font-size:1.25rem;color:#D4A840;
         border-bottom:1px solid rgba(212,168,80,.15);padding-bottom:6px;margin-bottom:13px}}
.or-box{{background:#0D1022;border:1px solid rgba(91,158,240,.18);border-radius:9px;
         padding:13px 16px;margin-bottom:13px;font-size:13px;line-height:1.65;color:#7890B8}}
.or-box strong{{color:#5B9EF0}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
thead tr{{background:#10131E}}
th{{padding:9px 10px;text-align:left;color:#5A6280;font-weight:600;
   font-size:.68rem;text-transform:uppercase;letter-spacing:.06em}}
td{{padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:middle}}
tr:hover{{background:rgba(255,255,255,.02)}}
.foot{{text-align:center;color:#2A3050;font-size:.72rem;
       padding:18px 0;border-top:1px solid rgba(255,255,255,.04);margin-top:28px}}
@media print{{body{{background:#fff;color:#111}}
  .header,.kpi,.or-box{{background:#f9f9f9;border-color:#ddd;color:#111}}
  .kpi .val{{color:#A07820}}.sec h2{{color:#A07820}}}}
</style>
</head>
<body><div class="page">
<div class="header">
  <h1>🍽️ MenuProfit <span>Pro</span></h1>
  <p>{period} Report &nbsp;·&nbsp; Generated {rdate} &nbsp;·&nbsp; Period basis: per {p_lbl}</p>
</div>
<div class="sec"><h2>At a Glance</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="lbl">Sales / {p_lbl}</div><div class="val">${fin['rev']*factor:,.0f}</div></div>
  <div class="kpi"><div class="lbl">Net Profit / {p_lbl}</div>
    <div class="val {'g' if fin['net']>0 else 'r'}">${fin['net']*factor:,.0f}</div></div>
  <div class="kpi {'g' if gain_wk*factor>0 else ''}"><div class="lbl">Extra if Repriced</div>
    <div class="val">${gain_wk*factor:+,.0f}</div><div class="sub">per {p_lbl.lower()}</div></div>
  <div class="kpi {'r' if fin['fc_pct']>35 else 'g'}"><div class="lbl">Food Cost %</div>
    <div class="val">{fin['fc_pct']:.1f}%</div><div class="sub">target &lt;32%</div></div>
  <div class="kpi {'g' if fin['safety']>25 else 'r'}"><div class="lbl">Safety Buffer</div>
    <div class="val">{fin['safety']:.1f}%</div></div>
  <div class="kpi w"><div class="lbl">Annual Profit</div><div class="val">${fin['annual']:,.0f}</div></div>
  <div class="kpi w"><div class="lbl">Breakeven Orders</div>
    <div class="val">{fin['bep']:.0f}</div><div class="sub">{fin['covers']:.0f} current</div></div>
  <div class="kpi {'g' if mc['p5']>0 else 'r'}"><div class="lbl">P5 Revenue Risk</div>
    <div class="val">${mc['p5']:,.0f}</div><div class="sub">90% CI</div></div>
</div></div>
<div class="sec"><h2>OR Analysis Summary</h2>
<div class="or-box">
  <strong>LP Shadow Prices (Taha Ch. 3):</strong><br>
  Each extra hour of kitchen time = <strong>${sh_l:.2f}</strong> additional weekly profit &nbsp;|&nbsp;
  Each extra $1 ingredient budget = <strong>${sh_b:.3f}</strong> additional weekly profit<br><br>
  <strong>Monte Carlo (Taha Ch. 19) — 8,000 simulations:</strong><br>
  Expected weekly profit: <strong>${mc['mean']:,.0f}</strong> &nbsp;|&nbsp;
  90% CI: <strong>${mc['p5']:,.0f} – ${mc['p95']:,.0f}</strong>
</div></div>
<div class="sec"><h2>Full Menu Breakdown</h2>
<table><thead><tr>
  <th>Dish</th><th>Category</th><th>Cost</th><th>Price</th>
  <th>Profit/Dish</th><th>Orders/Wk</th><th>Wkly Profit</th>
  <th>Suggested $</th><th>Food Cost%</th><th>Rating</th>
</tr></thead><tbody>{dish_rows}</tbody></table></div>
{ai_section}
<div class="foot">MenuProfit Pro &nbsp;·&nbsp; {period} Report &nbsp;·&nbsp; {rdate}</div>
</div></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("opt_prices", None), ("eng_df", None), ("fin", None),
             ("mc", None), ("sh_l", 0.0), ("sh_b", 0.0),
             ("ai_text", ""), ("ran", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <div>
    <div class="banner-title">🍽️ MenuProfit <span>Pro</span></div>
    <div class="banner-sub">Pricing intelligence for independent restaurants · Built on Operations Research</div>
  </div>
  <div class="banner-or-badge">LP · MONTE CARLO · AI</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Business Settings")
    weekly_fixed  = st.number_input("Fixed costs / week ($)", value=3500, step=100,
                                     help="Rent, wages, utilities — paid every week regardless of sales")
    labour_hrs    = st.number_input("Kitchen hours / week",    value=120,  step=5)
    weekly_budget = st.number_input("Ingredient budget / week ($)", value=4000, step=100)
    st.divider()
    st.markdown("### 📐 Optimiser Rules")
    min_mg = st.slider("Min margin %", 20, 60, 28)
    max_up = st.slider("Max price increase %", 5, 50, 35)
    st.divider()
    st.markdown("### 🤖 AI Key")
    api_key_input = st.text_input(
        "Anthropic API key", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Needed for AI Recommendations tab. Set ANTHROPIC_API_KEY env var to avoid pasting here.",
    )
    st.divider()
    st.markdown("### 📂 Upload Menu CSV")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded:
        try:
            imported_df = pd.read_csv(uploaded)
            required = list(DEFAULT.keys())
            missing = [c for c in required if c not in imported_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}"); uploaded = None
            else:
                st.success("✓ File loaded")
        except Exception as e:
            st.error(str(e)); uploaded = None

MIN_MARGIN   = min_mg / 100
MAX_PRICE_UP = max_up / 100

# ─────────────────────────────────────────────────────────────────────────────
#  MENU TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sh">📋 Your Menu</div>', unsafe_allow_html=True)
st.markdown('<div class="tip">Edit any cell. Add rows with the + button. '
            'Elasticity guide: −0.5 coffee/cocktails | −1.0 typical | −1.5 price-sensitive staples</div>',
            unsafe_allow_html=True)

base_df = pd.DataFrame(imported_df if uploaded else DEFAULT)
menu_df = st.data_editor(
    base_df, use_container_width=True, num_rows="dynamic",
    column_config={
        "Cost ($)":   st.column_config.NumberColumn("Cost ($)",    format="$%.2f", min_value=0.0),
        "Price ($)":  st.column_config.NumberColumn("Price ($)",   format="$%.2f", min_value=0.0),
        "Sold / Week":st.column_config.NumberColumn("Sold / Week", min_value=0),
        "Cook Time (min)": st.column_config.NumberColumn("Cook Time (min)", min_value=1),
        "Elasticity": st.column_config.NumberColumn("Elasticity",  format="%.1f",
                                                    min_value=-3.0, max_value=-0.1),
    },
    hide_index=True,
)

# Validation
if not menu_df.empty:
    bad = menu_df[menu_df["Cost ($)"] >= menu_df["Price ($)"]]
    if not bad.empty:
        st.warning(f"⚠️ Cost ≥ price: {', '.join(bad['Dish'].tolist())} — fix before running.")

c_run, c_exp = st.columns([3, 1])
with c_run:
    run_btn = st.button("🚀  Analyse Menu & Optimise Prices")
with c_exp:
    if st.session_state.eng_df is not None:
        buf = io.StringIO()
        st.session_state.eng_df.to_csv(buf, index=False)
        st.download_button("⬇️ Export CSV", buf.getvalue(), "menuprofit.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
#  RUN ENGINE
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if menu_df.empty:
        st.error("Add at least one dish."); st.stop()
    with st.spinner("Running LP, price optimiser, and Monte Carlo…"):
        rows = menu_df.to_dict("records")

        # 1. LP shadow prices (Taha Ch. 2–3)
        lp_res, sh_l, sh_b = run_lp(menu_df, labour_hrs, weekly_budget)

        # 2. Nonlinear price optimiser (Taha Ch. 10)
        opt_p, _ = optimise_prices(rows, weekly_fixed, MIN_MARGIN, MAX_PRICE_UP)

        # 3. Menu classification (Taha Ch. 15)
        eng_df, _, _ = classify_menu(menu_df)

        # 4. Financials
        fin = calc_fin(menu_df, weekly_fixed)

        # 5. Monte Carlo (Taha Ch. 19)
        mc = monte_carlo(menu_df, fixed=weekly_fixed)

        st.session_state.update(
            opt_prices=opt_p, eng_df=eng_df, fin=fin,
            mc=mc, sh_l=sh_l, sh_b=sh_b, ran=True, ai_text=""
        )
    st.success("✓ Analysis complete — scroll down for results")

# ─────────────────────────────────────────────────────────────────────────────
#  4-TAB RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.ran:
    eng_df     = st.session_state.eng_df
    opt_prices = st.session_state.opt_prices
    fin        = st.session_state.fin
    mc         = st.session_state.mc
    sh_l       = st.session_state.sh_l
    sh_b       = st.session_state.sh_b
    rows       = menu_df.to_dict("records")

    curr_profit = total_weekly_profit(np.array([r["Price ($)"] for r in rows]), rows, weekly_fixed)
    opt_profit  = total_weekly_profit(opt_prices, rows, weekly_fixed)
    wk_gain     = opt_profit - curr_profit

    t1, t2, t3, t4 = st.tabs([
        "📊  Overview",
        "💰  Price Calculator",
        "🤖  AI Recommendations",
        "📄  Reports",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════════════════
    with t1:
        st.markdown('<div class="sh">This Week at a Glance</div>', unsafe_allow_html=True)

        # Row 1 KPIs
        cols = st.columns(4)
        kpis = [
            ("Total Revenue",    f"${fin['rev']:,.0f}",    "this week",              "white"),
            ("Net Profit",       f"${fin['net']:,.0f}",    "after all costs",        "green" if fin["net"] > 0 else "red"),
            ("Annual Profit",    f"${fin['annual']:,.0f}", "×52 weeks",              "gold"),
            ("Food Cost %",      f"{fin['fc_pct']:.1f}%",  "target under 32%",      "green" if fin["fc_pct"] <= 32 else "red"),
        ]
        for col, (lbl, val, sub, cls) in zip(cols, kpis):
            col.markdown(
                '<div class="kpi-wrap">' + kpi_html(lbl, val, sub, cls) + '</div>',
                unsafe_allow_html=True,
            )

        # Row 2 KPIs
        cols2 = st.columns(4)
        kpis2 = [
            ("Safety Buffer",    f"{fin['safety']:.1f}%",  "above breakeven",        "green" if fin["safety"] > 25 else "red"),
            ("Breakeven Orders", f"{fin['bep']:.0f}/wk",   f"you have {fin['covers']:.0f}", "white"),
            ("MC Risk (P5)",     f"${mc['p5']:,.0f}",       "bad-week scenario",      "blue"),
            ("Gain if Repriced", f"${wk_gain:+,.0f}/wk",   f"${wk_gain*52:+,.0f}/yr","green" if wk_gain > 0 else "white"),
        ]
        for col, (lbl, val, sub, cls) in zip(cols2, kpis2):
            col.markdown(
                '<div class="kpi-wrap">' + kpi_html(lbl, val, sub, cls) + '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Menu engineering matrix + P&L side by side
        c_left, c_right = st.columns([3, 2])
        with c_left:
            _, avg_p, avg_q = classify_menu(menu_df)
            st.plotly_chart(fig_bcg(eng_df, avg_p, avg_q), use_container_width=True)
        with c_right:
            st.plotly_chart(fig_pl(fin), use_container_width=True)
            st.plotly_chart(fig_fc_bar(menu_df), use_container_width=True)

        # Weekly profit bar
        wk_profits = [
            (r["Price ($)"] - r["Cost ($)"]) * r["Sold / Week"] for r in rows
        ]
        st.plotly_chart(
            fig_bar(menu_df["Dish"].tolist(), wk_profits, "Weekly profit contribution by dish"),
            use_container_width=True,
        )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — PRICE CALCULATOR
    # ════════════════════════════════════════════════════════════════════════
    with t2:
        st.markdown('<div class="sh">💰 Price Optimiser</div>', unsafe_allow_html=True)

        # OR explanation box
        binding_res = "kitchen time" if sh_l > sh_b * 10 else "ingredient budget"
        st.markdown(
            f'<div class="or-box">'
            f'<strong>LP Analysis (Taha Ch. 2–3):</strong> '
            f'Your binding constraint is <strong>{binding_res}</strong>. '
            f'Each extra hour of kitchen time is worth <strong>${sh_l:.2f}/wk</strong> in profit. '
            f'Each extra $1 of ingredient budget generates <strong>${sh_b:.3f}/wk</strong>. '
            f'Loosen the binding constraint first before cutting costs elsewhere.'
            f'</div>',
            unsafe_allow_html=True,
        )

        # KPI row
        pc1, pc2, pc3 = st.columns(3)
        pc1.markdown('<div class="kpi-wrap">' +
                     kpi_html("Current Weekly Profit", f"${curr_profit:,.0f}", "", "white") +
                     '</div>', unsafe_allow_html=True)
        pc2.markdown('<div class="kpi-wrap">' +
                     kpi_html("Optimised Weekly Profit", f"${opt_profit:,.0f}", "", "green") +
                     '</div>', unsafe_allow_html=True)
        pc3.markdown('<div class="kpi-wrap">' +
                     kpi_html("Extra Per Year", f"${wk_gain*52:+,.0f}", "if you reprice today", "gold") +
                     '</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Shadow price chart + price comparison
        c_shad, c_comp = st.columns(2)
        with c_shad:
            st.plotly_chart(
                fig_shadow_prices(sh_l, sh_b, labour_hrs, weekly_budget),
                use_container_width=True,
            )
        with c_comp:
            st.plotly_chart(fig_price_comparison(rows, opt_prices), use_container_width=True)

        # Monte Carlo
        st.plotly_chart(fig_mc(mc), use_container_width=True)

        st.markdown(
            f'<div class="or-box">'
            f'<strong>Monte Carlo Insight (Taha Ch. 19):</strong> '
            f'Expected weekly profit is <strong>${mc["mean"]:,.0f}</strong>. '
            f'In a bad week (P5 scenario) you could earn as little as '
            f'<strong>${mc["p5"]:,.0f}</strong> — a ${mc["mean"]-mc["p5"]:,.0f} downside. '
            f'Your safety buffer of <strong>{fin["safety"]:.1f}%</strong> '
            f'{"provides reasonable cushion." if fin["safety"] > 15 else "is thin — reduce fixed costs or raise your lowest-margin prices."}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Suggested price table
        st.markdown("#### Suggested Prices")
        price_rows = []
        for i, r in enumerate(rows):
            p_c  = r["Price ($)"]
            p_o  = round(opt_prices[i] * 4) / 4
            chg  = (p_o - p_c) / p_c * 100
            cm_c = weekly_profit_item(p_c, r)
            cm_o = weekly_profit_item(p_o, r)
            t    = eng_df.iloc[i]["Type"]
            price_rows.append({
                "Dish":               r["Dish"],
                "Type":               f"{QUAD_ICON[t]} {t}",
                "Current Price":      f"${p_c:.2f}",
                "Suggested Price":    f"${p_o:.2f}",
                "Change":             f"{chg:+.1f}%",
                "Current Profit/Wk": f"${cm_c:,.0f}",
                "New Profit/Wk":     f"${cm_o:,.0f}",
                "Extra/Wk":          f"${cm_o-cm_c:+,.0f}",
                "CM/Min Cook":       f"${eng_df.iloc[i]['CM/Min']:.2f}",
            })
        st.dataframe(pd.DataFrame(price_rows), use_container_width=True, hide_index=True)
        st.markdown('<div class="tip">CM/Min Cook = contribution margin per minute of cook time — '
                    'the most efficient dishes for your kitchen capacity (LP efficiency metric).</div>',
                    unsafe_allow_html=True)

        # Interactive price tester
        st.markdown("---")
        st.markdown("#### 🎚️ Interactive Price Tester")
        sel     = st.selectbox("Dish to test", menu_df["Dish"].tolist())
        sel_row = menu_df[menu_df["Dish"] == sel].to_dict("records")[0]
        new_p   = st.slider(
            "Drag to test a price",
            min_value=float(sel_row["Cost ($)"] * 1.05),
            max_value=float(sel_row["Price ($)"] * 2.0),
            value=float(sel_row["Price ($)"]),
            step=0.25,
        )
        new_qty  = demand(new_p, sel_row["Price ($)"], sel_row["Sold / Week"], sel_row["Elasticity"])
        new_cm   = (new_p - sel_row["Cost ($)"]) * new_qty
        base_cm  = (sel_row["Price ($)"] - sel_row["Cost ($)"]) * sel_row["Sold / Week"]
        margin_p = (new_p - sel_row["Cost ($)"]) / new_p * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Est. Orders/Week", f"{new_qty:.0f}",   f"{new_qty-sel_row['Sold / Week']:+.0f}")
        m2.metric("Weekly Profit",    f"${new_cm:,.0f}",  f"${new_cm-base_cm:+,.0f}")
        m3.metric("Extra/Year",       f"${(new_cm-base_cm)*52:+,.0f}")
        m4.metric("Margin",           f"{margin_p:.1f}%")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — AI RECOMMENDATIONS
    # ════════════════════════════════════════════════════════════════════════
    with t3:
        st.markdown('<div class="sh">🤖 AI Recommendations</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="tip">'
            'Claude analyses your LP shadow prices, Monte Carlo forecast, menu engineering matrix, '
            'and optimised pricing in one structured brief. '
            'Set your Anthropic API key in the sidebar to unlock this tab.'
            '</div>',
            unsafe_allow_html=True,
        )

        api_key = api_key_input or os.environ.get("ANTHROPIC_API_KEY", "")

        if not ANTHROPIC_AVAILABLE:
            st.warning("Install the Anthropic package: `pip install anthropic`")
        elif not api_key:
            st.info("Add your Anthropic API key in the sidebar ↖")
        else:
            if st.button("🧠  Generate AI Recommendations"):
                prompt = build_ai_prompt(
                    eng_df, fin, opt_prices, sh_l, sh_b, mc
                )
                with st.spinner("Generating recommendations…"):
                    ai_text = stream_ai_response(prompt, api_key)
                st.session_state.ai_text = ai_text

            elif st.session_state.ai_text:
                st.markdown(
                    f'<div class="ai-box">{st.session_state.ai_text}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Scroll up and click Generate again to refresh.")

        # Show the prompt that will be sent (collapsible)
        with st.expander("🔍 View data sent to Claude"):
            if api_key or not ANTHROPIC_AVAILABLE:
                prompt_preview = build_ai_prompt(eng_df, fin, opt_prices, sh_l, sh_b, mc)
                st.code(prompt_preview, language="markdown")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — REPORTS
    # ════════════════════════════════════════════════════════════════════════
    with t4:
        st.markdown('<div class="sh">📄 Download Report</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="tip">'
            'Generates a full HTML report including OR analysis (LP shadow prices + Monte Carlo), '
            'dish breakdown, suggested prices, and AI recommendations (if generated). '
            'Open in any browser → Ctrl+P → Save as PDF.'
            '</div>',
            unsafe_allow_html=True,
        )

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            period = st.radio("Report period", ["Weekly", "Monthly"], horizontal=True)
        with r_col2:
            report_date = str(st.date_input("Report date", datetime.date.today()))

        # Preview numbers
        factor = 4.33 if period == "Monthly" else 1
        p_lbl  = "month" if period == "Monthly" else "week"
        st.markdown(f"**{period} figures preview:**")
        pv1, pv2, pv3, pv4 = st.columns(4)
        pv1.metric(f"Revenue / {p_lbl}",    f"${fin['rev']*factor:,.0f}")
        pv2.metric(f"Net Profit / {p_lbl}", f"${fin['net']*factor:,.0f}")
        pv3.metric("Annual profit",          f"${fin['annual']:,.0f}")
        pv4.metric("MC 90% range",           f"${mc['p5']:,.0f}–${mc['p95']:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📄  Generate Report"):
            html_report = generate_report(
                menu_df, eng_df, fin, opt_prices, mc,
                sh_l, sh_b, st.session_state.ai_text,
                period, report_date,
            )
            st.download_button(
                label=f"⬇️  Download {period} Report (HTML)",
                data=html_report,
                file_name=f"MenuProfit_{period}_{report_date}.html",
                mime="text/html",
            )
            st.success("Report ready — click the download button above.")

    # Footer
    st.markdown("""
    <div style="text-align:center;color:#2A3050;font-size:.72rem;
                padding:22px 0 6px;border-top:1px solid rgba(255,255,255,.04);margin-top:20px;">
      MenuProfit Pro &nbsp;·&nbsp; LP + Monte Carlo + AI &nbsp;·&nbsp; Operations Research applied
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:50px 16px;color:#3A4260;">
      <div style="font-size:2.4rem;margin-bottom:13px;">🍽️</div>
      <div style="font-size:1rem;line-height:1.8;">
        Edit your menu above, then tap
        <strong style="color:#D4A840;">Analyse Menu & Optimise Prices</strong>.<br>
        The LP solver, Monte Carlo simulation, and AI engine run in seconds.
      </div>
    </div>
    """, unsafe_allow_html=True)
