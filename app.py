"""
MenuProfit Pro
Restaurant profit tool — LP, EOQ, Monte Carlo, Markov, Queuing

Run:
    pip install streamlit pandas numpy plotly scipy openpyxl
    streamlit run app.py
"""

import io, math, datetime
from dataclasses import dataclass
from typing import List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize, differential_evolution, linprog

st.set_page_config(
    page_title="MenuProfit Pro",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');
*,html,body{box-sizing:border-box}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;font-size:14px}
[data-testid="stAppViewContainer"]{background:#070910;color:#DEE2F0}
[data-testid="stSidebar"]{background:#0C0E18!important}
[data-testid="stHeader"],[data-testid="stToolbar"]{display:none!important}
.banner{background:linear-gradient(118deg,#0D1025 0%,#090B15 55%,#0D1025 100%);
  border-bottom:1px solid rgba(210,162,72,.22);padding:18px 28px 14px;
  margin:-1rem -1rem 1.8rem;display:flex;align-items:center;gap:14px}
.b-title{font-family:'Cormorant Garamond',serif;font-size:1.85rem;font-weight:700;color:#fff;line-height:1}
.b-title span{color:#D2A248}
.b-sub{font-size:.76rem;color:#4A5578;margin-top:4px;letter-spacing:.03em}
.b-badges{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
.b-badge{background:rgba(210,162,72,.09);border:1px solid rgba(210,162,72,.22);
  border-radius:6px;padding:4px 11px;font-size:.68rem;color:#D2A248;
  letter-spacing:.1em;font-weight:600;white-space:nowrap}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.025);border-radius:10px;
  padding:4px;gap:2px;border:1px solid rgba(255,255,255,.05)}
.stTabs [data-baseweb="tab"]{border-radius:7px;color:#4A5578!important;
  font-size:.82rem!important;padding:8px 20px!important;font-weight:600!important;letter-spacing:.02em}
.stTabs [aria-selected="true"]{background:rgba(210,162,72,.13)!important;color:#D2A248!important}
.sh{font-family:'Cormorant Garamond',serif;font-size:1.2rem;font-weight:700;color:#D2A248;
  border-bottom:1px solid rgba(210,162,72,.14);padding-bottom:7px;margin:1.6rem 0 .9rem}
.krow{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:1rem}
.kpi{flex:1 1 120px;min-width:108px;background:linear-gradient(148deg,#0F1220,#0B0D18);
  border:1px solid rgba(255,255,255,.07);border-radius:11px;padding:13px 15px;text-align:center}
.kpi-l{font-size:.65rem;color:#3E4A6A;text-transform:uppercase;letter-spacing:.09em;margin-bottom:5px}
.kpi-v{font-size:1.5rem;font-weight:600;line-height:1.1}
.kpi-s{font-size:.65rem;color:#2E3858;margin-top:3px}
.gold{color:#D2A248}.green{color:#35C886}.red{color:#DC4545}.blue{color:#5498F2}.white{color:#C0C8E0}
.ibox{background:#0C1020;border:1px solid rgba(84,152,242,.18);border-radius:9px;
  padding:13px 16px;margin:.4rem 0 1rem;font-size:.81rem;line-height:1.7;color:#7A90B8}
.ibox strong{color:#5498F2}
.rec{border-radius:10px;padding:14px 17px;margin-bottom:10px;background:#0C1020;
  border:1px solid rgba(255,255,255,.06);border-left:3px solid #D2A248}
.rec.crit{border-left-color:#DC4545}
.rec.high{border-left-color:#F0A030}
.rec.med{border-left-color:#35C886}
.rec-top{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap}
.rec-cat{font-size:.65rem;font-weight:700;letter-spacing:.09em;padding:2px 8px;
  border-radius:20px;text-transform:uppercase}
.rec-cat.p{background:rgba(210,162,72,.12);color:#D2A248}
.rec-cat.i{background:rgba(84,152,242,.12);color:#5498F2}
.rec-cat.m{background:rgba(53,200,134,.12);color:#35C886}
.rec-cat.o{background:rgba(155,108,245,.12);color:#9B6CF5}
.rec-cat.b{background:rgba(240,160,48,.12);color:#F0A030}
.rec-dish{font-size:.72rem;color:#4A5578;font-style:italic}
.rec-imp{margin-left:auto;font-size:.76rem;font-weight:700;color:#35C886}
.rec-title{font-weight:600;font-size:.93rem;color:#DEE2F0;margin-bottom:4px}
.rec-insight{font-size:.80rem;color:#6070A0;line-height:1.65;margin-bottom:6px}
.rec-action{font-size:.80rem;color:#D2A248;font-weight:500}
.tip{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.05);border-radius:8px;
  padding:8px 13px;font-size:.77rem;color:#3E4A6A;margin:.3rem 0 1rem}
.stButton>button{background:linear-gradient(135deg,#9B7018,#D2A248)!important;color:#07090E!important;
  font-weight:700!important;border:none!important;border-radius:9px!important;
  padding:10px 22px!important;font-size:.88rem!important;width:100%!important;
  letter-spacing:.02em!important;transition:opacity .15s}
.stButton>button:hover{opacity:.88!important}
.psummary{display:flex;gap:10px;margin-bottom:1.2rem;flex-wrap:wrap}
.pchip{flex:1 1 120px;text-align:center;border-radius:9px;padding:10px 14px}
.pchip.crit{background:rgba(220,69,69,.1);border:1px solid rgba(220,69,69,.25)}
.pchip.high{background:rgba(240,160,48,.1);border:1px solid rgba(240,160,48,.25)}
.pchip.med{background:rgba(53,200,134,.1);border:1px solid rgba(53,200,134,.25)}
.pnum{font-size:1.6rem;font-weight:700}
.pnum.crit{color:#DC4545}.pnum.high{color:#F0A030}.pnum.med{color:#35C886}
.plbl{font-size:.68rem;letter-spacing:.07em;text-transform:uppercase;color:#4A5578;margin-top:2px}
@media(max-width:640px){.b-title{font-size:1.4rem}.kpi-v{font-size:1.25rem}
  .stTabs [data-baseweb="tab"]{font-size:.74rem!important;padding:6px 9px!important}}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#7A8AB0"),
    margin=dict(l=14, r=14, t=38, b=14),
)
# Disable click-to-expand on all charts
NO_EXPAND = dict(
    dragmode=False,
)
GOLD="#D2A248"; GREEN="#35C886"; RED="#DC4545"; BLUE="#5498F2"; PURPLE="#9B6CF5"; AMBER="#F0A030"
QUAD_C={"Star":GOLD,"Puzzle":BLUE,"Plow Horse":GREEN,"Dog":RED}
QUAD_I={"Star":"⭐","Puzzle":"🧩","Plow Horse":"🐎","Dog":"🐕"}

CHART_CFG = dict(
    config={
        "displayModeBar": False,
        "scrollZoom": False,
        "doubleClick": False,
        "showTips": False,
    }
)

# ── Default data ──────────────────────────────────────────────────────────────
DEFAULT = {
    "Dish":             ["Wagyu Burger","Truffle Pasta","Caesar Salad","Espresso Martini",
                         "Lava Cake","House Wine","Grilled Salmon","Garlic Bread"],
    "Category":         ["Mains","Mains","Starters","Drinks","Desserts","Drinks","Mains","Starters"],
    "Ingredient ($)":   [8.50, 6.20, 2.80, 1.90, 2.40, 3.20, 9.80, 0.80],
    "Price ($)":        [24.00,22.00,14.00,16.00,12.00,11.00,28.00, 7.00],
    "Sold/Wk":          [120,   85,  200,  180,   90,  250,   70, 160],
    "Prep (min)":       [  8,   10,    2,    1,    5,    0,   10,   2],
    "Cook (min)":       [ 12,   15,    5,    3,    8,    2,   18,   4],
    "Table Time (min)": [ 35,   40,   15,   20,   25,   10,   45,  10],
    "Wastage %":        [  8,   12,    5,    3,    6,    2,   10,   4],
    "Labour $/hr":      [ 30,   30,   30,   25,   30,   25,   32,  28],
    "Elasticity":       [-1.2,-1.0, -0.8, -0.6, -1.1, -0.9, -1.3,-0.7],
    "Stock":            [120,   80,  300,  200,  100,  400,   60, 250],
    "Reorder Pt":       [ 40,   30,  100,   60,   30,  100,   20,  80],
    "Lead Days":        [  2,    3,    1,    2,    2,    3,    2,   1],
    "Order Cost ($)":   [ 15,   20,   10,   12,   10,   18,   25,   8],
    "Hold Cost %/yr":   [ 25,   20,   30,   15,   30,   15,   25,  20],
    "Rating":           [4.7,  4.4,  4.2,  4.6,  4.8,  3.9,  4.5, 4.1],
    "Bundle Attach %":  [ 35,   22,   18,   40,   25,   30,   20,  45],
}

# ═════════════════════════════════════════════════════════════════════════════
#  MATH ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def true_cost(row):
    """Real cost = ingredient cost after waste + staff time cost."""
    food   = row["Ingredient ($)"] * (1 + row["Wastage %"] / 100)
    labour = (row["Prep (min)"] + row["Cook (min)"]) / 60 * row["Labour $/hr"]
    return round(food + labour, 2)

def true_margin_pct(price, tc):
    return (price - tc) / price * 100 if price > 0 else 0

def profit_per_min(price, tc, prep, cook):
    t = prep + cook
    return (price - tc) / t if t > 0 else 0

def demand(p_new, p_base, qty_base, elast):
    """How many we sell at a new price, based on price sensitivity."""
    if p_new <= 0: return 0.0
    return float(qty_base) * (p_new / float(p_base)) ** float(elast)

def weekly_profit_item(price, row):
    tc  = true_cost(row)
    qty = demand(price, row["Price ($)"], row["Sold/Wk"], row["Elasticity"])
    return (price - tc) * qty

def total_net_profit(prices, rows, fixed):
    return sum(weekly_profit_item(prices[i], rows[i]) for i in range(len(rows))) - fixed

def run_lp(df, kitchen_mins, ing_budget):
    """
    Find the best mix of dishes to serve given kitchen time and budget limits.
    Shadow prices tell us how much more profit we get by relaxing each limit.
    """
    tc      = df.apply(true_cost, axis=1).values
    cm      = (df["Price ($)"].values - tc).clip(min=0)
    total_t = (df["Prep (min)"] + df["Cook (min)"]).values
    table_t = df["Table Time (min)"].values
    ing_c   = df["Ingredient ($)"].values
    qtys    = df["Sold/Wk"].values.astype(float)

    A_ub = [total_t, ing_c, table_t]
    b_ub = [kitchen_mins, ing_budget, kitchen_mins * 2]

    res = linprog(-cm, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0.0, q) for q in qtys], method="highs")

    sh_k = sh_b = sh_s = 0.0
    if res.status == 0 and hasattr(res, "ineqlin") and res.ineqlin is not None:
        m = res.ineqlin.marginals
        if m is not None and len(m) >= 3:
            sh_k, sh_b, sh_s = abs(m[0]), abs(m[1]), abs(m[2])

    return res, sh_k, sh_b, sh_s

def optimise_prices(rows, fixed, min_mg=0.28, max_up=0.35):
    """Find the best prices for each dish to earn the most weekly profit."""
    tcs    = [true_cost(r) for r in rows]
    bounds = [
        (max(r["Price ($)"] * 0.85, tc * (1 + min_mg)),
         r["Price ($)"] * (1 + max_up))
        for r, tc in zip(rows, tcs)
    ]
    p0 = np.array([r["Price ($)"] for r in rows])
    r1 = minimize(lambda p: -total_net_profit(p, rows, fixed), p0,
                  bounds=bounds, method="L-BFGS-B",
                  options={"ftol": 1e-12, "maxiter": 2000})
    r2 = differential_evolution(lambda p: -total_net_profit(p, rows, fixed),
                                bounds, seed=42, maxiter=300, popsize=14, tol=1e-9)
    return (r2.x, -r2.fun) if -r2.fun >= -r1.fun else (r1.x, -r1.fun)

def eoq_model(row, demand_sigma_pct=0.15, service_z=1.645):
    """
    Work out the best order size so you never run out but don't over-order.
    EOQ = the amount to order each time to keep costs lowest.
    """
    D  = row["Sold/Wk"] * 52
    S  = row["Order Cost ($)"]
    H  = row["Ingredient ($)"] * row["Hold Cost %/yr"] / 100
    if H <= 0: H = 0.01

    eoq_qty    = math.sqrt(2 * D * S / H)
    orders_pa  = D / eoq_qty
    cycle_days = 365 / orders_pa

    sigma_wk = row["Sold/Wk"] * demand_sigma_pct
    lt_weeks = row["Lead Days"] / 7
    ss       = service_z * sigma_wk * math.sqrt(lt_weeks)
    rop      = (row["Sold/Wk"] / 7 * row["Lead Days"]) + ss

    hold_curr  = (row["Reorder Pt"] / 2) * H
    hold_eoq   = (eoq_qty / 2) * H
    orders_eoq = (D / eoq_qty) * S
    saving_pa  = (hold_curr + orders_pa * S) - (hold_eoq + orders_eoq)

    return {
        "eoq":          round(eoq_qty, 0),
        "orders_pa":    round(orders_pa, 1),
        "cycle_days":   round(cycle_days, 1),
        "safety_stock": round(ss, 0),
        "rop":          round(rop, 0),
        "annual_saving":round(saving_pa, 2),
        "stockout":     row["Stock"] < rop,
        "overstock":    row["Stock"] > eoq_qty * 2,
    }

def classify_menu(df):
    df = df.copy()
    df["True Cost ($)"]      = df.apply(true_cost, axis=1)
    df["True Profit ($)"]    = df["Price ($)"] - df["True Cost ($)"]
    df["True Margin %"]      = df.apply(lambda r: true_margin_pct(r["Price ($)"], r["True Cost ($)"]), axis=1)
    df["Profit/Min ($)"]     = df.apply(
        lambda r: profit_per_min(r["Price ($)"], r["True Cost ($)"], r["Prep (min)"], r["Cook (min)"]), axis=1)
    df["Wk True Profit ($)"] = df["True Profit ($)"] * df["Sold/Wk"]
    df["Food Cost %"]        = (df["True Cost ($)"] / df["Price ($)"] * 100).round(1)
    df["Bundle Sales/Wk"]    = (df["Sold/Wk"] * df["Bundle Attach %"] / 100).round(0).astype(int)

    avg_p = df["True Profit ($)"].mean()
    avg_q = df["Sold/Wk"].mean()

    def quad(r):
        hi_p = r["True Profit ($)"] >= avg_p
        hi_q = r["Sold/Wk"] >= avg_q
        if hi_p and hi_q: return "Star"
        if hi_p:          return "Puzzle"
        if hi_q:          return "Plow Horse"
        return "Dog"

    df["Type"] = df.apply(quad, axis=1)
    return df, avg_p, avg_q

def monte_carlo(df, fixed=0, n=8000, sigma=0.15):
    """
    Run 8,000 random demand scenarios to see the range of possible profits.
    P5 = a bad week. P50 = a normal week. P95 = a great week.
    """
    tc   = df.apply(true_cost, axis=1).values
    cm   = (df["Price ($)"].values - tc).clip(min=0)
    qtys = df["Sold/Wk"].values.astype(float)

    rng  = np.random.default_rng(42)
    sim  = rng.normal(qtys, qtys * sigma, (n, len(qtys))).clip(0)
    prof = (sim * cm).sum(axis=1) - fixed

    return {
        "mean": prof.mean(),
        "p5":   np.percentile(prof,  5),
        "p25":  np.percentile(prof, 25),
        "p50":  np.percentile(prof, 50),
        "p75":  np.percentile(prof, 75),
        "p95":  np.percentile(prof, 95),
        "sims": prof,
        "std":  prof.std(),
    }

def kitchen_analysis(df, available_mins):
    """How busy is the kitchen and which dish takes the most time."""
    total_t    = (df["Prep (min)"] + df["Cook (min)"]) * df["Sold/Wk"]
    load       = total_t.sum()
    util       = load / available_mins if available_mins > 0 else 0
    tc_vals    = df.apply(true_cost, axis=1).values
    cook_time  = df["Prep (min)"].values + df["Cook (min)"].values + 0.01
    pm         = (df["Price ($)"].values - tc_vals) / cook_time
    bottleneck = df.iloc[total_t.values.argmax()]["Dish"]

    return {
        "load_mins":     load,
        "util":          util,
        "util_pct":      min(util * 100, 100),
        "seating_load":  (df["Table Time (min)"] * df["Sold/Wk"]).sum(),
        "bottleneck":    bottleneck,
        "profit_per_min":np.array(pm),
        "dish_loads":    total_t.values,
    }

def markov_ltv(rating, avg_weekly_contribution, weekly_visits=1.0, discount_rate_annual=0.10):
    """
    Estimate how much a happy customer is worth over their lifetime.
    Higher rating = more likely to come back = higher value.
    """
    p_retain = min(0.96, max(0.30, 0.46 + (rating - 1.0) * 0.125))
    r_weekly = discount_rate_annual / 52
    visits   = p_retain / (1 - p_retain + r_weekly)
    clv      = avg_weekly_contribution * visits * weekly_visits
    return round(clv, 0), round(p_retain * 100, 1)

def calc_fin(df, eng_df, fixed):
    pool    = eng_df["Wk True Profit ($)"].sum()
    rev     = (df["Price ($)"].values * df["Sold/Wk"].values).sum()
    covers  = df["Sold/Wk"].sum()
    net     = pool - fixed
    avg_cm  = pool / covers if covers else 0
    bep     = fixed / avg_cm if avg_cm > 0 else float("inf")
    surplus = covers - bep
    safety  = net / pool * 100 if pool > 0 else 0
    fc_pct  = eng_df["Food Cost %"].mean()
    ing_only= (df["Ingredient ($)"].values * df["Sold/Wk"].values).sum() / rev * 100 if rev > 0 else 0
    return dict(
        pool=pool, rev=rev, net=net, monthly=net*4.33, annual=net*52,
        covers=covers, avg_cm=avg_cm, bep=bep, surplus=surplus,
        safety=safety, fc_pct=fc_pct, ing_pct=ing_only, fixed=fixed,
    )

# ═════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Rec:
    cat:      str
    priority: int
    dish:     str
    title:    str
    insight:  str
    action:   str
    impact:   float

def generate_recommendations(df, eng_df, fin, opt_prices, sh_k, sh_b, sh_s,
                              mc, eoq_data, kitchen, fixed,
                              available_kitchen_mins) -> List[Rec]:
    rows = df.to_dict("records")
    recs: List[Rec] = []

    # ── PRICING ──────────────────────────────────────────────────────────
    for i, row in enumerate(rows):
        p_c    = row["Price ($)"]
        p_o    = round(opt_prices[i] * 4) / 4
        tc     = true_cost(row)
        margin = true_margin_pct(p_c, tc)
        fc_p   = eng_df.iloc[i]["Food Cost %"]
        cm_c   = weekly_profit_item(p_c, row)
        cm_o   = weekly_profit_item(p_o, row)
        uplift = cm_o - cm_c
        pct_chg= (p_o - p_c) / p_c * 100

        if margin < 20:
            recs.append(Rec("Pricing", 1, row["Dish"],
                f"You are losing money on this dish — profit is only {margin:.1f}%",
                f"After counting {row['Wastage %']}% food waste and "
                f"{(row['Prep (min)']+row['Cook (min)']):.0f} min of staff time, "
                f"the real cost is ${tc:.2f}. Your margin is dangerously low.",
                f"Raise the price to at least ${tc * 1.30:.2f} to reach a safe profit level.",
                max(0, uplift) + 20))

        elif pct_chg > 3 and uplift > 15:
            pri = 1 if uplift > 80 else (2 if uplift > 30 else 3)
            demand_drop = abs((demand(p_o, p_c, row["Sold/Wk"], row["Elasticity"]) - row["Sold/Wk"]) /
                              row["Sold/Wk"] * 100)
            recs.append(Rec("Pricing", pri, row["Dish"],
                f"Raise price from ${p_c:.2f} to ${p_o:.2f} — earn +${uplift:.0f} more per week",
                f"Customers are not very price-sensitive on this dish. "
                f"Demand would only drop by about {demand_drop:.1f}% but each sale earns more. "
                f"The net result is more profit per week.",
                f"Change the menu price to ${p_o:.2f} (+{pct_chg:.1f}%). Expected gain: +${uplift:.0f}/week.",
                uplift))

        if fc_p > 38:
            fix_price = tc / 0.60
            recs.append(Rec("Pricing", 1, row["Dish"],
                f"Food cost is too high at {fc_p:.0f}% — target is under 32%",
                f"The real cost of this dish (including {row['Wastage %']}% waste) is "
                f"{fc_p:.0f}% of what you charge. That leaves very little room for profit.",
                f"Either raise the price to ${fix_price:.2f} or cut ingredient cost by "
                f"${tc - row['Price ($)'] * 0.38:.2f} per dish.",
                (fix_price - p_c) * row["Sold/Wk"]))

        if row["Rating"] >= 4.6 and pct_chg < 2:
            recs.append(Rec("Pricing", 3, row["Dish"],
                f"Customers love this dish (rating {row['Rating']}) — small price rise is safe",
                f"High-rated dishes keep customers coming back. "
                f"A small price increase is unlikely to put people off.",
                f"Try adding $0.75. At current demand that adds +${0.75*row['Sold/Wk']:.0f}/week.",
                0.75 * row["Sold/Wk"]))

    # ── INVENTORY ─────────────────────────────────────────────────────────
    for i, row in enumerate(rows):
        eq = eoq_data[i]

        if eq["stockout"]:
            recs.append(Rec("Inventory", 1, row["Dish"],
                f"Risk of running out — stock ({int(row['Stock'])}) is below safe level ({int(eq['rop'])})",
                f"With {row['Lead Days']} days to get a delivery and selling {row['Sold/Wk']} per week, "
                f"you need at least {eq['rop']:.0f} units in stock before you order more. "
                f"You only have {int(row['Stock'])} — order now.",
                f"Order {int(eq['eoq'])} units right away. Next order in {eq['cycle_days']:.0f} days.",
                row["Price ($)"] * row["Sold/Wk"] * (row["Lead Days"] / 7)))

        elif eq["overstock"]:
            hold_excess = (row["Stock"] - eq["eoq"] * 1.2) * row["Ingredient ($)"] * row["Hold Cost %/yr"] / 100 / 52
            recs.append(Rec("Inventory", 3, row["Dish"],
                f"Too much stock — you are paying to store it",
                f"You have {int(row['Stock'])} units but the ideal amount to order is only "
                f"{int(eq['eoq'])}. Extra stock costs money to store every week.",
                f"Order less next time. Aim for {int(eq['eoq'])} units every {eq['cycle_days']:.0f} days.",
                hold_excess))

        if eq["annual_saving"] > 50:
            recs.append(Rec("Inventory", 2, row["Dish"],
                f"Change your order pattern — save ${eq['annual_saving']/52:.0f}/week",
                f"The ideal order size for this dish is {int(eq['eoq'])} units. "
                f"Ordering this amount each time balances delivery costs vs storage costs, "
                f"saving ${eq['annual_saving']:.0f} per year.",
                f"Place a standing order for {int(eq['eoq'])} units every {eq['cycle_days']:.0f} days.",
                eq["annual_saving"] / 52))

    # ── MENU ENGINEERING ──────────────────────────────────────────────────
    for i, (_, row) in enumerate(eng_df.iterrows()):
        t = row["Type"]

        if t == "Dog":
            kitchen_freed = (row["Prep (min)"] + row["Cook (min)"]) * row["Sold/Wk"]
            opportunity   = kitchen_freed * sh_k
            recs.append(Rec("Menu", 1, row["Dish"],
                f"This dish has low profit AND low sales — consider removing it",
                f"{row['Dish']} earns only ${row['True Profit ($)']:.2f} per sale and sells just "
                f"{row['Sold/Wk']:.0f} times a week — both below average. "
                f"Kitchen time freed up could go to your better dishes.",
                f"Remove this dish or replace it. Freeing the kitchen time could be worth ~${opportunity:.0f}/week.",
                opportunity))

        elif t == "Puzzle":
            target_qty   = eng_df["Sold/Wk"].mean() * 0.9
            revenue_lift = max(0, (target_qty - row["Sold/Wk"]) * row["True Profit ($)"])
            recs.append(Rec("Menu", 2, row["Dish"],
                f"Good profit but not enough people order it — promote it",
                f"{row['Dish']} earns ${row['True Profit ($)']:.2f} per sale (above average) "
                f"but only sells {row['Sold/Wk']:.0f} times a week. "
                f"If it reached normal sales levels, you'd earn ${revenue_lift:.0f} more per week.",
                f"Put it on the specials board, add a photo, or ask staff to mention it. "
                f"Aim for {max(0, target_qty-row['Sold/Wk']):.0f} more orders per week.",
                revenue_lift))

        elif t == "Plow Horse":
            p_test = row["Price ($)"] * 1.07
            tc_val = row["True Cost ($)"]
            extra  = (p_test - tc_val - (row["Price ($)"] - tc_val)) * row["Sold/Wk"] * 0.92
            if extra > 0:
                recs.append(Rec("Menu", 2, row["Dish"],
                    f"Very popular but low profit — a small price rise will help",
                    f"{row['Dish']} sells {row['Sold/Wk']:.0f} times a week (above average) "
                    f"but the profit per sale is only {row['True Margin %']:.1f}%. "
                    f"A 7% price rise would only reduce orders by a small amount.",
                    f"Raise the price by $0.50–$1.00. At ${row['Price ($)']+0.75:.2f} you'd earn +${extra:.0f}/week.",
                    extra))

        elif t == "Star":
            premium_p = row["Price ($)"] * 1.10
            tc_val    = row["True Cost ($)"]
            extra     = (premium_p - tc_val - (row["Price ($)"] - tc_val)) * row["Sold/Wk"] * 0.95
            clv_val   = markov_ltv(row["Rating"], row["True Profit ($)"])[0]
            recs.append(Rec("Menu", 3, row["Dish"],
                f"Your best dish — keep quality high and consider a premium version",
                f"{row['Dish']} is top in both profit and popularity. "
                f"Customers who enjoy it are worth an estimated ${clv_val:,.0f} in repeat visits.",
                f"Consider a 'Premium {row['Dish']}' option at ${premium_p:.2f}. "
                f"If 20% of buyers upgrade, that adds +${extra*0.2:.0f}/week.",
                extra * 0.2))

    # ── OPERATIONS ────────────────────────────────────────────────────────
    util_pct = kitchen["util_pct"]

    if util_pct > 85:
        recs.append(Rec("Operations", 1, "Kitchen",
            f"Kitchen is at {util_pct:.0f}% capacity — things will slow down soon",
            f"When a kitchen goes above 80% busy, wait times grow fast and quality drops. "
            f"At {util_pct:.0f}% you are already at risk of delays and unhappy customers.",
            f"Remove your two least profitable dishes, or add a staff member to handle more orders. "
            f"Each extra minute of kitchen time is worth about ${sh_k:.2f} in profit.",
            sh_k * 40))

    elif util_pct < 50:
        idle_mins = max(0, available_kitchen_mins * 0.6 - kitchen["load_mins"])
        recs.append(Rec("Operations", 3, "Kitchen",
            f"Kitchen is only {util_pct:.0f}% busy — you have room to grow",
            f"Your kitchen has spare time. You could be earning more without adding staff.",
            f"Add 2–3 new dishes with high profit per minute to fill the spare kitchen time.",
            sh_k * idle_mins / 60 * 0.5))

    bn = kitchen["bottleneck"]
    recs.append(Rec("Operations", 2, bn,
        f"'{bn}' takes the most kitchen time each week",
        f"This dish uses up more kitchen minutes than any other. "
        f"Even saving 2 minutes per order would free up {eng_df.shape[0] * 2:.0f}+ minutes a week.",
        f"Prepare parts of {bn} in advance during quiet times. Aim to save 2–3 min per order.",
        sh_k * eng_df.shape[0] * 2))

    pm_thresh = eng_df["Profit/Min ($)"].quantile(0.25)
    for _, row in eng_df[eng_df["Profit/Min ($)"] < pm_thresh].iterrows():
        if row["Dish"] == bn: continue
        recs.append(Rec("Operations", 3, row["Dish"],
            f"Takes too long to make for what it earns — ${row['Profit/Min ($)']:.2f}/min",
            f"{row['Dish']} earns ${row['Profit/Min ($)']:.2f} per minute of kitchen time. "
            f"Other dishes earn much more. This dish is slowing down your kitchen.",
            f"Simplify the recipe or pre-prepare parts to cut cook time by 20%. "
            f"That would free up about ${sh_k*(row['Prep (min)']+row['Cook (min)'])*0.2*row['Sold/Wk']:.0f}/week.",
            sh_k * (row["Prep (min)"] + row["Cook (min)"]) * 0.15 * row["Sold/Wk"]))

    # ── BUNDLE ────────────────────────────────────────────────────────────
    for _, row in eng_df.iterrows():
        if row["Bundle Attach %"] >= 30:
            est_extra = row["Bundle Sales/Wk"] * row["True Profit ($)"] * 0.4
            recs.append(Rec("Bundle", 3, row["Dish"],
                f"People often order extras with this dish — make it official",
                f"{int(row['Bundle Attach %'])}% of orders of {row['Dish']} already include an add-on. "
                f"A proper combo deal on your menu or till could push that higher.",
                f"Create a '{row['Dish']} + [side/drink]' deal at a $1.50 combo price. "
                f"Estimated extra: ${est_extra:.0f}/week.",
                est_extra))

    # ── FINANCIAL HEALTH ─────────────────────────────────────────────────
    if fin["safety"] < 15:
        recs.append(Rec("Operations", 1, "All Dishes",
            f"You are very close to breaking even — a bad week could put you in the red",
            f"You need {fin['bep']:.0f} orders per week to cover all your costs. "
            f"You are getting {fin['covers']:.0f}. In a bad week your profit could drop to "
            f"${mc['p5']:,.0f} — below zero.",
            f"Raise prices on your popular low-margin dishes by $0.50 each "
            f"and think about removing your weakest dish.",
            abs(fin["net"] * 0.15)))

    if fin["fc_pct"] > 35:
        recs.append(Rec("Pricing", 2, "All Dishes",
            f"Your average food cost is {fin['fc_pct']:.1f}% — target is under 32%",
            f"Across all dishes, the real cost of food (including waste and staff time) "
            f"is too high. Every extra percent above 32% costs you ~${fin['rev'] * 0.01:,.0f}/week.",
            f"Fix your top 3 most expensive dishes first. "
            f"Getting to 32% would add ${(fin['fc_pct'] - 32) * fin['rev'] / 100:,.0f}/week.",
            (fin["fc_pct"] - 32) * fin["rev"] / 100 if fin["fc_pct"] > 32 else 0))

    return sorted(recs, key=lambda r: (r.priority, -r.impact))


# ═════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def kcard(lbl, val, sub="", cls="gold"):
    return (f'<div class="kpi"><div class="kpi-l">{lbl}</div>'
            f'<div class="kpi-v {cls}">{val}</div>'
            f'<div class="kpi-s">{sub}</div></div>')

def _no_zoom(fig, height):
    fig.update_layout(**DARK, **NO_EXPAND, height=height)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig

def fig_bcg(eng_df, avg_p, avg_q):
    fig = go.Figure()
    for t, grp in eng_df.groupby("Type"):
        fig.add_trace(go.Scatter(
            x=grp["Sold/Wk"], y=grp["True Profit ($)"],
            mode="markers+text", name=f"{QUAD_I[t]} {t}",
            text=grp["Dish"], textposition="top center",
            textfont=dict(size=9, color="#8A9AC8"),
            marker=dict(size=16, color=QUAD_C[t], opacity=.88,
                        line=dict(width=1.5, color="rgba(255,255,255,.1)"))))
    fig.add_hline(y=avg_p, line_dash="dot", line_color="rgba(255,255,255,.1)")
    fig.add_vline(x=avg_q, line_dash="dot", line_color="rgba(255,255,255,.1)")
    for lbl, ax, ay, xa, ya in [
        ("⭐ STARS",       eng_df["Sold/Wk"].max(), eng_df["True Profit ($)"].max(), "right","top"),
        ("🧩 PUZZLES",     eng_df["Sold/Wk"].min()*.88, eng_df["True Profit ($)"].max(), "left","top"),
        ("🐎 PLOW HORSES", eng_df["Sold/Wk"].max(), eng_df["True Profit ($)"].min()*.88, "right","bottom"),
        ("🐕 DOGS",        eng_df["Sold/Wk"].min()*.88, eng_df["True Profit ($)"].min()*.88, "left","bottom"),
    ]:
        fig.add_annotation(x=ax, y=ay, text=lbl, showarrow=False,
                           font=dict(size=8, color="rgba(255,255,255,.14)"),
                           xanchor=xa, yanchor=ya)
    fig.update_layout(title="Menu Performance Map — profit vs. how many you sell",
                      xaxis_title="Orders per week", yaxis_title="Profit per dish ($)")
    return _no_zoom(fig, 380)

def fig_profit_min(eng_df):
    sdf = eng_df.sort_values("Profit/Min ($)", ascending=False)
    fig = go.Figure(go.Bar(
        x=sdf["Dish"], y=sdf["Profit/Min ($)"],
        marker=dict(color=sdf["Profit/Min ($)"].tolist(),
                    colorscale=[[0,RED],[0.4,AMBER],[1,GREEN]],
                    showscale=False, line=dict(width=0)),
        text=[f"${v:.2f}" for v in sdf["Profit/Min ($)"]],
        textposition="outside", textfont=dict(size=10, color="#8A9AC8")))
    fig.update_layout(title="Profit earned per minute of kitchen time")
    return _no_zoom(fig, 260)

def fig_pl_waterfall(fin):
    fig = go.Figure(go.Waterfall(
        orientation="v",
        x=["Total Sales","− Food Cost","− Waste & Labour","− Fixed Costs","Net Profit"],
        y=[fin["rev"],
           -(fin["rev"] * fin["ing_pct"] / 100),
           -(fin["pool"] - fin["rev"] * (1 - fin["ing_pct"]/100)),
           -fin["fixed"], fin["net"]],
        connector={"line":{"color":"rgba(255,255,255,.05)"}},
        increasing={"marker":{"color":GREEN}},
        decreasing={"marker":{"color":RED}},
        totals={"marker":{"color":GOLD}},
        text=[f"${abs(v):,.0f}" for v in [
            fin["rev"],
            -(fin["rev"]*fin["ing_pct"]/100),
            -(fin["pool"]-fin["rev"]*(1-fin["ing_pct"]/100)),
            -fin["fixed"], fin["net"]]],
        textposition="outside", textfont=dict(size=10, color="#8A9AC8")))
    fig.update_layout(title="Where every dollar goes this week")
    return _no_zoom(fig, 280)

def fig_mc(mc):
    bins = np.linspace(mc["sims"].min(), mc["sims"].max(), 65)
    cnts, edges = np.histogram(mc["sims"], bins=bins)
    mid = (edges[:-1]+edges[1:])/2
    fig = go.Figure(go.Bar(x=mid, y=cnts,
                           marker=dict(color=mid,
                                       colorscale=[[0,RED],[.35,AMBER],[.65,BLUE],[1,GREEN]],
                                       showscale=False, line=dict(width=0)),
                           opacity=.85))
    for v, lbl, c in [
        (mc["p5"],  "Bad week",    RED),
        (mc["p50"], "Normal week", GOLD),
        (mc["p95"], "Great week",  GREEN),
    ]:
        fig.add_vline(x=v, line_color=c, line_dash="dash", line_width=1.5,
                      annotation_text=f"{lbl}  ${v:,.0f}",
                      annotation_font=dict(color=c, size=9))
    fig.update_layout(title="What your weekly profit could look like — 8,000 scenarios",
                      xaxis_title="Weekly net profit ($)", yaxis_title="How often")
    return _no_zoom(fig, 260)

def fig_shadow(sh_k, sh_b, sh_s):
    fig = go.Figure(go.Bar(
        x=["1 extra kitchen minute", "1 extra $1 food budget", "1 extra table minute"],
        y=[sh_k, sh_b, sh_s],
        marker=dict(color=[BLUE, GOLD, PURPLE], line=dict(width=0)),
        text=[f"${sh_k:.3f}", f"${sh_b:.4f}", f"${sh_s:.3f}"],
        textposition="outside", textfont=dict(size=11, color="#8A9AC8")))
    fig.update_layout(title="How much extra profit you gain by relaxing each limit",
                      yaxis_title="Extra weekly profit ($)")
    return _no_zoom(fig, 220)

def fig_eoq_table(eng_df, eoq_data):
    rows_d = []
    for i, (_, row) in enumerate(eng_df.iterrows()):
        eq = eoq_data[i]
        status = "🔴 Order Now" if eq["stockout"] else ("🟡 Too Much Stock" if eq["overstock"] else "✅ OK")
        rows_d.append({
            "Dish":               row["Dish"],
            "Current Stock":      int(row["Stock"]),
            "Best Order Size":    int(eq["eoq"]),
            "Safety Buffer":      int(eq["safety_stock"]),
            "Order When Below":   int(eq["rop"]),
            "Order Every (days)": eq["cycle_days"],
            "Annual Saving ($)":  f"${eq['annual_saving']:,.0f}",
            "Status":             status,
        })
    return pd.DataFrame(rows_d)

def fig_util_gauge(util_pct):
    color = GREEN if util_pct < 70 else (AMBER if util_pct < 85 else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=util_pct,
        title={"text":"Kitchen Busy %","font":{"size":11,"color":"#8A9AC8"}},
        number={"suffix":"%","font":{"size":20,"color":color}},
        gauge=dict(
            axis=dict(range=[0,100], tickcolor="#222"),
            bar=dict(color=color, thickness=0.22),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[
                {"range":[0,70],  "color":"rgba(53,200,134,.08)"},
                {"range":[70,85], "color":"rgba(240,160,48,.08)"},
                {"range":[85,100],"color":"rgba(220,69,69,.08)"},
            ],
            threshold=dict(line=dict(color=color,width=2),thickness=.75,value=util_pct))))
    return _no_zoom(fig, 190)

def fig_food_cost_bars(eng_df):
    sdf = eng_df.sort_values("Food Cost %", ascending=False)
    fig = go.Figure(go.Bar(
        x=sdf["Dish"], y=sdf["Food Cost %"],
        marker=dict(color=sdf["Food Cost %"].tolist(),
                    colorscale=[[0,GREEN],[.4,AMBER],[1,RED]],
                    showscale=False, line=dict(width=0)),
        text=[f"{v:.0f}%" for v in sdf["Food Cost %"]],
        textposition="outside", textfont=dict(size=10, color="#8A9AC8")))
    fig.add_hline(y=32, line_dash="dash", line_color=GOLD,
                  annotation_text="Target 32%", annotation_font_color=GOLD)
    fig.update_layout(title="Real food cost % per dish (ingredients + waste + staff time)")
    return _no_zoom(fig, 250)

def fig_price_compare(rows, opt_prices, eng_df):
    names = [r["Dish"] for r in rows]
    curr  = [r["Price ($)"] for r in rows]
    opt   = [round(p * 4)/4 for p in opt_prices]
    tcs   = [true_cost(r) for r in rows]
    fig   = go.Figure()
    fig.add_trace(go.Bar(name="Real Cost",        x=names, y=tcs,  marker_color=RED,  opacity=.65))
    fig.add_trace(go.Bar(name="Current Price",    x=names, y=curr, marker_color=BLUE, opacity=.75))
    fig.add_trace(go.Bar(name="Suggested Price",  x=names, y=opt,  marker_color=GOLD, opacity=.9))
    fig.update_layout(barmode="group", title="Real cost vs. what you charge vs. what you should charge",
                      yaxis_title="$ per dish")
    return _no_zoom(fig, 280)

def fig_clv(eng_df):
    clvs = []
    for _, row in eng_df.iterrows():
        clv, ret = markov_ltv(row["Rating"], row["True Profit ($)"])
        clvs.append({"Dish": row["Dish"], "CLV ($)": clv, "Repeat Visit %": ret})
    cdf = pd.DataFrame(clvs).sort_values("CLV ($)", ascending=False)
    fig = go.Figure(go.Bar(
        x=cdf["Dish"], y=cdf["CLV ($)"],
        marker=dict(color=cdf["CLV ($)"].tolist(),
                    colorscale=[[0,RED],[.5,AMBER],[1,GREEN]],
                    showscale=False, line=dict(width=0)),
        text=[f"${v:,.0f}" for v in cdf["CLV ($)"]],
        textposition="outside", textfont=dict(size=10, color="#8A9AC8")))
    fig.update_layout(title="How much a happy customer is worth over their lifetime",
                      yaxis_title="Estimated customer value ($)")
    return _no_zoom(fig, 250)

# ═════════════════════════════════════════════════════════════════════════════
#  HTML REPORT
# ═════════════════════════════════════════════════════════════════════════════

def generate_report(df, eng_df, fin, opt_prices, mc, sh_k, sh_b, sh_s,
                    eoq_data, kitchen, recs, period, rdate):
    factor = 4.33 if period == "Monthly" else 1
    plab   = "Month" if period == "Monthly" else "Week"
    rows   = df.to_dict("records")
    curr_p = total_net_profit([r["Price ($)"] for r in rows], rows, fin["fixed"])
    opt_p  = total_net_profit(opt_prices, rows, fin["fixed"])
    gain   = opt_p - curr_p

    dish_rows = ""
    for i, (_, row) in enumerate(eng_df.iterrows()):
        p_opt = round(opt_prices[i] * 4) / 4
        chg   = (p_opt - row["Price ($)"]) / row["Price ($)"] * 100
        dish_rows += f"""<tr>
          <td><b>{row['Dish']}</b></td><td>{row['Category']}</td>
          <td>${row['Ingredient ($)']:.2f}</td>
          <td style="color:#E06060">${row['True Cost ($)']:.2f}</td>
          <td>${row['Price ($)']:.2f}</td>
          <td style="color:{'#E06060' if chg>0.5 else '#AAA'}">${p_opt:.2f} ({chg:+.1f}%)</td>
          <td>${row['True Profit ($)']:.2f} ({row['True Margin %']:.0f}%)</td>
          <td>${row['Profit/Min ($)']:.2f}</td>
          <td>{int(row['Sold/Wk'])}</td>
          <td>${row['Wk True Profit ($)']:,.0f}</td>
          <td style="color:{'#E06060' if row['Food Cost %']>35 else '#60D090'}">{row['Food Cost %']:.0f}%</td>
          <td><span style="color:{QUAD_C[row['Type']]}">{QUAD_I[row['Type']]} {row['Type']}</span></td>
        </tr>"""

    inv_rows = ""
    for i, (_, row) in enumerate(eng_df.iterrows()):
        eq = eoq_data[i]
        st = "🔴 Order Now" if eq["stockout"] else ("🟡 Too Much" if eq["overstock"] else "✅ OK")
        inv_rows += f"""<tr>
          <td><b>{row['Dish']}</b></td>
          <td>{int(row['Stock'])}</td><td>{int(eq['eoq'])}</td>
          <td>{int(eq['safety_stock'])}</td><td>{int(eq['rop'])}</td>
          <td>{eq['cycle_days']:.0f}</td>
          <td>${eq['annual_saving']:,.0f}</td><td>{st}</td>
        </tr>"""

    rec_html = ""
    pri_map  = {1:"🔴 URGENT",2:"🟡 IMPORTANT",3:"🟢 WORTH DOING"}
    for r in recs[:15]:
        col = '#DC4545' if r.priority==1 else '#F0A030' if r.priority==2 else '#35C886'
        rec_html += f"""
        <div style="border-left:3px solid {col};background:#0C1020;border-radius:0 8px 8px 0;
                    padding:12px 16px;margin-bottom:10px;font-size:13px">
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:5px;flex-wrap:wrap">
            <span style="font-size:10px;font-weight:700;color:{col}">{pri_map[r.priority]}</span>
            <span style="font-size:10px;color:#D2A248;background:rgba(210,162,72,.1);
                         padding:1px 7px;border-radius:20px">{r.cat.upper()}</span>
            <span style="font-size:10px;color:#4A5578">{r.dish}</span>
            <span style="margin-left:auto;font-weight:700;color:#35C886">+${r.impact:.0f}/wk</span>
          </div>
          <div style="font-weight:600;color:#DEE2F0;margin-bottom:4px">{r.title}</div>
          <div style="color:#6070A0;line-height:1.6;margin-bottom:5px">{r.insight}</div>
          <div style="color:#D2A248;font-weight:500">→ {r.action}</div>
        </div>"""

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MenuProfit Pro — {period} Report — {rdate}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=DM+Sans:wght@300;400;600&display=swap');
*{{box-sizing:border-box;margin:0;padding:0;font-family:'DM Sans',sans-serif}}
body{{background:#070910;color:#DEE2F0;padding:0}}
.page{{max-width:1020px;margin:0 auto;padding:28px 20px}}
.hdr{{background:linear-gradient(118deg,#0D1025,#090B15);border-bottom:1px solid rgba(210,162,72,.2);
  padding:24px 28px;border-radius:14px;margin-bottom:24px}}
.hdr h1{{font-family:'Cormorant Garamond',serif;font-size:1.8rem;color:#fff}}
.hdr h1 span{{color:#D2A248}}.hdr p{{color:#4A5578;font-size:.78rem;margin-top:5px}}
.kgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:11px;margin-bottom:24px}}
.kbox{{background:linear-gradient(148deg,#0F1220,#0B0D18);border:1px solid rgba(255,255,255,.07);
  border-radius:11px;padding:15px;text-align:center}}
.kbox .l{{font-size:.65rem;color:#3E4A6A;text-transform:uppercase;letter-spacing:.09em;margin-bottom:5px}}
.kbox .v{{font-size:1.45rem;font-weight:600;color:#D2A248}}
.kbox .s{{font-size:.65rem;color:#2E3858;margin-top:3px}}
.kbox.g .v{{color:#35C886}}.kbox.r .v{{color:#DC4545}}.kbox.b .v{{color:#5498F2}}
.sec{{margin-bottom:24px}}
.sec h2{{font-family:'Cormorant Garamond',serif;font-size:1.2rem;color:#D2A248;
  border-bottom:1px solid rgba(210,162,72,.14);padding-bottom:6px;margin-bottom:12px}}
.ibox{{background:#0C1020;border:1px solid rgba(84,152,242,.16);border-radius:8px;
  padding:12px 15px;margin-bottom:12px;font-size:12.5px;line-height:1.65;color:#7A90B8}}
.ibox strong{{color:#5498F2}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
thead tr{{background:#0F1220}}
th{{padding:9px;text-align:left;color:#4A5578;font-weight:600;font-size:.65rem;
  text-transform:uppercase;letter-spacing:.06em}}
td{{padding:8px 9px;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:middle}}
tr:hover{{background:rgba(255,255,255,.02)}}
.foot{{text-align:center;color:#1E2840;font-size:.72rem;padding:16px 0;
  border-top:1px solid rgba(255,255,255,.04);margin-top:26px}}
@media print{{body{{background:#fff;color:#111}}
  .hdr,.kbox,table,.ibox{{background:#f8f8f8;border-color:#ddd;color:#111}}
  .kbox .v{{color:#A07020}}.sec h2{{color:#A07020}}}}
</style></head><body><div class="page">
<div class="hdr">
  <h1>🍽️ MenuProfit <span>Pro</span></h1>
  <p>{period} Report &nbsp;·&nbsp; {rdate}</p>
</div>
<div class="sec"><h2>Financial Summary</h2>
<div class="kgrid">
  <div class="kbox"><div class="l">Revenue / {plab}</div><div class="v">${fin['rev']*factor:,.0f}</div></div>
  <div class="kbox {'g' if fin['net']>0 else 'r'}"><div class="l">Net Profit / {plab}</div><div class="v">${fin['net']*factor:,.0f}</div></div>
  <div class="kbox {'g' if gain>0 else ''}"><div class="l">Extra if Repriced</div><div class="v">${gain*factor:+,.0f}</div><div class="s">per {plab.lower()}</div></div>
  <div class="kbox {'r' if fin['fc_pct']>35 else 'g'}"><div class="l">Food Cost %</div><div class="v">{fin['fc_pct']:.1f}%</div><div class="s">target under 32%</div></div>
  <div class="kbox {'g' if fin['safety']>25 else 'r'}"><div class="l">Safety Buffer</div><div class="v">{fin['safety']:.1f}%</div></div>
  <div class="kbox b"><div class="l">Bad Week Risk</div><div class="v">${mc['p5']:,.0f}</div></div>
  <div class="kbox"><div class="l">Annual Profit</div><div class="v">${fin['annual']:,.0f}</div></div>
  <div class="kbox"><div class="l">Break-Even Orders</div><div class="v">{fin['bep']:.0f}</div><div class="s">per week needed</div></div>
</div></div>
<div class="sec"><h2>Full Dish Breakdown</h2>
<table><thead><tr>
  <th>Dish</th><th>Category</th><th>Ingredient Cost</th><th>Real Cost</th>
  <th>Price</th><th>Suggested Price</th><th>Profit</th>
  <th>$/Min</th><th>Orders/Wk</th><th>Weekly Profit</th><th>Food Cost%</th><th>Type</th>
</tr></thead><tbody>{dish_rows}</tbody></table></div>
<div class="sec"><h2>Stock Levels</h2>
<table><thead><tr>
  <th>Dish</th><th>Stock</th><th>Best Order Size</th><th>Safety Buffer</th>
  <th>Order When Below</th><th>Order Every (days)</th><th>Annual Saving</th><th>Status</th>
</tr></thead><tbody>{inv_rows}</tbody></table></div>
<div class="sec"><h2>Top Actions</h2>{rec_html}</div>
<div class="foot">MenuProfit Pro &nbsp;·&nbsp; {period} Report &nbsp;·&nbsp; {rdate}</div>
</div></body></html>"""


# ═════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
for k, v in [("opt_prices",None),("eng_df",None),("fin",None),("mc",None),
             ("sh_k",0.),("sh_b",0.),("sh_s",0.),("kitchen",None),
             ("eoq_data",None),("recs",None),("ran",False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ═════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="banner">
  <div>
    <div class="b-title">🍽️ MenuProfit <span>Pro</span></div>
    <div class="b-sub">Restaurant profit tool · smart pricing · stock management · kitchen efficiency</div>
  </div>
  <div class="b-badges">
    <span class="b-badge">PRICE OPTIMISER</span>
    <span class="b-badge">STOCK ALERTS</span>
    <span class="b-badge">RISK SCENARIOS</span>
    <span class="b-badge">LIFETIME VALUE</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Your Business Numbers")
    weekly_fixed           = st.number_input("Fixed costs per week ($)",      value=3500, step=100)
    available_kitchen_mins = st.number_input("Kitchen minutes per week",       value=4800, step=60,
                                             help="Staff hours × number of cooks × 60")
    ing_budget             = st.number_input("Food budget per week ($)",       value=4000, step=100)
    st.divider()
    st.markdown("### 📐 Price Rules")
    min_mg = st.slider("Minimum profit margin %",   20, 60, 28)
    max_up = st.slider("Maximum price increase %",   5, 50, 35)
    st.divider()
    st.markdown("### 📦 Stock Settings")
    service_lvl  = st.slider("Stock safety level %",  80, 99, 95)
    demand_sigma = st.slider("Demand variability %",   5, 30, 15,
                             help="How much your weekly orders vary")
    st.divider()
    st.markdown("### 📂 Upload Your Menu")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    imported_df = None
    if uploaded:
        try:
            imported_df = pd.read_csv(uploaded)
            miss = [c for c in DEFAULT.keys() if c not in imported_df.columns]
            if miss: st.error(f"Missing columns: {miss}"); imported_df = None
            else:    st.success("✓ Menu loaded")
        except Exception as e:
            st.error(str(e))
    st.divider()
    st.caption(
        "**Price sensitivity guide**\n\n"
        "−0.3 – −0.6 : Drinks & coffee (not price sensitive)\n\n"
        "−0.7 – −1.0 : Typical main dishes\n\n"
        "−1.1 – −1.5 : Budget items like salads & sides"
    )

Z_SCORES = {80:1.282, 85:1.036, 90:1.282, 95:1.645, 99:2.326}
svc_z    = Z_SCORES.get(service_lvl, 1.645)

# ═════════════════════════════════════════════════════════════════════════════
#  MENU TABLE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sh">📋 Your Menu — edit any cell, add rows with ＋</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tip">'
    'Fill in all columns for each dish. Real cost = ingredient cost after waste + staff time. '
    'All results update when you click Run Analysis.'
    '</div>', unsafe_allow_html=True)

base_df = pd.DataFrame(imported_df if imported_df is not None else DEFAULT)
menu_df = st.data_editor(
    base_df, use_container_width=True, num_rows="dynamic",
    column_config={
        "Ingredient ($)":   st.column_config.NumberColumn("Ingredient ($)",   format="$%.2f", min_value=0.0),
        "Price ($)":        st.column_config.NumberColumn("Price ($)",        format="$%.2f", min_value=0.0),
        "Sold/Wk":          st.column_config.NumberColumn("Sold per Week",    min_value=0),
        "Prep (min)":       st.column_config.NumberColumn("Prep (min)",       min_value=0),
        "Cook (min)":       st.column_config.NumberColumn("Cook (min)",       min_value=1),
        "Table Time (min)": st.column_config.NumberColumn("Table Time (min)", min_value=5),
        "Wastage %":        st.column_config.NumberColumn("Wastage %",        format="%.0f%%", min_value=0, max_value=50),
        "Labour $/hr":      st.column_config.NumberColumn("Labour $/hr",      format="$%.0f",  min_value=0),
        "Elasticity":       st.column_config.NumberColumn("Price Sensitivity",format="%.1f",   min_value=-3.0, max_value=-0.1),
        "Stock":            st.column_config.NumberColumn("Stock (units)",    min_value=0),
        "Reorder Pt":       st.column_config.NumberColumn("Reorder Point",    min_value=0),
        "Lead Days":        st.column_config.NumberColumn("Lead Days",        min_value=1),
        "Order Cost ($)":   st.column_config.NumberColumn("Order Cost ($)",   format="$%.0f",  min_value=0),
        "Hold Cost %/yr":   st.column_config.NumberColumn("Hold Cost %/yr",   format="%.0f%%", min_value=1),
        "Rating":           st.column_config.NumberColumn("Rating (1–5)",     format="%.1f",   min_value=1.0, max_value=5.0),
        "Bundle Attach %":  st.column_config.NumberColumn("Bundle Attach %",  format="%.0f%%", min_value=0, max_value=100),
    },
    hide_index=True,
)

if not menu_df.empty:
    bad = menu_df[menu_df["Ingredient ($)"] >= menu_df["Price ($)"]]
    if not bad.empty:
        st.warning(f"⚠️ Ingredient cost is higher than selling price: {', '.join(bad['Dish'].tolist())} — please fix before running.")

c_run, c_exp = st.columns([3,1])
with c_run:
    run_btn = st.button("🚀  Run Analysis")
with c_exp:
    if st.session_state.eng_df is not None:
        buf = io.StringIO()
        st.session_state.eng_df.to_csv(buf, index=False)
        st.download_button("⬇️ Export CSV", buf.getvalue(), "menuprofit.csv", "text/csv")

# ═════════════════════════════════════════════════════════════════════════════
#  RUN ENGINE
# ═════════════════════════════════════════════════════════════════════════════
if run_btn:
    if menu_df.empty: st.error("Add at least one dish."); st.stop()
    with st.spinner("Crunching the numbers…"):
        rows = menu_df.to_dict("records")

        eng_df, avg_p, avg_q = classify_menu(menu_df)
        fin      = calc_fin(menu_df, eng_df, weekly_fixed)
        _, sh_k, sh_b, sh_s = run_lp(menu_df, available_kitchen_mins, ing_budget)
        opt_p, _ = optimise_prices(rows, weekly_fixed, min_mg/100, max_up/100)
        mc       = monte_carlo(menu_df, fixed=weekly_fixed, sigma=demand_sigma/100)
        eoq_data = [eoq_model(row, demand_sigma/100, svc_z) for row in rows]
        kitchen  = kitchen_analysis(menu_df, available_kitchen_mins)
        recs     = generate_recommendations(
            menu_df, eng_df, fin, opt_p, sh_k, sh_b, sh_s,
            mc, eoq_data, kitchen, weekly_fixed, available_kitchen_mins)

        st.session_state.update(
            opt_prices=opt_p, eng_df=eng_df, fin=fin, mc=mc,
            sh_k=sh_k, sh_b=sh_b, sh_s=sh_s,
            kitchen=kitchen, eoq_data=eoq_data, recs=recs, ran=True)

    st.success(f"✓ Done — {len(recs)} actions found")

# ═════════════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.ran:
    eng_df     = st.session_state.eng_df
    opt_prices = st.session_state.opt_prices
    fin        = st.session_state.fin
    mc         = st.session_state.mc
    sh_k       = st.session_state.sh_k
    sh_b       = st.session_state.sh_b
    sh_s       = st.session_state.sh_s
    kitchen    = st.session_state.kitchen
    eoq_data   = st.session_state.eoq_data
    recs       = st.session_state.recs
    rows       = menu_df.to_dict("records")

    curr_profit  = total_net_profit([r["Price ($)"] for r in rows], rows, weekly_fixed)
    opt_profit   = total_net_profit(opt_prices, rows, weekly_fixed)
    wk_gain      = opt_profit - curr_profit
    stockouts    = sum(1 for e in eoq_data if e["stockout"])
    critical_n   = sum(1 for r in recs if r.priority == 1)
    high_n       = sum(1 for r in recs if r.priority == 2)
    med_n        = sum(1 for r in recs if r.priority == 3)
    total_uplift = sum(r.impact for r in recs if r.priority <= 2)

    T1, T2, T3, T4 = st.tabs([
        "📊  Overview",
        "💰  Pricing",
        "💡  Actions",
        "📄  Report",
    ])

    # ── TAB 1: OVERVIEW ──────────────────────────────────────────────────
    with T1:
        st.markdown('<div class="sh">Business Health at a Glance</div>', unsafe_allow_html=True)

        c = st.columns(4)
        for col, (lbl, val, sub, cls) in zip(c, [
            ("Weekly Revenue",  f"${fin['rev']:,.0f}",   "total sales",            "white"),
            ("Net Profit",      f"${fin['net']:,.0f}",   "after all costs",         "green" if fin["net"]>0 else "red"),
            ("Annual Profit",   f"${fin['annual']:,.0f}","52 weeks",                "gold"),
            ("Food Cost %",     f"{fin['fc_pct']:.1f}%", "incl. waste + staff",    "green" if fin["fc_pct"]<=32 else "red"),
        ]):
            col.markdown('<div class="krow">'+kcard(lbl,val,sub,cls)+'</div>', unsafe_allow_html=True)

        c2 = st.columns(4)
        for col, (lbl, val, sub, cls) in zip(c2, [
            ("Safety Buffer",    f"{fin['safety']:.1f}%",      "above break-even",  "green" if fin["safety"]>25 else "red"),
            ("Bad Week Profit",  f"${mc['p5']:,.0f}",           "worst 5% scenario", "blue"),
            ("Kitchen Busy",     f"{kitchen['util_pct']:.0f}%", "of total time",     "green" if kitchen["util_pct"]<75 else "red"),
            ("Stock Alerts",     f"{stockouts}",                "dishes running low", "red" if stockouts>0 else "green"),
        ]):
            col.markdown('<div class="krow">'+kcard(lbl,val,sub,cls)+'</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns([3,2])
        with col_l:
            _, avg_p2, avg_q2 = classify_menu(menu_df)
            st.plotly_chart(fig_bcg(eng_df, avg_p2, avg_q2),
                            use_container_width=True, **CHART_CFG)
        with col_r:
            st.plotly_chart(fig_pl_waterfall(fin),
                            use_container_width=True, **CHART_CFG)
            st.plotly_chart(fig_util_gauge(kitchen["util_pct"]),
                            use_container_width=True, **CHART_CFG)

        st.plotly_chart(fig_profit_min(eng_df), use_container_width=True, **CHART_CFG)

        c_fc, c_clv = st.columns(2)
        with c_fc:
            st.plotly_chart(fig_food_cost_bars(eng_df), use_container_width=True, **CHART_CFG)
        with c_clv:
            st.plotly_chart(fig_clv(eng_df), use_container_width=True, **CHART_CFG)

        st.markdown('<div class="sh">Full Dish Table</div>', unsafe_allow_html=True)
        disp = eng_df[[
            "Dish","Category","Ingredient ($)","True Cost ($)","Price ($)",
            "True Profit ($)","True Margin %","Profit/Min ($)",
            "Sold/Wk","Wk True Profit ($)","Food Cost %","Rating","Bundle Sales/Wk","Type"
        ]].copy()
        for col in ["Ingredient ($)","True Cost ($)","Price ($)","True Profit ($)","Profit/Min ($)"]:
            disp[col] = disp[col].map("${:.2f}".format)
        disp["True Margin %"]      = disp["True Margin %"].map("{:.1f}%".format)
        disp["Food Cost %"]        = disp["Food Cost %"].map("{:.0f}%".format)
        disp["Wk True Profit ($)"] = disp["Wk True Profit ($)"].map("${:,.0f}".format)
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── TAB 2: PRICING ────────────────────────────────────────────────────
    with T2:
        st.markdown('<div class="sh">Best Prices for Each Dish</div>', unsafe_allow_html=True)

        binding = "kitchen time" if sh_k > sh_b * 8 else "food budget"
        st.markdown(
            f'<div class="ibox">'
            f'Your biggest limit right now is <strong>{binding}</strong>. '
            f'Each extra kitchen minute is worth <strong>${sh_k:.3f}</strong> more profit per week. '
            f'Switching to the suggested prices would add '
            f'<strong>${wk_gain:+,.0f}/week</strong> (${wk_gain*52:+,.0f}/year).'
            f'</div>', unsafe_allow_html=True)

        pc = st.columns(3)
        pc[0].markdown('<div class="krow">'+kcard("Current Weekly Profit",   f"${curr_profit:,.0f}", "", "white")+'</div>', unsafe_allow_html=True)
        pc[1].markdown('<div class="krow">'+kcard("Suggested Weekly Profit",  f"${opt_profit:,.0f}",  "", "green")+'</div>', unsafe_allow_html=True)
        pc[2].markdown('<div class="krow">'+kcard("Extra Per Year",           f"${wk_gain*52:+,.0f}", "if you reprice today", "gold")+'</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_shadow(sh_k, sh_b, sh_s), use_container_width=True, **CHART_CFG)
        with c2:
            st.plotly_chart(fig_price_compare(rows, opt_prices, eng_df), use_container_width=True, **CHART_CFG)

        st.plotly_chart(fig_mc(mc), use_container_width=True, **CHART_CFG)

        st.markdown("#### Suggested Prices")
        price_rows = []
        for i, r in enumerate(rows):
            p_c  = r["Price ($)"]
            p_o  = round(opt_prices[i] * 4) / 4
            tc   = true_cost(r)
            chg  = (p_o - p_c) / p_c * 100
            cm_c = weekly_profit_item(p_c, r)
            cm_o = weekly_profit_item(p_o, r)
            pm_c = profit_per_min(p_c, tc, r["Prep (min)"], r["Cook (min)"])
            pm_o = profit_per_min(p_o, tc, r["Prep (min)"], r["Cook (min)"])
            price_rows.append({
                "Dish":              r["Dish"],
                "Type":              f"{QUAD_I[eng_df.iloc[i]['Type']]} {eng_df.iloc[i]['Type']}",
                "Real Cost":         f"${tc:.2f}",
                "Current Price":     f"${p_c:.2f}",
                "Suggested Price":   f"${p_o:.2f}",
                "Change":            f"{chg:+.1f}%",
                "Current $/Min":     f"${pm_c:.2f}",
                "New $/Min":         f"${pm_o:.2f}",
                "Extra per Week":    f"${cm_o-cm_c:+,.0f}",
                "Extra per Year":    f"${(cm_o-cm_c)*52:+,.0f}",
            })
        st.dataframe(pd.DataFrame(price_rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="sh">Stock Levels</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ibox">The best order size keeps you from running out while avoiding '
            f'too much stock sitting around. Safety buffer uses {service_lvl}% confidence level.</div>',
            unsafe_allow_html=True)
        st.dataframe(fig_eoq_table(eng_df, eoq_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 🎚️ Try a Price")
        sel   = st.selectbox("Pick a dish", menu_df["Dish"].tolist(), key="pt")
        sel_r = menu_df[menu_df["Dish"] == sel].to_dict("records")[0]
        tc_sel= true_cost(sel_r)
        new_p = st.slider("Test price ($)",
                          float(tc_sel * 1.05), float(sel_r["Price ($)"] * 2.2),
                          float(sel_r["Price ($)"]), step=0.25, key="ps")
        new_qty = demand(new_p, sel_r["Price ($)"], sel_r["Sold/Wk"], sel_r["Elasticity"])
        base_cm = weekly_profit_item(sel_r["Price ($)"], sel_r)
        new_cm  = (new_p - tc_sel) * new_qty
        mg_pct  = (new_p - tc_sel) / new_p * 100
        clv, ret= markov_ltv(sel_r["Rating"], new_p - tc_sel)

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Est. Orders/Wk",  f"{new_qty:.0f}",     f"{new_qty-sel_r['Sold/Wk']:+.0f}")
        m2.metric("Weekly Profit",   f"${new_cm:,.0f}",    f"${new_cm-base_cm:+,.0f}")
        m3.metric("Profit Margin",   f"{mg_pct:.1f}%")
        m4.metric("Extra per Year",  f"${(new_cm-base_cm)*52:+,.0f}")
        m5.metric("Customer Value",  f"${clv:,.0f}",       f"Returns {ret}%")

    # ── TAB 3: ACTIONS ────────────────────────────────────────────────────
    with T3:
        st.markdown('<div class="sh">What to Do — Sorted by Importance</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="psummary">'
            f'<div class="pchip crit"><div class="pnum crit">{critical_n}</div><div class="plbl">🔴 Urgent</div></div>'
            f'<div class="pchip high"><div class="pnum high">{high_n}</div><div class="plbl">🟡 Important</div></div>'
            f'<div class="pchip med"><div class="pnum med">{med_n}</div><div class="plbl">🟢 Worth Doing</div></div>'
            f'<div class="pchip" style="background:rgba(210,162,72,.08);border:1px solid rgba(210,162,72,.2)">'
            f'<div class="pnum gold">${total_uplift:,.0f}</div><div class="plbl">Weekly Gain Available</div></div>'
            f'</div>', unsafe_allow_html=True)

        cats = ["All"] + sorted(set(r.cat for r in recs))
        filt = st.selectbox("Filter by type", cats, key="rc_filt")
        show_recs = recs if filt == "All" else [r for r in recs if r.cat == filt]

        CAT_CSS = {"Pricing":"p","Inventory":"i","Menu":"m","Operations":"o","Bundle":"b"}
        PRI_CSS = {1:"crit",2:"high",3:"med"}

        for r in show_recs:
            css  = PRI_CSS[r.priority]
            ccat = CAT_CSS.get(r.cat,"p")
            st.markdown(f"""
            <div class="rec {css}">
              <div class="rec-top">
                <span class="rec-cat {ccat}">{r.cat}</span>
                <span class="rec-dish">{r.dish}</span>
                <span class="rec-imp">+${r.impact:,.0f}/wk</span>
              </div>
              <div class="rec-title">{r.title}</div>
              <div class="rec-insight">{r.insight}</div>
              <div class="rec-action">→ {r.action}</div>
            </div>""", unsafe_allow_html=True)

    # ── TAB 4: REPORT ─────────────────────────────────────────────────────
    with T4:
        st.markdown('<div class="sh">Download Your Report</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="tip">Download as HTML, then open it in your browser and press Ctrl+P to save as PDF. '
            'Great for sharing with your accountant or team.</div>',
            unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        with rc1: period = st.radio("Period", ["Weekly","Monthly"], horizontal=True)
        with rc2: rdate  = str(st.date_input("Report date", datetime.date.today()))

        factor = 4.33 if period=="Monthly" else 1
        plab   = "month" if period=="Monthly" else "week"

        st.markdown(f"**Numbers for this {period.lower()} report:**")
        rv = st.columns(5)
        rv[0].metric(f"Revenue",       f"${fin['rev']*factor:,.0f}")
        rv[1].metric(f"Net Profit",    f"${fin['net']*factor:,.0f}")
        rv[2].metric("Annual Profit",   f"${fin['annual']:,.0f}")
        rv[3].metric("Extra if Repriced",f"${wk_gain*factor:+,.0f}/{plab}")
        rv[4].metric("Profit Range",    f"${mc['p5']*factor:,.0f}–${mc['p95']*factor:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### What if your fixed costs change?")
        sce = []
        for mult in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
            fc   = weekly_fixed * mult
            bep  = fc / fin["avg_cm"] if fin["avg_cm"] > 0 else 0
            net_w= (fin["pool"] - fc) * factor
            sce.append({
                f"Fixed Costs":          f"${fc*factor:,.0f}",
                "Orders to Break Even":  f"{bep:.0f}",
                "Your Extra Orders":     f"{fin['covers']-bep:+.0f}",
                f"Net Profit":           f"${net_w:+,.0f}",
                "Annual Profit":         f"${(fin['pool']-fc)*52:+,.0f}",
            })
        st.dataframe(pd.DataFrame(sce), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📄  Build Report"):
            html = generate_report(
                menu_df, eng_df, fin, opt_prices, mc,
                sh_k, sh_b, sh_s, eoq_data, kitchen, recs, period, rdate)
            st.download_button(
                f"⬇️  Download {period} Report (HTML)",
                data=html,
                file_name=f"MenuProfit_{period}_{rdate}.html",
                mime="text/html")
            st.success("Ready — click the button above to download.")

    st.markdown("""
    <div style="text-align:center;color:#1E2840;font-size:.72rem;
                padding:20px 0 6px;border-top:1px solid rgba(255,255,255,.04);margin-top:18px">
      MenuProfit Pro &nbsp;·&nbsp; Smart pricing · Stock management · Kitchen efficiency
    </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:52px 16px;color:#2E3858">
      <div style="font-size:2.5rem;margin-bottom:14px">🍽️</div>
      <div style="font-size:1rem;line-height:1.9">
        Edit your menu above, then tap<br>
        <strong style="color:#D2A248">🚀 Run Analysis</strong><br>
        Results appear in under 3 seconds.
      </div>
    </div>""", unsafe_allow_html=True)
