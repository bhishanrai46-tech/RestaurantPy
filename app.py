"""
MenuProfit Pro — Full SaaS Edition
====================================
Taha's Operations Research (10th Ed.) applied exhaustively:

  Ch. 2–3   Linear Programming      → Menu mix optimisation + dual/shadow prices
  Ch. 10    Nonlinear Optimisation   → Price-elasticity demand + global price solver
  Ch. 13    Inventory / EOQ         → Economic Order Qty, safety stock, reorder point
  Ch. 15    Decision Analysis        → Menu engineering matrix (Stars/Puzzles/Horses/Dogs)
  Ch. 17    Markov Chains            → Customer retention & lifetime value
  Ch. 18    Queuing Theory           → Kitchen throughput, utilisation, bottleneck analysis
  Ch. 19    Monte Carlo Simulation   → Revenue risk distribution, P5/P50/P95

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

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MenuProfit Pro",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');
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
.gold{color:#D2A248}.green{color:#35C886}.red{color:#DC4545}.blue{color:#5498F2}.white{color:#C0C8E0}.purple{color:#9B6CF5}
.ibox{background:#0C1020;border:1px solid rgba(84,152,242,.18);border-radius:9px;
  padding:13px 16px;margin:.4rem 0 1rem;font-size:.81rem;line-height:1.7;color:#7A90B8}
.ibox strong{color:#5498F2}
.rec{border-radius:10px;padding:14px 17px;margin-bottom:10px;background:#0C1020;
  border:1px solid rgba(255,255,255,.06);border-left:3px solid #D2A248;position:relative}
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
.rec-imp.neg{color:#DC4545}
.rec-title{font-weight:600;font-size:.93rem;color:#DEE2F0;margin-bottom:4px}
.rec-insight{font-size:.80rem;color:#6070A0;line-height:1.65;margin-bottom:6px}
.rec-action{font-size:.80rem;color:#D2A248;font-weight:500}
.rec-or{font-size:.65rem;color:#2A3550;margin-top:6px;font-style:italic}
.alert{border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:.82rem;line-height:1.6;
  border-left:3px solid #D2A248;background:rgba(210,162,72,.05)}
.alert.r{border-color:#DC4545;background:rgba(220,69,69,.05)}
.alert.g{border-color:#35C886;background:rgba(53,200,134,.04)}
.alert.b{border-color:#5498F2;background:rgba(84,152,242,.05)}
.tip{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.05);border-radius:8px;
  padding:8px 13px;font-size:.77rem;color:#3E4A6A;margin:.3rem 0 1rem}
.stButton>button{background:linear-gradient(135deg,#9B7018,#D2A248)!important;color:#07090E!important;
  font-weight:700!important;border:none!important;border-radius:9px!important;
  padding:10px 22px!important;font-size:.88rem!important;width:100%!important;
  letter-spacing:.02em!important;transition:opacity .15s}
.stButton>button:hover{opacity:.88!important}
[data-testid="stMetric"]{background:rgba(10,12,22,.9);border-radius:10px;
  border:1px solid rgba(255,255,255,.06);padding:10px!important}
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

# ─── Constants ────────────────────────────────────────────────────────────────
DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#7A8AB0"),
    margin=dict(l=14, r=14, t=38, b=14),
)
GOLD="#D2A248"; GREEN="#35C886"; RED="#DC4545"; BLUE="#5498F2"; PURPLE="#9B6CF5"; AMBER="#F0A030"
QUAD_C={"Star":GOLD,"Puzzle":BLUE,"Plow Horse":GREEN,"Dog":RED}
QUAD_I={"Star":"⭐","Puzzle":"🧩","Plow Horse":"🐎","Dog":"🐕"}

# ─── Default data ─────────────────────────────────────────────────────────────
DEFAULT = {
    "Dish":               ["Wagyu Burger","Truffle Pasta","Caesar Salad","Espresso Martini",
                           "Lava Cake","House Wine","Grilled Salmon","Garlic Bread"],
    "Category":           ["Mains","Mains","Starters","Drinks","Desserts","Drinks","Mains","Starters"],
    "Ingredient ($)":     [8.50, 6.20, 2.80, 1.90, 2.40, 3.20, 9.80, 0.80],
    "Price ($)":          [24.00,22.00,14.00,16.00,12.00,11.00,28.00, 7.00],
    "Sold/Wk":            [120,   85,  200,  180,   90,  250,   70, 160],
    "Prep (min)":         [  8,   10,    2,    1,    5,    0,   10,   2],
    "Cook (min)":         [ 12,   15,    5,    3,    8,    2,   18,   4],
    "Table Time (min)":   [ 35,   40,   15,   20,   25,   10,   45,  10],
    "Wastage %":          [  8,   12,    5,    3,    6,    2,   10,   4],
    "Labour $/hr":        [ 30,   30,   30,   25,   30,   25,   32,  28],
    "Elasticity":         [-1.2, -1.0, -0.8, -0.6, -1.1, -0.9, -1.3,-0.7],
    "Stock":              [120,   80,  300,  200,  100,  400,   60, 250],
    "Reorder Pt":         [ 40,   30,  100,   60,   30,  100,   20,  80],
    "Lead Days":          [  2,    3,    1,    2,    2,    3,    2,   1],
    "Order Cost ($)":     [ 15,   20,   10,   12,   10,   18,   25,   8],
    "Hold Cost %/yr":     [ 25,   20,   30,   15,   30,   15,   25,  20],
    "Rating":             [4.7,  4.4,  4.2,  4.6,  4.8,  3.9,  4.5, 4.1],
    "Bundle Attach %":    [ 35,   22,   18,   40,   25,   30,   20,  45],
}

# ═════════════════════════════════════════════════════════════════════════════
#  OR MATH ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def true_cost(row):
    real_food = row["Ingredient ($)"] * (1 + row["Wastage %"] / 100)
    labour    = (row["Prep (min)"] + row["Cook (min)"]) / 60 * row["Labour $/hr"]
    return round(real_food + labour, 2)

def true_margin_pct(price, tc):
    return (price - tc) / price * 100 if price > 0 else 0

def profit_per_min(price, tc, prep, cook):
    total_t = prep + cook
    return (price - tc) / total_t if total_t > 0 else 0

def demand(p_new, p_base, qty_base, elast):
    if p_new <= 0: return 0.0
    return float(qty_base) * (p_new / float(p_base)) ** float(elast)

def weekly_profit_item(price, row):
    tc  = true_cost(row)
    qty = demand(price, row["Price ($)"], row["Sold/Wk"], row["Elasticity"])
    return (price - tc) * qty

def total_net_profit(prices, rows, fixed):
    return sum(weekly_profit_item(prices[i], rows[i]) for i in range(len(rows))) - fixed

def run_lp(df, kitchen_mins, ing_budget):
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

    sh_kitchen = sh_budget = sh_seating = 0.0
    if res.status == 0 and hasattr(res, "ineqlin") and res.ineqlin is not None:
        m = res.ineqlin.marginals
        if m is not None and len(m) >= 3:
            sh_kitchen, sh_budget, sh_seating = abs(m[0]), abs(m[1]), abs(m[2])

    return res, sh_kitchen, sh_budget, sh_seating

def optimise_prices(rows, fixed, min_mg=0.28, max_up=0.35):
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
    D  = row["Sold/Wk"] * 52
    S  = row["Order Cost ($)"]
    H  = row["Ingredient ($)"] * row["Hold Cost %/yr"] / 100
    if H <= 0: H = 0.01

    eoq_qty    = math.sqrt(2 * D * S / H)
    orders_pa  = D / eoq_qty
    cycle_days = 365 / orders_pa

    sigma_wk   = row["Sold/Wk"] * demand_sigma_pct
    lt_weeks   = row["Lead Days"] / 7
    ss         = service_z * sigma_wk * math.sqrt(lt_weeks)
    rop        = (row["Sold/Wk"] / 7 * row["Lead Days"]) + ss

    annual_order_cost_curr = orders_pa * S
    hold_curr  = (row["Reorder Pt"] / 2) * H
    hold_eoq   = (eoq_qty / 2) * H
    orders_eoq = (D / eoq_qty) * S
    saving_pa  = (hold_curr + annual_order_cost_curr) - (hold_eoq + orders_eoq)

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

# ── FIXED: all numpy, no pandas Series in arithmetic ─────────────────────────
def monte_carlo(df, fixed=0, n=8000, sigma=0.15):
    tc   = df.apply(true_cost, axis=1).values          # numpy array
    cm   = (df["Price ($)"].values - tc).clip(min=0)   # numpy array — FIX
    qtys = df["Sold/Wk"].values.astype(float)

    rng  = np.random.default_rng(42)
    sim  = rng.normal(qtys, qtys * sigma, (n, len(qtys))).clip(0)
    prof = (sim * cm).sum(axis=1) - fixed

    return {
        "mean": prof.mean(), "p5":  np.percentile(prof,  5),
        "p25":  np.percentile(prof, 25), "p50": np.percentile(prof, 50),
        "p75":  np.percentile(prof, 75), "p95": np.percentile(prof, 95),
        "sims": prof, "std": prof.std(),
    }

def kitchen_analysis(df, available_mins):
    total_t  = (df["Prep (min)"] + df["Cook (min)"]) * df["Sold/Wk"]
    load     = total_t.sum()
    util     = load / available_mins if available_mins > 0 else 0
    tc_vals  = df.apply(true_cost, axis=1)
    pm       = ((df["Price ($)"].values - tc_vals.values) /
                (df["Prep (min)"].values + df["Cook (min)"].values + 0.01))
    bottleneck = df.iloc[total_t.values.argmax()]["Dish"]

    return {
        "load_mins":     load,
        "util":          util,
        "util_pct":      min(util * 100, 100),
        "seating_load":  (df["Table Time (min)"] * df["Sold/Wk"]).sum(),
        "bottleneck":    bottleneck,
        "profit_per_min": np.array(pm),
        "dish_loads":    total_t.values,
    }

def markov_ltv(rating, avg_weekly_contribution, weekly_visits=1.0, discount_rate_annual=0.10):
    p_retain  = min(0.96, max(0.30, 0.46 + (rating - 1.0) * 0.125))
    r_weekly  = discount_rate_annual / 52
    expected_visits = p_retain / (1 - p_retain + r_weekly)
    clv = avg_weekly_contribution * expected_visits * weekly_visits
    return round(clv, 0), round(p_retain * 100, 1)

def calc_fin(df, eng_df, fixed):
    pool   = eng_df["Wk True Profit ($)"].sum()
    rev    = (df["Price ($)"].values * df["Sold/Wk"].values).sum()
    covers = df["Sold/Wk"].sum()
    net    = pool - fixed
    avg_cm = pool / covers if covers else 0
    bep    = fixed / avg_cm if avg_cm > 0 else float("inf")
    surplus= covers - bep
    safety = net / pool * 100 if pool > 0 else 0
    fc_pct = eng_df["Food Cost %"].mean()
    ing_only=(df["Ingredient ($)"].values * df["Sold/Wk"].values).sum() / rev * 100 if rev > 0 else 0
    return dict(
        pool=pool, rev=rev, net=net, monthly=net*4.33, annual=net*52,
        covers=covers, avg_cm=avg_cm, bep=bep, surplus=surplus,
        safety=safety, fc_pct=fc_pct, ing_pct=ing_only, fixed=fixed,
    )

# ═════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE
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
    or_ref:   str

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
                f"Price is below true cost — margin only {margin:.1f}%",
                f"After accounting for {row['Wastage %']}% wastage and "
                f"{(row['Prep (min)']+row['Cook (min)']):.0f} min of labour "
                f"(${tc:.2f} true cost), your margin is {margin:.1f}% — dangerously thin.",
                f"Raise price to at least ${tc * 1.30:.2f} to reach a 23% true margin.",
                max(0, uplift) + 20,
                "LP shadow price analysis + True cost model (Taha Ch. 2–3, 10)"))

        elif pct_chg > 3 and uplift > 15:
            pri = 1 if uplift > 80 else (2 if uplift > 30 else 3)
            demand_drop = abs((demand(p_o, p_c, row["Sold/Wk"], row["Elasticity"]) - row["Sold/Wk"]) /
                              row["Sold/Wk"] * 100)
            recs.append(Rec("Pricing", pri, row["Dish"],
                f"Raise price ${p_c:.2f}→${p_o:.2f}, earn +${uplift:.0f}/wk",
                f"The price-elasticity model (ε={row['Elasticity']}) estimates demand falls by only "
                f"{demand_drop:.1f}% at the new price, but each unit earns ${p_o-tc:.2f} vs "
                f"${p_c-tc:.2f} — a net gain.",
                f"Update menu price to ${p_o:.2f} (+{pct_chg:.1f}%). Expected +${uplift:.0f}/week.",
                uplift,
                "Price elasticity demand model + Differential evolution optimiser (Taha Ch. 10)"))

        if fc_p > 38:
            fix_price = tc / 0.60
            recs.append(Rec("Pricing", 1, row["Dish"],
                f"Food cost {fc_p:.0f}% — renegotiate or reprice",
                f"True food cost (including {row['Wastage %']}% wastage) is {fc_p:.0f}% of revenue. "
                f"Industry target is <32%. Shadow price of kitchen time is ${sh_k:.2f}/min.",
                f"Raise price to ${fix_price:.2f} (target 40% FC) or reduce ingredient cost by "
                f"${tc - row['Price ($)'] * 0.38:.2f}.",
                (fix_price - p_c) * row["Sold/Wk"],
                "LP constraint analysis + True cost model (Taha Ch. 3, 13)"))

        if row["Rating"] >= 4.6 and pct_chg < 2:
            recs.append(Rec("Pricing", 3, row["Dish"],
                f"Rating {row['Rating']} — room for premium pricing",
                f"Markov retention analysis shows customers with a {row['Rating']}-star experience "
                f"have high repeat-visit probability (~{min(96, 46 + int((row['Rating']-1)*12.5))}%). "
                f"Premium-rated dishes carry pricing power that the elasticity model confirms.",
                f"Test a $0.50–$1.00 price increase. At current demand, "
                f"+$0.75 = +${0.75*row['Sold/Wk']:.0f}/wk.",
                0.75 * row["Sold/Wk"],
                "Markov CLV model + Elasticity pricing (Taha Ch. 10, 17)"))

    # ── INVENTORY ─────────────────────────────────────────────────────────
    for i, row in enumerate(rows):
        eq = eoq_data[i]

        if eq["stockout"]:
            recs.append(Rec("Inventory", 1, row["Dish"],
                f"Stockout risk — stock ({int(row['Stock'])}) below ROP ({int(eq['rop'])})",
                f"With {row['Lead Days']} days lead time and weekly demand of {row['Sold/Wk']}, "
                f"you need {eq['rop']:.0f} units before ordering (includes {eq['safety_stock']:.0f} "
                f"safety stock at 95% service level). Current stock of {int(row['Stock'])} is below this.",
                f"Order {int(eq['eoq'])} units immediately (EOQ). Next order in {eq['cycle_days']:.0f} days.",
                row["Price ($)"] * row["Sold/Wk"] * (row["Lead Days"] / 7),
                "EOQ + Safety stock model (Taha Ch. 13)"))

        elif eq["overstock"]:
            hold_excess = (row["Stock"] - eq["eoq"] * 1.2) * row["Ingredient ($)"] * row["Hold Cost %/yr"] / 100 / 52
            recs.append(Rec("Inventory", 3, row["Dish"],
                f"Overstock — holding cost ~${hold_excess:.0f}/wk",
                f"Current stock ({int(row['Stock'])} units) is over 2× EOQ ({int(eq['eoq'])}). "
                f"Excess inventory ties up capital at {row['Hold Cost %/yr']}% annual holding cost.",
                f"Reduce next order. Target {int(eq['eoq'])} units per cycle, every {eq['cycle_days']:.0f} days.",
                hold_excess,
                "EOQ inventory model — holding cost minimisation (Taha Ch. 13)"))

        if eq["annual_saving"] > 50:
            recs.append(Rec("Inventory", 2, row["Dish"],
                f"Switch to EOQ ordering — save ${eq['annual_saving']/52:.0f}/wk",
                f"Your current reorder quantity differs from the optimal EOQ of {int(eq['eoq'])} units. "
                f"The EOQ balances ordering cost (${row['Order Cost ($)']}) against holding cost "
                f"({row['Hold Cost %/yr']}%/yr), saving ${eq['annual_saving']:.0f}/year.",
                f"Set standing order for {int(eq['eoq'])} units every {eq['cycle_days']:.0f} days.",
                eq["annual_saving"] / 52,
                "Economic Order Quantity formula √(2DS/H) (Taha Ch. 13)"))

    # ── MENU ENGINEERING ──────────────────────────────────────────────────
    for i, row in enumerate(eng_df.iterrows()):
        _, row = row
        t = row["Type"]

        if t == "Dog":
            kitchen_freed = (row["Prep (min)"] + row["Cook (min)"]) * row["Sold/Wk"]
            opportunity   = kitchen_freed * sh_k
            recs.append(Rec("Menu", 1, row["Dish"],
                f"Dog: low profit + low demand — consider removal",
                f"Menu engineering matrix places {row['Dish']} in the Dog quadrant: "
                f"below-average profit (${row['True Profit ($)']:.2f}/dish) AND below-average "
                f"orders ({row['Sold/Wk']:.0f}/wk). Kitchen time freed = ${opportunity:.0f}/wk.",
                f"Replace with a Puzzle-converted item or bundle with a Star. "
                f"Removing frees ~${opportunity:.0f}/wk in kitchen capacity.",
                opportunity,
                "Menu engineering quadrant analysis (Taha Ch. 15 Decision Analysis)"))

        elif t == "Puzzle":
            target_qty    = eng_df["Sold/Wk"].mean() * 0.9
            revenue_lift  = max(0, (target_qty - row["Sold/Wk"]) * row["True Profit ($)"])
            recs.append(Rec("Menu", 2, row["Dish"],
                f"Puzzle: great margin but underordered — promote",
                f"{row['Dish']} earns ${row['True Profit ($)']:.2f}/dish (above average) but sells "
                f"only {row['Sold/Wk']:.0f}/wk. Reaching average ({target_qty:.0f}/wk) adds "
                f"${revenue_lift:.0f}/wk.",
                f"Feature on specials board, add photo, train staff to upsell. "
                f"Target: +{max(0, target_qty-row['Sold/Wk']):.0f} orders/wk.",
                revenue_lift,
                "Menu engineering quadrant (Taha Ch. 15) + Demand analysis"))

        elif t == "Plow Horse":
            p_test = row["Price ($)"] * 1.07
            tc_val = row["True Cost ($)"]
            extra  = (p_test - tc_val - (row["Price ($)"] - tc_val)) * row["Sold/Wk"] * 0.92
            if extra > 0:
                recs.append(Rec("Menu", 2, row["Dish"],
                    f"Plow Horse: popular but thin margin — small price lift",
                    f"{row['Dish']} sells {row['Sold/Wk']:.0f}/wk (above average) but margin is "
                    f"only {row['True Margin %']:.1f}%. Elasticity (ε={row['Elasticity']}) shows "
                    f"a 7% price increase drops demand by only ~{abs(row['Elasticity'])*7:.1f}%.",
                    f"Raise price by $0.50–$1.00. At ${row['Price ($)']+0.75:.2f}, "
                    f"est. +${extra:.0f}/wk.",
                    extra,
                    "Plow Horse strategy + price elasticity model (Taha Ch. 10, 15)"))

        elif t == "Star":
            premium_p = row["Price ($)"] * 1.10
            tc_val    = row["True Cost ($)"]
            extra     = (premium_p - tc_val - (row["Price ($)"] - tc_val)) * row["Sold/Wk"] * 0.95
            clv_val   = markov_ltv(row["Rating"], row["True Profit ($)"])[0]
            recs.append(Rec("Menu", 3, row["Dish"],
                f"Star: protect quality, explore premium variant",
                f"{row['Dish']} is your highest-performing dish. Markov CLV model shows "
                f"high-rated Stars retain customers worth ${clv_val:,.0f} lifetime value each.",
                f"Consider a 'Premium {row['Dish']}' at ${premium_p:.2f}. "
                f"If 20% of orders upgrade, +${extra*0.2:.0f}/wk.",
                extra * 0.2,
                "Star quadrant strategy + Markov CLV (Taha Ch. 15, 17)"))

    # ── OPERATIONS ────────────────────────────────────────────────────────
    util_pct = kitchen["util_pct"]

    if util_pct > 85:
        recs.append(Rec("Operations", 1, "Kitchen",
            f"Kitchen at {util_pct:.0f}% — near maximum capacity",
            f"Queuing theory (M/G/1 model, Taha Ch. 18) shows above 80% utilisation, "
            f"wait times grow exponentially. At {util_pct:.0f}%, throughput quality degrades.",
            f"Drop the two lowest Profit/Min dishes, or add 1 kitchen staff to extend "
            f"capacity by ~40 min/shift, worth ${sh_k * 40:.0f} in additional profit.",
            sh_k * 40,
            "M/G/1 queuing model — utilisation threshold (Taha Ch. 18)"))

    elif util_pct < 50:
        idle_mins = available_kitchen_mins * 0.6 - kitchen["load_mins"]
        recs.append(Rec("Operations", 3, "Kitchen",
            f"Kitchen only {util_pct:.0f}% utilised — add higher-margin items",
            f"Your kitchen has capacity headroom. Each idle kitchen minute is worth "
            f"${sh_k:.2f} in potential profit (shadow price).",
            f"Add 2–3 high-profit-per-minute items. "
            f"Target dishes earning >${eng_df['Profit/Min ($)'].quantile(0.75):.2f}/min.",
            sh_k * max(0, idle_mins) / 60 * 0.5,
            "Kitchen shadow price + LP capacity analysis (Taha Ch. 2–3, 18)"))

    bn = kitchen["bottleneck"]
    recs.append(Rec("Operations", 2, bn,
        f"'{bn}' is your kitchen bottleneck",
        f"This dish consumes the most kitchen minutes per week. A prep-time reduction of "
        f"2 min per order frees {eng_df.shape[0] * 2:.0f}+ kitchen minutes per week, "
        f"worth ${sh_k * eng_df.shape[0] * 2:.0f} at current shadow price.",
        f"Batch-prep {bn} components during off-peak hours. Target: cut prep by 2–3 min/order.",
        sh_k * eng_df.shape[0] * 2,
        "Bottleneck analysis + kitchen shadow price (Taha Ch. 18, 3)"))

    pm_thresh = eng_df["Profit/Min ($)"].quantile(0.25)
    for _, row in eng_df[eng_df["Profit/Min ($)"] < pm_thresh].iterrows():
        if row["Dish"] == bn: continue
        recs.append(Rec("Operations", 3, row["Dish"],
            f"Low efficiency: ${row['Profit/Min ($)']:.2f}/min cook time",
            f"At ${row['Profit/Min ($)']:.2f}/kitchen-minute, {row['Dish']} is in the bottom "
            f"quartile. The shadow price of kitchen time is ${sh_k:.2f}/min.",
            f"Simplify recipe or pre-portion to cut {row['Prep (min)']+row['Cook (min)']:.0f} "
            f"min by 20%. Frees ${sh_k*(row['Prep (min)']+row['Cook (min)'])*0.2*row['Sold/Wk']:.0f}/wk.",
            sh_k * (row["Prep (min)"] + row["Cook (min)"]) * 0.15 * row["Sold/Wk"],
            "Kitchen efficiency + LP shadow price (Taha Ch. 3, 18)"))

    # ── BUNDLE ────────────────────────────────────────────────────────────
    for _, row in eng_df.iterrows():
        if row["Bundle Attach %"] >= 30:
            est_extra = row["Bundle Sales/Wk"] * row["True Profit ($)"] * 0.4
            recs.append(Rec("Bundle", 3, row["Dish"],
                f"{row['Bundle Attach %']}% bundle rate — formalise upsell",
                f"{row['Dish']} already attaches to {int(row['Bundle Attach %'])}% of orders. "
                f"A structured add-on prompt could lift that to "
                f"{min(55, int(row['Bundle Attach %'])+10)}%.",
                f"Create a '{row['Dish']} + complement' combo at a $1.50 bundle premium. "
                f"Est. extra: ${est_extra:.0f}/wk.",
                est_extra,
                "Bundle attach rate analysis + demand model"))

    # ── FINANCIAL HEALTH ─────────────────────────────────────────────────
    if fin["safety"] < 15:
        recs.append(Rec("Operations", 1, "All Dishes",
            f"Safety buffer only {fin['safety']:.1f}% — near breakeven",
            f"You need {fin['bep']:.0f} orders/week to cover all costs. You have {fin['covers']:.0f}. "
            f"Monte Carlo P5 (${mc['p5']:,.0f} net) shows a bad week could put you below breakeven.",
            f"Increase prices on Plow Horses by $0.50 each, target "
            f"+{int(max(0, fin['bep'] - fin['surplus']))} extra orders/wk.",
            abs(fin["net"] * 0.15),
            "Safety buffer analysis + Monte Carlo risk (Taha Ch. 19)"))

    if fin["fc_pct"] > 35:
        recs.append(Rec("Pricing", 2, "All Dishes",
            f"Average true food cost {fin['fc_pct']:.1f}% — above 32% target",
            f"Across all dishes, your true food cost is {fin['fc_pct']:.1f}%. "
            f"Each percentage point above 32% costs ~${fin['rev'] * 0.01:,.0f}/week.",
            f"Address top 3 offenders. Reaching 32% adds "
            f"${(fin['fc_pct'] - 32) * fin['rev'] / 100:,.0f}/week.",
            (fin["fc_pct"] - 32) * fin["rev"] / 100 if fin["fc_pct"] > 32 else 0,
            "True cost model + LP resource analysis (Taha Ch. 3, 13)"))

    return sorted(recs, key=lambda r: (r.priority, -r.impact))


# ═════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def kcard(lbl, val, sub="", cls="gold"):
    return (f'<div class="kpi"><div class="kpi-l">{lbl}</div>'
            f'<div class="kpi-v {cls}">{val}</div>'
            f'<div class="kpi-s">{sub}</div></div>')

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
    for l, ax, ay, xa, ya in [
        ("⭐ STARS",       eng_df["Sold/Wk"].max()*1.0, eng_df["True Profit ($)"].max()*1.0, "right","top"),
        ("🧩 PUZZLES",     eng_df["Sold/Wk"].min()*.88, eng_df["True Profit ($)"].max()*1.0, "left","top"),
        ("🐎 PLOW HORSES", eng_df["Sold/Wk"].max()*1.0, eng_df["True Profit ($)"].min()*.88, "right","bottom"),
        ("🐕 DOGS",        eng_df["Sold/Wk"].min()*.88, eng_df["True Profit ($)"].min()*.88, "left","bottom"),
    ]:
        fig.add_annotation(x=ax, y=ay, text=l, showarrow=False,
                           font=dict(size=8, color="rgba(255,255,255,.14)"),
                           xanchor=xa, yanchor=ya)
    fig.update_layout(**DARK, height=380,
                      title="Menu Engineering Matrix (Taha Ch. 15)",
                      xaxis_title="Orders / Week", yaxis_title="True Profit / Dish ($)")
    return fig

def fig_profit_min(eng_df):
    sdf = eng_df.sort_values("Profit/Min ($)", ascending=False)
    fig = go.Figure(go.Bar(
        x=sdf["Dish"], y=sdf["Profit/Min ($)"],
        marker=dict(color=sdf["Profit/Min ($)"].tolist(),
                    colorscale=[[0,RED],[0.4,AMBER],[1,GREEN]],
                    showscale=False, line=dict(width=0)),
        text=[f"${v:.2f}" for v in sdf["Profit/Min ($)"]],
        textposition="outside", textfont=dict(size=10, color="#8A9AC8")))
    fig.update_layout(**DARK, height=260, xaxis_tickangle=-25,
                      title="Profit per kitchen minute — LP efficiency metric (Taha Ch. 3)")
    return fig

def fig_pl_waterfall(fin):
    fig = go.Figure(go.Waterfall(
        orientation="v",
        x=["Total Sales","− Ingredient Cost","− Wastage & Labour","− Fixed Costs","Net Profit"],
        y=[fin["rev"],
           -(fin["rev"] * fin["ing_pct"] / 100),
           -(fin["pool"] - fin["rev"] * (1 - fin["ing_pct"]/100)),
           -fin["fixed"], fin["net"]],
        connector={"line":{"color":"rgba(255,255,255,.05)"}},
        increasing={"marker":{"color":GREEN}},
        decreasing={"marker":{"color":RED}},
        totals={"marker":{"color":GOLD}},
        text=[f"${abs(v):,.0f}" for v in [fin["rev"],
               -(fin["rev"]*fin["ing_pct"]/100),
               -(fin["pool"]-fin["rev"]*(1-fin["ing_pct"]/100)),
               -fin["fixed"], fin["net"]]],
        textposition="outside", textfont=dict(size=10, color="#8A9AC8")))
    fig.update_layout(**DARK, title="True P&L Waterfall", height=280)
    return fig

def fig_mc(mc):
    bins = np.linspace(mc["sims"].min(), mc["sims"].max(), 65)
    cnts, edges = np.histogram(mc["sims"], bins=bins)
    mid = (edges[:-1]+edges[1:])/2
    fig = go.Figure(go.Bar(x=mid, y=cnts,
                           marker=dict(color=mid,
                                       colorscale=[[0,RED],[.35,AMBER],[.65,BLUE],[1,GREEN]],
                                       showscale=False, line=dict(width=0)),
                           opacity=.85))
    for v, l, c in [(mc["p5"],"P5  (bad week)",RED),(mc["p50"],"P50 (median)",GOLD),
                    (mc["p95"],"P95  (good week)",GREEN)]:
        fig.add_vline(x=v, line_color=c, line_dash="dash", line_width=1.5,
                      annotation_text=f"{l}  ${v:,.0f}",
                      annotation_font=dict(color=c, size=9))
    fig.update_layout(**DARK, height=260,
                      title="Monte Carlo — 8,000 demand simulations (Taha Ch. 19)",
                      xaxis_title="Weekly Net Profit ($)", yaxis_title="Frequency")
    return fig

def fig_shadow(sh_k, sh_b, sh_s):
    fig = go.Figure(go.Bar(
        x=["1 extra kitchen min", "1 extra $1 budget", "1 extra table min"],
        y=[sh_k, sh_b, sh_s],
        marker=dict(color=[BLUE, GOLD, PURPLE], line=dict(width=0)),
        text=[f"${sh_k:.3f}", f"${sh_b:.4f}", f"${sh_s:.3f}"],
        textposition="outside", textfont=dict(size=11, color="#8A9AC8")))
    fig.update_layout(**DARK, height=220,
                      title="LP Shadow Prices — marginal value of each constraint (Taha Ch. 3)",
                      yaxis_title="Extra weekly profit ($)")
    return fig

def fig_eoq_table(eng_df, eoq_data):
    rows_d = []
    for i, (_, row) in enumerate(eng_df.iterrows()):
        eq = eoq_data[i]
        status = "🔴 Stockout Risk" if eq["stockout"] else ("🟡 Overstock" if eq["overstock"] else "✅ OK")
        rows_d.append({
            "Dish":              row["Dish"],
            "Current Stock":     int(row["Stock"]),
            "EOQ (units)":       int(eq["eoq"]),
            "Safety Stock":      int(eq["safety_stock"]),
            "Reorder Point":     int(eq["rop"]),
            "Order Cycle (days)":eq["cycle_days"],
            "Annual Saving ($)": f"${eq['annual_saving']:,.0f}",
            "Status":            status,
        })
    return pd.DataFrame(rows_d)

def fig_util_gauge(util_pct):
    color = GREEN if util_pct < 70 else (AMBER if util_pct < 85 else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=util_pct,
        title={"text":"Kitchen Utilisation %","font":{"size":11,"color":"#8A9AC8"}},
        number={"suffix":"%","font":{"size":20,"color":color}},
        gauge=dict(
            axis=dict(range=[0,100], tickcolor="#222"),
            bar=dict(color=color, thickness=0.22),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[{"range":[0,70],"color":"rgba(53,200,134,.08)"},
                   {"range":[70,85],"color":"rgba(240,160,48,.08)"},
                   {"range":[85,100],"color":"rgba(220,69,69,.08)"}],
            threshold=dict(line=dict(color=color,width=2),thickness=.75,value=util_pct))))
    fig.update_layout(**DARK, height=190)
    return fig

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
    fig.update_layout(**DARK, height=250, xaxis_tickangle=-25,
                      title="True food cost % per dish (ingredient + wastage + labour ÷ price)")
    return fig

def fig_price_compare(rows, opt_prices, eng_df):
    names = [r["Dish"] for r in rows]
    curr  = [r["Price ($)"] for r in rows]
    opt   = [round(p * 4)/4 for p in opt_prices]
    tcs   = [true_cost(r) for r in rows]
    fig   = go.Figure()
    fig.add_trace(go.Bar(name="True Cost",       x=names, y=tcs,  marker_color=RED,  opacity=.65))
    fig.add_trace(go.Bar(name="Current Price",   x=names, y=curr, marker_color=BLUE, opacity=.75))
    fig.add_trace(go.Bar(name="Suggested Price", x=names, y=opt,  marker_color=GOLD, opacity=.9))
    fig.update_layout(**DARK, barmode="group", height=280, xaxis_tickangle=-25,
                      title="True cost vs. current vs. suggested price",
                      yaxis_title="$ per dish")
    return fig

def fig_clv(eng_df):
    clvs = []
    for _, row in eng_df.iterrows():
        clv, ret = markov_ltv(row["Rating"], row["True Profit ($)"])
        clvs.append({"Dish": row["Dish"], "CLV ($)": clv, "Retention %": ret})
    cdf = pd.DataFrame(clvs).sort_values("CLV ($)", ascending=False)
    fig = go.Figure(go.Bar(
        x=cdf["Dish"], y=cdf["CLV ($)"],
        marker=dict(color=cdf["CLV ($)"].tolist(),
                    colorscale=[[0,RED],[.5,AMBER],[1,GREEN]],
                    showscale=False, line=dict(width=0)),
        text=[f"${v:,.0f}" for v in cdf["CLV ($)"]],
        textposition="outside", textfont=dict(size=10, color="#8A9AC8")))
    fig.update_layout(**DARK, height=250, xaxis_tickangle=-25,
                      title="Customer Lifetime Value by dish — Markov model (Taha Ch. 17)",
                      yaxis_title="Estimated CLV ($)")
    return fig

# ═════════════════════════════════════════════════════════════════════════════
#  HTML REPORT GENERATOR
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
        st = "🔴 Reorder Now" if eq["stockout"] else ("🟡 Overstock" if eq["overstock"] else "✅ OK")
        inv_rows += f"""<tr>
          <td><b>{row['Dish']}</b></td>
          <td>{int(row['Stock'])}</td><td>{int(eq['eoq'])}</td>
          <td>{int(eq['safety_stock'])}</td><td>{int(eq['rop'])}</td>
          <td>{eq['cycle_days']:.0f}</td>
          <td>${eq['annual_saving']:,.0f}</td><td>{st}</td>
        </tr>"""

    rec_html = ""
    pri_map  = {1:"🔴 CRITICAL",2:"🟡 HIGH",3:"🟢 MEDIUM"}
    for r in recs[:15]:
        rec_html += f"""
        <div style="border-left:3px solid {'#DC4545' if r.priority==1 else '#F0A030' if r.priority==2 else '#35C886'};
                    background:#0C1020;border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:10px;font-size:13px">
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:5px;flex-wrap:wrap">
            <span style="font-size:10px;font-weight:700;color:{'#DC4545' if r.priority==1 else '#F0A030' if r.priority==2 else '#35C886'}">{pri_map[r.priority]}</span>
            <span style="font-size:10px;color:#D2A248;background:rgba(210,162,72,.1);padding:1px 7px;border-radius:20px">{r.cat.upper()}</span>
            <span style="font-size:10px;color:#4A5578">{r.dish}</span>
            <span style="margin-left:auto;font-weight:700;color:#35C886">+${r.impact:.0f}/wk</span>
          </div>
          <div style="font-weight:600;color:#DEE2F0;margin-bottom:4px">{r.title}</div>
          <div style="color:#6070A0;line-height:1.6;margin-bottom:5px">{r.insight}</div>
          <div style="color:#D2A248;font-weight:500">→ {r.action}</div>
          <div style="font-size:10px;color:#2A3550;margin-top:5px;font-style:italic">OR basis: {r.or_ref}</div>
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
th{{padding:9px 9px;text-align:left;color:#4A5578;font-weight:600;font-size:.65rem;
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
  <p>{period} Report &nbsp;·&nbsp; {rdate} &nbsp;·&nbsp; OR-Powered Restaurant Intelligence</p>
</div>
<div class="sec"><h2>Financial Summary</h2>
<div class="kgrid">
  <div class="kbox"><div class="l">Revenue / {plab}</div><div class="v">${fin['rev']*factor:,.0f}</div></div>
  <div class="kbox {'g' if fin['net']>0 else 'r'}"><div class="l">Net Profit / {plab}</div><div class="v">${fin['net']*factor:,.0f}</div></div>
  <div class="kbox {'g' if gain>0 else ''}"><div class="l">Extra if Repriced</div><div class="v">${gain*factor:+,.0f}</div><div class="s">per {plab.lower()}</div></div>
  <div class="kbox {'r' if fin['fc_pct']>35 else 'g'}"><div class="l">True Food Cost %</div><div class="v">{fin['fc_pct']:.1f}%</div><div class="s">target &lt;32%</div></div>
  <div class="kbox {'g' if fin['safety']>25 else 'r'}"><div class="l">Safety Buffer</div><div class="v">{fin['safety']:.1f}%</div></div>
  <div class="kbox b"><div class="l">MC P5 Risk</div><div class="v">${mc['p5']:,.0f}</div><div class="s">bad week scenario</div></div>
  <div class="kbox"><div class="l">Annual Profit</div><div class="v">${fin['annual']:,.0f}</div></div>
  <div class="kbox"><div class="l">Breakeven</div><div class="v">{fin['bep']:.0f}</div><div class="s">orders/wk needed</div></div>
</div></div>
<div class="sec"><h2>OR Analysis</h2>
<div class="ibox"><strong>LP Shadow Prices (Taha Ch. 3):</strong><br>
Each extra kitchen minute = <strong>${sh_k:.3f}</strong> profit &nbsp;|&nbsp;
Each extra $1 ingredient budget = <strong>${sh_b:.4f}</strong> profit &nbsp;|&nbsp;
Kitchen utilisation = <strong>{kitchen['util_pct']:.0f}%</strong><br><br>
<strong>Monte Carlo (Taha Ch. 19) — 8,000 simulations:</strong><br>
P5 = <strong>${mc['p5']:,.0f}</strong> &nbsp;|&nbsp; P50 = <strong>${mc['p50']:,.0f}</strong> &nbsp;|&nbsp;
P95 = <strong>${mc['p95']:,.0f}</strong></div></div>
<div class="sec"><h2>Full Dish Breakdown</h2>
<table><thead><tr>
  <th>Dish</th><th>Cat</th><th>Ingr Cost</th><th>True Cost</th>
  <th>Price</th><th>Suggested $</th><th>True Profit</th>
  <th>$/Min</th><th>Orders/Wk</th><th>Wk Profit</th><th>FC%</th><th>Type</th>
</tr></thead><tbody>{dish_rows}</tbody></table></div>
<div class="sec"><h2>Inventory — EOQ Analysis (Taha Ch. 13)</h2>
<table><thead><tr>
  <th>Dish</th><th>Stock</th><th>EOQ</th><th>Safety Stock</th>
  <th>Reorder Pt</th><th>Cycle (days)</th><th>Annual Saving</th><th>Status</th>
</tr></thead><tbody>{inv_rows}</tbody></table></div>
<div class="sec"><h2>Top Recommendations</h2>{rec_html}</div>
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
    <div class="b-sub">Restaurant profit intelligence · 10 OR models · 18 input variables</div>
  </div>
  <div class="b-badges">
    <span class="b-badge">LP · SHADOW PRICES</span>
    <span class="b-badge">EOQ · INVENTORY</span>
    <span class="b-badge">MONTE CARLO</span>
    <span class="b-badge">MARKOV CLV</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Business Parameters")
    weekly_fixed           = st.number_input("Fixed costs / week ($)",       value=3500, step=100)
    available_kitchen_mins = st.number_input("Kitchen minutes / week",        value=4800, step=60,
                                             help="Shift hours × number of chefs × 60")
    ing_budget             = st.number_input("Ingredient budget / week ($)",  value=4000, step=100)
    st.divider()
    st.markdown("### 📐 Optimiser Rules")
    min_mg = st.slider("Min true margin %",        20, 60, 28)
    max_up = st.slider("Max price increase %",      5, 50, 35)
    st.divider()
    st.markdown("### 📦 Inventory Settings")
    service_lvl  = st.slider("Inventory service level %", 80, 99, 95)
    demand_sigma = st.slider("Demand std dev %",           5, 30, 15,
                             help="Uncertainty in weekly demand")
    st.divider()
    st.markdown("### 📂 Upload Your Menu")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    imported_df = None
    if uploaded:
        try:
            imported_df = pd.read_csv(uploaded)
            miss = [c for c in DEFAULT.keys() if c not in imported_df.columns]
            if miss: st.error(f"Missing columns: {miss}"); imported_df = None
            else:    st.success("✓ Loaded")
        except Exception as e:
            st.error(str(e))
    st.divider()
    st.caption(
        "**Elasticity guide**\n\n"
        "−0.3 – −0.6 : Drinks, coffee\n\n"
        "−0.7 – −1.0 : Typical mains\n\n"
        "−1.1 – −1.5 : Price-sensitive items"
    )

Z_SCORES = {80:1.282, 85:1.036, 90:1.282, 95:1.645, 99:2.326}
svc_z    = Z_SCORES.get(service_lvl, 1.645)

# ═════════════════════════════════════════════════════════════════════════════
#  MENU TABLE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sh">📋 Your Menu — edit any cell, add rows with ＋</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tip">'
    '18 variables per dish power 10 OR models. True Cost = ingredient × (1+wastage%) + kitchen labour. '
    'All OR calculations update instantly when you re-run the analysis.'
    '</div>', unsafe_allow_html=True)

base_df = pd.DataFrame(imported_df if imported_df is not None else DEFAULT)
menu_df = st.data_editor(
    base_df, use_container_width=True, num_rows="dynamic",
    column_config={
        "Ingredient ($)":   st.column_config.NumberColumn("Ingredient ($)",   format="$%.2f", min_value=0.0),
        "Price ($)":        st.column_config.NumberColumn("Price ($)",        format="$%.2f", min_value=0.0),
        "Sold/Wk":          st.column_config.NumberColumn("Sold / Week",      min_value=0),
        "Prep (min)":       st.column_config.NumberColumn("Prep (min)",       min_value=0),
        "Cook (min)":       st.column_config.NumberColumn("Cook (min)",       min_value=1),
        "Table Time (min)": st.column_config.NumberColumn("Table Time (min)", min_value=5),
        "Wastage %":        st.column_config.NumberColumn("Wastage %",        format="%.0f%%", min_value=0, max_value=50),
        "Labour $/hr":      st.column_config.NumberColumn("Labour $/hr",      format="$%.0f",  min_value=0),
        "Elasticity":       st.column_config.NumberColumn("Elasticity",       format="%.1f",   min_value=-3.0, max_value=-0.1),
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
        st.warning(f"⚠️ Ingredient cost ≥ price: {', '.join(bad['Dish'].tolist())} — fix before running.")

c_run, c_exp = st.columns([3,1])
with c_run:
    run_btn = st.button("🚀  Run Full Analysis — LP + EOQ + Monte Carlo + All 10 OR Models")
with c_exp:
    if st.session_state.eng_df is not None:
        buf = io.StringIO()
        st.session_state.eng_df.to_csv(buf, index=False)
        st.download_button("⬇️ CSV", buf.getvalue(), "menuprofit.csv", "text/csv")

# ═════════════════════════════════════════════════════════════════════════════
#  RUN ENGINE
# ═════════════════════════════════════════════════════════════════════════════
if run_btn:
    if menu_df.empty: st.error("Add at least one dish."); st.stop()
    with st.spinner("Running LP · EOQ · Monte Carlo · Markov · Queuing analysis…"):
        rows = menu_df.to_dict("records")

        eng_df, avg_p, avg_q = classify_menu(menu_df)
        fin    = calc_fin(menu_df, eng_df, weekly_fixed)
        _, sh_k, sh_b, sh_s  = run_lp(menu_df, available_kitchen_mins, ing_budget)
        opt_p, _ = optimise_prices(rows, weekly_fixed, min_mg/100, max_up/100)
        mc     = monte_carlo(menu_df, fixed=weekly_fixed, sigma=demand_sigma/100)
        eoq_data = [eoq_model(row, demand_sigma/100, svc_z) for row in rows]
        kitchen  = kitchen_analysis(menu_df, available_kitchen_mins)
        recs     = generate_recommendations(
            menu_df, eng_df, fin, opt_p, sh_k, sh_b, sh_s,
            mc, eoq_data, kitchen, weekly_fixed, available_kitchen_mins)

        st.session_state.update(
            opt_prices=opt_p, eng_df=eng_df, fin=fin, mc=mc,
            sh_k=sh_k, sh_b=sh_b, sh_s=sh_s,
            kitchen=kitchen, eoq_data=eoq_data, recs=recs, ran=True)

    st.success(f"✓ Analysis complete — {len(recs)} recommendations generated")

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
        "💰  Price Calculator",
        "💡  Recommendations",
        "📄  Reports",
    ])

    # ── TAB 1: OVERVIEW ──────────────────────────────────────────────────
    with T1:
        st.markdown('<div class="sh">Business Health at a Glance</div>', unsafe_allow_html=True)

        c = st.columns(4)
        for col, (lbl, val, sub, cls) in zip(c, [
            ("Weekly Revenue",  f"${fin['rev']:,.0f}",   "total sales",           "white"),
            ("True Net Profit", f"${fin['net']:,.0f}",   "after all costs",        "green" if fin["net"]>0 else "red"),
            ("Annual Profit",   f"${fin['annual']:,.0f}","×52 weeks",              "gold"),
            ("True Food Cost",  f"{fin['fc_pct']:.1f}%", "incl wastage + labour", "green" if fin["fc_pct"]<=32 else "red"),
        ]):
            col.markdown('<div class="krow">'+kcard(lbl,val,sub,cls)+'</div>', unsafe_allow_html=True)

        c2 = st.columns(4)
        for col, (lbl, val, sub, cls) in zip(c2, [
            ("Safety Buffer",       f"{fin['safety']:.1f}%",      "above breakeven",    "green" if fin["safety"]>25 else "red"),
            ("MC Risk (P5)",        f"${mc['p5']:,.0f}",           "bad-week floor",     "blue"),
            ("Kitchen Utilisation", f"{kitchen['util_pct']:.0f}%", "of available time",  "green" if kitchen["util_pct"]<75 else "red"),
            ("Stockout Alerts",     f"{stockouts}",                "dishes below ROP",   "red" if stockouts>0 else "green"),
        ]):
            col.markdown('<div class="krow">'+kcard(lbl,val,sub,cls)+'</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns([3,2])
        with col_l:
            _, avg_p2, avg_q2 = classify_menu(menu_df)
            st.plotly_chart(fig_bcg(eng_df, avg_p2, avg_q2), use_container_width=True)
        with col_r:
            st.plotly_chart(fig_pl_waterfall(fin), use_container_width=True)
            st.plotly_chart(fig_util_gauge(kitchen["util_pct"]), use_container_width=True)

        st.plotly_chart(fig_profit_min(eng_df), use_container_width=True)

        c_fc, c_clv = st.columns(2)
        with c_fc:  st.plotly_chart(fig_food_cost_bars(eng_df), use_container_width=True)
        with c_clv: st.plotly_chart(fig_clv(eng_df), use_container_width=True)

        st.markdown('<div class="sh">Complete Dish Intelligence Table</div>', unsafe_allow_html=True)
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

    # ── TAB 2: PRICE CALCULATOR ───────────────────────────────────────────
    with T2:
        st.markdown('<div class="sh">Price Optimisation Engine</div>', unsafe_allow_html=True)

        binding = "kitchen time" if sh_k > sh_b * 8 else "ingredient budget"
        st.markdown(
            f'<div class="ibox">'
            f'<strong>LP Analysis (Taha Ch. 2–3):</strong> Binding constraint: <strong>{binding}</strong>. '
            f'Shadow prices — kitchen min: <strong>${sh_k:.3f}/wk</strong> · '
            f'ingredient $1: <strong>${sh_b:.4f}/wk</strong> · '
            f'seating min: <strong>${sh_s:.4f}/wk</strong>.<br>'
            f'Differential evolution optimiser found prices adding '
            f'<strong>${wk_gain:+,.0f}/wk</strong> (${wk_gain*52:+,.0f}/yr).'
            f'</div>', unsafe_allow_html=True)

        pc = st.columns(3)
        pc[0].markdown('<div class="krow">'+kcard("Current Weekly Profit",  f"${curr_profit:,.0f}", "", "white")+'</div>', unsafe_allow_html=True)
        pc[1].markdown('<div class="krow">'+kcard("Optimised Weekly Profit", f"${opt_profit:,.0f}",  "", "green")+'</div>', unsafe_allow_html=True)
        pc[2].markdown('<div class="krow">'+kcard("Extra Per Year", f"${wk_gain*52:+,.0f}", "if repriced today", "gold")+'</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(fig_shadow(sh_k, sh_b, sh_s), use_container_width=True)
        with c2: st.plotly_chart(fig_price_compare(rows, opt_prices, eng_df), use_container_width=True)

        st.plotly_chart(fig_mc(mc), use_container_width=True)

        st.markdown("#### Suggested Price Table")
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
                "Dish":            r["Dish"],
                "Type":            f"{QUAD_I[eng_df.iloc[i]['Type']]} {eng_df.iloc[i]['Type']}",
                "True Cost":       f"${tc:.2f}",
                "Current Price":   f"${p_c:.2f}",
                "Suggested Price": f"${p_o:.2f}",
                "Price Change":    f"{chg:+.1f}%",
                "Current $/Min":   f"${pm_c:.2f}",
                "New $/Min":       f"${pm_o:.2f}",
                "Extra/Week":      f"${cm_o-cm_c:+,.0f}",
                "Extra/Year":      f"${(cm_o-cm_c)*52:+,.0f}",
            })
        st.dataframe(pd.DataFrame(price_rows), use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="tip">$/Min = profit per kitchen minute — the LP efficiency metric.</div>',
            unsafe_allow_html=True)

        st.markdown('<div class="sh">Inventory — Economic Order Quantity (Taha Ch. 13)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="ibox"><strong>EOQ Model:</strong> EOQ = √(2DS/H). '
            f'Reorder Point = lead-time demand + safety stock (z×σ×√lead_time). '
            f'Safety stock uses z={svc_z:.3f} for {service_lvl}% service level.</div>',
            unsafe_allow_html=True)
        st.dataframe(fig_eoq_table(eng_df, eoq_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 🎚️ Interactive Price Tester")
        sel   = st.selectbox("Dish to test", menu_df["Dish"].tolist(), key="pt")
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
        m1.metric("Est. Orders/Wk", f"{new_qty:.0f}", f"{new_qty-sel_r['Sold/Wk']:+.0f}")
        m2.metric("Weekly Profit",  f"${new_cm:,.0f}", f"${new_cm-base_cm:+,.0f}")
        m3.metric("True Margin",    f"{mg_pct:.1f}%")
        m4.metric("Extra/Year",     f"${(new_cm-base_cm)*52:+,.0f}")
        m5.metric("Customer CLV",   f"${clv:,.0f}", f"Retention {ret}%")

    # ── TAB 3: RECOMMENDATIONS ────────────────────────────────────────────
    with T3:
        st.markdown('<div class="sh">Smart Recommendations — Algorithmic, OR-Grounded</div>',
                    unsafe_allow_html=True)

        st.markdown(
            f'<div class="psummary">'
            f'<div class="pchip crit"><div class="pnum crit">{critical_n}</div><div class="plbl">🔴 Critical</div></div>'
            f'<div class="pchip high"><div class="pnum high">{high_n}</div><div class="plbl">🟡 High Priority</div></div>'
            f'<div class="pchip med"><div class="pnum med">{med_n}</div><div class="plbl">🟢 Opportunities</div></div>'
            f'<div class="pchip" style="background:rgba(210,162,72,.08);border:1px solid rgba(210,162,72,.2)">'
            f'<div class="pnum gold">${total_uplift:,.0f}</div><div class="plbl">Est. Weekly Uplift</div></div>'
            f'</div>', unsafe_allow_html=True)

        cats = ["All"] + sorted(set(r.cat for r in recs))
        filt = st.selectbox("Filter by category", cats, key="rc_filt")
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
              <div class="rec-or">OR basis: {r.or_ref}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="ibox" style="margin-top:1rem">
          <strong>How recommendations are generated:</strong><br>
          • <strong>Pricing</strong> — LP shadow prices + price elasticity (Taha Ch. 3, 10)<br>
          • <strong>Inventory</strong> — EOQ + safety stock at {service_lvl}% service level (Taha Ch. 13)<br>
          • <strong>Menu Engineering</strong> — Decision analysis quadrant (Taha Ch. 15)<br>
          • <strong>Operations</strong> — M/G/1 kitchen utilisation + shadow price (Taha Ch. 3, 18)<br>
          • <strong>Bundle</strong> — Attach rate analysis + demand uplift estimation
        </div>""", unsafe_allow_html=True)

    # ── TAB 4: REPORTS ────────────────────────────────────────────────────
    with T4:
        st.markdown('<div class="sh">Download Professional Report</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="tip">Full HTML report: true cost · LP shadow prices · EOQ schedule · '
            'Monte Carlo risk · Markov CLV · all recommendations. '
            'Open in browser → Ctrl+P → Save as PDF.</div>', unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        with rc1: period = st.radio("Period", ["Weekly","Monthly"], horizontal=True)
        with rc2: rdate  = str(st.date_input("Report date", datetime.date.today()))

        factor = 4.33 if period=="Monthly" else 1
        plab   = "month" if period=="Monthly" else "week"

        st.markdown(f"**Preview — {period} figures:**")
        rv = st.columns(5)
        rv[0].metric(f"Revenue/{plab}",    f"${fin['rev']*factor:,.0f}")
        rv[1].metric(f"Net Profit/{plab}", f"${fin['net']*factor:,.0f}")
        rv[2].metric("Annual Profit",       f"${fin['annual']:,.0f}")
        rv[3].metric("Extra if Repriced",   f"${wk_gain*factor:+,.0f}/{plab}")
        rv[4].metric("MC 90% Range",        f"${mc['p5']*factor:,.0f}–${mc['p95']*factor:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Breakeven Sensitivity — what if fixed costs change?")
        sce = []
        for mult in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
            fc   = weekly_fixed * mult
            bep  = fc / fin["avg_cm"] if fin["avg_cm"] > 0 else 0
            net_w= (fin["pool"] - fc) * factor
            sce.append({
                f"Fixed Costs/{plab.capitalize()}": f"${fc*factor:,.0f}",
                "Orders to Break Even":             f"{bep:.0f}",
                "Your Surplus Orders":              f"{fin['covers']-bep:+.0f}",
                f"Net Profit/{plab.capitalize()}":  f"${net_w:+,.0f}",
                "Annual Profit":                    f"${(fin['pool']-fc)*52:+,.0f}",
            })
        st.dataframe(pd.DataFrame(sce), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📄  Generate & Download Full Report"):
            html = generate_report(
                menu_df, eng_df, fin, opt_prices, mc,
                sh_k, sh_b, sh_s, eoq_data, kitchen, recs, period, rdate)
            st.download_button(
                f"⬇️  Download {period} Report (HTML)",
                data=html,
                file_name=f"MenuProfit_{period}_{rdate}.html",
                mime="text/html")
            st.success("Ready — click the download button above.")

    st.markdown("""
    <div style="text-align:center;color:#1E2840;font-size:.72rem;
                padding:20px 0 6px;border-top:1px solid rgba(255,255,255,.04);margin-top:18px">
      MenuProfit Pro &nbsp;·&nbsp; LP · EOQ · Monte Carlo · Markov · Queuing &nbsp;·&nbsp;
      10 OR Models · 18 Input Variables
    </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:52px 16px;color:#2E3858">
      <div style="font-size:2.5rem;margin-bottom:14px">🍽️</div>
      <div style="font-size:1rem;line-height:1.9">
        Edit your menu above, then tap<br>
        <strong style="color:#D2A248">🚀 Run Full Analysis</strong><br>
        LP · EOQ · Monte Carlo · Markov CLV · Kitchen Queuing — all in &lt;3 seconds.
      </div>
    </div>""", unsafe_allow_html=True)
