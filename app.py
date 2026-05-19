"""
CafeIQ Pro — Cafe & Restaurant Optimization Dashboard
Stack: Streamlit · Pandas · NumPy · Plotly · SQLite / Supabase

Run:
    pip install streamlit pandas numpy plotly openpyxl
    streamlit run app.py

Optional Supabase (production DB):
    pip install supabase
    Set env vars: SUPABASE_URL  SUPABASE_KEY
"""

import io, os, math, time, datetime, random
from pathlib import Path
import sqlite3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CafeIQ Pro",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
#  THEME
# ════════════════════════════════════════════════════════════════════════════

AMBER  = "#F59E0B"; GREEN  = "#10B981"; RED   = "#F43F5E"
BLUE   = "#38BDF8"; PURPLE = "#A78BFA"; ORANG = "#FB923C"
TEAL   = "#2DD4BF"; SLATE  = "#64748B"

_CHART = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, system-ui, sans-serif", color="#64748B", size=11),
    margin=dict(l=14, r=14, t=42, b=14),
)
_CFG = dict(config={"displayModeBar": False, "scrollZoom": False})

def _chart(fig, h=270):
    fig.update_layout(**_CHART, height=h)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig

# ════════════════════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
*,html,body{box-sizing:border-box}
html,body,[class*="css"]{font-family:'DM Sans',system-ui,sans-serif;font-size:14px}
[data-testid="stAppViewContainer"]{background:#040812;color:#E2E8F0}
[data-testid="stSidebar"]{background:#070B18!important;border-right:1px solid rgba(245,158,11,.10)!important}
[data-testid="stHeader"],[data-testid="stToolbar"]{display:none!important}

.banner{background:linear-gradient(135deg,#060B1A 0%,#0A1020 60%,#060B1A 100%);
  border-bottom:1px solid rgba(245,158,11,.14);padding:20px 28px 16px;
  margin:-1rem -1rem 1.8rem;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.b-title{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:700;color:#fff;line-height:1}
.b-title span{color:#F59E0B}
.b-sub{font-size:.73rem;color:#334155;margin-top:5px;letter-spacing:.04em}
.b-right{margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.chip{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.18);
  border-radius:20px;padding:5px 13px;font-size:.67rem;color:#F59E0B;font-weight:600;
  letter-spacing:.07em;white-space:nowrap;display:flex;align-items:center;gap:5px}
.dot{width:6px;height:6px;border-radius:50%}
.dot-g{background:#10B981;animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

.sh{font-family:'Playfair Display',serif;font-size:1.12rem;font-weight:600;color:#F59E0B;
  border-bottom:1px solid rgba(245,158,11,.10);padding-bottom:8px;margin:1.6rem 0 1rem}

.kpi{background:linear-gradient(145deg,#0C1122,#070B18);border:1px solid rgba(255,255,255,.05);
  border-radius:14px;padding:16px 18px;position:relative;overflow:hidden;height:100%}
.kpi::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:14px 14px 0 0}
.kpi.ca::after{background:linear-gradient(90deg,#F59E0B,transparent)}
.kpi.cg::after{background:linear-gradient(90deg,#10B981,transparent)}
.kpi.cr::after{background:linear-gradient(90deg,#F43F5E,transparent)}
.kpi.cb::after{background:linear-gradient(90deg,#38BDF8,transparent)}
.kpi.cp::after{background:linear-gradient(90deg,#A78BFA,transparent)}
.kpi.co::after{background:linear-gradient(90deg,#FB923C,transparent)}
.kpi.ct::after{background:linear-gradient(90deg,#2DD4BF,transparent)}
.kpi-l{font-size:.65rem;color:#334155;text-transform:uppercase;letter-spacing:.1em;font-weight:600;margin-bottom:8px}
.kpi-v{font-size:1.65rem;font-weight:600;font-family:'DM Mono',monospace;line-height:1;color:#E2E8F0}
.kpi-v.ca{color:#F59E0B}.kpi-v.cg{color:#10B981}.kpi-v.cr{color:#F43F5E}
.kpi-v.cb{color:#38BDF8}.kpi-v.cp{color:#A78BFA}.kpi-v.co{color:#FB923C}.kpi-v.ct{color:#2DD4BF}
.kpi-s{font-size:.70rem;color:#1E293B;margin-top:5px}
.kpi-delta{position:absolute;top:13px;right:13px;font-size:.70rem;font-weight:700;
  padding:3px 9px;border-radius:20px}
.delta-up{background:rgba(16,185,129,.1);color:#10B981}
.delta-dn{background:rgba(244,63,94,.1);color:#F43F5E}

.al{border-radius:10px;padding:12px 16px;margin:.4rem 0 .8rem;
  font-size:.82rem;line-height:1.65;border-left:3px solid}
.al-info{background:rgba(56,189,248,.05);border-color:#38BDF8;color:#5A9BB5}
.al-warn{background:rgba(245,158,11,.06);border-color:#F59E0B;color:#A87A22}
.al-bad{background:rgba(244,63,94,.05);border-color:#F43F5E;color:#B03B50}
.al-good{background:rgba(16,185,129,.05);border-color:#10B981;color:#2E8060}
.al strong{font-weight:700;color:inherit}

.rc{background:#0C1122;border:1px solid rgba(255,255,255,.04);border-radius:12px;
  padding:14px 16px;margin-bottom:10px;border-left:3px solid #F59E0B}
.rc.rg{border-left-color:#10B981}.rc.rr{border-left-color:#F43F5E}
.rc.ro{border-left-color:#FB923C}.rc.rb{border-left-color:#38BDF8}
.rh{display:flex;align-items:center;gap:8px;margin-bottom:7px;flex-wrap:wrap}
.rb{font-size:.63rem;font-weight:700;letter-spacing:.08em;padding:2px 8px;
  border-radius:20px;text-transform:uppercase}
.rb-p{background:rgba(245,158,11,.1);color:#F59E0B}
.rb-m{background:rgba(16,185,129,.1);color:#10B981}
.rb-w{background:rgba(56,189,248,.1);color:#38BDF8}
.rb-s{background:rgba(167,139,250,.1);color:#A78BFA}
.rb-o{background:rgba(251,146,60,.1);color:#FB923C}
.rdish{font-size:.71rem;color:#1E293B;font-style:italic}
.rimpact{margin-left:auto;font-size:.75rem;font-weight:700;color:#10B981}
.rt{font-weight:600;font-size:.91rem;color:#E2E8F0;margin-bottom:5px}
.rins{font-size:.80rem;color:#334155;line-height:1.65;margin-bottom:5px}
.ract{font-size:.82rem;color:#F59E0B;font-weight:500}

.order{background:#0C1122;border:1px solid rgba(255,255,255,.04);border-radius:10px;
  padding:10px 14px;margin-bottom:7px;display:flex;justify-content:space-between;align-items:center}
.o-name{color:#E2E8F0;font-weight:500;font-size:.85rem}
.o-cat{font-size:.67rem;color:#1E293B;margin-top:2px}
.o-price{color:#F59E0B;font-family:'DM Mono',monospace;font-weight:600;font-size:.9rem}
.o-time{font-size:.67rem;color:#1E293B;text-align:right;margin-top:2px}

.prog-wrap{background:rgba(255,255,255,.05);border-radius:20px;height:10px;overflow:hidden;margin:.7rem 0}
.prog-fill{height:100%;border-radius:20px;background:linear-gradient(90deg,#92400E,#F59E0B)}

.stButton>button{background:linear-gradient(135deg,#92400E,#F59E0B)!important;color:#040812!important;
  font-weight:700!important;border:none!important;border-radius:10px!important;
  padding:10px 24px!important;font-size:.88rem!important;width:100%!important;letter-spacing:.02em!important}
.stButton>button:hover{opacity:.85!important}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.02);border-radius:12px;
  padding:4px;gap:2px;border:1px solid rgba(255,255,255,.04)}
.stTabs [data-baseweb="tab"]{border-radius:9px!important;color:#334155!important;
  font-size:.82rem!important;padding:9px 18px!important;font-weight:600!important}
.stTabs [aria-selected="true"]{background:rgba(245,158,11,.1)!important;color:#F59E0B!important}
.tip{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.04);border-radius:8px;
  padding:9px 14px;font-size:.76rem;color:#1E293B;margin:.3rem 0 1rem}
[data-testid="stSidebar"] label{color:#475569!important;font-size:.80rem!important}
.stDataFrame{border:1px solid rgba(255,255,255,.05)!important;border-radius:10px!important}
@media(max-width:640px){.b-title{font-size:1.5rem}.kpi-v{font-size:1.35rem}.b-right{display:none}}
</style>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  DATABASE  (SQLite default · Supabase optional)
# ════════════════════════════════════════════════════════════════════════════

SUPABASE_MODE = bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))
_sb = None
if SUPABASE_MODE:
    try:
        from supabase import create_client
        _sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    except ImportError:
        SUPABASE_MODE = False

@st.cache_resource
def _conn():
    c = sqlite3.connect("cafeiq.db", check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript("""
    CREATE TABLE IF NOT EXISTS menu_items(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT, category TEXT, selling_price REAL,
      ingredient_cost REAL, prep_time INTEGER, daily_sold INTEGER,
      waste_pct REAL, rating REAL, is_active INTEGER DEFAULT 1);
    CREATE TABLE IF NOT EXISTS hourly_sales(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      sale_date TEXT, hour INTEGER,
      total_orders INTEGER, total_revenue REAL);
    CREATE TABLE IF NOT EXISTS waste_log(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      log_date TEXT, ingredient TEXT, qty_kg REAL,
      unit_cost REAL, total_loss REAL, category TEXT, shift TEXT);
    CREATE TABLE IF NOT EXISTS inventory(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ingredient TEXT, on_hand REAL, unit TEXT,
      unit_cost REAL, reorder_pt REAL, lead_days INTEGER, weekly_usage REAL);
    CREATE TABLE IF NOT EXISTS staff_plan(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      hour INTEGER, staff_count INTEGER, hourly_rate REAL DEFAULT 26.0);
    """)
    c.commit()
    _seed(c)
    return c

# ── Seed realistic cafe data ──────────────────────────────────────────────────

_MENU = [
    # name, category, price, cost, prep_min, daily_sold, waste%, rating
    ("Espresso",          "Coffee", 4.00,  0.30, 2,  68, 2.0, 4.6),
    ("Long Black",        "Coffee", 4.50,  0.32, 2,  84, 2.0, 4.5),
    ("Cappuccino",        "Coffee", 5.50,  0.78, 3, 125, 3.0, 4.8),
    ("Flat White",        "Coffee", 5.50,  0.72, 3, 118, 3.0, 4.7),
    ("Iced Latte",        "Coffee", 7.00,  1.15, 4, 102, 4.5, 4.5),
    ("Chai Latte",        "Coffee", 6.00,  1.10, 3,  58, 5.0, 4.3),
    ("Hot Chocolate",     "Coffee", 6.00,  0.98, 3,  42, 4.0, 4.2),
    ("Avocado Toast",     "Food",  18.00,  6.20, 8,  48,18.0, 4.6),
    ("Eggs Benedict",     "Food",  21.00,  7.40,12,  36,11.0, 4.5),
    ("Granola Bowl",      "Food",  15.00,  4.10, 5,  32, 9.0, 4.4),
    ("Chicken Wrap",      "Food",  17.00,  7.20, 7,  24,13.0, 4.0),
    ("Caesar Salad",      "Food",  19.00,  5.60, 6,  18,10.0, 4.2),
    ("Banana Bread",      "Snacks", 6.50,  1.50, 1,  62, 7.0, 4.5),
    ("Chocolate Croissant","Snacks",6.00,  1.95, 1,  70, 9.0, 4.7),
    ("Blueberry Muffin",  "Snacks", 5.50,  1.35, 1,  44,10.0, 4.3),
]

_INVENTORY = [
    # ingredient, on_hand(kg), unit, unit_cost, reorder_pt, lead_days, weekly_usage
    ("Coffee Beans",  18.0, "kg", 28.0,  8.0, 2, 12.0),
    ("Full Cream Milk",42.0,"L",  1.80, 15.0, 1, 30.0),
    ("Almond Milk",   12.0, "L",  3.20,  5.0, 2,  8.0),
    ("Avocado",       28.0, "units",1.40, 12.0, 1, 22.0),
    ("Free-Range Eggs",72.0,"units",0.55, 30.0, 1, 48.0),
    ("Sourdough Bread",18.0,"loaves",4.50, 6.0, 1, 14.0),
    ("Chicken Breast", 8.0, "kg",  14.0,  4.0, 2,  6.0),
    ("Greek Yoghurt",  5.0, "kg",   9.0,  2.5, 2,  4.0),
    ("Butter",         3.5, "kg",  12.0,  1.5, 3,  2.5),
    ("Granola",        6.0, "kg",  11.0,  2.0, 3,  3.5),
]

_STAFF = [
    # hour, count, rate
    (6, 1, 26), (7, 3, 26), (8, 5, 26), (9, 5, 26),
    (10, 4, 26),(11, 4, 26),(12, 5, 26),(13, 5, 26),
    (14, 3, 26),(15, 3, 26),(16, 2, 26),(17, 1, 26),
]

# Hourly sales distribution (sum = 1)
_HOUR_W = {6:0.02,7:0.08,8:0.17,9:0.14,10:0.09,11:0.08,
           12:0.11,13:0.13,14:0.07,15:0.06,16:0.04,17:0.01}


def _seed(conn):
    if conn.execute("SELECT COUNT(*) FROM menu_items").fetchone()[0] > 0:
        return
    conn.executemany(
        "INSERT INTO menu_items(name,category,selling_price,ingredient_cost,"
        "prep_time,daily_sold,waste_pct,rating) VALUES(?,?,?,?,?,?,?,?)", _MENU)
    conn.executemany(
        "INSERT INTO inventory(ingredient,on_hand,unit,unit_cost,"
        "reorder_pt,lead_days,weekly_usage) VALUES(?,?,?,?,?,?,?)", _INVENTORY)
    conn.executemany(
        "INSERT INTO staff_plan(hour,staff_count,hourly_rate) VALUES(?,?,?)", _STAFF)

    rng  = np.random.default_rng(7)
    today = datetime.date.today()
    waste_items = [
        ("Coffee Grounds", "Beverages", 2.0),
        ("Milk",           "Beverages", 1.8),
        ("Avocado",        "Food",      1.4),
        ("Sourdough",      "Food",      4.5),
        ("Eggs",           "Food",      0.55),
        ("Pastries",       "Snacks",    3.8),
        ("Salad Leaves",   "Food",      6.0),
    ]
    for d in range(30):
        dt   = str(today - datetime.timedelta(days=d))
        is_we = (today - datetime.timedelta(days=d)).weekday() >= 5
        rev  = rng.normal(4200 if is_we else 3400, 380)
        for h, w in _HOUR_W.items():
            hr = rev * w * rng.uniform(0.88, 1.12)
            n  = max(1, round(hr / 8.5))
            conn.execute(
                "INSERT INTO hourly_sales(sale_date,hour,total_orders,total_revenue)"
                " VALUES(?,?,?,?)", (dt, h, n, round(hr, 2)))
        # Waste entries — 2-4 per day
        for _ in range(rng.integers(2, 5)):
            ing, cat, uc = waste_items[rng.integers(0, len(waste_items))]
            qty  = round(float(rng.uniform(0.2, 2.8)), 2)
            loss = round(qty * uc, 2)
            shift= rng.choice(["Morning","Afternoon","Evening"])
            conn.execute(
                "INSERT INTO waste_log(log_date,ingredient,qty_kg,unit_cost,"
                "total_loss,category,shift) VALUES(?,?,?,?,?,?,?)",
                (dt, ing, qty, uc, loss, cat, shift))
    conn.commit()


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ════════════════════════════════════════════════════════════════════════════

def load_menu(conn) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT * FROM menu_items WHERE is_active=1 ORDER BY category,name", conn)

def load_sales_30d(conn) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT sale_date,hour,SUM(total_orders) AS orders,"
        "SUM(total_revenue) AS revenue"
        " FROM hourly_sales WHERE sale_date>=date('now','-30 days')"
        " GROUP BY sale_date,hour ORDER BY sale_date,hour", conn)

def load_daily(conn) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT sale_date,SUM(total_orders) AS orders,"
        "SUM(total_revenue) AS revenue"
        " FROM hourly_sales WHERE sale_date>=date('now','-30 days')"
        " GROUP BY sale_date ORDER BY sale_date", conn)

def load_waste(conn) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT * FROM waste_log WHERE log_date>=date('now','-30 days')"
        " ORDER BY log_date DESC", conn)

def load_inventory(conn) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM inventory ORDER BY ingredient", conn)

def load_staff(conn) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM staff_plan ORDER BY hour", conn)


# ════════════════════════════════════════════════════════════════════════════
#  BUSINESS ENGINE
# ════════════════════════════════════════════════════════════════════════════

def enrich_menu(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["gross_profit"]   = d["selling_price"] - d["ingredient_cost"]
    d["gross_margin"]   = (d["gross_profit"] / d["selling_price"] * 100).round(1)
    d["food_cost_pct"]  = (d["ingredient_cost"] / d["selling_price"] * 100).round(1)
    d["waste_cost_wk"]  = (d["ingredient_cost"] * d["daily_sold"] * 7
                           * d["waste_pct"] / 100).round(2)
    d["weekly_revenue"] = (d["selling_price"] * d["daily_sold"] * 7).round(2)
    d["weekly_profit"]  = (d["gross_profit"]  * d["daily_sold"] * 7 *
                           (1 - d["waste_pct"]/100)).round(2)
    # Effective prep time: batch + rush model
    d["eff_prep"]       = (d["prep_time"] * np.where(d["daily_sold"]>80, 0.70,
                            np.where(d["daily_sold"]>40, 0.85, 1.0))).round(1)
    d["profit_per_min"] = (d["gross_profit"] / d["eff_prep"].clip(lower=0.5)).round(2)
    avg_p = d["gross_profit"].mean()
    avg_q = d["daily_sold"].mean()
    def _quad(r):
        if r["gross_profit"] >= avg_p and r["daily_sold"] >= avg_q: return "Star"
        if r["gross_profit"] >= avg_p: return "Puzzle"
        if r["daily_sold"] >= avg_q:   return "Plow Horse"
        return "Dog"
    d["segment"] = d.apply(_quad, axis=1)
    return d

def calc_kpis(daily: pd.DataFrame, menu_e: pd.DataFrame, staff: pd.DataFrame,
              fixed_wk: float):
    today = str(datetime.date.today())
    yesterday = str(datetime.date.today() - datetime.timedelta(days=1))
    rev_today = daily[daily.sale_date == today]["revenue"].sum() if today in daily.sale_date.values else 0
    rev_yest  = daily[daily.sale_date == yesterday]["revenue"].sum() if yesterday in daily.sale_date.values else 0
    avg_daily  = daily["revenue"].mean()
    rev_wk     = daily.tail(7)["revenue"].sum()
    rev_prev   = daily.iloc[-14:-7]["revenue"].sum() if len(daily) >= 14 else rev_wk
    gp_wk      = menu_e["weekly_profit"].sum()
    labor_hr   = staff["staff_count"].values * staff["hourly_rate"].values
    labor_wk   = labor_hr.sum() * 12  # 12 operating hours/day × 7 days (rough)
    food_cost  = menu_e["food_cost_pct"].mean()
    labor_pct  = labor_wk / (avg_daily * 7) * 100 if avg_daily else 0
    net_wk     = gp_wk - labor_wk - fixed_wk
    avg_order  = (menu_e["selling_price"] * menu_e["daily_sold"]).sum() / menu_e["daily_sold"].sum()
    waste_wk   = menu_e["waste_cost_wk"].sum()
    return dict(
        rev_today=rev_today, rev_yest=rev_yest,
        avg_daily=avg_daily, rev_wk=rev_wk, rev_prev=rev_prev,
        gp_wk=gp_wk, labor_wk=labor_wk, food_cost=food_cost,
        labor_pct=labor_pct, net_wk=net_wk, avg_order=avg_order, waste_wk=waste_wk,
    )

def effective_prep(base: float, qty_per_day: float,
                   is_rush: bool = False, prepped: bool = False) -> float:
    """T = BasePrep × BatchFactor × RushFactor − PrepBonus."""
    batch = 0.65 if qty_per_day > 80 else (0.80 if qty_per_day > 40 else 1.0)
    rush  = 1.20 if is_rush else 1.0
    prep_bonus = 0.25 if prepped else 0.0
    return max(0.5, base * batch * rush - prep_bonus)

def pricing_suggestions(menu_e: pd.DataFrame):
    """Return per-item optimal price suggestion and gain."""
    rows = []
    for _, r in menu_e.iterrows():
        cost = r["ingredient_cost"]
        cur  = r["selling_price"]
        mg   = r["gross_margin"]
        # Target margin depends on category
        target_mg = 72 if r["category"] == "Coffee" else 58 if r["category"] == "Snacks" else 62
        suggested  = round(cost / (1 - target_mg/100) / 0.5) * 0.5  # round to 50c
        suggested  = max(cur, suggested)  # never suggest lower
        extra_wk   = (suggested - cur) * r["daily_sold"] * 7
        rows.append({
            "name": r["name"], "category": r["category"],
            "current": cur, "suggested": suggested,
            "current_margin": mg, "target_margin": target_mg,
            "extra_wk": round(extra_wk, 0),
        })
    return pd.DataFrame(rows)

def eoq(weekly_usage, order_cost, unit_cost, hold_pct=0.25):
    annual_d = weekly_usage * 52
    H = unit_cost * hold_pct
    if H <= 0: H = 0.01
    q = math.sqrt(2 * annual_d * order_cost / H)
    return round(q, 1)


# ════════════════════════════════════════════════════════════════════════════
#  CHART FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def fig_revenue_trend(daily: pd.DataFrame):
    daily = daily.copy()
    daily["sale_date"] = pd.to_datetime(daily["sale_date"])
    daily["ma7"] = daily["revenue"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["sale_date"], y=daily["revenue"],
        fill="tozeroy", mode="none",
        fillcolor="rgba(245,158,11,0.08)", name="Daily Revenue"))
    fig.add_trace(go.Scatter(
        x=daily["sale_date"], y=daily["revenue"],
        mode="lines", line=dict(color=AMBER, width=2), name="Revenue",
        hovertemplate="$%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=daily["sale_date"], y=daily["ma7"],
        mode="lines", line=dict(color=GREEN, width=1.5, dash="dash"),
        name="7-day avg", hovertemplate="$%{y:,.0f}<extra></extra>"))
    fig.update_layout(title="Daily Revenue — Last 30 Days",
                      xaxis_title=None, yaxis_title="Revenue ($AUD)",
                      legend=dict(orientation="h", y=1.06, x=0,
                                  font=dict(size=10)))
    return _chart(fig, 260)

def fig_peak_heatmap(sales: pd.DataFrame):
    sales = sales.copy()
    sales["dow"] = pd.to_datetime(sales["sale_date"]).dt.dayofweek
    pivot = sales.groupby(["dow", "hour"])["revenue"].mean().reset_index()
    mat   = pivot.pivot(index="dow", columns="hour", values="revenue").fillna(0)
    days  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig   = go.Figure(go.Heatmap(
        z=mat.values,
        x=[f"{h}:00" for h in mat.columns],
        y=[days[i] for i in mat.index],
        colorscale=[[0,"#040812"],[0.3,"#451A03"],[0.6,AMBER],[1,"#FDE68A"]],
        showscale=False,
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Avg Revenue: $%{z:.0f}<extra></extra>"))
    fig.update_layout(title="Peak Revenue Heatmap — Avg by Day & Hour")
    return _chart(fig, 240)

def fig_menu_scatter(menu_e: pd.DataFrame):
    colors = {"Star": AMBER, "Puzzle": BLUE, "Plow Horse": GREEN, "Dog": RED}
    icons  = {"Star": "⭐", "Puzzle": "🧩", "Plow Horse": "🐎", "Dog": "🐕"}
    fig = go.Figure()
    for seg, grp in menu_e.groupby("segment"):
        fig.add_trace(go.Scatter(
            x=grp["daily_sold"], y=grp["gross_margin"],
            mode="markers+text", name=f"{icons[seg]} {seg}",
            text=grp["name"], textposition="top center",
            textfont=dict(size=9, color="#64748B"),
            marker=dict(size=14, color=colors[seg], opacity=0.85,
                        line=dict(width=1.2, color="rgba(255,255,255,0.1)")),
            hovertemplate="<b>%{text}</b><br>Daily: %{x} orders<br>Margin: %{y:.1f}%<extra></extra>"))
    avg_q = menu_e["daily_sold"].mean()
    avg_m = menu_e["gross_margin"].mean()
    fig.add_hline(y=avg_m, line_dash="dot", line_color="rgba(255,255,255,.08)")
    fig.add_vline(x=avg_q, line_dash="dot", line_color="rgba(255,255,255,.08)")
    fig.update_layout(title="Menu Performance Map — Volume vs Gross Margin",
                      xaxis_title="Daily Orders", yaxis_title="Gross Margin (%)")
    return _chart(fig, 320)

def fig_margin_bars(menu_e: pd.DataFrame):
    sdf = menu_e.sort_values("gross_margin", ascending=True)
    colors = [RED if v < 55 else (AMBER if v < 65 else GREEN) for v in sdf["gross_margin"]]
    fig = go.Figure(go.Bar(
        x=sdf["gross_margin"], y=sdf["name"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.0f}%" for v in sdf["gross_margin"]],
        textposition="outside", textfont=dict(size=10, color="#64748B"),
        hovertemplate="<b>%{y}</b><br>Margin: %{x:.1f}%<extra></extra>"))
    fig.add_vline(x=60, line_dash="dash", line_color=AMBER,
                  annotation_text="Target 60%", annotation_font_color=AMBER, annotation_font_size=9)
    fig.update_layout(title="Gross Margin % by Item", xaxis_title="Gross Margin %")
    return _chart(fig, 360)

def fig_weekly_profit(menu_e: pd.DataFrame):
    sdf = menu_e.sort_values("weekly_profit", ascending=False)
    fig = go.Figure(go.Bar(
        x=sdf["name"], y=sdf["weekly_profit"],
        marker=dict(color=sdf["weekly_profit"].tolist(),
                    colorscale=[[0, RED],[0.4, AMBER],[1, GREEN]],
                    showscale=False, line=dict(width=0)),
        text=[f"${v:,.0f}" for v in sdf["weekly_profit"]],
        textposition="outside", textfont=dict(size=10, color="#64748B"),
        hovertemplate="<b>%{x}</b><br>Weekly Profit: $%{y:,.0f}<extra></extra>"))
    fig.update_layout(title="Weekly Profit Contribution by Item", yaxis_title="Weekly Profit ($)")
    return _chart(fig, 260)

def fig_waste_trend(waste: pd.DataFrame):
    if waste.empty:
        return go.Figure()
    daily_waste = waste.groupby("log_date")["total_loss"].sum().reset_index()
    daily_waste["log_date"] = pd.to_datetime(daily_waste["log_date"])
    daily_waste = daily_waste.sort_values("log_date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_waste["log_date"], y=daily_waste["total_loss"],
        fill="tozeroy", fillcolor="rgba(244,63,94,0.07)", mode="none"))
    fig.add_trace(go.Scatter(
        x=daily_waste["log_date"], y=daily_waste["total_loss"],
        mode="lines+markers", line=dict(color=RED, width=2),
        marker=dict(size=5, color=RED),
        hovertemplate="$%{y:,.2f}<extra></extra>", name="Daily Waste"))
    fig.update_layout(title="Daily Waste Cost — Last 30 Days", yaxis_title="Waste Cost ($)")
    return _chart(fig, 230)

def fig_waste_by_ingredient(waste: pd.DataFrame):
    if waste.empty:
        return go.Figure()
    by_ing = waste.groupby("ingredient")["total_loss"].sum().reset_index()
    by_ing = by_ing.sort_values("total_loss", ascending=False).head(8)
    fig = go.Figure(go.Bar(
        x=by_ing["ingredient"], y=by_ing["total_loss"],
        marker=dict(color=by_ing["total_loss"].tolist(),
                    colorscale=[[0, AMBER],[1, RED]], showscale=False, line=dict(width=0)),
        text=[f"${v:.0f}" for v in by_ing["total_loss"]],
        textposition="outside", textfont=dict(size=10, color="#64748B"),
        hovertemplate="<b>%{x}</b><br>Loss: $%{y:,.2f}<extra></extra>"))
    fig.update_layout(title="Top Waste Cost by Ingredient (30 days)", yaxis_title="Total Loss ($)")
    return _chart(fig, 240)

def fig_waste_shift(waste: pd.DataFrame):
    if waste.empty:
        return go.Figure()
    by_shift = waste.groupby("shift")["total_loss"].sum().reset_index()
    fig = go.Figure(go.Bar(
        x=by_shift["shift"], y=by_shift["total_loss"],
        marker=dict(color=[ORANG, BLUE, PURPLE], line=dict(width=0)),
        text=[f"${v:.0f}" for v in by_shift["total_loss"]],
        textposition="outside", textfont=dict(size=11, color="#64748B")))
    fig.update_layout(title="Waste Cost by Shift", yaxis_title="Total Loss ($)")
    return _chart(fig, 220)

def fig_staff_vs_demand(sales: pd.DataFrame, staff: pd.DataFrame):
    hourly = sales.groupby("hour")["revenue"].mean().reset_index()
    staff_c = staff.copy()
    staff_c["hour"] = staff_c["hour"].astype(int)
    hourly["hour"]  = hourly["hour"].astype(int)
    merged = pd.merge(hourly, staff_c[["hour","staff_count"]], on="hour", how="left").fillna(0)
    # Estimate required staff from demand
    merged["req_staff"] = (merged["revenue"] / 180).round(0).clip(lower=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f"{h}:00" for h in merged["hour"]],
        y=merged["staff_count"],
        mode="lines+markers", name="Actual Staff",
        line=dict(color=BLUE, width=2.5),
        marker=dict(size=8, color=BLUE)))
    fig.add_trace(go.Scatter(
        x=[f"{h}:00" for h in merged["hour"]],
        y=merged["req_staff"],
        mode="lines+markers", name="Required (demand-based)",
        line=dict(color=AMBER, width=2, dash="dash"),
        marker=dict(size=7, color=AMBER)))
    fig.update_layout(title="Staff on Floor vs Demand-Based Requirement",
                      xaxis_title="Hour", yaxis_title="Staff Count",
                      legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)))
    return _chart(fig, 250)

def fig_labor_cost_hour(staff: pd.DataFrame):
    staff = staff.copy()
    staff["labor_cost"] = staff["staff_count"] * staff["hourly_rate"]
    fig = go.Figure(go.Bar(
        x=[f"{h}:00" for h in staff["hour"]],
        y=staff["labor_cost"],
        marker=dict(color=staff["labor_cost"].tolist(),
                    colorscale=[[0, GREEN],[0.5, AMBER],[1, RED]], showscale=False, line=dict(width=0)),
        text=[f"${v:.0f}" for v in staff["labor_cost"]],
        textposition="outside", textfont=dict(size=9, color="#64748B")))
    fig.update_layout(title="Labor Cost per Hour ($)", yaxis_title="Cost ($)")
    return _chart(fig, 210)

def fig_waterfall(kpis: dict):
    rev   = kpis["rev_wk"]
    fc    = -(kpis["gp_wk"] - rev) * -1
    labor = -kpis["labor_wk"]
    waste = -kpis["waste_wk"] * 7
    net   = kpis["net_wk"]
    fig = go.Figure(go.Waterfall(
        x=["Weekly Revenue","Gross Profit","−Labor","−Waste","Net Profit"],
        y=[rev, -(rev - kpis["gp_wk"]), labor, waste * 0, net],
        measure=["absolute","relative","relative","relative","total"],
        connector={"line":{"color":"rgba(255,255,255,.04)"}},
        increasing={"marker":{"color": GREEN}},
        decreasing={"marker":{"color": RED}},
        totals={"marker":{"color": AMBER}},
        text=[f"${abs(v):,.0f}" for v in [rev, -(rev-kpis["gp_wk"]), labor, 0, net]],
        textposition="outside", textfont=dict(size=10, color="#64748B")))
    fig.update_layout(title="Weekly Profit Waterfall", yaxis_title="$")
    return _chart(fig, 260)


# ════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

def gen_recommendations(menu_e, kpis, waste, staff_df, sales):
    recs = []

    # ── Pricing ──────────────────────────────────────────────────────────
    for _, r in menu_e.iterrows():
        mg = r["gross_margin"]
        fc = r["food_cost_pct"]
        if r["category"] == "Coffee" and mg < 68:
            target = round(r["ingredient_cost"] / 0.28 / 0.5) * 0.5
            gain   = (target - r["selling_price"]) * r["daily_sold"] * 7
            recs.append(("Pricing", "rr", "rb-p", r["name"],
                f"Coffee margin is only {mg:.0f}% — well below the 72% target",
                f"{r['name']} costs ${r['ingredient_cost']:.2f} to make. "
                f"At ${r['selling_price']:.2f} you're leaving money on the table. "
                f"Industry benchmark for coffee is 72%+ margin.",
                f"Raise price to ${target:.2f}. Expected gain: +${gain:,.0f}/week.",
                gain))
        elif r["category"] == "Food" and mg < 55:
            target = round(r["ingredient_cost"] / 0.40 / 0.5) * 0.5
            gain   = (target - r["selling_price"]) * r["daily_sold"] * 7
            recs.append(("Pricing", "rr", "rb-p", r["name"],
                f"{r['name']} food cost is dangerously high at {fc:.0f}%",
                f"You're spending ${r['ingredient_cost']:.2f} per item while charging "
                f"${r['selling_price']:.2f}. After waste and labour there's almost no profit.",
                f"Raise to ${target:.2f} or reduce portions to cut ingredient cost by "
                f"${r['ingredient_cost'] - r['selling_price']*0.40:.2f}.",
                gain))
        elif mg > 70 and r["rating"] >= 4.5 and r["daily_sold"] < menu_e["daily_sold"].mean():
            gain = 0.50 * r["daily_sold"] * 7
            recs.append(("Pricing", "rc", "rb-p", r["name"],
                f"High-margin + high-rated — small price rise is safe",
                f"{r['name']} has {mg:.0f}% margin and a {r['rating']} rating. "
                f"Customers already love it. A $0.50 increase is unlikely to reduce orders.",
                f"Add $0.50 to the price. Estimated extra: +${gain:,.0f}/week.",
                gain))

    # ── Menu engineering ──────────────────────────────────────────────────
    for _, r in menu_e.iterrows():
        seg = r["segment"]
        if seg == "Dog":
            recs.append(("Menu", "rr", "rb-m", r["name"],
                f"Remove or rework '{r['name']}' — low profit AND low sales",
                f"Only {r['daily_sold']:.0f} orders/day at {r['gross_margin']:.0f}% margin. "
                f"This dish ties up kitchen time worth "
                f"${r['eff_prep'] * r['daily_sold'] * r['profit_per_min']:.0f}/week in better uses.",
                "Remove it from the menu or redesign with 30% lower ingredient cost.",
                r["weekly_profit"] * 0.3))
        elif seg == "Puzzle":
            target = menu_e["daily_sold"].mean() * 0.9
            uplift = (target - r["daily_sold"]) * r["gross_profit"] * 7
            recs.append(("Menu", "rc", "rb-m", r["name"],
                f"'{r['name']}' has great margins but needs more exposure",
                f"{r['gross_margin']:.0f}% margin but only {r['daily_sold']:.0f} orders/day "
                f"vs {menu_e['daily_sold'].mean():.0f} average. "
                f"Promoting it could add ${max(0,uplift):,.0f}/week.",
                "Feature it on the specials board and train staff to recommend it.",
                max(0, uplift)))
        elif seg == "Plow Horse" and r["gross_margin"] < 58:
            recs.append(("Menu", "ro", "rb-m", r["name"],
                f"'{r['name']}' is popular but its margin is too thin",
                f"Sells {r['daily_sold']:.0f}/day (above average) but only {r['gross_margin']:.0f}% margin. "
                f"A $1 price rise would only slightly reduce orders.",
                f"Raise price by $1.00. Estimated weekly gain: +${r['daily_sold']*7:.0f}.",
                r["daily_sold"] * 7))

    # ── Waste ────────────────────────────────────────────────────────────
    if not waste.empty:
        top_waste = waste.groupby("ingredient")["total_loss"].sum().sort_values(ascending=False)
        for ing, loss in top_waste.head(3).items():
            if loss > 80:
                recs.append(("Waste", "ro", "rb-w", ing,
                    f"You lost ${loss:.0f} on {ing} in the last 30 days",
                    f"{ing} is your biggest waste item. This is ${loss/30:.1f}/day going in the bin. "
                    f"Common causes: over-ordering, poor FIFO rotation, or over-prepping.",
                    f"Order 15% less {ing} per week. Implement strict FIFO rotation. "
                    f"Saving target: ${loss*0.4/30:.0f}/day.",
                    loss * 0.4 / 4.33))

        shift_waste = waste.groupby("shift")["total_loss"].sum()
        worst_shift = shift_waste.idxmax() if not shift_waste.empty else None
        if worst_shift:
            recs.append(("Waste", "rc", "rb-w", f"{worst_shift} Shift",
                f"{worst_shift} shift generates the most waste",
                f"${shift_waste[worst_shift]:.0f} in waste over 30 days comes from the "
                f"{worst_shift.lower()} shift. This is often due to over-prepping for "
                f"demand that doesn't materialise.",
                f"Reduce {worst_shift.lower()} shift prep quantities by 10% and track results.",
                shift_waste[worst_shift] * 0.35 / 4.33))

    # ── Staffing ─────────────────────────────────────────────────────────
    if not sales.empty:
        hourly_avg = sales.groupby("hour")["revenue"].mean()
        merged = staff_df.copy()
        merged["hour"] = merged["hour"].astype(int)
        for _, s in merged.iterrows():
            h = int(s["hour"])
            avg_rev = hourly_avg.get(h, 0)
            req = max(1, round(avg_rev / 180))
            actual = s["staff_count"]
            overage = actual - req
            rate = s["hourly_rate"]
            if overage >= 2 and avg_rev > 0:
                saving = overage * rate * 7
                recs.append(("Staffing", "rc", "rb-s", f"{h}:00–{h+1}:00",
                    f"You are overstaffed at {h}:00 — {actual} staff for ~{req} needed",
                    f"Average revenue at {h}:00 is ${avg_rev:.0f}/hr. "
                    f"At $180 revenue per staff member, you only need {req} people. "
                    f"Having {actual} costs an extra ${overage*rate:.0f}/hr.",
                    f"Reduce to {req} staff between {h}:00–{h+1}:00. Weekly saving: ${saving:.0f}.",
                    saving))
            elif overage <= -2:
                recs.append(("Staffing", "rr", "rb-s", f"{h}:00–{h+1}:00",
                    f"Understaffed at {h}:00 — risk of slow service and lost sales",
                    f"Average revenue at {h}:00 is ${avg_rev:.0f}/hr requiring ~{req} staff, "
                    f"but only {actual} scheduled. Slow service during peak hurts repeat visits.",
                    f"Add {-overage} staff at {h}:00. Estimated revenue recovery: "
                    f"+${avg_rev*0.1*7:.0f}/week.",
                    avg_rev * 0.1 * 7))

    # Sort by estimated weekly impact descending
    recs.sort(key=lambda x: -x[7])
    return recs


# ════════════════════════════════════════════════════════════════════════════
#  LIVE SALES SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def simulate_live(menu: pd.DataFrame, daily_target: float):
    """Generate synthetic today's orders up to current minute."""
    rng   = np.random.default_rng(int(datetime.date.today().strftime("%Y%m%d")))
    now   = datetime.datetime.now()
    open_ = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if now < open_:
        return pd.DataFrame(columns=["time","name","category","price"]), 0.0

    elapsed_min = min(int((now - open_).total_seconds() // 60), 660)
    orders = []
    items  = menu.to_dict("records")
    wts    = np.array([r["daily_sold"] for r in items], dtype=float)
    wts   /= wts.sum()

    for m in range(elapsed_min):
        h = (open_ + datetime.timedelta(minutes=m)).hour
        w = _HOUR_W.get(h, 0.005)
        n = rng.poisson(w * daily_target / 8.5 / 60)
        for _ in range(n):
            item = items[rng.choice(len(items), p=wts)]
            orders.append({
                "time":     open_ + datetime.timedelta(minutes=m),
                "name":     item["name"],
                "category": item["category"],
                "price":    item["selling_price"],
            })
    df = pd.DataFrame(orders) if orders else pd.DataFrame(
        columns=["time","name","category","price"])
    revenue = df["price"].sum() if not df.empty else 0.0
    return df, revenue


# ════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATORS
# ════════════════════════════════════════════════════════════════════════════

def build_excel(menu_e, daily, waste, staff_df, kpis, pricing):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        # Summary sheet
        summary = pd.DataFrame({
            "Metric": [
                "Weekly Revenue","Weekly Net Profit","Avg Daily Revenue",
                "Food Cost %","Labor Cost %","Weekly Waste Cost","Avg Order Value"
            ],
            "Value": [
                f"${kpis['rev_wk']:,.0f}", f"${kpis['net_wk']:,.0f}",
                f"${kpis['avg_daily']:,.0f}", f"{kpis['food_cost']:.1f}%",
                f"{kpis['labor_pct']:.1f}%", f"${kpis['waste_wk']*7:.0f}",
                f"${kpis['avg_order']:.2f}"
            ]
        })
        summary.to_excel(xl, sheet_name="Summary", index=False)
        menu_e[[
            "name","category","selling_price","ingredient_cost",
            "gross_profit","gross_margin","weekly_profit","daily_sold","segment"
        ]].rename(columns={
            "name":"Dish","category":"Category","selling_price":"Price ($)",
            "ingredient_cost":"Cost ($)","gross_profit":"Gross Profit ($)",
            "gross_margin":"Margin (%)","weekly_profit":"Weekly Profit ($)",
            "daily_sold":"Daily Sold","segment":"Segment"
        }).to_excel(xl, sheet_name="Menu Analysis", index=False)
        pricing.rename(columns={
            "name":"Dish","current":"Current Price","suggested":"Suggested Price",
            "current_margin":"Current Margin %","target_margin":"Target Margin %",
            "extra_wk":"Extra/Week ($)"
        }).to_excel(xl, sheet_name="Pricing", index=False)
        daily.to_excel(xl, sheet_name="Daily Sales", index=False)
        waste.to_excel(xl, sheet_name="Waste Log", index=False)
    return buf.getvalue()


def build_html_report(menu_e, daily, waste, kpis, recs, period, rdate):
    pf = 4.33 if period == "Monthly" else 1
    pl = "Month" if period == "Monthly" else "Week"
    dish_rows = ""
    for _, r in menu_e.iterrows():
        color = "#F43F5E" if r["gross_margin"] < 55 else ("#F59E0B" if r["gross_margin"] < 65 else "#10B981")
        dish_rows += f"""<tr>
          <td><b>{r['name']}</b></td><td>{r['category']}</td>
          <td>${r['selling_price']:.2f}</td><td>${r['ingredient_cost']:.2f}</td>
          <td style="color:{color}">{r['gross_margin']:.0f}%</td>
          <td>{r['daily_sold']:.0f}</td>
          <td>${r['weekly_profit']:,.0f}</td>
          <td><span style="color:#F59E0B">{r['segment']}</span></td>
        </tr>"""

    rec_html = ""
    for (cat, css, badge_cls, dish, title, insight, action, impact) in recs[:12]:
        col = "#F43F5E" if css == "rr" else ("#10B981" if css == "rg" else
              "#FB923C" if css == "ro" else "#38BDF8")
        rec_html += f"""
        <div style="border-left:3px solid {col};background:#0C1122;border-radius:0 10px 10px 0;
             padding:12px 16px;margin-bottom:10px">
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:5px;flex-wrap:wrap">
            <span style="font-size:11px;font-weight:700;color:{col}">{cat.upper()}</span>
            <span style="font-size:11px;color:#F59E0B">{dish}</span>
            <span style="margin-left:auto;font-size:12px;font-weight:700;color:#10B981">
              +${impact:,.0f}/wk</span>
          </div>
          <div style="font-weight:600;color:#E2E8F0;margin-bottom:4px">{title}</div>
          <div style="color:#334155;line-height:1.6;margin-bottom:5px;font-size:13px">{insight}</div>
          <div style="color:#F59E0B;font-weight:500;font-size:13px">→ {action}</div>
        </div>"""

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>CafeIQ Pro Report — {rdate}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;600&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#040812;color:#E2E8F0;font-family:'DM Sans',sans-serif;padding:0}}
.page{{max-width:1040px;margin:0 auto;padding:28px 20px}}
.hdr{{background:linear-gradient(135deg,#060B1A,#0A1020);border-bottom:1px solid rgba(245,158,11,.15);
  padding:24px 28px;border-radius:14px;margin-bottom:22px}}
.hdr h1{{font-family:'Playfair Display',serif;font-size:1.8rem;color:#fff}}
.hdr h1 span{{color:#F59E0B}}.hdr p{{color:#334155;font-size:.78rem;margin-top:5px}}
.kgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:22px}}
.kbox{{background:linear-gradient(145deg,#0C1122,#070B18);border:1px solid rgba(255,255,255,.05);
  border-radius:12px;padding:14px 16px;text-align:center}}
.kbox .l{{font-size:.64rem;color:#334155;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px}}
.kbox .v{{font-size:1.4rem;font-weight:600;color:#F59E0B}}
.kbox.g .v{{color:#10B981}}.kbox.r .v{{color:#F43F5E}}.kbox.b .v{{color:#38BDF8}}
.sec{{margin-bottom:22px}}
.sec h2{{font-family:'Playfair Display',serif;font-size:1.15rem;color:#F59E0B;
  border-bottom:1px solid rgba(245,158,11,.1);padding-bottom:6px;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
thead tr{{background:#0C1122}}
th{{padding:9px;text-align:left;color:#334155;font-weight:600;font-size:.64rem;
  text-transform:uppercase;letter-spacing:.07em}}
td{{padding:8px 9px;border-bottom:1px solid rgba(255,255,255,.04)}}
tr:hover{{background:rgba(255,255,255,.01)}}
.foot{{text-align:center;color:#1E293B;font-size:.72rem;padding:16px 0;
  border-top:1px solid rgba(255,255,255,.04);margin-top:24px}}
@media print{{body,html{{background:#fff;color:#111}}
  .kbox,.hdr,table,.sec{{background:#f8f8f8;border-color:#ddd;color:#111}}}}
</style></head><body><div class="page">
<div class="hdr"><h1>☕ CafeIQ <span>Pro</span></h1>
<p>{period} Report &nbsp;·&nbsp; {rdate}</p></div>
<div class="kgrid">
  <div class="kbox"><div class="l">Revenue / {pl}</div><div class="v">${kpis['rev_wk']*pf:,.0f}</div></div>
  <div class="kbox {'g' if kpis['net_wk']>0 else 'r'}"><div class="l">Net Profit / {pl}</div><div class="v">${kpis['net_wk']*pf:,.0f}</div></div>
  <div class="kbox {'r' if kpis['food_cost']>35 else 'g'}"><div class="l">Food Cost %</div><div class="v">{kpis['food_cost']:.1f}%</div></div>
  <div class="kbox b"><div class="l">Avg Order Value</div><div class="v">${kpis['avg_order']:.2f}</div></div>
  <div class="kbox"><div class="l">Waste / {pl}</div><div class="v">${kpis['waste_wk']*7*pf:,.0f}</div></div>
  <div class="kbox"><div class="l">Annual Profit Est.</div><div class="v">${kpis['net_wk']*52:,.0f}</div></div>
</div>
<div class="sec"><h2>Menu Performance</h2>
<table><thead><tr><th>Item</th><th>Category</th><th>Price</th><th>Cost</th>
<th>Margin</th><th>Daily Sold</th><th>Wk Profit</th><th>Segment</th></tr></thead>
<tbody>{dish_rows}</tbody></table></div>
<div class="sec"><h2>Priority Actions</h2>{rec_html}</div>
<div class="foot">CafeIQ Pro &nbsp;·&nbsp; {period} Report &nbsp;·&nbsp; {rdate}</div>
</div></body></html>"""


# ════════════════════════════════════════════════════════════════════════════
#  HTML HELPERS
# ════════════════════════════════════════════════════════════════════════════

def kpi(label, value, sub="", cc="ca", delta=None, delta_up=True):
    d_html = ""
    if delta:
        cls  = "delta-up" if delta_up else "delta-dn"
        sign = "▲" if delta_up else "▼"
        d_html = f'<span class="kpi-delta {cls}">{sign} {delta}</span>'
    return (f'<div class="kpi {cc}">'
            f'<div class="kpi-l">{label}</div>'
            f'<div class="kpi-v {cc}">{value}</div>'
            f'<div class="kpi-s">{sub}</div>'
            f'{d_html}</div>')

def alert(text, kind="info"):
    cls = {"info":"al-info","warn":"al-warn","bad":"al-bad","good":"al-good"}
    return f'<div class="al {cls[kind]}">{text}</div>'


# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════════════════

for k, v in [("db_mode", "SQLite"), ("daily_target", 3500.0), ("fixed_wk", 1800.0),
              ("labor_rate", 26.0), ("refresh_cnt", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

conn = _conn()

# ════════════════════════════════════════════════════════════════════════════
#  BANNER
# ════════════════════════════════════════════════════════════════════════════

db_badge = f'<div class="chip"><span class="dot dot-g"></span>{"Supabase" if SUPABASE_MODE else "SQLite"} Live</div>'
st.markdown(f"""
<div class="banner">
  <div class="b-left">
    <div class="b-title">☕ CafeIQ <span>Pro</span></div>
    <div class="b-sub">Cafe & Restaurant Optimization · Profit · Waste · Staffing · Live Sales</div>
  </div>
  <div class="b-right">
    {db_badge}
    <div class="chip">📊 Real-time Analytics</div>
    <div class="chip">🇦🇺 AUD</div>
  </div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Business Settings")
    daily_target = st.number_input("Daily Revenue Target ($)", value=3500, step=100,
                                   key="daily_target")
    fixed_wk = st.number_input("Fixed Costs / Week ($)", value=1800, step=100,
                               help="Rent, utilities, insurance")
    labor_rate = st.number_input("Default Labor Rate ($/hr)", value=26, step=1)
    st.divider()
    st.markdown("### 📋 Quick Stats")
    menu = load_menu(conn)
    menu_e = enrich_menu(menu)
    daily = load_daily(conn)
    rev_today_est = daily.tail(1)["revenue"].values[0] if not daily.empty else 0
    st.metric("Items on Menu",    len(menu))
    st.metric("Today's Est. Rev", f"${rev_today_est:,.0f}")
    st.metric("Weekly Profit Est.",f"${menu_e['weekly_profit'].sum() - fixed_wk:,.0f}")
    st.divider()
    if SUPABASE_MODE:
        st.success("✓ Connected to Supabase")
    else:
        st.markdown("""
        <div style="font-size:.75rem;color:#334155;line-height:1.7">
        <strong style="color:#F59E0B">Supabase Mode</strong><br>
        Set env vars:<br>
        <code>SUPABASE_URL</code><br>
        <code>SUPABASE_KEY</code><br>
        Then: <code>pip install supabase</code>
        </div>""", unsafe_allow_html=True)
    st.divider()
    st.caption("CafeIQ Pro · Cafe Optimization Suite")


# ════════════════════════════════════════════════════════════════════════════
#  LOAD ALL DATA
# ════════════════════════════════════════════════════════════════════════════

sales   = load_sales_30d(conn)
waste   = load_waste(conn)
inv_df  = load_inventory(conn)
staff_df= load_staff(conn)
kpis    = calc_fin = calc_kpis(daily, menu_e, staff_df, fixed_wk)
pricing = pricing_suggestions(menu_e)
recs    = gen_recommendations(menu_e, kpis, waste, staff_df, sales)

# ════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════

T1, T2, T3, T4, T5, T6 = st.tabs([
    "📊 Dashboard", "🍽️ Menu", "♻️ Waste", "👥 Staffing", "⚡ Live Sales", "📄 Reports"
])


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 1: DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
with T1:
    st.markdown('<div class="sh">Business Health at a Glance</div>', unsafe_allow_html=True)

    # Alerts
    if kpis["food_cost"] > 38:
        st.markdown(alert(f'<strong>Food cost alert:</strong> Average food cost is '
                          f'{kpis["food_cost"]:.1f}% — target is under 32%. '
                          f'Check your top 3 highest-cost items.', "bad"), unsafe_allow_html=True)
    if kpis["net_wk"] < 0:
        st.markdown(alert(f'<strong>Operating at a loss:</strong> Estimated weekly net is '
                          f'${kpis["net_wk"]:,.0f}. Review pricing and fixed costs immediately.', "bad"),
                    unsafe_allow_html=True)

    # KPI row 1
    c = st.columns(4)
    rev_delta = (kpis["rev_wk"] - kpis["rev_prev"]) / kpis["rev_prev"] * 100 if kpis["rev_prev"] else 0
    for col, html in zip(c, [
        kpi("Weekly Revenue",    f"${kpis['rev_wk']:,.0f}",   "last 7 days", "ca",
            delta=f"{abs(rev_delta):.1f}%", delta_up=rev_delta>=0),
        kpi("Net Profit / Wk",  f"${kpis['net_wk']:,.0f}",   "after costs",
            "cg" if kpis["net_wk"]>0 else "cr"),
        kpi("Annual Profit Est.",f"${kpis['net_wk']*52:,.0f}","projected", "cp"),
        kpi("Avg Order Value",  f"${kpis['avg_order']:.2f}",  "per customer", "cb"),
    ]):
        col.markdown(html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row 2
    c2 = st.columns(4)
    fc_ok = kpis["food_cost"] <= 32
    for col, html in zip(c2, [
        kpi("Food Cost %",    f"{kpis['food_cost']:.1f}%",  "target <32%",  "cg" if fc_ok else "cr"),
        kpi("Labor Cost %",   f"{kpis['labor_pct']:.1f}%",  "target <30%",  "cg" if kpis["labor_pct"]<=30 else "cr"),
        kpi("Weekly Waste",   f"${kpis['waste_wk']*7:.0f}", "avoidable loss","co"),
        kpi("Menu Items",     str(len(menu_e)),              "active", "ct"),
    ]):
        col.markdown(html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.plotly_chart(fig_revenue_trend(daily), use_container_width=True, **_CFG)
    with col_r:
        st.plotly_chart(fig_waterfall(kpis), use_container_width=True, **_CFG)

    st.plotly_chart(fig_peak_heatmap(sales), use_container_width=True, **_CFG)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(fig_weekly_profit(menu_e), use_container_width=True, **_CFG)
    with c4:
        st.plotly_chart(fig_margin_bars(menu_e), use_container_width=True, **_CFG)

    # Full table
    st.markdown('<div class="sh">Full Menu Table</div>', unsafe_allow_html=True)
    disp = menu_e[[
        "name","category","selling_price","ingredient_cost","gross_profit",
        "gross_margin","food_cost_pct","daily_sold","weekly_profit","waste_pct","rating","segment"
    ]].copy()
    for c_ in ["selling_price","ingredient_cost","gross_profit","weekly_profit"]:
        disp[c_] = disp[c_].map("${:,.2f}".format)
    for c_ in ["gross_margin","food_cost_pct"]:
        disp[c_] = disp[c_].map("{:.1f}%".format)
    disp.columns = ["Item","Category","Price","Cost","Gross Profit","Margin%",
                    "Food Cost%","Daily Sold","Wk Profit","Waste%","Rating","Segment"]
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 2: MENU OPTIMIZATION
# ──────────────────────────────────────────────────────────────────────────────
with T2:
    st.markdown('<div class="sh">Menu Performance & Pricing</div>', unsafe_allow_html=True)

    seg_counts = menu_e["segment"].value_counts()
    dogs = seg_counts.get("Dog", 0)
    puzzles = seg_counts.get("Puzzle", 0)
    stars   = seg_counts.get("Star", 0)
    plow    = seg_counts.get("Plow Horse", 0)
    total_gain = pricing[pricing["extra_wk"]>0]["extra_wk"].sum()

    c = st.columns(5)
    for col, (lbl, val, cc) in zip(c, [
        (f"⭐ Stars",      stars,  "ca"),
        (f"🧩 Puzzles",   puzzles,"cb"),
        (f"🐎 Plow Horses",plow,  "cg"),
        (f"🐕 Dogs",       dogs,  "cr"),
        (f"Repricing Gain/Wk",f"${total_gain:,.0f}","ct"),
    ]):
        col.markdown(kpi(lbl, val if isinstance(val, str) else str(val), cc=cc), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    info_text = (
        "Stars ⭐ = high margin + high volume. "
        "Puzzles 🧩 = high margin but needs promotion. "
        "Plow Horses 🐎 = popular but low margin — consider a price rise. "
        "Dogs 🐕 = low margin + low volume — review or remove."
    )
    st.markdown(alert(info_text, "info"), unsafe_allow_html=True)

    st.plotly_chart(fig_menu_scatter(menu_e), use_container_width=True, **_CFG)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(fig_margin_bars(menu_e), use_container_width=True, **_CFG)
    with c6:
        # Profit per minute
        sdf = menu_e.sort_values("profit_per_min", ascending=False)
        fig = go.Figure(go.Bar(
            x=sdf["name"], y=sdf["profit_per_min"],
            marker=dict(color=sdf["profit_per_min"].tolist(),
                        colorscale=[[0, RED],[0.4, AMBER],[1, GREEN]],
                        showscale=False, line=dict(width=0)),
            text=[f"${v:.2f}" for v in sdf["profit_per_min"]],
            textposition="outside", textfont=dict(size=9, color="#64748B")))
        fig.update_layout(title="Profit per Kitchen Minute (batch-adjusted)", yaxis_title="$ / min")
        st.plotly_chart(_chart(fig, 270), use_container_width=True, **_CFG)

    # Pricing table
    st.markdown('<div class="sh">Pricing Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tip">Prices are suggested to hit category-specific margin targets: '
        'Coffee → 72% · Food → 62% · Snacks → 58%. Rounded to nearest $0.50.</div>',
        unsafe_allow_html=True)

    def _color_row(row):
        return (["color: #F43F5E"] * len(row) if row["extra_wk"] > 50
                else [""] * len(row))

    price_disp = pricing.copy()
    price_disp["change"] = price_disp.apply(
        lambda r: f"+${r['suggested']-r['current']:.2f}" if r["suggested"] > r["current"] else "—", axis=1)
    price_disp["current"] = price_disp["current"].map("${:.2f}".format)
    price_disp["suggested"] = price_disp["suggested"].map("${:.2f}".format)
    price_disp["extra_wk"] = price_disp["extra_wk"].map("${:,.0f}".format)
    price_disp = price_disp.rename(columns={
        "name":"Item","category":"Category","current":"Current Price",
        "suggested":"Suggested Price","change":"Change",
        "current_margin":"Current Margin%","target_margin":"Target Margin%",
        "extra_wk":"Extra/Week"
    })
    st.dataframe(price_disp, use_container_width=True, hide_index=True)

    # Price test widget
    st.markdown('<div class="sh">🎚️ What-If Price Tester</div>', unsafe_allow_html=True)
    sel = st.selectbox("Select item to test", menu_e["name"].tolist())
    sel_row = menu_e[menu_e["name"] == sel].iloc[0]
    new_price = st.slider(
        "Test price ($)", float(sel_row["ingredient_cost"] * 1.25),
        float(sel_row["selling_price"] * 2.0),
        float(sel_row["selling_price"]), step=0.25)
    new_mg   = (new_price - sel_row["ingredient_cost"]) / new_price * 100
    new_wk_p = (new_price - sel_row["ingredient_cost"]) * sel_row["daily_sold"] * 7
    cur_wk_p = sel_row["weekly_profit"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("New Price",       f"${new_price:.2f}")
    mc2.metric("New Margin",      f"{new_mg:.1f}%",    f"{new_mg-sel_row['gross_margin']:+.1f}%")
    mc3.metric("New Weekly Profit",f"${new_wk_p:,.0f}", f"${new_wk_p-cur_wk_p:+,.0f}")
    mc4.metric("Annual Impact",   f"${(new_wk_p-cur_wk_p)*52:+,.0f}")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 3: WASTE & INVENTORY
# ──────────────────────────────────────────────────────────────────────────────
with T3:
    st.markdown('<div class="sh">Waste Tracking & Inventory</div>', unsafe_allow_html=True)

    total_waste_30 = waste["total_loss"].sum() if not waste.empty else 0
    avg_daily_waste = total_waste_30 / 30 if total_waste_30 else 0
    top_waster = (waste.groupby("ingredient")["total_loss"].sum().idxmax()
                  if not waste.empty else "N/A")
    waste_pct_rev = total_waste_30 / (kpis["avg_daily"] * 30) * 100 if kpis["avg_daily"] else 0

    c = st.columns(4)
    for col, html in zip(c, [
        kpi("Waste Cost (30d)", f"${total_waste_30:,.0f}", "total avoidable", "cr"),
        kpi("Daily Waste Avg",  f"${avg_daily_waste:.0f}", "per day", "co"),
        kpi("Waste % of Rev",   f"{waste_pct_rev:.1f}%",  "target <3%",
            "cg" if waste_pct_rev<=3 else "cr"),
        kpi("Top Waster",       top_waster,                "by cost", "cp"),
    ]):
        col.markdown(html, unsafe_allow_html=True)

    if total_waste_30 > 300:
        st.markdown(alert(
            f'<strong>You lost ${total_waste_30:,.0f}</strong> to waste over the last 30 days '
            f'(${avg_daily_waste:.0f}/day). Reducing waste by 30% would save '
            f'${total_waste_30*0.3/4.33:.0f}/week.', "bad"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(fig_waste_trend(waste), use_container_width=True, **_CFG)
    with c8:
        st.plotly_chart(fig_waste_by_ingredient(waste), use_container_width=True, **_CFG)

    c9, c10 = st.columns([2,1])
    with c9:
        # Waste by category bar
        if not waste.empty:
            by_cat = waste.groupby("category")["total_loss"].sum().reset_index()
            fig = go.Figure(go.Bar(
                x=by_cat["category"], y=by_cat["total_loss"],
                marker=dict(color=[ORANG, RED, AMBER, PURPLE, BLUE][:len(by_cat)], line=dict(width=0)),
                text=[f"${v:.0f}" for v in by_cat["total_loss"]],
                textposition="outside", textfont=dict(size=11, color="#64748B")))
            fig.update_layout(title="Waste by Category (30 days)", yaxis_title="Loss ($)")
            st.plotly_chart(_chart(fig, 230), use_container_width=True, **_CFG)
    with c10:
        st.plotly_chart(fig_waste_shift(waste), use_container_width=True, **_CFG)

    # Inventory & EOQ
    st.markdown('<div class="sh">Inventory & Optimal Order Quantities</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tip">EOQ = Economic Order Quantity. Order this amount each time to '
        'minimise total holding + ordering costs. Red = reorder needed now.</div>',
        unsafe_allow_html=True)

    eoq_rows = []
    for _, r in inv_df.iterrows():
        q  = eoq(r["weekly_usage"], 15, r["unit_cost"])  # $15 order cost assumption
        ss = round(r["weekly_usage"] * 0.2 * r["lead_days"] / 7, 1)
        rop = round(r["weekly_usage"] / 7 * r["lead_days"] + ss, 1)
        status = "🔴 Order Now" if r["on_hand"] < rop else (
                 "🟡 Low Stock" if r["on_hand"] < rop * 1.5 else "✅ OK")
        eoq_rows.append({
            "Ingredient":    r["ingredient"],
            "On Hand":       f"{r['on_hand']:.1f} {r['unit']}",
            "EOQ":           f"{q:.0f} {r['unit']}",
            "Safety Stock":  f"{ss:.1f} {r['unit']}",
            "Reorder Point": f"{rop:.1f} {r['unit']}",
            "Lead Days":     r["lead_days"],
            "Unit Cost":     f"${r['unit_cost']:.2f}",
            "Weekly Usage":  f"{r['weekly_usage']:.1f} {r['unit']}",
            "Status":        status,
        })
    st.dataframe(pd.DataFrame(eoq_rows), use_container_width=True, hide_index=True)

    # Log new waste
    st.markdown('<div class="sh">➕ Log Waste Entry</div>', unsafe_allow_html=True)
    wc1, wc2, wc3, wc4 = st.columns(4)
    w_ing   = wc1.text_input("Ingredient")
    w_qty   = wc2.number_input("Quantity (kg/L/units)", min_value=0.0, step=0.1)
    w_uc    = wc3.number_input("Unit Cost ($)", min_value=0.0, step=0.5)
    w_shift = wc4.selectbox("Shift", ["Morning","Afternoon","Evening"])
    w_cat   = st.selectbox("Category", ["Beverages","Food","Snacks","Dairy","Produce"])
    if st.button("💾 Save Waste Entry"):
        if w_ing and w_qty > 0 and w_uc > 0:
            conn.execute(
                "INSERT INTO waste_log(log_date,ingredient,qty_kg,unit_cost,total_loss,category,shift)"
                " VALUES(?,?,?,?,?,?,?)",
                (str(datetime.date.today()), w_ing, w_qty, w_uc,
                 round(w_qty * w_uc, 2), w_cat, w_shift))
            conn.commit()
            st.success(f"✓ Logged ${w_qty * w_uc:.2f} waste for {w_ing}")
            st.rerun()
        else:
            st.warning("Fill in all fields.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 4: STAFFING
# ──────────────────────────────────────────────────────────────────────────────
with T4:
    st.markdown('<div class="sh">Staffing & Peak-Hour Optimization</div>', unsafe_allow_html=True)

    labor_cost_wk = (staff_df["staff_count"] * staff_df["hourly_rate"]).sum() * 7
    labor_pct_rev = labor_cost_wk / (kpis["avg_daily"] * 7) * 100 if kpis["avg_daily"] else 0
    # Overstaffed hours
    hourly_avg = sales.groupby("hour")["revenue"].mean()
    over_hrs = 0; under_hrs = 0
    for _, s in staff_df.iterrows():
        h = int(s["hour"])
        avg_r = hourly_avg.get(h, 0)
        req   = max(1, round(avg_r / 180))
        if s["staff_count"] - req >= 2: over_hrs += 1
        elif s["staff_count"] - req <= -2: under_hrs += 1
    saving_potential = over_hrs * labor_rate * 7

    c = st.columns(4)
    for col, html in zip(c, [
        kpi("Labor Cost / Wk",  f"${labor_cost_wk:,.0f}",     "all staff", "co"),
        kpi("Labor %",          f"{labor_pct_rev:.1f}%",        "of revenue",
            "cg" if labor_pct_rev<=30 else "cr"),
        kpi("Overstaffed Hours",str(over_hrs),                  "hours/day", "cr"),
        kpi("Saving Potential", f"${saving_potential:,.0f}/wk", "from overstaffing", "cg"),
    ]):
        col.markdown(html, unsafe_allow_html=True)

    if over_hrs > 2:
        st.markdown(alert(
            f'<strong>Overstaffing detected in {over_hrs} hour-slots per day.</strong> '
            f'Adjusting rosters could save up to ${saving_potential:,.0f}/week.', "warn"),
            unsafe_allow_html=True)
    if under_hrs > 0:
        st.markdown(alert(
            f'<strong>Understaffed during {under_hrs} peak hour-slots.</strong> '
            f'This risks slow service and lost revenue.', "bad"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(fig_staff_vs_demand(sales, staff_df), use_container_width=True, **_CFG)
    st.plotly_chart(fig_labor_cost_hour(staff_df), use_container_width=True, **_CFG)

    # Staffing breakdown table
    st.markdown('<div class="sh">Hour-by-Hour Staffing Analysis</div>', unsafe_allow_html=True)
    sf_table = []
    for _, s in staff_df.iterrows():
        h     = int(s["hour"])
        avg_r = hourly_avg.get(h, 0)
        req   = max(1, round(avg_r / 180))
        diff  = int(s["staff_count"]) - req
        cost  = s["staff_count"] * s["hourly_rate"]
        status = (f"🔴 Over by {diff}" if diff >= 2 else
                  f"🔴 Under by {abs(diff)}" if diff <= -2 else "✅ Optimal")
        sf_table.append({
            "Hour":          f"{h}:00–{h+1}:00",
            "Avg Revenue":   f"${avg_r:.0f}",
            "Actual Staff":  int(s["staff_count"]),
            "Required":      req,
            "Difference":    f"{diff:+d}",
            "Labor Cost/Hr": f"${cost:.0f}",
            "Status":        status,
        })
    st.dataframe(pd.DataFrame(sf_table), use_container_width=True, hide_index=True)

    # Edit staff plan
    st.markdown('<div class="sh">✏️ Update Staffing Plan</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    edit_h    = sc1.selectbox("Hour", list(range(6, 19)), key="sh")
    edit_cnt  = sc2.number_input("Staff Count", min_value=1, max_value=15, value=3, key="sn")
    edit_rate = sc3.number_input("Hourly Rate ($)", min_value=20, max_value=60, value=26, key="sr")
    if st.button("💾 Update Hour"):
        conn.execute(
            "UPDATE staff_plan SET staff_count=?, hourly_rate=? WHERE hour=?",
            (edit_cnt, edit_rate, edit_h))
        conn.commit()
        st.success(f"✓ Updated {edit_h}:00 → {edit_cnt} staff @ ${edit_rate}/hr")
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 5: LIVE SALES
# ──────────────────────────────────────────────────────────────────────────────
with T5:
    st.markdown('<div class="sh">⚡ Live Sales Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tip">Simulates today\'s order stream based on historical demand patterns. '
        'Click Refresh to update.</div>', unsafe_allow_html=True)

    col_ref, col_tog = st.columns([3,1])
    with col_ref:
        if st.button("🔄 Refresh Now"):
            st.session_state.refresh_cnt += 1
    with col_tog:
        auto_r = st.checkbox("Auto-refresh (30s)", value=False)

    orders_df, rev_today = simulate_live(menu, daily_target)

    pct = min(rev_today / daily_target * 100, 100)
    gap = daily_target - rev_today
    now_str = datetime.datetime.now().strftime("%H:%M:%S")

    # KPIs
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.markdown(kpi("Revenue Today",   f"${rev_today:,.0f}", f"as of {now_str}", "ca"), unsafe_allow_html=True)
    lc2.markdown(kpi("Daily Target",    f"${daily_target:,.0f}", "set in sidebar",  "cb"), unsafe_allow_html=True)
    lc3.markdown(kpi("Gap to Target",   f"${max(0,gap):,.0f}",   "remaining",
                     "cg" if gap<=0 else "co"), unsafe_allow_html=True)
    lc4.markdown(kpi("Orders Today",    str(len(orders_df)),      "simulated",       "ct"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Progress bar
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;font-size:.78rem;color:#334155">'
        f'<span>Daily progress</span><span style="color:#F59E0B">{pct:.0f}%</span></div>'
        f'<div class="prog-wrap"><div class="prog-fill" style="width:{pct:.0f}%"></div></div>',
        unsafe_allow_html=True)

    # Revenue by hour chart (today)
    if not orders_df.empty:
        orders_df["hour"] = orders_df["time"].dt.hour
        hourly_today = orders_df.groupby("hour")["price"].sum().reset_index()
        fig = go.Figure(go.Bar(
            x=[f"{h}:00" for h in hourly_today["hour"]],
            y=hourly_today["price"],
            marker=dict(color=hourly_today["price"].tolist(),
                        colorscale=[[0, "#0C1122"],[0.5, AMBER],[1, GREEN]],
                        showscale=False, line=dict(width=0)),
            text=[f"${v:.0f}" for v in hourly_today["price"]],
            textposition="outside", textfont=dict(size=10, color="#64748B")))
        fig.update_layout(title=f"Today's Revenue by Hour — {datetime.date.today().strftime('%A %d %b')}",
                          yaxis_title="Revenue ($)")
        st.plotly_chart(_chart(fig, 230), use_container_width=True, **_CFG)

    # Live order feed + top items side by side
    fc1, fc2 = st.columns([3, 2])
    with fc1:
        st.markdown('<div class="sh">Recent Orders</div>', unsafe_allow_html=True)
        if not orders_df.empty:
            last20 = orders_df.tail(20).iloc[::-1]
            for _, o in last20.iterrows():
                t_str = o["time"].strftime("%H:%M")
                st.markdown(
                    f'<div class="order">'
                    f'<div><div class="o-name">{o["name"]}</div>'
                    f'<div class="o-cat">{o["category"]}</div></div>'
                    f'<div><div class="o-price">${o["price"]:.2f}</div>'
                    f'<div class="o-time">{t_str}</div></div>'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(alert("No orders yet today — cafe opens at 7:00 AM.", "info"),
                        unsafe_allow_html=True)

    with fc2:
        st.markdown('<div class="sh">Today\'s Top Sellers</div>', unsafe_allow_html=True)
        if not orders_df.empty:
            top = orders_df.groupby("name").agg(
                orders=("price","count"), revenue=("price","sum")).reset_index()
            top = top.sort_values("revenue", ascending=False).head(8)
            for _, t in top.iterrows():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:8px 0;border-bottom:1px solid rgba(255,255,255,.04);'
                    f'font-size:.83rem">'
                    f'<span style="color:#E2E8F0">{t["name"]}</span>'
                    f'<span><span style="color:#64748B;font-size:.75rem">{t["orders"]} orders&nbsp;&nbsp;</span>'
                    f'<span style="color:#F59E0B;font-family:DM Mono,monospace">${t["revenue"]:.0f}</span></span>'
                    f'</div>', unsafe_allow_html=True)

    # Auto-refresh logic
    if auto_r:
        time.sleep(30)
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 6: REPORTS
# ──────────────────────────────────────────────────────────────────────────────
with T6:
    st.markdown('<div class="sh">Download Reports</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tip">Download detailed reports for your accountant or management team. '
        'HTML reports can be printed as PDF from your browser (Ctrl+P).</div>',
        unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        period = st.radio("Report Period", ["Weekly","Monthly"], horizontal=True)
    with rc2:
        rdate = str(st.date_input("Report Date", datetime.date.today()))

    pf = 4.33 if period == "Monthly" else 1
    pl = "month" if period == "Monthly" else "week"
    factor = pf

    st.markdown('<div class="sh">Period Summary</div>', unsafe_allow_html=True)
    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric("Revenue",        f"${kpis['rev_wk']*factor:,.0f}")
    sm2.metric("Gross Profit",   f"${kpis['gp_wk']*factor:,.0f}")
    sm3.metric("Net Profit",     f"${kpis['net_wk']*factor:,.0f}")
    sm4.metric("Total Waste",    f"${kpis['waste_wk']*7*factor:,.0f}")
    sm5.metric("Annual Proj.",   f"${kpis['net_wk']*52:,.0f}")

    # Scenario table
    st.markdown('<div class="sh">Fixed Cost Sensitivity</div>', unsafe_allow_html=True)
    scenarios = []
    for mult in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
        fc   = fixed_wk * mult
        net  = (kpis["gp_wk"] - kpis["labor_wk"] - fc) * factor
        scenarios.append({
            f"Fixed Costs/{pl.capitalize()}": f"${fc*factor:,.0f}",
            f"Net Profit/{pl.capitalize()}":  f"${net:,.0f}",
            "Annual Projection":             f"${(kpis['gp_wk']-kpis['labor_wk']-fc)*52:,.0f}",
            "vs Baseline":                   f"{(mult-1)*100:+.0f}%",
        })
    st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Download buttons
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.markdown("#### 📊 Excel Report")
        st.markdown("Full analysis across 5 sheets: Summary · Menu · Pricing · Sales · Waste")
        if st.button("📥 Build Excel"):
            xl_bytes = build_excel(menu_e, daily, waste, staff_df, kpis, pricing)
            st.download_button(
                "⬇️ Download Excel",
                data=xl_bytes,
                file_name=f"CafeIQ_{period}_{rdate}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with dl2:
        st.markdown("#### 📄 HTML / PDF Report")
        st.markdown("Styled management report with KPIs, menu table, and top actions.")
        if st.button("📥 Build HTML Report"):
            html = build_html_report(menu_e, daily, waste, kpis, recs, period, rdate)
            st.download_button(
                "⬇️ Download HTML",
                data=html,
                file_name=f"CafeIQ_{period}_{rdate}.html",
                mime="text/html")
            st.success("Open in browser and press Ctrl+P to save as PDF.")

    with dl3:
        st.markdown("#### 📋 CSV Export")
        st.markdown("Raw data: menu analysis, daily sales, and waste log.")
        if st.button("📥 Build CSV Pack"):
            # Zip-equivalent: offer three separate downloads
            csv_menu  = menu_e.to_csv(index=False)
            csv_sales = daily.to_csv(index=False)
            csv_waste = waste.to_csv(index=False)
            st.download_button("⬇️ Menu Analysis CSV",
                               csv_menu, f"cafeiq_menu_{rdate}.csv", "text/csv")
            st.download_button("⬇️ Daily Sales CSV",
                               csv_sales, f"cafeiq_sales_{rdate}.csv", "text/csv")
            st.download_button("⬇️ Waste Log CSV",
                               csv_waste, f"cafeiq_waste_{rdate}.csv", "text/csv")

    # Priority actions in report view
    st.markdown('<div class="sh">Priority Actions for This Report</div>', unsafe_allow_html=True)
    cat_filter = st.selectbox(
        "Filter by type",
        ["All","Pricing","Menu","Waste","Staffing"], key="rep_filt")
    cat_map = {"Pricing":"rb-p","Menu":"rb-m","Waste":"rb-w","Staffing":"rb-s"}
    border_map = {"rr":"rr","ro":"ro","rc":"rg","rb":"rb"}
    for (cat, css, badge_cls, dish, title, insight, action, impact) in recs:
        if cat_filter != "All" and cat != cat_filter:
            continue
        st.markdown(f"""
        <div class="rc {css}">
          <div class="rh">
            <span class="rb {badge_cls}">{cat}</span>
            <span class="rdish">{dish}</span>
            <span class="rimpact">+${impact:,.0f}/wk</span>
          </div>
          <div class="rt">{title}</div>
          <div class="rins">{insight}</div>
          <div class="ract">→ {action}</div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#1E293B;font-size:.72rem;padding:20px 0 6px;
  border-top:1px solid rgba(255,255,255,.04);margin-top:24px">
  CafeIQ Pro &nbsp;·&nbsp; Profit · Waste · Staffing · Live Sales
  &nbsp;·&nbsp; SQLite / Supabase Ready
</div>""", unsafe_allow_html=True)
