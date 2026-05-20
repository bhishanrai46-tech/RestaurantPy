"""
Seralung Opti — Cafe Business Optimization Software
====================================================
Single-file Streamlit application. No local module imports.
External dependencies: streamlit, pandas, numpy, plotly, supabase, python-dotenv

Run locally:    streamlit run app.py
Streamlit Cloud: push this file + requirements.txt to GitHub
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os
import io
import re
import smtplib
import warnings
from datetime import datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Seralung Opti",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 ── DESIGN TOKENS & GLOBAL CSS
# ═══════════════════════════════════════════════════════════════

# Color palette — warm neutrals, single forest-green accent
C = {
    "bg":         "#F7F6F3",
    "card":       "#FFFFFF",
    "border":     "#E5E3DE",
    "text":       "#1C1C1C",
    "muted":      "#6A6864",
    "accent":     "#2C5F2E",
    "accent_soft":"#EAF0E8",
    "success":    "#3D6B3F",
    "warning":    "#8B6914",
    "danger":     "#8B2E2E",
    "sidebar":    "#EFEDE8",
    "divider":    "#DEDAD4",
}

GLOBAL_CSS = f"""
<style>
/* ── Page ── */
.stApp {{
    background-color:{C['bg']};
    font-family:'Helvetica Neue',sans-serif;
    color:{C['text']};
}}
.block-container {{
    padding-top:2rem;
    padding-bottom:3rem;
    max-width:1120px;
}}
/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color:{C['sidebar']};
    border-right:1px solid {C['border']};
}}
/* ── Headings ── */
h1,h2,h3 {{
    font-family:'Georgia',serif;
    font-weight:400;
    color:{C['text']};
    letter-spacing:-0.02em;
}}
h2{{font-size:1.5rem;}}
h3{{font-size:1.1rem;}}
/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background:{C['card']};
    border:1px solid {C['border']};
    border-radius:6px;
    padding:1.25rem 1.5rem;
}}
[data-testid="metric-container"] label {{
    font-size:0.75rem;
    text-transform:uppercase;
    letter-spacing:0.06em;
    color:{C['muted']};
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family:'Georgia',serif;
    font-size:1.85rem;
    color:{C['text']};
}}
/* ── Buttons ── */
.stButton>button {{
    background-color:{C['accent']};
    color:#fff;
    border:none;
    border-radius:4px;
    padding:0.55rem 1.4rem;
    font-size:0.875rem;
    letter-spacing:0.02em;
    transition:opacity .15s;
}}
.stButton>button:hover{{opacity:.82;}}
/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap:0;
    border-bottom:1px solid {C['border']};
    background:transparent;
}}
.stTabs [data-baseweb="tab"] {{
    font-size:0.875rem;
    color:{C['muted']};
    padding:0.65rem 1.25rem;
    border-bottom:2px solid transparent;
    border-radius:0;
    background:transparent;
}}
.stTabs [aria-selected="true"] {{
    color:{C['text']};
    border-bottom:2px solid {C['accent']};
    font-weight:600;
    background:transparent;
}}
/* ── Dataframe ── */
.stDataFrame{{border:1px solid {C['border']};border-radius:6px;}}
/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {{
    background:{C['card']};
    border:1px dashed {C['border']};
    border-radius:6px;
}}
/* ── Alerts ── */
.stAlert{{border-radius:6px;font-size:0.875rem;}}
/* ── Hide Streamlit chrome ── */
#MainMenu,footer{{visibility:hidden;}}
hr{{border:none;border-top:1px solid {C['divider']};margin:1.25rem 0;}}
</style>
"""


def inject_css():
    """Inject global CSS into the page. Call once at app start."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_title(title: str, subtitle: str = ""):
    """Render a consistent page heading + optional subtitle."""
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(
            f"<p style='color:{C['muted']};font-size:0.875rem;"
            f"margin-top:-0.5rem;'>{subtitle}</p>",
            unsafe_allow_html=True,
        )
    st.markdown("---")


def card_html(content: str):
    """Render arbitrary HTML inside a bordered card."""
    st.markdown(
        f"<div style='background:{C['card']};border:1px solid {C['border']};"
        f"border-radius:6px;padding:1.25rem 1.5rem;margin-bottom:0.75rem;'>"
        f"{content}</div>",
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "info"):
    """
    Render a left-bordered callout block.
    kind: 'info' | 'success' | 'warning' | 'danger'
    """
    colors = {
        "info":    (C["accent_soft"],  C["accent"]),
        "success": ("#EAF2EA",         C["success"]),
        "warning": ("#FAF3E0",         C["warning"]),
        "danger":  ("#FAE8E8",         C["danger"]),
    }
    bg, border = colors.get(kind, colors["info"])
    st.markdown(
        f"<div style='background:{bg};border-left:3px solid {border};"
        f"border-radius:0 4px 4px 0;padding:0.9rem 1.2rem;"
        f"margin-bottom:0.75rem;font-size:0.875rem;"
        f"color:{C['text']};line-height:1.6;'>{text}</div>",
        unsafe_allow_html=True,
    )


def section_tag(text: str):
    """Small all-caps section label."""
    st.markdown(
        f"<p style='font-size:0.72rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:{C['muted']};margin-bottom:0.4rem;"
        f"font-weight:600;'>{text}</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 2 ── FORMATTERS
# ═══════════════════════════════════════════════════════════════

def fmt_currency(v, symbol="$") -> str:
    """Format a number as currency string. e.g. 1234.5 → '$1,234.50'"""
    if v is None:
        return f"{symbol}0.00"
    prefix = "-" if v < 0 else ""
    return f"{prefix}{symbol}{abs(v):,.2f}"


def fmt_pct(v, dp=1) -> str:
    """Format a number as percentage. e.g. 34.5 → '34.5%'"""
    if v is None:
        return "0.0%"
    return f"{v:.{dp}f}%"


def profit_margin(selling: float, cost: float) -> float:
    """Gross margin %: (selling - cost) / selling * 100"""
    if not selling or selling == 0:
        return 0.0
    return ((selling - cost) / selling) * 100


def suggested_price(cost: float, target_margin_pct: float) -> float:
    """Price needed to hit target margin: cost / (1 - margin/100)"""
    target_margin_pct = min(max(target_margin_pct, 0), 99)
    return round(cost / (1 - target_margin_pct / 100), 2)


def delta_str(current: float, previous: float, currency=False) -> str:
    """Return a delta string for st.metric. e.g. '+$120.00' or '+12.3%'"""
    diff = current - previous
    if currency:
        sign = "+" if diff >= 0 else ""
        return f"{sign}{fmt_currency(diff)}"
    pct = ((current - previous) / previous * 100) if previous else 0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


# ═══════════════════════════════════════════════════════════════
# SECTION 3 ── PLOTLY CHART BUILDERS
# ═══════════════════════════════════════════════════════════════

_CHART_COLORS = ["#2C5F2E", "#5A8A5D", "#8AB58C", "#C0D9C1", "#3D6B3F", "#1A3D1C"]


def _base_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply shared theme to any Plotly figure."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Georgia, serif", size=14, color=C["text"]),
            x=0, xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Helvetica Neue, sans-serif", color=C["text"], size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"], borderwidth=1),
    )
    fig.update_xaxes(showgrid=False, color=C["muted"], linecolor=C["border"])
    fig.update_yaxes(showgrid=True, gridcolor=C["border"], color=C["muted"],
                     linecolor="rgba(0,0,0,0)")
    return fig


def chart_revenue_line(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of daily revenue over time.
    Expects columns: sold_date, total_revenue
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["sold_date"], y=df["total_revenue"],
        mode="lines",
        line=dict(color=C["accent"], width=2),
        fill="tozeroy", fillcolor="rgba(44,95,46,0.07)",
        name="Revenue",
        hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    return _base_layout(fig, "Daily Revenue")


def chart_top_items_bar(df: pd.DataFrame, value_col="total_revenue") -> go.Figure:
    """
    Horizontal bar chart for top items.
    Expects columns: item_name, [value_col]
    """
    df = df.sort_values(value_col, ascending=True).tail(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[value_col], y=df["item_name"], orientation="h",
        marker=dict(color=C["accent"], line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>$%{x:,.2f}<extra></extra>",
    ))
    return _base_layout(fig, "Top Items by Revenue")


def chart_peak_hours(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of revenue by hour of day.
    Expects columns: hour, total_revenue
    """
    labels = [f"{int(h):02d}:00" for h in df["hour"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=df["total_revenue"],
        marker=dict(color=C["accent"], line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    return _base_layout(fig, "Revenue by Hour of Day")


def chart_category_donut(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart of revenue by category.
    Expects columns: category, total_revenue
    """
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=df["category"], values=df["total_revenue"],
        hole=0.55,
        marker=dict(colors=_CHART_COLORS),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<extra></extra>",
    ))
    return _base_layout(fig, "Revenue by Category")


def chart_cost_vs_price(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing cost_price vs selling_price per item.
    Expects columns: item_name, cost_price, selling_price
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Cost", x=df["item_name"], y=df["cost_price"],
                         marker_color="#C0D9C1",
                         hovertemplate="Cost: $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Selling Price", x=df["item_name"], y=df["selling_price"],
                         marker_color=C["accent"],
                         hovertemplate="Price: $%{y:.2f}<extra></extra>"))
    fig.update_layout(barmode="group")
    return _base_layout(fig, "Cost vs Selling Price")


# ═══════════════════════════════════════════════════════════════
# SECTION 4 ── DATA CLEANING PIPELINE
# ═══════════════════════════════════════════════════════════════

# Common abbreviation → canonical name map
ITEM_ALIASES = {
    "flat wht": "flat white", "flatwhite": "flat white",
    "long blk": "long black", "long blck": "long black",
    "cap": "cappuccino", "capp": "cappuccino",
    "lte": "latte", "oat lte": "oat milk latte",
    "oat latte": "oat milk latte",
    "cold brew coffee": "cold brew", "cb": "cold brew",
    "iced lat": "iced latte", "matcha": "matcha latte",
    "avo toast": "avocado toast", "avo on toast": "avocado toast",
    "eggs benny": "eggs benedict", "eggs bene": "eggs benedict",
    "gran bowl": "granola bowl", "bb": "banana bread",
    "chk sndwch": "chicken sandwich",
}

CATEGORY_KEYWORDS = {
    "Coffee":      ["latte","flat white","cappuccino","long black","espresso",
                    "cold brew","matcha","chai","macchiato","mocha","affogato"],
    "Cold Drinks": ["iced","smoothie","frappe","milkshake","slushie"],
    "Food":        ["toast","eggs","sandwich","salad","bowl","muffin",
                    "croissant","cake","scone","waffle","pancake","wrap",
                    "burger","bread"],
    "Drinks":      ["juice","water","soda","tea","kombucha"],
}

COLUMN_ALIASES = {
    "product": "item_name", "product_name": "item_name",
    "menu_item": "item_name", "name": "item_name",
    "quantity": "qty", "units_sold": "qty",
    "amount": "revenue", "total": "revenue", "sale_amount": "revenue",
    "date": "sold_at", "sale_date": "sold_at", "transaction_date": "sold_at",
    "item_category": "category", "type": "category",
}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full 8-step cleaning pipeline for raw POS/CSV data.
    Returns a cleaned DataFrame ready for database insertion.
    """
    df = df.copy()
    df = _normalize_columns(df)
    df = _remove_duplicates(df)
    df = _normalize_item_names(df)
    df = _parse_dates(df)
    df = _clean_numerics(df)
    df = _fill_missing(df)
    df = _assign_categories(df)
    df = _drop_bad_rows(df)
    return df.reset_index(drop=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, snake_case all column names, then apply alias map."""
    renamed = {}
    for col in df.columns:
        n = col.strip().lower()
        n = re.sub(r"[\s\-]+", "_", n)
        n = re.sub(r"[^\w]", "", n)
        n = re.sub(r"_+", "_", n).strip("_")
        renamed[col] = n
    df = df.rename(columns=renamed)
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicate rows."""
    return df.drop_duplicates()


def _normalize_item_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, collapse spaces, apply alias map."""
    if "item_name" not in df.columns:
        return df
    def clean_name(raw):
        if pd.isna(raw):
            return ""
        n = str(raw).strip().lower()
        n = re.sub(r"\s+", " ", n).strip(".,;:")
        return ITEM_ALIASES.get(n, n)
    df["item_name"] = df["item_name"].apply(clean_name)
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse sold_at column to datetime."""
    if "sold_at" in df.columns:
        df["sold_at"] = pd.to_datetime(df["sold_at"], errors="coerce")
    return df


def _clean_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Strip currency symbols and convert revenue/qty to numbers."""
    if "revenue" in df.columns:
        df["revenue"] = (
            df["revenue"].astype(str)
            .str.replace(r"[$£€RM,\s]", "", regex=True)
            .str.replace(r"[^\d.\-]", "", regex=True)
        )
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(
            df["qty"].astype(str).str.replace(r"[^\d]", "", regex=True),
            errors="coerce"
        ).fillna(1).astype(int)
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing category with 'Uncategorized', missing qty with 1."""
    if "category" not in df.columns:
        df["category"] = "Uncategorized"
    else:
        df["category"] = df["category"].fillna("Uncategorized")
    if "qty" in df.columns:
        df["qty"] = df["qty"].fillna(1).astype(int)
    return df


def _assign_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Infer category from item_name keywords where category is missing."""
    if "item_name" not in df.columns or "category" not in df.columns:
        return df
    def infer(row):
        if row["category"] != "Uncategorized":
            return row["category"]
        name = str(row["item_name"]).lower()
        for cat, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in name for kw in keywords):
                return cat
        return "Uncategorized"
    df["category"] = df.apply(infer, axis=1)
    return df


def _drop_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with empty item_name, zero/negative revenue, or unparseable dates."""
    df = df[df["item_name"].notna() & (df["item_name"].str.strip() != "")]
    if "revenue" in df.columns:
        df = df[pd.to_numeric(df["revenue"], errors="coerce").fillna(0) > 0]
    if "sold_at" in df.columns:
        df = df[df["sold_at"].notna()]
    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 5 ── DEMO DATA GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_demo_data() -> pd.DataFrame:
    """
    Build a synthetic 90-day cafe transaction dataset.
    Used when the user clicks 'Load demo data' — no Supabase needed.
    Returns a cleaned DataFrame ready for use.
    """
    rng   = np.random.default_rng(seed=42)
    start = datetime.now() - timedelta(days=90)

    menu = [
        ("flat white",       "Coffee",      4.50, 2.20),
        ("long black",       "Coffee",      4.00, 1.60),
        ("cappuccino",       "Coffee",      4.80, 2.10),
        ("latte",            "Coffee",      5.00, 2.30),
        ("oat milk latte",   "Coffee",      5.80, 2.90),
        ("matcha latte",     "Coffee",      6.00, 2.80),
        ("cold brew",        "Cold Drinks", 5.50, 1.80),
        ("iced latte",       "Cold Drinks", 5.50, 2.50),
        ("banana smoothie",  "Cold Drinks", 7.00, 3.20),
        ("avocado toast",    "Food",       14.00, 5.50),
        ("eggs benedict",    "Food",       18.00, 7.00),
        ("granola bowl",     "Food",       12.00, 4.20),
        ("banana bread",     "Food",        6.50, 2.00),
        ("croissant",        "Food",        5.00, 1.80),
        ("muffin",           "Food",        4.50, 1.50),
        ("chicken sandwich", "Food",       16.00, 6.50),
        ("caesar salad",     "Food",       15.00, 5.80),
        ("sparkling water",  "Drinks",      4.00, 0.80),
        ("orange juice",     "Drinks",      6.00, 1.50),
        ("chai latte",       "Coffee",      5.50, 2.40),
    ]
    weights = [12,10,12,14,8,6,7,8,3,9,5,4,8,10,8,6,5,5,4,6]
    w_norm  = [w/sum(weights) for w in weights]

    records = []
    for day in range(90):
        date = start + timedelta(days=day)
        for _ in range(int(rng.integers(55, 130))):
            idx  = rng.choice(len(menu), p=w_norm)
            name, cat, price, cost = menu[idx]
            qty  = int(rng.choice([1,1,1,2], p=[0.70,0.15,0.10,0.05]))
            hour = rng.choice(
                list(range(7, 19)),
                p=[0.15,0.18,0.15,0.08,0.06,0.10,0.08,0.06,0.05,0.04,0.03,0.02]
            )
            sold_at = date.replace(
                hour=int(hour),
                minute=int(rng.integers(0, 59)),
                second=0, microsecond=0
            )
            records.append({
                "item_name": name,
                "category":  cat,
                "qty":       qty,
                "revenue":   round(price * qty, 2),
                "cost":      round(cost * qty, 2),
                "sold_at":   sold_at,
            })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# SECTION 6 ── SUPABASE DATABASE LAYER
# ═══════════════════════════════════════════════════════════════

def get_supabase_client():
    """
    Return a Supabase client or None if credentials are not configured.
    Reads SUPABASE_URL and SUPABASE_KEY from st.secrets or environment.
    """
    try:
        from supabase import create_client
        # Try Streamlit secrets first, then environment variables
        url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
        key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))
        if url and key and url.startswith("https://"):
            return create_client(url, key)
    except Exception:
        pass
    return None


def db_save_transactions(records: list[dict]) -> bool:
    """
    Insert transaction records into Supabase in batches of 500.
    Returns True on success, False if Supabase is not available.
    """
    client = get_supabase_client()
    if not client:
        return False
    try:
        for i in range(0, len(records), 500):
            client.table("transactions").insert(records[i:i+500]).execute()
        return True
    except Exception as e:
        st.warning(f"Database write warning: {e}")
        return False


def db_load_transactions() -> pd.DataFrame:
    """
    Fetch all transactions from Supabase.
    Returns empty DataFrame if Supabase is unavailable.
    """
    client = get_supabase_client()
    if not client:
        return pd.DataFrame()
    try:
        resp = client.table("transactions").select("*").order("sold_at", desc=True).execute()
        df   = pd.DataFrame(resp.data or [])
        if not df.empty and "sold_at" in df.columns:
            df["sold_at"] = pd.to_datetime(df["sold_at"])
        return df
    except Exception:
        return pd.DataFrame()


def db_save_menu_items(items: list[dict]) -> bool:
    """Upsert menu items into Supabase. Returns True on success."""
    client = get_supabase_client()
    if not client or not items:
        return False
    try:
        client.table("menu_items").upsert(items, on_conflict="item_name").execute()
        return True
    except Exception:
        return False


def db_save_menu_costs(costs: list[dict]) -> bool:
    """Upsert cost records into Supabase. Returns True on success."""
    client = get_supabase_client()
    if not client or not costs:
        return False
    try:
        client.table("menu_costs").upsert(costs, on_conflict="item_name").execute()
        return True
    except Exception:
        return False


def db_load_menu_costs() -> pd.DataFrame:
    """Fetch all cost records from Supabase."""
    client = get_supabase_client()
    if not client:
        return pd.DataFrame()
    try:
        resp = client.table("menu_costs").select("*").execute()
        return pd.DataFrame(resp.data or [])
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# SECTION 7 ── ANALYTICS COMPUTATIONS
# ═══════════════════════════════════════════════════════════════

def calc_total_revenue(df: pd.DataFrame) -> float:
    """Sum all revenue rows."""
    if df.empty or "revenue" not in df.columns:
        return 0.0
    return float(df["revenue"].sum())


def calc_total_profit(txn_df: pd.DataFrame, costs_df: pd.DataFrame) -> float:
    """
    Estimate gross profit by joining transactions with cost data.
    Items without cost records are excluded.
    """
    if txn_df.empty or costs_df.empty or "cost_price" not in costs_df.columns:
        # Fall back to cost column if present in demo data
        if "cost" in txn_df.columns:
            return float((txn_df["revenue"] - txn_df["cost"]).sum())
        return 0.0
    merged = txn_df.merge(costs_df[["item_name","cost_price"]], on="item_name", how="inner")
    if merged.empty:
        return 0.0
    merged["line_profit"] = merged["revenue"] - merged["qty"] * merged["cost_price"]
    return float(merged["line_profit"].sum())


def calc_food_cost_pct(txn_df: pd.DataFrame, costs_df: pd.DataFrame) -> float:
    """Food cost % = total cost / total revenue × 100. Benchmark: 28–35%."""
    if txn_df.empty:
        return 0.0
    if "cost" in txn_df.columns:
        total_cost = float(txn_df["cost"].sum())
        total_rev  = float(txn_df["revenue"].sum())
        return (total_cost / total_rev * 100) if total_rev else 0.0
    if costs_df.empty or "cost_price" not in costs_df.columns:
        return 0.0
    merged = txn_df.merge(costs_df[["item_name","cost_price"]], on="item_name", how="inner")
    if merged.empty:
        return 0.0
    total_cost = float((merged["qty"] * merged["cost_price"]).sum())
    total_rev  = float(merged["revenue"].sum())
    return (total_cost / total_rev * 100) if total_rev else 0.0


def calc_avg_order(df: pd.DataFrame) -> float:
    """Mean revenue per transaction row."""
    if df.empty or "revenue" not in df.columns:
        return 0.0
    return float(df["revenue"].mean())


def calc_period_delta(df: pd.DataFrame, days: int = 30) -> dict:
    """
    Compare current period revenue/orders vs the prior equivalent period.
    Returns a dict with current_revenue, previous_revenue, change_pct, etc.
    """
    empty = {"current_revenue":0,"previous_revenue":0,"revenue_change_pct":0,
             "current_orders":0,"previous_orders":0}
    if df.empty or "sold_at" not in df.columns:
        return empty
    df = df.copy()
    df["sold_at"] = pd.to_datetime(df["sold_at"])
    now      = df["sold_at"].max()
    cur_start = now - timedelta(days=days)
    prv_start = cur_start - timedelta(days=days)
    cur = df[df["sold_at"] >= cur_start]
    prv = df[(df["sold_at"] >= prv_start) & (df["sold_at"] < cur_start)]
    curr_rev = float(cur["revenue"].sum())
    prev_rev = float(prv["revenue"].sum())
    chg      = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
    return {
        "current_revenue":    curr_rev,
        "previous_revenue":   prev_rev,
        "revenue_change_pct": round(chg, 1),
        "current_orders":     len(cur),
        "previous_orders":    len(prv),
    }


def calc_top_items(df: pd.DataFrame, n=10) -> pd.DataFrame:
    """Top N items by total revenue."""
    if df.empty:
        return pd.DataFrame()
    return (df.groupby("item_name")
              .agg(total_revenue=("revenue","sum"), total_qty=("qty","sum"))
              .reset_index()
              .sort_values("total_revenue", ascending=False)
              .head(n))


def calc_worst_items(df: pd.DataFrame, n=8) -> pd.DataFrame:
    """Bottom N items by total revenue."""
    if df.empty:
        return pd.DataFrame()
    return (df.groupby("item_name")
              .agg(total_revenue=("revenue","sum"), total_qty=("qty","sum"))
              .reset_index()
              .sort_values("total_revenue", ascending=True)
              .head(n))


def calc_peak_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and order count aggregated by hour of day (0–23)."""
    if df.empty or "sold_at" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["sold_at"] = pd.to_datetime(df["sold_at"])
    df["hour"] = df["sold_at"].dt.hour
    return (df.groupby("hour")
              .agg(total_revenue=("revenue","sum"), order_count=("qty","count"))
              .reset_index()
              .sort_values("hour"))


def calc_daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue aggregated by calendar day."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["sold_at"]  = pd.to_datetime(df["sold_at"])
    df["sold_date"] = df["sold_at"].dt.date
    daily = df.groupby("sold_date")["revenue"].sum().reset_index()
    daily.columns = ["sold_date", "total_revenue"]
    return daily.sort_values("sold_date")


def calc_category_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue by menu category."""
    if df.empty or "category" not in df.columns:
        return pd.DataFrame()
    g = df.groupby("category")["revenue"].sum().reset_index()
    g.columns = ["category","total_revenue"]
    return g.sort_values("total_revenue", ascending=False)


def filter_period(df: pd.DataFrame, days) -> pd.DataFrame:
    """Filter DataFrame to the last N days. Pass None for all time."""
    if df.empty or days is None or "sold_at" not in df.columns:
        return df
    df = df.copy()
    df["sold_at"] = pd.to_datetime(df["sold_at"])
    cutoff = df["sold_at"].max() - pd.Timedelta(days=days)
    return df[df["sold_at"] >= cutoff]


# ═══════════════════════════════════════════════════════════════
# SECTION 8 ── PRICE OPTIMIZER
# ═══════════════════════════════════════════════════════════════

MIN_MARGIN     = 55.0   # Minimum acceptable gross margin %
TARGET_BUMP    = 6.0    # Default recommended margin increase in pct points
MAX_PRICE_STEP = 0.05   # Max single price increase (5%) to avoid price shock


def build_margin_table(txn_df: pd.DataFrame, costs_dict: dict) -> pd.DataFrame:
    """
    Build a per-item margin analysis table.

    costs_dict: {item_name: cost_price}
    Derives selling_price from average revenue/qty in transactions.
    Returns DataFrame with margin and tier columns.
    """
    if txn_df.empty:
        return pd.DataFrame()

    # Compute unit price from transactions
    df = txn_df.copy()
    df["unit_price"] = df.apply(
        lambda r: r["revenue"] / r["qty"] if r["qty"] > 0 else r["revenue"], axis=1
    )
    summary = df.groupby("item_name").agg(
        selling_price=("unit_price", "mean"),
        quantity_sold=("qty", "sum"),
        category=("category", lambda x: x.mode()[0] if len(x) > 0 else "Uncategorized"),
    ).reset_index()

    # Attach cost prices
    summary["cost_price"] = summary["item_name"].map(costs_dict).fillna(0)
    summary = summary[summary["cost_price"] > 0]   # only items with cost data

    summary["margin_pct"] = summary.apply(
        lambda r: profit_margin(r["selling_price"], r["cost_price"]), axis=1
    )
    summary["tier"] = summary["margin_pct"].apply(_margin_tier)
    return summary.sort_values("margin_pct", ascending=True)


def _margin_tier(pct: float) -> str:
    """Low / Fair / Good / Strong margin tier label."""
    if pct < 55:  return "Low"
    if pct < 65:  return "Fair"
    if pct < 75:  return "Good"
    return "Strong"


def build_price_recommendations(margin_df: pd.DataFrame, bump: float = TARGET_BUMP) -> pd.DataFrame:
    """
    For each item in margin_df, compute a suggested price to hit
    (current_margin + bump), capped at a 5% single-step increase.
    Returns the same DataFrame with recommendation columns added.
    """
    if margin_df.empty:
        return pd.DataFrame()

    rows = []
    for _, r in margin_df.iterrows():
        target = min(r["margin_pct"] + bump, 85.0)
        ideal  = suggested_price(r["cost_price"], target)
        cap    = round(r["selling_price"] * (1 + MAX_PRICE_STEP), 2)
        sugg   = min(ideal, cap)
        sugg   = max(sugg, r["selling_price"])

        new_margin = profit_margin(sugg, r["cost_price"])
        increase   = sugg - r["selling_price"]
        gain       = (sugg - r["cost_price"] - (r["selling_price"] - r["cost_price"])) * r["quantity_sold"]

        rows.append({**r.to_dict(),
                     "suggested_price":   round(sugg, 2),
                     "new_margin_pct":    round(new_margin, 1),
                     "price_increase":    round(increase, 2),
                     "increase_pct":      round((increase / r["selling_price"] * 100) if r["selling_price"] else 0, 1),
                     "est_profit_gain":   round(gain, 2)})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# SECTION 9 ── RECOMMENDATIONS ENGINE
# ═══════════════════════════════════════════════════════════════

def generate_recommendations(txn_df: pd.DataFrame, margin_df: pd.DataFrame) -> list[dict]:
    """
    Run all insight generators and return a combined sorted list of
    recommendation dicts: {priority, type, title, detail, impact}
    priority: 1=Urgent, 2=Important, 3=Opportunity
    """
    recs = []
    recs += _recs_price_increases(margin_df)
    recs += _recs_remove_items(txn_df, margin_df)
    recs += _recs_combo_offers(txn_df)
    recs += _recs_promote_high_margin(margin_df)
    recs += _recs_peak_hours(txn_df)
    recs += _recs_inventory(txn_df)
    recs.sort(key=lambda r: (r["priority"], r["type"]))
    return recs


def _recs_price_increases(margin_df: pd.DataFrame) -> list[dict]:
    if margin_df.empty or "margin_pct" not in margin_df.columns:
        return []
    low = margin_df[margin_df["margin_pct"] < 60].head(4)
    result = []
    for _, r in low.iterrows():
        name  = r["item_name"].title()
        price = r["selling_price"]
        sugg  = round(price * 1.05, 2)
        result.append({
            "priority": 1, "type": "pricing",
            "title":  f"Increase the price of {name}",
            "detail": (f"{name} runs at {r['margin_pct']:.1f}% margin — below the 60% target. "
                       f"A small increase from {fmt_currency(price)} to {fmt_currency(sugg)} "
                       f"(5%) would improve margins without meaningful customer resistance."),
            "impact": f"Estimated +5–7% margin on {name}",
        })
    return result


def _recs_remove_items(txn_df: pd.DataFrame, margin_df: pd.DataFrame) -> list[dict]:
    if txn_df.empty or margin_df.empty:
        return []
    if "sold_at" not in txn_df.columns:
        return []
    dates = pd.to_datetime(txn_df["sold_at"])
    weeks = max((dates.max() - dates.min()).days / 7, 1)
    weekly_qty = (txn_df.groupby("item_name")["qty"].sum() / weeks).reset_index()
    weekly_qty.columns = ["item_name","avg_weekly_qty"]
    low_margin_items = set(margin_df[margin_df["margin_pct"] < 50]["item_name"])
    candidates = weekly_qty[
        (weekly_qty["avg_weekly_qty"] < 5) & (weekly_qty["item_name"].isin(low_margin_items))
    ]
    result = []
    for _, r in candidates.head(3).iterrows():
        name = r["item_name"].title()
        result.append({
            "priority": 2, "type": "menu",
            "title":  f"Consider removing {name} from the menu",
            "detail": (f"{name} averages {r['avg_weekly_qty']:.1f} units per week "
                       f"and has a below-target margin. Removing it reduces kitchen complexity "
                       f"and lets staff focus on faster, more profitable items."),
            "impact": "Reduces waste and kitchen complexity",
        })
    return result


def _recs_combo_offers(txn_df: pd.DataFrame) -> list[dict]:
    if txn_df.empty or "category" not in txn_df.columns:
        return []
    by_cat = txn_df.groupby(["item_name","category"])["revenue"].sum().reset_index()
    foods   = by_cat[by_cat["category"] == "Food"].sort_values("revenue", ascending=False)
    coffees = by_cat[by_cat["category"].isin(["Coffee","Cold Drinks"])].sort_values("revenue", ascending=False)
    if foods.empty or coffees.empty:
        return []
    food   = foods.iloc[0]["item_name"].title()
    coffee = coffees.iloc[0]["item_name"].title()
    return [{
        "priority": 3, "type": "marketing",
        "title":  f"Bundle {coffee} + {food} as a meal deal",
        "detail": (f"{coffee} and {food} are your top revenue earners in their categories. "
                   f"A named combo at a 5–8% bundle discount encourages upselling and "
                   f"increases average transaction value during the morning rush."),
        "impact": "Estimated 8–12% increase in average basket size",
    }]


def _recs_promote_high_margin(margin_df: pd.DataFrame) -> list[dict]:
    if margin_df.empty or "margin_pct" not in margin_df.columns:
        return []
    high = margin_df[margin_df["margin_pct"] > 70]
    if high.empty:
        return []
    top = high.sort_values("margin_pct", ascending=False).iloc[0]
    name = top["item_name"].title()
    return [{
        "priority": 2, "type": "marketing",
        "title":  f"Feature {name} as your daily special",
        "detail": (f"{name} carries a {top['margin_pct']:.1f}% gross margin — among your highest. "
                   f"Positioning it prominently on the menu board and training staff to recommend "
                   f"it will grow volume without any discounting."),
        "impact": f"20% volume increase on {name} adds proportionally to net profit",
    }]


def _recs_peak_hours(txn_df: pd.DataFrame) -> list[dict]:
    if txn_df.empty or "sold_at" not in txn_df.columns:
        return []
    df = txn_df.copy()
    df["sold_at"] = pd.to_datetime(df["sold_at"])
    df["hour"]    = df["sold_at"].dt.hour
    hourly = df.groupby("hour")["revenue"].sum()
    if hourly.empty:
        return []
    peak    = int(hourly.idxmax())
    slowest = int(hourly.idxmin())
    return [
        {"priority": 3, "type": "operations",
         "title":  f"Staff up between {peak:02d}:00 and {peak+1:02d}:00",
         "detail": (f"Your highest-revenue hour is {peak:02d}:00–{peak+1:02d}:00. "
                    f"Adequate staffing here reduces wait times and prevents lost sales."),
         "impact": "Prevents revenue loss during peak window"},
        {"priority": 3, "type": "operations",
         "title":  f"Run a promotion at {slowest:02d}:00 to lift slow hours",
         "detail": (f"Revenue is lowest at {slowest:02d}:00. A time-limited deal — coffee + snack "
                    f"bundle — can convert this dead hour into incremental revenue."),
         "impact": "Converts low-traffic hours into added revenue"},
    ]


def _recs_inventory(txn_df: pd.DataFrame) -> list[dict]:
    if txn_df.empty:
        return []
    top = txn_df.groupby("item_name")["qty"].sum().idxmax()
    return [{
        "priority": 3, "type": "operations",
        "title":  f"Review supply agreements for {top.title()}",
        "detail": (f"{top.title()} is your highest-volume item by units sold. "
                   f"A volume discount with your supplier or a 3-day buffer stock "
                   f"prevents stock-outs that directly cut revenue."),
        "impact": "Eliminates revenue loss from stock-outs on top seller",
    }]


# ═══════════════════════════════════════════════════════════════
# SECTION 10 ── REPORT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_report(txn_df: pd.DataFrame, costs_df: pd.DataFrame, days: int) -> dict:
    """
    Compile a report package for the given rolling window.
    Returns a dict of DataFrames and scalar KPIs.
    """
    df        = filter_period(txn_df, days)
    start     = (datetime.now() - timedelta(days=days)).strftime("%d %b %Y")
    end       = datetime.now().strftime("%d %b %Y")
    rev       = calc_total_revenue(df)
    profit    = calc_total_profit(df, costs_df)
    food_cost = calc_food_cost_pct(df, costs_df)
    orders    = len(df)
    avg       = calc_avg_order(df)

    return {
        "period":    f"{start} – {end}",
        "summary":   {"revenue": rev, "profit": profit,
                      "food_cost_pct": food_cost, "orders": orders, "avg_order": avg},
        "daily_df":  calc_daily_revenue(df),
        "top_df":    calc_top_items(df, 10),
        "worst_df":  calc_worst_items(df, 5),
        "hours_df":  calc_peak_hours(df),
    }


def export_csv(report: dict) -> bytes:
    """Combine all report DataFrames into a single downloadable CSV."""
    buf = io.StringIO()
    buf.write(f"Seralung Opti Report\nPeriod: {report['period']}\n"
              f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}\n\n")
    s = report["summary"]
    buf.write("--- Summary ---\n")
    buf.write(f"Revenue,{s['revenue']:.2f}\nProfit,{s['profit']:.2f}\n"
              f"Food Cost %,{s['food_cost_pct']:.1f}\nTransactions,{s['orders']}\n"
              f"Avg Order,{s['avg_order']:.2f}\n\n")
    for label, key in [("Top Items","top_df"),("Lowest Performing Items","worst_df"),
                       ("Daily Revenue","daily_df"),("Peak Hours","hours_df")]:
        df = report.get(key, pd.DataFrame())
        if not df.empty:
            buf.write(f"--- {label} ---\n")
            df.to_csv(buf, index=False)
            buf.write("\n")
    return buf.getvalue().encode("utf-8")


def send_email(recipient: str, period: str, csv_bytes: bytes) -> tuple[bool, str]:
    """
    Send the report CSV by email via SMTP.
    Reads credentials from st.secrets or environment variables.
    """
    try:
        host  = st.secrets.get("SMTP_HOST",  os.getenv("SMTP_HOST",  "smtp.gmail.com"))
        port  = int(st.secrets.get("SMTP_PORT", os.getenv("SMTP_PORT", "587")))
        user  = st.secrets.get("SMTP_USER",  os.getenv("SMTP_USER",  ""))
        pwd   = st.secrets.get("SMTP_PASSWORD", os.getenv("SMTP_PASSWORD", ""))
        frm   = st.secrets.get("REPORT_FROM_EMAIL", user)
    except Exception:
        host, port, user, pwd, frm = "smtp.gmail.com", 587, "", "", ""

    if not user or not pwd:
        return False, "Email credentials not configured. Add SMTP_USER and SMTP_PASSWORD to Streamlit secrets."

    msg = MIMEMultipart()
    msg["From"]    = frm
    msg["To"]      = recipient
    msg["Subject"] = f"Seralung Opti — Report for {period}"
    msg.attach(MIMEText(f"Please find attached the Seralung Opti report for {period}.\n\n— Seralung Opti", "plain"))
    att = MIMEBase("application", "octet-stream")
    att.set_payload(csv_bytes)
    encoders.encode_base64(att)
    att.add_header("Content-Disposition", f"attachment; filename=seralung_report.csv")
    msg.attach(att)

    try:
        with smtplib.SMTP(host, port) as srv:
            srv.ehlo(); srv.starttls(); srv.login(user, pwd)
            srv.sendmail(frm, recipient, msg.as_string())
        return True, f"Report sent to {recipient}."
    except Exception as e:
        return False, f"Email failed: {e}"


# ═══════════════════════════════════════════════════════════════
# SECTION 11 ── SESSION STATE HELPERS
# ═══════════════════════════════════════════════════════════════

def get_txn() -> pd.DataFrame:
    """Return the current session transactions DataFrame."""
    return st.session_state.get("transactions", pd.DataFrame())


def get_costs() -> dict:
    """Return the current session cost dictionary {item_name: cost_price}."""
    return st.session_state.get("costs", {})


def set_txn(df: pd.DataFrame):
    """Store transactions in session state."""
    st.session_state["transactions"] = df


def set_costs(d: dict):
    """Store costs dict in session state."""
    st.session_state["costs"] = d


def init_session():
    """Initialise session state keys on first load."""
    if "transactions" not in st.session_state:
        st.session_state["transactions"] = pd.DataFrame()
    if "costs" not in st.session_state:
        st.session_state["costs"] = {}
    if "report" not in st.session_state:
        st.session_state["report"] = None
    if "report_csv" not in st.session_state:
        st.session_state["report_csv"] = b""


# ═══════════════════════════════════════════════════════════════
# SECTION 12 ── SIDEBAR
# ═══════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """
    Render the sidebar controls and return a dict of current user settings:
    {period_days, period_label, target_margin_bump, rec_types, rec_priority_filter}
    """
    with st.sidebar:
        st.markdown(
            f"<div style='padding:1rem 0 1.25rem'>"
            f"<p style='font-family:Georgia,serif;font-size:1.2rem;"
            f"color:{C['text']};margin:0;letter-spacing:-0.02em;'>Seralung Opti</p>"
            f"<p style='font-size:0.72rem;color:{C['muted']};margin-top:0.1rem;"
            f"letter-spacing:0.04em;'>Cafe Business Intelligence</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Data import ──
        section_tag("Import Data")
        uploaded = st.file_uploader(
            "Upload POS / CSV export", type=["csv"],
            help="Required columns: item_name, qty, revenue, sold_at"
        )
        if uploaded:
            try:
                raw = pd.read_csv(uploaded)
                cleaned = clean_dataframe(raw)
                if cleaned.empty:
                    st.error("No valid rows found after cleaning. Check your CSV columns.")
                else:
                    set_txn(cleaned)
                    db_save_transactions(cleaned.to_dict(orient="records"))
                    st.success(f"Imported {len(cleaned):,} rows.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

        if st.button("Load demo data", use_container_width=True):
            demo = generate_demo_data()
            set_txn(demo)
            st.success(f"Demo loaded — {len(demo):,} transactions.")

        st.markdown("---")

        # ── Date filter ──
        section_tag("Date Range")
        period_option = st.selectbox(
            "Period", ["Last 7 days","Last 30 days","Last 90 days","All time"], index=2,
            label_visibility="collapsed"
        )
        period_map  = {"Last 7 days":7,"Last 30 days":30,"Last 90 days":90,"All time":None}
        period_days = period_map[period_option]

        st.markdown("---")

        # ── Price settings ──
        section_tag("Price Optimizer")
        margin_bump = st.slider(
            "Target margin increase (%)", min_value=2, max_value=15,
            value=6, step=1, label_visibility="visible"
        )

        st.markdown("---")
        st.markdown(
            f"<p style='font-size:0.7rem;color:{C['muted']};'>v1.0.0 · Seralung Opti</p>",
            unsafe_allow_html=True
        )

    return {
        "period_days":         period_days,
        "period_label":        period_option,
        "margin_bump":         margin_bump,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 13 ── PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════

def page_overview(txn_df: pd.DataFrame, costs_df_db: pd.DataFrame, settings: dict):
    """Render the Overview tab — KPIs, charts, top/worst items, peak hours."""
    page_title("Overview", f"Performance summary · {settings['period_label']}")

    if txn_df.empty:
        callout(
            "No transaction data loaded. Use <strong>Load demo data</strong> or "
            "upload a CSV file in the sidebar to get started.",
            kind="info"
        )
        return

    days = settings["period_days"]
    df   = filter_period(txn_df, days)

    if df.empty:
        callout("No data for the selected period.", kind="warning")
        return

    # ── KPI strip ──
    section_tag("Key Metrics")
    rev       = calc_total_revenue(df)
    profit    = calc_total_profit(df, costs_df_db)
    food_pct  = calc_food_cost_pct(df, costs_df_db)
    avg_ord   = calc_avg_order(df)
    delta     = calc_period_delta(df, days=min(days or 30, 30))

    k1, k2, k3, k4 = st.columns(4, gap="small")
    with k1:
        st.metric("Total Revenue", fmt_currency(rev),
                  delta=delta_str(delta["current_revenue"], delta["previous_revenue"], currency=True)
                  if delta["previous_revenue"] > 0 else None)
    with k2:
        st.metric("Gross Profit", fmt_currency(profit) if profit else "Add cost data")
    with k3:
        st.metric("Food Cost %", fmt_pct(food_pct) if food_pct else "Add cost data")
    with k4:
        st.metric("Avg Order Value", fmt_currency(avg_ord))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Revenue trend ──
    section_tag("Revenue Trend")
    daily = calc_daily_revenue(df)
    if not daily.empty:
        st.plotly_chart(chart_revenue_line(daily), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top + worst ──
    col_top, col_worst = st.columns(2, gap="large")
    with col_top:
        section_tag("Top Selling Items")
        top = calc_top_items(df)
        if not top.empty:
            st.plotly_chart(chart_top_items_bar(top), use_container_width=True)

    with col_worst:
        section_tag("Lowest Performing Items")
        worst = calc_worst_items(df)
        if not worst.empty:
            disp = worst.copy()
            disp.columns = ["Item","Revenue","Units Sold"]
            disp["Revenue"] = disp["Revenue"].apply(fmt_currency)
            st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Peak hours + category ──
    col_h, col_c = st.columns(2, gap="large")
    with col_h:
        section_tag("Peak Trading Hours")
        peak = calc_peak_hours(df)
        if not peak.empty:
            st.plotly_chart(chart_peak_hours(peak), use_container_width=True)

    with col_c:
        section_tag("Revenue by Category")
        cat = calc_category_revenue(df)
        if not cat.empty:
            st.plotly_chart(chart_category_donut(cat), use_container_width=True)

    # ── Raw data expander ──
    with st.expander("View raw transaction data"):
        disp = df.copy()
        if "revenue" in disp.columns:
            disp["revenue"] = disp["revenue"].apply(fmt_currency)
        cols = [c for c in ["item_name","category","qty","revenue","sold_at"] if c in disp.columns]
        st.dataframe(disp[cols].head(500), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 14 ── PAGE: PRICE CALCULATOR
# ═══════════════════════════════════════════════════════════════

def page_price_calculator(txn_df: pd.DataFrame, settings: dict):
    """Render the Price Calculator tab."""
    page_title("Price Calculator", "Margin analysis and optimized price recommendations.")

    if txn_df.empty:
        callout("No data loaded. Upload a CSV or load demo data in the sidebar.", kind="info")
        return

    # Derive item catalog from transactions
    df_items = txn_df.copy()
    df_items["unit_price"] = df_items.apply(
        lambda r: r["revenue"] / r["qty"] if r.get("qty", 1) > 0 else r["revenue"], axis=1
    )
    catalog = df_items.groupby("item_name").agg(
        selling_price=("unit_price","mean"),
        quantity_sold=("qty","sum"),
        category=("category", lambda x: x.mode()[0] if len(x) > 0 else "Uncategorized"),
    ).reset_index()

    costs_dict = get_costs()
    bump       = settings["margin_bump"]

    t_cost, t_analysis, t_recs = st.tabs(["Cost Entry","Margin Analysis","Price Recommendations"])

    # ── Tab: Cost Entry ──
    with t_cost:
        st.markdown("<br>", unsafe_allow_html=True)
        callout(
            "Enter the cost to produce each menu item (ingredients + direct labor per unit). "
            "This unlocks margin analysis and price recommendations.",
            kind="info"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        section_tag("Cost Prices")

        items_list = sorted(catalog["item_name"].tolist())
        new_costs  = {}

        for i in range(0, len(items_list), 4):
            cols = st.columns(4, gap="small")
            for col, name in zip(cols, items_list[i:i+4]):
                with col:
                    price = float(catalog.loc[catalog["item_name"] == name, "selling_price"].values[0])
                    val   = st.number_input(
                        label=name.title(),
                        min_value=0.0,
                        max_value=max(price * 0.99, 0.01),
                        value=float(costs_dict.get(name, 0.0)),
                        step=0.10, format="%.2f",
                        key=f"cost_{name}",
                    )
                    if val > 0:
                        new_costs[name] = val

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Save Cost Prices"):
            set_costs(new_costs)
            db_records = [{"item_name": k, "cost_price": v,
                           "profit_margin": profit_margin(
                               float(catalog.loc[catalog["item_name"]==k,"selling_price"].values[0]), v
                           )} for k, v in new_costs.items()]
            db_save_menu_costs(db_records)
            st.success(f"Saved {len(new_costs)} cost prices.")

    # ── Tab: Margin Analysis ──
    with t_analysis:
        st.markdown("<br>", unsafe_allow_html=True)
        if not costs_dict:
            callout("Enter cost prices in the Cost Entry tab first.", kind="info")
        else:
            margin_df = build_margin_table(txn_df, costs_dict)
            if margin_df.empty:
                callout("No items with cost data found.", kind="info")
            else:
                avg_m  = margin_df["margin_pct"].mean()
                low_n  = (margin_df["margin_pct"] < MIN_MARGIN).sum()
                best   = margin_df.loc[margin_df["margin_pct"].idxmax()]

                k1, k2, k3 = st.columns(3, gap="small")
                with k1: st.metric("Average Margin", fmt_pct(avg_m))
                with k2: st.metric("Items Below Target", f"{low_n} items")
                with k3: st.metric("Best Margin Item",
                                   f"{best['item_name'].title()} ({fmt_pct(best['margin_pct'])})")

                st.markdown("<br>", unsafe_allow_html=True)
                section_tag("Cost vs Selling Price")
                chart_data = margin_df[["item_name","cost_price","selling_price"]].head(14)
                st.plotly_chart(chart_cost_vs_price(chart_data), use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                section_tag("Margin Table")
                disp = margin_df[["item_name","category","selling_price","cost_price","margin_pct","tier"]].copy()
                disp.columns = ["Item","Category","Selling Price","Cost Price","Margin %","Tier"]
                disp["Selling Price"] = disp["Selling Price"].apply(fmt_currency)
                disp["Cost Price"]    = disp["Cost Price"].apply(fmt_currency)
                disp["Margin %"]      = disp["Margin %"].apply(fmt_pct)
                st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Tab: Price Recommendations ──
    with t_recs:
        st.markdown("<br>", unsafe_allow_html=True)
        if not costs_dict:
            callout("Enter cost prices in the Cost Entry tab first.", kind="info")
        else:
            margin_df = build_margin_table(txn_df, costs_dict)
            if margin_df.empty:
                callout("No items with cost data found.", kind="info")
            else:
                opt_df = build_price_recommendations(margin_df, bump=bump)

                # Annual impact estimate
                annual = opt_df["est_profit_gain"].sum() * 13  # 52 weeks / 4-week data window
                if annual > 0:
                    callout(
                        f"Applying all suggested price changes could generate an additional "
                        f"<strong>{fmt_currency(annual)}</strong> in gross profit over 12 months.",
                        kind="success"
                    )

                # High performer spotlight
                high_perf = margin_df[margin_df["margin_pct"] > 70].sort_values("margin_pct", ascending=False).head(5)
                if not high_perf.empty:
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_tag("High-Performing Items — Maintain and Promote")
                    cols = st.columns(min(len(high_perf), 5), gap="small")
                    for col, (_, r) in zip(cols, high_perf.iterrows()):
                        with col:
                            card_html(
                                f"<p style='font-size:0.82rem;font-weight:600;margin:0 0 0.25rem;'>"
                                f"{r['item_name'].title()}</p>"
                                f"<p style='font-family:Georgia,serif;font-size:1.1rem;margin:0;'>"
                                f"{fmt_pct(r['margin_pct'])}</p>"
                                f"<p style='font-size:0.72rem;color:{C['muted']};margin:0.15rem 0 0;'>"
                                f"{r['tier']} performer</p>"
                            )

                # Recommendations table
                st.markdown("<br>", unsafe_allow_html=True)
                section_tag("Price Adjustment Recommendations")
                needs = opt_df[opt_df["price_increase"] > 0]
                if needs.empty:
                    callout("All items are at or above the target margin. No changes needed.", kind="success")
                else:
                    disp = needs[[
                        "item_name","selling_price","suggested_price",
                        "price_increase","increase_pct","margin_pct","new_margin_pct","est_profit_gain"
                    ]].copy()
                    disp.columns = ["Item","Current Price","Suggested Price",
                                    "Increase ($)","Increase (%)","Current Margin","New Margin","Est. Gain"]
                    for c in ["Current Price","Suggested Price","Increase ($)","Est. Gain"]:
                        disp[c] = disp[c].apply(fmt_currency)
                    for c in ["Increase (%)","Current Margin","New Margin"]:
                        disp[c] = disp[c].apply(fmt_pct)
                    st.dataframe(disp, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 15 ── PAGE: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

def page_recommendations(txn_df: pd.DataFrame):
    """Render the Recommendations tab."""
    page_title("Recommendations", "Prioritized actions to grow profit and streamline operations.")

    if txn_df.empty:
        callout("No data loaded. Upload a CSV or load demo data in the sidebar.", kind="info")
        return

    costs_dict = get_costs()
    margin_df  = build_margin_table(txn_df, costs_dict) if costs_dict else pd.DataFrame()
    all_recs   = generate_recommendations(txn_df, margin_df)

    if not all_recs:
        callout("No recommendations generated. All items may already be performing well.", kind="success")
        return

    # Summary strip
    urgent_n = sum(1 for r in all_recs if r["priority"] == 1)
    imp_n    = sum(1 for r in all_recs if r["priority"] == 2)
    opp_n    = sum(1 for r in all_recs if r["priority"] == 3)
    k1, k2, k3 = st.columns(3, gap="small")
    with k1: st.metric("Urgent Actions",    str(urgent_n))
    with k2: st.metric("Important Actions", str(imp_n))
    with k3: st.metric("Opportunities",     str(opp_n))
    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    f_col1, f_col2 = st.columns(2, gap="small")
    with f_col1:
        show_types = st.multiselect(
            "Filter by type",
            ["pricing","menu","marketing","operations"],
            default=["pricing","menu","marketing","operations"],
        )
    with f_col2:
        min_priority = st.selectbox("Minimum priority", ["All","Urgent only","Important and above"])

    filtered = [r for r in all_recs if r["type"] in show_types]
    if min_priority == "Urgent only":
        filtered = [r for r in filtered if r["priority"] == 1]
    elif min_priority == "Important and above":
        filtered = [r for r in filtered if r["priority"] <= 2]

    if not filtered:
        callout("No recommendations match the selected filters.", kind="info")
        return

    st.markdown("<br>", unsafe_allow_html=True)

    PRIORITY_STYLE = {
        1: {"label":"Urgent",      "bg":"#FAE8E8","border":"#8B2E2E","text":"#8B2E2E"},
        2: {"label":"Important",   "bg":"#FAF3E0","border":"#8B6914","text":"#8B6914"},
        3: {"label":"Opportunity", "bg":"#EAF0E8","border":"#2C5F2E","text":"#2C5F2E"},
    }
    TYPE_LABEL = {"pricing":"Pricing","menu":"Menu","marketing":"Marketing","operations":"Operations"}

    for rec_type in ["pricing","menu","marketing","operations"]:
        group = [r for r in filtered if r["type"] == rec_type]
        if not group:
            continue
        section_tag(TYPE_LABEL.get(rec_type, rec_type.title()))
        for rec in group:
            p = PRIORITY_STYLE.get(rec["priority"], PRIORITY_STYLE[3])
            st.markdown(
                f"<div style='background:{C['card']};border:1px solid {C['border']};"
                f"border-left:3px solid {p['border']};border-radius:0 6px 6px 0;"
                f"padding:1.1rem 1.4rem;margin-bottom:0.75rem;'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:flex-start;margin-bottom:0.5rem;'>"
                f"<p style='font-family:Georgia,serif;font-size:0.95rem;"
                f"color:{C['text']};margin:0;'>{rec['title']}</p>"
                f"<span style='font-size:0.68rem;text-transform:uppercase;"
                f"letter-spacing:0.08em;background:{p['bg']};color:{p['text']};"
                f"padding:0.2rem 0.6rem;border-radius:3px;white-space:nowrap;"
                f"margin-left:1rem;font-weight:600;'>{p['label']}</span></div>"
                f"<p style='font-size:0.875rem;color:{C['muted']};margin:0 0 0.4rem;line-height:1.6;'>"
                f"{rec['detail']}</p>"
                f"<p style='font-size:0.78rem;color:{C['muted']};margin:0;font-style:italic;'>"
                f"Expected impact: {rec['impact']}</p></div>",
                unsafe_allow_html=True
            )
        st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 16 ── PAGE: REPORTS
# ═══════════════════════════════════════════════════════════════

def page_reports(txn_df: pd.DataFrame, costs_df_db: pd.DataFrame):
    """Render the Reports tab."""
    page_title("Reports", "Generate, preview, download, and email performance reports.")

    if txn_df.empty:
        callout("No data loaded. Upload a CSV or load demo data in the sidebar.", kind="info")
        return

    r_col, _ = st.columns([3, 5])
    with r_col:
        report_period = st.radio("Period", ["Weekly (7 days)","Monthly (30 days)"], horizontal=True)
        recipient     = st.text_input("Email report to (optional)", placeholder="owner@mycafe.com")

    days = 7 if "Weekly" in report_period else 30

    b1, b2, _ = st.columns([2, 2, 5])
    with b1:
        gen_clicked = st.button("Generate Report", use_container_width=True)
    with b2:
        send_clicked = st.button("Send by Email", use_container_width=True)

    if gen_clicked:
        with st.spinner("Building report…"):
            report = build_report(txn_df, costs_df_db, days)
            st.session_state["report"]     = report
            st.session_state["report_csv"] = export_csv(report)
        st.success("Report ready.")

    if send_clicked:
        if not recipient:
            st.warning("Enter a recipient email address above.")
        elif not st.session_state.get("report"):
            st.warning("Generate the report first.")
        else:
            with st.spinner("Sending…"):
                ok, msg = send_email(recipient,
                                     st.session_state["report"]["period"],
                                     st.session_state["report_csv"])
            st.success(msg) if ok else st.error(msg)

    report = st.session_state.get("report")
    if not report:
        callout("Click <strong>Generate Report</strong> to build your summary.", kind="info")
        return

    st.markdown("---")
    st.markdown(
        f"<p style='font-size:0.85rem;color:{C['muted']};'>Period: {report['period']}</p>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # KPIs
    section_tag("Summary")
    s = report["summary"]
    k1,k2,k3,k4,k5 = st.columns(5, gap="small")
    with k1: st.metric("Revenue",      fmt_currency(s["revenue"]))
    with k2: st.metric("Profit",       fmt_currency(s["profit"]))
    with k3: st.metric("Food Cost %",  fmt_pct(s["food_cost_pct"]))
    with k4: st.metric("Transactions", f"{s['orders']:,}")
    with k5: st.metric("Avg Order",    fmt_currency(s["avg_order"]))

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    if not report["daily_df"].empty:
        section_tag("Daily Revenue")
        st.plotly_chart(chart_revenue_line(report["daily_df"]), use_container_width=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        if not report["top_df"].empty:
            section_tag("Top Items")
            st.plotly_chart(chart_top_items_bar(report["top_df"]), use_container_width=True)
    with c2:
        if not report["hours_df"].empty:
            section_tag("Peak Hours")
            st.plotly_chart(chart_peak_hours(report["hours_df"]), use_container_width=True)

    # Worst items
    if not report["worst_df"].empty:
        st.markdown("<br>", unsafe_allow_html=True)
        section_tag("Lowest Performing Items")
        disp = report["worst_df"].copy()
        disp.columns = ["Item","Revenue","Units Sold"]
        disp["Revenue"] = disp["Revenue"].apply(fmt_currency)
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # Download
    st.markdown("---")
    st.download_button(
        label="Download CSV Report",
        data=st.session_state["report_csv"],
        file_name=f"seralung_report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 17 ── MAIN APP ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    """Bootstrap the app: CSS → session state → sidebar → tabs."""
    inject_css()
    init_session()

    settings = render_sidebar()

    txn_df      = get_txn()
    costs_df_db = pd.DataFrame()   # Supabase cost records if available

    # Try loading persisted costs from Supabase if session has no costs
    if not get_costs():
        db_costs = db_load_menu_costs()
        if not db_costs.empty and "item_name" in db_costs.columns and "cost_price" in db_costs.columns:
            set_costs(dict(zip(db_costs["item_name"], db_costs["cost_price"])))
            costs_df_db = db_costs

    # Try loading transactions from Supabase if session is empty
    if txn_df.empty:
        db_txn = db_load_transactions()
        if not db_txn.empty:
            set_txn(db_txn)
            txn_df = db_txn

    # ── Navigation tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Price Calculator",
        "Recommendations",
        "Reports",
    ])

    with tab1:
        page_overview(txn_df, costs_df_db, settings)

    with tab2:
        page_price_calculator(txn_df, settings)

    with tab3:
        page_recommendations(txn_df)

    with tab4:
        page_reports(txn_df, costs_df_db)


if __name__ == "__main__" or True:
    main()
