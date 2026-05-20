"""
Seralung Opti — Cafe Business Optimization Software
====================================================
Single-file Streamlit application. Zero local imports.
Requirements: streamlit, pandas, numpy, plotly, supabase, python-dotenv

Deploy:
  1. Push app.py + requirements.txt + .streamlit/config.toml to GitHub
  2. Connect repo to Streamlit Cloud — done.
"""

import os, io, re, smtplib, warnings
from datetime import datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Seralung Opti",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════

C = {
    "bg":          "#F7F6F3",
    "card":        "#FFFFFF",
    "border":      "#E5E3DE",
    "text":        "#1C1C1C",
    "muted":       "#5A5856",
    "accent":      "#2C5F2E",
    "accent_soft": "#EAF0E8",
    "success":     "#2D5A30",
    "warning":     "#7A5C12",
    "danger":      "#7A2828",
    "sidebar":     "#EFEDE8",
    "divider":     "#DEDAD4",
    "sidebar_text":"#1C1C1C",
}

# ── Global CSS — every rule uses !important to defeat dark-theme overrides ──
GLOBAL_CSS = f"""
<style>
/* ─── Force light background everywhere ─── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .stApp {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
}}

/* ─── Main content block ─── */
.block-container {{
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1120px !important;
    background-color: {C['bg']} !important;
}}

/* ─── Sidebar — force light, force dark text ─── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div {{
    background-color: {C['sidebar']} !important;
    color: {C['sidebar_text']} !important;
}}
[data-testid="stSidebar"] * {{
    color: {C['sidebar_text']} !important;
    border-color: {C['border']} !important;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown {{
    color: {C['sidebar_text']} !important;
}}
/* Sidebar border */
[data-testid="stSidebar"] {{
    border-right: 1px solid {C['border']} !important;
}}

/* ─── All text globally ─── */
p, span, div, label, li, td, th, a,
.stMarkdown, .stText {{
    color: {C['text']} !important;
}}

/* ─── Headings ─── */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Georgia', serif !important;
    font-weight: 400 !important;
    color: {C['text']} !important;
    letter-spacing: -0.02em !important;
}}
h2 {{ font-size: 1.5rem !important; }}
h3 {{ font-size: 1.1rem !important; }}

/* ─── Metric cards — the key fix for invisible numbers ─── */
[data-testid="metric-container"] {{
    background-color: {C['card']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 6px !important;
    padding: 1.25rem 1.5rem !important;
}}
[data-testid="metric-container"] > div {{
    background-color: {C['card']} !important;
}}
/* Metric label */
[data-testid="metric-container"] label,
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {{
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: {C['muted']} !important;
    font-weight: 600 !important;
}}
/* Metric value — THIS is what was invisible */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] div {{
    font-family: 'Georgia', serif !important;
    font-size: 1.85rem !important;
    color: {C['text']} !important;
    font-weight: 400 !important;
}}
/* Metric delta */
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] > div,
[data-testid="stMetricDelta"] svg {{
    font-size: 0.8rem !important;
}}

/* ─── Input widgets — force light backgrounds ─── */
.stSelectbox > div > div,
.stTextInput > div > div,
.stNumberInput > div > div,
[data-baseweb="select"],
[data-baseweb="input"] {{
    background-color: {C['card']} !important;
    color: {C['text']} !important;
    border-color: {C['border']} !important;
    border-radius: 4px !important;
}}
[data-baseweb="select"] * ,
[data-baseweb="input"] * {{
    color: {C['text']} !important;
    background-color: {C['card']} !important;
}}

/* ─── File uploader ─── */
[data-testid="stFileUploadDropzone"] {{
    background-color: {C['card']} !important;
    border: 1px dashed {C['border']} !important;
    border-radius: 6px !important;
    color: {C['text']} !important;
}}
[data-testid="stFileUploadDropzone"] * {{
    color: {C['text']} !important;
}}

/* ─── Buttons ─── */
.stButton > button {{
    background-color: {C['accent']} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.55rem 1.4rem !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    transition: opacity .15s ease !important;
}}
.stButton > button:hover {{
    opacity: 0.82 !important;
    color: #FFFFFF !important;
}}
/* Download button */
.stDownloadButton > button {{
    background-color: {C['accent']} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
}}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0 !important;
    border-bottom: 1px solid {C['border']} !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 0.875rem !important;
    color: {C['muted']} !important;
    padding: 0.65rem 1.3rem !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    background: transparent !important;
    font-weight: 400 !important;
}}
.stTabs [aria-selected="true"] {{
    color: {C['text']} !important;
    border-bottom: 2px solid {C['accent']} !important;
    font-weight: 600 !important;
    background: transparent !important;
}}
/* Tab panel background */
.stTabs [data-baseweb="tab-panel"] {{
    background-color: {C['bg']} !important;
    padding-top: 1rem !important;
}}

/* ─── Dataframe / table ─── */
[data-testid="stDataFrame"],
.stDataFrame {{
    border: 1px solid {C['border']} !important;
    border-radius: 6px !important;
    background-color: {C['card']} !important;
}}
[data-testid="stDataFrame"] * {{
    color: {C['text']} !important;
    background-color: {C['card']} !important;
}}

/* ─── Slider ─── */
[data-testid="stSlider"] > div > div > div > div {{
    background-color: {C['accent']} !important;
}}
[data-testid="stSlider"] [role="slider"] {{
    background-color: {C['accent']} !important;
    border-color: {C['accent']} !important;
}}

/* ─── Multiselect tags ─── */
[data-baseweb="tag"] {{
    background-color: {C['accent_soft']} !important;
    color: {C['accent']} !important;
}}

/* ─── Expander ─── */
[data-testid="stExpander"] {{
    border: 1px solid {C['border']} !important;
    border-radius: 6px !important;
    background-color: {C['card']} !important;
}}
[data-testid="stExpander"] summary {{
    color: {C['text']} !important;
}}

/* ─── Alerts / notifications ─── */
[data-testid="stAlert"] {{
    border-radius: 6px !important;
    font-size: 0.875rem !important;
}}

/* ─── Success/warning/error/info boxes ─── */
.stSuccess {{
    background-color: #EAF2EA !important;
    color: {C['success']} !important;
    border: 1px solid #B8D9BA !important;
}}
.stWarning {{
    background-color: #FAF3E0 !important;
    color: {C['warning']} !important;
}}
.stError {{
    background-color: #FAE8E8 !important;
    color: {C['danger']} !important;
}}

/* ─── Radio buttons ─── */
[data-testid="stRadio"] label {{
    color: {C['text']} !important;
}}

/* ─── Dividers ─── */
hr {{
    border: none !important;
    border-top: 1px solid {C['divider']} !important;
    margin: 1.25rem 0 !important;
}}

/* ─── Hide Streamlit chrome ─── */
#MainMenu, footer, [data-testid="stToolbar"] {{
    visibility: hidden !important;
    height: 0 !important;
}}

/* ─── Scrollbar ─── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 3px; }}
</style>
"""


def inject_css():
    """Inject the full CSS override into the page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_title(title: str, subtitle: str = ""):
    """Consistent page heading with subtitle."""
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(
            f"<p style='color:{C['muted']};font-size:0.875rem;"
            f"margin-top:-0.5rem;line-height:1.4;'>{subtitle}</p>",
            unsafe_allow_html=True,
        )
    st.markdown("---")


def card_html(content: str):
    """Bordered white card container."""
    st.markdown(
        f"<div style='background:{C['card']};border:1px solid {C['border']};"
        f"border-radius:6px;padding:1.25rem 1.5rem;margin-bottom:0.75rem;'>"
        f"{content}</div>",
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "info"):
    """
    Left-bordered callout block.
    kind: 'info' | 'success' | 'warning' | 'danger'
    """
    cfg = {
        "info":    (C["accent_soft"], C["accent"],  C["text"]),
        "success": ("#E6F2E6",        C["success"], C["success"]),
        "warning": ("#FAF3E0",        C["warning"], C["warning"]),
        "danger":  ("#FAE8E8",        C["danger"],  C["danger"]),
    }
    bg, border, text_col = cfg.get(kind, cfg["info"])
    st.markdown(
        f"<div style='background:{bg};border-left:3px solid {border};"
        f"border-radius:0 4px 4px 0;padding:0.9rem 1.2rem;"
        f"margin-bottom:0.75rem;font-size:0.875rem;"
        f"color:{C['text']};line-height:1.6;'>{text}</div>",
        unsafe_allow_html=True,
    )


def section_tag(text: str):
    """Small all-caps label above a group of content."""
    st.markdown(
        f"<p style='font-size:0.71rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:{C['muted']};margin-bottom:0.35rem;"
        f"font-weight:600;'>{text}</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════

def fmt_currency(v, symbol="$") -> str:
    """Format a number as a currency string."""
    if v is None:
        return f"{symbol}0.00"
    prefix = "-" if v < 0 else ""
    return f"{prefix}{symbol}{abs(v):,.2f}"


def fmt_pct(v, dp=1) -> str:
    """Format a number as a percentage string."""
    if v is None:
        return "0.0%"
    return f"{v:.{dp}f}%"


def profit_margin(selling: float, cost: float) -> float:
    """Gross margin %: (selling - cost) / selling * 100"""
    if not selling:
        return 0.0
    return ((selling - cost) / selling) * 100


def suggested_price(cost: float, target_pct: float) -> float:
    """Price needed to hit a target margin: cost / (1 - margin/100)"""
    target_pct = min(max(target_pct, 0), 99)
    return round(cost / (1 - target_pct / 100), 2)


def delta_str(current: float, previous: float, currency=False) -> str:
    """Delta string for st.metric."""
    diff = current - previous
    if currency:
        sign = "+" if diff >= 0 else ""
        return f"{sign}{fmt_currency(diff)}"
    pct  = ((current - previous) / previous * 100) if previous else 0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


# ═══════════════════════════════════════════════════════════════════════════
# 3. PLOTLY CHARTS
# ═══════════════════════════════════════════════════════════════════════════

_PALETTE = ["#2C5F2E", "#5A8A5D", "#8AB58C", "#C0D9C1", "#3D6B3F", "#1A3D1C"]


def _theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply shared chart theme."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Georgia, serif", size=14, color=C["text"]),
            x=0, xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Helvetica Neue, Arial, sans-serif",
                  color=C["text"], size=12),
        margin=dict(l=20, r=20, t=42, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)",
                    bordercolor=C["border"], borderwidth=1,
                    font=dict(color=C["text"])),
    )
    fig.update_xaxes(showgrid=False, color=C["muted"],
                     linecolor=C["border"], tickfont=dict(color=C["muted"]))
    fig.update_yaxes(showgrid=True, gridcolor=C["border"],
                     color=C["muted"], linecolor="rgba(0,0,0,0)",
                     tickfont=dict(color=C["muted"]))
    return fig


def chart_revenue_line(df: pd.DataFrame) -> go.Figure:
    """Line chart of daily revenue. Expects: sold_date, total_revenue"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["sold_date"], y=df["total_revenue"],
        mode="lines",
        line=dict(color=C["accent"], width=2),
        fill="tozeroy", fillcolor="rgba(44,95,46,0.07)",
        hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    return _theme(fig, "Daily Revenue")


def chart_top_bar(df: pd.DataFrame, val="total_revenue") -> go.Figure:
    """Horizontal bar — top items. Expects: item_name, [val]"""
    df = df.sort_values(val, ascending=True).tail(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[val], y=df["item_name"], orientation="h",
        marker=dict(color=C["accent"], line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>$%{x:,.2f}<extra></extra>",
    ))
    return _theme(fig, "Top Items by Revenue")


def chart_peak_hours(df: pd.DataFrame) -> go.Figure:
    """Bar chart by hour. Expects: hour, total_revenue"""
    labels = [f"{int(h):02d}:00" for h in df["hour"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=df["total_revenue"],
        marker=dict(color=C["accent"], line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    return _theme(fig, "Revenue by Hour of Day")


def chart_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart by category. Expects: category, total_revenue"""
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=df["category"], values=df["total_revenue"],
        hole=0.55, marker=dict(colors=_PALETTE),
        textinfo="label+percent",
        textfont=dict(color=C["text"]),
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<extra></extra>",
    ))
    return _theme(fig, "Revenue by Category")


def chart_cost_vs_price(df: pd.DataFrame) -> go.Figure:
    """Grouped bars — cost vs selling price. Expects: item_name, cost_price, selling_price"""
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Cost Price", x=df["item_name"], y=df["cost_price"],
                         marker_color="#C0D9C1",
                         hovertemplate="Cost: $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Selling Price", x=df["item_name"], y=df["selling_price"],
                         marker_color=C["accent"],
                         hovertemplate="Price: $%{y:.2f}<extra></extra>"))
    fig.update_layout(barmode="group")
    return _theme(fig, "Cost vs Selling Price")


# ═══════════════════════════════════════════════════════════════════════════
# 4. DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════

ITEM_ALIASES = {
    "flat wht":"flat white","flatwhite":"flat white",
    "long blk":"long black","long blck":"long black",
    "cap":"cappuccino","capp":"cappuccino",
    "lte":"latte","oat lte":"oat milk latte","oat latte":"oat milk latte",
    "cold brew coffee":"cold brew","cb":"cold brew",
    "iced lat":"iced latte","matcha":"matcha latte",
    "avo toast":"avocado toast","avo on toast":"avocado toast",
    "eggs benny":"eggs benedict","eggs bene":"eggs benedict",
    "gran bowl":"granola bowl","bb":"banana bread",
    "chk sndwch":"chicken sandwich",
}

COLUMN_ALIASES = {
    "product":"item_name","product_name":"item_name","menu_item":"item_name","name":"item_name",
    "quantity":"qty","units_sold":"qty",
    "amount":"revenue","total":"revenue","sale_amount":"revenue",
    "date":"sold_at","sale_date":"sold_at","transaction_date":"sold_at",
    "item_category":"category","type":"category",
}

CATEGORY_KW = {
    "Coffee":      ["latte","flat white","cappuccino","long black","espresso",
                    "cold brew","matcha","chai","macchiato","mocha","affogato"],
    "Cold Drinks": ["iced","smoothie","frappe","milkshake","slushie"],
    "Food":        ["toast","eggs","sandwich","salad","bowl","muffin","croissant",
                    "cake","scone","waffle","pancake","wrap","burger","bread"],
    "Drinks":      ["juice","water","soda","tea","kombucha"],
}


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Full 8-step cleaning pipeline for raw POS/CSV data."""
    df = df.copy()
    # 1. Normalise column names
    df.columns = [re.sub(r"_+","_",re.sub(r"[^\w]","",re.sub(r"[\s\-]+","_",c.strip().lower()))).strip("_")
                  for c in df.columns]
    df = df.rename(columns={k:v for k,v in COLUMN_ALIASES.items() if k in df.columns})
    # 2. Remove exact duplicates
    df = df.drop_duplicates()
    # 3. Normalise item names
    if "item_name" in df.columns:
        def _clean_name(n):
            if pd.isna(n): return ""
            n = re.sub(r"\s+"," ",str(n).strip().lower()).strip(".,;:")
            return ITEM_ALIASES.get(n, n)
        df["item_name"] = df["item_name"].apply(_clean_name)
    # 4. Parse dates
    if "sold_at" in df.columns:
        df["sold_at"] = pd.to_datetime(df["sold_at"], errors="coerce")
    # 5. Clean numerics
    if "revenue" in df.columns:
        df["revenue"] = pd.to_numeric(
            df["revenue"].astype(str).str.replace(r"[$£€RM,\s]","",regex=True)
                         .str.replace(r"[^\d.\-]","",regex=True), errors="coerce")
    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(
            df["qty"].astype(str).str.replace(r"[^\d]","",regex=True), errors="coerce"
        ).fillna(1).astype(int)
    # 6. Fill missing
    if "category" not in df.columns:
        df["category"] = "Uncategorized"
    else:
        df["category"] = df["category"].fillna("Uncategorized")
    if "qty" in df.columns:
        df["qty"] = df["qty"].fillna(1).astype(int)
    # 7. Assign categories from keywords
    if "item_name" in df.columns:
        def _infer_cat(row):
            if row.get("category","Uncategorized") != "Uncategorized":
                return row["category"]
            nm = str(row["item_name"]).lower()
            for cat, kws in CATEGORY_KW.items():
                if any(k in nm for k in kws): return cat
            return "Uncategorized"
        df["category"] = df.apply(_infer_cat, axis=1)
    # 8. Drop unresolvable rows
    if "item_name" in df.columns:
        df = df[df["item_name"].notna() & (df["item_name"].str.strip() != "")]
    if "revenue" in df.columns:
        df = df[pd.to_numeric(df["revenue"], errors="coerce").fillna(0) > 0]
    if "sold_at" in df.columns:
        df = df[df["sold_at"].notna()]
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 5. DEMO DATA
# ═══════════════════════════════════════════════════════════════════════════

def demo_data() -> pd.DataFrame:
    """Generate 90 days of synthetic cafe transaction data."""
    rng   = np.random.default_rng(42)
    start = datetime.now() - timedelta(days=90)
    menu  = [
        ("flat white",      "Coffee",      4.50, 2.20),
        ("long black",      "Coffee",      4.00, 1.60),
        ("cappuccino",      "Coffee",      4.80, 2.10),
        ("latte",           "Coffee",      5.00, 2.30),
        ("oat milk latte",  "Coffee",      5.80, 2.90),
        ("matcha latte",    "Coffee",      6.00, 2.80),
        ("cold brew",       "Cold Drinks", 5.50, 1.80),
        ("iced latte",      "Cold Drinks", 5.50, 2.50),
        ("banana smoothie", "Cold Drinks", 7.00, 3.20),
        ("avocado toast",   "Food",       14.00, 5.50),
        ("eggs benedict",   "Food",       18.00, 7.00),
        ("granola bowl",    "Food",       12.00, 4.20),
        ("banana bread",    "Food",        6.50, 2.00),
        ("croissant",       "Food",        5.00, 1.80),
        ("muffin",          "Food",        4.50, 1.50),
        ("chicken sandwich","Food",       16.00, 6.50),
        ("caesar salad",    "Food",       15.00, 5.80),
        ("sparkling water", "Drinks",      4.00, 0.80),
        ("orange juice",    "Drinks",      6.00, 1.50),
        ("chai latte",      "Coffee",      5.50, 2.40),
    ]
    wts = [12,10,12,14,8,6,7,8,3,9,5,4,8,10,8,6,5,5,4,6]
    wp  = [w/sum(wts) for w in wts]
    rows = []
    for d in range(90):
        day = start + timedelta(days=d)
        for _ in range(int(rng.integers(55, 130))):
            i = rng.choice(len(menu), p=wp)
            nm, cat, price, cost = menu[i]
            qty = int(rng.choice([1,1,1,2], p=[0.70,0.15,0.10,0.05]))
            hr  = rng.choice(list(range(7,19)),
                             p=[0.15,0.18,0.15,0.08,0.06,0.10,0.08,0.06,0.05,0.04,0.03,0.02])
            dt  = day.replace(hour=int(hr), minute=int(rng.integers(0,59)),
                              second=0, microsecond=0)
            rows.append({"item_name":nm,"category":cat,"qty":qty,
                         "revenue":round(price*qty,2),"cost":round(cost*qty,2),
                         "sold_at":dt})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 6. SUPABASE (optional — graceful fallback if not configured)
# ═══════════════════════════════════════════════════════════════════════════

def _supabase():
    """Return Supabase client or None."""
    try:
        from supabase import create_client
        url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL",""))
        key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY",""))
        if url and key and url.startswith("https://"):
            return create_client(url, key)
    except Exception:
        pass
    return None


def db_save_txn(records: list) -> None:
    c = _supabase()
    if not c: return
    try:
        for i in range(0, len(records), 500):
            c.table("transactions").insert(records[i:i+500]).execute()
    except Exception:
        pass


def db_load_txn() -> pd.DataFrame:
    c = _supabase()
    if not c: return pd.DataFrame()
    try:
        r  = c.table("transactions").select("*").order("sold_at", desc=True).execute()
        df = pd.DataFrame(r.data or [])
        if not df.empty and "sold_at" in df.columns:
            df["sold_at"] = pd.to_datetime(df["sold_at"])
        return df
    except Exception:
        return pd.DataFrame()


def db_save_costs(costs: list) -> None:
    c = _supabase()
    if not c: return
    try:
        c.table("menu_costs").upsert(costs, on_conflict="item_name").execute()
    except Exception:
        pass


def db_load_costs() -> pd.DataFrame:
    c = _supabase()
    if not c: return pd.DataFrame()
    try:
        r = c.table("menu_costs").select("*").execute()
        return pd.DataFrame(r.data or [])
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# 7. ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

def flt(df: pd.DataFrame, days) -> pd.DataFrame:
    """Filter to last N days. None = all time."""
    if df.empty or days is None or "sold_at" not in df.columns: return df
    df = df.copy()
    df["sold_at"] = pd.to_datetime(df["sold_at"])
    return df[df["sold_at"] >= df["sold_at"].max() - pd.Timedelta(days=days)]


def total_revenue(df): return float(df["revenue"].sum()) if not df.empty and "revenue" in df.columns else 0.0


def total_profit(txn: pd.DataFrame, costs: dict) -> float:
    """Gross profit using costs dict {item_name: cost_price}."""
    if txn.empty: return 0.0
    if "cost" in txn.columns:
        return float((txn["revenue"] - txn["cost"]).sum())
    if not costs: return 0.0
    df = txn.copy()
    df["cp"] = df["item_name"].map(costs).fillna(0)
    df = df[df["cp"] > 0]
    return float((df["revenue"] - df["qty"] * df["cp"]).sum()) if not df.empty else 0.0


def food_cost_pct(txn: pd.DataFrame, costs: dict) -> float:
    """Food cost % = total cost / total revenue × 100."""
    if txn.empty: return 0.0
    if "cost" in txn.columns:
        rev = float(txn["revenue"].sum())
        return float(txn["cost"].sum()) / rev * 100 if rev else 0.0
    if not costs: return 0.0
    df = txn.copy()
    df["cp"] = df["item_name"].map(costs).fillna(0)
    tc = float((df["qty"] * df["cp"]).sum())
    tr = float(df["revenue"].sum())
    return tc / tr * 100 if tr else 0.0


def avg_order(df): return float(df["revenue"].mean()) if not df.empty and "revenue" in df.columns else 0.0


def period_delta(df: pd.DataFrame, days: int) -> dict:
    """Current vs previous period comparison."""
    empty = {"cr":0,"pr":0,"pct":0}
    if df.empty or "sold_at" not in df.columns: return empty
    df = df.copy(); df["sold_at"] = pd.to_datetime(df["sold_at"])
    now = df["sold_at"].max()
    cur = df[df["sold_at"] >= now - timedelta(days=days)]
    prv = df[(df["sold_at"] >= now - timedelta(days=days*2)) &
             (df["sold_at"] <  now - timedelta(days=days))]
    cr, pr = float(cur["revenue"].sum()), float(prv["revenue"].sum())
    return {"cr":cr,"pr":pr,"pct":round((cr-pr)/pr*100,1) if pr else 0}


def top_items(df, n=10):
    if df.empty: return pd.DataFrame()
    return (df.groupby("item_name")
              .agg(total_revenue=("revenue","sum"),total_qty=("qty","sum"))
              .reset_index().sort_values("total_revenue",ascending=False).head(n))


def worst_items(df, n=8):
    if df.empty: return pd.DataFrame()
    return (df.groupby("item_name")
              .agg(total_revenue=("revenue","sum"),total_qty=("qty","sum"))
              .reset_index().sort_values("total_revenue",ascending=True).head(n))


def peak_hours(df):
    if df.empty or "sold_at" not in df.columns: return pd.DataFrame()
    d = df.copy(); d["sold_at"] = pd.to_datetime(d["sold_at"]); d["hour"] = d["sold_at"].dt.hour
    return d.groupby("hour").agg(total_revenue=("revenue","sum")).reset_index().sort_values("hour")


def daily_revenue(df):
    if df.empty: return pd.DataFrame()
    d = df.copy(); d["sold_at"] = pd.to_datetime(d["sold_at"]); d["sold_date"] = d["sold_at"].dt.date
    g = d.groupby("sold_date")["revenue"].sum().reset_index()
    g.columns = ["sold_date","total_revenue"]
    return g.sort_values("sold_date")


def cat_revenue(df):
    if df.empty or "category" not in df.columns: return pd.DataFrame()
    g = df.groupby("category")["revenue"].sum().reset_index()
    g.columns = ["category","total_revenue"]
    return g.sort_values("total_revenue",ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# 8. PRICE OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

MIN_MARGIN = 55.0


def margin_table(txn: pd.DataFrame, costs: dict) -> pd.DataFrame:
    """Build per-item margin analysis from transactions + cost dict."""
    if txn.empty or not costs: return pd.DataFrame()
    df = txn.copy()
    df["unit_price"] = df.apply(lambda r: r["revenue"]/r["qty"] if r.get("qty",1)>0 else r["revenue"], axis=1)
    s = df.groupby("item_name").agg(
        selling_price=("unit_price","mean"),
        quantity_sold=("qty","sum"),
        category=("category", lambda x: x.mode()[0] if len(x)>0 else "Other"),
    ).reset_index()
    s["cost_price"] = s["item_name"].map(costs).fillna(0)
    s = s[s["cost_price"] > 0]
    if s.empty: return pd.DataFrame()
    s["margin_pct"] = s.apply(lambda r: profit_margin(r["selling_price"], r["cost_price"]), axis=1)
    s["tier"] = s["margin_pct"].apply(lambda m: "Low" if m<55 else "Fair" if m<65 else "Good" if m<75 else "Strong")
    return s.sort_values("margin_pct", ascending=True)


def price_recs(mdf: pd.DataFrame, bump: float) -> pd.DataFrame:
    """Compute suggested prices for items below target margin."""
    if mdf.empty: return pd.DataFrame()
    rows = []
    for _, r in mdf.iterrows():
        target = min(r["margin_pct"] + bump, 85.0)
        ideal  = suggested_price(r["cost_price"], target)
        cap    = round(r["selling_price"] * 1.05, 2)
        sugg   = max(min(ideal, cap), r["selling_price"])
        new_m  = profit_margin(sugg, r["cost_price"])
        inc    = sugg - r["selling_price"]
        gain   = (sugg - r["cost_price"] - (r["selling_price"] - r["cost_price"])) * r["quantity_sold"]
        rows.append({**r.to_dict(),
                     "suggested_price": round(sugg,2),
                     "new_margin_pct":  round(new_m,1),
                     "price_increase":  round(inc,2),
                     "increase_pct":    round((inc/r["selling_price"]*100) if r["selling_price"] else 0,1),
                     "est_profit_gain": round(gain,2)})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 9. RECOMMENDATIONS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def recommendations(txn: pd.DataFrame, mdf: pd.DataFrame) -> list:
    """Generate all recommendations sorted by priority."""
    recs = []

    # Pricing
    if not mdf.empty and "margin_pct" in mdf.columns:
        for _, r in mdf[mdf["margin_pct"] < 60].head(4).iterrows():
            nm = r["item_name"].title()
            recs.append({"priority":1,"type":"pricing",
                "title": f"Increase the price of {nm}",
                "detail": f"{nm} runs at {r['margin_pct']:.1f}% margin — below the 60% target. "
                          f"A 5% increase from {fmt_currency(r['selling_price'])} to "
                          f"{fmt_currency(round(r['selling_price']*1.05,2))} improves margins without "
                          f"meaningful customer resistance.",
                "impact": f"Estimated +5–7% margin improvement on {nm}"})

    # Item removal
    if not txn.empty and not mdf.empty and "sold_at" in txn.columns:
        dates = pd.to_datetime(txn["sold_at"])
        weeks = max((dates.max()-dates.min()).days/7, 1)
        wq = (txn.groupby("item_name")["qty"].sum()/weeks).reset_index()
        wq.columns = ["item_name","wqty"]
        low_m = set(mdf[mdf["margin_pct"] < 50]["item_name"]) if "margin_pct" in mdf.columns else set()
        for _, r in wq[(wq["wqty"]<5) & (wq["item_name"].isin(low_m))].head(3).iterrows():
            nm = r["item_name"].title()
            recs.append({"priority":2,"type":"menu",
                "title": f"Consider removing {nm} from the menu",
                "detail": f"{nm} averages {r['wqty']:.1f} units/week with a below-target margin. "
                          f"Removing it reduces kitchen complexity and lets staff focus on "
                          f"faster, more profitable items.",
                "impact": "Reduces waste and kitchen complexity"})

    # Combo offer
    if not txn.empty and "category" in txn.columns:
        bc = txn.groupby(["item_name","category"])["revenue"].sum().reset_index()
        foods   = bc[bc["category"]=="Food"].sort_values("revenue",ascending=False)
        coffees = bc[bc["category"].isin(["Coffee","Cold Drinks"])].sort_values("revenue",ascending=False)
        if not foods.empty and not coffees.empty:
            f, c = foods.iloc[0]["item_name"].title(), coffees.iloc[0]["item_name"].title()
            recs.append({"priority":3,"type":"marketing",
                "title": f"Bundle {c} + {f} as a meal deal",
                "detail": f"{c} and {f} are your top revenue earners in their categories. "
                          f"A named combo at a 5–8% bundle discount encourages upselling and "
                          f"increases average transaction value during the morning rush.",
                "impact": "Estimated 8–12% increase in average basket size"})

    # Promote high-margin item
    if not mdf.empty and "margin_pct" in mdf.columns:
        hi = mdf[mdf["margin_pct"] > 70].sort_values("margin_pct",ascending=False)
        if not hi.empty:
            nm = hi.iloc[0]["item_name"].title()
            mg = hi.iloc[0]["margin_pct"]
            recs.append({"priority":2,"type":"marketing",
                "title": f"Feature {nm} as your daily special",
                "detail": f"{nm} carries a {mg:.1f}% gross margin — among your highest. "
                          f"Positioning it on the menu board and training staff to recommend it "
                          f"grows volume without discounting.",
                "impact": f"20% volume increase on {nm} adds directly to net profit"})

    # Peak hour staffing
    if not txn.empty and "sold_at" in txn.columns:
        d = txn.copy(); d["sold_at"] = pd.to_datetime(d["sold_at"]); d["hr"] = d["sold_at"].dt.hour
        hr = d.groupby("hr")["revenue"].sum()
        if not hr.empty:
            pk, sl = int(hr.idxmax()), int(hr.idxmin())
            recs.append({"priority":3,"type":"operations",
                "title": f"Staff up between {pk:02d}:00 and {pk+1:02d}:00",
                "detail": f"Your highest-revenue hour is {pk:02d}:00–{pk+1:02d}:00. "
                          f"Adequate staffing here reduces wait times and prevents lost sales.",
                "impact": "Prevents revenue loss during peak window"})
            recs.append({"priority":3,"type":"operations",
                "title": f"Run a promotion at {sl:02d}:00 to fill slow hours",
                "detail": f"Revenue drops at {sl:02d}:00. A time-limited coffee + snack deal "
                          f"can convert this quiet period into incremental revenue.",
                "impact": "Converts low-traffic hours into added revenue"})

    # Inventory
    if not txn.empty:
        top = txn.groupby("item_name")["qty"].sum().idxmax()
        recs.append({"priority":3,"type":"operations",
            "title": f"Review supply agreements for {top.title()}",
            "detail": f"{top.title()} is your highest-volume item. A volume discount with your "
                      f"supplier or a 3-day buffer stock prevents revenue-cutting stock-outs.",
            "impact": "Eliminates stock-out risk on top seller"})

    recs.sort(key=lambda r: (r["priority"], r["type"]))
    return recs


# ═══════════════════════════════════════════════════════════════════════════
# 10. REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_report(txn: pd.DataFrame, costs: dict, days: int) -> dict:
    """Compile report package for N rolling days."""
    df = flt(txn, days)
    return {
        "period": f"{(datetime.now()-timedelta(days=days)).strftime('%d %b %Y')} – {datetime.now().strftime('%d %b %Y')}",
        "summary": {"revenue": total_revenue(df), "profit": total_profit(df, costs),
                    "food_cost_pct": food_cost_pct(df, costs),
                    "orders": len(df), "avg_order": avg_order(df)},
        "daily_df":  daily_revenue(df),
        "top_df":    top_items(df, 10),
        "worst_df":  worst_items(df, 5),
        "hours_df":  peak_hours(df),
    }


def export_csv(report: dict) -> bytes:
    """Flatten report to downloadable CSV bytes."""
    buf = io.StringIO()
    buf.write(f"Seralung Opti Report\nPeriod: {report['period']}\n"
              f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}\n\n")
    s = report["summary"]
    buf.write(f"--- Summary ---\nRevenue,{s['revenue']:.2f}\nProfit,{s['profit']:.2f}\n"
              f"Food Cost %,{s['food_cost_pct']:.1f}\nTransactions,{s['orders']}\n"
              f"Avg Order,{s['avg_order']:.2f}\n\n")
    for lbl, key in [("Top Items","top_df"),("Lowest Items","worst_df"),
                     ("Daily Revenue","daily_df"),("Peak Hours","hours_df")]:
        df = report.get(key, pd.DataFrame())
        if not df.empty:
            buf.write(f"--- {lbl} ---\n"); df.to_csv(buf, index=False); buf.write("\n")
    return buf.getvalue().encode("utf-8")


def send_email_report(to: str, period: str, csv_bytes: bytes) -> tuple:
    """Send report CSV via SMTP."""
    try:
        host = st.secrets.get("SMTP_HOST",  os.getenv("SMTP_HOST",  "smtp.gmail.com"))
        port = int(st.secrets.get("SMTP_PORT", os.getenv("SMTP_PORT", "587")))
        user = st.secrets.get("SMTP_USER",  os.getenv("SMTP_USER",  ""))
        pwd  = st.secrets.get("SMTP_PASSWORD", os.getenv("SMTP_PASSWORD",""))
        frm  = st.secrets.get("REPORT_FROM_EMAIL", user)
    except Exception:
        return False, "Email credentials not configured."
    if not user or not pwd:
        return False, "Add SMTP_USER and SMTP_PASSWORD to Streamlit secrets."
    msg = MIMEMultipart()
    msg["From"] = frm; msg["To"] = to
    msg["Subject"] = f"Seralung Opti — Report for {period}"
    msg.attach(MIMEText(f"Report attached for {period}.\n\n— Seralung Opti", "plain"))
    att = MIMEBase("application","octet-stream"); att.set_payload(csv_bytes)
    encoders.encode_base64(att)
    att.add_header("Content-Disposition","attachment; filename=seralung_report.csv")
    msg.attach(att)
    try:
        with smtplib.SMTP(host, port) as srv:
            srv.ehlo(); srv.starttls(); srv.login(user, pwd)
            srv.sendmail(frm, to, msg.as_string())
        return True, f"Report sent to {to}."
    except Exception as e:
        return False, f"Email failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# 11. SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init_state():
    """Initialise all session state keys."""
    for k, v in [("txn", pd.DataFrame()), ("costs", {}),
                 ("report", None), ("report_csv", b"")]:
        if k not in st.session_state:
            st.session_state[k] = v


def get_txn() -> pd.DataFrame:
    return st.session_state.get("txn", pd.DataFrame())


def set_txn(df: pd.DataFrame):
    st.session_state["txn"] = df


def get_costs() -> dict:
    return st.session_state.get("costs", {})


def set_costs(d: dict):
    st.session_state["costs"] = d


# ═══════════════════════════════════════════════════════════════════════════
# 12. SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """Render sidebar controls and return settings dict."""
    with st.sidebar:
        st.markdown(
            f"<div style='padding:1rem 0 1.25rem;'>"
            f"<p style='font-family:Georgia,serif;font-size:1.2rem;"
            f"color:{C['text']};margin:0;letter-spacing:-0.02em;'>Seralung Opti</p>"
            f"<p style='font-size:0.72rem;color:{C['muted']};margin-top:0.1rem;'>"
            f"Cafe Business Intelligence</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        section_tag("IMPORT DATA")
        uploaded = st.file_uploader("Upload POS / CSV export", type=["csv"],
                                    help="Columns: item_name, qty, revenue, sold_at")
        if uploaded:
            try:
                raw     = pd.read_csv(uploaded)
                cleaned = clean_df(raw)
                if cleaned.empty:
                    st.error("No valid rows after cleaning. Check your CSV columns.")
                else:
                    set_txn(cleaned)
                    db_save_txn(cleaned.to_dict(orient="records"))
                    st.success(f"Imported {len(cleaned):,} rows.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

        if st.button("Load demo data", use_container_width=True):
            df = demo_data()
            set_txn(df)
            st.success(f"Demo loaded — {len(df):,} transactions.")

        st.markdown("---")
        section_tag("DATE RANGE")
        po = st.selectbox("Period",
                          ["Last 7 days","Last 30 days","Last 90 days","All time"],
                          index=2, label_visibility="collapsed")
        days = {"Last 7 days":7,"Last 30 days":30,"Last 90 days":90,"All time":None}[po]

        st.markdown("---")
        section_tag("PRICE OPTIMIZER")
        bump = st.slider("Target margin increase (%)", 2, 15, 6, 1)

        st.markdown("---")
        st.markdown(
            f"<p style='font-size:0.7rem;color:{C['muted']};'>v1.0.0 · Seralung Opti</p>",
            unsafe_allow_html=True,
        )

    return {"days": days, "label": po, "bump": bump}


# ═══════════════════════════════════════════════════════════════════════════
# 13. PAGE — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def pg_overview(txn: pd.DataFrame, costs: dict, settings: dict):
    page_title("Overview", f"Performance summary · {settings['label']}")

    if txn.empty:
        callout("No data loaded yet. Click <strong>Load demo data</strong> in the sidebar "
                "or upload a CSV file to begin.", kind="info")
        return

    df = flt(txn, settings["days"])
    if df.empty:
        callout("No data found for the selected period.", kind="warning")
        return

    # ── KPIs ──
    section_tag("KEY METRICS")
    rev  = total_revenue(df)
    pft  = total_profit(df, costs)
    fcp  = food_cost_pct(df, costs)
    aov  = avg_order(df)
    d    = period_delta(df, days=min(settings["days"] or 30, 30))

    k1, k2, k3, k4 = st.columns(4, gap="small")
    with k1:
        st.metric("Total Revenue", fmt_currency(rev),
                  delta=delta_str(d["cr"], d["pr"], currency=True) if d["pr"] > 0 else None)
    with k2:
        st.metric("Gross Profit", fmt_currency(pft) if pft else "Add cost data")
    with k3:
        st.metric("Food Cost %", fmt_pct(fcp) if fcp else "Add cost data")
    with k4:
        st.metric("Avg Order Value", fmt_currency(aov))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Revenue trend ──
    section_tag("REVENUE TREND")
    dr = daily_revenue(df)
    if not dr.empty:
        st.plotly_chart(chart_revenue_line(dr), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top + worst ──
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section_tag("TOP SELLING ITEMS")
        ti = top_items(df)
        if not ti.empty:
            st.plotly_chart(chart_top_bar(ti), use_container_width=True)

    with c2:
        section_tag("LOWEST PERFORMING ITEMS")
        wi = worst_items(df)
        if not wi.empty:
            d2 = wi.copy()
            d2.columns = ["Item","Revenue","Units Sold"]
            d2["Revenue"] = d2["Revenue"].apply(fmt_currency)
            st.dataframe(d2, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Peak hours + category ──
    c3, c4 = st.columns(2, gap="large")
    with c3:
        section_tag("PEAK TRADING HOURS")
        ph = peak_hours(df)
        if not ph.empty:
            st.plotly_chart(chart_peak_hours(ph), use_container_width=True)
    with c4:
        section_tag("REVENUE BY CATEGORY")
        cr = cat_revenue(df)
        if not cr.empty:
            st.plotly_chart(chart_donut(cr), use_container_width=True)

    with st.expander("View raw transaction data"):
        d3 = df.copy()
        if "revenue" in d3.columns:
            d3["revenue"] = d3["revenue"].apply(fmt_currency)
        cols = [c for c in ["item_name","category","qty","revenue","sold_at"] if c in d3.columns]
        st.dataframe(d3[cols].head(500), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# 14. PAGE — PRICE CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

def pg_price_calc(txn: pd.DataFrame, settings: dict):
    page_title("Price Calculator", "Margin analysis and optimised price recommendations.")

    if txn.empty:
        callout("No data loaded. Use the sidebar to upload or load demo data.", kind="info")
        return

    # Build item catalog from transactions
    df = txn.copy()
    df["unit_price"] = df.apply(lambda r: r["revenue"]/r["qty"] if r.get("qty",1)>0 else r["revenue"], axis=1)
    catalog = df.groupby("item_name").agg(
        selling_price=("unit_price","mean"),
        quantity_sold=("qty","sum"),
        category=("category", lambda x: x.mode()[0] if len(x)>0 else "Other"),
    ).reset_index()

    costs = get_costs()
    bump  = settings["bump"]

    t1, t2, t3 = st.tabs(["Cost Entry","Margin Analysis","Price Recommendations"])

    # ── Cost Entry ──
    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        callout("Enter the production cost per item (ingredients + direct labor). "
                "This enables margin analysis and pricing recommendations.", kind="info")
        st.markdown("<br>", unsafe_allow_html=True)
        section_tag("COST PRICES PER ITEM")

        items = sorted(catalog["item_name"].tolist())
        new_costs = {}
        for i in range(0, len(items), 4):
            cols = st.columns(4, gap="small")
            for col, nm in zip(cols, items[i:i+4]):
                with col:
                    sp = float(catalog.loc[catalog["item_name"]==nm,"selling_price"].values[0])
                    v  = st.number_input(nm.title(), min_value=0.0,
                                         max_value=max(round(sp*0.99,2), 0.01),
                                         value=float(costs.get(nm, 0.0)),
                                         step=0.10, format="%.2f", key=f"cost_{nm}")
                    if v > 0:
                        new_costs[nm] = v

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Save Cost Prices"):
            if new_costs:
                set_costs(new_costs)
                db_save_costs([{"item_name":k,"cost_price":v,
                    "profit_margin": profit_margin(
                        float(catalog.loc[catalog["item_name"]==k,"selling_price"].values[0]), v
                    )} for k,v in new_costs.items()])
                st.success(f"Saved {len(new_costs)} cost prices.")
            else:
                st.warning("Enter at least one cost price.")

    # ── Margin Analysis ──
    with t2:
        st.markdown("<br>", unsafe_allow_html=True)
        if not costs:
            callout("Enter cost prices in the Cost Entry tab first.", kind="info")
        else:
            mdf = margin_table(txn, costs)
            if mdf.empty:
                callout("No items with cost data found.", kind="info")
            else:
                avg_m = mdf["margin_pct"].mean()
                low_n = (mdf["margin_pct"] < MIN_MARGIN).sum()
                best  = mdf.loc[mdf["margin_pct"].idxmax()]
                k1,k2,k3 = st.columns(3, gap="small")
                with k1: st.metric("Average Margin",   fmt_pct(avg_m))
                with k2: st.metric("Items Below Target", f"{low_n} items")
                with k3: st.metric("Best Margin Item",
                                   f"{best['item_name'].title()} ({fmt_pct(best['margin_pct'])})")

                st.markdown("<br>", unsafe_allow_html=True)
                section_tag("COST VS SELLING PRICE")
                cpd = mdf[["item_name","cost_price","selling_price"]].head(14)
                st.plotly_chart(chart_cost_vs_price(cpd), use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                section_tag("FULL MARGIN TABLE")
                d2 = mdf[["item_name","category","selling_price","cost_price","margin_pct","tier"]].copy()
                d2.columns = ["Item","Category","Selling Price","Cost Price","Margin %","Tier"]
                d2["Selling Price"] = d2["Selling Price"].apply(fmt_currency)
                d2["Cost Price"]    = d2["Cost Price"].apply(fmt_currency)
                d2["Margin %"]      = d2["Margin %"].apply(fmt_pct)
                st.dataframe(d2, use_container_width=True, hide_index=True)

    # ── Price Recommendations ──
    with t3:
        st.markdown("<br>", unsafe_allow_html=True)
        if not costs:
            callout("Enter cost prices in the Cost Entry tab first.", kind="info")
        else:
            mdf = margin_table(txn, costs)
            if mdf.empty:
                callout("No items with cost data found.", kind="info")
            else:
                opt = price_recs(mdf, bump)
                annual = opt["est_profit_gain"].sum() * 13
                if annual > 0:
                    callout(f"Applying all suggested price changes could generate an additional "
                            f"<strong>{fmt_currency(annual)}</strong> in gross profit over 12 months.",
                            kind="success")

                hi = mdf[mdf["margin_pct"] > 70].sort_values("margin_pct",ascending=False).head(5)
                if not hi.empty:
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_tag("HIGH-PERFORMING ITEMS — PROMOTE THESE")
                    hcols = st.columns(min(len(hi),5), gap="small")
                    for col,(_, r) in zip(hcols, hi.iterrows()):
                        with col:
                            card_html(
                                f"<p style='font-size:0.82rem;font-weight:600;color:{C['text']};"
                                f"margin:0 0 0.25rem;'>{r['item_name'].title()}</p>"
                                f"<p style='font-family:Georgia,serif;font-size:1.15rem;"
                                f"color:{C['text']};margin:0;'>{fmt_pct(r['margin_pct'])}</p>"
                                f"<p style='font-size:0.72rem;color:{C['muted']};margin:0.15rem 0 0;'>"
                                f"{r['tier']} performer</p>"
                            )

                st.markdown("<br>", unsafe_allow_html=True)
                section_tag("PRICE ADJUSTMENT RECOMMENDATIONS")
                needs = opt[opt["price_increase"] > 0]
                if needs.empty:
                    callout("All items are at or above the target margin. No changes needed.", kind="success")
                else:
                    d2 = needs[["item_name","selling_price","suggested_price",
                                "price_increase","increase_pct",
                                "margin_pct","new_margin_pct","est_profit_gain"]].copy()
                    d2.columns = ["Item","Current","Suggested","Increase ($)","Increase (%)",
                                  "Current Margin","New Margin","Est. Gain"]
                    for c in ["Current","Suggested","Increase ($)","Est. Gain"]:
                        d2[c] = d2[c].apply(fmt_currency)
                    for c in ["Increase (%)","Current Margin","New Margin"]:
                        d2[c] = d2[c].apply(fmt_pct)
                    st.dataframe(d2, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# 15. PAGE — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════

def pg_recommendations(txn: pd.DataFrame):
    page_title("Recommendations", "Prioritized actions to grow profit and streamline operations.")

    if txn.empty:
        callout("No data loaded. Use the sidebar to upload or load demo data.", kind="info")
        return

    costs = get_costs()
    mdf   = margin_table(txn, costs) if costs else pd.DataFrame()
    recs  = recommendations(txn, mdf)

    if not recs:
        callout("No recommendations generated. Your business appears to be performing well.", kind="success")
        return

    urgent_n = sum(1 for r in recs if r["priority"]==1)
    imp_n    = sum(1 for r in recs if r["priority"]==2)
    opp_n    = sum(1 for r in recs if r["priority"]==3)

    k1,k2,k3 = st.columns(3, gap="small")
    with k1: st.metric("Urgent Actions",    str(urgent_n))
    with k2: st.metric("Important Actions", str(imp_n))
    with k3: st.metric("Opportunities",     str(opp_n))
    st.markdown("<br>", unsafe_allow_html=True)

    fc1, fc2 = st.columns(2, gap="small")
    with fc1:
        show_types = st.multiselect("Filter by type",
            ["pricing","menu","marketing","operations"],
            default=["pricing","menu","marketing","operations"])
    with fc2:
        min_pri = st.selectbox("Priority", ["All","Urgent only","Important and above"])

    filtered = [r for r in recs if r["type"] in show_types]
    if min_pri == "Urgent only":           filtered = [r for r in filtered if r["priority"]==1]
    elif min_pri == "Important and above": filtered = [r for r in filtered if r["priority"]<=2]

    if not filtered:
        callout("No recommendations match the selected filters.", kind="info")
        return

    st.markdown("<br>", unsafe_allow_html=True)

    PRI = {
        1: {"label":"Urgent",      "bg":"#FAE8E8","border":"#8B2E2E","txt":"#8B2E2E"},
        2: {"label":"Important",   "bg":"#FAF3E0","border":"#7A5C12","txt":"#7A5C12"},
        3: {"label":"Opportunity", "bg":"#EAF0E8","border":"#2C5F2E","txt":"#2C5F2E"},
    }
    TYPE_LBL = {"pricing":"Pricing","menu":"Menu","marketing":"Marketing","operations":"Operations"}

    for rt in ["pricing","menu","marketing","operations"]:
        grp = [r for r in filtered if r["type"]==rt]
        if not grp: continue
        section_tag(TYPE_LBL.get(rt, rt.upper()))
        for rec in grp:
            p = PRI.get(rec["priority"], PRI[3])
            st.markdown(
                f"<div style='background:{C['card']};border:1px solid {C['border']};"
                f"border-left:3px solid {p['border']};border-radius:0 6px 6px 0;"
                f"padding:1.1rem 1.4rem;margin-bottom:0.75rem;'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:flex-start;margin-bottom:0.5rem;'>"
                f"<p style='font-family:Georgia,serif;font-size:0.95rem;"
                f"color:{C['text']};margin:0;font-weight:400;'>{rec['title']}</p>"
                f"<span style='font-size:0.68rem;text-transform:uppercase;"
                f"letter-spacing:0.08em;background:{p['bg']};color:{p['txt']};"
                f"padding:0.2rem 0.65rem;border-radius:3px;white-space:nowrap;"
                f"margin-left:1rem;font-weight:700;'>{p['label']}</span></div>"
                f"<p style='font-size:0.875rem;color:{C['muted']};margin:0 0 0.4rem;"
                f"line-height:1.6;'>{rec['detail']}</p>"
                f"<p style='font-size:0.78rem;color:{C['muted']};margin:0;"
                f"font-style:italic;'>Expected impact: {rec['impact']}</p></div>",
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 16. PAGE — REPORTS
# ═══════════════════════════════════════════════════════════════════════════

def pg_reports(txn: pd.DataFrame, costs: dict):
    page_title("Reports", "Generate, preview, download, and email performance reports.")

    if txn.empty:
        callout("No data loaded. Use the sidebar to upload or load demo data.", kind="info")
        return

    rc, _ = st.columns([4,4])
    with rc:
        rp  = st.radio("Report period", ["Weekly (7 days)","Monthly (30 days)"], horizontal=True)
        to  = st.text_input("Email report to (optional)", placeholder="owner@mycafe.com")

    days = 7 if "Weekly" in rp else 30

    b1,b2,_ = st.columns([2,2,5])
    with b1:
        gen  = st.button("Generate Report",  use_container_width=True)
    with b2:
        send = st.button("Send by Email",    use_container_width=True)

    if gen:
        with st.spinner("Building report…"):
            rep = build_report(txn, costs, days)
            st.session_state["report"]     = rep
            st.session_state["report_csv"] = export_csv(rep)
        st.success("Report ready.")

    if send:
        if not to:           st.warning("Enter a recipient email address.")
        elif not st.session_state.get("report"): st.warning("Generate the report first.")
        else:
            with st.spinner("Sending…"):
                ok, msg = send_email_report(to, st.session_state["report"]["period"],
                                            st.session_state["report_csv"])
            st.success(msg) if ok else st.error(msg)

    rep = st.session_state.get("report")
    if not rep:
        callout("Click <strong>Generate Report</strong> to build your summary.", kind="info")
        return

    st.markdown("---")
    st.markdown(f"<p style='font-size:0.85rem;color:{C['muted']};'>Period: {rep['period']}</p>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    section_tag("SUMMARY")
    s = rep["summary"]
    k1,k2,k3,k4,k5 = st.columns(5, gap="small")
    with k1: st.metric("Revenue",      fmt_currency(s["revenue"]))
    with k2: st.metric("Profit",       fmt_currency(s["profit"]))
    with k3: st.metric("Food Cost %",  fmt_pct(s["food_cost_pct"]))
    with k4: st.metric("Transactions", f"{s['orders']:,}")
    with k5: st.metric("Avg Order",    fmt_currency(s["avg_order"]))

    st.markdown("<br>", unsafe_allow_html=True)

    if not rep["daily_df"].empty:
        section_tag("DAILY REVENUE")
        st.plotly_chart(chart_revenue_line(rep["daily_df"]), use_container_width=True)

    cc1,cc2 = st.columns(2, gap="large")
    with cc1:
        if not rep["top_df"].empty:
            section_tag("TOP ITEMS")
            st.plotly_chart(chart_top_bar(rep["top_df"]), use_container_width=True)
    with cc2:
        if not rep["hours_df"].empty:
            section_tag("PEAK HOURS")
            st.plotly_chart(chart_peak_hours(rep["hours_df"]), use_container_width=True)

    if not rep["worst_df"].empty:
        st.markdown("<br>", unsafe_allow_html=True)
        section_tag("LOWEST PERFORMING ITEMS")
        dw = rep["worst_df"].copy()
        dw.columns = ["Item","Revenue","Units Sold"]
        dw["Revenue"] = dw["Revenue"].apply(fmt_currency)
        st.dataframe(dw, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.download_button("Download CSV Report",
                       data=st.session_state["report_csv"],
                       file_name=f"seralung_report_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# 17. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    inject_css()
    init_state()

    settings = render_sidebar()
    txn      = get_txn()
    costs    = get_costs()

    # Try loading persisted data from Supabase if session is empty
    if txn.empty:
        db_txn = db_load_txn()
        if not db_txn.empty:
            set_txn(db_txn)
            txn = db_txn

    if not costs:
        db_c = db_load_costs()
        if not db_c.empty and "item_name" in db_c.columns and "cost_price" in db_c.columns:
            set_costs(dict(zip(db_c["item_name"], db_c["cost_price"])))
            costs = get_costs()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Price Calculator", "Recommendations", "Reports"
    ])
    with tab1: pg_overview(txn, costs, settings)
    with tab2: pg_price_calc(txn, settings)
    with tab3: pg_recommendations(txn)
    with tab4: pg_reports(txn, costs)


main()
