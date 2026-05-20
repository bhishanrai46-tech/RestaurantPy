"""
app.py
------
Seralung Opti — Cafe Business Optimization Software

Entry point for the Streamlit multi-page application.
Configures the page, applies global styling, and renders the
home screen with navigation guidance.

Run with:
    streamlit run app.py
"""

import streamlit as st
from utils.styling import apply_global_styles, page_header, COLORS, card, insight_callout

# -----------------------------------------------------------------------
# Page configuration — must be the first Streamlit call in the file
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Seralung Opti",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()


# -----------------------------------------------------------------------
# Sidebar branding
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 1rem 0 1.5rem 0;">
            <p style="
                font-family: Georgia, serif;
                font-size: 1.25rem;
                font-weight: 400;
                color: {COLORS['text_primary']};
                letter-spacing: -0.02em;
                margin: 0;
            ">Seralung Opti</p>
            <p style="
                font-size: 0.75rem;
                color: {COLORS['text_secondary']};
                margin-top: 0.15rem;
                letter-spacing: 0.02em;
            ">Cafe Business Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        f"<p style='font-size:0.78rem; color:{COLORS['text_secondary']}; "
        f"text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>Navigation</p>",
        unsafe_allow_html=True
    )
    st.page_link("pages/1_overview.py",         label="Overview")
    st.page_link("pages/2_price_calculator.py", label="Price Calculator")
    st.page_link("pages/3_recommendations.py",  label="Recommendations")
    st.page_link("pages/4_reports.py",          label="Reports")
    st.markdown("---")
    st.markdown(
        f"<p style='font-size:0.72rem; color:{COLORS['text_secondary']};'>"
        f"v1.0.0 · Seralung Opti</p>",
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------
# Home page content
# -----------------------------------------------------------------------
page_header(
    "Welcome to Seralung Opti",
    "Cafe business optimization for small and medium food service businesses."
)

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown(
        f"""
        <p style='font-size:0.95rem; color:{COLORS['text_secondary']}; line-height:1.75;'>
        Seralung Opti connects to your point-of-sale data and turns raw
        transaction records into clear, prioritized business decisions.
        It tells you which items to price higher, which to remove, and
        when your kitchen should be fully staffed.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### How to get started")
    st.markdown(
        f"""
        <ol style='font-size:0.9rem; color:{COLORS['text_secondary']}; line-height:2;'>
            <li>Go to <strong style='color:{COLORS['text_primary']};'>Overview</strong> and upload your POS export or CSV file.</li>
            <li>Review your revenue, profit, and food cost KPIs on the dashboard.</li>
            <li>Open <strong style='color:{COLORS['text_primary']};'>Price Calculator</strong> to see which items need a price adjustment.</li>
            <li>Read the <strong style='color:{COLORS['text_primary']};'>Recommendations</strong> for prioritized, actionable insights.</li>
            <li>Generate a weekly or monthly <strong style='color:{COLORS['text_primary']};'>Report</strong> to share with your team.</li>
        </ol>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    insight_callout(
        "<strong>No data yet?</strong> Go to Overview and click "
        "<em>Load demo data</em> to explore the app with 90 days "
        "of realistic sample cafe transactions.",
        kind="info"
    )
    insight_callout(
        "<strong>Connecting a POS system?</strong> Export your "
        "transactions as CSV with columns: "
        "<code>item_name, qty, revenue, sold_at, category</code>.",
        kind="success"
    )

st.markdown("---")

# Module summary cards
st.markdown("#### What each module does")
m1, m2, m3, m4 = st.columns(4, gap="small")

module_descriptions = [
    ("Overview",         "Revenue, profit, food cost KPIs, top and worst items, peak trading hours, and trend charts."),
    ("Price Calculator", "Margin analysis per item, suggested price increases, and projected profit impact."),
    ("Recommendations",  "Prioritized business actions: combos, removals, promotions, and operational hints."),
    ("Reports",          "Weekly and monthly report downloads with charts and optional email delivery."),
]

for col, (module_name, description) in zip([m1, m2, m3, m4], module_descriptions):
    with col:
        card(
            f"<p style='font-family: Georgia, serif; font-size:0.95rem; "
            f"color:{COLORS['text_primary']}; margin:0 0 0.5rem 0;'>{module_name}</p>"
            f"<p style='font-size:0.82rem; color:{COLORS['text_secondary']}; "
            f"line-height:1.55; margin:0;'>{description}</p>"
        )
