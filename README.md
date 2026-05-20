# Seralung Opti — Cafe Business Optimization Software

**Version:** 1.0.0  
**Stack:** Python · Streamlit · Supabase · Pandas · Plotly  
**Target:** Small to medium cafe and food service businesses

---

## What It Does

Seralung Opti connects to your POS system or accepts CSV exports and turns raw transaction data into clear, actionable business intelligence. It helps cafe owners understand which menu items drive profit, when peak hours occur, what to price, and what to change.

---

## Core Modules

| Module | Purpose |
|---|---|
| **Overview** | Revenue, profit, food cost, top/worst items, peak hours |
| **Price Calculator** | Optimized pricing with margin recommendations |
| **Recommendations** | Combo offers, removals, promotions, inventory hints |
| **Reports** | Weekly/monthly exports with charts and email delivery |

---

## Project Structure

```
seralung-opti/
├── app.py                        # Streamlit entry point and navigation
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
│
├── pages/                        # Streamlit multi-page app
│   ├── 1_overview.py             # Dashboard and KPIs
│   ├── 2_price_calculator.py     # Price optimization tool
│   ├── 3_recommendations.py      # Business recommendations
│   └── 4_reports.py              # Report generation
│
├── services/                     # Business logic layer
│   ├── data_ingestion.py         # CSV upload and POS parsing
│   ├── data_cleaning.py          # Data normalization pipeline
│   ├── analytics.py              # Revenue, profit, trend calculations
│   ├── price_optimizer.py        # Margin and pricing logic
│   ├── recommendations.py        # Insight generation engine
│   └── report_generator.py       # PDF/CSV report builder
│
├── database/                     # Data access layer
│   ├── client.py                 # Supabase connection
│   ├── schema.sql                # PostgreSQL table definitions
│   ├── menu_items.py             # menu_items CRUD
│   ├── menu_costs.py             # menu_costs CRUD
│   └── transactions.py           # transactions CRUD
│
├── utils/                        # Shared utilities
│   ├── styling.py                # Streamlit CSS theming
│   ├── formatters.py             # Number and currency helpers
│   ├── charts.py                 # Plotly chart factories
│   └── validators.py             # Data validation helpers
│
├── reports/                      # Report output handlers
│   ├── weekly_report.py          # 7-day rolling report
│   ├── monthly_report.py         # Calendar month report
│   └── email_sender.py           # SMTP email delivery
│
└── assets/
    └── sample_data/
        └── sample_transactions.csv   # Demo data for testing
```

---

## Quick Start

See [SETUP.md](SETUP.md) for full installation and configuration steps.

```bash
git clone https://github.com/your-org/seralung-opti.git
cd seralung-opti
pip install -r requirements.txt
cp .env.example .env          # fill in your Supabase credentials
streamlit run app.py
```

---

## Documentation Index

- [SETUP.md](SETUP.md) — Installation, environment setup, first run
- [DATABASE.md](DATABASE.md) — Schema reference, Supabase setup, migrations
- [CONTRIBUTING.md](CONTRIBUTING.md) — Code style, branching, pull request guide

---

## License

Proprietary. All rights reserved — Seralung Opti © 2024.
