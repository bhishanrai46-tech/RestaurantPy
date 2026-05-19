# ☕ CafeIQ Pro — Cafe & Restaurant Optimization Dashboard

A professional, data-driven dashboard for cafe owners to increase profit, reduce waste,
optimize staffing, and monitor live sales — powered by Python, Streamlit, and Plotly.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install streamlit pandas numpy plotly openpyxl

# 2. Run the app
streamlit run app.py
```

That's it. Demo data is seeded automatically on first run.

---

## 📊 Features

| Tab | What it does |
|-----|-------------|
| **Dashboard** | KPIs, revenue trend, peak heatmap, profit waterfall |
| **Menu** | BCG matrix, margin analysis, pricing recommendations, what-if tester |
| **Waste** | 30-day waste tracking, ingredient-level analysis, inventory & EOQ |
| **Staffing** | Demand vs staff heatmap, overstaffing detection, roster editor |
| **Live Sales** | Simulated real-time order stream, daily target progress |
| **Reports** | Excel (5 sheets), HTML/PDF, CSV exports |

---

## 🔑 Core Variables Tracked

- Food cost % (target < 32%)
- Labor cost % (target < 30%)
- Waste % of revenue (target < 3%)
- Gross margin per item
- Profit per kitchen minute (batch-adjusted)
- Peak-hour demand vs staffing
- Inventory turnover & EOQ
- Average order value
- Customer lifetime value proxy

---

## 🗄️ Database

**Default:** SQLite (`cafeiq.db`) — zero setup, created automatically.

**Production (Supabase/PostgreSQL):**

```bash
pip install supabase
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-anon-key"
```

The app detects the env vars and switches to Supabase automatically.
You'll need to run the same `CREATE TABLE` DDL from `app.py → _ddl()` in your Supabase SQL editor.

---

## 📐 Prep Time Model

Instead of fixed times, the app uses a realistic batch + rush model:

```
T_effective = BasePrep × BatchFactor × RushFactor − PrepBonus

BatchFactor  : 0.65 if daily_sold > 80 (batch cooking)
             : 0.80 if daily_sold > 40 (moderate batching)
             : 1.00 otherwise (single order)
RushFactor   : 1.20 during peak hours (8–9 AM, 12–1 PM)
PrepBonus    : −0.25 if ingredients are pre-prepped
```

---

## 🍽️ Sample Menu

Comes pre-loaded with 15 realistic Sydney cafe items across:
- **Coffee**: Espresso, Long Black, Cappuccino, Flat White, Iced Latte, Chai Latte
- **Food**: Avocado Toast, Eggs Benedict, Granola Bowl, Chicken Wrap, Caesar Salad
- **Snacks**: Banana Bread, Chocolate Croissant, Blueberry Muffin

All prices in AUD with realistic food costs and demand volumes.

---

## 📤 Exports

| Format | Contents |
|--------|----------|
| **Excel** | Summary · Menu Analysis · Pricing · Daily Sales · Waste Log |
| **HTML/PDF** | Styled management report (print to PDF from browser) |
| **CSV** | Menu, Sales, and Waste as separate files |

---

## ⚙️ Configuration (Sidebar)

- **Daily Revenue Target** — drives live sales progress bar
- **Fixed Costs / Week** — rent, utilities, insurance (affects net profit)
- **Default Labor Rate** — used for staffing cost calculations

---

## 🏗️ Architecture

```
app.py
├── Database layer      SQLite default · Supabase optional
├── Data loaders        pandas SQL queries
├── Business engine     margin, prep time, EOQ, recommendations
├── Chart functions     Plotly go/express (15+ charts)
├── Recommendation AI   rule-based, specific, actionable
├── Live simulation     Poisson arrival model by hour
└── Report builders     Excel, HTML, CSV
```

---

Built with ❤️ for cafe owners who want real numbers, not generic dashboards.
