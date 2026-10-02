# FMCG Retail Forecasting, Elasticity & Promo-ROI Studio

Streamlit app (Python) with a Groq LLM analyst. Forecast **sales value, sales units, inventory value and inventory units**
at **category, subcategory, brand, sub-brand and product** level, measure **price elasticity**, find **what drives sales**,
and evaluate **promotion and trade-investment ROI**.

> The data is synthetic and illustrative.

## Project layout

```
.
├── app.py                      # Streamlit UI (6 tabs)
├── engine.py                   # forecasting, elasticity, drivers, ROI, inventory (no Streamlit imports)
├── llm.py                      # Groq wrapper + prompts
├── holidays_calendar.py        # month-level holiday calendar (also used for future months)
├── build_dataset.py            # enriches the original CSV -> data/retailer_fmcg_enriched_data.csv
├── requirements.txt
├── .streamlit/secrets.toml.example
└── data/
    ├── retailer_fmcg_synthetic_dashboard_data.csv   # original file
    └── retailer_fmcg_enriched_data.csv              # file the app reads
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python build_dataset.py                                 # optional: regenerates the enriched CSV (seeded, reproducible)
streamlit run app.py
```

**Groq key** (any one of): paste it in the sidebar, or set `GROQ_API_KEY` in `.streamlit/secrets.toml`, or export the
`GROQ_API_KEY` environment variable. The default model is `llama-3.3-70b-versatile`; you can type any other Groq model id in the sidebar.
Never commit your key (`.streamlit/secrets.toml` is git-ignored).

## What is in the app

| Tab | What you can do |
|---|---|
| Overview | KPIs, trend of any measure split by category/subcategory/brand/sub-brand/product, promo intensity, trade vs incremental profit, performance table |
| Forecast | Choose level + item + measure (value, units, inventory value, inventory units), horizon 3-12 months, model (Auto, Seasonal naive, Holt-Winters, Ridge with drivers, Gradient Boosting, Ensemble), backtest comparison, prediction intervals, what-if scenario (promo share, discount depth, trade %, price change), forecast all items at a level, CSV download |
| Elasticity & Drivers | Price elasticity with confidence intervals at any level, discount simulator (best discount for gross profit), lift of each promo mechanic / holiday / trade spend, units bridge (baseline to actual), permutation feature importance |
| Promo & Trade ROI | Yes/No tables for promo, holiday and trade, ROI by promo type / holiday / retailer / country / brand / product / month, promo type x retailer heat-map, trade amount vs incremental gross profit |
| Inventory | Inventory value and stock cover trends, by-level view, stock-status mix (healthy / overstock / low), biggest overstock positions |
| AI Analyst | Executive summary and free-form Q&A grounded on a compact fact sheet (the LLM never sees raw rows) |

## New columns added to the data (`build_dataset.py`)

| Column | Meaning |
|---|---|
| `Inventory_Value_USD`, `Inventory_Value_LC` | **Closing inventory value** = `Closing_Inventory_Units x Unit_Cost_USD` (LC = local currency) |
| `Closing_Inventory_Units`, `Stock_Cover_Days`, `Stock_Status` | Month-end stock, cover (days), Healthy (15-60 d) / Low stock (<15 d) / Overstock (>60 d) |
| `Unit_Cost_USD` | Cost per unit (40-52% of list price depending on subcategory) |
| `List_Unit_Price_USD`, `Net_Unit_Price_USD`, `Price_Index` | Regular price, price after promo, net / list |
| `Promo_Flag`, `Promo_Type`, `Discount_Pct` | Promotion yes/no; Price Discount (TPR), Multibuy, Display / Feature, Bundle / Gift, Loyalty / Coupon; depth |
| `Holiday_Flag`, `Holiday_Name` | Ramadan, Ramadan & Eid al-Fitr, Eid al-Adha, White Friday, National Day, Christmas / New Year (country-specific) |
| `Trade_Flag`, `Trade_Pct_of_Sales`, `Trade_Investment_USD` | Trade investment yes/no, % of baseline sales, amount in USD |
| `Baseline_Units`, `Baseline_USD_Value`, `Baseline_Gross_Profit_USD` | What would have sold with no promo and no trade (holiday effect kept) |
| `Incremental_Units`, `Incremental_USD_Value`, `Incremental_Gross_Profit_USD` | Actual minus baseline |
| `Gross_Profit_USD` | Net revenue minus unit cost x units |
| `Trade_ROI_Pct` | (Incremental gross profit - trade investment) / trade investment x 100 (rows with trade only) |
| `Store`, `Sub_Brand`, `Year`, `Month_Num`, `FX_Rate_LC_per_USD`, `Base_Units_Original` | Helper columns; `Base_Units_Original` keeps the original units |

**Modified columns:** `Sales_Units`, `USD_Value` and `LC_Value` now include promo / holiday / trade effects (the original
numbers are the baseline). Total sales value is about 14% higher than in the original file.

**Cleaning:** garbled text in the Lux soap description and pack size fixed (`Lux Soft Touch 4 x 100 g`);
`Colgate Colgate Total` -> `Colgate Total`; `deodrants` -> `deodorants`.

### How the effects were simulated (so you can explain the data)

* Price elasticity by subcategory between -1.1 (toothpaste) and -1.9 (deodorants), applied to the discount.
* Promo mechanic lift on top of price: Display 1.20, Multibuy 1.15, Bundle 1.12, Loyalty 1.06, TPR 1.06.
* Holiday lifts of +4% to +18% depending on event and category; trade spend lifts units by about 3% per 1 pp of sales invested, scaled by retailer effectiveness.
* Promo probability depends on retailer and rises in holiday months.
* Rows with no promo and no trade have `Sales_Units == Baseline_Units`.

## Methods

* **Forecasting:** monthly series at the chosen level. Models: seasonal naive with YoY growth, damped Holt-Winters, Ridge regression and Gradient Boosting using drivers (promo share, discount depth, holiday share, trade intensity, price index) plus seasonality, and an ensemble. *Auto* picks the lowest-WAPE model on a rolling hold-out. Future holidays come from the calendar; other drivers default to the same month last year unless you set a scenario. Intervals come from backtest error and widen with the horizon.
* **Elasticity:** log-log OLS with product x country fixed effects, controlling for promo mechanic, holiday, trade %, trend and seasonality. TPR is the reference mechanic, so the price term also carries TPR's small non-price lift.
* **Drivers:** the coefficients give the lift of each lever; the units bridge splits incremental units across levers; permutation importance (Gradient Boosting, time-based hold-out) ranks features.
* **ROI:** incremental gross profit (after discount cost) vs trade investment.

## Caveats

* Each row is a sampled store-product-month, not a continuous store panel, so inventory is modelled as a month-end snapshot per row and forecasts work on monthly totals.
* Only 24 months of history: seasonal estimates are rough; use the backtest table to judge reliability.
* The baseline columns exist because the data is synthetic. With real data, estimate the baseline (e.g. from the driver model) before computing ROI.
* Elasticity is estimated from promotional price variation; it is not a regular-price elasticity.
