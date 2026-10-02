"""
FMCG Retail Forecasting, Elasticity & Promo-ROI Studio
Streamlit + scikit-learn + Groq LLM

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import engine as E
import llm

st.set_page_config(page_title="FMCG Forecasting & Promo Studio", page_icon="📈", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "retailer_fmcg_enriched_data.csv"


# ----------------------------------------------------------------------------
# Small UI helpers (work across Streamlit versions)
# ----------------------------------------------------------------------------
def show(fig):
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig, width="stretch")


def table(df: pd.DataFrame, **kw):
    try:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, width="stretch", hide_index=True, **kw)


def money(x: float) -> str:
    if x is None or np.isnan(x):
        return "-"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"${x / div:,.2f}{unit}"
    return f"${x:,.0f}"


def num(x: float) -> str:
    if x is None or np.isnan(x):
        return "-"
    for unit, div in (("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"{x / div:,.2f}{unit}"
    return f"{x:,.0f}"


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def fmt_metric(label: str, v: float) -> str:
    return money(v) if "USD" in label or "value" in label.lower() else num(v)


# ----------------------------------------------------------------------------
# Cached computations
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(source) -> pd.DataFrame:
    df = E.load_data(source)
    df["Month"] = df["Date"].dt.strftime("%Y-%m")
    df["Promo_Label"] = np.where(df["Promo_Flag"] == 1, "Promo: Yes", "Promo: No")
    df["Holiday_Label"] = np.where(df["Holiday_Flag"] == 1, "Holiday: Yes", "Holiday: No")
    df["Trade_Label"] = np.where(df["Trade_Flag"] == 1, "Trade investment: Yes", "Trade investment: No")
    return df


@st.cache_data(show_spinner=False)
def c_forecast(hist, target, h, model, weights, scenario, holdout, interval):
    return E.forecast_series(hist, target, h, model, weights, scenario, holdout=holdout, interval=interval)


@st.cache_data(show_spinner="Forecasting every item at this level...")
def c_forecast_all(df, level_col, target, h, model, scenario):
    return E.forecast_all_entities(df, level_col, target, h, model, scenario)


@st.cache_data(show_spinner="Estimating elasticities...")
def c_elasticity(df, level_col):
    return E.elasticity_table(df, level_col)


@st.cache_data(show_spinner="Fitting demand model...")
def c_demand_model(df):
    fm = E.fit_demand_model(df)
    decomp = E.driver_decomposition(df.reset_index(drop=True), fm)
    return {"coef": fm["coef"], "r2": fm["r2_within"], "n": fm["n"], "decomp": decomp}


@st.cache_data(show_spinner="Training driver model (permutation importance)...")
def c_importance(df):
    fi = E.feature_importance(df)
    return fi, fi.attrs.get("r2_holdout", np.nan)


@st.cache_data(show_spinner=False)
def c_fact_sheet(df, horizon: int = 6) -> dict:
    """Compact JSON-able summary handed to the Groq LLM."""
    k = E.overall_kpis(df)
    hist = E.aggregate_monthly(df, None)
    w = E.country_weights(df)
    months = sorted(df["Month"].unique())
    last12, prev12 = hist["USD_Value"].iloc[-12:].sum(), hist["USD_Value"].iloc[-24:-12].sum()
    facts = {
        "selection": {"rows": int(len(df)), "first_month": months[0], "last_month": months[-1],
                      "countries": sorted(df["Country"].unique().tolist()),
                      "retailers": sorted(df["Retailer"].unique().tolist()),
                      "categories": sorted(df["Category"].unique().tolist())},
        "kpis": {k2: (round(float(v), 2) if v == v else None) for k2, v in k.items()},
        "sales_last12m_usd": round(float(last12), 0),
        "sales_yoy_growth_pct": round(float((last12 / prev12 - 1) * 100), 1) if prev12 > 0 else None,
    }
    fc = {}
    for lab, col in E.METRICS.items():
        try:
            r = E.forecast_series(hist, col, horizon, "Auto (best on backtest)", w)
            agg = np.mean if col in E.STOCK_METRICS else np.sum
            fc[lab] = {"model": r.model_used, f"next_{horizon}m_{'avg_level' if col in E.STOCK_METRICS else 'total'}": round(float(agg(r.forecast["forecast"])), 0),
                       f"last_{horizon}m_{'avg_level' if col in E.STOCK_METRICS else 'total'}": round(float(agg(hist[col].iloc[-horizon:])), 0),
                       "backtest_wape_pct": round(float(r.backtest.loc[r.backtest['Model'] == r.model_used, 'WAPE %'].iloc[0]), 1)
                       if (r.backtest["Model"] == r.model_used).any() else None}
        except Exception:
            pass
    facts["forecast"] = fc
    br = df.groupby("Brand")["USD_Value"].sum().sort_values(ascending=False)
    facts["top_brands_by_sales_usd"] = br.head(5).round(0).to_dict()
    facts["bottom_brands_by_sales_usd"] = br.tail(3).round(0).to_dict()
    el = E.elasticity_table(df, "Subcategory")
    if len(el):
        facts["price_elasticity_by_subcategory"] = {r["Subcategory"]: round(float(r["Elasticity"]), 2) for _, r in el.iterrows()}
    if len(df) > 300:
        fm = E.fit_demand_model(df)
        c = fm["coef"].set_index("term")
        facts["driver_effects_on_units_pct"] = {t: round(float(c.loc[t, "effect_on_units_%"]), 1)
                                                for t in c.index if t.startswith(("promo:", "holiday:"))}
        if "trade_pct" in c.index:
            facts["trade_effect_units_pct_per_1pp_of_sales"] = round(float(c.loc["trade_pct", "effect_on_units_%"]), 2)
        facts["units_decomposition_pct_of_actual"] = {r["Driver"]: round(float(r["% of actual"]), 1)
                                                      for _, r in E.driver_decomposition(df.reset_index(drop=True), fm).iterrows()}
    rp = E.roi_table(df, "Promo_Type")
    facts["roi_by_promo_type"] = {r["Promo_Type"]: {"volume_lift_pct": round(float(r["Volume lift %"]), 1),
                                                     "incremental_gp_usd": round(float(r["Incremental_GP"]), 0),
                                                     "trade_usd": round(float(r["Trade_USD"]), 0),
                                                     "trade_roi_pct": None if r["Trade ROI %"] != r["Trade ROI %"] else round(float(r["Trade ROI %"]), 1)}
                                  for _, r in rp.iterrows()}
    rr = E.roi_table(df, "Retailer")
    facts["trade_roi_pct_by_retailer"] = {r["Retailer"]: (None if r["Trade ROI %"] != r["Trade ROI %"] else round(float(r["Trade ROI %"]), 1)) for _, r in rr.iterrows()}
    inv = E.inventory_summary(df[df["Date"] == df["Date"].max()], "Subcategory")
    facts["inventory_latest_month_by_subcategory"] = {
        r["Subcategory"]: {"inventory_usd": round(float(r["Inventory_USD"]), 0), "cover_days": round(float(r["Cover days"]), 1),
                           "overstock_pct": round(float(r["Overstock %"]), 1), "low_stock_pct": round(float(r["Low stock %"]), 1)}
        for _, r in inv.iterrows()}
    return facts


# ----------------------------------------------------------------------------
# Load data + sidebar filters
# ----------------------------------------------------------------------------
st.title("📈 FMCG Retail Forecasting, Elasticity & Promo ROI Studio")
st.caption("Forecast value, units and inventory at product / sub-brand / brand / category level, understand price "
           "elasticity and the drivers behind sales, and measure promotion & trade-investment ROI. "
           "Data is synthetic and illustrative.")

with st.sidebar:
    st.header("⚙️ Data")
    up = st.file_uploader("Upload enriched CSV (optional)", type="csv")
    if up is not None:
        raw = load_csv(up)
    elif DATA_PATH.exists():
        raw = load_csv(str(DATA_PATH))
    else:
        st.error(f"Data file not found at {DATA_PATH}. Run `python build_dataset.py` or upload a CSV.")
        st.stop()

    st.header("🔎 Filters")
    st.caption("Leave empty = all. Filters apply to every tab.")
    f_country = st.multiselect("Country", sorted(raw["Country"].unique()))
    f_retailer = st.multiselect("Retailer", sorted(raw["Retailer"].unique()))
    f_cat = st.multiselect("Category", sorted(raw["Category"].unique()))
    f_sub = st.multiselect("Subcategory", sorted(raw["Subcategory"].unique()))
    f_brand = st.multiselect("Brand", sorted(raw["Brand"].unique()))

    st.header("🤖 Groq LLM")
    key_in = st.text_input("Groq API key", type="password", help="Or set GROQ_API_KEY in .streamlit/secrets.toml / env var")
    model_pick = st.selectbox("Model", llm.DEFAULT_MODELS + ["(custom)"])
    model_name = st.text_input("Custom model id", value="") if model_pick == "(custom)" else model_pick


def get_api_key() -> str:
    if key_in:
        return key_in
    try:
        k = st.secrets.get("GROQ_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


f = raw.copy()
for col, sel in (("Country", f_country), ("Retailer", f_retailer), ("Category", f_cat),
                 ("Subcategory", f_sub), ("Brand", f_brand)):
    if sel:
        f = f[f[col].isin(sel)]
if f.empty:
    st.warning("No data for this combination of filters.")
    st.stop()
if f["Month"].nunique() < 18:
    st.warning("Fewer than 18 months of history in this selection - forecasts will be unreliable.")

tab_ov, tab_fc, tab_el, tab_roi, tab_inv, tab_ai = st.tabs(
    ["📊 Overview", "🔮 Forecast", "📉 Elasticity & Drivers", "🎯 Promo & Trade ROI", "📦 Inventory", "🤖 AI Analyst"])

# ----------------------------------------------------------------------------
# OVERVIEW
# ----------------------------------------------------------------------------
with tab_ov:
    k = E.overall_kpis(f)
    c = st.columns(6)
    c[0].metric("Sales value", money(k["sales_usd"]))
    c[1].metric("Sales units", num(k["units"]))
    c[2].metric("Inventory value (latest month)", money(k["inventory_usd"]))
    c[3].metric("Promo share of observations", f"{k['promo_share']:.1f}%")
    c[4].metric("Trade investment", money(k["trade_usd"]))
    c[5].metric("Trade ROI", "-" if np.isnan(k["trade_roi"]) else f"{k['trade_roi']:.0f}%")

    a, b, _ = st.columns([1, 1, 1])
    m_label = a.selectbox("Metric", list(E.METRICS), key="ov_metric")
    split = b.selectbox("Split by", list(E.LEVELS), index=3, key="ov_split")
    mcol, scol = E.METRICS[m_label], E.LEVELS[split]
    if scol is None:
        trend = f.groupby("Date", as_index=False)[mcol].sum()
        fig = px.line(trend, x="Date", y=mcol, markers=True, title=f"{m_label} by month")
    else:
        top = f.groupby(scol)[mcol].sum().nlargest(8).index
        g = f[f[scol].isin(top)].groupby(["Date", scol], as_index=False)[mcol].sum()
        fig = px.line(g, x="Date", y=mcol, color=scol, markers=True, title=f"{m_label} by month - top 8 {split.lower()}s")
    fig.update_layout(yaxis_title=m_label, legend_title=None)
    show(fig)

    c1, c2 = st.columns(2)
    mon = E.aggregate_monthly(f, None).reset_index()
    fig1 = go.Figure()
    fig1.add_bar(x=mon["Date"], y=mon["promo_share"] * 100, name="Promo share %")
    fig1.add_scatter(x=mon["Date"], y=mon["discount_depth"], name="Avg discount % (all rows)", mode="lines+markers", yaxis="y2")
    fig1.update_layout(title="Promotion intensity", yaxis_title="Promo share %",
                       yaxis2=dict(title="Avg discount %", overlaying="y", side="right"), legend=dict(orientation="h"))
    with c1:
        show(fig1)
    fig2 = go.Figure()
    fig2.add_bar(x=mon["Date"], y=mon["Trade_Investment_USD"], name="Trade investment $")
    fig2.add_scatter(x=mon["Date"], y=mon["Incremental_Gross_Profit_USD"], name="Incremental gross profit $", mode="lines+markers")
    fig2.update_layout(title="Trade investment vs incremental gross profit", legend=dict(orientation="h"))
    with c2:
        show(fig2)

    st.subheader("Performance by level")
    lv = st.selectbox("Level", [l for l in E.LEVELS if l != "Total"], index=2, key="ov_level")
    lc = E.LEVELS[lv]
    perf = f.groupby(lc).agg(Sales_USD=("USD_Value", "sum"), Units=("Sales_Units", "sum"),
                             Promo_share_pct=("Promo_Flag", "mean"),
                             Trade_USD=("Trade_Investment_USD", "sum")).reset_index()
    perf["Promo_share_pct"] = perf["Promo_share_pct"] * 100
    inv_latest = f[f["Date"] == f["Date"].max()].groupby(lc)["Inventory_Value_USD"].sum().rename("Inventory_USD_latest")
    perf = perf.merge(inv_latest, on=lc, how="left").sort_values("Sales_USD", ascending=False)
    table(perf.round(1))
    st.download_button("Download this table (CSV)", csv_bytes(perf), f"performance_by_{lv.lower()}.csv", "text/csv")

# ----------------------------------------------------------------------------
# FORECAST
# ----------------------------------------------------------------------------
with tab_fc:
    st.markdown("Pick **what** to forecast (level, item, measure), **how** (model, horizon) and optionally test a "
                "**what-if promo / trade / price plan** for the future months.")
    r1 = st.columns([1, 1.4, 1.2, 1, 1.4])
    lvl_label = r1[0].selectbox("Forecast level", list(E.LEVELS), index=3, key="fc_level")
    lcol = E.LEVELS[lvl_label]
    if lcol:
        ents = sorted(f[lcol].unique())
        entity = r1[1].selectbox(lvl_label, ents, key="fc_entity")
        sub = f[f[lcol] == entity]
        title_ent = f"{lvl_label}: {entity}"
    else:
        r1[1].selectbox("Item", ["All selected data"], disabled=True)
        sub, title_ent = f, "Total (all filtered data)"
    metric_label = r1[2].selectbox("Measure", list(E.METRICS), key="fc_metric")
    horizon = r1[3].slider("Horizon (months)", 3, 12, 6, key="fc_h")
    model = r1[4].selectbox("Model", E.MODEL_NAMES, key="fc_model")
    target = E.METRICS[metric_label]
    is_stock = target in E.STOCK_METRICS

    hist = E.aggregate_monthly(sub, None)
    weights = E.country_weights(sub)

    with st.expander("What-if scenario for the forecast period (promotion / holiday / trade / price)"):
        use_scn = st.checkbox("Apply scenario", value=False, key="scn_on")
        ref = hist.iloc[-12:]
        s1, s2, s3, s4 = st.columns(4)
        scn_promo = s1.slider("Promo share %", 0, 100, int(round(ref["promo_share"].mean() * 100)), key="scn_p")
        scn_depth = s2.slider("Avg discount depth %", 0, 40, int(round(ref["discount_depth"].mean())), key="scn_d")
        scn_trade = s3.slider("Trade investment % of sales", 0.0, 12.0, float(round(ref["trade_intensity"].mean(), 1)), 0.1, key="scn_t")
        scn_price = s4.slider("List price change %", -10, 10, 0, key="scn_pr")
        st.caption("Holiday timing comes from the built-in calendar (Ramadan, Eid, White Friday, national days, Christmas). "
                   "Scenario levers are used by the driver-based models (Ridge, Gradient Boosting, Ensemble, Auto).")
    with st.expander("Advanced"):
        a1, a2 = st.columns(2)
        holdout = a1.selectbox("Backtest window (months)", [3, 6], index=0)
        interval = a2.selectbox("Prediction interval", [0.80, 0.90, 0.95], index=0, format_func=lambda x: f"{int(x*100)}%")

    scenario = ({"promo_share": scn_promo / 100, "discount_depth": float(scn_depth), "trade_intensity": float(scn_trade),
                 "price_change_pct": float(scn_price)} if use_scn else None)

    if len(hist) < 14:
        st.error("Not enough history to forecast this selection.")
    else:
        res = c_forecast(hist, target, horizon, model, weights, scenario, holdout, interval)
        base_res = c_forecast(hist, target, horizon, model, weights, None, holdout, interval) if use_scn else None
        fcast = res.forecast
        agg = np.mean if is_stock else np.sum
        lbl = "avg level" if is_stock else "total"
        last_h = agg(hist[target].iloc[-horizon:])
        ly_idx = [d - pd.DateOffset(years=1) for d in fcast.index]
        ly_vals = [hist.loc[d, target] for d in ly_idx if d in hist.index]
        ly = agg(ly_vals) if len(ly_vals) == horizon else np.nan
        nxt = agg(fcast["forecast"])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Forecast {lbl} (next {horizon}M)", fmt_metric(metric_label, nxt))
        k2.metric(f"vs last {horizon}M actual", f"{(nxt / last_h - 1) * 100:+.1f}%" if last_h else "-")
        k3.metric("vs same months last year", "-" if np.isnan(ly) else f"{(nxt / ly - 1) * 100:+.1f}%")
        wape = res.backtest.loc[res.backtest["Model"] == res.model_used, "WAPE %"]
        k4.metric("Model used / backtest WAPE", f"{res.model_used.split(' (')[0]}", f"{wape.iloc[0]:.1f}% WAPE" if len(wape) else None, delta_color="off")

        if use_scn and res.model_used in ("Seasonal naive + growth", "Holt-Winters (damped, seasonal)"):
            st.info("The selected model does not use drivers, so the scenario has no effect. "
                    "Choose Regression with drivers, Gradient Boosting, Ensemble or Auto.")
        if use_scn and base_res is not None and res.model_used not in ("Seasonal naive + growth", "Holt-Winters (damped, seasonal)"):
            b = agg(base_res.forecast["forecast"])
            st.success(f"Scenario vs baseline plan ({base_res.model_used.split(' (')[0]}): "
                       f"{fmt_metric(metric_label, nxt)} vs {fmt_metric(metric_label, b)} ({(nxt / b - 1) * 100:+.1f}%).")

        fig = go.Figure()
        fig.add_scatter(x=hist.index, y=hist[target], name="Actual", mode="lines+markers")
        fig.add_scatter(x=fcast.index, y=fcast["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fcast.index, y=fcast["lower"], fill="tonexty", line=dict(width=0),
                        fillcolor="rgba(255,127,14,0.18)", name=f"{int(interval*100)}% interval", hoverinfo="skip")
        fig.add_scatter(x=fcast.index, y=fcast["forecast"], name=f"Forecast ({'scenario' if use_scn else 'plan'})",
                        mode="lines+markers", line=dict(dash="dash", color="#ff7f0e"))
        if base_res is not None:
            fig.add_scatter(x=base_res.forecast.index, y=base_res.forecast["forecast"], name="Baseline plan",
                            mode="lines", line=dict(dash="dot", color="grey"))
        fig.update_layout(title=f"{metric_label} - {title_ent}", yaxis_title=metric_label, legend=dict(orientation="h"))
        show(fig)

        out = fcast.copy()
        out.index.name = "Month"
        out = out.reset_index()
        out["Month"] = out["Month"].dt.strftime("%Y-%m")
        fx = res.future_exog.reset_index(drop=True)
        out["Promo share %"] = fx["promo_share"] * 100
        out["Avg discount %"] = fx["discount_depth"]
        out["Holiday share %"] = fx["holiday_share"] * 100
        out["Trade % of sales"] = fx["trade_intensity"]
        out.insert(1, "Level", title_ent)
        out.insert(2, "Measure", metric_label)
        out.insert(3, "Model", res.model_used)
        table(out.round(1))
        st.download_button("Download forecast (CSV)", csv_bytes(out), "forecast.csv", "text/csv")

        st.subheader("Model comparison (backtest on the last months of history)")
        table(res.backtest.round(2))
        st.caption("WAPE = sum of absolute errors / sum of actuals. Auto picks the model with the lowest WAPE. "
                   "Intervals are derived from backtest error and widen with the horizon.")

    if lcol:
        with st.expander(f"Forecast every {lvl_label.lower()} at once"):
            if st.button("Run for all", key="fc_all"):
                allt = c_forecast_all(f, lcol, target, horizon, model, scenario)
                st.session_state["fc_all_tbl"] = allt
            if "fc_all_tbl" in st.session_state and lcol in st.session_state["fc_all_tbl"].columns:
                allt = st.session_state["fc_all_tbl"]
                table(allt.round(1))
                st.download_button("Download all forecasts (CSV)", csv_bytes(allt), "forecast_all_items.csv", "text/csv")

# ----------------------------------------------------------------------------
# ELASTICITY & DRIVERS
# ----------------------------------------------------------------------------
with tab_el:
    sub_a, sub_b = st.tabs(["Price elasticity", "What drives sales?"])

    with sub_a:
        st.markdown("Own-price elasticity from a **log-log demand model** with product x country fixed effects, controlling "
                    "for promo mechanic, holiday, trade investment, trend and seasonality. "
                    "Elasticity of **-1.5** means a 10% price cut lifts units by roughly 15%.")
        el_lvl = st.selectbox("Level", [l for l in E.LEVELS if l != "Total"], index=1, key="el_level")
        ecol = E.LEVELS[el_lvl]
        et = c_elasticity(f, ecol)
        if et.empty:
            st.info("Not enough rows / price variation at this level for the current filters. Widen the filters or pick a higher level.")
        else:
            et = et.copy()
            et["Significant (95%)"] = np.where(et["p-value"] < 0.05, "Yes", "No")
            fig = go.Figure(go.Bar(
                x=et["Elasticity"], y=et[ecol], orientation="h",
                error_x=dict(type="data", symmetric=False, array=et["CI high"] - et["Elasticity"], arrayminus=et["Elasticity"] - et["CI low"]),
                marker_color=np.where(et["p-value"] < 0.05, "#1f77b4", "#bbbbbb")))
            fig.update_layout(title=f"Price elasticity by {el_lvl.lower()} (bars = 95% CI; grey = not significant)",
                              xaxis_title="Elasticity", height=max(320, 40 * len(et) + 120), yaxis=dict(autorange="reversed"))
            show(fig)
            table(et.round(3))
            st.download_button("Download elasticities (CSV)", csv_bytes(et), "elasticities.csv", "text/csv")

            st.subheader("Discount simulator")
            pick = st.selectbox(f"Choose {el_lvl.lower()}", et[ecol].tolist(), key="el_pick")
            e_val = float(et.loc[et[ecol] == pick, "Elasticity"].iloc[0])
            s1, s2 = st.columns(2)
            cost_ratio = s1.slider("Unit cost as % of list price", 20, 80, 43) / 100
            mech = s2.slider("Non-price promo lift (display, etc.) %", 0, 30, 0) / 100 + 1
            rc = E.response_curve(e_val, np.arange(0, 41, 2.5), mech, cost_ratio)
            fig = px.line(rc.melt("Discount %", var_name="Index", value_name="Value (no-promo = 100)"),
                          x="Discount %", y="Value (no-promo = 100)", color="Index", markers=True,
                          title=f"{pick}: elasticity {e_val:.2f}")
            fig.add_hline(y=100, line_dash="dot", line_color="grey")
            show(fig)
            best = rc.loc[rc["Gross profit index"].idxmax()]
            if best["Discount %"] == 0:
                st.warning("At this elasticity and cost structure, no discount depth increases gross profit - price promotion destroys margin.")
            else:
                st.success(f"Gross-profit-maximising discount is about **{best['Discount %']:.1f}%** "
                           f"(units index {best['Units index']:.0f}, gross profit index {best['Gross profit index']:.0f}).")

    with sub_b:
        if len(f) < 400:
            st.info("Select a broader slice of data (>= 400 rows) to fit the driver models.")
        else:
            dm = c_demand_model(f)
            coef = dm["coef"].set_index("term")
            st.markdown(f"**Driver model** (log units, fixed effects) - within R² = {dm['r2']:.2f}, rows = {dm['n']:,}.")
            rows = []
            for t, r in coef.iterrows():
                if t.startswith(("promo:", "holiday:")) or t == "trade_pct":
                    rows.append({"Driver": t.replace("promo: ", "Promo - ").replace("holiday: ", "Holiday - ").replace("trade_pct", "Trade investment (per +1pp of sales)"),
                                 "Effect on units %": (np.exp(r["coef"]) - 1) * 100,
                                 "CI low %": (np.exp(r["ci_low"]) - 1) * 100, "CI high %": (np.exp(r["ci_high"]) - 1) * 100,
                                 "p-value": r["p_value"], "Significant": "Yes" if r["p_value"] < 0.05 else "No"})
            dtab = pd.DataFrame(rows).sort_values("Effect on units %", ascending=False)
            fig = go.Figure(go.Bar(x=dtab["Effect on units %"], y=dtab["Driver"], orientation="h",
                                   error_x=dict(type="data", symmetric=False, array=dtab["CI high %"] - dtab["Effect on units %"],
                                                arrayminus=dtab["Effect on units %"] - dtab["CI low %"]),
                                   marker_color=np.where(dtab["p-value"] < 0.05, "#2ca02c", "#bbbbbb")))
            fig.update_layout(title="Estimated lift in units vs. no promo / no holiday / no trade (95% CI)",
                              xaxis_title="% change in units", yaxis=dict(autorange="reversed"), height=max(340, 36 * len(dtab) + 120))
            show(fig)
            st.caption("Price discount effect is shown through the elasticity tab. Promo mechanics are measured versus "
                       "'Price Discount (TPR)', whose effect is carried by the price term.")
            table(dtab.round(3))

            st.subheader("Where did the incremental units come from?")
            dd = dm["decomp"]
            wf = go.Figure(go.Waterfall(
                x=dd["Driver"].tolist() + ["Actual units"], y=dd["Units"].tolist() + [0],
                measure=["absolute"] + ["relative"] * (len(dd) - 1) + ["total"],
                text=[f"{v:,.0f}" for v in dd["Units"]] + [f"{dd['Units'].sum():,.0f}"]))
            wf.update_layout(title="Units bridge: estimated baseline to actual", showlegend=False)
            show(wf)

            st.subheader("Which features matter most? (permutation importance)")
            fi, r2h = c_importance(f)
            groups = {"Price index": "Commercial lever", "Discount %": "Commercial lever", "Promo flag": "Commercial lever",
                      "Promo type": "Commercial lever", "Holiday flag": "Commercial lever", "Holiday name": "Commercial lever",
                      "Trade flag": "Commercial lever", "Trade % of sales": "Commercial lever"}
            fi["Group"] = fi["Feature"].map(groups).fillna("Structural / context")
            figi = px.bar(fi.sort_values("Importance %"), x="Importance %", y="Feature", color="Group", orientation="h",
                          title=f"Share of predictive impact (hold-out R² = {r2h:.2f})")
            show(figi)
            lev = fi[fi["Group"] == "Commercial lever"].copy()
            if lev["Importance"].clip(lower=0).sum() > 0:
                lev["Share among levers %"] = lev["Importance"].clip(lower=0) / lev["Importance"].clip(lower=0).sum() * 100
                st.markdown("**Among commercial levers only**")
                table(lev[["Feature", "Importance", "Share among levers %"]].round(3))
            st.caption("Permutation importance = how much prediction error rises when a feature is shuffled "
                       "(Gradient-Boosting model trained on earlier months, scored on the latest 4 months). "
                       "Structural features such as store size dominate raw volume; the levers table isolates what you can control.")

# ----------------------------------------------------------------------------
# PROMO & TRADE ROI
# ----------------------------------------------------------------------------
with tab_roi:
    st.markdown("**Trade ROI = (incremental gross profit - trade investment) / trade investment.** Incremental = actual minus "
                "the no-promo / no-trade baseline. Gross profit already reflects the cost of discounts.")
    kk = E.overall_kpis(f)
    c = st.columns(5)
    c[0].metric("Incremental units", num(f["Incremental_Units"].sum()))
    c[1].metric("Incremental sales value", money(f["Incremental_USD_Value"].sum()))
    c[2].metric("Incremental gross profit", money(kk["inc_gp"]))
    c[3].metric("Trade investment", money(kk["trade_usd"]))
    c[4].metric("Trade ROI", "-" if np.isnan(kk["trade_roi"]) else f"{kk['trade_roi']:.0f}%")

    st.subheader("Yes / No comparisons")
    y1, y2, y3 = st.columns(3)
    for colw, lab in ((y1, "Promo_Label"), (y2, "Holiday_Label"), (y3, "Trade_Label")):
        t = f.groupby(lab).agg(Rows=("Sales_Units", "size"), Avg_units=("Sales_Units", "mean"), Avg_sales_USD=("USD_Value", "mean"),
                               Avg_gross_profit_USD=("Gross_Profit_USD", "mean"),
                               Incremental_units=("Incremental_Units", "sum"), Trade_USD=("Trade_Investment_USD", "sum")).reset_index()
        with colw:
            table(t.round(1))

    dims = {"Promo type": "Promo_Type", "Promo (Yes/No)": "Promo_Label", "Holiday": "Holiday_Name", "Holiday (Yes/No)": "Holiday_Label",
            "Trade investment (Yes/No)": "Trade_Label", "Retailer": "Retailer", "Country": "Country", "Category": "Category",
            "Subcategory": "Subcategory", "Brand": "Brand", "Sub-brand": "Sub_Brand", "Product": "Product_Description", "Month": "Month"}
    by_label = st.selectbox("Analyse ROI by", list(dims), key="roi_by")
    bycol = dims[by_label]
    rt = E.roi_table(f, bycol)
    show_cols = [bycol, "Rows", "Sales_USD", "Units", "Volume lift %", "Value lift %", "Incremental_Units", "Incremental_USD",
                 "Incremental_GP", "Trade_USD", "Trade ROI %", "Avg_Discount"]
    table(rt[show_cols].round(1))
    st.download_button("Download ROI table (CSV)", csv_bytes(rt[show_cols]), "roi_table.csv", "text/csv")

    g1, g2 = st.columns(2)
    rt_plot = rt[rt["Trade_USD"] > 0].sort_values("Trade ROI %")
    if len(rt_plot):
        fig = px.bar(rt_plot, x="Trade ROI %", y=bycol, orientation="h", title=f"Trade ROI % by {by_label.lower()}",
                     color="Trade ROI %", color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
        with g1:
            show(fig)
    fig = px.bar(rt.sort_values("Volume lift %"), x="Volume lift %", y=bycol, orientation="h", title=f"Volume lift % by {by_label.lower()}")
    with g2:
        show(fig)

    st.subheader("Promo mechanic x retailer: trade ROI heat-map")
    tg = f[f["Trade_Investment_USD"] > 0].groupby(["Promo_Type", "Retailer"])[["Incremental_Gross_Profit_USD", "Trade_Investment_USD"]].sum()
    hm = ((tg["Incremental_Gross_Profit_USD"] - tg["Trade_Investment_USD"]) / tg["Trade_Investment_USD"] * 100).unstack("Retailer")
    if hm.size:
        fig = px.imshow(hm.round(0), text_auto=True, color_continuous_scale="RdYlGn", color_continuous_midpoint=0, aspect="auto",
                        labels=dict(color="Trade ROI %"))
        show(fig)

    st.subheader("Trade amount vs incremental gross profit")
    sc = f[f["Trade_Investment_USD"] > 0]
    if len(sc):
        sc = sc.sample(min(3000, len(sc)), random_state=1)
        fig = px.scatter(sc, x="Trade_Investment_USD", y="Incremental_Gross_Profit_USD", color="Promo_Type", opacity=0.6,
                         hover_data=["Product_Description", "Retailer", "Month"], title="Each dot = one store-product-month with trade spend")
        fig.add_shape(type="line", x0=0, y0=0, x1=sc["Trade_Investment_USD"].max(), y1=sc["Trade_Investment_USD"].max(),
                      line=dict(dash="dot", color="grey"))
        show(fig)
        st.caption("Dots above the dotted line (incremental gross profit > trade spend) paid back.")

# ----------------------------------------------------------------------------
# INVENTORY
# ----------------------------------------------------------------------------
with tab_inv:
    st.markdown("Inventory is the **closing stock at month-end** for each store-product-month, valued at unit cost "
                "(`Inventory_Value_USD = Closing_Inventory_Units x Unit_Cost_USD`). Stock cover = closing units / monthly sales x 30.")
    latest = f[f["Date"] == f["Date"].max()]
    inv_v, inv_u = latest["Inventory_Value_USD"].sum(), latest["Closing_Inventory_Units"].sum()
    cover = inv_u / latest["Sales_Units"].sum() * 30
    over = latest.loc[latest["Stock_Status"] == "Overstock", "Inventory_Value_USD"].sum()
    c = st.columns(5)
    c[0].metric("Inventory value (latest month)", money(inv_v))
    c[1].metric("Inventory units", num(inv_u))
    c[2].metric("Stock cover (days)", f"{cover:.0f}")
    c[3].metric("Overstock value", money(over), f"{over / inv_v * 100:.0f}% of stock" if inv_v else None, delta_color="off")
    c[4].metric("Low-stock observations", f"{(latest['Stock_Status'] == 'Low stock').mean() * 100:.1f}%")

    mon = f.groupby("Date").agg(inv=("Inventory_Value_USD", "sum"), inv_u=("Closing_Inventory_Units", "sum"),
                                units=("Sales_Units", "sum"), sales=("USD_Value", "sum")).reset_index()
    mon["Cover days"] = mon["inv_u"] / mon["units"] * 30
    i1, i2 = st.columns(2)
    with i1:
        show(px.line(mon, x="Date", y="inv", markers=True, title="Inventory value (USD)", labels={"inv": "USD"}))
    with i2:
        show(px.line(mon, x="Date", y="Cover days", markers=True, title="Stock cover (days)"))

    il = st.selectbox("Inventory by", [l for l in E.LEVELS if l != "Total"] + ["Country", "Retailer"], index=1, key="inv_lvl")
    icol = E.LEVELS.get(il, il)
    it = E.inventory_summary(latest, icol).sort_values("Inventory_USD", ascending=False)
    fig = px.bar(it, x=icol, y="Inventory_USD", color="Cover days", color_continuous_scale="RdYlGn_r",
                 title=f"Latest-month inventory value by {il.lower()} (colour = cover days)")
    show(fig)
    table(it.round(1))
    st.download_button("Download inventory table (CSV)", csv_bytes(it), "inventory_by_level.csv", "text/csv")

    ss = f.groupby(["Month", "Stock_Status"]).size().reset_index(name="n")
    fig = px.bar(ss, x="Month", y="n", color="Stock_Status", barmode="relative", title="Stock status mix by month",
                 color_discrete_map={"Healthy": "#2ca02c", "Overstock": "#ff7f0e", "Low stock": "#d62728"})
    fig.update_layout(barnorm="percent", yaxis_title="% of observations")
    show(fig)

    st.subheader("Biggest overstock positions (latest month)")
    ov = latest[latest["Stock_Status"] == "Overstock"].nlargest(15, "Inventory_Value_USD")[
        ["Retailer", "Store_City", "Product_Description", "Closing_Inventory_Units", "Inventory_Value_USD", "Stock_Cover_Days", "Promo_Type", "Holiday_Name"]]
    table(ov.round(1))

# ----------------------------------------------------------------------------
# AI ANALYST (Groq)
# ----------------------------------------------------------------------------
with tab_ai:
    st.markdown("The LLM reads a compact **fact sheet** computed from your current filters (KPIs, forecasts, elasticities, "
                "driver effects, ROI, inventory). It never receives the raw rows.")
    api_key = get_api_key()
    if not api_key:
        st.warning("Add your Groq API key in the sidebar (or `GROQ_API_KEY` in `.streamlit/secrets.toml`) to enable this tab.")
    if st.checkbox("Show the fact sheet sent to the LLM", value=False):
        st.json(c_fact_sheet(f))

    if st.button("📝 Generate executive summary", disabled=not api_key):
        with st.spinner("Asking Groq..."):
            facts = c_fact_sheet(f)
            msgs = llm.build_messages(facts, [], llm.EXEC_SUMMARY_PROMPT)
            st.session_state["exec_summary"] = llm.ask(api_key, model_name, msgs, max_tokens=1500)
    if st.session_state.get("exec_summary"):
        st.markdown(st.session_state["exec_summary"])

    st.divider()
    st.subheader("Ask a question")
    st.caption("Examples: *Which brand should I cut promotions on?* - *Why is inventory high for deodorants?* - "
               "*What discount depth is best for body wash?*")
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Ask about your forecasts, elasticity, ROI or inventory...", disabled=not api_key)
    if q:
        st.session_state["chat"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                facts = c_fact_sheet(f)
                msgs = llm.build_messages(facts, st.session_state["chat"][:-1], q)
                ans = llm.ask(api_key, model_name, msgs)
            st.markdown(ans)
        st.session_state["chat"].append({"role": "assistant", "content": ans})
    if st.session_state["chat"] and st.button("Clear chat"):
        st.session_state["chat"] = []
        st.rerun()
