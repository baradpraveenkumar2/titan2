"""
engine.py  -  analytics core for the Streamlit app (no Streamlit imports here,
so everything can be unit-tested / reused in notebooks).

Contents
  * data loading + level helpers
  * monthly aggregation at any hierarchy level
  * forecasting models (Seasonal-naive, Holt-Winters, Ridge-with-drivers, Gradient Boosting)
    + rolling backtest + prediction intervals
  * elasticity / driver regression (log-log with fixed effects)
  * driver decomposition, permutation feature importance
  * promo & trade ROI tables
  * inventory helpers
Only numpy / pandas / scipy / scikit-learn are required.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from holidays_calendar import holiday_for, NONE, ALL_HOLIDAYS

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Levels & metrics
# ----------------------------------------------------------------------------
LEVELS = {
    "Total": None,
    "Category": "Category",
    "Subcategory": "Subcategory",
    "Brand": "Brand",
    "Sub-brand": "Sub_Brand",
    "Product": "Product_Description",
}
METRICS = {
    "Sales value (USD)": "USD_Value",
    "Sales units": "Sales_Units",
    "Inventory value (USD)": "Inventory_Value_USD",
    "Inventory units": "Closing_Inventory_Units",
}
STOCK_METRICS = {"Inventory_Value_USD", "Closing_Inventory_Units"}   # levels (use average), not flows (use sum)
PROMO_TYPES = ["Price Discount (TPR)", "Multibuy", "Display / Feature", "Bundle / Gift", "Loyalty / Coupon"]

# exogenous drivers available for the forecast (all monthly, at the selected level)
EXOG = ["promo_share", "discount_depth", "holiday_share", "trade_intensity", "price_index"]
EXOG_LABELS = {
    "promo_share": "Promo share (% of observations on promo)",
    "discount_depth": "Avg discount depth (%)",
    "holiday_share": "Holiday share (% of observations in a holiday month)",
    "trade_intensity": "Trade investment (% of baseline sales)",
    "price_index": "Avg price index (net / list price)",
}


def load_data(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Holiday_Name"] = df["Holiday_Name"].fillna(NONE)
    return df


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def aggregate_monthly(df: pd.DataFrame, level_col: str | None, entity: str | None = None) -> pd.DataFrame:
    """Monthly series (one row per month) for one entity of a level (or the whole filtered data)."""
    d = df if level_col is None or entity is None else df[df[level_col] == entity]
    g = d.groupby("Date")
    out = pd.DataFrame({
        "USD_Value": g["USD_Value"].sum(),
        "Sales_Units": g["Sales_Units"].sum(),
        "Inventory_Value_USD": g["Inventory_Value_USD"].sum(),
        "Closing_Inventory_Units": g["Closing_Inventory_Units"].sum(),
        "Baseline_USD_Value": g["Baseline_USD_Value"].sum(),
        "Trade_Investment_USD": g["Trade_Investment_USD"].sum(),
        "Incremental_Gross_Profit_USD": g["Incremental_Gross_Profit_USD"].sum(),
        "promo_share": g["Promo_Flag"].mean(),
        "discount_depth": g["Discount_Pct"].mean(),
        "holiday_share": g["Holiday_Flag"].mean(),
        "price_index": g["Price_Index"].mean(),
        "n_obs": g.size(),
    })
    out["trade_intensity"] = out["Trade_Investment_USD"] / out["Baseline_USD_Value"].replace(0, np.nan) * 100
    out["trade_intensity"] = out["trade_intensity"].fillna(0)
    out["Sales_USD_per_obs"] = out["USD_Value"] / out["n_obs"]
    return out.sort_index()


def country_weights(df: pd.DataFrame) -> dict:
    return (df["Country"].value_counts(normalize=True)).to_dict()


def future_holiday_share(weights: dict, month: pd.Timestamp) -> float:
    m = month.strftime("%Y-%m")
    return float(sum(w for c, w in weights.items() if holiday_for(c, m) != NONE))


# ----------------------------------------------------------------------------
# Forecasting models
# ----------------------------------------------------------------------------
MODEL_NAMES = [
    "Auto (best on backtest)",
    "Seasonal naive + growth",
    "Holt-Winters (damped, seasonal)",
    "Regression with drivers (Ridge)",
    "Gradient Boosting with drivers",
    "Ensemble (average of all)",
]


def _time_features(dates: pd.DatetimeIndex, t0: pd.Timestamp) -> pd.DataFrame:
    t = ((dates.year - t0.year) * 12 + (dates.month - t0.month)).astype(float)
    m = dates.month.to_numpy()
    return pd.DataFrame({
        "t": t,
        "s1": np.sin(2 * np.pi * m / 12), "c1": np.cos(2 * np.pi * m / 12),
        "s2": np.sin(4 * np.pi * m / 12), "c2": np.cos(4 * np.pi * m / 12),
    }, index=dates)


def _fc_seasonal_naive(y: pd.Series, h: int) -> np.ndarray:
    """Same month last year x recent YoY growth (falls back to last value)."""
    v = y.to_numpy(float)
    n = len(v)
    if n < 13:
        return np.repeat(v[-1], h)
    last12, prev12 = v[-12:].sum(), v[-24:-12].sum() if n >= 24 else np.nan
    g = (last12 / prev12) if (prev12 and not np.isnan(prev12) and prev12 > 0) else 1.0
    g = float(np.clip(g, 0.7, 1.4))
    out = []
    for i in range(h):
        out.append(v[n - 12 + (i % 12)] * (g if i < 12 else g ** 2))
    return np.array(out)


def _hw_run(params, v, m, init, h=0):
    a, b, g, phi = params
    L, T, S = init
    S = list(S)
    fitted = np.zeros(len(v))
    for i, x in enumerate(v):
        s = S[i % m]
        fitted[i] = L + phi * T + s
        L_new = a * (x - s) + (1 - a) * (L + phi * T)
        T = b * (L_new - L) + (1 - b) * phi * T
        S[i % m] = g * (x - L_new) + (1 - g) * s
        L = L_new
    fc = []
    n = len(v)
    for k in range(1, h + 1):
        damp = sum(phi ** j for j in range(1, k + 1))
        fc.append(L + damp * T + S[(n + k - 1) % m])
    return fitted, np.array(fc)


def _fc_holt_winters(y: pd.Series, h: int) -> np.ndarray:
    v = y.to_numpy(float)
    n, m = len(v), 12
    if n < 2 * m - 3:   # not enough history for seasonal init -> damped trend only
        m = 1
    if m == 12:
        L0 = v[:12].mean()
        S0 = v[:12] - L0
        T0 = (v[12:24].mean() - v[:12].mean()) / 12 if n >= 24 else 0.0
    else:
        L0, S0, T0 = v[0], np.zeros(1), (v[-1] - v[0]) / max(n - 1, 1)
    init = (L0, T0, S0)

    def sse(p):
        f, _ = _hw_run(p, v, m, init)
        return float(np.sum((v - f) ** 2))

    best = minimize(sse, x0=[0.3, 0.05, 0.1, 0.9], bounds=[(0.01, 0.99), (0.0, 0.5), (0.0, 0.99), (0.8, 0.98)],
                    method="L-BFGS-B")
    _, fc = _hw_run(best.x, v, m, init, h)
    return fc


def _prep_xy(hist: pd.DataFrame, target: str, exog: list[str], t0):
    tf = _time_features(hist.index, t0)
    X = pd.concat([tf, hist[exog]], axis=1)
    y = np.log1p(hist[target].to_numpy(float))
    return X, y


def _fc_ridge(hist, target, exog, future, alpha=3.0):
    t0 = hist.index[0]
    X, y = _prep_xy(hist, target, exog, t0)
    Xf = pd.concat([_time_features(future.index, t0), future[exog]], axis=1)
    sc = StandardScaler().fit(X)
    mdl = Ridge(alpha=alpha).fit(sc.transform(X), y)
    return np.expm1(mdl.predict(sc.transform(Xf)))


def _fc_gbm(hist, target, exog, future):
    """Linear trend on log scale + gradient boosting on the residual using seasonality & drivers."""
    t0 = hist.index[0]
    X, y = _prep_xy(hist, target, exog, t0)
    Xf = pd.concat([_time_features(future.index, t0), future[exog]], axis=1)
    trend = Ridge(alpha=1e-3).fit(X[["t"]], y)
    resid = y - trend.predict(X[["t"]])
    cols = [c for c in X.columns if c != "t"]
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                   subsample=0.8, random_state=0).fit(X[cols], resid)
    return np.expm1(trend.predict(Xf[["t"]]) + gb.predict(Xf[cols]))


def _run_model(name, hist, target, exog, future):
    h = len(future)
    if name == "Seasonal naive + growth":
        return _fc_seasonal_naive(hist[target], h)
    if name == "Holt-Winters (damped, seasonal)":
        return _fc_holt_winters(hist[target], h)
    if name == "Regression with drivers (Ridge)":
        return _fc_ridge(hist, target, exog, future)
    if name == "Gradient Boosting with drivers":
        return _fc_gbm(hist, target, exog, future)
    raise ValueError(name)


BASE_MODELS = MODEL_NAMES[1:5]


def backtest(hist: pd.DataFrame, target: str, exog: list[str], holdout: int = 3) -> pd.DataFrame:
    """Single-origin backtest: train on all but the last `holdout` months, score on those months."""
    train, test = hist.iloc[:-holdout], hist.iloc[-holdout:]
    rows, preds = [], {}
    for name in BASE_MODELS:
        try:
            p = np.clip(_run_model(name, train, target, exog, test), 0, None)
        except Exception:
            continue
        preds[name] = p
        a = test[target].to_numpy(float)
        rows.append({"Model": name,
                     "WAPE %": float(np.abs(a - p).sum() / max(a.sum(), 1e-9) * 100),
                     "MAPE %": float(np.mean(np.abs((a - p) / np.where(a == 0, np.nan, a))) * 100),
                     "Bias %": float((p.sum() - a.sum()) / max(a.sum(), 1e-9) * 100)})
    if preds:
        ens = np.mean(list(preds.values()), axis=0)
        a = test[target].to_numpy(float)
        rows.append({"Model": "Ensemble (average of all)",
                     "WAPE %": float(np.abs(a - ens).sum() / max(a.sum(), 1e-9) * 100),
                     "MAPE %": float(np.mean(np.abs((a - ens) / np.where(a == 0, np.nan, a))) * 100),
                     "Bias %": float((ens.sum() - a.sum()) / max(a.sum(), 1e-9) * 100)})
    return pd.DataFrame(rows).sort_values("WAPE %").reset_index(drop=True)


@dataclass
class ForecastResult:
    forecast: pd.DataFrame        # index = future months; columns: forecast, lower, upper
    model_used: str
    backtest: pd.DataFrame
    future_exog: pd.DataFrame


def build_future_exog(hist: pd.DataFrame, h: int, weights: dict, scenario: dict | None = None) -> pd.DataFrame:
    """Future driver values = same month last year (baseline plan), holiday from the calendar,
    optionally overridden by scenario values (None -> keep baseline)."""
    scenario = scenario or {}
    last = hist.index[-1]
    idx = pd.date_range(last + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    rows = []
    for d in idx:
        ly = d - pd.DateOffset(years=1)
        ref = hist.loc[ly] if ly in hist.index else hist.iloc[-12:].mean()
        r = {c: float(ref[c]) for c in ["promo_share", "discount_depth", "trade_intensity", "price_index"]}
        r["holiday_share"] = future_holiday_share(weights, d)
        rows.append(r)
    fx = pd.DataFrame(rows, index=idx)
    if scenario.get("promo_share") is not None:
        fx["promo_share"] = scenario["promo_share"]
    if scenario.get("discount_depth") is not None:
        fx["discount_depth"] = scenario["discount_depth"]
    if scenario.get("trade_intensity") is not None:
        fx["trade_intensity"] = scenario["trade_intensity"]
    if scenario.get("price_change_pct"):
        fx["price_index"] = fx["price_index"] * (1 + scenario["price_change_pct"] / 100)
    return fx


def forecast_series(hist: pd.DataFrame, target: str, h: int, model: str, weights: dict,
                    scenario: dict | None = None, exog: list[str] | None = None,
                    holdout: int = 3, interval: float = 0.80) -> ForecastResult:
    exog = exog or EXOG
    future = build_future_exog(hist, h, weights, scenario)
    bt = backtest(hist, target, exog, holdout)
    chosen = model
    if model.startswith("Auto"):
        chosen = bt.iloc[0]["Model"] if len(bt) else "Seasonal naive + growth"
    if chosen == "Ensemble (average of all)":
        p = np.mean([_run_model(m, hist, target, exog, future) for m in BASE_MODELS], axis=0)
    else:
        p = _run_model(chosen, hist, target, exog, future)
    p = np.clip(p, 0, None)
    # prediction interval: backtest WAPE-based relative error, widening with horizon
    wape = float(bt.loc[bt["Model"] == chosen, "WAPE %"].iloc[0]) / 100 if (len(bt) and (bt["Model"] == chosen).any()) else 0.10
    z = stats.norm.ppf(0.5 + interval / 2)
    sigma_rel = max(wape * 1.25, 0.03)           # WAPE ~ 0.8 sigma for normal errors
    widen = np.sqrt(1 + np.arange(h) / 6)
    half = z * sigma_rel * p * widen
    out = pd.DataFrame({"forecast": p, "lower": np.clip(p - half, 0, None), "upper": p + half}, index=future.index)
    return ForecastResult(out, chosen, bt, future)


def forecast_all_entities(df: pd.DataFrame, level_col: str, target: str, h: int, model: str,
                          scenario: dict | None = None) -> pd.DataFrame:
    """Forecast every entity of a level (e.g. every brand) and return a tidy table."""
    rows = []
    for ent in sorted(df[level_col].unique()):
        sub = df[df[level_col] == ent]
        hist = aggregate_monthly(sub, None)
        w = country_weights(sub)
        try:
            res = forecast_series(hist, target, h, model, w, scenario)
        except Exception:
            continue
        agg = np.mean if target in STOCK_METRICS else np.sum
        last_n = agg(hist[target].iloc[-h:])
        nxt = agg(res.forecast["forecast"])
        rows.append({
            level_col: ent, "Model": res.model_used,
            f"Last {h}M actual": last_n,
            f"Next {h}M forecast": nxt,
            "Growth vs last period %": (nxt / last_n - 1) * 100 if last_n else np.nan,
            "Backtest WAPE %": float(res.backtest.loc[res.backtest["Model"] == res.model_used, "WAPE %"].iloc[0])
            if (res.backtest["Model"] == res.model_used).any() else np.nan,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Elasticity / driver regression
# ----------------------------------------------------------------------------
def _design(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Design matrix for the log-log demand model."""
    X = pd.DataFrame(index=d.index)
    X["ln_price_index"] = np.log(d["Price_Index"].clip(lower=0.2))
    for p in PROMO_TYPES[1:]:                      # TPR is the reference mechanic: its effect is carried by the price term
        X[f"promo: {p}"] = (d["Promo_Type"] == p).astype(float)
    for h in ALL_HOLIDAYS:
        X[f"holiday: {h}"] = (d["Holiday_Name"] == h).astype(float)
    X["trade_pct"] = d["Trade_Pct_of_Sales"]       # per 1 pp of sales invested
    t = (d["Date"].dt.year - 2024) * 12 + d["Date"].dt.month
    X["trend"] = t.astype(float)
    m = d["Date"].dt.month
    for k in (1, 2):
        X[f"sin{k}"] = np.sin(2 * np.pi * k * m / 12)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * m / 12)
    y = np.log(d["Sales_Units"].clip(lower=1))
    return X, y


def fit_demand_model(d: pd.DataFrame, fe_cols=("Product_Description", "Country")) -> dict:
    """Within-transformed OLS (fixed effects absorbed). Returns coefficients, SE, CI, p-values."""
    d = d.reset_index(drop=True)
    X, y = _design(d)
    grp = d[list(fe_cols)].astype(str).agg("|".join, axis=1)
    Xd = X - X.groupby(grp).transform("mean")
    yd = y - y.groupby(grp).transform("mean")
    keep = [c for c in Xd.columns if Xd[c].std() > 1e-9]       # drop constant columns (e.g. holiday not present)
    A = Xd[keep].to_numpy()
    beta, *_ = np.linalg.lstsq(A, yd.to_numpy(), rcond=None)
    resid = yd.to_numpy() - A @ beta
    n, k = A.shape
    n_fe = grp.nunique()
    dof = max(n - k - n_fe, 1)
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    pval = 2 * (1 - stats.t.cdf(np.abs(tval), dof))
    ss_tot = (yd @ yd)
    coef = pd.DataFrame({"term": keep, "coef": beta, "se": se, "t": tval, "p_value": pval,
                         "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se})
    coef["effect_on_units_%"] = (np.exp(coef["coef"]) - 1) * 100       # for dummies
    return {"coef": coef, "r2_within": float(1 - (resid @ resid) / ss_tot), "n": n, "keep": keep, "X": X, "y": y,
            "beta": pd.Series(beta, index=keep)}


def elasticity_table(df: pd.DataFrame, level_col: str, min_rows: int = 150) -> pd.DataFrame:
    """Own-price elasticity (promo price) for every entity of a level."""
    rows = []
    for ent, sub in df.groupby(level_col):
        if len(sub) < min_rows or sub["Price_Index"].nunique() < 5:
            continue
        try:
            fm = fit_demand_model(sub)
        except Exception:
            continue
        c = fm["coef"].set_index("term")
        if "ln_price_index" not in c.index:
            continue
        r = c.loc["ln_price_index"]
        rows.append({level_col: ent, "Elasticity": r["coef"], "CI low": r["ci_low"], "CI high": r["ci_high"],
                     "p-value": r["p_value"], "Rows": len(sub),
                     "Type": "Elastic (|e|>1)" if abs(r["coef"]) > 1 else "Inelastic (|e|<1)"})
    return pd.DataFrame(rows).sort_values("Elasticity").reset_index(drop=True)


def response_curve(elasticity: float, discounts=np.arange(0, 41, 5), mech_lift: float = 1.0,
                   cost_ratio: float = 0.43) -> pd.DataFrame:
    """Units / revenue / gross-profit index vs discount depth (list price = 1, baseline units = 1)."""
    p = 1 - discounts / 100
    units = p ** elasticity * mech_lift
    rev = units * p
    gp = units * (p - cost_ratio)
    base_gp = 1 - cost_ratio
    return pd.DataFrame({"Discount %": discounts, "Units index": units * 100,
                         "Revenue index": rev * 100, "Gross profit index": gp / base_gp * 100})


def driver_decomposition(d: pd.DataFrame, fm: dict) -> pd.DataFrame:
    """Split incremental units vs a no-promo/no-holiday/no-trade counterfactual into drivers,
    using the fitted log-linear coefficients (allocation proportional to log contribution)."""
    beta, X = fm["beta"], fm["X"].loc[d.index] if not d.index.equals(fm["X"].index) else fm["X"]
    groups = {
        "Price discount": [c for c in beta.index if c == "ln_price_index"],
        "Promo mechanic": [c for c in beta.index if c.startswith("promo:")],
        "Holiday / event": [c for c in beta.index if c.startswith("holiday:")],
        "Trade investment": [c for c in beta.index if c == "trade_pct"],
    }
    contrib = pd.DataFrame({g: (X[cols] * beta[cols]).sum(axis=1) if cols else 0.0 for g, cols in groups.items()})
    total_log = contrib.sum(axis=1)
    actual = d["Sales_Units"].astype(float)
    base_hat = actual / np.exp(total_log)
    inc = actual - base_hat
    denom = total_log.replace(0, np.nan)
    share = contrib.div(denom, axis=0).fillna(0)
    units = share.mul(inc, axis=0).sum()
    out = pd.DataFrame({"Driver": ["Estimated baseline"] + list(units.index),
                        "Units": [base_hat.sum()] + list(units.values)})
    out["% of actual"] = out["Units"] / actual.sum() * 100
    return out


def feature_importance(d: pd.DataFrame, target: str = "Sales_Units", test_months: int = 4,
                       n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance of business drivers using a Gradient-Boosting model (time-based holdout).
    Importance = increase in MAE when the feature is shuffled (grouped features shuffled together)."""
    d = d.sort_values("Date").reset_index(drop=True)
    feats = pd.DataFrame({
        "Price index": d["Price_Index"], "Discount %": d["Discount_Pct"],
        "Promo flag": d["Promo_Flag"], "Promo type": d["Promo_Type"].astype("category").cat.codes,
        "Holiday flag": d["Holiday_Flag"], "Holiday name": d["Holiday_Name"].astype("category").cat.codes,
        "Trade flag": d["Trade_Flag"], "Trade % of sales": d["Trade_Pct_of_Sales"],
        "Store area": d["Store_Area_sqm"], "Product": d["Product_Description"].astype("category").cat.codes,
        "Country": d["Country"].astype("category").cat.codes, "Retailer": d["Retailer"].astype("category").cat.codes,
        "Month of year": d["Date"].dt.month, "Time trend": (d["Date"].dt.year - 2024) * 12 + d["Date"].dt.month,
    })
    y = np.log(d[target].clip(lower=1))
    cut = d["Date"].sort_values().unique()[-test_months]
    tr, te = d["Date"] < cut, d["Date"] >= cut
    mdl = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.08, max_iter=250, random_state=0)
    mdl.fit(feats[tr], y[tr])
    pi = permutation_importance(mdl, feats[te], y[te], n_repeats=n_repeats, random_state=0,
                                scoring="neg_mean_absolute_error")
    out = pd.DataFrame({"Feature": feats.columns, "Importance": pi.importances_mean,
                        "Std": pi.importances_std}).sort_values("Importance", ascending=False)
    out["Importance %"] = out["Importance"].clip(lower=0) / out["Importance"].clip(lower=0).sum() * 100
    pred = mdl.predict(feats[te])
    r2 = 1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    out.attrs["r2_holdout"] = float(r2)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Promo & trade ROI
# ----------------------------------------------------------------------------
def roi_table(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    g = df.groupby(by)
    t = g.agg(Rows=("Sales_Units", "size"), Sales_USD=("USD_Value", "sum"), Baseline_USD=("Baseline_USD_Value", "sum"),
              Units=("Sales_Units", "sum"), Baseline_Units=("Baseline_Units", "sum"),
              Incremental_Units=("Incremental_Units", "sum"), Incremental_USD=("Incremental_USD_Value", "sum"),
              Incremental_GP=("Incremental_Gross_Profit_USD", "sum"), Trade_USD=("Trade_Investment_USD", "sum"),
              Avg_Discount=("Discount_Pct", "mean")).reset_index()
    t["Volume lift %"] = t["Incremental_Units"] / t["Baseline_Units"].replace(0, np.nan) * 100
    t["Value lift %"] = t["Incremental_USD"] / t["Baseline_USD"].replace(0, np.nan) * 100
    t["Trade ROI %"] = np.where(t["Trade_USD"] > 0, (t["Incremental_GP"] - t["Trade_USD"]) / t["Trade_USD"].replace(0, np.nan) * 100, np.nan)
    t["Incremental GP per $ trade"] = np.where(t["Trade_USD"] > 0, t["Incremental_GP"] / t["Trade_USD"].replace(0, np.nan), np.nan)
    return t


def overall_kpis(df: pd.DataFrame) -> dict:
    tr = df["Trade_Investment_USD"].sum()
    inc_gp = df["Incremental_Gross_Profit_USD"].sum()
    trd = df[df["Trade_Investment_USD"] > 0]
    roi = (trd["Incremental_Gross_Profit_USD"].sum() - trd["Trade_Investment_USD"].sum()) / trd["Trade_Investment_USD"].sum() * 100 \
        if trd["Trade_Investment_USD"].sum() > 0 else np.nan
    return {
        "sales_usd": df["USD_Value"].sum(), "units": df["Sales_Units"].sum(),
        "inventory_usd": df[df["Date"] == df["Date"].max()]["Inventory_Value_USD"].sum(),
        "promo_share": df["Promo_Flag"].mean() * 100, "trade_usd": tr, "inc_gp": inc_gp,
        "trade_roi": roi, "avg_discount_on_promo": df.loc[df["Promo_Flag"] == 1, "Discount_Pct"].mean(),
    }


# ----------------------------------------------------------------------------
# Inventory
# ----------------------------------------------------------------------------
def inventory_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by)
    t = g.agg(Inventory_USD=("Inventory_Value_USD", "sum"), Inventory_Units=("Closing_Inventory_Units", "sum"),
              Sales_Units=("Sales_Units", "sum"), Sales_USD=("USD_Value", "sum"),
              Overstock_rows=("Stock_Status", lambda s: (s == "Overstock").mean() * 100),
              Low_stock_rows=("Stock_Status", lambda s: (s == "Low stock").mean() * 100)).reset_index()
    t["Cover days"] = t["Inventory_Units"] / t["Sales_Units"] * 30
    t = t.rename(columns={"Overstock_rows": "Overstock %", "Low_stock_rows": "Low stock %"})
    return t
