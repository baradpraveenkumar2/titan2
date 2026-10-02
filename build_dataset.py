"""
build_dataset.py
----------------
Enriches the original synthetic retailer FMCG file with:

  * Inventory columns (closing units, unit cost, INVENTORY VALUE, cover days, status)
  * Promotion columns (flag, type, discount depth, net price)
  * Holiday columns (flag, name)
  * Trade investment columns (flag, amount, % of sales)
  * Counterfactual baseline + incremental units / value / gross profit + Trade ROI
  * Small data-quality fixes (see CLEANING below)

The original USD_Value / Sales_Units are treated as the *baseline* (no promo,
no holiday, no trade).  Promo / holiday / trade effects are layered on top with
known elasticities so that the elasticity & driver models in the app have a
real signal to find.  Everything is seeded -> fully reproducible.

Run:  python build_dataset.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from holidays_calendar import holiday_for, NONE

SRC = "data/retailer_fmcg_synthetic_dashboard_data.csv"
DST = "data/retailer_fmcg_enriched_data.csv"
SEED = 42

# ----------------------------------------------------------------------------
# Assumptions used to simulate the new columns (documented in README)
# ----------------------------------------------------------------------------
# True promo-price elasticity by subcategory (units % change per 1% price change)
PRICE_ELASTICITY = {
    "toothpaste": -1.1, "mouthwash": -1.5, "toothbrush": -1.3,
    "bar soap": -1.7, "deodorants": -1.9, "bodywash": -1.6, "fc liquids": -1.4,
}
# Gross-margin structure: unit cost as share of list price
COST_RATIO = {
    "toothpaste": 0.42, "mouthwash": 0.40, "toothbrush": 0.45,
    "bar soap": 0.46, "deodorants": 0.41, "bodywash": 0.43, "fc liquids": 0.52,
}
PROMO_TYPES = ["Price Discount (TPR)", "Multibuy", "Display / Feature",
               "Bundle / Gift", "Loyalty / Coupon"]
PROMO_TYPE_P = [0.38, 0.20, 0.17, 0.10, 0.15]
# non-price lift of the promo mechanic (on top of the price effect)
MECHANIC_LIFT = {
    "Price Discount (TPR)": 1.06, "Multibuy": 1.15, "Display / Feature": 1.20,
    "Bundle / Gift": 1.12, "Loyalty / Coupon": 1.06,
}
HOLIDAY_LIFT = {   # (HC, PC)
    "Ramadan": (1.14, 1.08), "Ramadan & Eid al-Fitr": (1.18, 1.15),
    "Eid al-Adha": (1.08, 1.10), "White Friday": (1.12, 1.16),
    "National Day": (1.04, 1.07), "Christmas / New Year": (1.05, 1.10),
}
RETAILER_PROMO_P = {"Carrefour": 0.30, "LuLu": 0.30, "SPAR": 0.22,
                    "Reliance": 0.20, "DMart": 0.18}
RETAILER_TRADE_EFF = {"LuLu": 1.15, "Carrefour": 1.00, "SPAR": 0.85,
                      "Reliance": 0.90, "DMart": 1.05}
TRADE_PCT_RANGE = {   # trade spend as % of baseline sales value
    "Price Discount (TPR)": (1, 5), "Multibuy": (2, 6), "Display / Feature": (3, 7),
    "Bundle / Gift": (3, 8), "Loyalty / Coupon": (1, 4), "No Promotion": (1, 3),
}
TRADE_LIFT_PER_PCT = 3.0   # +3 units-% per 1% of sales invested (before retailer effect)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """CLEANING: fix text corruption / typos found in the source file."""
    df = df.copy()
    # Lux soap pack text contained garbled non-Latin characters
    m = df["Product_Description"].str.startswith("Lux Soft Touch")
    df.loc[m, "Product_Description"] = "Lux Soft Touch 4 x 100 g"
    df.loc[df["Pack_Size"].str.contains("soap", case=False, na=False), "Pack_Size"] = "4 x 100 g"
    # 'Colgate Colgate Total 150 g' -> 'Colgate Total 150 g'
    df["Product_Description"] = df["Product_Description"].str.replace(
        "Colgate Colgate", "Colgate", regex=False)
    df["Subcategory"] = df["Subcategory"].replace({"deodrants": "deodorants"})
    return df


def main(src: str = SRC, dst: str = DST) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    df = clean(pd.read_csv(src))
    n = len(df)

    # ---------------- identifiers / helper columns ---------------------------
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month_Num"] = df["Date"].dt.month
    df["Store"] = df["Retailer"] + " - " + df["Store_City"]
    df["Sub_Brand"] = np.where(
        df.apply(lambda r: r["Sub_Brand_Variant"].startswith(r["Brand"]), axis=1),
        df["Sub_Brand_Variant"], df["Brand"] + " " + df["Sub_Brand_Variant"])

    fx = (df["LC_Value"] / df["USD_Value"]).groupby(df["Local_Currency"]).transform("median")
    df["FX_Rate_LC_per_USD"] = fx.round(4)

    # ---------------- baseline (original figures) ----------------------------
    base_units = df["Sales_Units"].astype(float)
    list_price = df["USD_Value"] / base_units
    df["Base_Units_Original"] = df["Sales_Units"]
    df["List_Unit_Price_USD"] = list_price.round(4)

    # ---------------- holiday ------------------------------------------------
    month_str = df["Date"].dt.strftime("%Y-%m")
    key = df["Country"] + "|" + month_str
    hmap = {k: holiday_for(*k.split("|")) for k in key.unique()}
    df["Holiday_Name"] = key.map(hmap)
    df["Holiday_Flag"] = (df["Holiday_Name"] != NONE).astype(int)
    is_pc = (df["Category"] == "PC").to_numpy()
    hol_mult = np.ones(n)
    for h, (hc, pc) in HOLIDAY_LIFT.items():
        mk = (df["Holiday_Name"] == h).to_numpy()
        hol_mult[mk] = np.where(is_pc[mk], pc, hc)
    baseline_units = np.maximum(1, np.round(base_units * hol_mult))

    # ---------------- promotion ----------------------------------------------
    p_promo = df["Retailer"].map(RETAILER_PROMO_P).to_numpy() * np.where(df["Holiday_Flag"] == 1, 1.35, 1.0)
    p_promo = np.clip(p_promo, 0, 0.7)
    promo_flag = (rng.random(n) < p_promo).astype(int)
    ptype = rng.choice(PROMO_TYPES, size=n, p=PROMO_TYPE_P)
    ptype = np.where(promo_flag == 1, ptype, "No Promotion")

    disc = np.zeros(n)
    u = rng.random(n)
    m = ptype == "Price Discount (TPR)"; disc[m] = 10 + 20 * u[m]
    m = ptype == "Multibuy";             disc[m] = rng.choice([20, 25, 33], size=m.sum())
    m = ptype == "Bundle / Gift";        disc[m] = 8 + 7 * u[m]
    m = ptype == "Loyalty / Coupon";     disc[m] = 5 + 10 * u[m]
    # Display / Feature has no price cut
    disc = np.round(disc, 1)
    df["Promo_Flag"] = promo_flag
    df["Promo_Type"] = ptype
    df["Discount_Pct"] = disc
    df["Net_Unit_Price_USD"] = (list_price * (1 - disc / 100)).round(4)
    df["Price_Index"] = (1 - disc / 100).round(4)       # net price / list price

    # ---------------- trade investment ---------------------------------------
    p_trade = np.where(promo_flag == 1, 0.65, 0.12)
    trade_flag = (rng.random(n) < p_trade).astype(int)
    lo = np.array([TRADE_PCT_RANGE[t][0] for t in ptype], dtype=float)
    hi = np.array([TRADE_PCT_RANGE[t][1] for t in ptype], dtype=float)
    trade_pct = np.where(trade_flag == 1, lo + (hi - lo) * rng.random(n), 0.0)
    baseline_value = baseline_units * list_price.to_numpy()
    df["Trade_Flag"] = trade_flag
    df["Trade_Pct_of_Sales"] = np.round(trade_pct, 2)
    df["Trade_Investment_USD"] = np.round(baseline_value * trade_pct / 100, 2)

    # ---------------- units / value with all effects -------------------------
    elast = df["Subcategory"].map(PRICE_ELASTICITY).to_numpy()
    price_eff = (1 - disc / 100) ** elast
    mech_eff = np.array([MECHANIC_LIFT.get(t, 1.0) for t in ptype])
    trade_eff = 1 + TRADE_LIFT_PER_PCT * (trade_pct / 100) * df["Retailer"].map(RETAILER_TRADE_EFF).to_numpy()
    has_effect = (promo_flag == 1) | (trade_flag == 1)
    noise = np.where(has_effect, np.exp(rng.normal(0, 0.05, n)), 1.0)   # no promo & no trade -> units == baseline
    units = np.maximum(1, np.round(baseline_units * price_eff * mech_eff * trade_eff * noise))

    df["Baseline_Units"] = baseline_units.astype(int)
    df["Sales_Units"] = units.astype(int)
    df["USD_Value"] = (units * df["Net_Unit_Price_USD"]).round(2)
    df["LC_Value"] = (df["USD_Value"] * df["FX_Rate_LC_per_USD"]).round(2)

    # ---------------- cost / gross profit / incrementals ---------------------
    ratio = df["Subcategory"].map(COST_RATIO)
    ref_price = list_price.groupby([df["Product_Description"], df["Country"]]).transform("median")
    df["Unit_Cost_USD"] = (ref_price * ratio).round(4)
    df["Baseline_USD_Value"] = (baseline_units * list_price).round(2)
    df["Gross_Profit_USD"] = (units * (df["Net_Unit_Price_USD"] - df["Unit_Cost_USD"])).round(2)
    df["Baseline_Gross_Profit_USD"] = (baseline_units * (list_price - df["Unit_Cost_USD"])).round(2)
    df["Incremental_Units"] = (df["Sales_Units"] - df["Baseline_Units"]).astype(int)
    df["Incremental_USD_Value"] = (df["USD_Value"] - df["Baseline_USD_Value"]).round(2)
    df["Incremental_Gross_Profit_USD"] = (df["Gross_Profit_USD"] - df["Baseline_Gross_Profit_USD"]).round(2)
    # Trade ROI = (incremental gross profit - trade spend) / trade spend (only rows with trade)
    tr = df["Trade_Investment_USD"]
    df["Trade_ROI_Pct"] = np.where(
        tr > 0, (df["Incremental_Gross_Profit_USD"] - tr) / tr.where(tr > 0, np.nan) * 100, np.nan).round(1)

    # ---------------- inventory ----------------------------------------------
    next_month = (df["Date"] + pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m")
    nkey = df["Country"] + "|" + next_month
    nmap = {k: holiday_for(*k.split("|")) for k in nkey.unique()}
    next_holiday = nkey.map(nmap) != NONE
    cover_m = np.where(is_pc, 1.10, 1.30) * np.exp(rng.normal(0, 0.38, n))
    cover_m = cover_m * np.where(next_holiday, 1.25, 1.0)          # pre-build ahead of a holiday
    cover_m = cover_m * np.where(promo_flag == 1, 0.85, 1.0)      # promo draws stock down
    stockout = rng.random(n) < 0.04
    cover_m = np.where(stockout, cover_m * 0.12, cover_m)
    closing = np.maximum(0, np.round(units * cover_m))
    df["Closing_Inventory_Units"] = closing.astype(int)
    df["Inventory_Value_USD"] = (closing * df["Unit_Cost_USD"]).round(2)
    df["Inventory_Value_LC"] = (df["Inventory_Value_USD"] * df["FX_Rate_LC_per_USD"]).round(2)
    df["Stock_Cover_Days"] = np.round(closing / units * 30, 1)
    df["Stock_Status"] = np.select(
        [df["Stock_Cover_Days"] < 15, df["Stock_Cover_Days"] > 60],
        ["Low stock", "Overstock"], default="Healthy")

    df["Data_Status"] = "Synthetic illustrative data (enriched)"
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    order = [
        "Retailer", "Store", "Store_City", "Country", "Store_Area_sqm",
        "Product_Description", "Brand", "Sub_Brand_Variant", "Sub_Brand", "Manufacturer",
        "Subcategory", "Category", "Pack_Size", "Pack_Type",
        "Date", "Month", "Year", "Month_Num", "Period_Type", "Fiscal_Year", "YTD_Period",
        "Local_Currency", "FX_Rate_LC_per_USD",
        "USD_Value", "LC_Value", "Sales_Units",
        "List_Unit_Price_USD", "Net_Unit_Price_USD", "Price_Index", "Unit_Cost_USD",
        "Promo_Flag", "Promo_Type", "Discount_Pct",
        "Holiday_Flag", "Holiday_Name",
        "Trade_Flag", "Trade_Pct_of_Sales", "Trade_Investment_USD",
        "Base_Units_Original", "Baseline_Units", "Baseline_USD_Value",
        "Incremental_Units", "Incremental_USD_Value",
        "Gross_Profit_USD", "Baseline_Gross_Profit_USD", "Incremental_Gross_Profit_USD", "Trade_ROI_Pct",
        "Closing_Inventory_Units", "Inventory_Value_USD", "Inventory_Value_LC",
        "Stock_Cover_Days", "Stock_Status", "Data_Status",
    ]
    df = df[order]
    df.to_csv(dst, index=False)
    print(f"Wrote {dst}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    main()
