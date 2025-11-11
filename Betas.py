# =====================================================
# CAPM Beta Estimation with Real EUR Risk-Free Rate
# =====================================================

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as web

# -------------------- USER SETTINGS --------------------

# Fetch the real Euro risk-free rate (3M German Bund yield from FRED)
rf_series = web.DataReader("IR3TIB01DEM156N", "fred")  # 3-Month EUR interbank yield
RF_ANNUAL = rf_series.iloc[-1].values[0] / 100  # Convert from % to decimal
print(f"Using real EUR risk-free rate: {RF_ANNUAL:.4f} ({RF_ANNUAL*100:.2f}%)")

YEARS = 3
INTERVAL = "1wk"                   # weekly data
MARKET_TICKER = "^STOXX"          # EURO STOXX 600 (EUR)
WEEKS_PER_YEAR = 52

# Portfolio tickers mapping: (Label, YahooTicker, Trading Currency)
# Prefer European listings where possible to minimize FX conversion.
PORTFOLIO = [
    ("Unilever",          "UNA.AS",   "EUR"),
    ("Danone",            "BN.PA",    "EUR"),
    ("Carrefour",         "CA.PA",    "EUR"),
    ("RWE",               "RWE.DE",   "EUR"),
    ("Shell",             "SHELL.AS", "EUR"),
    ("Deutsche Bank",     "DBK.DE",   "EUR"),
    ("HSBC",              "HSBA.L",   "GBp"),  # London listing, pence
    ("Santander",         "SAN.MC",   "EUR"),
    ("Telefonica",        "TEF.MC",   "EUR"),
    ("Siemens",           "SIE.DE",   "EUR"),
    ("Airbus",            "AIR.PA",   "EUR"),
    ("Microsoft",         "MSFT",     "USD"),
    ("Apple",             "AAPL",     "USD"),
    ("Walmart",           "WMT",      "USD"),
]

# FX series from Yahoo Finance
FX = {
    "USD_per_EUR": "EURUSD=X",   # USD per EUR
    "GBP_per_EUR": "EURGBP=X",   # GBP per EUR
}
# -------------------------------------------------------

rf_week = (1 + RF_ANNUAL) ** (1 / WEEKS_PER_YEAR) - 1
end = datetime.today()
start = end - timedelta(days=365 * YEARS + 10)


def download_prices(tickers, start_dt, end_dt, interval="1wk") -> pd.DataFrame:
    """Download adjusted close for a list of tickers via yfinance."""
    df = yf.download(
        tickers,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            df = df["Adj Close"].copy()
        elif "Close" in df.columns.get_level_values(0):
            df = df["Close"].copy()
    else:
        if "Adj Close" in df.columns:
            df = df["Adj Close"].to_frame()
        elif "Close" in df.columns:
            df = df["Close"].to_frame()
        else:
            raise KeyError("No 'Adj Close' or 'Close' in yfinance output.")
    df = df.dropna(how="all")
    df.index.name = "Date"
    return df


# ---------- 1) Download raw data ----------
all_yahoo = [MARKET_TICKER] + [t[1] for t in PORTFOLIO] + list(FX.values())
raw = download_prices(all_yahoo, start, end, INTERVAL)

# Split into components
market = raw[MARKET_TICKER].rename("MKT")
fx_usd_per_eur = raw[FX["USD_per_EUR"]].rename("USD_per_EUR")   # USD per EUR
fx_gbp_per_eur = raw[FX["GBP_per_EUR"]].rename("GBP_per_EUR")   # GBP per EUR

# Local-currency price panel
prices_local = pd.DataFrame({label: raw[yahoo] for (label, yahoo, _) in PORTFOLIO})
prices_local["MKT"] = market

# FX table for auditability
fx_tbl = pd.concat([fx_usd_per_eur, fx_gbp_per_eur], axis=1)


# ---------- 2) Convert all stocks to EUR ----------
def to_eur(series: pd.Series, ccy: str) -> pd.Series:
    if ccy == "EUR":
        return series
    elif ccy == "USD":
        # EUR price = USD price / (USD per EUR)
        return series.div(fx_usd_per_eur, axis=0)
    elif ccy == "GBp":
        # EUR price = GBp / (GBp per EUR) ; GBp per EUR = 100 * (GBP per EUR)
        gbp_per_eur_in_pence = 100.0 * fx_gbp_per_eur
        return series.div(gbp_per_eur_in_pence, axis=0)
    else:
        raise ValueError(f"Unknown currency: {ccy}")


prices_eur = pd.DataFrame(index=prices_local.index)
for (label, yahoo, ccy) in PORTFOLIO:
    prices_eur[label] = to_eur(prices_local[label], ccy)

prices_eur["MKT"] = market  # already EUR


# ---------- 3) Weekly log returns & excess returns ----------
returns = np.log(prices_eur / prices_eur.shift(1)).dropna()
returns.index.name = "Date"

rx = returns - rf_week         # subtract weekly rf from every column
rx_mkt = rx["MKT"]


# ---------- 4) Betas (cov/var == SLOPE) ----------
def beta_from_excess(rx_stock: pd.Series, rx_market: pd.Series) -> float:
    """Return CAPM beta using population cov/var on aligned excess returns."""
    df = pd.concat([rx_stock, rx_market], axis=1, join="inner").dropna()
    if len(df) < 20:
        return np.nan
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=0)[0, 1]
    var_m = np.var(df.iloc[:, 1], ddof=0)
    return np.nan if var_m == 0 else cov / var_m


betas = {label: beta_from_excess(rx[label], rx_mkt) for (label, _, _) in PORTFOLIO}
betas_df = pd.Series(betas, name="beta_raw").to_frame()
betas_df["beta_adj"] = 0.67 * betas_df["beta_raw"] + 0.33   # Blume adjustment (optional)


# ---------- 5) Save CSVs ----------
prices_local.to_csv("data_prices_local.csv")
fx_tbl.to_csv("data_fx.csv")
prices_eur.to_csv("data_prices_eur.csv")
returns.to_csv("data_weekly_returns.csv")
rx.to_csv("data_weekly_excess_returns.csv")
betas_df.to_csv("data_betas.csv")


# ---------- 6) Save a single Excel workbook with tidy sheets ----------
with pd.ExcelWriter("beta_dataset.xlsx", engine="xlsxwriter") as writer:
    prices_local.to_excel(writer, sheet_name="prices_local")
    fx_tbl.to_excel(writer, sheet_name="fx_rates")
    prices_eur.to_excel(writer, sheet_name="prices_eur_EUR")
    returns.to_excel(writer, sheet_name="weekly_returns_log")
    rx.to_excel(writer, sheet_name="weekly_excess_returns")
    betas_df.to_excel(writer, sheet_name="betas")

print("\nCreated files in the current folder:")
print("  - data_prices_local.csv")
print("  - data_fx.csv")
print("  - data_prices_eur.csv")
print("  - data_weekly_returns.csv")
print("  - data_weekly_excess_returns.csv")
print("  - data_betas.csv")
print("  - beta_dataset.xlsx")
