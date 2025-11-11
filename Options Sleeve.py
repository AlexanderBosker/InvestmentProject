# =====================================================
# Options Sleeve Builder (robust price sourcing)
# =====================================================
# Creates: options_sleeve.xlsx with sheets:
#  - inputs
#  - underlying_prices (history + S0/S1)
#  - bs_at_t0, bs_at_t1
#  - position_summary
#  - scenarios
# =====================================================

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass

# ---------------- CONFIG ----------------
AUM_TOTAL_EUR       = 100_000_000
EQUITY_SLEEVE_W     = 0.75
EQUITY_AUM_EUR      = AUM_TOTAL_EUR * EQUITY_SLEEVE_W

INDEX_TICKER        = "^STOXX"          # EURO STOXX 600 (EUR)
CONTRACT_MULTIPLIER = 10.0               # € per index point

DATE_T0             = pd.Timestamp("2025-09-26")
DATE_T1             = pd.Timestamp("2025-11-07")
MATURITY_DATE       = pd.Timestamp("2025-12-19")   # December 2025

RISK_FREE_ANNUAL    = 0.0203            # 2.03%
DIVIDEND_YIELD      = 0.00              # set to your best estimate
SIGMA_ANNUAL        = 0.1270            # 12.70%

HEDGE_RATIO         = 0.30              # hedge 30% of equity sleeve delta
USE_COVERED_CALL    = False             # optionally add short call overwrite
CALL_OTM_PCT        = 0.07

# ---- where to get prices ----
# "local"  -> read S0/S1 from your beta_dataset.xlsx (recommended)
# "yfinance" -> fetch from Yahoo (requires internet)
PRICE_SOURCE        = "local"
LOCAL_BETA_DATASET  = "beta_dataset.xlsx"  # path to your file
# ----------------------------------------

# --------------- helpers ----------------
def yearfrac_365(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    return (d1 - d0).days / 365.0

def norm_cdf(x: float) -> float:
    return 0.5*(1.0 + math.erf(x/math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return (1.0/math.sqrt(2.0*math.pi)) * math.exp(-0.5*x*x)

@dataclass
class BSInputs:
    S: float; K: float; r: float; q: float; sigma: float; T: float

@dataclass
class BSResult:
    price: float; delta: float; gamma: float; vega: float; theta: float; d1: float; d2: float

def black_scholes(option_type: str, p: BSInputs) -> BSResult:
    if p.T <= 0 or p.sigma <= 0 or p.S <= 0 or p.K <= 0:
        if option_type.upper() == "C":
            price = max(p.S - p.K, 0.0); delta = 1.0 if p.S > p.K else 0.0
        else:
            price = max(p.K - p.S, 0.0); delta = -1.0 if p.S < p.K else 0.0
        return BSResult(price, delta, 0.0, 0.0, 0.0, float("nan"), float("nan"))

    d1 = (math.log(p.S/p.K) + (p.r - p.q + 0.5*p.sigma**2)*p.T)/(p.sigma*math.sqrt(p.T))
    d2 = d1 - p.sigma*math.sqrt(p.T)
    Nd1, Nd2, nd1 = norm_cdf(d1), norm_cdf(d2), norm_pdf(d1)
    disc_r, disc_q = math.exp(-p.r*p.T), math.exp(-p.q*p.T)

    if option_type.upper() == "C":
        price = p.S*disc_q*Nd1 - p.K*disc_r*Nd2
        delta = disc_q*Nd1
        theta = (-p.S*disc_q*nd1*p.sigma/(2*math.sqrt(p.T))
                 + p.q*p.S*disc_q*Nd1 - p.r*p.K*disc_r*Nd2)
    else:
        price = p.K*disc_r*norm_cdf(-d2) - p.S*disc_q*norm_cdf(-d1)
        delta = -disc_q*norm_cdf(-d1)
        theta = (-p.S*disc_q*nd1*p.sigma/(2*math.sqrt(p.T))
                 - p.q*p.S*disc_q*norm_cdf(-d1) + p.r*p.K*disc_r*norm_cdf(-d2))

    gamma = (disc_q*nd1)/(p.S*p.sigma*math.sqrt(p.T))
    vega  = p.S*disc_q*nd1*math.sqrt(p.T)   # per 1.0 vol (not per 1%)
    return BSResult(price, delta, gamma, vega, theta, d1, d2)

# ------- price sourcing (robust) -------
def get_levels_from_local(path: str):
    """Use your beta_dataset.xlsx -> sheet 'prices_eur_EUR', column 'MKT'."""
    df = pd.read_excel(path, sheet_name="prices_eur_EUR", index_col=0, parse_dates=True)
    assert "MKT" in df.columns, "Column 'MKT' not found in prices_eur_EUR."
    # nearest on-or-before dates
    t0 = df.index[df.index.get_indexer([DATE_T0], method="pad")[0]]
    t1 = df.index[df.index.get_indexer([DATE_T1], method="pad")[0]]
    S0 = float(df.loc[t0, "MKT"]); S1 = float(df.loc[t1, "MKT"])
    hist = df[["MKT"]].copy()
    hist.rename(columns={"MKT":"AdjClose"}, inplace=True)
    hist.index.name = "Date"
    return hist.reset_index(), t0, S0, t1, S1

def get_levels_from_yfinance(ticker: str):
    import yfinance as yf
    hist = yf.download(ticker,
                       start=(DATE_T0 - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                       end=(DATE_T1 + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                       interval="1d", auto_adjust=False, progress=False)
    if hist.empty:
        raise RuntimeError("No data from yfinance.")
    # 'Adj Close' selection may be Series or DataFrame depending on yfinance/pandas versions
    adj = hist["Adj Close"]
    if isinstance(adj, pd.Series):
        px = adj.to_frame(name="AdjClose")
    else:  # DataFrame
        # if single column, rename; else take the first column
        if adj.shape[1] == 1:
            px = adj.copy()
            px.columns = ["AdjClose"]
        else:
            px = adj.iloc[:, [0]].copy()
            px.columns = ["AdjClose"]
    px = px.dropna()
    # nearest on-or-before dates
    t0 = px.index[px.index.get_indexer([DATE_T0], method="pad")[0]]
    t1 = px.index[px.index.get_indexer([DATE_T1], method="pad")[0]]
    S0 = float(px.loc[t0, "AdjClose"]); S1 = float(px.loc[t1, "AdjClose"])
    px = px.reset_index().rename(columns={"Date":"Date"})
    return px, t0, S0, t1, S1

if PRICE_SOURCE.lower() == "local":
    prices_df, t0_used, S0, t1_used, S1 = get_levels_from_local(LOCAL_BETA_DATASET)
else:
    prices_df, t0_used, S0, t1_used, S1 = get_levels_from_yfinance(INDEX_TICKER)

# ------------- BS at t0/t1 -------------
T0_Y = yearfrac_365(DATE_T0, MATURITY_DATE)
T1_Y = yearfrac_365(DATE_T1, MATURITY_DATE)

K_put = S0  # ATM
bs0_put = black_scholes("P", BSInputs(S0, K_put, RISK_FREE_ANNUAL, DIVIDEND_YIELD, SIGMA_ANNUAL, T0_Y))
bs1_put = black_scholes("P", BSInputs(S1, K_put, RISK_FREE_ANNUAL, DIVIDEND_YIELD, SIGMA_ANNUAL, T1_Y))

target_delta_eur = HEDGE_RATIO * EQUITY_AUM_EUR
delta_per_contract_eur = abs(bs0_put.delta) * CONTRACT_MULTIPLIER * S0
contracts_put = int(round(target_delta_eur / delta_per_contract_eur))

start_eur_put = bs0_put.price * CONTRACT_MULTIPLIER * contracts_put
end_eur_put   = bs1_put.price * CONTRACT_MULTIPLIER * contracts_put
pnl_put       = end_eur_put - start_eur_put
hpr_put       = pnl_put / abs(start_eur_put) if start_eur_put != 0 else np.nan
hedge_ratio_realized = (abs(bs0_put.delta) * CONTRACT_MULTIPLIER * S0 * contracts_put) / EQUITY_AUM_EUR

# ---- optional covered call ----
rows = []
if USE_COVERED_CALL:
    K_call = S0 * (1.0 + CALL_OTM_PCT)
    bs0_call = black_scholes("C", BSInputs(S0, K_call, RISK_FREE_ANNUAL, DIVIDEND_YIELD, SIGMA_ANNUAL, T0_Y))
    bs1_call = black_scholes("C", BSInputs(S1, K_call, RISK_FREE_ANNUAL, DIVIDEND_YIELD, SIGMA_ANNUAL, T1_Y))
    overwrite_notional = 0.20 * EQUITY_AUM_EUR
    contracts_call = -int(round(overwrite_notional / (CONTRACT_MULTIPLIER * S0)))
    start_eur_call = bs0_call.price * CONTRACT_MULTIPLIER * contracts_call
    end_eur_call   = bs1_call.price * CONTRACT_MULTIPLIER * contracts_call
    pnl_call       = end_eur_call - start_eur_call
    hpr_call       = pnl_call / abs(start_eur_call) if start_eur_call != 0 else np.nan

# ---- scenarios at t1 ----
scen_moves = np.array([-0.10,-0.05,-0.02,0.0,0.02,0.05,0.10])
scen_rows = []
for m in scen_moves:
    S_scen = S1*(1.0+m)
    out = black_scholes("P", BSInputs(S_scen, K_put, RISK_FREE_ANNUAL, DIVIDEND_YIELD, SIGMA_ANNUAL, T1_Y))
    scen_rows.append({
        "Move_%": m*100, "S_scen": S_scen, "Put_price": out.price,
        "Put_value_EUR": out.price*CONTRACT_MULTIPLIER*contracts_put,
        "PnL_vs_Start_EUR": out.price*CONTRACT_MULTIPLIER*contracts_put - start_eur_put,
        "Delta": out.delta, "Gamma": out.gamma, "Vega_per_1vol": out.vega, "Theta_per_year": out.theta
    })
scenarios_df = pd.DataFrame(scen_rows)

# ---- build tables ----
inputs_df = pd.DataFrame({
    "Param":["AUM_TOTAL_EUR","EQUITY_SLEEVE_W","EQUITY_AUM_EUR","INDEX_TICKER","CONTRACT_MULTIPLIER",
             "DATE_T0","DATE_T1","MATURITY_DATE","RISK_FREE_ANNUAL","DIVIDEND_YIELD","SIGMA_ANNUAL",
             "HEDGE_RATIO","K_put","Price_Source","t0_used","t1_used","S0","S1"],
    "Value":[AUM_TOTAL_EUR,EQUITY_SLEEVE_W,EQUITY_AUM_EUR,INDEX_TICKER,CONTRACT_MULTIPLIER,
             DATE_T0.date(),DATE_T1.date(),MATURITY_DATE.date(),RISK_FREE_ANNUAL,DIVIDEND_YIELD,SIGMA_ANNUAL,
             HEDGE_RATIO,K_put,PRICE_SOURCE,str(t0_used.date()),str(t1_used.date()),S0,S1]
})

bs_t0 = pd.DataFrame({
    "S0":[S0],"K":[K_put],"T0_Y":[T0_Y],"r":[RISK_FREE_ANNUAL],"q":[DIVIDEND_YIELD],
    "sigma":[SIGMA_ANNUAL],"Type":["Put"],
    "Price0":[bs0_put.price],"Delta0":[bs0_put.delta],"Gamma0":[bs0_put.gamma],
    "Vega0":[bs0_put.vega],"Theta0":[bs0_put.theta],"d1_0":[bs0_put.d1],"d2_0":[bs0_put.d2]
})
bs_t1 = pd.DataFrame({
    "S1":[S1],"K":[K_put],"T1_Y":[T1_Y],"r":[RISK_FREE_ANNUAL],"q":[DIVIDEND_YIELD],
    "sigma":[SIGMA_ANNUAL],"Type":["Put"],
    "Price1":[bs1_put.price],"Delta1":[bs1_put.delta],"Gamma1":[bs1_put.gamma],
    "Vega1":[bs1_put.vega],"Theta1":[bs1_put.theta],"d1_1":[bs1_put.d1],"d2_1":[bs1_put.d2]
})

position = pd.DataFrame({
    "Instrument":["Protective Put"], "Contracts":[contracts_put], "Multiplier":[CONTRACT_MULTIPLIER],
    "Start_Price":[bs0_put.price], "End_Price":[bs1_put.price],
    "Start_EUR":[start_eur_put], "End_EUR":[end_eur_put], "PnL_EUR":[pnl_put], "HPR":[hpr_put],
    "Target_Hedge_Ratio":[HEDGE_RATIO], "Realized_Hedge_Ratio_at_t0":[hedge_ratio_realized],
    "Delta0":[bs0_put.delta], "Gamma0":[bs0_put.gamma], "Vega0":[bs0_put.vega], "Theta0":[bs0_put.theta]
})

totals = position[["Start_EUR","End_EUR","PnL_EUR"]].sum()
greeks = position[["Delta0","Gamma0","Vega0","Theta0"]].sum()
totals_row = pd.DataFrame({
    "Instrument":["TOTAL"], "Contracts":[position["Contracts"].sum()],
    "Multiplier":[CONTRACT_MULTIPLIER], "Start_Price":[np.nan], "End_Price":[np.nan],
    "Start_EUR":[totals["Start_EUR"]], "End_EUR":[totals["End_EUR"]], "PnL_EUR":[totals["PnL_EUR"]],
    "HPR":[totals["PnL_EUR"]/abs(totals["Start_EUR"]) if totals["Start_EUR"]!=0 else np.nan],
    "Target_Hedge_Ratio":[HEDGE_RATIO], "Realized_Hedge_Ratio_at_t0":[hedge_ratio_realized],
    "Delta0":[greeks["Delta0"]], "Gamma0":[greeks["Gamma0"]], "Vega0":[greeks["Vega0"]], "Theta0":[greeks["Theta0"]]
})
position_summary = pd.concat([position, totals_row], ignore_index=True)

# ---- save workbook ----
out_path = "options_sleeve.xlsx"
with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
    inputs_df.to_excel(xw, sheet_name="inputs", index=False)
    prices_df.to_excel(xw, sheet_name="underlying_prices", index=False)
    bs_t0.to_excel(xw, sheet_name="bs_at_t0", index=False)
    bs_t1.to_excel(xw, sheet_name="bs_at_t1", index=False)
    position_summary.to_excel(xw, sheet_name="position_summary", index=False)
    scenarios_df.to_excel(xw, sheet_name="scenarios", index=False)

print(f"Created {out_path}")
