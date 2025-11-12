#!/usr/bin/env python3
# Build R (T×N simple weekly returns), Σ (N×N sample covariance), σ (N×1 sample stdev)
# Sheet: 'TxN Matrix' | Prices/Log region: A:P | Rows up to 160
# Header row is at Excel row 2 (A2 = "Date")

import os, re, sys, argparse
import numpy as np
import pandas as pd

# ---------- YOUR SETTINGS ----------
DEFAULT_XLSX_PATH  = r"C:\Users\TABos\OneDrive\Bureaublad\Year 4\Semester 1\Finance\Finance Workbook Investment Assignment.xlsx"
DEFAULT_SHEET      = "TxN Matrix"
PRICES_RANGE       = "A:P"      # Prices in EUR and/or Weekly log returns
USE_LOG_RETURNS    = True       # prefer Weekly log returns (converted to simple); else use prices
MIN_OBS_PER_ASSET  = 30
HEADER_ROW_EXCEL   = 2          # <-- A2 contains 'Date'
MAX_DATA_ROW_EXCEL = 160
NON_EQUITY_PATTERNS = [r"\bmkt\b", r"market", r"\brf\b", r"risk[-\s]*free", r"benchmark", r"stoxx", r"index"]
# -----------------------------------

def any_match(s, pats): s = (s or "").lower(); return any(re.search(p, s, re.I) for p in pats)

def clean_headers(cols):
    out, seen = [], {}
    for c in cols:
        c = "" if c is None else str(c).strip()
        if c in seen: seen[c]+=1; c=f"{c}.{seen[c]}"; 
        else: seen[c]=1
        out.append(c)
    return out

def parse_datetime_series(raw):
    s_dt = pd.to_datetime(raw, errors="coerce")
    if s_dt.notna().mean() < 0.5:
        s_num = pd.to_numeric(raw, errors="coerce")
        if s_num.notna().any():
            guess = pd.to_datetime(s_num, unit="D", origin="1899-12-30", errors="coerce")
            if guess.notna().mean() > s_dt.notna().mean(): s_dt = guess
    return s_dt

def detect_header_row_by_date(raw):
    # look for a row where col A (or any cell) equals 'date' (case-insensitive)
    for r in range(0, min(10, len(raw))):
        row_vals = [str(x).strip().lower() for x in list(raw.iloc[r].values)]
        if "date" in row_vals: return r
        if str(raw.iloc[r,0]).strip().lower() == "date": return r
    return None

def split_cols_by_type(df, date_col):
    rtn, px = [], []
    for c in df.columns:
        if c == date_col: continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
        if len(s)==0: px.append(c); continue
        medabs = s.abs().median(); frac_small = (s.abs()<=1.5).mean()
        (rtn if (medabs<0.5 and frac_small>0.9) else px).append(c)
    return df[[date_col]+rtn], df[[date_col]+px]

def drop_non_equity(df): return df[[c for c in df.columns if not any_match(c, NON_EQUITY_PATTERNS)]]

def build_R_Sigma_sigma(xlsx, sheet, usecols, use_log, min_obs, header_row_excel, max_row_excel):
    print(f"Reading: {xlsx} [sheet='{sheet}']")
    # Read raw (no header) up to MAX row to robustly locate actual header
    raw = pd.read_excel(xlsx, sheet_name=sheet, usecols=usecols, header=None, engine="openpyxl")
    if max_row_excel: raw = raw.iloc[:max_row_excel, :].copy()

    # Prefer explicit header row if provided (Excel row index -> zero-based)
    header_row = header_row_excel-1 if header_row_excel else None
    # Validate that row looks like a header (has 'Date'); else auto-detect
    if header_row is None or "date" not in [str(x).strip().lower() for x in raw.iloc[header_row].values]:
        hdr_guess = detect_header_row_by_date(raw)
        header_row = hdr_guess if hdr_guess is not None else 0

    df = raw.copy()
    df.columns = clean_headers(df.iloc[header_row].values)
    df = df.iloc[header_row+1: , :].copy()

    # Date column
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols: raise RuntimeError("No 'Date' header found after header detection.")
    date_col = date_cols[0]
    df[date_col] = parse_datetime_series(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Split 'returns-like' vs 'prices-like'
    returns_like, prices_like = split_cols_by_type(df, date_col)
    returns_like = drop_non_equity(returns_like)
    prices_like  = drop_non_equity(prices_like)

    # Build R (simple)
    if use_log and returns_like.shape[1] > 1:
        LR = returns_like.set_index(date_col).apply(pd.to_numeric, errors="coerce")
        R = np.expm1(LR)  # exp(log)-1
        source = "weekly log returns (converted to simple)"
    elif prices_like.shape[1] > 1:
        PX = prices_like.set_index(date_col).apply(pd.to_numeric, errors="coerce")
        R = PX.pct_change().replace([np.inf,-np.inf], np.nan)
        source = "prices in EUR (pct_change)"
    else:
        raise RuntimeError("No usable returns-like or prices-like equity columns found in A:P.")

    # Clean
    R = R.dropna(how="all").dropna(axis=1, how="all")
    R = R.loc[:, ~R.columns.duplicated(keep="first")]
    R = R[R.columns[R.notna().sum() >= min_obs]]
    if R.shape[1] < 1: raise RuntimeError("After cleaning, no equity columns remain.")

    Sigma = R.cov()
    sigma = R.std(ddof=1).to_frame("stdev")

    os.makedirs("outputs", exist_ok=True)
    R.to_csv("outputs/R_simple.csv")
    Sigma.to_csv("outputs/cov_matrix.csv")
    sigma.to_csv("outputs/stdev_vector.csv")
    with pd.ExcelWriter("outputs/equity_data_bundle.xlsx", engine="openpyxl") as xw:
        R.to_excel(xw, sheet_name="R_simple")
        Sigma.to_excel(xw, sheet_name="Covariance")
        sigma.to_excel(xw, sheet_name="Stdev")

    print(f"Header row used (0-based): {header_row}")
    print(f"Date column: {date_col}")
    print(f"Built R from: {source} | Shapes -> R: {R.shape}, Σ: {Sigma.shape}, σ: {sigma.shape}")
    return R, Sigma, sigma

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", default=DEFAULT_XLSX_PATH)
    p.add_argument("--sheet", default=DEFAULT_SHEET)
    p.add_argument("--range", default=PRICES_RANGE)
    p.add_argument("--from-log", action="store_true")
    p.add_argument("--min-obs", type=int, default=MIN_OBS_PER_ASSET)
    p.add_argument("--header-row", type=int, default=HEADER_ROW_EXCEL, help="Excel row number of header (e.g., 2 if A2='Date').")
    p.add_argument("--max-row", type=int, default=MAX_DATA_ROW_EXCEL, help="Cut off reading after this Excel row.")
    a = p.parse_args()
    try:
        build_R_Sigma_sigma(
            xlsx=a.xlsx, sheet=a.sheet, usecols=a.range,
            use_log=(a.from_log or USE_LOG_RETURNS),
            min_obs=a.min_obs, header_row_excel=a.header_row, max_row_excel=a.max_row
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        # quick debug: show first few header candidates
        try:
            dbg = pd.read_excel(a.xlsx, sheet_name=a.sheet, usecols=a.range, header=None, nrows=5, engine="openpyxl")
            print("Top rows preview:\n", dbg, file=sys.stderr)
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
