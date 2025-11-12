#!/usr/bin/env python3
"""
Calculate portfolio weights & risk from your Excel bundle:
- Loads: R_simple (optional), Covariance_weekly (or Covariance)
- Computes: Equal-weight, GMV (unconstrained), GMV (long-only with per-asset cap)
- Outputs: weights + risk contributions + weekly/annual variance & volatility
- Saves everything next to your Excel file in: <same folder>\weights_outputs\weights_results.xlsx
"""

import os
import numpy as np
import pandas as pd

# ========= EDIT HERE IF YOUR PATH CHANGES =========
INFILE = r"C:\Users\TABos\OneDrive\Bureaublad\Year 4\Semester 1\Finance\Part2_Point2_Outputs.xlsx"
ANNUAL_PERIODS = 52      # weekly → annual
LONG_ONLY_CAP  = 0.20    # set to None for no per-asset cap (still long-only)
# ==================================================

# Try to import SciPy for long-only optimization (SLSQP). If missing, we’ll skip long-only.
try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def annualize_var(var_weekly, periods=ANNUAL_PERIODS):
    return float(var_weekly) * periods

def annualize_vol(vol_weekly, periods=ANNUAL_PERIODS):
    return float(vol_weekly) * np.sqrt(periods)

def port_var(w, Sigma):
    w = np.asarray(w, dtype=float).reshape(-1)
    return float(w @ Sigma @ w)

def risk_contributions(w, Sigma):
    """
    Marginal: m_i = (Σ w)_i
    Abs contrib: c_i = w_i * m_i
    Percent: pc_i = c_i / (w'Σw)
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    m = Sigma @ w
    abs_c = w * m
    total = float(w @ m)
    pct_c = abs_c / total if total > 0 else np.zeros_like(abs_c)
    return m, abs_c, pct_c

def summarize(name, cols, w, Sigma, periods=ANNUAL_PERIODS):
    w = np.asarray(w, dtype=float).reshape(-1)
    var_w = port_var(w, Sigma)
    vol_w = np.sqrt(var_w)
    var_a = annualize_var(var_w, periods)
    vol_a = annualize_vol(vol_w, periods)
    m, c_abs, c_pct = risk_contributions(w, Sigma)
    table = pd.DataFrame({
        "Asset": cols,
        "Weight": w,
        "Marginal_Contrib": m,
        "Abs_Contrib_to_Var": c_abs,
        "Pct_Contrib_to_Var": c_pct
    })
    meta = pd.DataFrame([{
        "Name": name,
        "weekly_variance": var_w,
        "weekly_volatility": vol_w,
        "annual_variance": var_a,
        "annual_volatility": vol_a
    }])
    return table, meta

def gmv_unconstrained(Sigma):
    """ w* = Σ^{-1} 1 / (1' Σ^{-1} 1) """
    N = Sigma.shape[0]
    one = np.ones(N)
    invS = np.linalg.pinv(Sigma)  # numerically stable
    w = invS @ one / (one @ invS @ one)
    return w

def gmv_long_only(Sigma, cap=None):
    """ Minimize w'Σw s.t. sum(w)=1, w>=0, and optional per-asset cap """
    if not SCIPY_OK:
        raise RuntimeError("SciPy not installed; run: pip install scipy")

    N = Sigma.shape[0]
    x0 = np.ones(N) / N
    bounds = [(0.0, cap if cap is not None else 1.0)] * N
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    def obj(w): return float(w @ Sigma @ w)

    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 10000, 'ftol': 1e-12})
    if not res.success:
        raise RuntimeError(f"Long-only GMV failed: {res.message}")
    return res.x


def main():
    # --- Load inputs ---
    # Prefer Covariance_weekly; fallback to Covariance
    try:
        cov_df = pd.read_excel(INFILE, sheet_name="Covariance_weekly", engine="openpyxl", index_col=0)
    except Exception:
        cov_df = pd.read_excel(INFILE, sheet_name="Covariance", engine="openpyxl", index_col=0)

    Sigma = cov_df.values.astype(float)
    cols  = list(cov_df.columns)

    # R_simple is optional; load if present (for traceability)
    try:
        R = pd.read_excel(INFILE, sheet_name="R_simple", engine="openpyxl", index_col=0)
    except Exception:
        R = None

    N = Sigma.shape[0]

    # --- Compute weights sets ---
    results_tables = []
    summary_rows = []

    # Equal-weight
    w_eq = np.ones(N) / N
    tbl_eq, meta_eq = summarize("Equal-Weight", cols, w_eq, Sigma)
    results_tables.append(("Equal_Weight", tbl_eq))
    summary_rows.append(meta_eq)

    # GMV unconstrained
    w_uc = gmv_unconstrained(Sigma)
    tbl_uc, meta_uc = summarize("GMV Unconstrained", cols, w_uc, Sigma)
    results_tables.append(("GMV_Unconstrained", tbl_uc))
    summary_rows.append(meta_uc)

    # GMV long-only (with cap)
    tbl_lo = meta_lo = None
    if SCIPY_OK:
        try:
            w_lo = gmv_long_only(Sigma, cap=LONG_ONLY_CAP)
            cap_label = "" if LONG_ONLY_CAP is None else f" (cap {LONG_ONLY_CAP:.0%})"
            tbl_lo, meta_lo = summarize(f"GMV Long-Only{cap_label}", cols, w_lo, Sigma)
            results_tables.append(("GMV_LongOnly", tbl_lo))
            summary_rows.append(meta_lo)
        except Exception as e:
            print(f"[Note] Long-only GMV skipped: {e}")
    else:
        print("[Note] SciPy not installed → skipping long-only GMV. Install with: pip install scipy")

    # Optional: add your own custom weights by editing below
    # Example:
    # custom = pd.DataFrame({"Asset": cols, "Weight": w_eq}).assign(Weight=lambda d: d["Weight"])  # copy of equal
    # tbl_c, meta_c = summarize("Custom", cols, custom["Weight"].values, Sigma)
    # results_tables.append(("Custom_Weights", tbl_c))
    # summary_rows.append(meta_c)

    # --- Save outputs next to input file ---
    outdir = os.path.join(os.path.dirname(INFILE), "weights_outputs")
    os.makedirs(outdir, exist_ok=True)
    outxlsx = os.path.join(outdir, "weights_results.xlsx")

    with pd.ExcelWriter(outxlsx, engine="openpyxl") as xw:
        # Inputs
        cov_df.to_excel(xw, sheet_name="Sigma_weekly")
        if R is not None:
            R.to_excel(xw, sheet_name="R_simple")

        # Results
        for sheetname, df in results_tables:
            df.to_excel(xw, sheet_name=sheetname, index=False)

        # Summary
        summary = pd.concat(summary_rows, ignore_index=True)
        summary.to_excel(xw, sheet_name="Summary", index=False)

    # --- Console summary ---
    def fmt_row(r):
        return (f"{r['Name']}: "
                f"σ_weekly={r['weekly_volatility']:.4%}, "
                f"σ_annual={r['annual_volatility']:.4%}, "
                f"var_weekly={r['weekly_variance']:.6f}, "
                f"var_annual={r['annual_variance']:.6f}")

    print("\n=== Portfolio Risk Summary ===")
    for _, row in summary.iterrows():
        print(fmt_row(row))

    print(f"\nSaved results to: {outxlsx}")


if __name__ == "__main__":
    main()
