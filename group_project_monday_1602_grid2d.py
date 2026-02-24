from __future__ import annotations

from dateutil.relativedelta import relativedelta
from datetime import datetime
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


"""
We simulate a portfolio over time. At certain dates ("rebalances") we:
  - choose which assets are tradable (investable universe),
  - estimate expected return and risk from recent data,
  - choose portfolio weights by solving an optimisation problem,
  - pay transaction costs if we changed holdings,
  - then let wealth evolve day-by-day until the next rebalance.

We tune two hyperparameters:
  - gamma: how strongly we penalise portfolio risk (variance term),
  - lambda: how strongly we shrink weights toward zero (stability term).

We produce two parameter choices:
  1) Risk-targeted choice: realised volatility closest to target_vol_annual.
  2) Return-seeking choice: maximum final wealth in-sample.

Tuning
----------------
1) COARSE grid search over (gamma, lambda) on a broad range.
2) Two FINE searches in small local boxes:
     - a local box around the coarse return-seeking winner,
     - a local box around the coarse risk-targeted winner.

This reduces runtime compared to a full fine grid over the whole region.

definitions
----------------------------------------
- Let r_t be the vector of realised daily returns for the risky assets on day t.
- Let w be the vector of portfolio weights in risky assets.
- Let w_c be the cash weight: w_c = 1 - sum(w) (cash is “leftover”).

Constraints on risky weights:
  - 0 <= w_i <= cap          (no shorting, and per-asset cap)
  - sum(w) <= 1             (cash allowed)

Daily wealth update:
  V_{t+1} = V_t * (1 + w^T r_t + w_c * rf_daily)

At each rebalance we estimate:
  - mu: average daily returns over a lookback window,
  - cov: covariance matrix of daily returns over the same window.

We then choose w by minimising (conceptually):
  f(w) = - mu^T w + (gamma/2) * w^T cov w + lambda * ||w||^2
subject to the constraints above.

Interpretation:
  - The “- mu^T w” term prefers higher expected return.
  - The “w^T cov w” term penalises risk (variance).
  - The “||w||^2” term shrinks weights toward zero (regularisation).
"""


# ===========================================================
# 0) RUN SETTINGS (what the run will do)
# ===========================================================

# Backtest window length (in calendar years)
years = 5

# End date is “today” in your local environment; start date is years years earlier
end_date = datetime.today()
start_date = end_date - relativedelta(years=years)

# Lookback window (trading days) for:
#   - liquidity ranking (Close × Volume averaged)
#   - estimating mu/cov from returns
lookback_days = 63  # ~3 months of trading days

# Universe sizes
parent_size = 250   # size of initial pool from Yahoo screener ("most active")
n_select = 50       # number of assets we actually optimise over

# Portfolio constraints
cap = 0.08          # maximum risky weight per asset (8%)

# Risk-free interest rate used for the cash component of the portfolio
rf_annual = 0.03
rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0  # convert annual to daily

# Transaction costs
# Turnover at rebalance: sum_i |w_i(new) - w_i(old)|
# Cost deducted at rebalance: cost = kappa * turnover * current wealth
kappa = 0.001
apply_costs = True

# Target annualised volatility for the risk-targeted selection
target_vol_annual = 0.10  # 10%

# Rebalance schedules to run
modes = ["monthly", "quarterly"]

# Benchmark used for beta calculation (does not affect optimisation itself)
benchmark = "SPY"

# Caching to speed up repeated runs
cache_parent = True
cache_prices = True

# Cache directory (stores parent list + downloaded price panels)
cache_root = Path("outputs_gp2a_liquidity")
cache_dir = cache_root / "_cache"
cache_root.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

# Whether to display plots at the end of each mode
show_plots = True


# ===========================================================
# 1) GRID SEARCH SETTINGS (coarse grid + two local fine grids)
# ===========================================================

# Search bounds for gamma (risk penalty strength)
gamma_min = 0.0
gamma_max = 100.0

# Search bounds for lambda (ridge penalty strength)
# IMPORTANT: mu/cov are computed from DAILY returns, so lambda must be small.
lam_min = 1e-6
lam_max = 1e-2

# Coarse step sizes (fewer points)
gamma_step_coarse = 20.0
lam_step_coarse = 1e-3

# Fine step sizes (more resolution near a winner)
gamma_step_fine = 10.0
lam_step_fine = 5e-4

# Half-width of the local fine box around a coarse winner
# Example: if center_gamma=60 and gamma_window=10, gamma in [50,70] (clamped to bounds).
gamma_window = 10.0
lam_window = 1e-3


# ===========================================================
# 2) SOLVER SETTINGS (how hard we solve each optimisation)
# ===========================================================

# Grid runs: “good enough” solutions for comparing parameter values
tol_grid = 5e-4
max_iter_grid = 150

# Final runs: higher accuracy for the parameter values we report
tol_final = 1e-6
max_iter_final = 5000


# ===========================================================
# 3) SMALL UTILITIES (printing, grids, statistics)
# ===========================================================

def explain_block(title: str, lines: list[str]) -> None:
    """
    Print a sectioned block to the console.

    This is used frequently so the run is easy to follow:
      - what stage we are in,
      - what settings are being used,
      - what results were found.
    """
    bar = "=" * max(60, len(title) + 8)
    print("\n" + bar)
    print(f"==  {title}")
    print(bar)
    for s in lines:
        print(s)
    print(bar + "\n", flush=True)


def _dstr(dt: datetime | pd.Timestamp) -> str:
    """Format a date as YYYY-MM-DD."""
    return pd.Timestamp(dt).strftime("%Y-%m-%d")


def _hash_list(xs: list[str]) -> str:
    """
    Create a short cache key based on a list of tickers.
    If the ticker list changes, the hash changes, so we do not reuse the wrong cache file.
    """
    s = "|".join(xs)
    return str(abs(hash(s)))[:10]


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into the interval [lo, hi]."""
    return float(min(max(x, lo), hi))


def fixed_step_grid(vmin: float, vmax: float, step: float) -> np.ndarray:
    """
    Create a fixed-step grid:
      vmin, vmin+step, vmin+2*step, ... <= vmax
    """
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(np.floor((vmax - vmin) / step + 1e-12)) + 1
    vals = vmin + step * np.arange(n, dtype=float)
    vals = vals[vals <= vmax + 1e-12]
    return vals


def ensure_in_grid(vals: np.ndarray, x: float, tol: float = 1e-12) -> np.ndarray:
    """
    Ensure a particular value x appears in vals (within tolerance).
    Useful because “clamping +stepping” could skip an endpoint by rounding.
    """
    if len(vals) == 0:
        return np.array([float(x)], dtype=float)
    if np.any(np.isclose(vals, x, atol=tol, rtol=0.0)):
        return vals
    return np.unique(np.sort(np.append(vals, float(x))))


def annual_vol(daily_returns: pd.Series) -> float:
    """
    Annualised volatility from daily returns:
      vol_ann = std(daily) *sqrt(252)

    We treat 252 trading days as a typical year.
    """
    r = pd.Series(daily_returns).dropna()
    if len(r) < 5:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(252.0))


def max_drawdown(wealth: pd.Series) -> float:
    """
    Maximum drawdown is the worst percentage drop from a previous peak.

    If V_t is wealth at time t, define peak_t = max_{s<=t} V_s.
    Drawdown at t is V_t/ peak_t- 1.
    Max drawdown is min over t of that drawdown.
    """
    V = pd.Series(wealth).dropna()
    if len(V) < 5:
        return float("nan")
    peak = V.cummax()
    dd = V / peak - 1.0
    return float(dd.min())


# ===========================================================
# 4) PARENT UNIVERSE (Yahoo "most active")
# ===========================================================

def get_parent_universe(size: int = 250) -> list[str]:
    """
    Build a “parent universe” from Yahoo’s 'most active' list via yfinance.

    - This is a starting pool of tickers.
    - Later we choose a smaller investable universe from this pool using liquidity.
    - We cache the tickers so reruns are quick.
    """
    cache_path = cache_dir / f"parent_most_actives_{size}.json"

    # Try loading cached tickers first
    if cache_parent and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            tickers = d.get("tickers", [])
            # Sanity check: cache should not be tiny
            if isinstance(tickers, list) and len(tickers) >= min(size, 50):
                return tickers[:size]
        except Exception:
            # If cache is unreadable, we ignore it and fetch fresh
            pass

    # yfinance must provide yf.screen for this approach
    if not hasattr(yf, "screen"):
        raise RuntimeError(
            "yfinance does not provide yf.screen() in this environment. "
            "Update yfinance (pip install -U yfinance) and restart your kernel."
        )

    # Different yfinance versions accept different kwargs; try a few possibilities
    out = None
    last_err = None
    for kwargs in [{"count": size}, {"size": size}, {}]:
        try:
            out = yf.screen("most_actives", **kwargs)
            break
        except Exception as e:
            last_err = e
            out = None

    if out is None:
        raise RuntimeError(f"Could not retrieve most_actives list: {last_err}")

    tickers: list[str] = []

    # yfinance may return a dict with “quotes”
    if isinstance(out, dict):
        quotes = out.get("quotes", [])
        if isinstance(quotes, list):
            for q in quotes:
                sym = q.get("symbol") if isinstance(q, dict) else None
                if isinstance(sym, str):
                    tickers.append(sym)

    # or a DataFrame with a symbol column
    if isinstance(out, pd.DataFrame):
        for col in ["symbol", "Symbol", "ticker", "Ticker"]:
            if col in out.columns:
                tickers = [str(x) for x in out[col].dropna().tolist()]
                break

    # Deduplicate while preserving order
    seen = set()
    tickers2 = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            tickers2.append(t)

    tickers2 = tickers2[:size]
    if len(tickers2) == 0:
        raise RuntimeError("No tickers found for the parent universe.")

    # Save cache
    if cache_parent:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"size": size, "tickers": tickers2}, f, indent=2)

    return tickers2


# ===========================================================
# 5) DATA DOWNLOAD + RETURNS
# ===========================================================

def download_parent_data(
    parent: list[str],
    start_date: datetime,
    end_date: datetime,
    batch_size: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download market data for the parent universe.

    We download three panels (dates × tickers):
      - adj: Adjusted Close (used to compute returns)
      - clo: Close (used for liquidity score Close × Volume)
      - vol: Volume (used for liquidity score Close × Volume)

    We cache the result keyed by:
      - date range
      - parent ticker list (via a short hash)
    """
    parent_tag = _hash_list(parent)
    cache_file = cache_dir / f"prices_{_dstr(start_date)}_{_dstr(end_date)}_{parent_tag}.pkl"

    # If cached, load immediately
    if cache_prices and cache_file.exists():
        try:
            adj, clo, vol = pd.read_pickle(cache_file)
            return adj, clo, vol
        except Exception:
            pass

    adj_list, clo_list, vol_list = [], [], []

    # Download in batches to reduce request size and improve reliability
    for i in range(0, len(parent), batch_size):
        batch = parent[i:i + batch_size]

        # yf.download returns a DataFrame with either MultiIndex columns (multi tickers)
        # or single columns (single ticker)
        df = yf.download(
            batch,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if df is None or len(df) == 0:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            adj_b = df["Adj Close"].copy()
            clo_b = df["Close"].copy()
            vol_b = df["Volume"].copy()
        else:
            # Single ticker fallback
            t = batch[0]
            adj_b = df["Adj Close"].to_frame(t)
            clo_b = df["Close"].to_frame(t)
            vol_b = df["Volume"].to_frame(t)

        adj_list.append(adj_b)
        clo_list.append(clo_b)
        vol_list.append(vol_b)

    # Combine all batches into full panels
    adj = pd.concat(adj_list, axis=1).sort_index()
    clo = pd.concat(clo_list, axis=1).sort_index()
    vol = pd.concat(vol_list, axis=1).sort_index()

    # Clean obvious numeric issues
    adj = adj.replace([np.inf, -np.inf], np.nan)
    clo = clo.replace([np.inf, -np.inf], np.nan)
    vol = vol.replace([np.inf, -np.inf], np.nan)

    # Keep only tickers that successfully downloaded
    common_cols = [c for c in parent if c in adj.columns]
    adj = adj[common_cols]
    clo = clo[common_cols]
    vol = vol[common_cols]

    # Save cache
    if cache_prices:
        try:
            pd.to_pickle((adj, clo, vol), cache_file)
        except Exception:
            pass

    return adj, clo, vol


def download_benchmark_prices(ticker: str, start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Download benchmark adjusted close prices for beta calculation.
    """
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False, threads=True)
    if df is None or len(df) == 0:
        raise RuntimeError("Benchmark download failed.")
    s = df["Adj Close"].copy()
    s.name = ticker
    return s


def simple_returns(adj: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Adjusted Close prices to simple returns:
      R_t = P_t / P_{t-1} - 1

    This is computed per ticker and aligned by date.
    """
    return adj.pct_change(fill_method=None)


# ===========================================================
# 6) LIQUIDITY SELECTION + MOMENT ESTIMATION
# ===========================================================

def liquidity_rank(
    clo: pd.DataFrame,
    vol: pd.DataFrame,
    asof_date: pd.Timestamp,
    lookback_days: int,
    top_n: int,
) -> list[str]:
    """
    Choose the investable universe using a simple liquidity score.

    Liquidity score for each ticker i:
      ADV$_i = mean over lookback window of (Close_i * Volume_i)

    We rank by ADV$ and select the top_n tickers.
    """
    asof_date = pd.Timestamp(asof_date)

    # If asof_date is not in the index (non-trading day), move to the previous trading day
    if asof_date not in clo.index:
        asof_date = clo.index[clo.index.get_loc(asof_date, method="pad")]

    # Window endpoints in index coordinates
    idx = clo.index.get_loc(asof_date)
    lo = max(0, idx - lookback_days + 1)
    win = clo.index[lo:idx + 1]

    # Dollar volume = close * volume
    adv = (clo.loc[win] * vol.loc[win]).mean(axis=0, skipna=True)

    # Drop missing and sort descending (largest liquidity first)
    adv = adv.dropna().sort_values(ascending=False)

    return list(adv.index[:top_n])


def estimate_moments(
    ret_all: pd.DataFrame,
    tickers: list[str],
    end_date: pd.Timestamp,
    lookback_days: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Estimate mu and cov from daily returns.

    Inputs:
      - ret_all: returns panel (dates × tickers)
      - tickers: list of chosen investable tickers
      - end_date: rebalance date (use returns up to this date)
      - lookback_days: number of trading days to use

    Outputs:
      - mu: sample mean vector (daily expected returns)
      - cov: sample covariance matrix (daily return covariance)
    """
    end_date = pd.Timestamp(end_date)
    if end_date not in ret_all.index:
        end_date = ret_all.index[ret_all.index.get_loc(end_date, method="pad")]

    idx = ret_all.index.get_loc(end_date)
    lo = max(0, idx - lookback_days + 1)
    win = ret_all.index[lo:idx + 1]

    # Extract returns for the chosen tickers and window
    R = ret_all.loc[win, tickers].copy()

    # Drop any day with missing return for any ticker
    # This ensures mu/cov are based on aligned data.
    R = R.dropna(axis=0, how="any")

    if len(R) < 10:
        # Not enough data to estimate a covariance matrix reliably
        return None, None

    mu = R.mean(axis=0).to_numpy()
    cov = np.cov(R.to_numpy(), rowvar=False, ddof=1)
    return mu, cov


# ===========================================================
# 7) REBALANCE DATES + SEGMENTS
# ===========================================================

def make_rebalance_dates(mode: str, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """
    Turn a daily trading index into rebalance dates.

    monthly:
      - last trading day of each calendar month

    quarterly:
      - last trading day of each calendar quarter
    """
    dates = pd.DatetimeIndex(dates)

    if mode == "monthly":
        grp = pd.Series(np.arange(len(dates)), index=dates).groupby([dates.year, dates.month])
        idxs = grp.max().to_numpy()
        return list(dates[idxs])

    if mode == "quarterly":
        grp = pd.Series(np.arange(len(dates)), index=dates).groupby([dates.year, ((dates.month - 1) // 3) + 1])
        idxs = grp.max().to_numpy()
        return list(dates[idxs])

    raise ValueError("Unknown mode")


def prepare_mode_segments(
    mode: str,
    ret_all: pd.DataFrame,
    clo: pd.DataFrame,
    vol: pd.DataFrame,
) -> tuple[list[dict], list[pd.Timestamp]]:
    """
    Build a list of segments, one per rebalance.

    Each segment contains everything needed to run the backtest between two rebalances:
      - t0: rebalance date
      - tickers: investable universe at t0
      - mu, cov: estimated from last lookback_days ending at t0
      - cov_lmax: largest eigenvalue of cov (used for PGD step size)
      - hold_R: realised daily returns from (t0, t1] used to evolve wealth

    Universe refresh rule:
      - At the first rebalance date in each calendar year:
          choose a new investable universe by liquidity.
      - Otherwise:
          keep the previous investable universe (reduces unnecessary churn).
    """
    all_dates = ret_all.index.dropna()
    rebal_dates = make_rebalance_dates(mode, all_dates)

    update_dates: list[pd.Timestamp] = []
    years_seen: set[int] = set()
    segs: list[dict] = []

    for k, t0 in enumerate(rebal_dates):
        t0 = pd.Timestamp(t0)

        # Determine whether this rebalance date starts a new year (first time we see this year)
        if t0.year not in years_seen:
            years_seen.add(t0.year)
            update_dates.append(t0)

        # Choose investable universe tickers
        if len(segs) == 0 or t0 in update_dates:
            tickers = liquidity_rank(clo, vol, t0, lookback_days=lookback_days, top_n=n_select)
        else:
            tickers = segs[-1]["tickers"]

        # Estimate mu and cov from returns up to t0
        mu, cov = estimate_moments(ret_all, tickers, t0, lookback_days=lookback_days)
        if mu is None or cov is None:
            continue

        # Precompute the largest eigenvalue of cov:
        # This avoids repeatedly calling eigvalsh inside every optimisation solve.
        try:
            eigs = np.linalg.eigvalsh(cov)
            cov_lmax = float(np.max(eigs))
            if not np.isfinite(cov_lmax) or cov_lmax <= 0:
                cov_lmax = 1.0
        except Exception:
            cov_lmax = 1.0

        # Define end of holding period (next rebalance date, or last date in data)
        if k < len(rebal_dates) - 1:
            t1 = pd.Timestamp(rebal_dates[k + 1])
        else:
            t1 = pd.Timestamp(all_dates[-1])

        # Returns used to evolve wealth between t0 and t1 (exclusive of t0, inclusive of t1)
        hold_idx = ret_all.index[(ret_all.index > t0) & (ret_all.index <= t1)]
        hold_R = ret_all.loc[hold_idx, tickers].copy()
        hold_R = hold_R.dropna(axis=0, how="any")

        segs.append(
            {
                "t0": t0,
                "t1": t1,
                "tickers": tickers,
                "mu": mu,
                "cov": cov,
                "cov_lmax": cov_lmax,
                "hold_R": hold_R,
            }
        )

    return segs, update_dates


# ===========================================================
# 8) OPTIMISATION (Projected Gradient Descent)
# ===========================================================

def project_weights_with_cash(v: np.ndarray, cap: float, tol: float = 1e-12, max_iter: int = 2000) -> np.ndarray:
    """
    Projection step: force a vector v into the feasible weight set.

    Feasible risky weights w satisfy:
      - 0 <= w_i <= cap
      - sum(w) <= 1

    If sum(clipped) <= 1, clipping is enough.
    If sum(clipped) > 1, we “shift down” by tau and clip again:
      w = clip(v - tau, 0, cap)
    and pick tau so sum(w) = 1.
    """
    v = np.asarray(v, dtype=float)

    # Step 1: simple clipping to [0, cap]
    w = np.clip(v, 0.0, cap)

    # If we already satisfy sum <= 1, we are done
    if float(w.sum()) <= 1.0:
        return w

    # Otherwise we need a stricter projection so sum(w)=1
    lo, hi = -1e6, 1e6

    def f(tau: float) -> float:
        # f(tau) measures how far the sum is above 1
        return float(np.clip(v - tau, 0.0, cap).sum() - 1.0)

    # Bisection: find tau such that f(tau) ~ 0
    mid = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        if abs(val) < tol:
            break
        if val > 0:
            lo = mid
        else:
            hi = mid

    w = np.clip(v - mid, 0.0, cap)

    # Safety normalisation in case of tiny numerical overshoot
    s = float(w.sum())
    if s > 1.0:
        w *= (1.0 / s)

    return w


def pgd_solve(
    mu: np.ndarray,
    cov: np.ndarray,
    cov_lmax: float,
    gamma: float,
    lam: float,
    cap: float,
    tol: float,
    max_iter: int,
    w0: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve the constrained weight choice using projected gradient descent (PGD).

    Objective (informally):
      f(w) = - mu^T w + (gamma/2) w^T cov w + lam * ||w||^2

    Gradient:
      grad(w) = -mu + gamma * cov w + 2*lam*w

    Iteration:
      w_{k+1} = Proj( w_k - step * grad(w_k) )

    Step size:
      L ≈ gamma * lambda_max(cov) + 2*lam
      step ≈ 1/L  (capped to avoid instability)
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)

    # Starting point (warm-start if provided, otherwise all-cash)
    if w0 is None or np.asarray(w0).shape != (n,):
        w = np.zeros(n, dtype=float)
    else:
        w = np.asarray(w0, dtype=float).copy()

    # Compute a simple smoothness bound for step size
    L = gamma * float(cov_lmax) + 2.0 * lam
    if not np.isfinite(L) or L <= 0:
        L = 1.0
    step = min(1.0 / L, 0.25)

    # PGD loop
    for _ in range(max_iter):
        # Gradient of the objective at current w
        grad = -mu + gamma * (cov @ w) + 2.0 * lam * w

        # Gradient step (unconstrained)
        v = w - step * grad

        # Projection step to enforce constraints
        w_new = project_weights_with_cash(v, cap=cap)

        # Stop when weights stop changing much
        if float(np.linalg.norm(w_new - w, ord=2)) < tol:
            w = w_new
            break

        w = w_new

    return w


# ===========================================================
# 9) BACKTEST SIMULATION
# ===========================================================

def simulate_from_segments(
    segs: list[dict],
    gamma: float,
    lam: float,
    tol: float,
    max_iter: int,
    warm_init: np.ndarray | None = None,
    verbose_every: int = 0,
    tag: str = "",
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    """
    Run a full backtest given (gamma, lambda) and a list of precomputed segments.

    For each segment (rebalance):
      1) Solve for new risky weights w_r.
      2) Compute cash weight w_c = 1 - sum(w_r).
      3) Compute turnover relative to previous risky weights.
      4) Deduct transaction cost proportional to turnover.
      5) Update wealth day-by-day over the holding returns hold_R.

    Returns:
      - V: wealth time series over trading days
      - TO: turnover at each rebalance date
      - C: transaction cost paid at each rebalance date
      - warm: the final risky weights (can be reused as warm-start)
    """
    V_list: list[float] = []         # wealth values over time
    idx_all: list[pd.Timestamp] = [] # corresponding dates

    TO_rows = []  # turnover records at rebalance dates
    C_rows = []   # cost records at rebalance dates

    # Initial wealth (absolute scale is arbitrary for comparisons)
    Vt = 1_000_000.0

    # Keep previous risky weights so we can measure turnover
    w_prev: np.ndarray | None = None

    for k, seg in enumerate(segs):
        # Segment ingredients
        t0 = seg["t0"]
        mu = seg["mu"]
        cov = seg["cov"]
        cov_lmax = seg["cov_lmax"]
        hold_R: pd.DataFrame = seg["hold_R"]

        # Warm-start priority:
        # - if warm_init is supplied (external), use it,
        # - otherwise use w_prev (previous rebalance weights),
        # - otherwise start at zeros (handled inside pgd_solve).
        w0 = warm_init if warm_init is not None else w_prev

        # Solve for risky weights
        w_r = pgd_solve(
            mu=mu,
            cov=cov,
            cov_lmax=cov_lmax,
            gamma=float(gamma),
            lam=float(lam),
            cap=cap,
            tol=tol,
            max_iter=max_iter,
            w0=w0,
        )

        # Cash weight = remaining budget
        w_c = 1.0 - float(np.sum(w_r))
        w_c = max(0.0, w_c)  # numerical guard

        # Turnover: total absolute change in risky weights
        if w_prev is None:
            turnover = float(np.sum(np.abs(w_r)))
        else:
            turnover = float(np.sum(np.abs(w_r - w_prev)))

        # Transaction cost is paid immediately at rebalance (reduces wealth)
        cost = kappa * turnover * Vt if apply_costs else 0.0
        Vt -= cost

        # Store turnover and cost at the rebalance date
        TO_rows.append(pd.Series({"turnover": turnover}, name=t0))
        C_rows.append(pd.Series({"cost": cost}, name=t0))

        # Evolve wealth day-by-day over the holding period
        # Wealth update:
        #   V_{t+1} = V_t * (1 + w_r^T r_t + w_c * rf_daily)
        if len(hold_R) > 0:
            for dt, rvec in hold_R.iterrows():
                r = rvec.to_numpy(dtype=float)
                Vt *= (1.0 + float(np.dot(w_r, r)) + w_c * rf_daily)
                V_list.append(Vt)
                idx_all.append(pd.Timestamp(dt))
        else:
            # If no return rows (rare), still record the wealth at t0
            V_list.append(Vt)
            idx_all.append(pd.Timestamp(t0))

        # Carry risky weights forward for next rebalance turnover
        w_prev = w_r

        if verbose_every and (k % verbose_every == 0) and k > 0:
            print(f"  [{tag}] segment {k}/{len(segs)}  wealth={Vt:,.0f}", flush=True)

    V = pd.Series(V_list, index=pd.DatetimeIndex(idx_all)).sort_index()
    TO = pd.DataFrame(TO_rows)
    C = pd.DataFrame(C_rows)
    warm = w_prev

    return V, TO, C, warm


# ===========================================================
# 10) PERFORMANCE METRICS
# ===========================================================

def perf_table(V: pd.Series, bench_prices: pd.Series) -> dict:
    """
    Compute a set of metrics from a wealth curve V.

    - CAGR: compound annual growth rate from start to finish
    - AnnVol: annualised volatility from daily returns
    - Sharpe: annualised mean excess return divided by annual volatility
    - MaxDD: maximum drawdown
    - beta: regression-like slope vs benchmark daily returns
    """
    V = pd.Series(V).dropna()
    if len(V) < 10:
        return {}

    # Daily portfolio returns
    Rp = V.pct_change(fill_method=None).dropna()
    if len(Rp) < 10:
        return {}

    # CAGR approximation using number of daily steps
    cagr = float((V.iloc[-1] / V.iloc[0]) ** (252.0 / len(Rp)) - 1.0)

    # Annual volatility
    vol = annual_vol(Rp)

    # Sharpe ratio (annualised)
    sharpe = float((Rp.mean() * 252.0 - rf_annual) / vol) if (np.isfinite(vol) and vol > 1e-12) else float("nan")

    # Max drawdown
    mdd = max_drawdown(V)

    # Beta vs benchmark:
    #   beta = Cov(Rp, Rb) / Var(Rb)
    bench = bench_prices.reindex(V.index).dropna()
    beta = float("nan")
    if len(bench) > 10:
        Rb = bench.pct_change(fill_method=None).dropna()
        idx = Rp.index.intersection(Rb.index)
        if len(idx) > 10:
            x = np.asarray(Rb.loc[idx]).reshape(-1)
            y = np.asarray(Rp.loc[idx]).reshape(-1)
            m = min(len(x), len(y))
            x, y = x[:m], y[:m]
            if m > 10 and np.std(x) > 1e-12:
                beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))

    return {
        "CAGR_port": cagr,
        "AnnVol_port": float(vol),
        "Sharpe_port": sharpe,
        "MaxDD_port": float(mdd),
        "beta_vs_SPY": beta,
        "FinalV": float(V.iloc[-1]),
    }


# ===========================================================
# 11) PLOTTING
# ===========================================================

def plot_grid_heatmap(
    title: str,
    gamma_vals: np.ndarray,
    lam_vals: np.ndarray,
    grid_df: pd.DataFrame,
    points: list[dict] | None = None,
    value_col: str = "finalV",
) -> None:
    """
    Make a heatmap from grid_df, which stores results for a grid of (gamma, lambda).

    We pivot grid_df into a matrix with:
      rows = gamma values
      cols = lambda values
      entries = value_col (e.g. final wealth, or -vol error)
    """
    M = (
        grid_df.pivot(index="gamma", columns="lambda", values=value_col)
        .reindex(index=list(gamma_vals), columns=list(lam_vals))
        .to_numpy()
    )
    MM = np.ma.masked_invalid(M)

    plt.figure(figsize=(10.5, 4.8))
    im = plt.imshow(MM, origin="lower", aspect="auto")
    plt.colorbar(im, label=value_col)

    # Select a small number of ticks to keep labels readable
    G, L = len(gamma_vals), len(lam_vals)
    x_ticks = np.linspace(0, L - 1, min(L, 8), dtype=int)
    y_ticks = np.linspace(0, G - 1, min(G, 8), dtype=int)

    plt.xticks(x_ticks, [f"{lam_vals[j]:g}" for j in x_ticks], rotation=45, ha="right")
    plt.yticks(y_ticks, [f"{gamma_vals[i]:g}" for i in y_ticks])

    plt.xlabel("lambda")
    plt.ylabel("gamma")
    plt.title(title)

    # Optional: mark specific points (e.g. chosen parameters)
    if points:
        for p in points:
            g = float(p["gamma"])
            lam = float(p["lambda"])
            marker = p.get("marker", "x")
            name = p.get("name", "")

            gi = int(np.argmin(np.abs(np.asarray(gamma_vals) - g)))
            lj = int(np.argmin(np.abs(np.asarray(lam_vals) - lam)))

            plt.scatter([lj], [gi], s=70, marker=marker)
            if name:
                plt.text(lj + 0.2, gi + 0.2, name, fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_wealth_two_curves(mode: str, V_risk: pd.Series, V_return: pd.Series) -> None:
    """
    Plot wealth over time for two strategies on the same chart.
    """
    plt.figure(figsize=(10, 4))
    V_risk.plot(label="Risk-targeted choice")
    V_return.plot(label="Return-seeking choice")
    plt.legend()
    plt.title(f"{mode}: wealth over time")
    plt.tight_layout()
    plt.show()


# ===========================================================
# 12) GRID SEARCH HELPERS
# ===========================================================

def run_grid_search(
    segs: list[dict],
    gamma_vals: np.ndarray,
    lam_vals: np.ndarray,
    tol: float,
    max_iter: int,
    tag: str,
) -> pd.DataFrame:
    """
    Evaluate a grid of (gamma, lambda) choices.

    For each pair:
      - run the full backtest over all segments,
      - record final wealth and realised annual volatility,
      - compute err = |vol - target_vol_annual|.

    Warm-starting is used to speed up the grid:
      - across lambdas within a gamma row,
      - and from one gamma row to the next.
    """
    rows: list[dict] = []

    # Warm-start for the next gamma row
    warm_prev_row = None

    # Track the best wealth seen so far (for progress printing)
    best_seen = -np.inf

    for i, g in enumerate(gamma_vals):
        # Warm-start within this gamma row
        warm = warm_prev_row

        for lam in lam_vals:
            # Run backtest
            V, _, _, warm = simulate_from_segments(
                segs=segs,
                gamma=float(g),
                lam=float(lam),
                tol=tol,
                max_iter=max_iter,
                warm_init=warm,
                verbose_every=0,
                tag=tag,
            )

            # Realised portfolio returns from wealth
            Rp = V.pct_change(fill_method=None).dropna()

            # Annualised volatility from daily returns
            v_ann = annual_vol(Rp)

            # Final wealth at end of backtest
            finalV = float(V.iloc[-1]) if len(V) else float("nan")

            rows.append(
                {
                    "gamma": float(g),
                    "lambda": float(lam),
                    "vol": float(v_ann),
                    "finalV": float(finalV),
                }
            )

            if np.isfinite(finalV) and finalV > best_seen:
                best_seen = float(finalV)

        # Carry warm-start to next gamma row (often improves convergence)
        warm_prev_row = warm

        print(
            f"  [{tag}] gamma row {i+1}/{len(gamma_vals)}  gamma={float(g):g}  best wealth so far={best_seen:,.0f}",
            flush=True,
        )

    df = pd.DataFrame(rows)

    # Vol error relative to target
    df["err"] = (df["vol"] - target_vol_annual).abs()

    return df


def pick_best_overall(grid_df: pd.DataFrame) -> tuple[float, float]:
    """
    Return the (gamma, lambda) pair with the highest final wealth.
    """
    row = grid_df.loc[int(grid_df["finalV"].idxmax())]
    return float(row["gamma"]), float(row["lambda"])


def pick_vol_match(grid_df: pd.DataFrame) -> tuple[float, float]:
    """
    Return a (gamma, lambda) pair that gets annual volatility close to target.

    To avoid overfitting to a single lambda outlier, we do:
      - for each gamma, keep the lambda with smallest err (tie-break: higher wealth),
      - then choose the gamma with smallest err (tie-break: higher wealth).
    """
    per_gamma = (
        grid_df.sort_values(["gamma", "err", "finalV"], ascending=[True, True, False])
        .groupby("gamma", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    row = per_gamma.sort_values(["err", "finalV"], ascending=[True, False]).iloc[0]
    return float(row["gamma"]), float(row["lambda"])


def local_fine_grid(center_gamma: float, center_lam: float) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Build a small fixed-step grid around a “center” point.

    Box:
      gamma in [center_gamma - gamma_window, center_gamma + gamma_window]
      lambda in [center_lam - lam_window,  center_lam + lam_window]

    Values are clamped to the global bounds, then expanded into fixed-step grids.
    """
    g_low = clamp(center_gamma - gamma_window, gamma_min, gamma_max)
    g_high = clamp(center_gamma + gamma_window, gamma_min, gamma_max)
    l_low = clamp(center_lam - lam_window, lam_min, lam_max)
    l_high = clamp(center_lam + lam_window, lam_min, lam_max)

    gamma_vals = fixed_step_grid(g_low, g_high, gamma_step_fine)
    lam_vals = fixed_step_grid(l_low, l_high, lam_step_fine)

    # Ensure the center point is explicitly included
    gamma_vals = ensure_in_grid(gamma_vals, center_gamma)
    lam_vals = ensure_in_grid(lam_vals, center_lam)

    return gamma_vals, lam_vals, g_low, g_high, l_low, l_high


# ===========================================================
# 13) RUN ONE MODE (monthly or quarterly)
# ===========================================================

def run_mode(mode: str, ret_all: pd.DataFrame, clo: pd.DataFrame, vol: pd.DataFrame, bench_prices: pd.Series) -> dict:
    """
    Run the entire pipeline for one rebalance schedule:
      - build segments,
      - coarse grid search,
      - fine search around coarse return-seeking winner,
      - fine search around coarse risk-targeted winner,
      - final backtests for each of the two chosen parameter sets.
    """
    explain_block(
        f"Running mode: {mode}",
        [
            "Stages:",
            "  1) build segments (rebalance dates + universes + mu/cov + holding returns),",
            "  2) coarse grid over (gamma, lambda),",
            "  3) fine local grid for return-seeking choice,",
            "  4) fine local grid for risk-targeted choice,",
            "  5) final backtests and plots.",
        ],
    )

    # Build segments once; grid searches reuse them
    t_prep = time.perf_counter()
    segs, update_dates = prepare_mode_segments(mode, ret_all, clo, vol)
    prep_time = time.perf_counter() - t_prep

    explain_block(
        "Segments prepared",
        [
            f"Rebalances (segments): {len(segs)}",
            f"Yearly universe refresh dates: {len(update_dates)}",
            f"Preparation time: {prep_time:.2f} seconds",
            "",
            "Each segment stores cov_lmax to avoid eigenvalue computations inside the optimiser.",
        ],
    )

    # -------------------------
    # COARSE GRID SEARCH
    # -------------------------
    gamma_c = fixed_step_grid(gamma_min, gamma_max, gamma_step_coarse)
    lam_c = fixed_step_grid(lam_min, lam_max, lam_step_coarse)

    explain_block(
        "Coarse grid search",
        [
            f"Gamma values: {list(gamma_c)}",
            f"Lambda: {lam_c.min():g} to {lam_c.max():g} step {lam_step_coarse:g} (n={len(lam_c)})",
            f"Total combinations: {len(gamma_c) * len(lam_c)}",
            f"Grid solver: tol={tol_grid:g}, max_iter={max_iter_grid}",
        ],
    )

    t0 = time.perf_counter()
    grid_c = run_grid_search(segs, gamma_c, lam_c, tol=tol_grid, max_iter=max_iter_grid, tag=f"{mode}-coarse")
    coarse_time = time.perf_counter() - t0

    # Coarse winners
    g_best_c, l_best_c = pick_best_overall(grid_c)
    g_vol_c, l_vol_c = pick_vol_match(grid_c)

    explain_block(
        "Coarse results",
        [
            f"Time taken: {coarse_time:.1f} seconds",
            f"Observed annual vol range: {grid_c['vol'].min():.2%} to {grid_c['vol'].max():.2%}",
            f"Return-seeking coarse point: gamma={g_best_c:g}, lambda={l_best_c:g}",
            f"Risk-targeted coarse point:  gamma={g_vol_c:g}, lambda={l_vol_c:g}",
            f"Target volatility: {target_vol_annual:.2%}",
        ],
    )

    # -------------------------
    # FINE LOCAL SEARCH: RETURN-SEEKING
    # -------------------------
    gamma_f_ret, lam_f_ret, gLr, gHr, lLr, lHr = local_fine_grid(g_best_c, l_best_c)

    explain_block(
        "Fine local search (return-seeking)",
        [
            f"Center: gamma={g_best_c:g}, lambda={l_best_c:g}",
            f"Gamma box: {gLr:g} to {gHr:g} step {gamma_step_fine:g} (n={len(gamma_f_ret)})",
            f"Lambda box: {lLr:g} to {lHr:g} step {lam_step_fine:g} (n={len(lam_f_ret)})",
            f"Total combinations: {len(gamma_f_ret) * len(lam_f_ret)}",
        ],
    )

    t1 = time.perf_counter()
    grid_f_ret = run_grid_search(
        segs, gamma_f_ret, lam_f_ret,
        tol=tol_grid, max_iter=max_iter_grid,
        tag=f"{mode}-fine-return"
    )
    fine_ret_time = time.perf_counter() - t1

    g_best_f, l_best_f = pick_best_overall(grid_f_ret)

    explain_block(
        "Fine results (return-seeking)",
        [
            f"Time taken: {fine_ret_time:.1f} seconds",
            f"Chosen point: gamma={g_best_f:g}, lambda={l_best_f:g}",
        ],
    )

    # -------------------------
    # FINE LOCAL SEARCH: RISK-TARGETED
    # -------------------------
    gamma_f_vol, lam_f_vol, gLv, gHv, lLv, lHv = local_fine_grid(g_vol_c, l_vol_c)

    explain_block(
        "Fine local search (risk-targeted)",
        [
            f"Center: gamma={g_vol_c:g}, lambda={l_vol_c:g}",
            f"Gamma box: {gLv:g} to {gHv:g} step {gamma_step_fine:g} (n={len(gamma_f_vol)})",
            f"Lambda box: {lLv:g} to {lHv:g} step {lam_step_fine:g} (n={len(lam_f_vol)})",
            f"Total combinations: {len(gamma_f_vol) * len(lam_f_vol)}",
        ],
    )

    t2 = time.perf_counter()
    grid_f_vol = run_grid_search(
        segs, gamma_f_vol, lam_f_vol,
        tol=tol_grid, max_iter=max_iter_grid,
        tag=f"{mode}-fine-risk"
    )
    fine_vol_time = time.perf_counter() - t2

    g_vol_f, l_vol_f = pick_vol_match(grid_f_vol)

    explain_block(
        "Fine results (risk-targeted)",
        [
            f"Time taken: {fine_vol_time:.1f} seconds",
            f"Chosen point: gamma={g_vol_f:g}, lambda={l_vol_f:g}",
        ],
    )

    # -------------------------
    # FINAL BACKTESTS (tighter solver)
    # -------------------------
    explain_block(
        "Final backtests",
        [
            "Two full backtests are run with tighter solver settings.",
            f"Final solver: tol={tol_final:g}, max_iter={max_iter_final}",
            "",
            "1) Risk-targeted choice (closest to volatility target).",
            "2) Return-seeking choice (highest final wealth in its local fine grid).",
        ],
    )

    print(f"{mode}: running risk-targeted backtest  (gamma={g_vol_f:g}, lambda={l_vol_f:g})", flush=True)
    V_risk, TO_risk, C_risk, _ = simulate_from_segments(
        segs,
        gamma=g_vol_f,
        lam=l_vol_f,
        tol=tol_final,
        max_iter=max_iter_final,
        warm_init=None,
        verbose_every=10,
        tag=f"{mode}-risk",
    )
    perf_risk = perf_table(V_risk, bench_prices)
    print(f"{mode}: risk-targeted summary: {perf_risk}", flush=True)

    print(f"{mode}: running return-seeking backtest (gamma={g_best_f:g}, lambda={l_best_f:g})", flush=True)
    V_ret, TO_ret, C_ret, _ = simulate_from_segments(
        segs,
        gamma=g_best_f,
        lam=l_best_f,
        tol=tol_final,
        max_iter=max_iter_final,
        warm_init=None,
        verbose_every=0,
        tag=f"{mode}-return",
    )
    perf_ret = perf_table(V_ret, bench_prices)
    print(f"{mode}: return-seeking summary: {perf_ret}", flush=True)

    # -------------------------
    # PLOTS
    # -------------------------
    if show_plots:
        # Heatmap for return-seeking fine grid (colour = final wealth)
        plot_grid_heatmap(
            title=f"{mode}: fine grid around return-seeking point (final wealth)",
            gamma_vals=gamma_f_ret,
            lam_vals=lam_f_ret,
            grid_df=grid_f_ret,
            points=[{"name": "Return", "gamma": g_best_f, "lambda": l_best_f, "marker": "^"}],
            value_col="finalV",
        )

        # Heatmap for risk-targeted fine grid (colour = -err so higher is better)
        grid_f_vol_plot = grid_f_vol.copy()
        grid_f_vol_plot["neg_err"] = -grid_f_vol_plot["err"]
        plot_grid_heatmap(
            title=f"{mode}: fine grid around risk-targeted point (closer to target is better)",
            gamma_vals=gamma_f_vol,
            lam_vals=lam_f_vol,
            grid_df=grid_f_vol_plot,
            points=[{"name": "Risk", "gamma": g_vol_f, "lambda": l_vol_f, "marker": "o"}],
            value_col="neg_err",
        )

        # Wealth curves for the two final choices
        plot_wealth_two_curves(mode, V_risk, V_ret)

        # Turnover comparison plot
        plt.figure(figsize=(10, 3))
        TO_risk["turnover"].plot(label="Risk-targeted turnover")
        TO_ret["turnover"].plot(label="Return-seeking turnover")
        plt.legend()
        plt.title(f"{mode}: turnover over time")
        plt.tight_layout()
        plt.show()

    return {
        "mode": mode,
        "rebalances": int(len(segs)),
        "update_dates": update_dates,
        "coarse_best": (g_best_c, l_best_c),
        "coarse_risk": (g_vol_c, l_vol_c),
        "fine_best": (g_best_f, l_best_f),
        "fine_risk": (g_vol_f, l_vol_f),
        "perf_risk_targeted": perf_risk,
        "perf_return_seeking": perf_ret,
    }


# ===========================================================
# 14) MAIN
# ===========================================================

def main() -> tuple[dict, pd.DataFrame]:
    """
    Main entry point:
      - prints settings,
      - builds parent universe,
      - downloads data,
      - computes returns,
      - downloads benchmark,
      - runs each rebalance mode,
      - prints a summary table.
    """
    t_all = time.perf_counter()

    explain_block(
        "Run settings",
        [
            f"Backtest window: {_dstr(start_date)} to {_dstr(end_date)} ({years} years)",
            f"Lookback window: {lookback_days} trading days",
            f"Parent universe size: {parent_size}",
            f"Investable universe size: {n_select}",
            f"Max risky weight per asset: {cap:.2%}",
            f"Transaction cost rate: {kappa:g} (applied: {apply_costs})",
            f"Annual risk-free rate: {rf_annual:.2%} (daily ≈ {rf_daily:.6f})",
            f"Target annual volatility: {target_vol_annual:.2%}",
            f"Modes: {modes}",
            "",
            "Coarse grid:",
            f"  gamma step {gamma_step_coarse:g}, lambda step {lam_step_coarse:g}",
            "Fine local grids:",
            f"  gamma step {gamma_step_fine:g}, lambda step {lam_step_fine:g}",
            f"  windows: gamma ±{gamma_window:g}, lambda ±{lam_window:g}",
        ],
    )

    # 1) Parent universe tickers
    parent = get_parent_universe(parent_size)
    print("Parent universe tickers:", len(parent), flush=True)

    # 2) Download price/volume panels for parent universe
    adj, clo, vol = download_parent_data(parent, start_date, end_date, batch_size=80)
    print("Panels:", "adj", adj.shape, "close", clo.shape, "vol", vol.shape, flush=True)

    # 3) Compute daily returns panel from adjusted close
    ret_all = simple_returns(adj)
    print("Returns panel:", ret_all.shape, flush=True)

    # 4) Download benchmark series for beta calculation
    bench_prices = download_benchmark_prices(benchmark, start_date, end_date)
    print("Benchmark length:", len(bench_prices.dropna()), flush=True)

    # Run each rebalance schedule independently
    results: dict[str, dict] = {}
    perf_rows: list[dict] = []

    for mode in modes:
        res = run_mode(mode, ret_all, clo, vol, bench_prices)
        results[mode] = res

        # Prepare one row per mode for a compact summary table
        perf_rows.append({
            "mode": mode,
            "FinalV_RiskTarget": res["perf_risk_targeted"].get("FinalV", np.nan),
            "AnnVol_RiskTarget": res["perf_risk_targeted"].get("AnnVol_port", np.nan),
            "Sharpe_RiskTarget": res["perf_risk_targeted"].get("Sharpe_port", np.nan),
            "FinalV_ReturnSeek": res["perf_return_seeking"].get("FinalV", np.nan),
            "AnnVol_ReturnSeek": res["perf_return_seeking"].get("AnnVol_port", np.nan),
            "Sharpe_ReturnSeek": res["perf_return_seeking"].get("Sharpe_port", np.nan),
            "CoarseBest": str(res["coarse_best"]),
            "CoarseRisk": str(res["coarse_risk"]),
            "FineBest": str(res["fine_best"]),
            "FineRisk": str(res["fine_risk"]),
        })

    perf_df = pd.DataFrame(perf_rows)

    explain_block("Summary table", ["Comparison of the two final choices in each mode:"])
    print(perf_df.to_string(index=False), flush=True)

    total_time = time.perf_counter() - t_all
    explain_block(
        "Finished",
        [
            f"Total runtime: {total_time:.1f} seconds",
            f"Cache folder: {str(cache_dir.resolve())}",
        ],
    )

    return results, perf_df


if __name__ == "__main__":
    RESULTS, PERF_DF = main()