from __future__ import annotations #To not evaluate annotations immediately

from dataclasses import dataclass # To allow definition of modes/configurations
from datetime import datetime #
from dateutil.relativedelta import relativedelta
from pathlib import Path #Should work on any device now
import time #check how long it takes

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


"""
Portfolio backtest with a calibration/test split (date-based) plus simple baselines.

How to read this file (big picture)
-----------------------------------
This script does:

  1) Define run controls and model settings near the top of the file.
  2) Build a parent universe (Yahoo Finance 'most_actives') and download price/volume data.
     - Downloads are cached so re-runs are fast and reproducible for a fixed ticker list.
  3) Convert Adjusted Close prices into a clean panel of daily SIMPLE returns.
  4) Build rebalance-to-rebalance "segments" for a given rebalance frequency (monthly / quarterly).
  5) Split segments into:
       - calibration window (used ONLY to choose hyperparameters), and
       - test window (out-of-sample evaluation).
     The split is snapped to a rebalance date to avoid look-ahead leakage.
  6) Calibrate (gamma, lambda) on calibration segments using a grid search:
       - one parameter pair for the "return-seeking" strategy,
       - one parameter pair for the "risk-controlled" strategy.
  7) Run the two strategies out-of-sample on the test window, alongside:
       - equal-weight baseline, and
       - SPY buy-and-hold benchmark.
  8) Save CSV tables and PNG plots into a timestamped output folder.

"""



# Pandas print settings: makes console tables easier to read (does not affect computations).
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)

# Global run dates (set in main; used for the run manifest)
START_DATE = None  # type: ignore
END_DATE = None    # type: ignore


# ======================================================================
# These switches are intended for day-to-day use: speed/accuracy, which modes to run, and output behaviour.
# only change these controls deliberately.
# 0) QUICK CONTROLS (edit these first)
# ======================================================================

# ---- Speed / accuracy profile for calibration ----
# FAST: fewer grid points + fewer solver iterations (quick experimentation)
# BALANCED: default; good trade-off
# ACCURATE: larger grids / stricter solver settings (slower)
# Calibration speed/accuracy profile. This changes solver tolerances / iteration budgets and grid resolution.
# It does NOT change the objective or constraints; it only changes how thoroughly we search for (gamma, lambda).
RUN_PROFILE: str = "BALANCED"  # "FAST" | "BALANCED" | "ACCURATE"

# ---- Rebalance modes to run ----
# Choose: "monthly", "quarterly", or "both".
# This affects ONLY which backtests are executed; the optimisation/backtest method is unchanged.
# Which rebalance frequencies to execute. 'both' runs two independent pipelines (monthly and quarterly).
RUN_MODES: str = "monthly"  # "monthly" | "quarterly" | "both"

# ---- Calibration scope ----
# For speed: calibrate monthly and reuse for quarterly.
CALIBRATION_SCOPE: str = "MONTHLY_ONLY"  # "MONTHLY_ONLY" | "QUARTERLY_ONLY" | "BOTH"

# ---- Cash cap toggle ----
# If disabled, cash isn't allowed at all
# If enabled, enforce cash <= CASH_CAP (e.g. 10%) by requiring sum(weights) >= 1 - CASH_CAP.
# If enabled, we require the risky weights to sum to at least (1 - CASH_CAP), which limits how much can sit in cash.
ENABLE_CASH_CAP: bool = True
CASH_CAP: float = 1  # "max X% in cash" rule

# ---- What to save/show ----
SAVE_CSVS: bool = True
SAVE_PLOTS: bool = True
SHOW_PLOTS: bool = True
# When True, we store per-rebalance weights in a long-format table (can be large).
# This is required for plots that show weights over time (e.g. top-10 holdings).
SAVE_WEIGHTS_CSV: bool = True  # big files; set False to speed up

# ---- Benchmark ----
BENCHMARK_TICKER: str = "SPY"


# ======================================================================
# Model/backtest configuration. These are method parameters.
# For reproducibility: if we change these values, results will change.
# 1) CONFIGURATION (mostly fixed(try not to tweak these unless you have to))
# ======================================================================

@dataclass
class Profile:
    # coarse grids
    gamma_step_coarse: float
    lam_n_coarse: int
    # fine search
    gamma_window: float
    lam_n_fine: int
    lam_factor_window: float
    # solver budget for calibration (used in coarse and fine)
    tol_cal: float
    max_iter_cal: int
    # solver budget for test backtests (used only on TEST)
    tol_test: float
    max_iter_test: int


# Profiles control the grid-search resolution and PGD solver budgets.
# FAST is for debugging; ACCURATE is for final runs; BALANCED is a sensible default.
PROFILES: dict[str, Profile] = {
    "FAST": Profile(
        gamma_step_coarse=20.0,   # 0,20,...,100 (6 points)
        lam_n_coarse=6,           # 6 lambdas
        gamma_window=3.0,         # refine gamma in +/- 3
        lam_n_fine=3,             # 3 lambdas around best
        lam_factor_window=1.8,
        tol_cal=1e-3,
        max_iter_cal=80,
        tol_test=5e-5,
        max_iter_test=800,
    ),
    "BALANCED": Profile(
        gamma_step_coarse=10.0,   # 0,10,...,100 (11 points)
        lam_n_coarse=8,           # 8 lambdas (cheaper than 10)
        gamma_window=5.0,         # refine gamma in +/- 5
        lam_n_fine=5,             # 5 lambdas around best
        lam_factor_window=2.0,
        tol_cal=5e-4,
        max_iter_cal=500,
        tol_test=1e-5,
        max_iter_test=1500,
    ),
    "ACCURATE": Profile(
        gamma_step_coarse=5.0,    # 0,5,...,100 (21 points)
        lam_n_coarse=10,          # 10 lambdas
        gamma_window=6.0,         # refine gamma in +/- 6
        lam_n_fine=7,             # 7 lambdas around best
        lam_factor_window=2.5,
        tol_cal=2e-4,
        max_iter_cal=1000,
        tol_test=5e-6,
        max_iter_test=1000,
    ),
}


@dataclass(frozen=True)
# Core experiment settings: horizon, universe size, lookback window, constraints, costs, and calibration ranges.
class Config:
    # horizon and split
    years_total: int = 8 #total years of data downloaded
    test_years: int = 3 #time at the end where we just test the lambda/gamma, and run the optmiser as it would in the fiture
    
    # lookback for liquidity + moments
    lookback_days: int = 63

    # universe sizes (Keep at 250, we struggle to obtain more with our current yahoo call function)
    parent_size: int = 250
    n_select: int = 50

    # constraints
    cap: float = 0.10  # max weight per risky asset

    # rates and costs
    rf_annual: float = 0.037  #How much cash earns annually 
    kappa: float = 0.001  #bps transaction costs
    apply_costs: bool = True

    # risk-targeting (selection target, not a hard constraint)
    target_vol_annual: float = 0.15

    # grid ranges (fixed)
    gamma_min: float = 0.0
    gamma_max: float = 100.0
    lam_min: float = 1e-4
    lam_max: float = 1

    # tie-break stability
    near_opt_eps_return: float = 5e-3  # within 0.5% of best wealth -> min vol -> max gama

    # outputs
    outputs_root: str = "outputs_gp2a_liquidity"

    # wealth scaling
    initial_wealth: float = 10_000_000.0


# CFG holds the run configuration used throughout the script (single source of truth).
CFG = Config()
PROFILE = PROFILES.get(RUN_PROFILE.upper(), PROFILES["BALANCED"])


# ======================================================================
# 2) PATHS (cache + run outputs)
# ======================================================================

ROOT_OUT = Path(CFG.outputs_root)
CACHE_DIR = ROOT_OUT / "_cache"

#the above are created ONLY IF MISSING

ROOT_OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT_OUT / f"run_{RUN_ID}" #the folder where everything goes

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 3) SMALL UTILITIES
# ======================================================================

# ----------------------------------------------------------------------
# Function: explain_block - this prints in the console for everything that's happening
# ----------------------------------------------------------------------
def explain_block(title: str, lines: list[str]) -> None:
    bar = "=" * 70
    print("\n" + bar)
    print(f"==  {title}")
    print(bar)
    for s in lines:
        print(s)
    print(bar + "\n", flush=True)


# ----------------------------------------------------------------------
# Function: clamp for fine windows in grids
# ----------------------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


# ----------------------------------------------------------------------
# Function: fixed_step_grid
# ----------------------------------------------------------------------
def fixed_step_grid(vmin: float, vmax: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(np.floor((vmax - vmin) / step + 1e-12)) + 1
    vals = vmin + step * np.arange(n, dtype=float)
    return vals[vals <= vmax + 1e-12]


# ----------------------------------------------------------------------
# Function: log_grid (lambda is sensitive to scale, not linear increase)
#Therefore better to search for it multiplicatively
# ----------------------------------------------------------------------
def log_grid(vmin: float, vmax: float, n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be >= 2 for log_grid")
    if vmin <= 0 or vmax <= 0:
        raise ValueError("vmin and vmax must be > 0 for log_grid")
    if vmax < vmin:
        raise ValueError("vmax must be >= vmin for log_grid")
    return np.geomspace(vmin, vmax, num=n, dtype=float)


# ----------------------------------------------------------------------
# Function: ensure_in_grid
# ----------------------------------------------------------------------
def ensure_in_grid(vals: np.ndarray, x: float, tol: float = 1e-12) -> np.ndarray:
    if len(vals) == 0:
        return np.array([float(x)], dtype=float)
    if np.any(np.isclose(vals, x, atol=tol, rtol=0.0)):
        return vals
    return np.sort(np.append(vals, float(x)))


# ----------------------------------------------------------------------
# Function: save_csv
# ----------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


# ----------------------------------------------------------------------
# Function: save_kv_csv
# ----------------------------------------------------------------------
def save_kv_csv(kv: dict, path: Path) -> None:
    df = pd.DataFrame({"key": list(kv.keys()), "value": list(kv.values())}).set_index("key")
    save_csv(df, path)


# ----------------------------------------------------------------------
# Function: _rf_daily
# ----------------------------------------------------------------------
def _rf_daily(rf_annual: float) -> float:
    return (1.0 + float(rf_annual)) ** (1.0 / 252.0) - 1.0


# ======================================================================
# Universe: we pull a 'most active' list from Yahoo at runtime, then cache the resulting ticker list and price panels.
# Important: the ticker list is included in the cache key to avoid mixing different universes across runs.
# 4) PARENT UNIVERSE + DATA DOWNLOAD (cached)
# ======================================================================

# ----------------------------------------------------------------------
# Function: _hash_list
# ----------------------------------------------------------------------
def _hash_list(xs: list[str]) -> str:
    return str(abs(hash("|".join(xs))))[:10]


# Fetch the parent ticker list. We cache the list on disk so re-runs are stable even if Yahoo's screen changes later.
# ----------------------------------------------------------------------
# Function: get_parent_tickers_yahoo_most_active:
    #1. check the cache, if it isn't there it downloads new data
    #2. If exists: read tickers, clean strings, return first n if list looks sufficiently populated.
    #3. If not cached (or cache invalid): call Yahoo “screener predefined saved” endpoint most_actives using requests.
    #4. Parse symbols, deduplicate while preserving order.
    #5. If nothing returned, fall back to ["AAPL", "MSFT"...] as emergency.
    #6. Save the tickers to cache CSV, return first n.
# ----------------------------------------------------------------------
def get_parent_tickers_yahoo_most_active(n: int) -> list[str]:
    """Pull parent universe using Yahoo 'most actives'. Cached."""
    cache_path = CACHE_DIR / f"parent_most_active_{n}.csv"
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            xs = df["ticker"].astype(str).tolist()
            xs = [x.strip().upper() for x in xs if isinstance(x, str) and len(x.strip()) > 0]
            if len(xs) >= max(10, n // 2):
                return xs[:n]
        except Exception:
            pass

    import requests

    url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    params = {"scrIds": "most_actives", "count": int(n), "start": 0}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    quotes = j.get("finance", {}).get("result", [{}])[0].get("quotes", [])

    tickers: list[str] = []
    for q in quotes:
        t = q.get("symbol", None)
        if isinstance(t, str) and len(t) > 0:
            tickers.append(t.strip().upper())

    seen = set()
    dedup = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            dedup.append(t)

    if len(dedup) == 0:
        dedup = ["AAPL", "MSFT", "AMZN", "GOOG", "META"]

    pd.DataFrame({"ticker": dedup}).to_csv(cache_path, index=False)
    return dedup[:n]


# Download and cache the full parent-universe data panel (Adjusted Close, Close, Volume).
# The ticker list is part of the cache key to avoid silently mixing universes.

# Download Adjusted Close / Close / Volume for the parent universe.
# Data are cached using (start_date, end_date, hash(ticker_list)) so results remain reproducible for a fixed universe.
# ----------------------------------------------------------------------
# Function: download_parent_data
# ----------------------------------------------------------------------
def download_parent_data(
    parent: list[str],
    start_date: datetime,
    end_date: datetime,
    batch_size: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download AdjClose/Close/Volume for parent tickers, with caching."""
    tag = _hash_list(parent)
    cache_file = CACHE_DIR / f"prices_{start_date:%Y-%m-%d}_{end_date:%Y-%m-%d}_{tag}.pkl"
    if cache_file.exists():
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass

    adj_list, clo_list, vol_list = [], [], []
    for i in range(0, len(parent), batch_size):
        batch = parent[i:i + batch_size]
        # Yahoo download call. We use batches to avoid request-size limits and to keep failures local to a batch.
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
            t = batch[0]
            adj_b = df["Adj Close"].to_frame(t)
            clo_b = df["Close"].to_frame(t)
            vol_b = df["Volume"].to_frame(t)

        adj_list.append(adj_b)
        clo_list.append(clo_b)
        vol_list.append(vol_b)

    adj = pd.concat(adj_list, axis=1).sort_index().replace([np.inf, -np.inf], np.nan)
    clo = pd.concat(clo_list, axis=1).sort_index().replace([np.inf, -np.inf], np.nan)
    vol = pd.concat(vol_list, axis=1).sort_index().replace([np.inf, -np.inf], np.nan)

    cols = [c for c in parent if c in adj.columns]
    adj, clo, vol = adj[cols], clo[cols], vol[cols]

    pd.to_pickle((adj, clo, vol), cache_file)
    return adj, clo, vol


# ----------------------------------------------------------------------
# Function: download_benchmark_prices, returns a panda series named ticker for each one
# ----------------------------------------------------------------------
def download_benchmark_prices(ticker: str, start_date: datetime, end_date: datetime) -> pd.Series:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False, threads=True)
    if df is None or len(df) == 0:
        raise RuntimeError("Benchmark download failed.")

    if isinstance(df.columns, pd.MultiIndex):
        ac = df["Adj Close"]
        if isinstance(ac, pd.DataFrame):
            s = ac[ticker] if ticker in ac.columns else ac.iloc[:, 0]
        else:
            s = ac
    else:
        ac = df["Adj Close"]
        s = ac.iloc[:, 0] if isinstance(ac, pd.DataFrame) else ac

    s = pd.Series(s).copy()
    s.name = ticker
    return s


# ----------------------------------------------------------------------
# Function: simple_returns
# ----------------------------------------------------------------------
def simple_returns(adj: pd.DataFrame) -> pd.DataFrame:
    return adj.pct_change(fill_method=None)


# ======================================================================
# Each year we select a smaller investable universe (n_select) based on average dollar volume over the lookback window.
# Then, using the same lookback window, we estimate mean returns and covariance for Markowitz-style optimisation.
# 5) LIQUIDITY SELECTION + MOMENTS
# ======================================================================

# Rank tickers by average dollar volume (price × volume) over the lookback window, with safeguards.
# ----------------------------------------------------------------------
# Function: liquidity_rank
# ----------------------------------------------------------------------
def liquidity_rank(
    clo: pd.DataFrame,
    vol: pd.DataFrame,
    asof_date: pd.Timestamp,
    lookback_days: int,
    top_n: int,
    min_obs: int,
) -> list[str]:
    asof_date = pd.Timestamp(asof_date)
    if asof_date not in clo.index:
        asof_date = clo.index[clo.index.get_loc(asof_date, method="pad")]

    idx = clo.index.get_loc(asof_date)
    lo = max(0, idx - int(lookback_days) + 1)
    win = clo.index[lo:idx + 1]

    dv = (clo.loc[win] * vol.loc[win]).replace([np.inf, -np.inf], np.nan)
    valid_counts = dv.notna().sum(axis=0)
    last_valid = dv.loc[asof_date].notna()

    def _select(min_obs_req: int) -> pd.Series:
        elig = (valid_counts >= int(min_obs_req)) & last_valid
        s = dv.mean(axis=0, skipna=True)
        s = s[elig].dropna()
        return s.sort_values(ascending=False)

    ranked = _select(min_obs)
    if len(ranked) < top_n:
        ranked = _select(max(10, int(min_obs) // 2))
    if len(ranked) < top_n:
        s = dv.mean(axis=0, skipna=True)
        s = s[last_valid].dropna().sort_values(ascending=False)
        ranked = s

    return list(ranked.index[:top_n])


# Estimate (mu, cov) from log/simple returns over the lookback window, after dropping rows with missing values.
# ----------------------------------------------------------------------
# Function: estimate_moments
# ----------------------------------------------------------------------
def estimate_moments(
    ret_all: pd.DataFrame,
    tickers: list[str],
    end_date: pd.Timestamp,
    lookback_days: int,
    min_obs: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    end_date = pd.Timestamp(end_date)
    if end_date not in ret_all.index:
        end_date = ret_all.index[ret_all.index.get_loc(end_date, method="pad")]

    idx = ret_all.index.get_loc(end_date)
    lo = max(0, idx - int(lookback_days) + 1)
    win = ret_all.index[lo:idx + 1]

    R = ret_all.loc[win, tickers].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if len(R) < int(min_obs):
        return None, None

    mu = R.mean(axis=0).to_numpy(dtype=float)
    cov = np.cov(R.to_numpy(dtype=float), rowvar=False, ddof=1)
    return mu, cov


# ======================================================================
# We convert the full daily return panel into rebalance-to-rebalance segments.
# Each segment stores: tickers, estimated moments, and the realised holding-period daily returns.
# 6) REBALANCE DATES + SEGMENTS
# ======================================================================

# Compute the list of rebalance dates for the chosen frequency (month-end or quarter-end trading days).
# ----------------------------------------------------------------------
# Function: make_rebalance_dates
# ----------------------------------------------------------------------
def make_rebalance_dates(mode: str, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
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


# Build rebalance-to-rebalance 'segments' for a given mode (monthly/quarterly).
# Each segment contains: the tickers traded during that holding period, estimated mu/cov from the lookback,
# and the realised returns during the holding window.

# Build the segment list used by both calibration and test backtests for a given rebalance mode.
# Key design choice: universe is refreshed once per year (stable holdings set within each year).
# ----------------------------------------------------------------------
# Function: prepare_mode_segments
# ----------------------------------------------------------------------
def prepare_mode_segments(
    mode: str,
    ret_all: pd.DataFrame,
    clo: pd.DataFrame,
    vol: pd.DataFrame,
) -> tuple[list[dict], list[pd.Timestamp]]:
    all_dates = ret_all.index.dropna()
    rebal_dates = make_rebalance_dates(mode, all_dates)

    min_obs_est = max(int(np.ceil(0.70 * CFG.lookback_days)), 30)
    min_obs_liq = min_obs_est
    min_obs_hold = 5

    update_dates: list[pd.Timestamp] = []
    years_seen: set[int] = set()
    segs: list[dict] = []

    for k, t0 in enumerate(rebal_dates):
        t0 = pd.Timestamp(t0)

        # Universe refresh once per year (stable within each year).
        if t0.year not in years_seen:
            years_seen.add(t0.year)
            update_dates.append(t0)

        if len(segs) == 0 or t0 in update_dates:
            tickers = liquidity_rank(
                clo=clo,
                vol=vol,
                asof_date=t0,
                lookback_days=CFG.lookback_days,
                top_n=CFG.n_select,
                min_obs=min_obs_liq,
            )
        else:
            tickers = segs[-1]["tickers"]

        if len(tickers) < 2:
            continue

        mu, cov = estimate_moments(
            ret_all=ret_all,
            tickers=tickers,
            end_date=t0,
            lookback_days=CFG.lookback_days,
            min_obs=min_obs_est,
        )
        if mu is None or cov is None:
            continue

        try:
            # We precompute the largest eigenvalue of the covariance as a Lipschitz constant proxy for PGD step sizing.
            cov_lmax = float(np.max(np.linalg.eigvalsh(cov)))
            if not np.isfinite(cov_lmax) or cov_lmax <= 0:
                cov_lmax = 1.0
        except Exception:
            cov_lmax = 1.0

        t1 = pd.Timestamp(rebal_dates[k + 1]) if k < len(rebal_dates) - 1 else pd.Timestamp(all_dates[-1])

        hold_idx = ret_all.index[(ret_all.index > t0) & (ret_all.index <= t1)]
        hold_R = (
            ret_all.loc[hold_idx, tickers]
            .copy()
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis=0, how="any")
        )
        if len(hold_R) < min_obs_hold:
            continue

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


# Split segments into calibration and test without look-ahead leakage.
# We 'snap' the calendar split date forward to the next rebalance date.

# Split segments into calibration vs test without look-ahead leakage.
# We snap the calendar split forward to the next rebalance date so no TEST information leaks into calibration.
# ----------------------------------------------------------------------
# Function: split_segments_no_leak
# ----------------------------------------------------------------------
def split_segments_no_leak(segs: list[dict], calendar_split: pd.Timestamp) -> tuple[list[dict], list[dict], pd.Timestamp]:
    calendar_split = pd.Timestamp(calendar_split)
    t0s = [pd.Timestamp(s["t0"]) for s in segs]
    candidates = [t0 for t0 in t0s if t0 > calendar_split]
    if len(candidates) == 0:
        raise RuntimeError("Split failed: no rebalance date found after the calendar split date.")

    test_start = min(candidates)
    cal_segs = [s for s in segs if pd.Timestamp(s["t0"]) < test_start]
    test_segs = [s for s in segs if pd.Timestamp(s["t0"]) >= test_start]

    if len(cal_segs) == 0 or len(test_segs) == 0:
        raise RuntimeError("Split failed: calibration or test segments are empty.")

    return cal_segs, test_segs, test_start


# ======================================================================
# The optimisation step uses PGD under long-only + cap constraints.
# Constraints: 0 <= w_i <= cap and sum(w) <= 1 (cash is the remainder, if allowed). Optional: enforce sum(w) >= 1 - CASH_CAP.
# 7) PROJECTION + PGD SOLVER
# ======================================================================

# ----------------------------------------------------------------------
# Function: _project_to_sum
# ----------------------------------------------------------------------
def _project_to_sum(v: np.ndarray, cap: float, target_sum: float, tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = int(v.size)
    if n == 0:
        return v.copy()

    target_sum = float(max(target_sum, 0.0))
    target_sum = float(min(target_sum, n * float(cap)))

    # Tight tau bracket -> faster bisection
    lo = float(np.min(v) - cap)
    hi = float(np.max(v) + cap)

    def f(tau: float) -> float:
        return float(np.clip(v - tau, 0.0, cap).sum() - target_sum)

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
    s = float(w.sum())
    if s > target_sum + 1e-10 and s > 0.0:
        w *= (target_sum / s)
    return w


# Project a weight vector onto the feasible set: box constraints plus an upper bound on total risky exposure.
# If cash cap is enabled, we also enforce a minimum risky exposure at the end of PGD iterations.
# ----------------------------------------------------------------------
# Function: project_weights
# ----------------------------------------------------------------------
def project_weights(v: np.ndarray, cap: float, *, sum_upper: float = 1.0, min_sum: float = 0.0) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = int(v.size)
    if n == 0:
        return v.copy()

    sum_upper = float(min(max(sum_upper, 0.0), 1.0))
    max_sum = n * float(cap)
    min_sum = float(min(max(min_sum, 0.0), max_sum))

    w = np.clip(v, 0.0, cap)
    s = float(w.sum())

    if s > sum_upper + 1e-12:
        w = _project_to_sum(v, cap=cap, target_sum=sum_upper)
        s = float(w.sum())

    if s < min_sum - 1e-12:
        w = _project_to_sum(v, cap=cap, target_sum=min_sum)

    return w


# Projected Gradient Descent (PGD) solver for the Markowitz-style objective.
# IMPORTANT: This is the *method* core; do not change without re-validating results.

# Solve the Markowitz-style portfolio problem using PGD.
# Objective implemented by the gradient: minimise  -mu^T w + (gamma/2) w^T Sigma w + lambda ||w||_2^2
#
# Interpretation: gamma penalises variance (risk aversion); lambda shrinks weights (ridge regularisation / stability).
# Importantly, when the 2d grid shows that optimal gamma/values pairs are on a boundary, consider
# shifitng their min/max to allow for a better solution
# ----------------------------------------------------------------------
# Function: pgd_solve
# ----------------------------------------------------------------------
def pgd_solve(
    mu: np.ndarray,
    cov: np.ndarray,
    cov_lmax: float,
    gamma: float,
    lam: float,
    cap: float,
    tol: float,
    max_iter: int,
    w0: np.ndarray | None,
) -> np.ndarray:
    """
    PGD solver with tolerance stopping.

    Runtime note:
    - During iterations we only enforce sum(w)<=1 (cheap).
    - If ENABLE_CASH_CAP=True, we enforce sum(w)>=1-CASH_CAP once at the end.
      This keeps the cash cap but avoids slow "exact-sum" projections every iteration.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = int(mu.size)

    if w0 is None or np.asarray(w0).shape != (n,):
        w = np.full(n, 1.0 / n, dtype=float)
    else:
        w = np.asarray(w0, dtype=float).copy()
    w = project_weights(w, cap=cap, sum_upper=1.0, min_sum=0.0)

    # Step size is set using an upper bound on the gradient Lipschitz constant: gamma * lambda_max(Sigma) + 2*lambda.
    L = gamma * float(cov_lmax) + 2.0 * float(lam)
    if not np.isfinite(L) or L <= 0:
        L = 1.0
    step = min(1.0 / L, 0.25)

    for _ in range(int(max_iter)):
        # Gradient of the objective (see comment above). PGD alternates a gradient step and a projection back to constraints.
        grad = -mu + gamma * (cov @ w) + 2.0 * lam * w
        v = w - step * grad
        w_new = project_weights(v, cap=cap, sum_upper=1.0, min_sum=0.0)
        if float(np.linalg.norm(w_new - w, ord=2)) < float(tol):
            w = w_new
            break
        w = w_new

    if ENABLE_CASH_CAP:
        w = project_weights(w, cap=cap, sum_upper=1.0, min_sum=1.0 - float(CASH_CAP))

    return w


# ======================================================================
# Given a list of segments and chosen (gamma, lambda), we simulate daily wealth through the TEST window.
# At each rebalance: compute weights, pay transaction costs, then compound daily returns over the holding period.
# 8) SIMULATION
# ======================================================================

# Backtest the PGD strategy on a sequence of segments. Returns wealth series plus turnover/cost tables and weights.
# ----------------------------------------------------------------------
# Function: simulate_strategy_from_segments
# ----------------------------------------------------------------------
def simulate_strategy_from_segments(
    segs: list[dict],
    gamma: float,
    lam: float,
    tol: float,
    max_iter: int,
    initial_wealth: float,
    store_weights: bool,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    rf_daily = _rf_daily(CFG.rf_annual)

    Vt = float(initial_wealth)
    w_prev: pd.Series | None = None

    wealth_vals: list[float] = []
    wealth_idx: list[pd.Timestamp] = []

    to_rows = []
    c_rows = []
    weights_rows = [] if store_weights else None

    for seg in segs:
        t0 = pd.Timestamp(seg["t0"])
        tickers: list[str] = seg["tickers"]
        mu = seg["mu"]
        cov = seg["cov"]
        cov_lmax = seg["cov_lmax"]
        hold_R: pd.DataFrame = seg["hold_R"]

        w0 = None if w_prev is None else w_prev.reindex(tickers, fill_value=0.0).to_numpy(dtype=float)

        w_r = pgd_solve(
            mu=mu,
            cov=cov,
            cov_lmax=cov_lmax,
            gamma=float(gamma),
            lam=float(lam),
            cap=CFG.cap,
            tol=float(tol),
            max_iter=int(max_iter),
            w0=w0,
        )

        w_curr = pd.Series(w_r, index=tickers, dtype=float)
        # Implied cash weight is whatever is left after allocating to risky assets.
        w_c = max(0.0, 1.0 - float(np.sum(w_r)))

        # Turnover is the L1 change in risky weights at the rebalance.
        # It proxies how much trading we do, and is used to compute transaction costs.
        if w_prev is None:
            turnover = float(w_curr.abs().sum())
        else:
            turnover = float(w_curr.subtract(w_prev, fill_value=0.0).abs().sum())

        # Simple proportional cost model: cost = kappa * turnover * current wealth.
        # We subtract costs immediately at the rebalance before the next holding period starts.
        cost = (CFG.kappa * turnover * Vt) if CFG.apply_costs else 0.0
        Vt = max(0.0, Vt - cost)

        to_rows.append(pd.Series({"turnover": turnover}, name=t0))
        c_rows.append(pd.Series({"cost": cost}, name=t0))

        if store_weights:
            for tkr, wi in w_curr.items():
                weights_rows.append({"rebalance_date": t0, "ticker": tkr, "weight": float(wi)})

        if len(hold_R) > 0:
            R = hold_R.to_numpy(dtype=float)
            # Daily gross-return factor for the portfolio over this holding window.
            # - risky sleeve contributes (R @ w_r)
            # - cash sleeve (implied) contributes w_c * rf_daily
            # Daily portfolio gross-return factor: 1 + risky_return + cash_return.
            # Cash earns rf_daily on the leftover weight (1 - sum(risky weights)).
            factors = 1.0 + (R @ w_r) + w_c * rf_daily
            seg_wealth = Vt * np.cumprod(factors)
            Vt = float(seg_wealth[-1])
            wealth_vals.extend(seg_wealth.tolist())
            wealth_idx.extend([pd.Timestamp(d) for d in hold_R.index])

        w_prev = w_curr

    V = pd.Series(wealth_vals, index=pd.DatetimeIndex(wealth_idx)).sort_index()
    TO = pd.DataFrame(to_rows)
    C = pd.DataFrame(c_rows)
    W = pd.DataFrame(weights_rows) if store_weights else None
    return V, TO, C, W


# Baseline backtest: equal-weight subject to the same long-only + cap (+ optional cash cap) constraints.
# ----------------------------------------------------------------------
# Function: simulate_equal_weight_from_segments
# ----------------------------------------------------------------------
def simulate_equal_weight_from_segments(
    segs: list[dict],
    initial_wealth: float,
    store_weights: bool,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    rf_daily = _rf_daily(CFG.rf_annual)

    Vt = float(initial_wealth)
    w_prev: pd.Series | None = None

    wealth_vals: list[float] = []
    wealth_idx: list[pd.Timestamp] = []

    to_rows = []
    c_rows = []
    weights_rows = [] if store_weights else None

    for seg in segs:
        t0 = pd.Timestamp(seg["t0"])
        tickers: list[str] = seg["tickers"]
        hold_R: pd.DataFrame = seg["hold_R"]

        n = len(tickers)
        if n == 0:
            continue

        v = np.full(n, 1.0 / n, dtype=float)
        w_r = project_weights(
            v,
            cap=CFG.cap,
            sum_upper=1.0,
            min_sum=(1.0 - float(CASH_CAP)) if ENABLE_CASH_CAP else 0.0,
        )
        w_curr = pd.Series(w_r, index=tickers, dtype=float)
        # Implied cash weight is whatever is left after allocating to risky assets.
        w_c = max(0.0, 1.0 - float(np.sum(w_r)))

        # Turnover is the L1 change in risky weights at the rebalance.
        # It proxies how much trading we do, and is used to compute transaction costs.
        if w_prev is None:
            turnover = float(w_curr.abs().sum())
        else:
            turnover = float(w_curr.subtract(w_prev, fill_value=0.0).abs().sum())

        # Simple proportional cost model: cost = kappa * turnover * current wealth.
        # We subtract costs immediately at the rebalance before the next holding period starts.
        cost = (CFG.kappa * turnover * Vt) if CFG.apply_costs else 0.0
        Vt = max(0.0, Vt - cost)

        to_rows.append(pd.Series({"turnover": turnover}, name=t0))
        c_rows.append(pd.Series({"cost": cost}, name=t0))

        if store_weights:
            for tkr, wi in w_curr.items():
                weights_rows.append({"rebalance_date": t0, "ticker": tkr, "weight": float(wi)})

        if len(hold_R) > 0:
            R = hold_R.to_numpy(dtype=float)
            # Daily gross-return factor for the portfolio over this holding window.
            # - risky sleeve contributes (R @ w_r)
            # - cash sleeve (implied) contributes w_c * rf_daily
            factors = 1.0 + (R @ w_r) + w_c * rf_daily
            seg_wealth = Vt * np.cumprod(factors)
            Vt = float(seg_wealth[-1])
            wealth_vals.extend(seg_wealth.tolist())
            wealth_idx.extend([pd.Timestamp(d) for d in hold_R.index])

        w_prev = w_curr

    V = pd.Series(wealth_vals, index=pd.DatetimeIndex(wealth_idx)).sort_index()
    TO = pd.DataFrame(to_rows)
    C = pd.DataFrame(c_rows)
    W = pd.DataFrame(weights_rows) if store_weights else None
    return V, TO, C, W


# ----------------------------------------------------------------------
# Function: build_spy_buyhold_wealth
# ----------------------------------------------------------------------
def build_spy_buyhold_wealth(bench_prices: pd.Series, target_index: pd.DatetimeIndex, initial_wealth: float) -> pd.Series:
    p = pd.Series(bench_prices).sort_index().copy()
    p_aligned = p.reindex(target_index, method="pad").dropna()
    if len(p_aligned) == 0:
        raise RuntimeError("SPY alignment failed: no prices available on the target date index.")
    p0 = float(p_aligned.iloc[0])
    V = float(initial_wealth) * (p_aligned / p0)
    V.name = BENCHMARK_TICKER
    return V


# ======================================================================
# Performance metrics are computed from the daily wealth curve: annualised vol, Sharpe, max drawdown, etc.
# These are for reporting; they do not feed back into the optimisation.
# 9) METRICS
# ======================================================================

# ----------------------------------------------------------------------
# Function: annual_vol
# ----------------------------------------------------------------------
def annual_vol(R: pd.Series) -> float:
    R = pd.Series(R).dropna()
    if len(R) < 2:
        return float("nan")
    return float(R.std(ddof=1) * np.sqrt(252.0))


# ----------------------------------------------------------------------
# Function: max_drawdown
# ----------------------------------------------------------------------
def max_drawdown(V: pd.Series) -> float:
    V = pd.Series(V).dropna()
    if len(V) < 2:
        return float("nan")
    peak = V.cummax()
    dd = (V / peak) - 1.0
    return float(dd.min())


# ----------------------------------------------------------------------
# Function: drawdown_series
# ----------------------------------------------------------------------
def drawdown_series(V: pd.Series) -> pd.Series:
    V = pd.Series(V).dropna()
    peak = V.cummax()
    return (V / peak) - 1.0


# Performance metrics computed from the daily wealth series.
# Used for compact tables (final wealth, annualised vol, max drawdown, etc.).

# Compute key reporting metrics from wealth: final value, CAGR, annualised vol, Sharpe (excess over rf), max DD.
# ----------------------------------------------------------------------
# Function: perf_metrics
# ----------------------------------------------------------------------
def perf_metrics(V: pd.Series) -> dict:
    V = pd.Series(V).dropna()
    if len(V) < 10:
        return {}

    Rp = V.pct_change(fill_method=None).dropna()
    if len(Rp) < 10:
        return {}

    cagr = float((V.iloc[-1] / V.iloc[0]) ** (252.0 / len(Rp)) - 1.0)
    vol = annual_vol(Rp)
    sharpe = float((Rp.mean() * 252.0 - CFG.rf_annual) / vol) if (np.isfinite(vol) and vol > 1e-12) else float("nan")
    mdd = max_drawdown(V)

    return {
        "FinalV": float(V.iloc[-1]),
        "CAGR": cagr,
        "AnnVol": float(vol),
        "Sharpe": sharpe,
        "MaxDD": float(mdd),
    }


# ======================================================================
# Calibration chooses (gamma, lambda) using only the calibration segments.
# We run a coarse grid search followed by small 'fine' local refinements around the coarse winners.
# 10) CALIBRATION (coarse + fine; no FINAL stage)
# ======================================================================

# Evaluate a grid of (gamma, lambda) pairs by running a full backtest on the calibration segments for each pair.
# This is the most expensive part of the script; runtime roughly scales with (#pairs × #segments × PGD iterations).
# ----------------------------------------------------------------------
# Function: run_grid_search
# ----------------------------------------------------------------------
def run_grid_search(
    segs: list[dict],
    gamma_vals: np.ndarray,
    lam_vals: np.ndarray,
    tol: float,
    max_iter: int,
    tag: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    best_seen = -np.inf

    for i, g in enumerate(gamma_vals):
        for lam in lam_vals:
            V, _, _, _ = simulate_strategy_from_segments(
                segs=segs,
                gamma=float(g),
                lam=float(lam),
                tol=tol,
                max_iter=max_iter,
                initial_wealth=CFG.initial_wealth,
                store_weights=False,
            )
            Rp = V.pct_change(fill_method=None).dropna()
            v_ann = annual_vol(Rp)
            finalV = float(V.iloc[-1]) if len(V) else float("nan")

            rows.append({"gamma": float(g), "lambda": float(lam), "vol": float(v_ann), "finalV": float(finalV)})

            if np.isfinite(finalV) and finalV > best_seen:
                best_seen = float(finalV)

        print(
            f"  [{tag}] gamma row {i+1}/{len(gamma_vals)} gamma={float(g):g} best wealth so far={best_seen:,.0f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    # Volatility targeting error: absolute difference between achieved annual vol and the configured target.
    df["err"] = (df["vol"] - CFG.target_vol_annual).abs()
    return df


# Select parameters for the return-seeking strategy:
# 1) maximise final wealth, 2) among near-optimal wealth (within eps), pick minimum vol, 3) prefer larger gamma then lambda.
# ----------------------------------------------------------------------
# Function: pick_return
# ----------------------------------------------------------------------
def pick_return(grid_df: pd.DataFrame) -> tuple[float, float]:
    df = grid_df[np.isfinite(grid_df["finalV"])].copy()
    if len(df) == 0:
        raise RuntimeError("No finite finalV values in calibration grid.")
    best = float(df["finalV"].max())
    thresh = (1.0 - float(CFG.near_opt_eps_return)) * best
    cand = df[df["finalV"] >= thresh].copy()
    cand = cand.sort_values(["vol", "gamma", "lambda"], ascending=[True, False, False]).reset_index(drop=True)
    row = cand.iloc[0]
    return float(row["gamma"]), float(row["lambda"])


# Select parameters for the risk-targeted strategy:
# 1) minimise vol error to target, 2) tie-break by higher final wealth, 3) prefer larger gamma for stability.
# ----------------------------------------------------------------------
# Function: pick_risk
# ----------------------------------------------------------------------
def pick_risk(grid_df: pd.DataFrame) -> tuple[float, float]:
    df = grid_df[np.isfinite(grid_df["err"])].copy()
    if len(df) == 0:
        raise RuntimeError("No finite err values in calibration grid.")
    df = df.sort_values(["err", "finalV", "gamma"], ascending=[True, False, False]).reset_index(drop=True)
    row = df.iloc[0]
    return float(row["gamma"]), float(row["lambda"])


# ----------------------------------------------------------------------
# Function: refine_gamma
# ----------------------------------------------------------------------
def refine_gamma(
    segs: list[dict],
    center_gamma: float,
    lam_fixed: float,
    tol: float,
    max_iter: int,
    tag: str,
) -> pd.DataFrame:
    g_low = clamp(center_gamma - PROFILE.gamma_window, CFG.gamma_min, CFG.gamma_max)
    g_high = clamp(center_gamma + PROFILE.gamma_window, CFG.gamma_min, CFG.gamma_max)
    gamma_vals = fixed_step_grid(g_low, g_high, 1.0)
    gamma_vals = ensure_in_grid(gamma_vals, center_gamma)
    return run_grid_search(segs, gamma_vals, np.array([lam_fixed], dtype=float), tol, max_iter, tag)


# ----------------------------------------------------------------------
# Function: refine_lambda
# ----------------------------------------------------------------------
def refine_lambda(
    segs: list[dict],
    gamma_fixed: float,
    center_lam: float,
    tol: float,
    max_iter: int,
    tag: str,
) -> pd.DataFrame:
    l_low = max(CFG.lam_min, float(center_lam) / float(PROFILE.lam_factor_window))
    l_high = min(CFG.lam_max, float(center_lam) * float(PROFILE.lam_factor_window))
    lam_vals = log_grid(l_low, l_high, int(PROFILE.lam_n_fine))
    lam_vals = ensure_in_grid(lam_vals, center_lam)
    return run_grid_search(segs, np.array([gamma_fixed], dtype=float), lam_vals, tol, max_iter, tag)


# ======================================================================
# Plotting helpers. These only read precomputed series/tables and produce figures (no effect on computations).
# Figures are both shown and saved to disk when the relevant toggles are enabled.
# 11) PLOTS
# ======================================================================

# ----------------------------------------------------------------------
# Function: plot_calibration_heatmap
# ----------------------------------------------------------------------
def plot_calibration_heatmap(
    mode: str,
    grid_df: pd.DataFrame,
    gamma_vals: np.ndarray,
    lam_vals: np.ndarray,
    chosen_points: list[dict],
    out_path: Path,
) -> None:
    M = (
        grid_df.pivot(index="gamma", columns="lambda", values="finalV")
        .reindex(index=list(gamma_vals), columns=list(lam_vals))
        .to_numpy()
    )
    MM = np.ma.masked_invalid(M)

    plt.figure(figsize=(10.5, 4.8))
    im = plt.imshow(MM, origin="lower", aspect="auto")
    plt.colorbar(im, label="final wealth (calibration)")

    G, L = len(gamma_vals), len(lam_vals)
    x_ticks = np.linspace(0, L - 1, min(L, 8), dtype=int)
    y_ticks = np.linspace(0, G - 1, min(G, 8), dtype=int)

    plt.xticks(x_ticks, [f"{lam_vals[j]:g}" for j in x_ticks], rotation=45, ha="right")
    plt.yticks(y_ticks, [f"{gamma_vals[i]:g}" for i in y_ticks])

    plt.xlabel("lambda")
    plt.ylabel("gamma")
    plt.title(f"{mode}: calibration coarse grid (final wealth)")

    ax = plt.gca()
    for p in chosen_points:
        g = float(p["gamma"])
        lam = float(p["lambda"])
        label = p.get("label", "")
        marker = p.get("marker", "x")

        # chosen may not be exactly on coarse grid -> highlight nearest coarse cell
        gi = int(np.argmin(np.abs(np.asarray(gamma_vals) - g)))
        lj = int(np.argmin(np.abs(np.asarray(lam_vals) - lam)))

        ax.add_patch(Rectangle((lj - 0.5, gi - 0.5), 1.0, 1.0, fill=False, linewidth=2.0))
        plt.scatter([lj], [gi], s=90, marker=marker, label=label if label else None)

    if any(p.get("label", "") for p in chosen_points):
        plt.legend(loc="best")

    plt.tight_layout()
    if SAVE_PLOTS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ----------------------------------------------------------------------
# Function: plot_wealth_comparison
# ----------------------------------------------------------------------
def plot_wealth_comparison(mode: str, series: dict[str, pd.Series], out_path: Path) -> None:
    plt.figure(figsize=(11, 4.6))
    for name, V in series.items():
        V = pd.Series(V).dropna()
        if len(V) > 0:
            plt.plot(V.index, V.values, label=name)
    plt.title(f"{mode}: TEST wealth comparison")
    plt.xlabel("date")
    plt.ylabel("portfolio value")
    plt.legend(loc="best")
    plt.tight_layout()
    if SAVE_PLOTS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ----------------------------------------------------------------------
# Function: plot_drawdown_comparison
# ----------------------------------------------------------------------
def plot_drawdown_comparison(mode: str, series: dict[str, pd.Series], out_path: Path) -> None:
    plt.figure(figsize=(11, 4.6))
    for name, V in series.items():
        dd = drawdown_series(V)
        plt.plot(dd.index, dd.values, label=name)
    plt.title(f"{mode}: TEST drawdown comparison")
    plt.xlabel("date")
    plt.ylabel("drawdown")
    plt.legend(loc="best")
    plt.tight_layout()
    if SAVE_PLOTS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ----------------------------------------------------------------------
# Function: plot_turnover_comparison
# ----------------------------------------------------------------------
def plot_turnover_comparison(mode: str, series: dict[str, pd.DataFrame], out_path: Path) -> None:
    plt.figure(figsize=(11, 4.6))
    for name, df in series.items():
        if "turnover" in df.columns and len(df) > 0:
            plt.plot(df.index, df["turnover"].values, label=name)
    plt.title(f"{mode}: TEST turnover at rebalances")
    plt.xlabel("rebalance date")
    plt.ylabel("turnover (L1)")
    plt.legend(loc="best")
    plt.tight_layout()
    if SAVE_PLOTS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()




# Extra diagnostics requested for interpretation/reporting.
# They are computed from TEST outputs (wealth curves, stored weights, and calibration coarse grid).
# ---- Additional TEST plots 
# These functions create extra figures for interpretation and reportingg


# ----------------------------------------------------------------------
# Function: _rolling_sharpe_series
# ----------------------------------------------------------------------
def _rolling_sharpe_series(R: pd.Series, rf_daily: float, window_days: int) -> pd.Series:
    """Rolling (annualised) Sharpe ratio of a daily return series.

    Sharpe_t = sqrt(252) * mean(excess_returns over window) / std(excess_returns over window).
    """
    R = pd.Series(R).dropna()
    if window_days <= 1:
        raise ValueError("window_days must be >= 2 for rolling Sharpe.")
    excess = R - float(rf_daily)
    mu = excess.rolling(window_days).mean()
    sd = excess.rolling(window_days).std(ddof=0)
    sharpe = (mu / sd) * np.sqrt(252.0)
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
    return sharpe


# ----------------------------------------------------------------------
# Function: plot_rolling_sharpe_comparison
# ----------------------------------------------------------------------
def plot_rolling_sharpe_comparison(
    mode: str,
    wealth_series: dict[str, pd.Series],
    out_path: Path,
    window_days: int = 63,
) -> None:
    """Plot rolling Sharpe ratios on the TEST window for multiple wealth series."""
    rf_daily = _rf_daily(CFG.rf_annual)

    plt.figure(figsize=(11, 4.6))
    for name, V in wealth_series.items():
        V = pd.Series(V).dropna()
        if len(V) < window_days + 2:
            continue
        R = V.pct_change(fill_method=None).dropna()
        S = _rolling_sharpe_series(R, rf_daily=rf_daily, window_days=int(window_days))
        if len(S.dropna()) > 0:
            plt.plot(S.index, S.values, label=name)

    plt.title(f"{mode}: TEST rolling Sharpe (window={window_days} trading days)")
    plt.xlabel("date")
    plt.ylabel("rolling Sharpe (annualised)")
    plt.legend(loc="best")
    plt.tight_layout()
    if SAVE_PLOTS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ----------------------------------------------------------------------
# Function: plot_top_weights_over_time
# ----------------------------------------------------------------------
def plot_top_weights_over_time(
    mode: str,
    weights_long: pd.DataFrame,
    out_path: Path,
    top_k: int = 10,
    title_suffix: str = "",
) -> None:
    """Plot the top-k weights over time (rebalance dates).

    weights_long is the long-format DataFrame returned by simulate_* functions:
      columns: rebalance_date, ticker, weight.
    """
    if weights_long is None or len(weights_long) == 0:
        return

    df = weights_long.copy()
    # Ensure the x-axis is a datetime index.
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    wide = (
        df.pivot_table(index="rebalance_date", columns="ticker", values="weight", aggfunc="first")
        .sort_index()
        .fillna(0.0)
    )

    # Choose "top" tickers by average absolute weight across the test period.
    avg_abs = wide.abs().mean(axis=0).sort_values(ascending=False)
    top = list(avg_abs.head(int(top_k)).index)

    plt.figure(figsize=(11, 4.6))
    for tkr in top:
        plt.plot(wide.index, wide[tkr].values, label=str(tkr))

    ttl = f"{mode}: TEST top-{int(top_k)} weights over time"
    if title_suffix:
        ttl += f" ({title_suffix})"
    plt.title(ttl)
    plt.xlabel("rebalance date")
    plt.ylabel("portfolio weight")
    plt.legend(loc="best", ncol=2, fontsize=8)
    plt.tight_layout()
    if SAVE_PLOTS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ----------------------------------------------------------------------
# Function: plot_calibration_best_lambda_by_gamma
# ----------------------------------------------------------------------
def plot_calibration_best_lambda_by_gamma(
    mode: str,
    grid_df: pd.DataFrame,
    out_path: Path,
    chosen_points: list[dict] | None = None,
) -> None:
    """Calibration diagnostic: for each gamma, show best final wealth over lambdas."""
    if grid_df is None or len(grid_df) == 0:
        return

    df = grid_df.copy()
    # For each gamma row, pick lambda giving the maximum finalV.
    best = (
        df.sort_values(["gamma", "finalV"], ascending=[True, False])
        .groupby("gamma", as_index=False)
        .first()
        .sort_values("gamma")
    )

    plt.figure(figsize=(11, 4.6))
    plt.plot(best["gamma"].values, best["finalV"].values, marker="o", linewidth=1.5)

    # Mark chosen parameter pairs (if provided) for quick visual checking.
    if chosen_points:
        for p in chosen_points:
            g = float(p["gamma"])
            # Find the closest gamma on the coarse grid.
            gi = int(np.argmin(np.abs(best["gamma"].to_numpy(dtype=float) - g)))
            plt.scatter([best["gamma"].iloc[gi]], [best["finalV"].iloc[gi]], s=80, marker=p.get("marker", "x"))

    plt.title(f"{mode}: calibration coarse diagnostic (best lambda per gamma)")
    plt.xlabel("gamma")
    plt.ylabel("best final wealth (over lambdas)")
    plt.tight_layout()
    if SAVE_PLOTS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ----------------------------------------------------------------------
# Function: plot_calibration_err_heatmap
# ----------------------------------------------------------------------
def plot_calibration_err_heatmap(mode: str, grid_df: pd.DataFrame, out_path: Path) -> None:
    """Calibration diagnostic: heatmap of absolute volatility error |vol - target|."""
    if grid_df is None or len(grid_df) == 0:
        return

    gammas = np.sort(grid_df["gamma"].unique())
    lams = np.sort(grid_df["lambda"].unique())

    M = (
        grid_df.pivot(index="gamma", columns="lambda", values="err")
        .reindex(index=list(gammas), columns=list(lams))
        .to_numpy()
    )
    MM = np.ma.masked_invalid(M)

    plt.figure(figsize=(10.5, 4.8))
    im = plt.imshow(MM, origin="lower", aspect="auto")
    plt.colorbar(im, label="|ann_vol - target_vol|")

    G, L = len(gammas), len(lams)
    x_ticks = np.linspace(0, L - 1, min(L, 8), dtype=int)
    y_ticks = np.linspace(0, G - 1, min(G, 8), dtype=int)

    plt.xticks(x_ticks, [f"{lams[j]:g}" for j in x_ticks], rotation=45, ha="right")
    plt.yticks(y_ticks, [f"{gammas[i]:g}" for i in y_ticks])

    plt.xlabel("lambda")
    plt.ylabel("gamma")
    plt.title(f"{mode}: calibration coarse grid (vol error heatmap)")
    plt.tight_layout()
    if SAVE_PLOTS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ======================================================================
# Orchestrates coarse grid + fine refinement, saves outputs, and produces calibration plots.
# 12) CALIBRATION WRAPPER
# ======================================================================

# ----------------------------------------------------------------------
# Function: calibrate_on_segments
# ----------------------------------------------------------------------
def calibrate_on_segments(mode: str, cal_segs: list[dict], mode_dir: Path) -> tuple[tuple[float, float], tuple[float, float], pd.DataFrame]:
    gamma_c = fixed_step_grid(CFG.gamma_min, CFG.gamma_max, float(PROFILE.gamma_step_coarse))
    lam_c = log_grid(CFG.lam_min, CFG.lam_max, int(PROFILE.lam_n_coarse))

    explain_block(
        "Calibration: coarse grid",
        [
            f"gamma: {CFG.gamma_min:g}..{CFG.gamma_max:g} step {PROFILE.gamma_step_coarse:g}  (n={len(gamma_c)})",
            f"lambda: {CFG.lam_min:g}..{CFG.lam_max:g} logspace (n={len(lam_c)})",
            f"combos: {len(gamma_c) * len(lam_c)}",
            f"solver (cal): tol={PROFILE.tol_cal:g}, max_iter={PROFILE.max_iter_cal}",
            f"profile: {RUN_PROFILE.upper()}",
        ],
    )

    # Save a run manifest (inputs and key switches) so results can be reproduced later.
    if SAVE_CSVS:
        manifest = {
            "run_id": RUN_ID,
            "start_date": f"{START_DATE:%Y-%m-%d}" if START_DATE is not None else "",
            "end_date": f"{END_DATE:%Y-%m-%d}" if END_DATE is not None else "",
            "years_total": CFG.years_total,
            "test_years": CFG.test_years,
            "lookback_days": CFG.lookback_days,
            "parent_size": CFG.parent_size,
            "n_select": CFG.n_select,
            "cap": CFG.cap,
            "rf_annual": CFG.rf_annual,
            "kappa": CFG.kappa,
            "apply_costs": CFG.apply_costs,
            "run_profile": RUN_PROFILE.upper(),
            "calibration_scope": "PER_MODE_NO_REUSE",
            "enable_cash_cap": ENABLE_CASH_CAP,
            "cash_cap": CASH_CAP if ENABLE_CASH_CAP else np.nan,
            "benchmark": BENCHMARK_TICKER,
        }
        save_kv_csv(manifest, OUT_DIR / "run_manifest.csv")

    t0 = time.perf_counter()
    grid_c = run_grid_search(
        segs=cal_segs,
        gamma_vals=gamma_c,
        lam_vals=lam_c,
        tol=float(PROFILE.tol_cal),
        max_iter=int(PROFILE.max_iter_cal),
        tag=f"{mode}-cal-coarse",
    )
    coarse_s = time.perf_counter() - t0

    if SAVE_CSVS:
        save_csv(grid_c.set_index(["gamma", "lambda"]), mode_dir / "calibration_grid_coarse.csv")

    g_ret_c, l_ret_c = pick_return(grid_c)
    g_risk_c, l_risk_c = pick_risk(grid_c)

    explain_block(
        "Calibration: coarse winners",
        [
            f"Coarse return-seeking: gamma={g_ret_c:g}, lambda={l_ret_c:g}",
            f"Coarse risk-targeted:  gamma={g_risk_c:g}, lambda={l_risk_c:g}  (target vol={CFG.target_vol_annual:.2%})",
            f"Coarse grid time: {coarse_s:.2f} seconds",
        ],
    )

    # Fine refinement
    t1 = time.perf_counter()

    grid_ret_g = refine_gamma(cal_segs, g_ret_c, l_ret_c, float(PROFILE.tol_cal), int(PROFILE.max_iter_cal), f"{mode}-fine-ret-gamma")
    g_ret_f, _ = pick_return(grid_ret_g)

    grid_ret_l = refine_lambda(cal_segs, g_ret_f, l_ret_c, float(PROFILE.tol_cal), int(PROFILE.max_iter_cal), f"{mode}-fine-ret-lambda")
    _, l_ret_f = pick_return(grid_ret_l)

    grid_risk_g = refine_gamma(cal_segs, g_risk_c, l_risk_c, float(PROFILE.tol_cal), int(PROFILE.max_iter_cal), f"{mode}-fine-risk-gamma")
    g_risk_f, _ = pick_risk(grid_risk_g)

    grid_risk_l = refine_lambda(cal_segs, g_risk_f, l_risk_c, float(PROFILE.tol_cal), int(PROFILE.max_iter_cal), f"{mode}-fine-risk-lambda")
    _, l_risk_f = pick_risk(grid_risk_l)

    fine_s = time.perf_counter() - t1

    ret_params = (float(g_ret_f), float(l_ret_f))
    risk_params = (float(g_risk_f), float(l_risk_f))

    explain_block(
        "Calibration: final chosen params (no FINAL stage)",
        [
            f"Return-seeking (final): gamma={ret_params[0]:g}, lambda={ret_params[1]:g}",
            f"Risk-targeted  (final): gamma={risk_params[0]:g}, lambda={risk_params[1]:g}  (target vol={CFG.target_vol_annual:.2%})",
            f"Fine refinement time: {fine_s:.2f} seconds",
        ],
    )

    if SAVE_CSVS:
        chosen_df = pd.DataFrame(
            [
                {"choice": "return_seeking", "gamma": ret_params[0], "lambda": ret_params[1]},
                {"choice": "risk_targeted", "gamma": risk_params[0], "lambda": risk_params[1]},
            ]
        ).set_index("choice")
        save_csv(chosen_df, mode_dir / "chosen_params.csv")

        save_csv(grid_ret_g.set_index(["gamma", "lambda"]), mode_dir / "calibration_fine_return_gamma.csv")
        save_csv(grid_ret_l.set_index(["gamma", "lambda"]), mode_dir / "calibration_fine_return_lambda.csv")
        save_csv(grid_risk_g.set_index(["gamma", "lambda"]), mode_dir / "calibration_fine_risk_gamma.csv")
        save_csv(grid_risk_l.set_index(["gamma", "lambda"]), mode_dir / "calibration_fine_risk_lambda.csv")

    # All plots are created here. Adding plots here does not change strategy outputs; it only increases runtime slightly for figure rendering.
    if SAVE_PLOTS or SHOW_PLOTS:
        plot_calibration_heatmap(
            mode=mode,
            grid_df=grid_c,
            gamma_vals=gamma_c,
            lam_vals=lam_c,
            chosen_points=[
                {"gamma": ret_params[0], "lambda": ret_params[1], "label": "return", "marker": "x"},
                {"gamma": risk_params[0], "lambda": risk_params[1], "label": "risk", "marker": "o"},
            ],
            out_path=mode_dir / "calibration_heatmap_coarse.png",
        )

    return ret_params, risk_params, grid_c


# ======================================================================
# Runs the full pipeline for a single rebalance frequency (monthly or quarterly): build segments -> split -> calibrate -> test -> outputs.
# 13) RUN ONE MODE
# ======================================================================

# ----------------------------------------------------------------------
# Function: run_mode
# ----------------------------------------------------------------------
def run_mode(
    mode: str,
    ret_all: pd.DataFrame,
    clo: pd.DataFrame,
    vol: pd.DataFrame,
    bench_prices: pd.Series,
    params_by_mode: dict[str, tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    mode_dir = OUT_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    explain_block(
        f"Mode: {mode}",
        [
            "Stages:",
            "  1) build segments",
            "  2) split into calibration and test (date-based, snapped to rebalance to avoid leakage)",
            "  3) calibration grid search (optional)",
            "  4) test backtests: two strategies + equal-weight baseline + SPY",
            "  5) save CSV outputs + show/save PNG plots",
        ],
    )

    t_prep = time.perf_counter()
    segs, update_dates = prepare_mode_segments(mode, ret_all, clo, vol)
    prep_s = time.perf_counter() - t_prep

    calendar_split = pd.Timestamp(datetime.today() - relativedelta(years=CFG.test_years))
    cal_segs, test_segs, test_start = split_segments_no_leak(segs, calendar_split=calendar_split)

    explain_block(
        "Segments and split",
        [
            f"Total segments: {len(segs)}",
            f"Yearly universe refresh dates: {len(update_dates)}",
            f"Prep time: {prep_s:.2f} seconds",
            "",
            f"Calendar split date: {calendar_split.date()}",
            f"Snapped TEST start rebalance date: {pd.Timestamp(test_start).date()}",
            f"Calibration segments: {len(cal_segs)}",
            f"Test segments: {len(test_segs)}",
        ],
    )

    if SAVE_CSVS:
        save_kv_csv(
            {
                "mode": mode,
                "run_id": RUN_ID,
                "profile": RUN_PROFILE.upper(),
                "calibration_scope": "PER_MODE_NO_REUSE",
                "cash_cap_enabled": ENABLE_CASH_CAP,
                "cash_cap": CASH_CAP,
                "years_total": CFG.years_total,
                "test_years": CFG.test_years,
                "calendar_split_date": str(calendar_split.date()),
                "snapped_test_start_rebalance": str(pd.Timestamp(test_start).date()),
                "lookback_days": CFG.lookback_days,
                "parent_size": CFG.parent_size,
                "n_select": CFG.n_select,
                "cap": CFG.cap,
                "rf_annual": CFG.rf_annual,
                "kappa": CFG.kappa,
                "apply_costs": CFG.apply_costs,
                "target_vol_annual": CFG.target_vol_annual,
                "tol_cal": PROFILE.tol_cal,
                "max_iter_cal": PROFILE.max_iter_cal,
                "tol_test": PROFILE.tol_test,
                "max_iter_test": PROFILE.max_iter_test,
            },
            mode_dir / "config_used.csv",
        )

    # Calibrate (gamma, lambda) on the CALIBRATION window for this mode.
    # We then freeze these parameters and evaluate out-of-sample on TEST.
    ret_params, risk_params, grid_c = calibrate_on_segments(mode, cal_segs, mode_dir)
    params_by_mode[mode] = (ret_params, risk_params)

    explain_block(
        "TEST backtests",
        [
            "We now freeze (gamma, lambda) from calibration and evaluate out-of-sample on TEST.",
            f"Strategy return-seeking: gamma={ret_params[0]:g}, lambda={ret_params[1]:g}",
            f"Strategy risk-targeted:  gamma={risk_params[0]:g}, lambda={risk_params[1]:g}",
            "Baseline: equal-weight on the same investable universe.",
            f"Benchmark: {BENCHMARK_TICKER} buy-and-hold.",
        ],
    )

    V_ret, TO_ret, C_ret, W_ret = simulate_strategy_from_segments(
        segs=test_segs,
        gamma=ret_params[0],
        lam=ret_params[1],
        tol=float(PROFILE.tol_test),
        max_iter=int(PROFILE.max_iter_test),
        initial_wealth=CFG.initial_wealth,
        store_weights=SAVE_WEIGHTS_CSV,
    )
    V_risk, TO_risk, C_risk, W_risk = simulate_strategy_from_segments(
        segs=test_segs,
        gamma=risk_params[0],
        lam=risk_params[1],
        tol=float(PROFILE.tol_test),
        max_iter=int(PROFILE.max_iter_test),
        initial_wealth=CFG.initial_wealth,
        store_weights=SAVE_WEIGHTS_CSV,
    )
    V_eq, TO_eq, C_eq, W_eq = simulate_equal_weight_from_segments(
        segs=test_segs,
        initial_wealth=CFG.initial_wealth,
        store_weights=SAVE_WEIGHTS_CSV,
    )

    common_idx = V_ret.index.intersection(V_risk.index).intersection(V_eq.index)
    V_ret = V_ret.reindex(common_idx).dropna()
    V_risk = V_risk.reindex(common_idx).dropna()
    V_eq = V_eq.reindex(common_idx).dropna()

    V_spy = build_spy_buyhold_wealth(bench_prices, target_index=common_idx, initial_wealth=CFG.initial_wealth)

    met_ret = perf_metrics(V_ret)
    met_risk = perf_metrics(V_risk)
    met_eq = perf_metrics(V_eq)
    met_spy = perf_metrics(V_spy)

    summary = pd.DataFrame(
        [
            {"series": "strategy_return", **met_ret, "gamma": ret_params[0], "lambda": ret_params[1]},
            {"series": "strategy_risk", **met_risk, "gamma": risk_params[0], "lambda": risk_params[1]},
            {"series": "equal_weight", **met_eq, "gamma": np.nan, "lambda": np.nan},
            {"series": BENCHMARK_TICKER, **met_spy, "gamma": np.nan, "lambda": np.nan},
        ]
    ).set_index("series")

    explain_block(
        "TEST metrics summary",
        [
            str(
                summary[["FinalV", "AnnVol", "MaxDD", "gamma", "lambda"]]
                .rename(columns={"FinalV": "final_wealth", "AnnVol": "ann_vol", "MaxDD": "max_dd"})
            ),
        ],
    )

    if SAVE_CSVS:
        save_csv(summary, mode_dir / "test_metrics_summary.csv")
        save_csv(V_ret.to_frame("strategy_return"), mode_dir / "wealth_strategy_return.csv")
        save_csv(V_risk.to_frame("strategy_risk"), mode_dir / "wealth_strategy_risk.csv")
        save_csv(V_eq.to_frame("equal_weight"), mode_dir / "wealth_equal_weight.csv")
        save_csv(V_spy.to_frame(BENCHMARK_TICKER), mode_dir / f"wealth_{BENCHMARK_TICKER}.csv")

        save_csv(TO_ret, mode_dir / "turnover_strategy_return.csv")
        save_csv(TO_risk, mode_dir / "turnover_strategy_risk.csv")
        save_csv(TO_eq, mode_dir / "turnover_equal_weight.csv")

        save_csv(C_ret, mode_dir / "costs_strategy_return.csv")
        save_csv(C_risk, mode_dir / "costs_strategy_risk.csv")
        save_csv(C_eq, mode_dir / "costs_equal_weight.csv")

        if SAVE_WEIGHTS_CSV:
            if W_ret is not None:
                save_csv(W_ret, mode_dir / "weights_strategy_return.csv")
            if W_risk is not None:
                save_csv(W_risk, mode_dir / "weights_strategy_risk.csv")
            if W_eq is not None:
                save_csv(W_eq, mode_dir / "weights_equal_weight.csv")

    if SAVE_PLOTS or SHOW_PLOTS:
        plot_wealth_comparison(
            mode,
            {
                "strategy_return": V_ret,
                "strategy_risk": V_risk,
                "equal_weight": V_eq,
                BENCHMARK_TICKER: V_spy,
            },
            mode_dir / "wealth_curves_test.png",
        )
        plot_drawdown_comparison(
            mode,
            {
                "strategy_return": V_ret,
                "strategy_risk": V_risk,
                "equal_weight": V_eq,
                BENCHMARK_TICKER: V_spy,
            },
            mode_dir / "drawdown_curves_test.png",
        )
        plot_turnover_comparison(
            mode,
            {
                "turnover_strategy_return": TO_ret,
                "turnover_strategy_risk": TO_risk,
                "turnover_equal_weight": TO_eq,
            },
            mode_dir / "turnover_test.png",
        )


        # ------------------------------------------------------------------
        # Extra TEST-only diagnostic plots (do not affect strategy results)
        # ------------------------------------------------------------------
        # 1) Rolling Sharpe ratio (windowed risk-adjusted performance).
        plot_rolling_sharpe_comparison(
            mode,
            {
                "strategy_return": V_ret,
                "strategy_risk": V_risk,
                "equal_weight": V_eq,
                BENCHMARK_TICKER: V_spy,
            },
            mode_dir / "rolling_sharpe_test.png",
            window_days=int(CFG.lookback_days),
        )

        # 2) Top-10 weights over time (requires store_weights=True / SAVE_WEIGHTS_CSV=True).
        plot_top_weights_over_time(
            mode,
            W_ret,
            mode_dir / "top10_weights_strategy_return_test.png",
            top_k=10,
            title_suffix="strategy_return",
        )
        plot_top_weights_over_time(
            mode,
            W_risk,
            mode_dir / "top10_weights_strategy_risk_test.png",
            top_k=10,
            title_suffix="strategy_risk",
        )

        # 3) Calibration diagnostics (plotted at the end, but based on the coarse grid run).
        plot_calibration_best_lambda_by_gamma(
            mode,
            grid_c,
            mode_dir / "calibration_diag_best_lambda_by_gamma.png",
            chosen_points=[
                {"gamma": ret_params[0], "marker": "x"},
                {"gamma": risk_params[0], "marker": "o"},
            ],
        )
        plot_calibration_err_heatmap(
            mode,
            grid_c,
            mode_dir / "calibration_diag_vol_error_heatmap.png",
        )
    return ret_params, risk_params


# ======================================================================
# Entry point: compute date range, download/cached data, then run the requested rebalance modes.
# At the end we also aggregate per-mode summaries into a single CSV for convenience.
# 14) MAIN
# ======================================================================

# ----------------------------------------------------------------------
# Function: main
# ----------------------------------------------------------------------
def main() -> None:
    start_date = datetime.today() - relativedelta(years=CFG.years_total)
    end_date = datetime.today()

    global START_DATE, END_DATE
    START_DATE = start_date
    END_DATE = end_date

    # Decide which rebalance frequencies to run.
    # (This is a top-level control only; it does not change the strategy mechanics.)
    _m = RUN_MODES.strip().lower()
    if _m == "monthly":
        modes_to_run = ["monthly"]
    elif _m == "quarterly":
        modes_to_run = ["quarterly"]
    else:
        modes_to_run = ["monthly", "quarterly"]

    explain_block(
        "Run settings",
        [
            f"Date range: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d} (end date = today)",
            f"Total years downloaded: {CFG.years_total}",
            f"Test years (calendar): {CFG.test_years}",
            f"Lookback (trading days): {CFG.lookback_days}",
            f"Parent size: {CFG.parent_size}, investable size: {CFG.n_select}",
            f"Cap per asset: {CFG.cap:.2%}",
            f"rf_annual: {CFG.rf_annual:.2%}, kappa: {CFG.kappa:g}, apply_costs: {CFG.apply_costs}",
            f"Calibration profile: {RUN_PROFILE.upper()}",
            f"Calibration scope: per-mode (no reuse)",
            f"Cash cap enabled: {ENABLE_CASH_CAP} (cash <= {CASH_CAP:.2%})" if ENABLE_CASH_CAP else "Cash cap enabled: False",
            f"Modes: ['monthly', 'quarterly']",
            f"Benchmark: {BENCHMARK_TICKER}",
            f"Outputs: {OUT_DIR}",
        ],
    )

    t0 = time.perf_counter()
    print(f"requested parent_size: {CFG.parent_size}")
    parent = get_parent_tickers_yahoo_most_active(CFG.parent_size)
    print(f"len(parent) returned: {len(parent)}")

    adj, clo, vol = download_parent_data(parent, start_date=start_date, end_date=end_date)
    ret_all = simple_returns(adj)

    print(f"adj columns downloaded: {adj.shape[1]}")
    missing = sorted(set(parent) - set(adj.columns))
    print(f"missing from download: {len(missing)}")
    print(f"example missing: {missing[:10] if len(missing) else []}")

    bench_prices = download_benchmark_prices(BENCHMARK_TICKER, start_date=start_date, end_date=end_date)

    dl_s = time.perf_counter() - t0
    explain_block(
        "Data downloaded / prepared",
        [
            f"Parent tickers: {len(parent)}",
            f"AdjClose panel shape: {adj.shape}",
            f"Returns panel shape: {ret_all.shape}",
            f"Benchmark series length: {len(bench_prices)}",
            f"Total download/prepare time: {dl_s:.2f} seconds",
            "",
            "Note: parent universe is Yahoo 'most actives' at runtime (limitation disclosed in report).",
        ],
    )

    params_by_mode: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}

    for _mode in modes_to_run:
        run_mode(_mode, ret_all, clo, vol, bench_prices, params_by_mode)

    # Combine per-mode test summaries into one CSV for convenience.
    if SAVE_CSVS:
        combined = []
        for _mode in modes_to_run:
            p = OUT_DIR / _mode / "test_metrics_summary.csv"
            if p.exists():
                df = pd.read_csv(p, index_col=0)
                df.insert(0, "mode", _mode)
                combined.append(df)
        if len(combined) > 0:
            all_modes = pd.concat(combined, axis=0)
            save_csv(all_modes, OUT_DIR / "test_metrics_summary_all_modes.csv")

    explain_block(
        "Done",
        [
            f"All outputs saved under: {OUT_DIR}",
            "Per-mode subfolders contain CSV tables and PNG figures.",
        ],
    )


if __name__ == "__main__":
    main()