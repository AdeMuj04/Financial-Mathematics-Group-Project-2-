from dateutil.relativedelta import relativedelta
from datetime import datetime
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


###############################################
# big picture 
#
# We run a rolling backtest with:
#   - a parent list of tickers (top "most active" on Yahoo at runtime)
#   - a yearly universe refresh: at certain rebalance dates we re-pick the top 50
#     by a simple liquidity proxy (ADV$ = average(Close * Volume) over last 63 days)
#   - a mean/cov estimate using the last 63 days of returns (intersection-only dates)
#   - an optimisation at each rebalance (long-only, per-asset cap, cash allowed),
#     using projected gradient descent (PGD) and a ridge penalty
#   - holdings drift between rebalances (risky assets move with returns; cash grows at rf)
#   - turnover + proportional transaction costs at each rebalance
#
# Then we repeat the whole thing for different rebalance frequencies and compare to SPY.
#
# We *do* cache big Yahoo downloads because otherwise re-running is miserable.
# Plots are displayed (not saved). Summary is printed to console.
# ########################################



# basic run settings
# ----------------------------------------

# how far back we backtest
years = 5
# snap to the most recent Monday so the cache key is stable for the whole week
_today = datetime.today()
end_date = _today - relativedelta(days=_today.weekday())  # 0=Mon, ..., 6=Sun -> go back to Mon
end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
start_date = end_date - relativedelta(years=years)

# we use 3 months for:
#   (i) liquidity ranking window
#  (ii) mean/cov estimation window
lookback_months = 3
lookback_days = 63  # ~3 months of trading days

# universe choice
parent_size = 250   # pull top 250 "most active" from Yahoo screen
n_select = 50       # at universe refresh we keep top 50 by liquidity

# constraints
cap = 0.08          # max risky weight per asset (8%)

# risk-free and costs
rf_annual = 0.03
rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0

# transaction costs: proportional to turnover * wealth
# 10 bps per $ traded => kappa = 0.001
kappa = 0.001
apply_costs = True

# target vol used for choosing gamma (we pick gamma with realised vol closest to this)
target_vol_annual = 0.10

# rebalance modes we compare
modes = [   "quarterly"]

# grids (log-spaced)
n_grid_gamma = 11
n_grid_lambda = 11
gamma_values = np.logspace(-3, 3, n_grid_gamma)   # 1e-3 ... 1e3
lam_values = np.logspace(-6, 0, n_grid_lambda)    # 1e-6 ... 1

benchmark = "SPY"

# caching: we cache the Yahoo screen + large price pulls
cache_parent = True
cache_prices = True

# solver settings:
# - grid runs: looser tolerance + fewer iterations (faster)
# - final run: tighter tolerance + more iterations (more accurate)
tol_grid = 2e-5
max_iter_grid = 1200
tol_final = 1e-6
max_iter_final = 5000

# cache folder (plots are not saved)
cache_root = Path("outputs_gp2a_liquidity")
cache_dir = cache_root / "_cache"
cache_root.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

show_plots = True

#-----------

# small helpers


def _dstr(dt):
    # date -> YYYY-MM-DD for filenames
    return pd.Timestamp(dt).strftime("%Y-%m-%d")


def _hash_list(xs):
    # deterministic cache tag — uses md5 so the hash is stable across Python sessions
    s = "|".join(xs)
    return hashlib.md5(s.encode()).hexdigest()[:10]


def annual_vol(r):
    # annualised volatility from a series of daily returns
    r = pd.Series(r).dropna()
    if len(r) < 5:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(252.0))


def max_drawdown(V):
    # max drawdown of a wealth/equity curve
    V = pd.Series(V).dropna()
    if len(V) < 5:
        return np.nan
    peak = V.cummax()
    dd = V / peak - 1.0
    return float(dd.min())

#-------------

# parent universe from Yahoo "most actives"


def get_parent_universe(size=250):
    """
    We grab a parent list from Yahoo's "most active" screen via yfinance.
    This is a practical way to avoid maintaining a hand-written ticker list.

    Output: list of tickers (strings), length ~ size.
    """
    cache_path = cache_dir / f"parent_most_actives_{size}.json"

    # if cached parent exists, use it (saves a screen call)
    if cache_parent and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        tickers = d.get("tickers", [])
        if isinstance(tickers, list) and len(tickers) > 0:
            return tickers[:size]

    # yfinance "screen" exists only in newer versions
    if not hasattr(yf, "screen"):
        raise RuntimeError("yfinance doesn't have yf.screen() here. update yfinance and retry.")

    out = None
    last_err = None

    # yfinance versions differ in kwargs ("count" vs "size" etc), so we try a few
    for kwargs in [{"count": size}, {"size": size}, {}]:
        for attempt in range(4):
            try:
                out = yf.screen("most_actives", **kwargs)
                break
            except Exception as e:
                msg = str(e).lower()
                if ("too many requests" in msg or "429" in msg or "rate" in msg) and attempt < 3:
                    wait = 10 * (attempt + 1)
                    print(f"  yf.screen rate-limited, retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    last_err = e
                    out = None
                    break
        if out is not None:
            break

    if out is None:
        raise RuntimeError(f"could not call yf.screen('most_actives'): {last_err}")

    tickers = []

    # sometimes we get dicts, sometimes a DataFrame
    if isinstance(out, dict):
        quotes = out.get("quotes", [])
        if isinstance(quotes, list):
            for q in quotes:
                if isinstance(q, dict) and isinstance(q.get("symbol"), str):
                    tickers.append(q["symbol"])

    if isinstance(out, pd.DataFrame):
        for col in ["symbol", "Symbol", "ticker", "Ticker"]:
            if col in out.columns:
                tickers = [str(x) for x in out[col].dropna().tolist()]
                break

    # dedupe while keeping order
    seen = set()
    tickers2 = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            tickers2.append(t)

    tickers2 = tickers2[:size]
    if len(tickers2) == 0:
        raise RuntimeError("got no tickers from yf.screen()")

    # cache the parent list so we can re-run without hitting the screen each time
    if cache_parent:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {"source": "yfinance.screen most_actives", "size": size, "tickers": tickers2},
                f,
                indent=2
            )

    return tickers2



# download price/volume panels (cached)


def _download_batch(tickers, start, end, retries=5, base_delay=5):
    """
    Download a batch of tickers from Yahoo with retry + exponential back-off
    to handle Yahoo Finance rate-limiting (HTTP 429).
    """
    for attempt in range(retries):
        try:
            return yf.download(
                tickers,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                actions=False,
                group_by="column",
                threads=False
            )
        except Exception as e:
            msg = str(e).lower()
            if ("too many requests" in msg or "429" in msg or "rate" in msg) and attempt < retries - 1:
                wait = base_delay * (2 ** attempt)
                print(f"  rate-limited, retrying in {wait}s (attempt {attempt+1}/{retries})...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"download failed after {retries} retries")


def download_parent_data(parent_tickers, start, end, batch_size=80):
    """
    We download three panels for the parent universe:
      - adj close (used for returns)
      - close (used for liquidity calculation)
      - volume (used for liquidity calculation)

    We cache these on disk because this is the slowest part of the pipeline.
    """
    h = _hash_list(parent_tickers)
    tag = f"{len(parent_tickers)}_{h}_{_dstr(start)}_{_dstr(end)}"

    p_adj = cache_dir / f"adj_{tag}.csv"
    p_clo = cache_dir / f"close_{tag}.csv"
    p_vol = cache_dir / f"vol_{tag}.csv"

    # load cached files if they exist
    if cache_prices and p_adj.exists() and p_clo.exists() and p_vol.exists():
        adj = pd.read_csv(p_adj, index_col=0, parse_dates=True)
        clo = pd.read_csv(p_clo, index_col=0, parse_dates=True)
        vol = pd.read_csv(p_vol, index_col=0, parse_dates=True)
        return adj, clo, vol

    # otherwise download in chunks
    adj_list, clo_list, vol_list = [], [], []

    for i in range(0, len(parent_tickers), batch_size):
        if i > 0:
            time.sleep(3)  # avoid hitting Yahoo rate limit between batches
        batch = parent_tickers[i:i + batch_size]
        raw = _download_batch(batch, start, end)

        if not isinstance(raw.columns, pd.MultiIndex):
            raise ValueError("expected MultiIndex columns from yfinance for multi-ticker download")

        lvl0 = raw.columns.get_level_values(0)

        # we prefer Adj Close when it exists; if not, we fall back to Close
        if "Adj Close" in set(lvl0):
            adj_b = raw["Adj Close"].copy()
        else:
            adj_b = raw["Close"].copy()

        clo_b = raw["Close"].copy()
        vol_b = raw["Volume"].copy()

        # make sure each is a DataFrame
        if isinstance(adj_b, pd.Series):
            adj_b = adj_b.to_frame()
        if isinstance(clo_b, pd.Series):
            clo_b = clo_b.to_frame()
        if isinstance(vol_b, pd.Series):
            vol_b = vol_b.to_frame()

        adj_list.append(adj_b)
        clo_list.append(clo_b)
        vol_list.append(vol_b)

    adj = pd.concat(adj_list, axis=1).sort_index()
    clo = pd.concat(clo_list, axis=1).sort_index()
    vol = pd.concat(vol_list, axis=1).sort_index()

    # reorder columns to match the parent_tickers ordering
    adj = adj.loc[:, [t for t in parent_tickers if t in adj.columns]]
    clo = clo.loc[:, [t for t in parent_tickers if t in clo.columns]]
    vol = vol.loc[:, [t for t in parent_tickers if t in vol.columns]]

    if cache_prices:
        adj.to_csv(p_adj)
        clo.to_csv(p_clo)
        vol.to_csv(p_vol)

    return adj, clo, vol


def simple_returns(prices):
    """
    Simple returns: R_t = P_t/P_{t-1} - 1.
    We use fill_method=None so pandas doesn't silently forward-fill missing values.
    """
    return prices.pct_change(fill_method=None).dropna(how="all")



# rebalance schedules


def rebalance_idx_for_mode(dates, mode, start_trade_date):
    """
    Given a full trading-day index, we build a list of integer indices (tk) where we rebalance.

    - monthly: last trading day of each month
    - quarterly: last trading day of each quarter
    - biweekly10: every 10 trading days
    """
    dates = pd.DatetimeIndex(dates)
    start_trade_date = pd.Timestamp(start_trade_date)

    # we only consider rebalances on/after start_trade_date
    ok = np.where(dates >= start_trade_date)[0]
    if len(ok) == 0:
        return []
    first = int(ok[0])

    if mode == "biweekly10":
        return list(range(first, len(dates), 10))

    s = dates.to_series()

    if mode == "monthly":
        last = s.groupby([dates.year, dates.month]).max()
        reb_dates = pd.DatetimeIndex(last.values)
    elif mode == "quarterly":
        q = ((dates.month - 1) // 3) + 1
        last = s.groupby([dates.year, q]).max()
        reb_dates = pd.DatetimeIndex(last.values)
    else:
        raise ValueError("mode must be monthly / biweekly10 / quarterly")

    idx = []
    for d in reb_dates:
        if d < start_trade_date:
            continue
        if d in dates:
            idx.append(int(dates.get_loc(d)))

    return sorted(set(idx))



# yearly universe updates + liquidity filter


def anniversary_list(start_dt, years_):
    """
    We create a list of anniversary dates:
      start_dt + 1 year, +2 years, ..., +years_ years.
    """
    base = pd.Timestamp(start_dt)
    return [base + pd.DateOffset(years=k) for k in range(1, years_ + 1)]


def next_rebalance_date(reb_idx, dates, dt):
    """
    Map an arbitrary date dt to the *next* available rebalance date in our schedule.
    """
    for i in reb_idx:
        if dates[i] >= dt:
            return dates[i]
    return None


def build_universe_update_flags(reb_idx, dates, start_trade_date):
    """
    We decide which rebalance dates trigger a universe refresh.

    Rule:
      - first rebalance date at/after start_trade_date
      - then each yearly anniversary of start_date, mapped forward to the next rebalance date
    """
    flags = {}

    first = next_rebalance_date(reb_idx, dates, pd.Timestamp(start_trade_date))
    if first is not None:
        flags[first] = True

    for a in anniversary_list(start_date, years):
        d = next_rebalance_date(reb_idx, dates, a)
        if d is not None:
            flags[d] = True

    return flags


def pick_top_liquidity(close_df, vol_df, adj_df, parent_tickers, dates, tk, lb_days, n_pick):
    """
    Liquidity score: ADV$ = mean(Close * Volume) over last lb_days, ending at tk-1.
    We only rank tickers that have complete data over that lookback.
    """
    if tk - lb_days < 1:
        raise ValueError("not enough history for liquidity lookback")

    w_dates = dates[tk - lb_days:tk]

    # only tickers that exist in all panels
    cols = [t for t in parent_tickers if t in close_df.columns and t in vol_df.columns and t in adj_df.columns]
    if len(cols) < n_pick:
        raise ValueError("parent tickers with data columns is too small")

    clo_w = close_df.loc[w_dates, cols]
    vol_w = vol_df.loc[w_dates, cols]
    adj_w = adj_df.loc[w_dates, cols]

    # require no missing values across the whole liquidity window
    ok = (~clo_w.isna()).all(axis=0) & (~vol_w.isna()).all(axis=0) & (~adj_w.isna()).all(axis=0)
    good = ok[ok].index.tolist()

    if len(good) < n_pick:
        raise ValueError(f"not enough tickers with full lookback data (got {len(good)} need {n_pick})")

    adv = (clo_w[good] * vol_w[good]).mean(axis=0).sort_values(ascending=False)
    return adv.head(n_pick).index.tolist()


def build_universe_by_rebalance_date(ret_all, close_all, vol_all, adj_all, parent_tickers, reb_idx, start_trade_date):
    """
    For every rebalance date, we attach a universe (list of tickers).
    If it’s a refresh date, we recompute the top 50 by liquidity;
    otherwise we keep the previous universe.
    """
    dates = ret_all.index
    flags = build_universe_update_flags(reb_idx, dates, start_trade_date)

    universe_by_date = {}
    current = None
    updates_log = []

    for tk in reb_idx:
        d = dates[tk]
        if current is None or flags.get(d, False):
            current = pick_top_liquidity(
                close_df=close_all,
                vol_df=vol_all,
                adj_df=adj_all,
                parent_tickers=parent_tickers,
                dates=dates,
                tk=tk,
                lb_days=lookback_days,
                n_pick=n_select
            )
            updates_log.append(str(d.date()))
        universe_by_date[d] = current

    return universe_by_date, updates_log



# projection + PGD optimiser


def projection(v, total=1.0, caps=None, tol=1e-12, max_iter=2000):
    """
    Euclidean projection onto the capped simplex:

        sum_i w_i = total
        0 <= w_i <= caps_i

    We do a scalar shift tau + clipping, found via bisection.
    This stays close to the style we already used earlier.
    """
    v = np.asarray(v, dtype=float).ravel()
    caps = np.asarray(caps, dtype=float).ravel()
    if v.shape != caps.shape:
        raise ValueError("caps shape mismatch")

    if total < 0 or total > float(caps.sum()) + 1e-12:
        raise ValueError("infeasible total for caps")

    lo = float((v - caps).min())
    hi = float(v.max())

    for _ in range(max_iter):
        tau = 0.5 * (lo + hi)
        w = np.clip(v - tau, 0.0, caps)
        s = float(w.sum())

        if abs(s - total) < tol:
            return w
        if s > total:
            lo = tau
        else:
            hi = tau

    return np.clip(v - 0.5 * (lo + hi), 0.0, caps)


def ridge_grad(w, cov_matrix, mu, lam=1.0, gamma=1.0, rf=0.0):
    """
    We work with annualised moments (mu, cov). We include a cash weight.

    Objective we minimise (in annual units):

        f(w) = -(mu^T w_r + rf*w_cash) + (gamma/2) w_r^T Sigma w_r + lam ||w_r||^2

    - w_r is the risky block (length p)
    - w_cash is a single scalar
    - Sigma is risky-only covariance (cash has zero variance/covariance)
    - ridge penalty only on risky weights
    """
    p = len(mu)
    w_r = w[:p]
    g = np.zeros_like(w)
    g[:p] = gamma * (cov_matrix @ w_r) - mu + 2.0 * lam * w_r
    g[p] = -rf
    return g


def auto_step(gamma, lam, eigmax):
    """
    PGD is sensitive to step size. We pick a conservative step based on a Lipschitz bound:

        L ~ gamma * lambda_max(Sigma) + 2*lam

    and use step ~ 1/L (capped).
    """
    L = gamma * max(float(eigmax), 1e-12) + 2.0 * lam
    return float(min(1.0 / max(L, 1e-12), 100.0))


def ridge_portfolio_pgd(cov_matrix, mu, lam, gamma=1.0, cap=0.08, rf=0.0,
                        tol=1e-6, max_iter=5000, w_init=None, eigmax=1.0):
    """
    We solve for weights [w_risky (p entries), w_cash] with constraints:

      - sum weights = 1
      - 0 <= w_i <= cap for each risky weight
      - 0 <= w_cash <= 1

    Algorithm:
      1) take a gradient step (unconstrained)
      2) project back onto constraints
      3) repeat until convergence or max_iter
    """
    p = len(mu)
    caps = np.concatenate([np.full(p, cap), np.array([1.0])])

    # initialise weights: either warm-start from w_init or a simple feasible guess
    if w_init is None or len(w_init) != p + 1:
        w_r = np.full(p, 1.0 / p)
        w_r = np.minimum(w_r, cap)
        w_cash = 1.0 - float(w_r.sum())
        if w_cash < 0:
            w_r = w_r / w_r.sum()
            w_cash = 0.0
        w = np.concatenate([w_r, np.array([w_cash])])
        w = projection(w, total=1.0, caps=caps)
    else:
        w = projection(np.asarray(w_init, dtype=float), total=1.0, caps=caps)

    step_size = auto_step(gamma, lam, eigmax)

    for it in range(1, max_iter + 1):
        grad = ridge_grad(w, cov_matrix, mu, lam=lam, gamma=gamma, rf=rf)
        w_new = projection(w - step_size * grad, total=1.0, caps=caps)

        if np.linalg.norm(w_new - w) < tol:
            return w_new, it, step_size

        w = w_new

    return w, max_iter, step_size



# moments (intersection only) and annualisation


def estimate_moments_annualised(ret_all, tickers, dates, tk, lb_days):
    """
    Estimation window is last lb_days ending at tk-1.

    We enforce intersection-only dates:
      - drop any row where any ticker return is missing
      - if that shortens the sample too much, we fail early

    We annualise:
      mu_a = 252 * mu_daily
      cov_a = 252 * cov_daily
    """
    if tk - lb_days < 1:
        raise ValueError("not enough history for estimation")

    w_dates = dates[tk - lb_days:tk]
    Rw = ret_all.loc[w_dates, tickers].dropna(axis=0, how="any")

    if len(Rw) < lb_days:
        raise ValueError(f"intersection window too short: got {len(Rw)} need {lb_days}")

    mu_d = Rw.mean(axis=0).to_numpy()
    cov_d = Rw.cov(ddof=1).to_numpy()
    cov_d = 0.5 * (cov_d + cov_d.T)

    mu_a = 252.0 * mu_d
    cov_a = 252.0 * cov_d
    return mu_a, cov_a



# precompute segments per mode (runtime fix)


def prepare_mode_segments(mode, ret_all, close_all, vol_all, adj_all, parent_tickers):
    """
    This is where we do all the expensive / repeated logic once per mode.

    For each rebalance date d_k we create a "segment" that contains:
      - the universe (tickers)
      - the estimated mu and cov at that date
      - the holding-period return matrix from d_k to d_{k+1} (intersection-only)

    Then grid searches just reuse these segments (huge speed-up).
    """
    dates = ret_all.index
    start_trade_date = pd.Timestamp(start_date) + pd.DateOffset(months=lookback_months)

    # build rebalance indices for this mode
    reb_idx = rebalance_idx_for_mode(dates, mode, start_trade_date)

    # we also need enough data for liquidity + estimation lookback
    reb_idx = [tk for tk in reb_idx if tk >= lookback_days]
    if len(reb_idx) < 5:
        raise ValueError("too few rebalances after lookback filter")

    # build universe mapping (ticker list for each rebalance date)
    universe_by_date, update_dates = build_universe_by_rebalance_date(
        ret_all, close_all, vol_all, adj_all, parent_tickers, reb_idx, start_trade_date
    )

    segs = []
    for k, tk in enumerate(reb_idx):
        d = dates[tk]
        tickers = universe_by_date[d]

        # compute annualised moments once
        mu_a, cov_a = estimate_moments_annualised(ret_all, tickers, dates, tk, lookback_days)
        eigmax = float(np.linalg.eigvalsh(cov_a).max()) if cov_a.size else 0.0

        # holding period is from tk to next rebalance (exclusive of next)
        t_next = reb_idx[k + 1] if (k + 1) < len(reb_idx) else len(dates)
        hold_df = ret_all.iloc[tk:t_next][tickers].dropna(axis=0, how="any")

        segs.append({
            "date": d,
            "tickers": tickers,
            "mu": mu_a,
            "cov": cov_a,
            "eigmax": eigmax,
            "hold_dates": hold_df.index,
            "hold_R": hold_df.to_numpy(dtype=float),
        })

    return segs, update_dates



# simulation (vectorised drift between rebalances)


def simulate_from_segments(segs, gamma, lam, tol, max_iter, warm_init=None, verbose_every=0, tag=""):
    """
    We simulate wealth over time:

    At each rebalance date:
      1) solve for target weights (risky + cash)
      2) compute turnover vs current risky weights
      3) deduct transaction costs (kappa * turnover * wealth)
      4) set holdings to match target weights

    Between rebalances:
      - risky holdings drift with realised asset returns
      - cash drifts with rf_daily

    Implementation detail:
      - we drift using cumulative products, so we avoid looping day-by-day in Python.
    """
    v0 = 10_000_000.0

    # holdings are: risky allocations A_prev (vector) + cash (scalar)
    cash = float(v0)
    prev_tickers = []
    A_prev = np.zeros(0, dtype=float)

    # time series outputs
    V_dates, V_vals = [], []
    to_dates, to_vals = [], []
    c_dates, c_vals = [], []

    # warm start stores last solution per rebalance index
    warm_out = [None] * len(segs)

    for j, seg in enumerate(segs):
        d = seg["date"]
        tickers = seg["tickers"]
        p = len(tickers)

        # wealth just before rebalancing
        V_pre = cash + float(A_prev.sum())
        if V_pre <= 0:
            raise ValueError("portfolio value <= 0")

        # map current risky weights by ticker so we can handle universe changes cleanly
        pre_map = {}
        if len(prev_tickers) > 0 and len(A_prev) == len(prev_tickers):
            w_pre_vec = A_prev / V_pre
            for i, t in enumerate(prev_tickers):
                pre_map[t] = float(w_pre_vec[i])

        # warm-start if provided
        w0 = None
        if warm_init is not None and warm_init[j] is not None:
            w0 = warm_init[j]

        # solve the optimisation for this segment
        w, iters, step_used = ridge_portfolio_pgd(
            cov_matrix=seg["cov"],
            mu=seg["mu"],
            lam=float(lam),
            gamma=float(gamma),
            cap=cap,
            rf=rf_annual,     # objective uses annual rf to match annual mu/cov
            tol=tol,
            max_iter=max_iter,
            w_init=w0,
            eigmax=seg["eigmax"],
        )
        warm_out[j] = w.copy()

        w_r = w[:p]
        w_cash = float(w[p])

        # turnover on the union of old + new risky tickers:
        # - for tickers we keep: abs(new - old)
        # - for new tickers: abs(new - 0)
        # - for removed tickers: abs(0 - old)
        to = 0.0
        for i, t in enumerate(tickers):
            to += abs(float(w_r[i]) - pre_map.get(t, 0.0))
            pre_map[t] = None
        for t, val in pre_map.items():
            if val is not None:
                to += abs(0.0 - float(val))

        # proportional transaction costs charged immediately at rebalance
        cost = 0.0
        V_post = V_pre
        if apply_costs and kappa > 0:
            if kappa * to >= 1.0:
                raise ValueError("kappa*turnover >= 1 (wipeout)")
            cost = kappa * V_pre * to
            V_post = V_pre - cost

        # set allocations right after rebalance (post-cost wealth)
        A0 = V_post * w_r
        cash0 = V_post * w_cash

        # record turnover/cost at this rebalance date
        to_dates.append(d)
        to_vals.append(float(to))
        c_dates.append(d)
        c_vals.append(float(cost))

        # progress printing
        if verbose_every and ((j == 0) or (j == len(segs) - 1) or ((j + 1) % int(verbose_every) == 0)):
            print(f"{tag}: reb {j+1}/{len(segs)} {d.date()} TO={to:.4f} iters={iters} step={step_used:.3e}", flush=True)

        # drift through holding period (vectorised)
        R = seg["hold_R"]
        dd = seg["hold_dates"]

        if R.size > 0:
            # risky: multiply holdings by cumulative product of (1+R)
            fac = np.cumprod(1.0 + R, axis=0)
            A_path = A0[None, :] * fac

            # cash: deterministic compounding at rf_daily
            m = R.shape[0]
            rf_fac = (1.0 + rf_daily) ** np.arange(1, m + 1)
            cash_path = cash0 * rf_fac

            # wealth path
            V_path = A_path.sum(axis=1) + cash_path

            V_dates.extend(list(dd))
            V_vals.extend(list(V_path))

            # end-of-period holdings become the new starting point
            A_end = A_path[-1, :]
            cash_end = float(cash_path[-1])
        else:
            # no holding days in this segment (rare, but possible)
            A_end = A0
            cash_end = cash0

        prev_tickers = tickers
        A_prev = np.asarray(A_end, dtype=float)
        cash = float(cash_end)

    V = pd.Series(V_vals, index=pd.DatetimeIndex(V_dates), name="V").sort_index()
    TO = pd.DataFrame({"turnover": to_vals}, index=pd.DatetimeIndex(to_dates)).sort_index()
    C = pd.DataFrame({"cost": c_vals}, index=pd.DatetimeIndex(c_dates)).sort_index()

    return V, TO, C, warm_out



# benchmark + perf summary

def download_benchmark_prices(ticker, start, end):
    """
    Download benchmark price series (we use Adj Close if available).
    Cached, same reason as everything else.
    """
    cache_path = cache_dir / f"bench_{ticker}_{_dstr(start)}_{_dstr(end)}.csv"
    if cache_prices and cache_path.exists():
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = ticker
        return s

    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        actions=False,
        group_by="column"
    )

    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        if "Adj Close" in set(lvl0):
            p = data["Adj Close"]
        else:
            p = data["Close"]
        if isinstance(p, pd.DataFrame):
            p = p.iloc[:, 0]
    else:
        p = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

    p = p.dropna()
    p.name = ticker

    if cache_prices:
        p.to_frame().to_csv(cache_path)

    return p


def perf_table(V, bench_prices):
    """
    Basic performance metrics on daily returns:
      - CAGR
      - annual vol
      - Sharpe using rf_daily
      - max drawdown
      - beta vs benchmark
    """
    V = V.dropna()
    if len(V) < 10:
        return {"note": "not enough portfolio data"}

    Rp = V.pct_change(fill_method=None).dropna()
    Rb = bench_prices.pct_change(fill_method=None).dropna()

    common = Rp.index.intersection(Rb.index)
    Rp = Rp.loc[common]
    Rb = Rb.loc[common]

    if len(common) < 50:
        return {"note": "not enough overlap with benchmark"}

    Rex_p = Rp - rf_daily
    Rex_b = Rb - rf_daily

    sharpe_p = float(Rex_p.mean() / Rex_p.std(ddof=1) * np.sqrt(252.0))
    sharpe_b = float(Rex_b.mean() / Rex_b.std(ddof=1) * np.sqrt(252.0))

    cagr_p = float((1.0 + Rp).prod() ** (252.0 / len(Rp)) - 1.0)
    cagr_b = float((1.0 + Rb).prod() ** (252.0 / len(Rb)) - 1.0)

    beta = float(np.cov(Rp, Rb, ddof=1)[0, 1] / np.var(Rb, ddof=1))

    return {
        "CAGR_port": cagr_p,
        "AnnVol_port": annual_vol(Rp),
        "Sharpe_port": sharpe_p,
        "MaxDD_port": max_drawdown(V.loc[common]),
        "CAGR_SPY": cagr_b,
        "AnnVol_SPY": annual_vol(Rb),
        "Sharpe_SPY": sharpe_b,
        "beta_vs_SPY": beta,
        "n_days": int(len(common)),
    }



# heatmap: what we tested during gamma +lambda selection
def plot_tested_heatmap(mode, gamma_vals, lam_vals, gdf, ldf, gamma_star, lambda_star):
    """
    We are not doing a full 2D grid search here.
    We test:
      - all gamma values with lambda fixed at lam_vals[0]
      - all lambda values with gamma fixed at gamma_star

    So the “tested points” form a cross in (gamma, lambda) space.
    We visualise final wealth at those tested points.
    """
    G = len(gamma_vals)
    L = len(lam_vals)

    M = np.full((G, L), np.nan)

    # gamma sweep: fill column j=0
    for i in range(G):
        M[i, 0] = float(gdf.iloc[i]["finalV"])

    # lambda sweep: fill row at gamma_star
    gamma_idx = int(np.where(np.isclose(gamma_vals, gamma_star))[0][0])
    for j in range(L):
        M[gamma_idx, j] = float(ldf.iloc[j]["finalV"])

    MM = np.ma.masked_invalid(M)

    plt.figure(figsize=(10, 4.5))
    im = plt.imshow(MM, origin="lower", aspect="auto")
    plt.colorbar(im, label="final wealth (USD)")

    # keep tick labels readble
    x_ticks = np.linspace(0, L - 1, min(L, 6), dtype=int)
    y_ticks = np.linspace(0, G - 1, min(G, 6), dtype=int)

    plt.xticks(x_ticks, [f"{lam_vals[j]:.1e}" for j in x_ticks], rotation=45, ha="right")
    plt.yticks(y_ticks, [f"{gamma_vals[i]:.1e}" for i in y_ticks])

    plt.xlabel("lambda")
    plt.ylabel("gamma")
    plt.title(f"{mode}: tested (gamma, lambda) points")

    lam_idx = int(np.where(np.isclose(lam_vals, lambda_star))[0][0])
    plt.scatter([lam_idx], [gamma_idx], s=60, marker="x")
    plt.tight_layout()
    plt.show()



# run one mode


def run_mode(mode, ret_all, clo, vol, adj, parent, bench_prices):
    """
    For one rebalance mode we do:

      (A) Precompute segments (universes + moments + holding returns)
      (B) Choose gamma:
            run gamma grid with lambda fixed at smallest value
            pick gamma with realised vol closest to target
      (C) Choose lambda:
            run lambda grid at gamma_star
            pick lambda that gives max final wealth
      (D) Final run with tighter solver settings
      (E) Print + plot
    """
    t_prep = time.perf_counter()
    segs, update_dates = prepare_mode_segments(mode, ret_all, clo, vol, adj, parent)
    prep_time = time.perf_counter() - t_prep

    print(f"{mode}: rebalances={len(segs)} updates={len(update_dates)} prep={prep_time:.1f}s", flush=True)
    if len(update_dates) > 0:
        print(f"{mode}: update dates: {update_dates}", flush=True)

    # ----------------------------
    #ganma selection
    ###
    lam_for_gamma = float(lam_values[0])
    print(f"{mode}: gamma grid (n={len(gamma_values)}) with lambda={lam_for_gamma:.1e}", flush=True)

    warm = None
    rows_g = []
    t0 = time.perf_counter()

    for i, g in enumerate(gamma_values):
        V, TO, C, warm = simulate_from_segments(
            segs, gamma=float(g), lam=lam_for_gamma,
            tol=tol_grid, max_iter=max_iter_grid,
            warm_init=warm, verbose_every=0, tag=mode
        )
        Rp = V.pct_change(fill_method=None).dropna()
        vol_g = annual_vol(Rp)
        finalV = float(V.iloc[-1]) if len(V) else np.nan
        rows_g.append({"gamma": float(g), "vol": float(vol_g), "finalV": float(finalV)})

        # only print some points so the console doesn’t become noise
        if (i == 0) or (i == len(gamma_values) - 1) or ((i + 1) % 4 == 0):
            print(f"  gamma {i+1}/{len(gamma_values)} {g:.2e} vol={vol_g:.2%} V={finalV:,.0f}", flush=True)

    gdf = pd.DataFrame(rows_g)
    gdf["err"] = (gdf["vol"] - target_vol_annual).abs()
    gamma_star = float(gdf.loc[gdf["err"].idxmin(), "gamma"])
    vol_star = float(gdf.loc[gdf["err"].idxmin(), "vol"])

    print(
        f"{mode}: gamma*={gamma_star:.3g} (vol={vol_star:.2%}, target={target_vol_annual:.2%}) "
        f"t={time.perf_counter()-t0:.1f}s",
        flush=True
    )

    # --------------------------
    # lambda selection
    #
    print(f"{mode}: lambda grid (n={len(lam_values)}) at gamma={gamma_star:.3g}", flush=True)

    warm = None
    rows_l = []
    best_lam = None
    best_V = -np.inf
    t1 = time.perf_counter()

    for i, lam in enumerate(lam_values):
        V, TO, C, warm = simulate_from_segments(
            segs, gamma=float(gamma_star), lam=float(lam),
            tol=tol_grid, max_iter=max_iter_grid,
            warm_init=warm, verbose_every=0, tag=mode
        )

        finalV = float(V.iloc[-1]) if len(V) else np.nan
        Rp = V.pct_change(fill_method=None).dropna()
        vol_l = annual_vol(Rp)

        rows_l.append({"lambda": float(lam), "finalV": float(finalV), "vol": float(vol_l)})

        if finalV > best_V:
            best_V = finalV
            best_lam = float(lam)

        if (i == 0) or (i == len(lam_values) - 1) or ((i + 1) % 4 == 0):
            print(f"  lam {i+1}/{len(lam_values)} {lam:.2e} V={finalV:,.0f} vol={vol_l:.2%}", flush=True)

    ldf = pd.DataFrame(rows_l)
    lambda_star = float(best_lam)

    print(f"{mode}: lambda*={lambda_star:.3g} (best V={best_V:,.0f}) t={time.perf_counter()-t1:.1f}s", flush=True)

    # --------------------------
    # final run (tighter solver)
    
    print(f"{mode}: final run gamma={gamma_star:.3g}, lambda={lambda_star:.3g}", flush=True)

    V, TO, C, _ = simulate_from_segments(
        segs, gamma=float(gamma_star), lam=float(lambda_star),
        tol=tol_final, max_iter=max_iter_final,
        warm_init=None, verbose_every=10, tag=mode
    )

    perf = perf_table(V, bench_prices)
    print(f"{mode}: summary {perf}", flush=True)

    # --------------------------------------
    # plots (display only)
    
    if show_plots:
        plot_tested_heatmap(mode, gamma_values, lam_values, gdf, ldf, gamma_star, lambda_star)

        plt.figure(figsize=(10, 4))
        V.plot()
        plt.title(f"{mode}: portfolio value")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 3))
        TO["turnover"].plot()
        plt.title(f"{mode}: turnover")
        plt.tight_layout()
        plt.show()

    return {
        "mode": mode,
        "rebalances": int(len(segs)),
        "update_dates": update_dates,
        "gamma_star": gamma_star,
        "lambda_star": lambda_star,
        "perf": perf,
        "gamma_grid_df": gdf,
        "lambda_grid_df": ldf,
        "V": V,
        "TO": TO,
        "C": C,
    }


# main
# ------------------------------------

def main():
    """
    Main entrypoint:
      1) build parent list
      2) download/cached panels for parent
      3) compute returns
      4) download benchmark prices
      5) loop over rebalance modes
      6) print a small comparison table
    """
    t_all = time.perf_counter()

    print("getting parent tickers...", flush=True)
    parent = get_parent_universe(parent_size)
    print("parent size:", len(parent), flush=True)

    print("downloading prices/volume (cached)...", flush=True)
    adj, clo, vol = download_parent_data(parent, start_date, end_date, batch_size=80)

    # returns from adj close (simple returns)
    ret_all = simple_returns(adj)
    print("returns:", ret_all.shape, flush=True)

    print("benchmark:", benchmark, flush=True)
    bench_prices = download_benchmark_prices(benchmark, start_date, end_date)

    print("grid:", flush=True)
    print("  gamma:", f"{gamma_values[0]:.1e} .. {gamma_values[-1]:.1e}", "n=", len(gamma_values), flush=True)
    print("  lam:  ", f"{lam_values[0]:.1e} .. {lam_values[-1]:.1e}", "n=", len(lam_values), flush=True)
    print("  lookback_days:", lookback_days, "cap:", cap, "kappa:", kappa, "rf_annual:", rf_annual, flush=True)

    results = {}
    perf_rows = []

    for mode in modes:
        print("\nmode:", mode, flush=True)
        res = run_mode(mode, ret_all, clo, vol, adj, parent, bench_prices)
        results[mode] = res

        p = res["perf"]
        perf_rows.append({
            "mode": mode,
            "CAGR_port": p.get("CAGR_port", np.nan),
            "AnnVol_port": p.get("AnnVol_port", np.nan),
            "Sharpe_port": p.get("Sharpe_port", np.nan),
            "MaxDD_port": p.get("MaxDD_port", np.nan),
            "beta_vs_SPY": p.get("beta_vs_SPY", np.nan),
        })

    perf_df = pd.DataFrame(perf_rows)
    print("\nperf table:", flush=True)
    print(perf_df.to_string(index=False), flush=True)

    print("\ndone in", round(time.perf_counter() - t_all, 1), "seconds", flush=True)
    print("cache folder:", str(cache_dir.resolve()), flush=True)

    return results, perf_df


if __name__ == "__main__":
    RESULTS, PERF_DF = main()
