# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 22:56:08 2026

@author: slopingcorn
"""

from mpl_toolkits.mplot3d import Axes3D
from dateutil.relativedelta import relativedelta
from datetime import datetime
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'GOOG', 'TSLA',
    'LLY', 'AVGO', 'JPM', 'UNH', 'XOM', 'V', 'MA', 'HD', 'PG', 'COST', 'MRK',
    'ABBV', 'CVX', 'PEP', 'ADBE', 'KO', 'CRM', 'NFLX', 'AMD', 'TMO', 'WMT',
    'DIS', 'MCD', 'INTC', 'ACN', 'QCOM', 'CSCO', 'TXN', 'NEE', 'LIN', 'PM',
    'HON', 'AMGN', 'IBM', 'UPS', 'MS', 'GS', 'SBUX', 'CAT', 'BA'
]

years = 5
end_date = datetime.today() - relativedelta(years = 0) 
start_date = end_date - relativedelta(years=years)





adj_close = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    adj_close[ticker] = data['Close']



# add synthetic CASH
rfr_cash = 0.1
rf_daily_log = rfr_cash / 252.0


cash_prices = pd.Series(np.exp(rf_daily_log * np.arange(len(adj_close))), index=adj_close.index, name="CASH")
cash_prices /= cash_prices.iloc[0]
adj_close["CASH"] = cash_prices

tickers = list(adj_close.columns)

#%%

log_returns = np.log(adj_close / adj_close.shift(1)).dropna()
mu = log_returns.mean().values * 252
cov_matrix = log_returns.cov().values * 252

def expected_return(w, log_returns):
    return np.sum(log_returns.mean().values * w) * 252

def standard_deviation(w, cov_matrix):
    return np.sqrt(w.T @ cov_matrix @ w)

def sharpe_ratio(w, log_returns, cov_matrix, rfr):
    return (expected_return(w, log_returns) - rfr) / standard_deviation(w, cov_matrix)

mu_s = pd.Series(mu, index=log_returns.columns)
print("mu(CASH) =", mu_s["CASH"])
print("var(CASH) =", np.diag(cov_matrix)[log_returns.columns.get_loc("CASH")])

def projection(v, total=1.0, cap=0.4, tol=1e-12, max_iter=2000):
    """
    A projection which takes an input of weights (v), shifts it by a certain scalar tau to make it sum to "total" and applies certain constraints to project in desired range 

    Parameters
    ----------
    v : array
        vector of input weights (the weights we are projecting)
    total : TYPE, optional
        what the sum of the weights should equal. The default is 1.0.
    cap : float, optional
        maximum weight of any one single stock. The default is 0.4.
    tol : float, optional
        maximum difference between 1 and the projected weights total (want this to be as close to 0 as possible). The default is 1e-12.
    max_iter : int, optional
        Max number of bisection iterations. The default is 2000.

    Raises
    ------
    ValueError
        checks if the total is invalid

    Returns
    -------
    W : array
        Projected weights to be used in gradient descent
    """
    
    #create vector n with length equal to the number of rows (same as the number of assets)
    n = v.shape[0]
    
    #check if the total weights input is less than 0 or higher than the number of stocks * cap
    if total < 0 or total > n * cap:
        raise ValueError(f"Infeasible: total={total} not in [0, {n*cap}] for cap={cap}")

    #generate bounds for bisection (min and max of tau)
    lo = v.min() - cap
    hi = v.max() 
    
    for _ in range(max_iter):
        
        #find tau by bisecting the lower bounds of tau 
        tau = 0.5 * (lo + hi)
        
        #shift the weights by tau (so the sum equals "total") and project the shifted (weights - tau) between 0 and cap
        w = np.clip(v - tau, 0.0, cap)
        
        #create a variable for the sum of all the weihgts
        w_sum = w.sum()

        #check if the sum adds to "total" within tolerance (if yes, return projected weights - if no, change lo/hi and recompute bisection to find tau so it does sum to "total")
        if abs(w_sum - total) < tol:
            return w
        if w_sum > total:
            lo = tau 
        else:
            hi = tau

    return np.clip(v - 0.5*(lo+hi), 0.0, cap)


#function to calculate the objective function of ridge regression

def ridge_objective(w, cov_matrix, mu, lam = 1, gamma = 1):
    return 0.5 * gamma * (w.T @ cov_matrix @ w) - (mu.T @ w) + 0.5 * lam * (w.T @ w)

def ridge_grad(w, cov_matrix, mu, lam = 1, gamma = 1):
    return gamma * (cov_matrix @ w) - mu + lam * w


# def ridge_objective(w, cov_matrix, mu, lam):
#     return 0.5 * (w.T @ cov_matrix @ w) - (mu.T @ w) + 0.5 * lam * (w.T @ w)

# #function to calculate the gradient of ridge regression
# def ridge_grad(w, cov_matrix, mu, lam):
#     return cov_matrix @ w - mu + lam * w

def ridge_portfolio_pgd(cov_matrix, mu, lam, step_size, gamma=1.0, cap=0.4,
                        tol=1e-8, max_iter=20000, w_init=None):
    """
    Projected GD for ridge mean-variance objective with constraints:
      sum(w)=1, 0<=w<=cap
    """
    p = len(mu)
    w = (np.ones(p) / p) if w_init is None else w_init.copy()

    obj_vals = []
    for _ in range(max_iter):
        grad = ridge_grad(w, cov_matrix, mu, lam=lam, gamma=gamma)
        w_new = projection(w - step_size * grad, total=1.0, cap=cap)

        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break

        w = w_new
        # logging the *actual* objective (now includes gamma)
        obj_vals.append(0.5 * gamma * (w.T @ cov_matrix @ w) - (mu.T @ w) + 0.5 * lam * (w.T @ w))

    return w, obj_vals

# Parameters
risk_free_rate = rfr_cash
cap = 0.08
threshold = 0.0
lam = 1
step_size = 0.01
gamma = 8



weights, obj_vals_ridge = ridge_portfolio_pgd(cov_matrix, mu, lam, step_size, cap=cap, gamma = gamma)

#create a pandas series of tickers and corresponding weights 
weights_series = pd.Series(weights, index=tickers)










#apply heuristic of eliminating companies below threshold weight:
weights_series = weights_series[weights_series.abs() > threshold]
weights_series /= weights_series.sum()


#recalcaulate the metrics after the reweighting
w_full = pd.Series(0.0, index=log_returns.columns)
w_full.loc[weights_series.index] = weights_series.values
w_full_np = w_full.values










#print the companies ticker and corresponding weights to 4dp
print("\nRidge Portfolio Weights:\n")
for t, w in weights_series.sort_values(ascending=False).items():
    print(f"{t}: {w:.16f}")

#calculate and print the key data as defined earlier
ridge_return = expected_return(w_full_np, log_returns)
ridge_vol = standard_deviation(w_full_np, cov_matrix)
ridge_sharpe = sharpe_ratio(w_full_np, log_returns, cov_matrix, risk_free_rate)

print("\nRIDGE Portfolio Performance:")
print(f"Expected Return: {ridge_return:.4f}")
print(f"Volatility:      {ridge_vol:.4f}")
print(f"Sharpe Ratio:    {ridge_sharpe:.4f}")



plot_series = weights_series.reindex(weights_series.abs().sort_values(ascending=False).index)

plt.figure(figsize=(10, 6))
plt.bar(plot_series.index, plot_series.values)
plt.xticks(rotation=45)
plt.ylabel("Weight")
plt.title("Ridge Regression Portfolio Weights (sorted by |weight|)")
plt.tight_layout()
plt.show()


# --- grid search settings ---
lam_values = np.arange(1, 21, 1)
gamma_values = np.arange(1, 21, 1)

cap = 0.08
threshold = 0.0
step_size = 0.01
risk_free_rate = rfr_cash  # keep consistent with your synthetic cash

# results holder: rows=gamma, cols=lambda
sharpe_grid = np.full((len(gamma_values), len(lam_values)), np.nan, dtype=float)

# column names for alignment
asset_cols = log_returns.columns

for gi, gamma in enumerate(gamma_values):
    w_start = None
    print(f"gamma = {gamma}")

    for li, lam in enumerate(lam_values):
        print(f"  lambda = {lam}")

        # solve
        weights, _ = ridge_portfolio_pgd(
            cov_matrix, mu, lam=lam, step_size=step_size, gamma=gamma,
            cap=cap, tol=1e-8, max_iter=20000, w_init=w_start
        )
        w_start = weights  # warm-start next lambda

        # convert to Series with correct labels
        weights_series = pd.Series(weights, index=asset_cols)

        # heuristic rebalance
        weights_series = weights_series[weights_series.abs() > threshold]
        weights_series = weights_series / weights_series.sum()

        # build full vector aligned to log_returns columns
        w_full = pd.Series(0.0, index=asset_cols)
        w_full.loc[weights_series.index] = weights_series.values
        w_full_np = w_full.values

        # compute sharpe; guard against ~0 vol
        vol = standard_deviation(w_full_np, cov_matrix)
        sharpe = np.nan if vol < 1e-12 else sharpe_ratio(w_full_np, log_returns, cov_matrix, risk_free_rate)

        sharpe_grid[gi, li] = sharpe


# --- HEATMAP plot ---
fig, ax = plt.subplots(figsize=(10, 7))

im = ax.imshow(
    sharpe_grid,
    origin="lower",
    aspect="auto",
    cmap="RdBu",  # low=red, high=blue
    extent=[lam_values.min(), lam_values.max(), gamma_values.min(), gamma_values.max()],
    interpolation="nearest"
)

ax.set_xlabel("lambda (λ)")
ax.set_ylabel("gamma (γ)")
ax.set_title("Sharpe ratio heatmap over (λ, γ) grid")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Sharpe ratio")

# --- OPTIONAL: mark the best (lambda, gamma) ---
best_idx = np.nanargmax(sharpe_grid)
best_gi, best_li = np.unravel_index(best_idx, sharpe_grid.shape)
best_gamma = gamma_values[best_gi]
best_lam = lam_values[best_li]
best_sharpe = sharpe_grid[best_gi, best_li]

ax.scatter(best_lam, best_gamma, marker="x", s=150, color="black")
ax.text(best_lam, best_gamma, f"  {best_sharpe:.3f}", color="black", va="center")

print("Best (lambda, gamma, sharpe):", best_lam, best_gamma, best_sharpe)

plt.tight_layout()
plt.show()
