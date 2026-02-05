# -*- coding: utf-8 -*-



from dateutil.relativedelta import relativedelta
from datetime import datetime
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


tickers = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'GOOG', 'TSLA',
    'LLY', 'AVGO', 'JPM', 'UNH', 'XOM', 'V', 'MA', 'HD', 'PG', 'COST', 'MRK',
    'ABBV', 'CVX', 'PEP', 'ADBE', 'KO', 'CRM', 'NFLX', 'AMD', 'TMO', 'WMT',
    'DIS', 'MCD', 'INTC', 'ACN', 'QCOM', 'CSCO', 'TXN', 'NEE', 'LIN', 'PM',
    'HON', 'AMGN', 'IBM', 'UPS', 'MS', 'GS', 'SBUX', 'CAT', 'BA'
]


years = 2
end_date = datetime.today() - relativedelta(years = 10)   # your original choice
start_date = end_date - relativedelta(years=years)


adj_close = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    adj_close[ticker] = data['Close']



log_returns = np.log(adj_close / adj_close.shift(1)).dropna()
mu = log_returns.mean().values * 252
cov_matrix = log_returns.cov().values * 252



#define application of constraints (minimum and maximum weights, sum to 1)
def apply_constraints(w, cap=0.4):
    """
    Simple projection used in your code: clip to [0, cap] then renormalize (exclude weights less than threshold and redistributes proportionally).
    """
    # (w must lie between 0 and cap)
    w = np.clip(w, 0.0, cap)
    
    #redistribute weights proportionally so it sums to 1
    if w.sum() > 0:
        w /= w.sum()
    return w

def expected_return(w, log_returns):
    return np.sum(log_returns.mean().values * w) * 252

def standard_deviation(w, cov_matrix):
    return np.sqrt(w.T @ cov_matrix @ w)

def sharpe_ratio(w, log_returns, cov_matrix, rfr):
    return (expected_return(w, log_returns) - rfr) / standard_deviation(w, cov_matrix)





def ridge_closed_form(cov_matrix, mu, lam):
    p = len(mu)
    Id = np.identity(p)
    return np.linalg.solve(cov_matrix + lam * Id, mu)

def ridge_objective(w, cov_matrix, mu, lam):
    return 0.5 * (w.T @ cov_matrix @ w) - (mu.T @ w) + 0.5 * lam * (w.T @ w)

def ridge_grad(w, cov_matrix, mu, lam):
    return cov_matrix @ w - mu + lam * w

#ridge regression using gradient descent
def ridge_portfolio_gd(cov_matrix, mu, lam, step_size, tol=1e-6, max_iter=50000, cap=0.05):
    """
    Ridge gradient descent with FIXED step size, then projection
    to (constraints) (long-only, sum-to-1, max weight = cap).
    """
    p = len(mu)
    w = np.ones(p) / p
    obj_vals = []

    for i in range(max_iter):
        grad = ridge_grad(w, cov_matrix, mu, lam)

        w_new = w - step_size * grad
        w_new = apply_constraints(w_new, cap)

        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break

        w = w_new
        obj_vals.append(ridge_objective(w, cov_matrix, mu, lam))

    return w, obj_vals




# Parameters
risk_free_rate = 0.04
cap = 1
threshold = 0.03
lam = 5
step_size = 0.01

#ridge regression
#gradient descent approximation 
# weights_ridge, obj_vals_ridge = ridge_portfolio_gd(cov_matrix, mu, lam, step_size, cap=cap)


#closed-form solution then apply constraints
weights_ridge = apply_constraints(ridge_closed_form(cov_matrix, mu, lam), cap=cap)
obj_vals_ridge = []


#create a pandas series of tickers and corresponding weights
weights_ridge_series = pd.Series(weights_ridge, index=tickers)

#check if absolute of weights is greater than ridge and redistribute proportionally 
weights_ridge_series = weights_ridge_series[weights_ridge_series.abs() > threshold]
weights_ridge_series /= weights_ridge_series.sum()




#print the companies ticker and corresponding weights to 4dp
print("\nRidge Portfolio Weights:\n")
for t, w in weights_ridge_series.items():
    print(f"{t}: {w:.4f}")

#calculate and print the key data as defined earlier
ridge_return = expected_return(weights_ridge, log_returns)
ridge_vol = standard_deviation(weights_ridge, cov_matrix)
ridge_sharpe = sharpe_ratio(weights_ridge, log_returns, cov_matrix, risk_free_rate)

print("\nRIDGE Portfolio Performance:")
print(f"Expected Return: {ridge_return:.4f}")
print(f"Volatility:      {ridge_vol:.4f}")
print(f"Sharpe Ratio:    {ridge_sharpe:.4f}")


#plot the weights on a bar chart
plt.figure(figsize=(10, 6))
plt.bar(weights_ridge_series.index, weights_ridge_series.values)
plt.xticks(rotation=45)
plt.ylabel("Weight")
plt.title("Ridge Regression Portfolio Weights")
plt.tight_layout()
plt.show()
