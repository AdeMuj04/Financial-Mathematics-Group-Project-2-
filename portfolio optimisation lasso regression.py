# -*- coding: utf-8 -*-



#import libraries
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


#select tickers to be considered
tickers = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'GOOG', 'TSLA',
    'LLY', 'AVGO', 'JPM', 'UNH', 'XOM', 'V', 'MA', 'HD', 'PG', 'COST', 'MRK',
    'ABBV', 'CVX', 'PEP', 'ADBE', 'KO', 'CRM', 'NFLX', 'AMD', 'TMO', 'WMT',
    'DIS', 'MCD', 'INTC', 'ACN', 'QCOM', 'CSCO', 'TXN', 'NEE', 'LIN', 'PM',
    'HON', 'AMGN', 'IBM', 'UPS', 'MS', 'GS', 'SBUX', 'CAT', 'BA'
]

#select length of period for analysis 

years = 2
#select end (auto calculates start date based on years above)
end_date = datetime.today() - relativedelta(years = 10)
start_date = end_date - relativedelta(years = years)

#download adjusted close prices from yahoo finance and store in a dataframe
adj_close = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    adj_close[ticker] = data['Close']



#calculate log returns for all tickers
log_returns = np.log(adj_close / adj_close.shift(1)).dropna()

#calculate historical returns and covariance of downloaded data
mu = log_returns.mean().values * 252
cov_matrix = log_returns.cov().values * 252


def project_capped_simplex(w, cap=0.4):
    w = np.clip(w, 0.0, cap)
    if w.sum() > 0:
        w /= w.sum()
    return w


def expected_return(w, log_returns):
    return np.sum(log_returns.mean().values * w) * 252

def standard_deviation(w, cov_matrix):
    return np.sqrt(w.T @ cov_matrix @ w)

def sharpe_ratio(w, log_returns, cov_matrix, rfr):
    return (expected_return(w, log_returns) - rfr) / standard_deviation(w, cov_matrix)




def soft_threshold(w, thresh):
    return np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)

def lasso_grad(w, cov_matrix, mu):
    return cov_matrix @ w - mu




def lasso_portfolio_gd(cov_matrix, mu, lam, step_size, tol=1e-4, max_iter=50000, cap = 0.05):
    p = len(mu)
    w = np.ones(p) / p
    obj_vals = []

    for k in range(max_iter):
        grad = lasso_grad(w, cov_matrix, mu)
        
        w_temp = w - step_size * grad
        w_temp = soft_threshold(w_temp, lam * step_size)
        w_new  = project_capped_simplex(w_temp, cap)
        
        
        w_new = np.maximum(w_new, 0)
        if w_new.sum() > 0:
            w_new /= w_new.sum()

        if np.linalg.norm(w_new - w) < tol:
            break

        w = w_new
        obj = 0.5 * w.T @ cov_matrix @ w - mu.T @ w + lam * np.sum(np.abs(w))
        obj_vals.append(obj)

    return w, obj_vals




lam = 0.1
step_size = 0.01
risk_free_rate = 0.02
cap = 1
threshold = 0.05

weights_lasso, obj_vals = lasso_portfolio_gd(cov_matrix, mu, lam, step_size, cap = cap)




weights = pd.Series(weights_lasso, index=tickers)


weights = weights[weights.abs() > threshold]
weights /= weights.sum()

print("\nLasso Portfolio Weights:\n")
for t, w in weights.items():
    print(f"{t}: {w:.4f}")





lasso_return = expected_return(weights_lasso, log_returns)
lasso_vol = standard_deviation(weights_lasso, cov_matrix)
lasso_sharpe = sharpe_ratio(weights_lasso, log_returns, cov_matrix, risk_free_rate)

print("\nPortfolio Performance:")
print(f"Expected Return: {lasso_return:.4f}")
print(f"Volatility:      {lasso_vol:.4f}")
print(f"Sharpe Ratio:    {lasso_sharpe:.4f}")




plt.figure(figsize=(10, 6))
plt.bar(weights.index, weights.values)
plt.xticks(rotation=45)
plt.ylabel("Weight")
plt.title("Lasso Portfolio Weights")
plt.tight_layout()
plt.show()
