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


years = 5
end_date = datetime.today() - relativedelta(years = 0)   # your original choice
start_date = end_date - relativedelta(years=years)


adj_close = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    adj_close[ticker] = data['Close']


log_returns = np.log(adj_close / adj_close.shift(1)).dropna()
mu = log_returns.mean().values * 252
cov_matrix = log_returns.cov().values * 252

def expected_return(w, log_returns):
    return np.sum(log_returns.mean().values * w) * 252

def standard_deviation(w, cov_matrix):
    return np.sqrt(w.T @ cov_matrix @ w)

def sharpe_ratio(w, log_returns, cov_matrix, rfr):
    return (expected_return(w, log_returns) - rfr) / standard_deviation(w, cov_matrix)

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
def ridge_objective(w, cov_matrix, mu, lam):
    return 0.5 * (w.T @ cov_matrix @ w) - (mu.T @ w) + 0.5 * lam * (w.T @ w)

#function to calculate the gradient of ridge regression
def ridge_grad(w, cov_matrix, mu, lam):
    return cov_matrix @ w - mu + lam * w


def ridge_portfolio_pgd(cov_matrix, mu, lam, step_size, cap=0.4, tol=1e-8, max_iter=20000):
    """
    Projected GD for ridge mean-variance objective with constraints:
      sum(w)=1, 0<=w<=cap

    Parameters
    ----------
    cov_matrix : matrix
        Covariance matrix
    mu : float
        expected return.
    lam : float
        regularisation strength.
    step_size : float
        fixed step gradient descent stepsize

    cap : float, optional
        maximum weight that can be given to any single stock. The default is 0.4.
    tol : float, optional
        stopping critera for grdadient descent (if the distance between the new and old weights is sufficiently small this will trigger). The default is 1e-8.
    max_iter : int, optional
        maximum number of iterations of gradient descent (if tol isnt reached before this number of iteratiions is reached, this will stop the loop). The default is 20000.

    Returns
    -------
    w : array
        weights of each stock.
    obj_vals : float
        objective values when output weights are input into the original objective function.
    """
 
    

    #p: length of expected returns
    p = len(mu)
    
    #create a temporary weights vector of equal entries 
    w = np.ones(p) / p
    
    #empty array for objective values to be stored in
    obj_vals = []

    for _ in range(max_iter):
        
        #calculate descent direction (gradient)
        grad = ridge_grad(w, cov_matrix, mu, lam)
        
        #calculate next step w_new =  (w - t*grad(f(x)), then project on desired conditions
        w_new = projection(w - step_size * grad, total = 1.0, cap = cap)
        
        #check if tolerance has been reached - if not repeat another iteration
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        
        #change weights to the weights we just projected
        w = w_new
        
        #append objective values to the array
        obj_vals.append(0.5 * w.T @ cov_matrix @ w - mu.T @ w + 0.5 * lam * w.T @ w)

    return w, obj_vals


# Parameters
risk_free_rate = 0.04
cap = 1
threshold = 0.03
lam = 2
step_size = 0.01

weights, obj_vals_ridge = ridge_portfolio_pgd(cov_matrix, mu, lam, step_size, cap=cap)

#create a pandas series of tickers and corresponding weights 
weights_series = pd.Series(weights, index=tickers)





#(uncomment below to use the heuristic)

#apply heuristic of eliminating companies below threshold weight:
# weights_series = weights_series[weights_series.abs() > threshold]
# weights_series /= weights_series.sum()












#print the companies ticker and corresponding weights to 4dp
print("\nRidge Portfolio Weights:\n")
for t, w in weights_series.items():
    print(f"{t}: {w:.4f}")

#calculate and print the key data as defined earlier
ridge_return = expected_return(weights, log_returns)
ridge_vol = standard_deviation(weights, cov_matrix)
ridge_sharpe = sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

print("\nRIDGE Portfolio Performance:")
print(f"Expected Return: {ridge_return:.4f}")
print(f"Volatility:      {ridge_vol:.4f}")
print(f"Sharpe Ratio:    {ridge_sharpe:.4f}")


#plot the weights on a bar chart
plt.figure(figsize=(10, 6))
plt.bar(weights_series.index, weights_series.values)
plt.xticks(rotation=45)
plt.ylabel("Weight")
plt.title("Ridge Regression Portfolio Weights")
plt.tight_layout()
plt.show()

