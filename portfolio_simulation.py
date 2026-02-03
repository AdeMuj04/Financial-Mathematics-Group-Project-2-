# -*- coding: utf-8 -*-
"""
Portfolio Optimization with Simulation and Rebalancing
Display-only version - Shows plots without saving files
"""
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ============================================================================
# SECTION 1: OPTIMIZATION FUNCTIONS
# These functions handle portfolio optimization (finding best weights)
# ============================================================================

def fetch_price_data(tickers, start_date, end_date):
    """
    Downloads historical stock prices from Yahoo Finance
    
    What it does:
    - Takes a list of stock tickers (like ['AAPL', 'MSFT'])
    - Downloads closing prices for the date range
    - Returns a DataFrame with prices for each stock
    """
    adj_close_df = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            adj_close_df[ticker] = data['Close']
        except:
            print(f"Warning: Could not fetch data for {ticker}")
    return adj_close_df.dropna(axis=1)  # Remove stocks with missing data

def calculate_log_returns(prices):
    """
    Calculates daily returns using logarithmic method
    
    What it does:
    - Log returns = ln(Price_today / Price_yesterday)
    - More mathematically sound than simple returns
    - Makes it easier to calculate portfolio statistics
    
    Example:
    If AAPL goes from $100 to $105, log return = ln(105/100) = 0.0488 (4.88%)
    """
    return np.log(prices / prices.shift(1)).dropna()

def standard_deviation(weights, cov_matrix):
    """
    Calculates portfolio volatility (risk)
    
    What it does:
    - Uses portfolio theory formula: sqrt(weights^T × Covariance × weights)
    - Accounts for how stocks move together (correlation)
    - Returns annualized standard deviation (volatility)
    
    Example:
    If portfolio has 20% volatility, we expect it to fluctuate ±20% per year
    """
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    """
    Calculates expected annual return of the portfolio
    
    What it does:
    - Takes average daily return for each stock
    - Multiplies by portfolio weights
    - Annualizes by multiplying by 252 (trading days per year)
    
    Example:
    If AAPL weight = 30%, AAPL avg daily return = 0.1%, contribution = 0.3 × 0.1% = 0.03% per day
    """
    return np.sum(log_returns.mean() * weights) * 252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    """
    Calculates Sharpe ratio (risk-adjusted return)
    
    What it does:
    - Sharpe = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
    - Measures return per unit of risk
    - Higher is better (getting more return for the risk you take)
    
    Example:
    Portfolio return = 15%, Risk-free = 2%, Volatility = 20%
    Sharpe = (15% - 2%) / 20% = 0.65
    """
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    """
    Negative Sharpe ratio for optimization
    
    Why negative?
    - The optimizer MINIMIZES functions
    - We want to MAXIMIZE Sharpe ratio
    - So we minimize the negative of Sharpe (which is the same as maximizing Sharpe)
    """
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

def optimize_portfolio(tickers, log_returns, cov_matrix, risk_free_rate=0.02, max_weight=0.4, threshold=1e-2):
    """
    Finds the optimal portfolio weights that maximize Sharpe ratio
    
    What it does:
    1. Sets up constraints: weights must sum to 1 (100%)
    2. Sets up bounds: each weight between 0% and 40% (no shorting, max concentration)
    3. Uses an optimizer to find weights that maximize Sharpe ratio
    4. Filters out tiny weights below 1% threshold
    5. Renormalizes weights to sum to 100%
    
    Example output:
    {
        'weights': {'AAPL': 0.35, 'MSFT': 0.40, 'GOOGL': 0.25},
        'expected_return': 0.18,  # 18% annual return
        'volatility': 0.22,       # 22% annual volatility
        'sharpe_ratio': 0.73      # Risk-adjusted return measure
    }
    """
    # Constraint: all weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    
    # Bounds: each weight between 0 and max_weight (default 40%)
    bounds = [(0, max_weight) for _ in range(len(tickers))]
    
    # Start with equal weights as initial guess
    initial_weights = np.array([1/len(tickers)] * len(tickers))
    
    # Run the optimizer (SLSQP = Sequential Least Squares Programming)
    optimized_results = minimize(
        neg_sharpe_ratio,  # Function to minimize (negative Sharpe)
        initial_weights,   # Starting point
        args=(log_returns, cov_matrix, risk_free_rate),  # Additional arguments
        method='SLSQP',    # Optimization algorithm
        constraints=constraints,  # Weights must sum to 1
        bounds=bounds      # Each weight between 0 and max_weight
    )
    
    optimal_weights = optimized_results.x
    
    # Filter out tiny weights and renormalize
    weights = pd.Series(optimal_weights, index=tickers)
    weights = weights[weights.abs() > threshold]  # Keep only weights > 1%
    weights /= weights.sum()  # Make sure they sum to 1 again
    
    # Calculate portfolio metrics with optimal weights
    full_weights = np.zeros(len(tickers))
    for ticker, weight in weights.items():
        idx = tickers.index(ticker)
        full_weights[idx] = weight
    
    portfolio_return = expected_return(full_weights, log_returns)
    portfolio_volatility = standard_deviation(full_weights, cov_matrix)
    portfolio_sharpe = sharpe_ratio(full_weights, log_returns, cov_matrix, risk_free_rate)
    
    return {
        'weights': weights.to_dict(),
        'expected_return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': portfolio_sharpe
    }


# ============================================================================
# SECTION 2: PORTFOLIO SIMULATOR CLASS
# This simulates what happens when you actually invest money over time
# ============================================================================

class PortfolioSimulator:
    """
    Simulates real portfolio performance with periodic rebalancing
    
    What it tracks:
    - Day-by-day portfolio value
    - When and how to rebalance
    - Actual trades executed
    - Performance metrics
    """
    
    def __init__(self, tickers, initial_investment=10000, risk_free_rate=0.02, 
                 max_weight=0.4, threshold=1e-2):
        """
        Initialize the simulator
        
        Parameters explained:
        - tickers: List of stocks to include ['AAPL', 'MSFT', ...]
        - initial_investment: How much money you start with ($10,000 default)
        - risk_free_rate: Risk-free rate for Sharpe calculation (2% default)
        - max_weight: Maximum % in any one stock (40% default)
        - threshold: Minimum % to hold a position (1% default)
        """
        self.tickers = tickers
        self.initial_investment = initial_investment
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.threshold = threshold
        
        # These will store simulation results
        self.portfolio_history = []
        self.rebalance_events = []
        self.daily_values = []
        
    def simulate(self, start_date, end_date, rebalance_frequency='annual', 
                 lookback_years=5, transaction_cost=0.0):
        """
        Run the full portfolio simulation
        
        What this does (step-by-step):
        
        STEP 1: SETUP
        - Download price data for all stocks for the entire period
        - Determine when rebalancing will happen (e.g., Jan 1 each year)
        
        STEP 2: INITIAL INVESTMENT (Day 1)
        - Look back at past data to find optimal weights
        - Buy stocks according to those weights
        - Record what you bought
        
        STEP 3: DAILY TRACKING
        - Every day, calculate portfolio value based on current prices
        - Track how much each stock is worth
        - Record daily value
        
        STEP 4: REBALANCING (e.g., every Jan 1)
        - Calculate how current weights have drifted from target
        - Re-run optimization using recent data to get NEW optimal weights
        - Calculate what trades to make to get back to target
        - Execute those trades (sell some stocks, buy others)
        - Pay transaction costs if applicable
        
        STEP 5: REPEAT
        - Continue daily tracking until end date
        - Rebalance whenever the schedule says to
        
        RESULT: A complete history of your portfolio value over time
        """
        
        print("\n" + "="*80)
        print("PORTFOLIO SIMULATION WITH REBALANCING")
        print("="*80)
        print(f"Initial Investment: ${self.initial_investment:,.2f}")
        print(f"Simulation Period: {start_date} to {end_date}")
        print(f"Rebalance Frequency: {rebalance_frequency}")
        print(f"Number of Assets: {len(self.tickers)}")
        print("="*80 + "\n")
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # STEP 1: Download all price data
        print("Fetching price data...")
        simulation_prices = fetch_price_data(self.tickers, start_date, end_date)
        self.tickers = list(simulation_prices.columns)  # Update in case some failed
        print(f"Successfully fetched data for {len(self.tickers)} assets\n")
        
        # Get list of rebalancing dates
        rebalance_dates = self._get_rebalance_dates(simulation_prices.index, rebalance_frequency)
        
        # Initialize holdings (number of shares of each stock)
        holdings = {}  # Example: {'AAPL': 10.5, 'MSFT': 8.2, ...}
        
        # STEP 2: INITIAL INVESTMENT
        print("INITIAL PORTFOLIO SETUP")
        print("-"*80)
        
        # Get historical data to optimize (look back 5 years from start)
        opt_start = start_date - timedelta(days=lookback_years*365)
        opt_prices = fetch_price_data(self.tickers, opt_start, start_date)
        opt_returns = calculate_log_returns(opt_prices)
        opt_cov_matrix = opt_returns.cov() * 252  # Annualized covariance
        
        # Find optimal weights
        optimal = optimize_portfolio(
            self.tickers, opt_returns, opt_cov_matrix, 
            self.risk_free_rate, self.max_weight, self.threshold
        )
        
        print("Initial Optimal Weights:")
        for ticker, weight in sorted(optimal['weights'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {ticker}: {weight:.4f} ({weight*100:.2f}%)")
        print(f"\nExpected Annual Return: {optimal['expected_return']:.4f} ({optimal['expected_return']*100:.2f}%)")
        print(f"Expected Volatility: {optimal['volatility']:.4f} ({optimal['volatility']*100:.2f}%)")
        print(f"Sharpe Ratio: {optimal['sharpe_ratio']:.4f}\n")
        
        # Buy stocks on day 1
        first_day_prices = simulation_prices.iloc[0]
        cash_remaining = self.initial_investment
        
        for ticker, weight in optimal['weights'].items():
            target_value = self.initial_investment * weight  # How much $ to invest in this stock
            shares = target_value / first_day_prices[ticker]  # How many shares to buy
            holdings[ticker] = shares
            cash_remaining -= shares * first_day_prices[ticker]
        
        # Record initial purchase
        self.rebalance_events.append({
            'date': simulation_prices.index[0],
            'action': 'initial_purchase',
            'weights': optimal['weights'],
            'portfolio_value': self.initial_investment,
            'cash': cash_remaining
        })
        
        # STEP 3 & 4: DAILY TRACKING AND PERIODIC REBALANCING
        print("Running simulation...")
        for i, date in enumerate(simulation_prices.index):
            current_prices = simulation_prices.loc[date]
            
            # Calculate today's portfolio value
            portfolio_value = cash_remaining
            for ticker, shares in holdings.items():
                portfolio_value += shares * current_prices[ticker]
            
            # Record today's value
            self.daily_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': cash_remaining
            })
            
            # Check if today is a rebalancing date
            if date in rebalance_dates and date != simulation_prices.index[0]:
                print(f"\n{'='*80}")
                print(f"REBALANCING EVENT - {date.date()}")
                print(f"{'='*80}")
                print(f"Portfolio Value Before Rebalancing: ${portfolio_value:,.2f}")
                
                # Calculate current weights (how portfolio has drifted)
                current_weights = {}
                for ticker, shares in holdings.items():
                    current_value = shares * current_prices[ticker]
                    current_weights[ticker] = current_value / portfolio_value
                
                print("\nCurrent Allocation:")
                for ticker, weight in sorted(current_weights.items(), key=lambda x: x[1], reverse=True):
                    if weight > 0.001:
                        print(f"  {ticker}: {weight:.4f} ({weight*100:.2f}%)")
                
                # Re-optimize using recent data
                opt_end = date
                opt_start = date - timedelta(days=lookback_years*365)
                opt_prices = fetch_price_data(self.tickers, opt_start, opt_end)
                opt_returns = calculate_log_returns(opt_prices)
                opt_cov_matrix = opt_returns.cov() * 252
                
                optimal = optimize_portfolio(
                    self.tickers, opt_returns, opt_cov_matrix,
                    self.risk_free_rate, self.max_weight, self.threshold
                )
                
                print("\nNew Optimal Weights:")
                for ticker, weight in sorted(optimal['weights'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {ticker}: {weight:.4f} ({weight*100:.2f}%)")
                print(f"\nExpected Annual Return: {optimal['expected_return']:.4f} ({optimal['expected_return']*100:.2f}%)")
                print(f"Expected Volatility: {optimal['volatility']:.4f} ({optimal['volatility']*100:.2f}%)")
                print(f"Sharpe Ratio: {optimal['sharpe_ratio']:.4f}")
                
                # Calculate how far we've drifted
                max_drift = max([abs(current_weights.get(ticker, 0) - optimal['weights'].get(ticker, 0)) 
                                for ticker in set(list(current_weights.keys()) + list(optimal['weights'].keys()))])
                print(f"\nMaximum Drift from Target: {max_drift:.4f} ({max_drift*100:.2f}%)")
                
                # Execute rebalancing trades
                print("\nExecuting Trades:")
                transaction_costs = 0
                
                # SELL: Reduce positions that are overweight
                for ticker in list(holdings.keys()):
                    current_value = holdings[ticker] * current_prices[ticker]
                    target_weight = optimal['weights'].get(ticker, 0)
                    target_value = portfolio_value * target_weight
                    
                    if target_value < current_value:  # We have too much, need to sell
                        shares_to_sell = (current_value - target_value) / current_prices[ticker]
                        if shares_to_sell > 0.01:
                            sell_value = shares_to_sell * current_prices[ticker]
                            transaction_costs += sell_value * transaction_cost
                            cash_remaining += sell_value - (sell_value * transaction_cost)
                            holdings[ticker] -= shares_to_sell
                            print(f"  SELL {shares_to_sell:.4f} shares of {ticker} @ ${current_prices[ticker]:.2f} = ${sell_value:,.2f}")
                        
                        if holdings[ticker] < 0.01:  # Position too small, close it
                            del holdings[ticker]
                
                # BUY: Increase positions that are underweight
                for ticker, target_weight in optimal['weights'].items():
                    target_value = portfolio_value * target_weight
                    current_value = holdings.get(ticker, 0) * current_prices[ticker]
                    
                    if target_value > current_value:  # We don't have enough, need to buy
                        shares_to_buy = (target_value - current_value) / current_prices[ticker]
                        if shares_to_buy > 0.01:
                            buy_value = shares_to_buy * current_prices[ticker]
                            cost_with_fee = buy_value * (1 + transaction_cost)
                            
                            if cost_with_fee <= cash_remaining:
                                transaction_costs += buy_value * transaction_cost
                                cash_remaining -= cost_with_fee
                                holdings[ticker] = holdings.get(ticker, 0) + shares_to_buy
                                print(f"  BUY {shares_to_buy:.4f} shares of {ticker} @ ${current_prices[ticker]:.2f} = ${buy_value:,.2f}")
                
                if transaction_costs > 0:
                    print(f"\nTotal Transaction Costs: ${transaction_costs:,.2f}")
                
                # Calculate new portfolio value after rebalancing
                new_portfolio_value = cash_remaining
                for ticker, shares in holdings.items():
                    new_portfolio_value += shares * current_prices[ticker]
                
                print(f"Portfolio Value After Rebalancing: ${new_portfolio_value:,.2f}")
                
                # Record this rebalancing event
                self.rebalance_events.append({
                    'date': date,
                    'action': 'rebalance',
                    'weights': optimal['weights'],
                    'portfolio_value': new_portfolio_value,
                    'cash': cash_remaining,
                    'transaction_costs': transaction_costs,
                    'drift': max_drift
                })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.daily_values).set_index('date')
        
        print("\n" + "="*80)
        print("SIMULATION COMPLETE")
        print("="*80)
        
        return results_df
    
    def _get_rebalance_dates(self, date_index, frequency):
        """
        Generates the dates when rebalancing should occur
        
        What it does:
        - 'annual': First trading day of each year
        - 'semi-annual': First trading day of Jan and July
        - 'quarterly': First trading day of Jan, Apr, Jul, Oct
        - 'monthly': First trading day of each month
        """
        rebalance_dates = [date_index[0]]  # Always include first day
        
        if frequency == 'annual':
            for year in range(date_index[0].year, date_index[-1].year + 1):
                year_dates = date_index[date_index.year == year]
                if len(year_dates) > 0 and year_dates[0] not in rebalance_dates:
                    rebalance_dates.append(year_dates[0])
        
        elif frequency == 'semi-annual':
            for year in range(date_index[0].year, date_index[-1].year + 1):
                for month in [1, 7]:
                    period_dates = date_index[(date_index.year == year) & (date_index.month == month)]
                    if len(period_dates) > 0 and period_dates[0] not in rebalance_dates:
                        rebalance_dates.append(period_dates[0])
        
        elif frequency == 'quarterly':
            for year in range(date_index[0].year, date_index[-1].year + 1):
                for month in [1, 4, 7, 10]:
                    quarter_dates = date_index[(date_index.year == year) & (date_index.month == month)]
                    if len(quarter_dates) > 0 and quarter_dates[0] not in rebalance_dates:
                        rebalance_dates.append(quarter_dates[0])
        
        elif frequency == 'monthly':
            for year in range(date_index[0].year, date_index[-1].year + 1):
                for month in range(1, 13):
                    month_dates = date_index[(date_index.year == year) & (date_index.month == month)]
                    if len(month_dates) > 0 and month_dates[0] not in rebalance_dates:
                        rebalance_dates.append(month_dates[0])
        
        return sorted(set(rebalance_dates))
    
    def get_performance_summary(self):
        """
        Calculate and display performance statistics
        
        What it calculates:
        - Total return: How much you made/lost overall
        - Annualized return: Average yearly return
        - Annualized volatility: How much it fluctuated
        - Sharpe ratio: Risk-adjusted return
        - Maximum drawdown: Worst peak-to-trough decline
        - Drawdown duration: How long the worst decline lasted
        """
        if not self.daily_values:
            print("No simulation data available.")
            return None
        
        values_df = pd.DataFrame(self.daily_values).set_index('date')
        
        initial_value = values_df['value'].iloc[0]
        final_value = values_df['value'].iloc[-1]
        
        # Basic return calculations
        total_return = (final_value - initial_value) / initial_value
        daily_returns = values_df['value'].pct_change().dropna()
        
        # Annualized metrics
        num_years = len(values_df) / 252  # 252 trading days per year
        annualized_return = (1 + total_return) ** (1 / num_years) - 1
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        
        # Drawdown analysis
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ( running_max - cumulative) / running_max
        max_drawdown = drawdown.min()
        
        # Find longest drawdown period
        drawdown_start = None
        max_drawdown_days = 0
        current_drawdown_days = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_drawdown_days = i - drawdown_start
                max_drawdown_days = max(max_drawdown_days, current_drawdown_days)
            else:
                drawdown_start = None
                current_drawdown_days = 0
        
        summary = {
            'Initial Investment': initial_value,
            'Final Value': final_value,
            'Total Return': total_return,
            'Total Return ($)': final_value - initial_value,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Max Drawdown Duration (days)': max_drawdown_days,
            'Number of Rebalances': len([e for e in self.rebalance_events if e['action'] == 'rebalance']),
            'Simulation Days': len(values_df)
        }
        
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Initial Investment: ${summary['Initial Investment']:,.2f}")
        print(f"Final Portfolio Value: ${summary['Final Value']:,.2f}")
        print(f"Total Profit/Loss: ${summary['Total Return ($)']:,.2f}")
        print(f"Total Return: {summary['Total Return']*100:.2f}%")
        print(f"Annualized Return: {summary['Annualized Return']*100:.2f}%")
        print(f"Annualized Volatility: {summary['Annualized Volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {summary['Sharpe Ratio']:.4f}")
        print(f"Maximum Drawdown: {abs(summary['Maximum Drawdown']*100):.2f}%")
        print(f"Max Drawdown Duration: {summary['Max Drawdown Duration (days)']} days")
        print(f"Number of Rebalances: {summary['Number of Rebalances']}")
        print("="*80)
        
        return summary
    
    def plot_performance(self, benchmark_ticker='SPY'):
        """
        Create visual charts of portfolio performance
        
        What it shows:
        1. Portfolio value over time vs benchmark (SPY)
        2. Cumulative returns (%)
        3. Drawdown chart (shows worst declines)
        
        Red dashed lines = rebalancing dates
        """
        if not self.daily_values:
            print("No simulation data available.")
            return
        
        values_df = pd.DataFrame(self.daily_values).set_index('date')
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # CHART 1: Portfolio value over time
        ax1 = axes[0]
        ax1.plot(values_df.index, values_df['value'], linewidth=2, label='Portfolio', color='#2E86AB')
        
        # Add benchmark comparison
        try:
            benchmark = yf.download(benchmark_ticker, start=values_df.index[0], 
                                   end=values_df.index[-1], progress=False)['Close']
            benchmark_normalized = (benchmark / benchmark.iloc[0]) * self.initial_investment
            ax1.plot(benchmark.index, benchmark_normalized, linewidth=2, 
                    label=f'{benchmark_ticker} (Benchmark)', color='#A23B72', alpha=0.7)
        except:
            pass
        
        # Mark rebalancing dates with red lines
        for event in self.rebalance_events:
            if event['action'] == 'rebalance':
                ax1.axvline(x=event['date'], color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # CHART 2: Cumulative returns
        ax2 = axes[1]
        daily_returns = values_df['value'].pct_change().dropna()
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        ax2.plot(cumulative_returns.index, cumulative_returns * 100, 
                linewidth=2, color='#F18F01')
        ax2.fill_between(cumulative_returns.index, 0, cumulative_returns * 100, 
                         alpha=0.3, color='#F18F01')
        
        # Mark rebalancing dates
        for event in self.rebalance_events:
            if event['action'] == 'rebalance':
                ax2.axvline(x=event['date'], color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        ax2.set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Return (%)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # CHART 3: Drawdown (shows declines from peak)
        ax3 = axes[2]
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        ax3.fill_between(drawdown.index, 0, drawdown * 100, 
                        color='#C73E1D', alpha=0.6)
        ax3.plot(drawdown.index, drawdown * 100, linewidth=1, color='#8B0000')
        
        # Mark rebalancing dates
        for event in self.rebalance_events:
            if event['action'] == 'rebalance':
                ax3.axvline(x=event['date'], color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        ax3.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Drawdown (%)', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()  # This displays the plot instead of saving
        
        return fig
    
    def plot_allocation_evolution(self):
        """
        Shows how portfolio weights changed at each rebalancing
        
        What it shows:
        - Stacked bar chart
        - Each bar = one rebalancing event
        - Colors = different stocks
        - Height = % of portfolio
        """
        if not self.rebalance_events:
            print("No rebalancing data available.")
            return
        
        # Extract weights from each rebalancing event
        dates = [event['date'] for event in self.rebalance_events]
        all_tickers = set()
        for event in self.rebalance_events:
            all_tickers.update(event['weights'].keys())
        
        weights_over_time = {ticker: [] for ticker in all_tickers}
        
        for event in self.rebalance_events:
            for ticker in all_tickers:
                weights_over_time[ticker].append(event['weights'].get(ticker, 0))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort by average weight to show biggest holdings first
        avg_weights = {ticker: np.mean(weights) for ticker, weights in weights_over_time.items()}
        sorted_tickers = sorted(avg_weights.keys(), key=lambda x: avg_weights[x], reverse=True)
        
        # Only show top 15 holdings for clarity
        top_n = min(15, len(sorted_tickers))
        bottom = np.zeros(len(dates))
        
        colors = plt.cm.tab20(np.linspace(0, 1, top_n))
        
        for i, ticker in enumerate(sorted_tickers[:top_n]):
            values = np.array(weights_over_time[ticker]) * 100
            ax.bar(range(len(dates)), values, bottom=bottom, label=ticker, color=colors[i])
            bottom += values
        
        ax.set_xlabel('Rebalancing Event', fontsize=11)
        ax.set_ylabel('Portfolio Weight (%)', fontsize=11)
        ax.set_title('Portfolio Allocation Evolution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates], rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()  # Display instead of saving
        
        return fig


# ============================================================================
# SECTION 3: MAIN EXECUTION
# This is where you actually run the simulation
# ============================================================================

if __name__ == "__main__":
    # Your stock list
    tickers = [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'GOOG', 'TSLA',
        'LLY', 'AVGO', 'JPM', 'UNH', 'XOM', 'V', 'MA', 'HD', 'PG', 'COST', 'MRK',
        'ABBV', 'CVX', 'PEP', 'ADBE', 'KO', 'CRM', 'NFLX', 'AMD', 'TMO', 'WMT',
        'DIS', 'MCD', 'INTC', 'ACN', 'QCOM', 'CSCO', 'TXN', 'NEE', 'LIN', 'PM',
        'HON', 'AMGN', 'IBM', 'UPS', 'MS', 'GS', 'SBUX', 'CAT', 'BA'
    ]
    
    # Create the simulator
    simulator = PortfolioSimulator(
        tickers=tickers,
        initial_investment=1000000,     # Start with $10,000
        risk_free_rate=0.02,          # 2% risk-free rate
        max_weight=0.4,               # Max 40% in any stock
        threshold=1e-2                # Min 1% to hold
    )
    
    # Run the simulation
    results = simulator.simulate(
        start_date='2020-01-01',
        end_date='2024-12-31',
        rebalance_frequency='semi-annual',  # Rebalance once per year
        lookback_years=5,              # Use 5 years of data for optimization
        transaction_cost=0.0           # No transaction costs
    )
    
    # Get performance statistics
    summary = simulator.get_performance_summary()
    
    # Display the charts
    print("\nGenerating performance charts...")
    simulator.plot_performance(benchmark_ticker='SPY')
    
    print("\nGenerating allocation evolution chart...")
    simulator.plot_allocation_evolution()
    
    print("\n✅ Simulation complete! Charts displayed.")