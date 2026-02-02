# -*- coding: utf-8 -*-
"""
Portfolio Optimization with Simulation and Rebalancing
Integrates with existing optimization code to simulate real investment performance
"""
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ============================================================================
# EXISTING OPTIMIZATION CODE (with minor modifications for reusability)
# ============================================================================

def fetch_price_data(tickers, start_date, end_date):
    """Fetch adjusted close prices for tickers"""
    adj_close_df = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            adj_close_df[ticker] = data['Close']
        except:
            print(f"Warning: Could not fetch data for {ticker}")
    return adj_close_df.dropna(axis=1)  # Drop tickers with missing data

def calculate_log_returns(prices):
    """Calculate log returns from price data"""
    return np.log(prices / prices.shift(1)).dropna()

def standard_deviation(weights, cov_matrix):
    """Calculate portfolio standard deviation"""
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    """Calculate expected annual return"""
    return np.sum(log_returns.mean() * weights) * 252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    """Calculate Sharpe ratio"""
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    """Negative Sharpe ratio for minimization"""
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

def optimize_portfolio(tickers, log_returns, cov_matrix, risk_free_rate=0.02, max_weight=0.4, threshold=1e-2):
    """
    Find optimal portfolio weights using Sharpe ratio maximization
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    log_returns : DataFrame
        Log returns for each asset
    cov_matrix : DataFrame
        Covariance matrix
    risk_free_rate : float
        Risk-free rate (default 2%)
    max_weight : float
        Maximum weight per asset (default 40%)
    threshold : float
        Minimum weight threshold (default 1%)
    
    Returns:
    --------
    dict : Contains optimal weights, expected return, volatility, Sharpe ratio
    """
    # Constraints and bounds
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, max_weight) for _ in range(len(tickers))]
    initial_weights = np.array([1/len(tickers)] * len(tickers))
    
    # Optimize
    optimized_results = minimize(
        neg_sharpe_ratio, 
        initial_weights, 
        args=(log_returns, cov_matrix, risk_free_rate), 
        method='SLSQP', 
        constraints=constraints, 
        bounds=bounds
    )
    
    optimal_weights = optimized_results.x
    
    # Filter out weights below threshold and renormalize
    weights = pd.Series(optimal_weights, index=tickers)
    weights = weights[weights.abs() > threshold]
    weights /= weights.sum()
    
    # Calculate portfolio metrics
    # Create full weights array for metrics calculation
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
# SIMULATION AND REBALANCING CODE
# ============================================================================

class PortfolioSimulator:
    """
    Simulates portfolio performance with periodic rebalancing
    """
    
    def __init__(self, tickers, initial_investment=10000, risk_free_rate=0.02, 
                 max_weight=0.4, threshold=1e-2):
        """
        Initialize portfolio simulator
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        initial_investment : float
            Initial investment amount
        risk_free_rate : float
            Annual risk-free rate
        max_weight : float
            Maximum weight per asset
        threshold : float
            Minimum weight threshold
        """
        self.tickers = tickers
        self.initial_investment = initial_investment
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.threshold = threshold
        
        # Track simulation results
        self.portfolio_history = []
        self.rebalance_events = []
        self.daily_values = []
        
    def simulate(self, start_date, end_date, rebalance_frequency='annual', 
                 lookback_years=5, transaction_cost=0.0):
        """
        Run portfolio simulation with rebalancing
        
        Parameters:
        -----------
        start_date : str or datetime
            Simulation start date
        end_date : str or datetime
            Simulation end date
        rebalance_frequency : str
            'annual', 'semi-annual', 'quarterly', 'monthly'
        lookback_years : int
            Years of historical data for optimization
        transaction_cost : float
            Transaction cost as percentage (e.g., 0.001 for 0.1%)
        
        Returns:
        --------
        DataFrame : Daily portfolio values
        """
        print("\n" + "="*80)
        print("PORTFOLIO SIMULATION WITH REBALANCING")
        print("="*80)
        print(f"Initial Investment: ${self.initial_investment:,.2f}")
        print(f"Simulation Period: {start_date} to {end_date}")
        print(f"Rebalance Frequency: {rebalance_frequency}")
        print(f"Number of Assets: {len(self.tickers)}")
        print("="*80 + "\n")
        
        # Convert dates if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Fetch price data for entire simulation period
        print("Fetching price data...")
        simulation_prices = fetch_price_data(self.tickers, start_date, end_date)
        
        # Update tickers list in case some were dropped
        self.tickers = list(simulation_prices.columns)
        print(f"Successfully fetched data for {len(self.tickers)} assets\n")
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(
            simulation_prices.index, rebalance_frequency
        )
        
        # Initialize portfolio holdings (shares of each stock)
        holdings = {}  # {ticker: number of shares}
        
        # INITIAL OPTIMIZATION AND PURCHASE
        print("INITIAL PORTFOLIO SETUP")
        print("-"*80)
        
        # Get historical data for initial optimization
        opt_start = start_date - timedelta(days=lookback_years*365)
        opt_prices = fetch_price_data(self.tickers, opt_start, start_date)
        opt_returns = calculate_log_returns(opt_prices)
        opt_cov_matrix = opt_returns.cov() * 252
        
        # Optimize
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
        
        # Initial purchase
        first_day_prices = simulation_prices.iloc[0]
        cash_remaining = self.initial_investment
        
        for ticker, weight in optimal['weights'].items():
            target_value = self.initial_investment * weight
            shares = target_value / first_day_prices[ticker]
            holdings[ticker] = shares
            cash_remaining -= shares * first_day_prices[ticker]
        
        # Track initial portfolio
        self.rebalance_events.append({
            'date': simulation_prices.index[0],
            'action': 'initial_purchase',
            'weights': optimal['weights'],
            'portfolio_value': self.initial_investment,
            'cash': cash_remaining
        })
        
        # SIMULATE DAY BY DAY
        print("Running simulation...")
        for i, date in enumerate(simulation_prices.index):
            current_prices = simulation_prices.loc[date]
            
            # Calculate portfolio value
            portfolio_value = cash_remaining
            for ticker, shares in holdings.items():
                portfolio_value += shares * current_prices[ticker]
            
            # Record daily value
            self.daily_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': cash_remaining
            })
            
            # Check if rebalancing date
            if date in rebalance_dates and date != simulation_prices.index[0]:
                print(f"\n{'='*80}")
                print(f"REBALANCING EVENT - {date.date()}")
                print(f"{'='*80}")
                print(f"Portfolio Value Before Rebalancing: ${portfolio_value:,.2f}")
                
                # Calculate current allocation
                current_weights = {}
                for ticker, shares in holdings.items():
                    current_value = shares * current_prices[ticker]
                    current_weights[ticker] = current_value / portfolio_value
                
                print("\nCurrent Allocation:")
                for ticker, weight in sorted(current_weights.items(), key=lambda x: x[1], reverse=True):
                    if weight > 0.001:  # Only show meaningful weights
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
                
                # Calculate drift
                max_drift = max([abs(current_weights.get(ticker, 0) - optimal['weights'].get(ticker, 0)) 
                                for ticker in set(list(current_weights.keys()) + list(optimal['weights'].keys()))])
                print(f"\nMaximum Drift from Target: {max_drift:.4f} ({max_drift*100:.2f}%)")
                
                # Execute rebalancing trades
                print("\nExecuting Trades:")
                transaction_costs = 0
                
                # Sell positions not in new optimal portfolio or reduce overweight positions
                for ticker in list(holdings.keys()):
                    current_value = holdings[ticker] * current_prices[ticker]
                    target_weight = optimal['weights'].get(ticker, 0)
                    target_value = portfolio_value * target_weight
                    
                    if target_value < current_value:
                        # Sell some shares
                        shares_to_sell = (current_value - target_value) / current_prices[ticker]
                        if shares_to_sell > 0.01:  # Only execute meaningful trades
                            sell_value = shares_to_sell * current_prices[ticker]
                            transaction_costs += sell_value * transaction_cost
                            cash_remaining += sell_value - (sell_value * transaction_cost)
                            holdings[ticker] -= shares_to_sell
                            print(f"  SELL {shares_to_sell:.4f} shares of {ticker} @ ${current_prices[ticker]:.2f} = ${sell_value:,.2f}")
                        
                        if holdings[ticker] < 0.01:
                            del holdings[ticker]
                
                # Buy new positions or increase underweight positions
                for ticker, target_weight in optimal['weights'].items():
                    target_value = portfolio_value * target_weight
                    current_value = holdings.get(ticker, 0) * current_prices[ticker]
                    
                    if target_value > current_value:
                        # Buy more shares
                        shares_to_buy = (target_value - current_value) / current_prices[ticker]
                        if shares_to_buy > 0.01:  # Only execute meaningful trades
                            buy_value = shares_to_buy * current_prices[ticker]
                            cost_with_fee = buy_value * (1 + transaction_cost)
                            
                            if cost_with_fee <= cash_remaining:
                                transaction_costs += buy_value * transaction_cost
                                cash_remaining -= cost_with_fee
                                holdings[ticker] = holdings.get(ticker, 0) + shares_to_buy
                                print(f"  BUY {shares_to_buy:.4f} shares of {ticker} @ ${current_prices[ticker]:.2f} = ${buy_value:,.2f}")
                
                if transaction_costs > 0:
                    print(f"\nTotal Transaction Costs: ${transaction_costs:,.2f}")
                
                # Record rebalancing event
                new_portfolio_value = cash_remaining
                for ticker, shares in holdings.items():
                    new_portfolio_value += shares * current_prices[ticker]
                
                print(f"Portfolio Value After Rebalancing: ${new_portfolio_value:,.2f}")
                
                self.rebalance_events.append({
                    'date': date,
                    'action': 'rebalance',
                    'weights': optimal['weights'],
                    'portfolio_value': new_portfolio_value,
                    'cash': cash_remaining,
                    'transaction_costs': transaction_costs,
                    'drift': max_drift
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.daily_values).set_index('date')
        
        print("\n" + "="*80)
        print("SIMULATION COMPLETE")
        print("="*80)
        
        return results_df
    
    def _get_rebalance_dates(self, date_index, frequency):
        """Generate rebalancing dates"""
        rebalance_dates = [date_index[0]]
        
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
        """Generate performance summary statistics"""
        if not self.daily_values:
            print("No simulation data available.")
            return None
        
        values_df = pd.DataFrame(self.daily_values).set_index('date')
        
        initial_value = values_df['value'].iloc[0]
        final_value = values_df['value'].iloc[-1]
        
        # Calculate returns
        total_return = (final_value - initial_value) / initial_value
        daily_returns = values_df['value'].pct_change().dropna()
        
        # Annualized metrics
        num_years = len(values_df) / 252
        annualized_return = (1 + total_return) ** (1 / num_years) - 1
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        
        # Drawdown analysis
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
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
        print(f"Maximum Drawdown: {summary['Maximum Drawdown']*100:.2f}%")
        print(f"Max Drawdown Duration: {summary['Max Drawdown Duration (days)']} days")
        print(f"Number of Rebalances: {summary['Number of Rebalances']}")
        print("="*80)
        
        return summary
    
    def plot_performance(self, benchmark_ticker='SPY'):
        """Plot portfolio performance"""
        if not self.daily_values:
            print("No simulation data available.")
            return
        
        values_df = pd.DataFrame(self.daily_values).set_index('date')
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Portfolio value over time
        ax1 = axes[0]
        ax1.plot(values_df.index, values_df['value'], linewidth=2, label='Portfolio', color='#2E86AB')
        
        # Fetch benchmark
        try:
            benchmark = yf.download(benchmark_ticker, start=values_df.index[0], 
                                   end=values_df.index[-1], progress=False)['Close']
            benchmark_normalized = (benchmark / benchmark.iloc[0]) * self.initial_investment
            ax1.plot(benchmark.index, benchmark_normalized, linewidth=2, 
                    label=f'{benchmark_ticker} (Benchmark)', color='#A23B72', alpha=0.7)
        except:
            pass
        
        # Mark rebalancing dates
        for event in self.rebalance_events:
            if event['action'] == 'rebalance':
                ax1.axvline(x=event['date'], color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot 2: Returns over time
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
        
        # Plot 3: Drawdown
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
        plt.savefig('/mnt/user-data/outputs/portfolio_simulation_results.png', 
                   dpi=300, bbox_inches='tight')
        print("\nPerformance charts saved to 'portfolio_simulation_results.png'")
        return fig
    
    def plot_allocation_evolution(self):
        """Plot how portfolio allocation changes over time"""
        if not self.rebalance_events:
            print("No rebalancing data available.")
            return
        
        # Extract weights over time
        dates = [event['date'] for event in self.rebalance_events]
        all_tickers = set()
        for event in self.rebalance_events:
            all_tickers.update(event['weights'].keys())
        
        weights_over_time = {ticker: [] for ticker in all_tickers}
        
        for event in self.rebalance_events:
            for ticker in all_tickers:
                weights_over_time[ticker].append(event['weights'].get(ticker, 0))
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort tickers by average weight
        avg_weights = {ticker: np.mean(weights) for ticker, weights in weights_over_time.items()}
        sorted_tickers = sorted(avg_weights.keys(), key=lambda x: avg_weights[x], reverse=True)
        
        # Only plot top holdings for clarity
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
        plt.savefig('/mnt/user-data/outputs/allocation_evolution.png', 
                   dpi=300, bbox_inches='tight')
        print("Allocation evolution chart saved to 'allocation_evolution.png'")
        return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Your existing ticker list
    tickers = [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'GOOG', 'TSLA',
        'LLY', 'AVGO', 'JPM', 'UNH', 'XOM', 'V', 'MA', 'HD', 'PG', 'COST', 'MRK',
        'ABBV', 'CVX', 'PEP', 'ADBE', 'KO', 'CRM', 'NFLX', 'AMD', 'TMO', 'WMT',
        'DIS', 'MCD', 'INTC', 'ACN', 'QCOM', 'CSCO', 'TXN', 'NEE', 'LIN', 'PM',
        'HON', 'AMGN', 'IBM', 'UPS', 'MS', 'GS', 'SBUX', 'CAT', 'BA'
    ]
    
    # Create simulator
    simulator = PortfolioSimulator(
        tickers=tickers,
        initial_investment=10000,
        risk_free_rate=0.02,
        max_weight=0.4,
        threshold=1e-2
    )
    
    # Run simulation
    # Example: Simulate from Jan 2020 to Dec 2024 with annual rebalancing
    results = simulator.simulate(
        start_date='2020-01-01',
        end_date='2024-12-31',
        rebalance_frequency='annual',  # Can be: 'annual', 'semi-annual', 'quarterly', 'monthly'
        lookback_years=5,
        transaction_cost=0.0  # 0% transaction costs (can add 0.001 for 0.1% costs)
    )
    
    # Get performance summary
    summary = simulator.get_performance_summary()
    
    # Generate plots
    simulator.plot_performance(benchmark_ticker='SPY')
    simulator.plot_allocation_evolution()
    
    print("\n✅ Simulation complete! Check the generated charts.")
