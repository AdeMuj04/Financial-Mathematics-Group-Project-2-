"""
Portfolio Rebalancing Engine Core Module
Implement portfolio rebalancing strategies including time-based triggers and threshold-based triggers
"""

import numpy as np
import pandas as pd
from datetime import datetime
from scipy import optimize
from .data_loader import get_current_weights, calculate_returns

class RebalanceEngine:
    def __init__(self, tickers, target_weights, rebalance_interval='quarterly', threshold=0.1):
        """
        Initialize rebalancing engine
        
        Parameters:
            tickers: List of asset tickers
            target_weights: Target weight vector
            rebalance_interval: Rebalancing frequency ('monthly', 'quarterly', 'annually')
            threshold: Threshold trigger ratio
        """
        self.tickers = tickers
        self.target_weights = np.array(target_weights)
        self.rebalance_interval = rebalance_interval
        self.threshold = threshold
        self.trade_log = []
        self.portfolio_values = []
        self.weights_history = []
        
    def backtest(self, prices):
        """
        Historical data backtesting
        
        Parameters:
            prices: Historical closing price DataFrame
        """
        if prices.empty:
            print("Price data is empty, cannot execute backtest")
            return
        
        dates = prices.index
        initial_capital = 10000000  # Initial capital: 10 million GBP
        
        # Initialize portfolio
        current_weights = self.target_weights.copy()
        portfolio_value = initial_capital
        
        print(f"Starting backtest, time range: {dates[0]} to {dates[-1]}")
        
        for i, date in enumerate(dates):
            # Get current prices
            current_prices = prices.loc[date]
            
            # Calculate current weights
            if i == 0:
                # First day: initialize with target weights
                current_weights = self.target_weights.copy()
            else:
                # Calculate actual weights based on price changes
                current_weights = self._calculate_actual_weights(
                    prices.iloc[i-1], prices.iloc[i], current_weights
                )
            
            # Record weight history
            self.weights_history.append({
                'date': date,
                'weights': current_weights.copy()
            })
            
            # Check if rebalancing is triggered
            if self._should_rebalance(date, current_weights):
                print(f"{date}: Rebalancing triggered")
                
                # Generate rebalancing instructions
                trades = self._generate_trades(current_prices, current_weights)
                
                # Execute trades and update weights
                current_weights = self._execute_trades(current_weights, trades)
                
                # Record trade log
                self.trade_log.append({
                    'date': date,
                    'trades': trades,
                    'reason': self._get_rebalance_reason(date, current_weights)
                })
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(current_prices, current_weights, initial_capital)
            self.portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'weights': current_weights.copy()
            })
        
        print(f"Backtest completed, final portfolio value: ${portfolio_value:,.2f}")
    
    def _calculate_actual_weights(self, prev_prices, curr_prices, prev_weights):
        """
        Calculate actual weights based on price changes
        
        Parameters:
            prev_prices: Previous period prices
            curr_prices: Current prices
            prev_weights: Previous period weights
            
        Returns:
            ndarray: Current actual weights
        """
        if len(prev_weights) != len(self.tickers):
            return self.target_weights.copy()
        
        # Calculate value changes for each asset
        value_changes = curr_prices / prev_prices
        new_weights = prev_weights * value_changes
        new_weights = new_weights / np.sum(new_weights)  # Normalization
        
        return new_weights
    
    def _should_rebalance(self, date, current_weights):
        """
        Check if rebalancing conditions are triggered
        
        Parameters:
            date: Current date
            current_weights: Current weights
            
        Returns:
            bool: Whether rebalancing is needed
        """
        # Time-based trigger condition
        if self._is_time_trigger(date):
            return True
        
        # Threshold-based trigger condition
        if self._is_threshold_trigger(current_weights):
            return True
        
        return False
    
    def _is_time_trigger(self, date):
        """
        Check time-based trigger condition
        """
        if self.rebalance_interval == 'monthly':
            return date.day == 1  # First day of each month
        elif self.rebalance_interval == 'quarterly':
            return date.day == 1 and date.month % 3 == 1  # First day of each quarter
        elif self.rebalance_interval == 'annually':
            return date.day == 1 and date.month == 1  # First day of each year
        return False
    
    def _is_threshold_trigger(self, current_weights):
        """
        Check threshold-based trigger condition
        """
        deviation = np.abs(current_weights - self.target_weights)
        return np.any(deviation > self.threshold)
    
    def _get_rebalance_reason(self, date, current_weights):
        """
        Get rebalancing reason
        """
        if self._is_time_trigger(date):
            return "Time-based trigger"
        elif self._is_threshold_trigger(current_weights):
            max_deviation = np.max(np.abs(current_weights - self.target_weights))
            return f"Threshold trigger (maximum deviation: {max_deviation:.2%})"
        return "Unknown reason"
    
    def _generate_trades(self, current_prices, current_weights):
        """
        Generate specific buy/sell instructions
        
        Parameters:
            current_prices: Current prices
            current_weights: Current weights
            
        Returns:
            dict: Trade instructions
        """
        if len(current_weights) != len(self.tickers):
            return {}
        
        # Assume total portfolio value is 10 million GBP (for calculating trade amounts)
        total_portfolio_value = 10000000
        
        # Calculate target values and current values
        target_values = total_portfolio_value * self.target_weights
        current_values = total_portfolio_value * current_weights
        
        trades = {}
        
        for i, ticker in enumerate(self.tickers):
            target = target_values[i]
            current = current_values[i]
            delta = target - current
            
            if abs(delta) > 100:  # Minimum trade amount threshold
                if delta > 0:
                    trades[ticker] = ('BUY', delta, current_prices.iloc[i])
                elif delta < 0:
                    trades[ticker] = ('SELL', -delta, current_prices.iloc[i])
        
        return trades
    
    def _execute_trades(self, current_weights, trades):
        """
        Execute trades and update weights
        
        Parameters:
            current_weights: Current weights
            trades: Trade instructions
            
        Returns:
            ndarray: Updated weights
        """
        new_weights = current_weights.copy()
        
        if not trades:
            return new_weights
        
        # Simplified trade execution: directly adjust to target weights
        # In practical applications, this would consider transaction costs, slippage, etc.
        new_weights = self.target_weights.copy()
        
        return new_weights
    
    def _calculate_portfolio_value(self, prices, weights, initial_capital):
        """
        Calculate portfolio value
        
        Parameters:
            prices: Current prices
            weights: Current weights
            initial_capital: Initial capital
            
        Returns:
            float: Portfolio value
        """
        if len(weights) != len(prices):
            return initial_capital
        
        # Assume initial purchase with target weights
        return initial_capital * np.sum(weights * prices / prices)  # Simplified calculation
    
    def generate_report(self):
        """
        Generate backtest report
        """
        if not self.trade_log:
            print("No trade records")
            return
        
        print("\n=== Portfolio Rebalancing Backtest Report ===")
        print(f"Backtest period: {self.portfolio_values[0]['date']} to {self.portfolio_values[-1]['date']}")
        print(f"Initial capital: $10,000,000")
        print(f"Final value: ${self.portfolio_values[-1]['value']:,.2f}")
        
        total_return = (self.portfolio_values[-1]['value'] - 10000000) / 10000000 * 100
        print(f"Total return: {total_return:.2f}%")
        
        print(f"\nRebalancing count: {len(self.trade_log)}")
        print("Trade details:")
        for i, trade in enumerate(self.trade_log, 1):
            print(f"\n{i}. {trade['date']} - {trade['reason']}")
            for ticker, (action, amount, price) in trade['trades'].items():
                print(f"   {ticker}: {action} ${amount:,.2f} (price: ${price:.2f})")
    
    def get_performance_metrics(self):
        """
        Get performance metrics
        """
        if not self.portfolio_values:
            return {}
        
        values = [p['value'] for p in self.portfolio_values]
        dates = [p['date'] for p in self.portfolio_values]
        
        # Calculate return series
        returns = []
        for i in range(1, len(values)):
            ret = (values[i] - values[i-1]) / values[i-1]
            returns.append(ret)
        
        if not returns:
            return {}
        
        returns = np.array(returns)
        
        metrics = {
            'total_return': (values[-1] - values[0]) / values[0],
            'annualized_return': np.mean(returns) * 252,  # Annualized
            'volatility': np.std(returns) * np.sqrt(252),  # Annualized volatility
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(values),
            'trades_count': len(self.trade_log)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, values):
        """
        Calculate maximum drawdown
        """
        if len(values) < 2:
            return 0
        
        peak = values[0]
        max_dd = 0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd