#!/usr/bin/env python3
"""
Complete AI Trading System
Integrates: Data Collection + Ensemble Engine + Portfolio Allocation + Live Trading
"""

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import savgol_filter
import math
import warnings
warnings.filterwarnings('ignore')

# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import alpaca_trade_api as tradeapi

# =============================================================================
# CONFIGURATION
# =============================================================================

# Alpaca API credentials
API_KEY = "PK6TOP0STK4VO996JZ5Q"
API_SECRET = "PiNdt7ccSjzydlk35W5GOR5q9Ng1tAc6VEiSADeY"
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading

# Trading configuration
TRADING_CONFIG = {
    'symbols': ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN", "NFLX"],
    'data_days_back': 30,  # Days of historical data for training
    'prediction_confidence_threshold': 0.65,  # Minimum confidence to trade
    'buy_threshold': 0.02,  # 2% expected return to buy
    'sell_threshold': -0.015,  # -1.5% expected return to sell
    'stop_loss_pct': 0.03,  # 3% stop loss
    'take_profit_pct': 0.08,  # 8% take profit
    'max_positions': 6,  # Maximum concurrent positions
    'rebalance_frequency_hours': 24,  # Rebalance every 24 hours
}

print("🚀 Complete AI Trading System Starting...")
print(f"Symbols: {TRADING_CONFIG['symbols']}")
print(f"Mode: Paper Trading (Safe)")

# =============================================================================
# DATA COLLECTION MODULE
# =============================================================================

class DataCollector:
    def __init__(self, api_key, api_secret):
        self.client = StockHistoricalDataClient(api_key, api_secret)
        
    def get_historical_data(self, symbols, days_back=30):
        """Collect historical data for multiple symbols"""
        print(f"\n📊 Collecting historical data for {len(symbols)} symbols...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back * 2)  # Extra buffer for weekends
        
        all_data = {}
        
        for symbol in symbols:
            try:
                print(f"   Fetching {symbol}...", end=" ")
                
                bars_request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    start=start_date,
                    end=end_date,
                    timeframe=TimeFrame.Minute
                )
                
                bars = self.client.get_stock_bars(bars_request)
                
                if hasattr(bars, 'data') and symbol in bars.data:
                    symbol_bars = bars.data[symbol]
                    
                    # Convert to DataFrame
                    bars_data = []
                    for bar in symbol_bars:
                        bars_data.append({
                            'timestamp': bar.timestamp,
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume
                        })
                    
                    df = pd.DataFrame(bars_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    
                    # Resample to 5-minute intervals
                    df_resampled = df.resample('5T').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    
                    # Create ensemble engine format
                    final_data = pd.DataFrame({
                        'Bid_Price': df_resampled['low'],
                        'Ask_Price': df_resampled['high'],
                        'Price': df_resampled['close']
                    })
                    
                    all_data[symbol] = final_data
                    print(f"✅ {len(final_data)} points")
                    
                else:
                    print(f"❌ No data")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:30]}")
                
        print(f"📊 Data collection complete: {len(all_data)}/{len(symbols)} symbols")
        return all_data

# =============================================================================
# ENSEMBLE ENGINE MODULE
# =============================================================================

class EnsembleEngineNet(nn.Module):
    def __init__(self, input_dim, num_layers=3, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extraction
        self.shared_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Cross-attention for price relationships
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Prediction heads
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.bid_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.ask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # [price, bid, ask] confidence
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Feature extraction
        out = x
        for layer in self.shared_layers:
            out = torch.relu(layer(out))
        
        # Cross-attention
        out_expanded = out.unsqueeze(1)
        attended, attention_weights = self.cross_attention(out_expanded, out_expanded, out_expanded)
        out = attended.squeeze(1)
        
        # Predictions
        price_pred = self.price_head(out)
        bid_pred = self.bid_head(out)
        ask_pred = self.ask_head(out)
        confidence = self.confidence_head(out)
        
        return price_pred.squeeze(-1), bid_pred.squeeze(-1), ask_pred.squeeze(-1), confidence, attention_weights

class EnsembleEngine:
    def __init__(self):
        self.models = {}  # Store trained models for each symbol
        self.feature_columns = [
            "Price_Lag1", "Price_Lag2", "Price_Return", "Bid_Lag1", "Ask_Lag1", 
            "Spread_Lag1", "Fourier_Trend_Lag1", "Fourier_Residual_Lag1", 
            "Fourier_Magnitude_1", "Fourier_Magnitude_2"
        ]
        
    def create_features(self, df):
        """Create features for ensemble engine"""
        # Basic lag features
        df['Price_Lag1'] = df['Price'].shift(1)
        df['Price_Lag2'] = df['Price'].shift(2)
        df['Price_Return'] = df['Price'].pct_change()
        df['Bid_Lag1'] = df['Bid_Price'].shift(1)
        df['Ask_Lag1'] = df['Ask_Price'].shift(1)
        df['Spread_Lag1'] = (df['Ask_Price'] - df['Bid_Price']).shift(1)
        
        # Fourier features (simplified)
        df['Fourier_Trend_Lag1'] = df['Price'].rolling(window=20).mean().shift(1)
        df['Fourier_Residual_Lag1'] = (df['Price'] - df['Price'].rolling(window=20).mean()).shift(1)
        df['Fourier_Magnitude_1'] = df['Price'].rolling(window=10).std()
        df['Fourier_Magnitude_2'] = df['Price'].rolling(window=5).std()
        
        return df.dropna().reset_index(drop=True)
    
    def train_model(self, symbol, data):
        """Train ensemble model for a specific symbol"""
        print(f"   🧠 Training AI model for {symbol}...", end=" ")
        
        # Create features
        df = self.create_features(data.copy())
        
        if len(df) < 50:
            print("❌ Insufficient data")
            return False
        
        # Prepare training data (use 80% for training)
        train_size = int(len(df) * 0.8)
        
        features = df[self.feature_columns].values.astype(np.float32)
        prices = df['Price'].values.astype(np.float32)
        bids = df['Bid_Price'].values.astype(np.float32)
        asks = df['Ask_Price'].values.astype(np.float32)
        
        X_train = torch.tensor(features[:train_size-1], dtype=torch.float32)
        y_price = torch.tensor(prices[1:train_size], dtype=torch.float32)
        y_bid = torch.tensor(bids[1:train_size], dtype=torch.float32)
        y_ask = torch.tensor(asks[1:train_size], dtype=torch.float32)
        
        # Create and train model
        model = EnsembleEngineNet(input_dim=len(self.feature_columns))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(200):  # Quick training
            optimizer.zero_grad()
            
            pred_price, pred_bid, pred_ask, confidence, _ = model(X_train)
            
            loss_price = criterion(pred_price, y_price)
            loss_bid = criterion(pred_bid, y_bid)
            loss_ask = criterion(pred_ask, y_ask)
            
            total_loss = loss_price + loss_bid + loss_ask
            total_loss.backward()
            optimizer.step()
        
        self.models[symbol] = {
            'model': model,
            'features': df,
            'train_size': train_size
        }
        
        print(f"✅ Trained ({len(df)} points)")
        return True
    
    def predict(self, symbol, current_data=None):
        """Generate prediction for a symbol"""
        if symbol not in self.models:
            return None
        
        model_info = self.models[symbol]
        model = model_info['model']
        df = model_info['features']
        
        # Use latest available data
        latest_features = df[self.feature_columns].iloc[-1].values.astype(np.float32)
        
        model.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(latest_features, dtype=torch.float32).unsqueeze(0)
            pred_price, pred_bid, pred_ask, confidence, _ = model(features_tensor)
            
            current_price = df['Price'].iloc[-1]
            predicted_price = pred_price.item()
            price_confidence = confidence[0, 0].item()
            
            expected_return = (predicted_price - current_price) / current_price
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'expected_return': expected_return,
                'confidence': price_confidence,
                'signal_strength': price_confidence * abs(expected_return)
            }

# =============================================================================
# PORTFOLIO ALLOCATION MODULE
# =============================================================================

class PortfolioAllocator:
    def __init__(self):
        pass
    
    def calculate_risk_score(self, symbol, prediction_data, historical_data):
        """Calculate risk score for a symbol"""
        if prediction_data is None:
            return 50.0  # Default medium risk
        
        # Base risk on prediction confidence and volatility
        confidence = prediction_data['confidence']
        
        # Calculate historical volatility
        if len(historical_data) > 20:
            returns = historical_data['Price'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            volatility_risk = min(volatility * 100, 80)  # Cap at 80%
        else:
            volatility_risk = 30  # Default
        
        # Combine confidence and volatility for risk score
        confidence_risk = (1 - confidence) * 100
        risk_score = (confidence_risk + volatility_risk) / 2
        
        return min(max(risk_score, 5), 95)  # Keep between 5-95%
    
    def allocate_portfolio(self, balance, symbols, predictions, historical_data):
        """Allocate portfolio using bell curve distribution"""
        print(f"\n💰 Allocating ${balance:,.2f} across {len(symbols)} symbols...")
        
        # Calculate risk scores for each symbol
        stocks_data = []
        for symbol in symbols:
            prediction = predictions.get(symbol)
            hist_data = historical_data.get(symbol)
            
            if prediction and hist_data is not None:
                risk = self.calculate_risk_score(symbol, prediction, hist_data)
                stocks_data.append({
                    'ticker': symbol,
                    'risk': risk,
                    'prediction': prediction
                })
        
        if not stocks_data:
            print("❌ No valid symbols for allocation")
            return {}
        
        # Bell curve allocation (from master_allocater.py)
        allocations = self.bellcurve_allocation({
            'balance': balance,
            'stocks': stocks_data
        })
        
        return allocations
    
    def bellcurve_allocation(self, data):
        """Bell curve allocation algorithm"""
        balance = data["balance"]
        stocks = data["stocks"]
        raw_allocs = []
        tickers = []
        
        m = 1000  # Number of steps for binomial/Gaussian
        
        for stock in stocks:
            ticker = stock["ticker"]
            risk = stock["risk"]
            left_pct = risk  # risk is percent chance to go down
            q = left_pct / 100
            p = 1 - q
            
            # Raw allocation based on success probability
            alloc = balance * (p ** 2)
            raw_allocs.append(alloc)
            tickers.append(ticker)
        
        # Scale allocations to fit balance
        sum_raw = sum(raw_allocs)
        if sum_raw > balance:
            squares = [a ** 2 for a in raw_allocs]
            sum_squares = sum(squares)
            scaled_allocs = [math.floor(balance * (s / sum_squares)) for s in squares]
            
            # Adjust last allocation
            diff = balance - sum(scaled_allocs)
            if diff > 0:
                scaled_allocs[-1] += diff
        else:
            scaled_allocs = [math.floor(a) for a in raw_allocs]
            diff = balance - sum(scaled_allocs)
            if diff > 0:
                scaled_allocs[-1] += diff
        
        allocation_dict = {tickers[i]: scaled_allocs[i] for i in range(len(tickers))}
        remainder = balance - sum(scaled_allocs)
        allocation_dict["cash_remainder"] = remainder
        
        return allocation_dict

# =============================================================================
# TRADING EXECUTION MODULE
# =============================================================================

class TradingExecutor:
    def __init__(self, api_key, secret_key, base_url):
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.positions = {}
        
    def get_account_info(self):
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value)
            }
        except Exception as e:
            print(f"❌ Error getting account info: {e}")
            return None
    
    def get_current_positions(self):
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            current_positions = {}
            
            for pos in positions:
                current_positions[pos.symbol] = {
                    'quantity': int(pos.qty),
                    'market_value': float(pos.market_value),
                    'unrealized_pnl': float(pos.unrealized_pnl),
                    'avg_cost': float(pos.avg_cost)
                }
            
            return current_positions
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            return {}
    
    def execute_trade(self, symbol, action, quantity, current_price):
        """Execute a trade"""
        try:
            if action == 'BUY' and quantity > 0:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                
                # Set stop loss and take profit
                stop_price = current_price * (1 - TRADING_CONFIG['stop_loss_pct'])
                take_profit_price = current_price * (1 + TRADING_CONFIG['take_profit_pct'])
                
                # Submit protective orders
                self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='stop',
                    stop_price=stop_price,
                    time_in_force='gtc'
                )
                
                self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='limit',
                    limit_price=take_profit_price,
                    time_in_force='gtc'
                )
                
                print(f"   ✅ BUY {quantity} shares of {symbol} at ${current_price:.2f}")
                return True
                
            elif action == 'SELL' and quantity > 0:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                
                print(f"   ✅ SELL {quantity} shares of {symbol} at ${current_price:.2f}")
                return True
                
        except Exception as e:
            print(f"   ❌ Trade failed for {symbol}: {e}")
            return False
        
        return False

# =============================================================================
# MAIN TRADING SYSTEM
# =============================================================================

class CompleteTradingSystem:
    def __init__(self):
        self.data_collector = DataCollector(API_KEY, API_SECRET)
        self.ensemble_engine = EnsembleEngine()
        self.portfolio_allocator = PortfolioAllocator()
        self.trading_executor = TradingExecutor(API_KEY, API_SECRET, BASE_URL)
        
        self.last_rebalance = None
        self.historical_data = {}
        
    def initialize_system(self):
        """Initialize the complete trading system"""
        print("\n🔧 Initializing Complete AI Trading System...")
        
        # Get account info
        account_info = self.trading_executor.get_account_info()
        if not account_info:
            print("❌ Failed to connect to Alpaca account")
            return False
        
        print(f"✅ Connected to Alpaca account")
        print(f"   Portfolio Value: ${account_info['portfolio_value']:,.2f}")
        print(f"   Buying Power: ${account_info['buying_power']:,.2f}")
        
        # Collect historical data
        self.historical_data = self.data_collector.get_historical_data(
            TRADING_CONFIG['symbols'], 
            TRADING_CONFIG['data_days_back']
        )
        
        if not self.historical_data:
            print("❌ Failed to collect historical data")
            return False
        
        # Train ensemble models
        print(f"\n🧠 Training AI models for {len(self.historical_data)} symbols...")
        trained_count = 0
        
        for symbol, data in self.historical_data.items():
            if self.ensemble_engine.train_model(symbol, data):
                trained_count += 1
        
        print(f"✅ Trained {trained_count}/{len(self.historical_data)} AI models")
        
        if trained_count == 0:
            print("❌ No models trained successfully")
            return False
        
        print("🚀 System initialization complete!")
        return True
    
    def generate_trading_signals(self):
        """Generate trading signals for all symbols"""
        print(f"\n📊 Generating AI trading signals...")
        
        predictions = {}
        signals = []
        
        for symbol in TRADING_CONFIG['symbols']:
            prediction = self.ensemble_engine.predict(symbol)
            if prediction:
                predictions[symbol] = prediction
                
                # Determine trading signal
                confidence = prediction['confidence']
                expected_return = prediction['expected_return']
                
                if (confidence >= TRADING_CONFIG['prediction_confidence_threshold'] and 
                    expected_return >= TRADING_CONFIG['buy_threshold']):
                    signal = 'BUY'
                elif (confidence >= TRADING_CONFIG['prediction_confidence_threshold'] and 
                      expected_return <= TRADING_CONFIG['sell_threshold']):
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                
                signals.append({
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'expected_return': expected_return,
                    'current_price': prediction['current_price'],
                    'predicted_price': prediction['predicted_price']
                })
        
        # Sort by signal strength
        signals.sort(key=lambda x: x['confidence'] * abs(x['expected_return']), reverse=True)
        
        print(f"📊 Generated {len(signals)} trading signals")
        for signal in signals[:5]:  # Show top 5
            print(f"   {signal['symbol']}: {signal['signal']} | "
                  f"Confidence: {signal['confidence']:.1%} | "
                  f"Expected Return: {signal['expected_return']:.1%}")
        
        return predictions, signals
    
    def rebalance_portfolio(self):
        """Rebalance portfolio based on AI predictions"""
        print(f"\n⚖️ Rebalancing Portfolio...")
        
        # Get current account state
        account_info = self.trading_executor.get_account_info()
        current_positions = self.trading_executor.get_current_positions()
        
        if not account_info:
            print("❌ Cannot get account info for rebalancing")
            return False
        
        # Generate predictions and signals
        predictions, signals = self.generate_trading_signals()
        
        # Calculate optimal allocation
        available_cash = account_info['buying_power']
        allocations = self.portfolio_allocator.allocate_portfolio(
            available_cash, 
            TRADING_CONFIG['symbols'], 
            predictions, 
            self.historical_data
        )
        
        print(f"\n💰 Optimal Allocation (${available_cash:,.2f} available):")
        for symbol, amount in allocations.items():
            if symbol != 'cash_remainder' and amount > 0:
                print(f"   {symbol}: ${amount:,.2f}")
        
        # Execute rebalancing trades
        trades_executed = 0
        
        for signal in signals:
            symbol = signal['symbol']
            current_price = signal['current_price']
            target_allocation = allocations.get(symbol, 0)
            
            if target_allocation <= 0:
                continue
            
            # Calculate target shares
            target_shares = int(target_allocation / current_price)
            current_shares = current_positions.get(symbol, {}).get('quantity', 0)
            
            # Determine trade needed
            shares_diff = target_shares - current_shares
            
            if abs(shares_diff) > 0:
                if shares_diff > 0:
                    # Need to buy more
                    if (signal['signal'] == 'BUY' and 
                        len(current_positions) < TRADING_CONFIG['max_positions']):
                        if self.trading_executor.execute_trade(symbol, 'BUY', shares_diff, current_price):
                            trades_executed += 1
                elif shares_diff < 0:
                    # Need to sell some
                    if signal['signal'] == 'SELL':
                        if self.trading_executor.execute_trade(symbol, 'SELL', abs(shares_diff), current_price):
                            trades_executed += 1
        
        print(f"✅ Rebalancing complete: {trades_executed} trades executed")
        self.last_rebalance = datetime.now()
        return True
    
    def run_trading_session(self, duration_hours=24):
        """Run the complete trading system"""
        print(f"\n🚀 Starting {duration_hours}-hour AI Trading Session...")
        
        if not self.initialize_system():
            print("❌ System initialization failed")
            return
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Initial rebalance
        self.rebalance_portfolio()
        
        while datetime.now() < end_time:
            try:
                # Check if market is open
                clock = self.trading_executor.api.get_clock()
                if not clock.is_open:
                    print("💤 Market closed, waiting...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # Check if rebalance is needed
                if (self.last_rebalance is None or 
                    (datetime.now() - self.last_rebalance).total_seconds() > 
                    TRADING_CONFIG['rebalance_frequency_hours'] * 3600):
                    
                    # Update historical data
                    print("\n🔄 Updating market data...")
                    new_data = self.data_collector.get_historical_data(
                        TRADING_CONFIG['symbols'], 
                        days_back=7  # Quick update
                    )
                    
                    # Update models with new data
                    for symbol, data in new_data.items():
                        if len(data) > 50:
                            self.ensemble_engine.train_model(symbol, data)
                            self.historical_data[symbol] = data
                    
                    # Rebalance portfolio
                    self.rebalance_portfolio()
                
                # Generate current signals for monitoring
                predictions, signals = self.generate_trading_signals()
                
                # Display current status
                account_info = self.trading_executor.get_account_info()
                positions = self.trading_executor.get_current_positions()
                
                print(f"\n📈 Current Status ({datetime.now().strftime('%H:%M:%S')})")
                print(f"   Portfolio Value: ${account_info['portfolio_value']:,.2f}")
                print(f"   Active Positions: {len(positions)}")
                
                # Show top signals
                buy_signals = [s for s in signals if s['signal'] == 'BUY']
                sell_signals = [s for s in signals if s['signal'] == 'SELL']
                
                if buy_signals:
                    print(f"   🟢 Top Buy Signal: {buy_signals[0]['symbol']} "
                          f"({buy_signals[0]['expected_return']:.1%} expected)")
                
                if sell_signals:
                    print(f"   🔴 Top Sell Signal: {sell_signals[0]['symbol']} "
                          f"({sell_signals[0]['expected_return']:.1%} expected)")
                
                # Wait before next check
                time.sleep(300)  # Check every 5 minutes
                
            except KeyboardInterrupt:
                print("\n⏹️ Trading session stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in trading session: {e}")
                time.sleep(60)  # Wait 1 minute on error
        
        print(f"\n🏁 Trading session completed!")
        
        # Final status report
        final_account = self.trading_executor.get_account_info()
        final_positions = self.trading_executor.get_current_positions()
        
        print(f"\n📊 Final Report:")
        print(f"   Final Portfolio Value: ${final_account['portfolio_value']:,.2f}")
        print(f"   Final Positions: {len(final_positions)}")
        
        for symbol, pos in final_positions.items():
            pnl_pct = (pos['unrealized_pnl'] / (pos['avg_cost'] * pos['quantity'])) * 100
            print(f"     {symbol}: {pos['quantity']} shares | "
                  f"P&L: ${pos['unrealized_pnl']:.2f} ({pnl_pct:.1f}%)")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the complete trading system"""
    print("=" * 80)
    print("🤖 COMPLETE AI TRADING SYSTEM")
    print("=" * 80)
    print("Features:")
    print("  ✅ Live data collection from Alpaca")
    print("  ✅ Advanced AI price prediction")
    print("  ✅ Risk-based portfolio allocation")
    print("  ✅ Automated trade execution")
    print("  ✅ Paper trading (safe mode)")
    print("=" * 80)
    
    # Create and run the complete system
    trading_system = CompleteTradingSystem()
    
    try:
        # Run for 24 hours (or until stopped)
        trading_system.run_trading_session(duration_hours=24)
    except KeyboardInterrupt:
        print("\n👋 System shutdown requested")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    
    print("\n🎯 Complete AI Trading System finished!")

if __name__ == "__main__":
    main() 