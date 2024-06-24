#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
import pandas as pd
csv_file_path = 'BTC-Hourly.csv'
df_downsampled = pd.read_csv(csv_file_path)

# Reverse the DataFrame
df_downsampled = df_downsampled[::-1]

# Convert the date column to datetime format
df_downsampled['Date'] = pd.to_datetime(df_downsampled['date'])

# Set the Date column as the index
df_downsampled.set_index('Date', inplace=True)
df_downsampled = df_downsampled[df_downsampled.index.year != 2022]
df_downsampled = df_downsampled[df_downsampled.index.year != 2018]
# Downsample the data by selecting every 5th row
df_downsampled = df_downsampled.iloc[::5]
df_downsampled.rename(columns={'close': 'Close'}, inplace=True)
df_downsampled.rename(columns={'low': 'Low'}, inplace=True)
df_downsampled.rename(columns={'high': 'High'}, inplace=True)
# Function to download historical data using yfinance
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to preprocess downloaded data
def preprocess_data(data):
    data['Date'] = data.index
    data.set_index('Date', inplace=True)
    return data

# Function to create a simple moving average (SMA) crossover signal
def generate_sma_signal(data, short_window=40, long_window=100):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    # Create short simple moving average
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

    # Create long simple moving average
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

# Function to calculate returns, maximum drawdown, and manage risk
def calculate_returns_drawdown_risk(df, position_size=0.1, atr_multiplier=1.5):
    df['Returns'] = df['Close'].pct_change()
    df['Cumulative Returns'] = (1 + df['Returns']).cumprod() - 1

    # Calculate drawdown
    df['Roll_Max'] = df['Cumulative Returns'].cummax()
    df['Drawdown'] = df['Cumulative Returns'] - df['Roll_Max']

    # Calculate maximum drawdown
    df['Max_Drawdown'] = -df['Drawdown'].cummin()

    # Calculate Average True Range (ATR)
    df['ATR'] = df['High'] - df['Low']

    # Dynamic position sizing based on ATR
    df['Position_Size'] = position_size * df['ATR'].rolling(window=20).mean() * atr_multiplier

    # Limit position size to 20% of the portfolio
    df['Position_Size'] = np.minimum(df['Position_Size'], 0.2)

    # Calculate risk-adjusted returns
    df['Risk_Adj_Returns'] = df['Returns'] * df['Position_Size']

    # Calculate cumulative risk-adjusted returns
    df['Cumulative_Risk_Adj_Returns'] = (1 + df['Risk_Adj_Returns']).cumprod() - 1

    return df['Max_Drawdown'].iloc[-1], df['Cumulative_Risk_Adj_Returns'].iloc[-1]

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=input_shape))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



# Function to calculate average dip in running trade
def calculate_metrics(data):
    # Check if the necessary columns are present
    if 'combined_signal' not in data.columns or 'signal' not in data.columns:
        raise ValueError("Columns 'combined_signal' and 'signal' are required for metric calculation.")
    
    # Extract relevant columns
    combined_signal = data['combined_signal']
    signal = data['signal']

    # Create a column for daily returns based on signals
    data['daily_returns'] = data['Close'].pct_change()

    # Create a column for daily strategy returns
    data['strategy_returns'] = data['daily_returns'] * combined_signal.shift(1)

    # Calculate metrics
    metrics = {
        'Gross Profit': np.sum(data[data['strategy_returns'] > 0]['strategy_returns']),
        'Gross Loss': np.sum(data[data['strategy_returns'] < 0]['strategy_returns']),
        'Net Profit': np.sum(data['strategy_returns']),
        'Total Closed Trades': np.sum(np.abs(combined_signal.diff() != 0)),
        'Win Rate': np.sum(data['strategy_returns'] > 0) / np.sum(data['strategy_returns'] != 0),
        'Max Drawdown': calculate_max_drawdown(data['Close']),
        'Average Winning Trade (in USDT)': np.mean(data[data['strategy_returns'] > 0]['strategy_returns']),
        'Average Losing Trade (in USDT)': np.mean(data[data['strategy_returns'] < 0]['strategy_returns']),
        'Buy and Hold Return of BTC': (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1,
        'Largest Winning Trade (in USDT)': np.max(data[data['strategy_returns'] > 0]['strategy_returns']),
        'Largest Losing Trade (in USDT)': np.min(data[data['strategy_returns'] < 0]['strategy_returns']),
        'Sharpe Ratio': calculate_sharpe_ratio(data['strategy_returns']),
        'Sortino Ratio': calculate_sortino_ratio(data['strategy_returns']),
        'Average Holding Duration per Trade': calculate_average_holding_duration(data, combined_signal),
        'Max Dip in Running Trade': calculate_max_dip(data['Close'], combined_signal),
        'Average Dip in Running Trade': calculate_average_dip(data['Close'], combined_signal),
    }

    return metrics

# Function to calculate max drawdown
def calculate_max_drawdown(prices):
    roll_max = prices.cummax()
    daily_drawdown = prices/roll_max - 1.0
    max_drawdown = daily_drawdown.cummin()
    return np.abs(max_drawdown.min())

# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(returns):
    annualized_return = np.mean(returns) * 24 * 365  # Assuming hourly data
    annualized_volatility = np.std(returns) * np.sqrt(24 * 365)  # Assuming hourly data
    sharpe_ratio = annualized_return / annualized_volatility
    return sharpe_ratio

# Function to calculate Sortino ratio
def calculate_sortino_ratio(returns):
    downside_returns = returns[returns < 0]
    annualized_return = np.mean(returns) * 24 * 365  # Assuming hourly data
    downside_volatility = np.std(downside_returns) * np.sqrt(24 * 365)  # Assuming hourly data
    sortino_ratio = annualized_return / downside_volatility
    return sortino_ratio

# Function to calculate average holding duration per trade
# Function to calculate average holding duration per trade
def calculate_average_holding_duration(data, signals):
    trade_entries = data['Close'][signals.diff() != 0]
    trade_exits = data['Close'][signals.shift(1).diff() != 0]
    
    # Ensure the length of trade_entries and trade_exits is the same
    min_length = min(len(trade_entries), len(trade_exits))
    trade_entries = trade_entries[:min_length]
    trade_exits = trade_exits[:min_length]

    holding_durations = (trade_exits.index.values - trade_entries.index.values).astype('timedelta64[s]').astype(int) / 3600  # Convert to hours
    average_holding_duration = holding_durations.mean()
    return average_holding_duration

# Function to calculate max dip in running trade
def calculate_max_dip(prices, signals):
    max_dip = 0
    current_dip = 0
    for i in range(1, len(prices)):
        if signals.iloc[i] == 0 and signals.iloc[i - 1] == 1:
            current_dip = 0
        elif signals.iloc[i] == 1:
            current_dip += prices.iloc[i - 1] - prices.iloc[i]
            max_dip = max(max_dip, current_dip)
    return max_dip

# Function to calculate average dip in running trade
def calculate_average_dip(prices, signals):
    dip_values = []
    current_dip = 0
    for i in range(1, len(prices)):
        if signals.iloc[i] == 0 and signals.iloc[i - 1] == 1:
            current_dip = 0
        elif signals.iloc[i] == 1:
            current_dip += prices.iloc[i - 1] - prices.iloc[i]
            dip_values.append(current_dip)
    return np.mean(dip_values)

def combined_strategy_backtest(data):
    # SMA Signal
    sma_signals = generate_sma_signal(data)

    # LSTM Model
    lstm_model = create_lstm_model((60, 1))

    # Combine SMA and LSTM signals
    signals = sma_signals.copy()
    signals['lstm_signal'] = 0.0

    for i in range(60, len(data)):
        x_train = data['Close'].iloc[i-60:i].values.reshape(1, -1, 1)
        lstm_prediction = lstm_model.predict(x_train)[0, 0]

        if lstm_prediction > data['Close'].iloc[i]:
            signals['lstm_signal'].iloc[i] = 1.0
        else:
            signals['lstm_signal'].iloc[i] = 0.0

    # Add 'lstm_signal' to the data DataFrame
    data['lstm_signal'] = signals['lstm_signal']

    # Create the 'combined_signal' column
    signals['combined_signal'] = signals['signal'] + signals['lstm_signal']

    # ... (unchanged)
    print("Signals DataFrame:")
    print(signals)

    # Apply risk management and calculate returns
    # (Add your risk management and position sizing logic here)

    # Add 'combined_signal' and 'signal' columns to the data DataFrame
    data['combined_signal'] = signals['combined_signal']
    data['signal'] = signals['signal']

    # Print relevant columns for debugging
    print("Columns after combining signals:")
    print(data[['signal', 'lstm_signal', 'combined_signal']])

    # Check if the necessary columns are present
    if 'combined_signal' not in data.columns or 'signal' not in data.columns:
        raise ValueError("Columns 'combined_signal' and 'signal' are required for metric calculation.")

    # Calculate maximum drawdown and cumulative returns using the original data
    max_drawdown, cumulative_returns = calculate_returns_drawdown_risk(data)

    print(f"Max Drawdown: {max_drawdown}, Cumulative Returns: {cumulative_returns}")

    # Calculate additional metrics using the original data
    metrics = calculate_metrics(data)

    print("Metrics:")
    print(metrics)

    # Extract relevant columns from the signals DataFrame
    extracted_signals = signals[['signal', 'lstm_signal', 'combined_signal']]

    return extracted_signals, max_drawdown, cumulative_returns, metrics




# Download BTC/USD data for 2023
btc_data_2023 = download_data("BTC-USD", start_date="2023-01-01", end_date="2023-12-31")

# Preprocess BTC/USD data for 2023
btc_data_2023 = preprocess_data(btc_data_2023)

# Backtest combined strategy for each year in 2023
drawdown_by_year_2023 = {}
returns_by_year_2023 = {}
metrics_by_year_2023 = {}

for year in btc_data_2023.index.year.unique():
    year_data = btc_data_2023[btc_data_2023.index.year == year].copy()
    combined_strategy_backtest(year_data)
    drawdown, returns = calculate_returns_drawdown_risk(year_data)
    metrics = calculate_metrics(year_data)

    drawdown_by_year_2023[year] = drawdown
    returns_by_year_2023[year] = returns
    metrics_by_year_2023[year] = metrics

    print(f'Year: {year}, Max Drawdown: {drawdown:.4f}, Cumulative Returns: {returns:.4f}')
    print(f'Metrics for {year}: {metrics}')

# Backtest combined strategy for each year in earlier data
# You can use the existing df_downsampled DataFrame or load earlier data if needed
# Backtest combined strategy for each year in earlier data
# You can use the existing df DataFrame or load earlier data if needed
# Backtest combined strategy for each year in earlier data
# Use the correct DataFrame (replace df with df_downsampled)
drawdown_by_year_earlier = {}
returns_by_year_earlier = {}
metrics_by_year_earlier = {}

for year in df_downsampled.index.year.unique():
    year_data = df_downsampled[df_downsampled.index.year == year].copy()
    combined_strategy_backtest(year_data)
    drawdown, returns = calculate_returns_drawdown_risk(year_data)
    metrics = calculate_metrics(year_data)

    drawdown_by_year_earlier[year] = drawdown
    returns_by_year_earlier[year] = returns
    metrics_by_year_earlier[year] = metrics

    print(f'Year: {year}, Max Drawdown: {drawdown:.4f}, Cumulative Returns: {returns:.4f}')
    print(f'Metrics for {year}: {metrics}')


# Overall performance for 2023
print(f'Overall Performance - Max Drawdown (2023): {max(drawdown_by_year_2023.values()):.4f}, Total Cumulative Returns (2023): {sum(returns_by_year_2023.values()):.4f}')

# Overall performance for earlier data
print(f'Overall Performance - Max Drawdown (Earlier): {max(drawdown_by_year_earlier.values()):.4f}, Total Cumulative Returns (Earlier): {sum(returns_by_year_earlier.values()):.4f}')

