import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import datetime

# --- Configuration (must match train_model.py) ---
ETF_TICKERS = [
    "SPY", "IVV", "VOO", "QQQ", "DIA", 
    "IWM", "VUG", "BND", "GLD", "TLT",
    "XLK", "XLF", "XLE", "XLU", "XLV",
    "XLI", "XLY", "XLP", "XLB", "XLC"
]
START_DATE = "2024-01-01" # Using a more recent period for testing, unseen by training
END_DATE = "2024-12-31" 
SEQUENCE_LENGTH = 60
PREDICTION_DAYS = 1

# --- Neural Network Model (must match train_model.py) ---
class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StockPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  
        
        out = self.fc(out[:, -1, :]) 
        return out

# --- Data Fetching and Feature Calculation (must match train_model.py) ---
def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    processed_data = {}

    for ticker in tickers:
        ticker_df = pd.DataFrame()
        required_metrics = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for metric in required_metrics:
            if (metric, ticker) in data.columns:
                ticker_df[metric] = data[(metric, ticker)]
        
        if not ticker_df.empty:
            ticker_df.dropna(inplace=True)
            if not ticker_df.empty:
                processed_data[ticker] = ticker_df
            else:
                print(f"Warning: {ticker} has no complete data after dropping NaNs in fetch_data.")
        else:
            print(f"Warning: No data found for {ticker}.")
            
    return processed_data

def calculate_features(df):
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        return pd.DataFrame()

    df['Daily_Return'] = df['Close'].pct_change()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Volume_Change'] = df['Volume'].pct_change()

    df['Volatility'] = df['Daily_Return'].rolling(window=14).std()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['Close'].rolling(window=20).std() * 2)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    selected_features = [
        'Close', 'Volume', 'Daily_Return',
        'SMA_5', 'SMA_10', 'SMA_20',
        'MACD', 'Signal_Line', 'RSI', 'Volume_Change',
        'Volatility', 'ATR', 'BB_Middle', 'BB_Upper', 'BB_Lower'
    ]
    final_features = [f for f in selected_features if f in df.columns]
    return df[final_features]

def create_sequences(data_df, sequence_length, prediction_days):
    X, y = [], []
    
    if len(data_df) < sequence_length + prediction_days:
        return np.array([]), np.array([])

    for i in range(len(data_df) - sequence_length - prediction_days + 1):
        features = data_df.iloc[i : i + sequence_length].values
        
        target_price_today = data_df['Close'].iloc[i + sequence_length - 1]
        target_price_future = data_df['Close'].iloc[i + sequence_length + prediction_days - 1]
        
        if target_price_today == 0 or np.isnan(target_price_today):
            continue
            
        price_change_percent = (target_price_future - target_price_today) / target_price_today
        
        if price_change_percent < -0.001:
            target = 0 # Down
        elif price_change_percent > 0.001:
            target = 2 # Up
        else:
            target = 1 # Hold
        
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y)

def test_model_accuracy():
    print("Loading model and scalers...")
    model_path = "stock_predictor_model.pth"
    scalers_path = "etf_scalers.pkl"

    try:
        scalers = joblib.load(scalers_path)
    except FileNotFoundError:
        print(f"Error: Scalers file '{scalers_path}' not found. Please ensure train_model.py has been run successfully.")
        return

    if not scalers:
        print("Error: No scalers loaded. Cannot determine input dimension.")
        return
    
    sample_scaler = list(scalers.values())[0]
    input_dim = sample_scaler.n_features_in_

    hidden_dim = 100
    num_layers = 2
    output_dim = 3

    model = StockPredictor(input_dim, hidden_dim, num_layers, output_dim)
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please ensure train_model.py has been run successfully.")
        return
    
    print("Model and scalers loaded.")

    X_test_combined, y_test_combined = [], []
    
    # Fetch historical data for testing (different period than training)
    print(f"Fetching historical data for testing from {START_DATE} to {END_DATE}...")
    etf_raw_data = fetch_data(ETF_TICKERS, START_DATE, END_DATE)

    for ticker, data_df in etf_raw_data.items():
        if data_df.empty:
            print(f"Skipping {ticker} due to empty raw data.")
            continue
        
        print(f"Processing {ticker} for testing (raw data length: {len(data_df)})...")
        features_df = calculate_features(data_df.copy())
        
        if features_df.empty:
            print(f"Skipping {ticker} due to insufficient data for feature calculation.")
            continue
        
        if ticker not in scalers:
            print(f"Scaler for {ticker} not found for testing. Skipping.")
            continue
        scaler = scalers[ticker]

        # Use the *fitted* scaler to transform new data
        scaled_features = scaler.transform(features_df)
        scaled_features_df = pd.DataFrame(scaled_features, columns=features_df.columns, index=features_df.index)
        
        X_etf_test, y_etf_test = create_sequences(scaled_features_df, SEQUENCE_LENGTH, PREDICTION_DAYS)
        
        if X_etf_test.size > 0:
            X_test_combined.append(X_etf_test)
            y_test_combined.append(y_etf_test)
            
    if not X_test_combined:
        print("No test sequences created. Exiting.")
        return

    X_test_combined = np.vstack(X_test_combined)
    y_test_combined = np.hstack(y_test_combined)

    X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_combined, dtype=torch.long)

    print("\nEvaluating model on historical test data...")
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
        print("\n--- Classification Report ---")
        target_names = ["Down", "Hold", "Up"]
        print(classification_report(y_test_tensor.numpy(), predicted.numpy(), target_names=target_names))
        
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(y_test_tensor.numpy(), predicted.numpy()))
        
        accuracy = (predicted == y_test_tensor).float().mean()
        print(f"\nOverall Test Accuracy: {accuracy.item():.4f}")


if __name__ == "__main__":
    test_model_accuracy()
