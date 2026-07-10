import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import datetime

# --- Configuration (must match train_model.py) ---
ETF_TICKERS = [
    "SPY", "IVV", "VOO", "QQQ", "DIA", 
    "IWM", "VUG", "BND", "GLD", "TLT",
    "XLK", "XLF", "XLE", "XLU", "XLV",
    "XLI", "XLY", "XLP", "XLB", "XLC"
]
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
def fetch_data_single_ticker(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    # yfinance returns a MultiIndex DataFrame even for single ticker if multiple columns are present.
    # Flatten the column names here.
    if isinstance(data.columns, pd.MultiIndex):
        # Assuming the structure is ('Metric', 'Ticker')
        data.columns = data.columns.get_level_values(0)
    
    return data

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

def predict_next_day():
    print("Loading model and scalers...")
    model_path = "stock_predictor_model.pth"
    scalers_path = "etf_scalers.pkl"

    # Load scalers
    try:
        scalers = joblib.load(scalers_path)
    except FileNotFoundError:
        print(f"Error: Scalers file '{scalers_path}' not found. Please ensure train_model.py has been run successfully.")
        return

    # Determine input_dim from one of the scalers
    if not scalers:
        print("Error: No scalers loaded. Cannot determine input dimension.")
        return
    
    # Get a sample scaler to determine the number of features
    sample_scaler = list(scalers.values())[0]
    input_dim = sample_scaler.n_features_in_
    # Note: n_features_in_ is the number of features the scaler was fit on, not the input_dim for LSTM directly
    # Input_dim for LSTM is the number of features PER DAY, which is what scaler.n_features_in_ represents.

    hidden_dim = 100 # Must match the trained model
    num_layers = 2 # Must match the trained model
    output_dim = 3 # For (down, hold, up) classification

    model = StockPredictor(input_dim, hidden_dim, num_layers, output_dim)
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please ensure train_model.py has been run successfully.")
        return
    
    print("Model and scalers loaded.")

    predictions = {}
    today = datetime.date.today()
    # Fetch data up to today for SEQUENCE_LENGTH + max_feature_window (e.g. 26 for EMA)
    # A safe buffer would be around 100 days to ensure enough data for feature calculation
    fetch_start_date = today - datetime.timedelta(days=SEQUENCE_LENGTH + 100) 

    for ticker in ETF_TICKERS:
        print(f"Fetching recent data for {ticker}...")
        raw_data = fetch_data_single_ticker(ticker, fetch_start_date, today)
        
        if raw_data.empty:
            print(f"  No recent data for {ticker}. Skipping.")
            continue

        features_df = calculate_features(raw_data)
        
        if features_df.empty:
            print(f"  Insufficient data to calculate features for {ticker}. Skipping.")
            continue
        
        if ticker not in scalers:
            print(f"  Scaler for {ticker} not found. Skipping.")
            continue
        scaler = scalers[ticker]

        # Get the last SEQUENCE_LENGTH entries for prediction
        if len(features_df) < SEQUENCE_LENGTH:
            print(f"  Not enough features data for {ticker} ({len(features_df)} < {SEQUENCE_LENGTH}). Skipping.")
            continue
        
        recent_sequence = features_df.tail(SEQUENCE_LENGTH)
        scaled_sequence = scaler.transform(recent_sequence)
        
        # Convert to tensor and add batch dimension
        X_predict = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0) # (1, SEQUENCE_LENGTH, input_dim)

        with torch.no_grad():
            output = model(X_predict)
            probabilities = torch.softmax(output, dim=1)
            _, predicted_class = torch.max(output, 1)

            # Map predicted class to readable label
            class_map = {0: "Down", 1: "Hold", 2: "Up"}
            prediction_label = class_map[predicted_class.item()]
            confidence = probabilities[0, predicted_class.item()].item()

            predictions[ticker] = {
                "label": prediction_label,
                "confidence": confidence,
                "probabilities": probabilities[0].tolist(),
                "last_price": raw_data['Close'].iloc[-1]
            }
            print(f"  {ticker} prediction: {prediction_label} (Confidence: {confidence:.2f})")
    
    if not predictions:
        print("No predictions could be made for any ETF.")
        return

    # --- ETF Selection Logic ---
    print("\n--- Selecting an ETF ---")
    
    # Prioritize 'Up' predictions with higher confidence
    up_candidates = [
        (ticker, pred["confidence"], pred["last_price"]) 
        for ticker, pred in predictions.items() if pred["label"] == "Up"
    ]
    
    if up_candidates:
        # Sort by confidence (descending)
        up_candidates.sort(key=lambda x: x[1], reverse=True)
        best_choice_ticker, best_choice_confidence, best_choice_price = up_candidates[0]
        prediction_action = "Buy"
        # Predict a price range (simple heuristic for now)
        # Assuming for 'Up', a small increase like 0.5% to 2%
        lower_bound = best_choice_price * (1 + 0.005)
        upper_bound = best_choice_price * (1 + 0.02)
        print(f"Recommended ETF: {best_choice_ticker}")
        print(f"Action: {prediction_action}")
        print(f"Predicted price movement: Up (Confidence: {best_choice_confidence:.2f})")
        print(f"Expected price range next day: {lower_bound:.2f} - {upper_bound:.2f}")
    else:
        # If no 'Up' predictions, check 'Hold'
        hold_candidates = [
            (ticker, pred["confidence"], pred["last_price"]) 
            for ticker, pred in predictions.items() if pred["label"] == "Hold"
        ]
        if hold_candidates:
            hold_candidates.sort(key=lambda x: x[1], reverse=True)
            best_choice_ticker, best_choice_confidence, best_choice_price = hold_candidates[0]
            prediction_action = "Hold"
            # For 'Hold', assume price range around +/- 0.5%
            lower_bound = best_choice_price * (1 - 0.005)
            upper_bound = best_choice_price * (1 + 0.005)
            print(f"Recommended ETF: {best_choice_ticker}")
            print(f"Action: {prediction_action}")
            print(f"Predicted price movement: Hold (Confidence: {best_choice_confidence:.2f})")
            print(f"Expected price range next day: {lower_bound:.2f} - {upper_bound:.2f}")
        else:
            # If no 'Up' or 'Hold', just pick the 'Down' with highest confidence for 'Down'
            down_candidates = [
                (ticker, pred["confidence"], pred["last_price"]) 
                for ticker, pred in predictions.items() if pred["label"] == "Down"
            ]
            if down_candidates:
                down_candidates.sort(key=lambda x: x[1], reverse=True)
                best_choice_ticker, best_choice_confidence, best_choice_price = down_candidates[0]
                prediction_action = "Sell/Avoid"
                # For 'Down', assume price range around -0.5% to -2%
                lower_bound = best_choice_price * (1 - 0.02)
                upper_bound = best_choice_price * (1 - 0.005)
                print(f"Recommended ETF: {best_choice_ticker}")
                print(f"Action: {prediction_action}")
                print(f"Predicted price movement: Down (Confidence: {best_choice_confidence:.2f})")
                print(f"Expected price range next day: {lower_bound:.2f} - {upper_bound:.2f}")
            else:
                print("No clear recommendation based on predictions.")


if __name__ == "__main__":
    predict_next_day()
