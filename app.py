from flask import Flask, request, send_file
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

@app.route("/")
def home():
    return "Intraday Stock Predictor API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    ticker = data.get("ticker", "AAPL")

    # Load Data
    df = yf.download(ticker, period="60d", interval="5m")[["Open", "High", "Low", "Close", "Volume"]]

    # Preprocess
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    window_size = 60
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i, 3])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, verbose=0)

    # Predict Next 24h (288 steps)
    last_seq = scaled[-window_size:]
    predictions = []
    current_seq = last_seq.reshape(1, window_size, X.shape[2])
    for _ in range(288):
        pred = model.predict(current_seq, verbose=0)
        new_step = np.zeros((1, X.shape[2]))
        new_step[0, 3] = pred
        current_seq = np.append(current_seq[:, 1:, :], new_step.reshape(1, 1, X.shape[2]), axis=1)
        predictions.append(pred[0][0])

    future_full = np.zeros((len(predictions), 5))
    future_full[:, 3] = predictions
    predicted_future = scaler.inverse_transform(future_full)[:, 3]

    # Plot Graph
    plt.figure(figsize=(14, 6))
    plt.plot(df.index[-len(y_test):], scaler.inverse_transform(np.concatenate([np.zeros((len(y_test), 3)), y_test.reshape(-1, 1), np.zeros((len(y_test), 1))], axis=1))[:, 3], label="Actual (Test Data)")
    plt.plot(df.index[-len(y_test):], scaler.inverse_transform(np.concatenate([np.zeros((len(y_test), 3)), model.predict(X_test).reshape(-1, 1), np.zeros((len(y_test), 1))], axis=1))[:, 3], label="Predicted (Test Data)", alpha=0.7)
    plt.plot(pd.date_range(df.index[-1], periods=289, freq="5min")[1:], predicted_future, label="Predicted Next 24h", linestyle="dashed")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"{ticker} Stock Price Prediction (Next 24h)")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')
