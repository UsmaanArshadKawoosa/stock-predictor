from flask import Flask, request, send_file, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import io

app = Flask(__name__)

# Load LSTM model
model = load_model("intraday_new_lstm_model.h5", compile=False)

# Load scaler if you have one
try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
except:
    scaler = None

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    ticker = data.get("ticker")

    if not ticker:
        return jsonify({"error": "Ticker not provided"}), 400

    # Fetch last 60 minutes of stock data
    try:
        df = yf.download(ticker, period="1d", interval="1m")
    except Exception as e:
        return jsonify({"error": f"Failed to fetch data: {str(e)}"}), 500

    if df.empty:
        return jsonify({"error": "No data found for ticker"}), 404

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60)

    # Scale data if scaler is available
    input_data = df.values
    if scaler:
        input_data = scaler.transform(input_data)

    input_data = np.expand_dims(input_data, axis=0)  # Shape: (1, 60, 5)

    # Make prediction
    prediction = model.predict(input_data)[0][0]

    # Plot actual vs prediction
    plt.figure(figsize=(6, 4))
    plt.plot(df['Close'], label='Actual')
    plt.axhline(y=prediction, color='r', linestyle='--', label='Predicted')
    plt.title(f"{ticker} Intraday Prediction")
    plt.legend()
    plt.tight_layout()

    # Save plot to memory
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
