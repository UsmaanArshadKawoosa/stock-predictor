# Stock Predictor

A small Flask-based inference service that uses a pre-trained Keras LSTM model to provide intraday (1-minute interval) stock price predictions and returns a plot image with the predicted level over the recent closes.

## Overview
This repository contains an inference-only demo: a Flask app (app.py) that downloads the most recent intraday data for a requested ticker using yfinance, prepares the last 60 rows (Open, High, Low, Close, Volume), optionally applies a saved scaler, and feeds the data into a pre-trained Keras LSTM model (intraday_new_lstm_model.h5) to produce a single predicted value. The app returns a JPEG plot showing actual recent closes with the predicted level.

## Features
- Download recent intraday data via yfinance (1m interval, last trading day)
- Prepare input window of 60 timesteps with 5 features (O/H/L/C/V)
- Optional feature scaling if a scaler pickle is provided. The app will try `scaler.pkl` and fall back to `scaler.save` if present.
- Load a Keras LSTM model (`.h5`) and produce a single prediction
- Return a plot image showing actual closes and the predicted level

## Tech Stack
- Python 3.x
- Flask
- TensorFlow / Keras
- NumPy, pandas
- scikit-learn (for scaler usage)
- yfinance (data retrieval)
- matplotlib (plotting)

Dependencies are listed in `requirements.txt`.

## Project structure
- app.py — Flask inference service and the recommended entry point
- intraday_new_lstm_model.h5 — pre-trained Keras model (inference artifact)
- scaler.save — a saved scaler file present in the repo. The app will attempt to load `scaler.pkl` first and fall back to `scaler.save`.
- requirements.txt — pinned dependencies
- Procfile — simple command for hosting (web: python app.py)

## How it works (pipeline)
1. Data collection: yfinance.download(ticker, period="1d", interval="1m")
2. Preprocessing: select last 60 rows and columns ['Open','High','Low','Close','Volume']
3. Feature scaling: if a saved scaler pickle is present (the app tries `scaler.pkl` then `scaler.save`), it is applied to the 60x5 input
4. Model: input reshaped to (1, 60, 5) and passed to a Keras LSTM model loaded from `intraday_new_lstm_model.h5`
5. Output: model returns a scalar prediction; the app plots recent Close prices and the predicted value and returns a JPEG image

## Data
- Source: Yahoo Finance via the `yfinance` Python package
- Tickers: any ticker supported by yfinance (user-supplied per request)
- Timeframe: intraday 1-minute interval, data fetched for `period='1d'` and the last 60 rows are used
- Features used: Open, High, Low, Close, Volume
- Preprocessing: optional scaler (pickle) applied if available; otherwise raw values are used

## Model
- Type: Keras / TensorFlow model saved in HDF5 format (`intraday_new_lstm_model.h5`)
- Role: single LSTM-based model used for producing a scalar intraday prediction

Note: training scripts, model architecture/source code, and training dataset are not included in this repository. The `.h5` file is an inference artifact only.

## Training
No training code or notebooks are included. There are no training logs, checkpoints, or scripts in this repository.

## Prediction / Inference
Start the Flask app and POST to the `/predict` endpoint with JSON specifying a ticker.

Example (after starting the app locally):

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"ticker\": \"AAPL\"}" --output prediction.jpg

The response is a JPEG image containing the recent close prices and a horizontal line for the predicted value.

## Evaluation
No evaluation metrics (RMSE/MAE/R^2) or test/evaluation notebooks are included in this repository. The repo contains only inference artifacts and the Flask service.

## Installation (Windows example)
1. Create and activate a virtual environment (Windows PowerShell):

    python -m venv venv
    .\venv\Scripts\Activate.ps1

2. Install dependencies:

    pip install --upgrade pip
    pip install -r requirements.txt

## Usage
1. Ensure the model file `intraday_new_lstm_model.h5` is present in the repository root (it is tracked in this repo).
2. Optionally provide a scaler pickle named `scaler.pkl` in the repository root if you want the app to apply scaling. If `scaler.pkl` is not present the app will fall back to `scaler.save` (the repository currently contains `scaler.save`).
3. Start the app:

    python app.py

4. Send a POST to `/predict` with JSON {"ticker": "AAPL"} to receive a JPEG plot.

## Configuration
- Model path: hard-coded in `app.py` as `intraday_new_lstm_model.h5`.
- Scaler path: `app.py` attempts to open `scaler.pkl` first then `scaler.save`. If neither file is present the app proceeds without scaling (scaler=None).
- Data fetch: `yf.download` uses `period='1d'` and `interval='1m'` and then the last 60 rows are selected. To change, edit the `app.py` source.

## Results
This repository does not contain reproducible numeric evaluation results. Only model and scaler artifacts for inference are present.

## Limitations
- Training code and evaluation artifacts are not included — this repository is inference-only.
- No unit tests or CI configuration present.
- The app uses live yfinance data; behavior depends on network and Yahoo Finance availability.
- The app will attempt `scaler.pkl` then `scaler.save`; the repository currently contains `scaler.save`.
- Minimal input validation and error handling; do not expose publicly without additional hardening.

## Notable repository issue (handled automatically)
- The app now tries `scaler.pkl` first and falls back to `scaler.save` if present, so no immediate manual action is required to use the bundled scaler. `scaler.save` remains in the repo and will be used automatically if `scaler.pkl` is not found.

## Future Improvements
- Add training scripts, model architecture code, and a reproducible training pipeline
- Add example notebook demonstrating data preparation and model evaluation
- Add unit tests and CI (linting, type checks)
- Add better API input validation and JSON responses (optionally return numeric prediction alongside the plot)
- Consider packaging the app (Dockerfile) and secure configuration (do not commit secrets)

