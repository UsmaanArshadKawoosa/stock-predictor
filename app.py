from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("intraday_lstm_model.h5")

@app.route("/")
def home():
    return "Stock Predictor API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    predicted_class = int(np.round(prediction[0]))
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
