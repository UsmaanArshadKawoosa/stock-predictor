from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('intraday_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_features = data['features']
    prediction = model.predict([input_features])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)