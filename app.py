from flask import Flask, request, jsonify
from flasgger import Swagger
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
Swagger(app)

iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)

@app.route('/')
def home():
    return jsonify({"message": "Witaj w API serwującym model ML!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get("features")
    if not features or len(features) != 4:
        return jsonify({"error": "Brak wymaganych cech lub nieprawidłowa liczba cech (4)"}), 400

    prediction = model.predict([features]).tolist()
    predicted_class = iris.target_names[prediction[0]]

    return jsonify({"prediction": predicted_class})

@app.route('/info')
def info():
    return jsonify({
        "model_type": "LogisticRegression",
        "number_of_features": 4,
        "feature_names": iris.feature_names,
        "classes": iris.target_names.tolist()
    })

@app.route('/health')
def health():
    return jsonify({"status": "ok"})