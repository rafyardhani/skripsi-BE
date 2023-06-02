from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# load model from .pkl file
model = joblib.load("decision.pkl")

# define endpoint to receive input and return prediction


@app.route("/predict", methods=["POST"])
def predict():
    # get input data from request
    input_data = request.json["features"]
    # convert input features to numpy array
    X = np.array(input_data).reshape(1, -1)
    # make prediction using loaded KNN model
    y_pred = model.predict(X)
    # return prediction as JSON response
    return jsonify({"prediction": int(y_pred[0])})
