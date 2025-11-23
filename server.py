import json
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all domains (simplest fix for public APIs)
CORS(app)

# 1. Load the model at startup
# Ensure the model path is correct relative to this file
MODEL_PATH = os.path.join("outputs", "xgb_final.json")

if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print(f"WARNING: Model not found at {MODEL_PATH}")
    model = None

# 2. Define the exact features the model expects (in order)
#    (Taken from your outputs/selected_features.txt)
EXPECTED_FEATURES = [
    "B_11_last", "B_1_last", "B_2_last", "B_2_mean6", 
    "B_37_last", "B_7_mean3", "B_9_last", "D_42_mean12", 
    "D_64_O_mean3", "P_2_last", "P_2_mean3", "R_1_mean12", 
    "R_1_mean3", "S_3_mean6"
]

@app.route('/', methods=['GET'])
def home():
    return "Credit Default Prediction API is running. Use /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded", "status": "error"}), 500

    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Convert single record to list if needed
        if isinstance(data, dict):
            data = [data]
            
        # Create DataFrame to ensure feature alignment
        df = pd.DataFrame(data)
        
        # Ensure all columns exist (fill missing with 0.0)
        # and strictly order them to match the model's expectation
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        
        df = df[EXPECTED_FEATURES]
        
        # Convert to DMatrix (XGBoost's native format)
        dtest = xgb.DMatrix(df)
        
        # Predict
        scores = model.predict(dtest)
        
        # Return results
        response = {
            "predictions": [float(s) for s in scores],
            "status": "success"
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
