from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import sys
import pickle

app = Flask(__name__)

# Path ke file model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Debug: cek file yang ada
print("=" * 50)
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))
print("BASE_DIR:", BASE_DIR)
print("Files in BASE_DIR:", os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else "Not found")
print("=" * 50)

# Load model
def load_model():
    try:
        # Coba berbagai kemungkinan path
        possible_paths = [
            os.path.join(BASE_DIR, 'xgb_hybrid_mi_rfe_optimized.pkl'),
            os.path.join(os.getcwd(), 'xgb_hybrid_mi_rfe_optimized.pkl'),
            'xgb_hybrid_mi_rfe_optimized.pkl'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading model from: {path}")
                return joblib.load(path)
        
        raise FileNotFoundError("Model file not found")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "Alzheimer Prediction API",
        "endpoints": {
            "/predict": "POST - Send JSON with features",
            "/health": "GET - Health check",
            "/files": "GET - List files"
        },
        "model_loaded": model is not None
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "files": os.listdir('.')
    })

@app.route('/files')
def list_files():
    return jsonify({
        "files": os.listdir('.'),
        "model_exists": os.path.exists('xgb_hybrid_mi_rfe_optimized.pkl')
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Predict
        prob = model.predict_proba(df)[0, 1]
        pred = model.predict(df)[0]
        
        return jsonify({
            "prediction": int(pred),
            "probability": float(prob),
            "label": "Alzheimer" if pred == 1 else "Non-Alzheimer"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)