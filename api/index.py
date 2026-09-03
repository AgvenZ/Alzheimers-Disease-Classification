from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import sys

app = Flask(__name__)

# Path ke root project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Debug: cek file yang ada
print("=" * 50)
print("Current directory:", os.getcwd())
print("Files in current dir:", os.listdir('.'))
print("BASE_DIR:", BASE_DIR)
print("Files in BASE_DIR:", os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else "Not found")
print("=" * 50)

def load_model():
    try:
        # Cari file di root
        model_path = os.path.join(BASE_DIR, 'xgb_hybrid_mi_rfe_optimized.pkl')
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "Alzheimer Prediction API is running!",
        "model_loaded": model is not None,
        "files": os.listdir(BASE_DIR)
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        df = pd.DataFrame([data])
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
    app.run()