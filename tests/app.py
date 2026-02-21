"""
Simple Flask API for Fraud Detection
Reuses prediction logic from predict.py
"""

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import sys
from pathlib import Path

# Add src to path so we can import
sys.path.append(str(Path(__file__).parent / "src" / "models"))

# Setup
app = Flask(__name__)

# Load model and scaler
MODEL_PATH = Path("models/best_model_tuned.joblib")
SCALER_PATH = Path("models/scaler.joblib")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None
    scaler = None


def make_prediction(data):
    """
    Simple prediction function
    Input: dict or list of dicts
    Output: predictions
    """
    if model is None or scaler is None:
        raise ValueError("Model not loaded")
    
    # Convert to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    
    # Scale
    df_scaled = scaler.transform(df)
    
    # Predict
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]
    
    return predictions, probabilities


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Fraud Detection API',
        'status': 'running',
        'endpoints': {
            '/predict': 'POST - Single transaction',
            '/predict/batch': 'POST - Multiple transactions',
            '/health': 'GET - Health check'
        }
    })


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


@app.route('/predict', methods=['POST'])
def predict_single():
    """Single transaction prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        predictions, probabilities = make_prediction(data)
        
        prediction = int(predictions[0])
        probability = float(probabilities[0])
        
        # Risk level
        if probability >= 0.8:
            risk_level = "high"
        elif probability >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return jsonify({
            'fraud_prediction': prediction,
            'fraud_probability': probability,
            'risk_level': risk_level,
            'message': 'Fraudulent transaction detected!' if prediction == 1 else 'Transaction appears legitimate'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400





if __name__ == '__main__':
    print("🚀 Starting Fraud Detection API...")
    print(f"📂 Model: {MODEL_PATH}")
    print(f"📂 Scaler: {SCALER_PATH}")
    
    # Run
    app.run(debug=True, host='0.0.0.0', port=5000)