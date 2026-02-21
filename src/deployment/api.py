"""
Flask API for Credit Risk Prediction
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and artifacts
model = None
feature_artifacts = None


def load_model_and_artifacts():
    """Load trained model and feature artifacts"""
    global model, feature_artifacts
    
    try:
        # Load model
        model_path = Path('models/xgboost_model.pkl')
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load feature artifacts
        artifacts_path = Path('data/processed/feature_artifacts.pkl')
        feature_artifacts = joblib.load(artifacts_path)
        logger.info(f"Feature artifacts loaded from {artifacts_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model/artifacts: {e}")
        return False


def preprocess_input(data: dict) -> pd.DataFrame:
    """Preprocess input data"""
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Apply feature engineering (simplified version)
    # In production, use the same FeatureEngineer class
    from features.build_features import FeatureEngineer
    
    engineer = FeatureEngineer()
    engineer.label_encoders = feature_artifacts['label_encoders']
    engineer.scaler = feature_artifacts['scaler']
    engineer.feature_names = feature_artifacts['feature_names']
    
    # Transform
    df_transformed = engineer.transform(df, fit=False)
    
    # Remove target if exists
    if 'target' in df_transformed.columns:
        df_transformed = df_transformed.drop('target', axis=1)
    
    return df_transformed


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'artifacts_loaded': feature_artifacts is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make credit risk prediction
    
    Request body (JSON):
    {
        "checking_status": "A11",
        "duration": 12,
        "credit_history": "A34",
        ...
    }
    
    Response:
    {
        "prediction": 0,
        "probability": 0.23,
        "risk_level": "Low Risk",
        "recommendation": "Approve"
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Preprocess
        X = preprocess_input(data)
        
        # Predict
        probability = model.predict_proba(X)[0, 1]
        prediction = int(probability >= 0.5)
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
        elif probability < 0.5:
            risk_level = "Medium Risk"
        elif probability < 0.7:
            risk_level = "High Risk"
        else:
            risk_level = "Very High Risk"
        
        # Generate recommendation
        recommendation = "Reject" if prediction == 1 else "Approve"
        
        # Response
        result = {
            'prediction': prediction,
            'probability': float(probability),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'explanation': {
                '0': 'Good credit - Low risk of default',
                '1': 'Bad credit - High risk of default'
            }[str(prediction)]
        }
        
        logger.info(f"Prediction made: {result}")
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Make batch predictions
    
    Request body (JSON):
    {
        "samples": [
            {"checking_status": "A11", "duration": 12, ...},
            {"checking_status": "A12", "duration": 24, ...}
        ]
    }
    
    Response:
    {
        "predictions": [0, 1],
        "probabilities": [0.23, 0.67],
        "risk_levels": ["Low Risk", "High Risk"]
    }
    """
    try:
        data = request.get_json()
        
        if 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        samples = data['samples']
        
        # Process each sample
        predictions = []
        probabilities = []
        risk_levels = []
        recommendations = []
        
        for sample in samples:
            X = preprocess_input(sample)
            prob = model.predict_proba(X)[0, 1]
            pred = int(prob >= 0.5)
            
            # Risk level
            if prob < 0.3:
                risk_level = "Low Risk"
            elif prob < 0.5:
                risk_level = "Medium Risk"
            elif prob < 0.7:
                risk_level = "High Risk"
            else:
                risk_level = "Very High Risk"
            
            predictions.append(pred)
            probabilities.append(float(prob))
            risk_levels.append(risk_level)
            recommendations.append("Reject" if pred == 1 else "Approve")
        
        result = {
            'count': len(samples),
            'predictions': predictions,
            'probabilities': probabilities,
            'risk_levels': risk_levels,
            'recommendations': recommendations
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        import json
        
        # Load model metadata
        metadata_path = Path('models/xgboost_metadata.json')
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Load metrics
        metrics_path = Path('models/xgboost_metrics.json')
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            metrics = {}
        
        info = {
            'model_name': 'XGBoost Credit Risk Model',
            'version': '1.0.0',
            'metadata': metadata,
            'performance': metrics.get('validation', {}),
            'feature_count': len(feature_artifacts['feature_names']) if feature_artifacts else 0
        }
        
        return jsonify(info), 200
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Starting Credit Risk Prediction API")
    logger.info("="*60)
    
    # Load model and artifacts
    if load_model_and_artifacts():
        logger.info("Model and artifacts loaded successfully")
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Failed to load model/artifacts. Exiting.")
