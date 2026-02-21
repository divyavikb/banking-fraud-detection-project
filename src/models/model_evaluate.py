import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from datetime import datetime

# Create logger
logger = logging.getLogger("predict")
logger.setLevel(logging.INFO)

# Console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def load_model(model_path):
    """Load saved model"""
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def generate_evaluation_report(y_true, y_pred, y_pred_proba, model, report_path):
    """Generate comprehensive evaluation report"""
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

 
        
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Business metrics
    fpr = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    
    # Classification report
    class_report = classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraudulent'])
    
    # Create report content
    report = []
    report.append("=" * 80)
    report.append("FRAUD DETECTION MODEL - EVALUATION REPORT")
    report.append("=" * 80)
    report.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Model Type: {type(model).__name__}")
    report.append(f"Test Samples: {len(y_true):,}")
    report.append(f"Actual Frauds: {y_true.sum():,} ({y_true.mean()*100:.2f}%)")
    report.append(f"Predicted Frauds: {y_pred.sum():,} ({y_pred.mean()*100:.2f}%)")
    
    report.append("\n" + "=" * 80)
    report.append("1. PERFORMANCE METRICS")
    report.append("=" * 80)
    report.append(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    report.append(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    report.append(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    report.append(f"F1-Score:  {f1:.4f}")
    report.append(f"ROC-AUC:   {roc_auc:.4f}")
    
    report.append("\n" + "=" * 80)
    report.append("2. CONFUSION MATRIX")
    report.append("=" * 80)
    report.append("\n                 Predicted")
    report.append("                 Legit  Fraud")
    report.append(f"Actual  Legit    {tn:>6,}  {fp:>6,}")
    report.append(f"        Fraud    {fn:>6,}  {tp:>6,}")
    
    report.append(f"\nTrue Negatives (TN):  {tn:,} - Correctly identified legitimate")
    report.append(f"False Positives (FP): {fp:,} - Legitimate flagged as fraud")
    report.append(f"False Negatives (FN): {fn:,} - Fraud missed")
    report.append(f"True Positives (TP):  {tp:,} - Correctly identified fraud")
    
    report.append("\n" + "=" * 80)
    report.append("3. ERROR ANALYSIS")
    report.append("=" * 80)
    report.append(f"\nFalse Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
    report.append(f"  → Out of 100 legitimate transactions, {fpr*100:.1f} are incorrectly flagged")
    report.append(f"\nFalse Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")
    report.append(f"  → Out of 100 fraudulent transactions, {fnr*100:.1f} are missed")
    report.append(f"\nSpecificity: {specificity:.4f} ({specificity*100:.2f}%)")
    report.append(f"  → Ability to correctly identify legitimate transactions")
    
    report.append("\n" + "=" * 80)
    report.append("4. DETAILED CLASSIFICATION REPORT")
    report.append("=" * 80)
    report.append(f"\n{class_report}")
    
    report.append("\n" + "=" * 80)
    report.append("5. BUSINESS IMPACT (Assuming $100 avg transaction)")
    report.append("=" * 80)
    avg_transaction = 100
    report.append(f"\nFraud Prevented: ${tp * avg_transaction:,.2f}")
    report.append(f"Fraud Missed: ${fn * avg_transaction:,.2f}")
    report.append(f"Customer Friction (False Alarms): {fp:,} transactions")
    report.append(f"\nFraud Detection Rate: {recall*100:.2f}%")
    report.append(f"Precision (when we flag fraud, how often correct): {precision*100:.2f}%")
    
    report.append("\n" + "=" * 80)
    report.append("6. PROBABILITY DISTRIBUTION")
    report.append("=" * 80)
    
    # Probability bins
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    labels = ['0-30%', '30-50%', '50-70%', '70-90%', '90-100%']
    prob_dist = pd.cut(y_pred_proba, bins=bins, labels=labels).value_counts().sort_index()
    
    report.append("\nPredicted Probability Distribution:")
    for label, count in prob_dist.items():
        report.append(f"  {label:>10}: {count:>6,} transactions ({count/len(y_pred_proba)*100:>5.2f}%)")
    
    report.append("\n" + "=" * 80)
    report.append("7. RECOMMENDATION")
    report.append("=" * 80)
    
    if f1 >= 0.85 and roc_auc >= 0.95:
        report.append("\nMODEL PERFORMANCE: EXCELLENT")
        report.append("   The model is production-ready with high accuracy and reliability.")
    elif f1 >= 0.75 and roc_auc >= 0.85:
        report.append("\n MODEL PERFORMANCE: GOOD")
        report.append("   The model performs well and can be deployed with monitoring.")
    elif f1 >= 0.65:
        report.append("\n  MODEL PERFORMANCE: ACCEPTABLE")
        report.append("   Consider further tuning or feature engineering before deployment.")
    else:
        report.append("\n MODEL PERFORMANCE: NEEDS IMPROVEMENT")
        report.append("   Model requires significant improvements before production use.")
    
    if fpr > 0.05:
        report.append(f"\n  High False Positive Rate ({fpr*100:.2f}%)")
        report.append("   This may cause customer friction. Consider adjusting threshold.")
    
    if fnr > 0.10:
        report.append(f"\n High False Negative Rate ({fnr*100:.2f}%)")
        report.append("   Significant fraud is being missed. Review feature engineering.")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    # Write to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"\n✅ Evaluation report saved to: {report_path}")

    
    # Also print to console
    print('\n'.join(report))


def predict(data_path, model_path, scaler_path):
    """Make predictions on new data"""
    
    # Load model and scaler
    model = load_model(model_path)
    scaler = load_model(scaler_path)
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded: {df.shape}")
    
    # Separate features and target (if exists)
    if 'TX_FRAUD' in df.columns:
        X = df.drop(columns=['TX_FRAUD'])
        y_true = df['TX_FRAUD']
        has_labels = True
    else:
        X = df
        has_labels = False
    
    # Scale features
    X_scaled = scaler.transform(X)
    logger.info("Data scaled successfully")
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    logger.info("Predictions made successfully")
    
    # Create results dataframe
    results = pd.DataFrame({
        'fraud_prediction': y_pred,
        'fraud_probability': y_pred_proba
    })
    
    # Add actual labels if available
    if has_labels:
        results['actual_fraud'] = y_true.values
        
        # Show accuracy
        accuracy = (y_pred == y_true).mean()
        logger.info(f"\nAccuracy: {accuracy:.4f}")

        
        # Show confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
        logger.info(f"  FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
    
    # Show sample predictions
    logger.info("\nSample Predictions:")
    logger.info(results.head(10).to_string())
    
    # Count fraud predictions
    fraud_count = y_pred.sum()
    fraud_pct = (fraud_count / len(y_pred)) * 100
    logger.info(f"\nPredicted Frauds: {fraud_count:,} ({fraud_pct:.2f}%)")
    
    return results, y_true if has_labels else None, y_pred, y_pred_proba, model


if __name__ == "__main__":
    # Current path
    current_path = Path(__file__)
    # Root path
    root_path = current_path.parent.parent.parent
    
    # Paths
    data_path = root_path / "data" / "final" / "test.csv"
    model_path = root_path / "models" / "best_model_tuned.joblib"
    scaler_path = root_path / "models" / "scaler.joblib"
    
    # Make predictions
    results, y_true, y_pred, y_pred_proba, model = predict(data_path, model_path, scaler_path)
    
    # Save predictions
    output_path = root_path / "data" / "processed" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"\nPredictions saved to {output_path}")

     
    
    # Generate evaluation report (if labels available)
    if y_true is not None:
        report_path = root_path / "reports" / "evaluation_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        generate_evaluation_report(y_true, y_pred, y_pred_proba, model, report_path)