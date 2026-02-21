# 🚀 Quick Start Guide

Get up and running with the Credit Risk Prediction ML project in minutes!

## ⚡ 5-Minute Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/banking-ml-project.git
cd banking-ml-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Download Data & Train Model

```bash
# Download and prepare data (automated)
make data

# Build features
make features

# Train model (takes ~2-3 minutes)
make train

# Generate visualizations
make visualize
```

### 3. Make Predictions

```bash
# Predict on test data
python src/models/predict_model.py \
  --model-path models/xgboost_model.pkl \
  --input data/processed/test_features.csv \
  --output predictions.csv
```

## 🎯 Try the API

### Option 1: Local Flask API

```bash
# Start API server
python src/deployment/api.py

# Test in another terminal
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "checking_status": "A11",
    "duration": 12,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 1500,
    "savings_status": "A61",
    "employment": "A73",
    "installment_commitment": 4,
    "personal_status": "A93",
    "other_parties": "A101",
    "residence_since": 4,
    "property_magnitude": "A121",
    "age": 35,
    "other_payment_plans": "A143",
    "housing": "A152",
    "existing_credits": 1,
    "job": "A173",
    "num_dependents": 1,
    "own_telephone": "A192",
    "foreign_worker": "A201"
  }'
```

### Option 2: Docker

```bash
# Build and run with Docker
docker-compose up --build

# API will be available at http://localhost:5000
```

## ☁️ Deploy to AWS

### Prerequisites
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
```

### Deploy to SageMaker

```bash
# Set environment variable
export SAGEMAKER_ROLE=arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole

# Deploy
python src/deployment/sagemaker_deploy.py
```

## 📊 View Results

```bash
# Open visualizations
open reports/figures/

# Check model metrics
cat models/xgboost_metrics.json

# View feature importance
head models/xgboost_feature_importance.csv
```

## 🧪 Run Tests

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=src tests/
```

## 📖 Common Tasks

### Train Different Models

```bash
# Logistic Regression
python src/models/train_model.py --model logistic

# Random Forest
python src/models/train_model.py --model random_forest

# LightGBM
python src/models/train_model.py --model lightgbm
```

### Use SMOTE for Imbalanced Data

```bash
python src/models/train_model.py --model xgboost --use-smote
```

### Cross-Validation

```bash
python src/models/train_model.py --model xgboost --cv 5
```

## 🐛 Troubleshooting

### Issue: Module not found
```bash
# Make sure you installed the package
pip install -e .
```

### Issue: Data not found
```bash
# Run data download script
make data
```

### Issue: AWS credentials error
```bash
# Check AWS configuration
aws configure list

# Set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

## 📚 Next Steps

1. **Explore Notebooks**: Check `notebooks/` for detailed analysis
2. **Read Documentation**: See `docs/` for comprehensive guides
3. **Customize Models**: Modify hyperparameters in `src/models/train_model.py`
4. **Add Features**: Extend `src/features/build_features.py`
5. **Deploy Production**: Set up CI/CD with GitHub Actions

## 🆘 Need Help?

- **Issues**: Create an issue on GitHub
- **Documentation**: Read full docs in `docs/`
- **Examples**: See `notebooks/` for examples
- **API Docs**: Visit `/docs` endpoint when API is running

---

**Pro Tip**: Use `make help` to see all available commands!
