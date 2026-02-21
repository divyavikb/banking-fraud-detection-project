# 📦 Complete Setup Instructions

## Project Overview

This is a **production-ready** Credit Risk Prediction ML project following industry-standard Cookiecutter Data Science structure. Perfect for banking domain ML projects with AWS deployment.

## 📁 Project Structure

```
banking-ml-project/
├── LICENSE                     # MIT License
├── Makefile                    # Automation commands
├── README.md                   # Main documentation
├── QUICKSTART.md              # 5-minute setup guide
├── CONTRIBUTING.md            # Contribution guidelines
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── setup.py                   # Package installation
├── tox.ini                    # Testing automation
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Multi-container setup
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
│
├── .github/workflows/
│   └── ml-pipeline.yml       # CI/CD pipeline
│
├── data/
│   ├── external/             # Third-party data
│   ├── interim/              # Intermediate processed data
│   ├── processed/            # Final datasets for modeling
│   └── raw/                  # Original immutable data
│
├── docs/                      # Sphinx documentation
│
├── models/                    # Trained models (.pkl files)
│
├── notebooks/                 # Jupyter notebooks for EDA
│
├── references/                # Data dictionaries, manuals
│
├── reports/                   # Generated analysis
│   └── figures/              # Visualizations (PNG, PDF)
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py   # Download/process data
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py    # Train models
│   │   └── predict_model.py  # Make predictions
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualize.py      # Create plots
│   └── deployment/
│       ├── __init__.py
│       ├── api.py            # Flask REST API
│       └── sagemaker_deploy.py # AWS deployment
│
└── tests/                     # Unit and integration tests
    ├── __init__.py
    └── test_models.py        # Model tests
```

## 🚀 Complete Setup Guide

### Step 1: Initial Setup

```bash
# 1. Create project directory
mkdir banking-ml-project
cd banking-ml-project

# 2. Copy all files from the outputs folder into this directory

# 3. Create virtual environment
python3.9 -m venv venv

# 4. Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate      # On Windows

# 5. Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional, for development)
pip install -r requirements-dev.txt

# Install project as package
pip install -e .
```

### Step 3: Setup Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your credentials
nano .env  # or use any text editor

# Required variables:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY  
# - AWS_REGION
# - SAGEMAKER_ROLE (for AWS deployment)
```

### Step 4: Run the Complete Pipeline

```bash
# Option 1: Use Makefile (Recommended)
make all  # Runs: data → features → train → evaluate → visualize

# Option 2: Step by step
make data       # Download and prepare data
make features   # Build features
make train      # Train XGBoost model
make evaluate   # Evaluate model
make visualize  # Generate plots
```

### Step 5: Verify Installation

```bash
# Check data
ls -lh data/processed/

# Check trained model
ls -lh models/

# Check visualizations
ls -lh reports/figures/

# Run tests
make test
```

## 📊 Usage Examples

### 1. Train Different Models

```bash
# Logistic Regression
python src/models/train_model.py --model logistic --output models/

# Random Forest
python src/models/train_model.py --model random_forest --output models/

# XGBoost (default)
python src/models/train_model.py --model xgboost --output models/

# LightGBM
python src/models/train_model.py --model lightgbm --output models/
```

### 2. Make Predictions

```bash
# Predict on test set
python src/models/predict_model.py \
  --model-path models/xgboost_model.pkl \
  --input data/processed/test_features.csv \
  --output predictions.csv

# View predictions
head predictions.csv
```

### 3. Run REST API

```bash
# Start Flask API
python src/deployment/api.py

# Test API (in another terminal)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json

# Or use the health check
curl http://localhost:5000/health
```

### 4. Docker Deployment

```bash
# Build Docker image
docker build -t credit-risk-api .

# Run container
docker run -p 5000:5000 credit-risk-api

# Or use docker-compose
docker-compose up --build
```

## ☁️ AWS Deployment

### Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **SageMaker Execution Role** created

### Setup AWS

```bash
# Install AWS CLI (if not installed)
pip install awscli

# Configure AWS credentials
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (us-east-1)
# - Output format (json)

# Create SageMaker execution role (one-time setup)
# Visit: https://console.aws.amazon.com/iam/
# Create role with SageMaker permissions
# Copy ARN: arn:aws:iam::123456789:role/SageMakerRole
```

### Deploy to SageMaker

```bash
# Set environment variable
export SAGEMAKER_ROLE="arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole"

# Deploy
python src/deployment/sagemaker_deploy.py

# Check deployment
cat endpoint_info.json
```

### Test SageMaker Endpoint

```python
import boto3
import json

client = boto3.client('sagemaker-runtime', region_name='us-east-1')

response = client.invoke_endpoint(
    EndpointName='credit-risk-2024-01-15-12-30-00',
    ContentType='application/json',
    Body=json.dumps({
        'checking_status': 'A11',
        'duration': 12,
        'credit_history': 'A34',
        # ... more features
    })
)

result = json.loads(response['Body'].read())
print(result)
```

## 🧪 Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

### Code Quality

```bash
# Format code with Black
black src/ tests/

# Lint with Flake8
flake8 src/ tests/

# Type check with MyPy
mypy src/

# Run all quality checks
make quality
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-model

# Make changes and commit
git add .
git commit -m "feat: add neural network model"

# Push to GitHub
git push origin feature/new-model

# Create Pull Request on GitHub
```

## 📈 Monitoring & Maintenance

### Model Performance

```bash
# Check model metrics
cat models/xgboost_metrics.json

# View feature importance
cat models/xgboost_feature_importance.csv

# Generate new visualizations
python src/visualization/visualize.py
```

### Update Dependencies

```bash
# Update all packages
pip list --outdated
pip install --upgrade <package-name>

# Update requirements file
pip freeze > requirements.txt
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`
```bash
# Solution: Install package in editable mode
pip install -e .
```

**Issue**: `FileNotFoundError: data/processed/train.csv`
```bash
# Solution: Run data preparation
make data
```

**Issue**: `AWS credentials not found`
```bash
# Solution: Configure AWS CLI
aws configure
# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

**Issue**: Docker build fails
```bash
# Solution: Ensure data and models exist
make data
make train
# Then rebuild
docker-compose up --build
```

## 📚 Additional Resources

- **Makefile Help**: Run `make help` to see all commands
- **API Documentation**: Visit `http://localhost:5000/docs` when API is running
- **AWS SageMaker Docs**: https://docs.aws.amazon.com/sagemaker/
- **Scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/

## 🎯 Next Steps

1. **Customize**: Modify hyperparameters in `src/models/train_model.py`
2. **Add Features**: Extend `src/features/build_features.py`
3. **New Models**: Add model configurations in `train_model.py`
4. **Documentation**: Update `README.md` and create notebooks
5. **Deploy**: Set up GitHub Actions CI/CD pipeline

## 📞 Support

- Create an issue on GitHub
- Check documentation in `docs/`
- Review example notebooks in `notebooks/`

---

**Congratulations!** You now have a fully functional ML project ready for production deployment! 🎉
