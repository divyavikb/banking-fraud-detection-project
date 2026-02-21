# Credit Risk Prediction - ML Banking Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready machine learning project for predicting credit risk in banking applications. Built with industry-standard practices and AWS deployment capabilities.

## 🎯 Project Overview

This project predicts whether a loan applicant will default on their loan using historical credit data. It includes complete ML pipeline from data processing to model deployment on AWS SageMaker.

### Business Problem
Banks need to assess credit risk to minimize loan defaults while maximizing loan approvals for qualified applicants. This model helps automate credit decisions with high accuracy.

### Key Features
- ✅ Complete ML pipeline (data → features → model → deployment)
- ✅ Multiple ML algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM)
- ✅ Handles imbalanced data (SMOTE, class weights)
- ✅ Model explainability (SHAP values)
- ✅ AWS SageMaker deployment ready
- ✅ REST API for predictions
- ✅ Comprehensive testing
- ✅ CI/CD pipeline with GitHub Actions

## 📊 Dataset

**Source**: German Credit Risk Dataset (UCI ML Repository)
- **Records**: 1,000 loan applications
- **Features**: 20 attributes (credit history, employment, demographics)
- **Target**: Credit risk (Good/Bad)
- **Imbalance**: 70% Good / 30% Bad

## 🏗️ Project Structure

```
banking-ml-project/
├── LICENSE
├── Makefile                    <- Commands like `make data` or `make train`
├── README.md                   <- Top-level README
├── data
│   ├── external               <- Third party data sources
│   ├── interim                <- Intermediate transformed data
│   ├── processed              <- Final datasets for modeling
│   └── raw                    <- Original immutable data
├── docs                       <- Sphinx documentation
├── models                     <- Trained models and predictions
├── notebooks                  <- Jupyter notebooks
│   ├── 1.0-eda.ipynb         <- Exploratory Data Analysis
│   ├── 2.0-preprocessing.ipynb
│   ├── 3.0-modeling.ipynb
│   └── 4.0-evaluation.ipynb
├── references                 <- Data dictionaries and manuals
├── reports                    <- Generated analysis
│   └── figures               <- Graphics and figures
├── requirements.txt           <- Python dependencies
├── setup.py                   <- Makes project pip installable
├── src                        <- Source code
│   ├── __init__.py
│   ├── data
│   │   └── make_dataset.py   <- Download/generate data
│   ├── features
│   │   └── build_features.py <- Feature engineering
│   ├── models
│   │   ├── train_model.py    <- Train models
│   │   └── predict_model.py  <- Generate predictions
│   ├── visualization
│   │   └── visualize.py      <- Create visualizations
│   └── deployment
│       ├── api.py            <- Flask API
│       └── sagemaker_deploy.py
├── tests                      <- Unit tests
└── tox.ini                    <- Tox configuration
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
pip
virtualenv
AWS CLI (for deployment)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/banking-ml-project.git
cd banking-ml-project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -e .  # Install project as package
```

4. **Download data**
```bash
make data
```

## 📖 Usage

### Using Makefile Commands

```bash
# Download and prepare data
make data

# Build features
make features

# Train models
make train

# Evaluate models
make evaluate

# Generate predictions
make predict

# Run all steps
make all

# Clean generated files
make clean

# Run tests
make test

# Generate documentation
make docs
```

### Manual Workflow

```bash
# 1. Prepare dataset
python src/data/make_dataset.py

# 2. Build features
python src/features/build_features.py

# 3. Train model
python src/models/train_model.py --model xgboost

# 4. Make predictions
python src/models/predict_model.py --input data/processed/test.csv
```

## 🧪 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.76 | 0.74 | 0.72 | 0.73 | 0.82 |
| Random Forest | 0.78 | 0.77 | 0.75 | 0.76 | 0.85 |
| **XGBoost** | **0.82** | **0.81** | **0.79** | **0.80** | **0.89** |
| LightGBM | 0.81 | 0.80 | 0.78 | 0.79 | 0.88 |

**Best Model**: XGBoost with hyperparameter tuning

## 🎨 Features Engineering

### Derived Features
- Credit utilization ratio
- Income to debt ratio  
- Age groups
- Credit history score
- Employment stability index

### Feature Selection
- Correlation analysis
- Feature importance (XGBoost)
- Recursive Feature Elimination
- SHAP values

## 📊 Model Explainability

The project includes SHAP (SHapley Additive exPlanations) for model interpretability:

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

## ☁️ AWS Deployment

### SageMaker Deployment

```bash
# Configure AWS credentials
aws configure

# Deploy to SageMaker
python src/deployment/sagemaker_deploy.py

# Test endpoint
python src/deployment/test_endpoint.py
```

### Architecture
- **Training**: SageMaker Training Jobs
- **Hosting**: SageMaker Endpoint (ml.t2.medium)
- **Storage**: S3 for data and models
- **Monitoring**: CloudWatch metrics
- **API**: Lambda + API Gateway

### Cost Estimate
- Training: ~$0.50 per run
- Endpoint: ~$0.05/hour (ml.t2.medium)
- Storage: Minimal (<$1/month)

## 🔄 CI/CD Pipeline

GitHub Actions workflow automatically:
1. Runs tests on push
2. Checks code quality (black, flake8)
3. Trains model on main branch
4. Deploys to SageMaker (on tag)

```bash
# Trigger deployment
git tag v1.0.0
git push origin v1.0.0
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_features.py
```

## 📚 Documentation

Generate documentation:
```bash
cd docs
make html
open _build/html/index.html
```

## 🛠️ Development

### Code Style
```bash
# Format code
black src/

# Lint
flake8 src/

# Type checking
mypy src/
```

### Adding New Models
1. Create model class in `src/models/`
2. Add training logic in `train_model.py`
3. Update `Makefile` and tests
4. Document in `docs/`

## 📈 Monitoring

- **Data Drift**: Monitor feature distributions
- **Model Performance**: Track accuracy over time
- **SageMaker Metrics**: CPU, memory, latency
- **Alerts**: CloudWatch alarms for degradation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- German Credit Dataset - UCI ML Repository
- AWS SageMaker Examples
- Cookiecutter Data Science Project Structure

## 📞 Contact

- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)

## 🗺️ Roadmap

- [ ] Add deep learning models
- [ ] Implement real-time predictions
- [ ] Add A/B testing framework
- [ ] Create web dashboard
- [ ] Multi-model ensemble
- [ ] AutoML integration

---

**Note**: Remember to update AWS credentials and never commit them to version control. Use AWS Secrets Manager or environment variables.
