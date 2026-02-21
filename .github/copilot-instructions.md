# Copilot Instructions for Banking ML Project

This is a **Credit Risk Prediction** ML project following Cookiecutter Data Science structure. It implements a complete ML pipeline with data validation, feature engineering, multiple model algorithms, and AWS SageMaker deployment.

## Architecture Overview

The project follows a **linear data pipeline**: Raw Data → Processed Data → Feature Engineering → Model Training → Predictions/Deployment

**Key Modules:**
- `src/data/` - Data ingestion, validation, and preparation
- `src/features/` - Feature engineering using sklearn patterns (StandardScaler, LabelEncoder)
- `src/models/` - Multiple algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM) with imbalance handling
- `src/deployment/` - Flask API and AWS SageMaker integration
- `src/visualization/` - EDA and analysis plots
- `tests/` - Unit and integration tests using pytest
- `models/` - Trained model artifacts (.pkl files via joblib)
- `data/raw/` → `data/interim/` → `data/processed/` - Standard data workflow

## Critical Developer Workflows

### Data & Training Pipeline
```bash
make data              # Download/prepare raw data (src/data/make_dataset.py)
make features          # Build features from processed data (src/features/build_features.py)
make train             # Train default XGBoost model → models/xgboost_model.pkl
make train-all         # Train logistic, random_forest, xgboost, lightgbm
make evaluate          # Evaluate trained models on test set
make hyperparameter-tuning  # Optuna-based optimization
make visualize         # Generate all plots and EDA reports
```

### Testing & Code Quality
```bash
make test              # Run pytest on tests/ directory
make test-coverage     # Run with coverage report (HTML in htmlcov/)
tox                    # Run tests for Python 3.9, 3.10, 3.11 + lint/docs
make quality           # Run format-check + lint + type-check
make format            # Auto-format with black
make lint              # Flake8 checks
make type-check        # Mypy type validation
```

### Deployment
```bash
make docker-build      # Build Docker image for API
make docker-run        # Run API container locally (port 5000)
make sagemaker-deploy  # Deploy to AWS SageMaker endpoint
make api-test          # Test prediction endpoint
```

## Project-Specific Patterns

### Model Configuration Pattern
Models are defined in [src/models/train_model.py](src/models/train_model.py#L28) using `MODEL_CONFIGS` dict:
```python
MODEL_CONFIGS = {
    'xgboost': {
        'class': XGBClassifier,
        'params': {'max_depth': 6, 'scale_pos_weight': 2, ...}  # scale_pos_weight for imbalance
    }
    # ... other models
}
```
New models should follow this pattern. Use `scale_pos_weight` or `class_weight='balanced'` for imbalanced binary classification.

### Feature Engineering Pattern
[src/features/build_features.py](src/features/build_features.py) implements sklearn-style `FeatureEngineer` class:
- `create_derived_features()` - Builds domain-specific features (credit utilization, age groups, employment scores)
- `encode_categorical_features()` - LabelEncoder with fit/transform for reusability
- `fit()` / `transform()` - Standard sklearn patterns for train/test consistency

**Critical**: Features are persisted as artifacts (`data/processed/feature_artifacts.pkl`) so API can apply same transformations.

### Class Imbalance Handling
The dataset is ~70% Good / 30% Bad credit. Project uses two strategies:
1. **SMOTE oversampling** - Available via `CreditRiskModel(use_smote=True)`
2. **Class weights** - Built into model configs (e.g., `scale_pos_weight=2` for XGBoost)

Evaluation metrics in [src/models/train_model.py](src/models/train_model.py) report precision, recall, F1, and ROC-AUC.

### Data Validation
[src/data/data_validation.py](src/data/data_validation.py) uses great-expectations and pandera for schema validation. Tests data quality before feature engineering.

### REST API Design
[src/deployment/api.py](src/deployment/api.py) provides Flask API:
- `/health` - Liveness check
- `/predict` - Takes JSON input, returns credit risk prediction + probability
- Loads model and feature_artifacts on startup
- Applies same preprocessing as training pipeline

## Key Dependencies & Integration Points

| Component | Package | Key Pattern |
|-----------|---------|-------------|
| **Model Serialization** | joblib | `joblib.dump(model, path)` for persistence |
| **Data Imbalance** | imbalanced-learn | `SMOTE()` for oversampling |
| **AWS Deployment** | boto3, sagemaker | Deploy via [src/deployment/sagemaker_deploy.py](src/deployment/sagemaker_deploy.py) |
| **Model Explainability** | SHAP | Integration available in feature_extraction.py |
| **Hyperparameter Tuning** | optuna | `make hyperparameter-tuning` uses Optuna trial optimization |
| **Code Quality** | black, flake8, mypy | Run via `make quality` (100 char line limit) |

## Testing Patterns

Tests in [tests/test_models.py](tests/test_models.py) follow pytest conventions:
- Fixtures for sample data generation
- Parametrized tests for multiple models (`@pytest.mark.parametrize`)
- tmp_path for temporary file handling
- Coverage threshold enforced via tox config

When adding features/models, add corresponding tests in `tests/`.

## Common Tasks for AI Agents

### Adding a New Model Algorithm
1. Add config to `MODEL_CONFIGS` in [src/models/train_model.py](src/models/train_model.py#L28)
2. Implement Train method to handle the new class
3. Add unit test with parametrized test if not already covered
4. Test with: `python src/models/train_model.py --model <name> --output models/`

### Updating Feature Engineering
1. Modify `create_derived_features()` in [src/features/build_features.py](src/features/build_features.py)
2. Run `make features` to rebuild feature artifacts
3. Retrain models with: `make train-all`
4. Validate new features with existing tests

### Preparing for Deployment
1. Train final model: `make train-all` (evaluates all models)
2. Run quality checks: `make quality`
3. Run tests: `make test-coverage`
4. Build Docker: `make docker-build` (validates Dockerfile)
5. Deploy: `make sagemaker-deploy` (requires AWS credentials via `make aws-configure`)

### Environment & Python Version
- **Supported**: Python 3.9, 3.10, 3.11 (enforced by tox)
- **Virtualenv**: Required - use `python -m venv venv` before pip install
- **Install**: `pip install -e .` after requirements (editable mode for development)

## Code Style & Conventions

- **Formatter**: Black (enforced via `make format`)
- **Linter**: Flake8 (max 100 char lines in tox.ini)
- **Type Checking**: Mypy on src/ (mypy.ini may define ignores)
- **Logging**: Uses basicConfig with `logging.getLogger(__name__)` pattern
- **CLI**: Click decorators for command-line interfaces (see train_model.py)
- **Imports**: `from pathlib import Path` for file handling, `import joblib` for model I/O

## File Location Reference

- **Model training entry point**: [src/models/train_model.py](src/models/train_model.py)
- **Feature engineering logic**: [src/features/build_features.py](src/features/build_features.py)
- **API server**: [src/deployment/api.py](src/deployment/api.py)
- **Data preparation**: [src/data/make_dataset.py](src/data/make_dataset.py)
- **Test templates**: [tests/test_models.py](tests/test_models.py)
- **Configuration**: [Makefile](Makefile) (all workflows), [setup.py](setup.py) (dependencies), [tox.ini](tox.ini) (testing)
