.PHONY: clean data features train evaluate predict test docs help all
.DEFAULT_GOAL := help

PYTHON_INTERPRETER = python3

##@ General

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

requirements:  ## Install Python dependencies
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

dev-requirements:  ## Install development dependencies
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt

install:  ## Install package in editable mode
	$(PYTHON_INTERPRETER) -m pip install -e .

##@ Data

data:  ## Download and prepare raw data
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

download-data:  ## Download raw dataset only
	$(PYTHON_INTERPRETER) src/data/make_dataset.py --download-only

validate-data:  ## Validate data quality
	$(PYTHON_INTERPRETER) src/data/validate_data.py data/raw/credit_data.csv

##@ Features

features:  ## Build features from processed data
	$(PYTHON_INTERPRETER) src/features/build_features.py data/processed data/processed

feature-analysis:  ## Analyze feature importance
	$(PYTHON_INTERPRETER) src/features/analyze_features.py

##@ Models

train:  ## Train default model (XGBoost)
	$(PYTHON_INTERPRETER) src/models/train_model.py --model xgboost --output models/

train-all:  ## Train all models
	$(PYTHON_INTERPRETER) src/models/train_model.py --model logistic --output models/
	$(PYTHON_INTERPRETER) src/models/train_model.py --model random_forest --output models/
	$(PYTHON_INTERPRETER) src/models/train_model.py --model xgboost --output models/
	$(PYTHON_INTERPRETER) src/models/train_model.py --model lightgbm --output models/

train-cv:  ## Train with cross-validation
	$(PYTHON_INTERPRETER) src/models/train_model.py --model xgboost --cv 5 --output models/

hyperparameter-tuning:  ## Run hyperparameter optimization
	$(PYTHON_INTERPRETER) src/models/optimize_hyperparameters.py --model xgboost --trials 50

evaluate:  ## Evaluate trained model
	$(PYTHON_INTERPRETER) src/models/evaluate_model.py models/xgboost_model.pkl data/processed/test.csv

predict:  ## Make predictions on new data
	$(PYTHON_INTERPRETER) src/models/predict_model.py models/xgboost_model.pkl data/processed/test.csv

##@ Visualization

visualize:  ## Generate all visualizations
	$(PYTHON_INTERPRETER) src/visualization/visualize.py

eda-report:  ## Generate EDA HTML report
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --execute notebooks/1.0-eda.ipynb --to html --output-dir reports/

##@ Testing

test:  ## Run pytest tests
	pytest tests/ -v

test-coverage:  ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term tests/

test-integration:  ## Run integration tests only
	pytest tests/integration/ -v

##@ Code Quality

lint:  ## Lint code with flake8
	flake8 src/ tests/

format:  ## Format code with black
	black src/ tests/

format-check:  ## Check code formatting
	black --check src/ tests/

type-check:  ## Type checking with mypy
	mypy src/

quality:  ## Run all quality checks
	make format-check
	make lint
	make type-check

##@ AWS Deployment

aws-configure:  ## Configure AWS credentials
	aws configure

sagemaker-deploy:  ## Deploy model to SageMaker
	$(PYTHON_INTERPRETER) src/deployment/sagemaker_deploy.py

sagemaker-test:  ## Test SageMaker endpoint
	$(PYTHON_INTERPRETER) src/deployment/test_endpoint.py

sagemaker-delete:  ## Delete SageMaker endpoint
	$(PYTHON_INTERPRETER) src/deployment/delete_endpoint.py

lambda-deploy:  ## Deploy Lambda function
	cd src/deployment && ./deploy_lambda.sh

api-test:  ## Test API endpoint
	$(PYTHON_INTERPRETER) src/deployment/test_api.py

##@ Documentation

docs:  ## Generate Sphinx documentation
	cd docs && make html
	@echo "Documentation generated in docs/_build/html/index.html"

docs-serve:  ## Serve documentation locally
	cd docs/_build/html && $(PYTHON_INTERPRETER) -m http.server 8000

notebook-docs:  ## Convert notebooks to HTML
	jupyter nbconvert --to html notebooks/*.ipynb --output-dir reports/

##@ Docker

docker-build:  ## Build Docker image
	docker build -t banking-ml-project:latest .

docker-run:  ## Run Docker container
	docker run -p 5000:5000 banking-ml-project:latest

docker-push:  ## Push to Docker Hub
	docker tag banking-ml-project:latest yourusername/banking-ml-project:latest
	docker push yourusername/banking-ml-project:latest

##@ Cleaning

clean:  ## Remove build artifacts
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

clean-data:  ## Remove processed data
	rm -rf data/interim/*
	rm -rf data/processed/*

clean-models:  ## Remove trained models
	rm -rf models/*

clean-reports:  ## Remove generated reports
	rm -rf reports/figures/*
	rm -rf reports/*.html

clean-all:  ## Remove all generated files
	make clean
	make clean-data
	make clean-models
	make clean-reports
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage

##@ Complete Workflows

all:  ## Run complete ML pipeline
	make data
	make features
	make train
	make evaluate
	make visualize

ci:  ## Run CI pipeline (test + quality)
	make test
	make quality

deploy:  ## Complete deployment pipeline
	make train
	make evaluate
	make sagemaker-deploy

##@ Info

show-structure:  ## Show project structure
	tree -L 3 -I '__pycache__|*.pyc|.git'

show-models:  ## List trained models
	ls -lh models/

show-data:  ## Show data files
	du -sh data/*/
	@echo "\nData files:"
	find data/ -type f -name "*.csv" -o -name "*.pkl"

info:  ## Show project information
	@echo "Python version: $$($(PYTHON_INTERPRETER) --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Project path: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'N/A')"
