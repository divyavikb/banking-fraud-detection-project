from setuptools import find_packages, setup

setup(
    name='banking_ml_project',
    packages=find_packages(),
    version='0.1.0',
    description='Credit Risk Prediction ML Project for Banking Domain',
    author='Your Name',
    license='MIT',
    install_requires=[
        'numpy>=1.24.3',
        'pandas>=2.0.3',
        'scikit-learn>=1.3.0',
        'xgboost>=1.7.6',
        'lightgbm>=4.0.0',
        'imbalanced-learn>=0.11.0',
        'shap>=0.42.1',
        'matplotlib>=3.7.2',
        'seaborn>=0.12.2',
        'boto3>=1.28.25',
        'sagemaker>=2.175.0',
        'flask>=2.3.2',
        'python-dotenv>=1.0.0',
        'pyyaml>=6.0.1',
        'click>=8.1.6',
        'tqdm>=4.65.0',
        'joblib>=1.3.1',
        'loguru>=0.7.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.0.0',
            'mypy>=1.4.1',
        ],
        'docs': [
            'sphinx>=7.1.0',
            'sphinx-rtd-theme>=1.2.2',
        ],
    },
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'train-model=src.models.train_model:main',
            'predict=src.models.predict_model:main',
            'build-features=src.features.build_features:main',
        ],
    },
)
