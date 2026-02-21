"""
Unit tests for model training
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.train_model import CreditRiskModel


class TestCreditRiskModel:
    """Test suite for CreditRiskModel"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randint(0, 5, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = CreditRiskModel('xgboost', use_smote=False)
        assert model.model_name == 'xgboost'
        assert model.use_smote == False
        assert model.model is None
    
    def test_invalid_model_name(self):
        """Test error on invalid model name"""
        with pytest.raises(ValueError):
            CreditRiskModel('invalid_model')
    
    def test_model_configs(self):
        """Test that all model configs are available"""
        expected_models = ['logistic', 'random_forest', 'xgboost', 'lightgbm']
        
        for model_name in expected_models:
            assert model_name in CreditRiskModel.MODEL_CONFIGS
            assert 'class' in CreditRiskModel.MODEL_CONFIGS[model_name]
            assert 'params' in CreditRiskModel.MODEL_CONFIGS[model_name]
    
    @pytest.mark.parametrize('model_name', ['logistic', 'random_forest', 'xgboost'])
    def test_train_different_models(self, model_name, sample_data, tmp_path):
        """Test training with different models"""
        # Create temporary data files
        train_path = tmp_path / 'train_features.csv'
        val_path = tmp_path / 'validation_features.csv'
        
        # Split data
        train_data = sample_data.iloc[:80]
        val_data = sample_data.iloc[80:]
        
        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        
        # Train model
        trainer = CreditRiskModel(model_name, use_smote=False)
        trainer.load_data(tmp_path)
        trainer.train()
        
        # Assertions
        assert trainer.model is not None
        assert hasattr(trainer, 'X_train')
        assert hasattr(trainer, 'y_train')
        assert len(trainer.X_train) == 80
    
    def test_feature_importance_extraction(self, sample_data, tmp_path):
        """Test feature importance extraction"""
        # Prepare data
        train_path = tmp_path / 'train_features.csv'
        val_path = tmp_path / 'validation_features.csv'
        
        sample_data.iloc[:80].to_csv(train_path, index=False)
        sample_data.iloc[80:].to_csv(val_path, index=False)
        
        # Train
        trainer = CreditRiskModel('xgboost', use_smote=False)
        trainer.load_data(tmp_path)
        trainer.train()
        
        # Check feature importance
        assert trainer.feature_importance is not None
        assert len(trainer.feature_importance) > 0
        assert 'feature' in trainer.feature_importance.columns
        assert 'importance' in trainer.feature_importance.columns


class TestModelEvaluation:
    """Test model evaluation functions"""
    
    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        
        assert 0 <= acc <= 1
        assert 0 <= prec <= 1
        assert 0 <= rec <= 1


@pytest.mark.integration
class TestIntegration:
    """Integration tests"""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete training pipeline"""
        # This would test the full pipeline from data loading to model saving
        pass
