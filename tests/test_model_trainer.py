"""Tests for model trainer module"""
import pytest
import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_trainer import FakeNewsClassifier

class TestFakeNewsClassifier:
    """Test FakeNewsClassifier class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        X_train = [
            "this is fake news",
            "this is real news",
            "fake story about aliens",
            "real report on climate",
            "fake celebrity rumor",
            "real scientific study"
        ]
        y_train = np.array([1, 0, 1, 0, 1, 0])
        
        X_test = [
            "fake news article",
            "real news article"
        ]
        y_test = np.array([1, 0])
        
        return X_train, y_train, X_test, y_test
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        classifier = FakeNewsClassifier(model_type='logistic')
        assert classifier.model_type == 'logistic'
        assert not classifier.is_fitted
    
    def test_train_model(self, sample_data):
        """Test model training"""
        X_train, y_train, X_test, y_test = sample_data
        
        classifier = FakeNewsClassifier(model_type='logistic')
        metrics = classifier.train(X_train, y_train, X_test, y_test)
        
        assert classifier.is_fitted
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' in metrics
        assert 0 <= metrics['train_accuracy'] <= 1
        assert 0 <= metrics['val_accuracy'] <= 1
    
    def test_predict(self, sample_data):
        """Test prediction"""
        X_train, y_train, X_test, y_test = sample_data
        
        classifier = FakeNewsClassifier(model_type='logistic')
        classifier.train(X_train, y_train)
        
        predictions = classifier.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction"""
        X_train, y_train, X_test, y_test = sample_data
        
        classifier = FakeNewsClassifier(model_type='logistic')
        classifier.train(X_train, y_train)
        
        probabilities = classifier.predict_proba(X_test)
        
        assert probabilities.shape == (len(X_test), 2)
        assert all(0 <= p <= 1 for row in probabilities for p in row)
        assert all(abs(sum(row) - 1.0) < 0.01 for row in probabilities)
    
    def test_evaluate(self, sample_data):
        """Test model evaluation"""
        X_train, y_train, X_test, y_test = sample_data
        
        classifier = FakeNewsClassifier(model_type='logistic')
        classifier.train(X_train, y_train)
        
        metrics = classifier.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'classification_report' in metrics
        assert 'confusion_matrix' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_save_load_model(self, sample_data):
        """Test model saving and loading"""
        X_train, y_train, X_test, y_test = sample_data
        
        # Train model
        classifier = FakeNewsClassifier(model_type='logistic')
        classifier.train(X_train, y_train)
        
        # Make prediction with original model
        original_pred = classifier.predict(X_test)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pkl')
            vectorizer_path = os.path.join(tmpdir, 'vectorizer.pkl')
            
            classifier.save(model_path, vectorizer_path)
            
            # Load model
            new_classifier = FakeNewsClassifier()
            new_classifier.load(model_path, vectorizer_path)
            
            # Make prediction with loaded model
            loaded_pred = new_classifier.predict(X_test)
            
            # Predictions should be the same
            assert np.array_equal(original_pred, loaded_pred)
    
    def test_different_model_types(self, sample_data):
        """Test different model types"""
        X_train, y_train, X_test, y_test = sample_data
        
        model_types = ['logistic', 'naive_bayes', 'random_forest', 'svm']
        
        for model_type in model_types:
            classifier = FakeNewsClassifier(model_type=model_type)
            metrics = classifier.train(X_train, y_train)
            
            assert classifier.is_fitted
            assert 'train_accuracy' in metrics
    
    def test_predict_without_training(self):
        """Test that prediction fails without training"""
        classifier = FakeNewsClassifier()
        
        with pytest.raises(ValueError):
            classifier.predict(["test"])
