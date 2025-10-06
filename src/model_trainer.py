"""Model training module for fake news detection"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class FakeNewsClassifier:
    """Fake news classifier with multiple model options"""
    
    def __init__(self, model_type='logistic', max_features=5000):
        """
        Initialize classifier
        
        Args:
            model_type (str): Type of model ('logistic', 'naive_bayes', 'random_forest', 'svm')
            max_features (int): Maximum number of features for TF-IDF
        """
        self.model_type = model_type
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.model = self._get_model(model_type)
        self.is_fitted = False
    
    def _get_model(self, model_type):
        """Get model based on type"""
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': LinearSVC(max_iter=1000, random_state=42)
        }
        return models.get(model_type, LogisticRegression(max_iter=1000, random_state=42))
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data (optional)
            y_val: Validation labels (optional)
            
        Returns:
            dict: Training metrics
        """
        # Vectorize training data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        self.is_fitted = True
        
        # Calculate training accuracy
        train_pred = self.model.predict(X_train_vec)
        train_acc = accuracy_score(y_train, train_pred)
        
        metrics = {
            'train_accuracy': train_acc,
            'model_type': self.model_type
        }
        
        # Validate if validation data provided
        if X_val is not None and y_val is not None:
            X_val_vec = self.vectorizer.transform(X_val)
            val_pred = self.model.predict(X_val_vec)
            val_acc = accuracy_score(y_val, val_pred)
            metrics['val_accuracy'] = val_acc
            metrics['classification_report'] = classification_report(y_val, val_pred)
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Text data to predict
            
        Returns:
            np.array: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Text data to predict
            
        Returns:
            np.array: Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_vec = self.vectorizer.transform(X)
        
        # Check if model supports predict_proba
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_vec)
        else:
            # For models without predict_proba, return binary predictions
            predictions = self.model.predict(X_vec)
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions] = 1
            return proba
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test text data
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
        }
        
        return metrics
    
    def save(self, model_path, vectorizer_path):
        """
        Save model and vectorizer
        
        Args:
            model_path (str): Path to save model
            vectorizer_path (str): Path to save vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
    
    def load(self, model_path, vectorizer_path):
        """
        Load model and vectorizer
        
        Args:
            model_path (str): Path to load model from
            vectorizer_path (str): Path to load vectorizer from
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.is_fitted = True

def train_and_compare_models(X_train, y_train, X_test, y_test):
    """
    Train and compare multiple models
    
    Args:
        X_train: Training text data
        y_train: Training labels
        X_test: Test text data
        y_test: Test labels
        
    Returns:
        dict: Results for all models
    """
    model_types = ['logistic', 'naive_bayes', 'random_forest', 'svm']
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        classifier = FakeNewsClassifier(model_type=model_type)
        
        # Train
        train_metrics = classifier.train(X_train, y_train, X_test, y_test)
        
        # Evaluate
        eval_metrics = classifier.evaluate(X_test, y_test)
        
        results[model_type] = {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'model': classifier
        }
        
        print(f"{model_type} - Test Accuracy: {eval_metrics['accuracy']:.4f}")
    
    return results
