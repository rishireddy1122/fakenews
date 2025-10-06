"""
Script to train a logistic regression model for fake news detection.
"""

import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def train_logistic_regression():
    """
    Train a logistic regression model on the preprocessed data.
    """
    print("Loading preprocessed data...")
    
    # Load training data
    with open('models/train_data.pkl', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    # Load test data
    with open('models/test_data.pkl', 'rb') as f:
        X_test, y_test = pickle.load(f)
    
    print("Training Logistic Regression model...")
    
    # Initialize and train model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='liblinear'
    )
    
    model.fit(X_train, y_train)
    
    print("Model training complete!")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Real News', 'Fake News']))
    
    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Save the trained model
    with open('models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\nModel saved to 'models/logistic_regression_model.pkl'")
    
    return model, train_accuracy, test_accuracy

if __name__ == "__main__":
    train_logistic_regression()
