"""
Script to make predictions on new text using the trained model.
"""

import pickle
import re

def clean_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def predict_news(text):
    """
    Predict whether a news article is fake or real.
    
    Args:
        text (str): News article text
        
    Returns:
        tuple: (prediction, probability)
    """
    # Load the trained model
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load the vectorizer
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Preprocess the text
    cleaned = clean_text(text)
    
    # Vectorize the text
    text_vectorized = vectorizer.transform([cleaned])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    # Convert prediction to label
    label = "Fake News" if prediction == 1 else "Real News"
    confidence = probability[prediction] * 100
    
    return label, confidence

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "Scientists discover cure for all diseases overnight",
        "Local government announces new infrastructure project",
        "Aliens confirmed to be living among us says anonymous source"
    ]
    
    print("Making predictions on sample texts:\n")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"Text {i}: {text}")
        label, confidence = predict_news(text)
        print(f"Prediction: {label} (Confidence: {confidence:.2f}%)\n")
