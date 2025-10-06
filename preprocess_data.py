"""
Script to preprocess the fake news dataset.
This includes text cleaning, tokenization, and feature extraction.
"""

import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords

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

def preprocess_data(input_path='data/fake_news_data.csv'):
    """
    Preprocess the dataset and create train/test splits.
    
    Args:
        input_path (str): Path to input CSV file
    """
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Split data
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Extracting features using TF-IDF...")
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    # Save vectorizer
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save processed data
    with open('models/train_data.pkl', 'wb') as f:
        pickle.dump((X_train_tfidf, y_train), f)
    
    with open('models/test_data.pkl', 'wb') as f:
        pickle.dump((X_test_tfidf, y_test), f)
    
    print(f"Preprocessing complete!")
    print(f"Training samples: {X_train_tfidf.shape[0]}")
    print(f"Test samples: {X_test_tfidf.shape[0]}")
    print(f"Number of features: {X_train_tfidf.shape[1]}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
