"""Data preprocessing module for fake news detection"""
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class TextPreprocessor:
    """Text preprocessing utilities for fake news detection"""
    
    def __init__(self):
        """Initialize preprocessor with NLTK resources"""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def clean_text(self, text):
        """
        Clean and preprocess text data
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without stopwords
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def stem_text(self, text):
        """
        Apply stemming to text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Stemmed text
        """
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def preprocess(self, text, remove_stopwords=True, apply_stemming=True):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stopwords
            apply_stemming (bool): Whether to apply stemming
            
        Returns:
            str: Fully preprocessed text
        """
        text = self.clean_text(text)
        
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        if apply_stemming:
            text = self.stem_text(text)
        
        return text
    
    def preprocess_dataframe(self, df, text_column, target_column=None):
        """
        Preprocess entire dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column to preprocess
            target_column (str): Name of target column (optional)
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df = df.copy()
        df['processed_text'] = df[text_column].apply(self.preprocess)
        
        # Remove rows with empty processed text
        df = df[df['processed_text'].str.len() > 0]
        
        if target_column:
            # Ensure target column exists and is properly formatted
            df = df[df[target_column].notna()]
        
        return df
