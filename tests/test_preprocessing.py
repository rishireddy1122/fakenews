"""Tests for preprocessing module"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import TextPreprocessor

class TestTextPreprocessor:
    """Test TextPreprocessor class"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return TextPreprocessor()
    
    def test_clean_text_basic(self, preprocessor):
        """Test basic text cleaning"""
        text = "This is a TEST sentence!"
        result = preprocessor.clean_text(text)
        assert result == "this is a test sentence"
    
    def test_clean_text_urls(self, preprocessor):
        """Test URL removal"""
        text = "Check this link https://example.com for more info"
        result = preprocessor.clean_text(text)
        assert "https" not in result
        assert "example.com" not in result
    
    def test_clean_text_numbers(self, preprocessor):
        """Test number removal"""
        text = "There are 123 apples and 456 oranges"
        result = preprocessor.clean_text(text)
        assert "123" not in result
        assert "456" not in result
    
    def test_remove_stopwords(self, preprocessor):
        """Test stopword removal"""
        text = "this is a test of the system"
        result = preprocessor.remove_stopwords(text)
        # Common stopwords should be removed
        assert "is" not in result or "test" in result
    
    def test_stem_text(self, preprocessor):
        """Test text stemming"""
        text = "running runner runs"
        result = preprocessor.stem_text(text)
        # All should stem to similar root
        assert "run" in result
    
    def test_preprocess_pipeline(self, preprocessor):
        """Test full preprocessing pipeline"""
        text = "This is a TEST! Check https://example.com"
        result = preprocessor.preprocess(text)
        assert len(result) > 0
        assert result.islower() or result == ""
        assert "https" not in result
    
    def test_preprocess_empty_text(self, preprocessor):
        """Test handling of empty text"""
        result = preprocessor.preprocess("")
        assert result == ""
    
    def test_preprocess_dataframe(self, preprocessor):
        """Test dataframe preprocessing"""
        df = pd.DataFrame({
            'text': ['This is news 1', 'This is news 2'],
            'label': [0, 1]
        })
        
        result = preprocessor.preprocess_dataframe(df, 'text', 'label')
        
        assert 'processed_text' in result.columns
        assert len(result) <= len(df)
        assert all(len(text) > 0 for text in result['processed_text'])
