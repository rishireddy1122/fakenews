"""
Main script to run the complete pipeline:
1. Collect data
2. Preprocess data
3. Train model
4. Make predictions
"""

from collect_data import collect_data
from preprocess_data import preprocess_data
from train_model import train_logistic_regression
from predict import predict_news

def run_complete_pipeline():
    """
    Run the complete fake news detection pipeline.
    """
    print("="*60)
    print("FAKE NEWS DETECTION PIPELINE")
    print("="*60)
    
    # Step 1: Collect data
    print("\n[Step 1/4] Collecting data...")
    print("-"*60)
    collect_data()
    
    # Step 2: Preprocess data
    print("\n[Step 2/4] Preprocessing data...")
    print("-"*60)
    preprocess_data()
    
    # Step 3: Train model
    print("\n[Step 3/4] Training model...")
    print("-"*60)
    train_logistic_regression()
    
    # Step 4: Test predictions
    print("\n[Step 4/4] Testing predictions...")
    print("-"*60)
    
    sample_texts = [
        "Breaking news: miracle cure discovered",
        "Government releases annual economic report"
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}: {text}")
        label, confidence = predict_news(text)
        print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nTrained model saved in 'models/' directory")
    print("You can use 'predict.py' to make predictions on new texts")

if __name__ == "__main__":
    run_complete_pipeline()
