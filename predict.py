"""Prediction script for fake news detection"""
import argparse
import os
from src.preprocessing import TextPreprocessor
from src.model_trainer import FakeNewsClassifier

def main(args):
    """Main prediction function"""
    print("=== Fake News Detection Prediction ===\n")
    
    # Load model
    model_path = os.path.join(args.model_dir, 'model.pkl')
    vectorizer_path = os.path.join(args.model_dir, 'vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Model not found. Please train a model first using train.py")
        return
    
    print("Loading model...")
    classifier = FakeNewsClassifier()
    classifier.load(model_path, vectorizer_path)
    
    # Preprocess input
    preprocessor = TextPreprocessor()
    
    if args.text:
        # Single text prediction
        texts = [args.text]
    elif args.file:
        # Read from file
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Please provide either --text or --file argument")
        return
    
    # Make predictions
    print("\nMaking predictions...\n")
    
    for i, text in enumerate(texts, 1):
        processed_text = preprocessor.preprocess(text)
        
        if not processed_text:
            print(f"Text {i}: [Empty after preprocessing]")
            continue
        
        prediction = classifier.predict([processed_text])[0]
        proba = classifier.predict_proba([processed_text])[0]
        
        label = "FAKE" if prediction == 1 else "REAL"
        confidence = proba[prediction] * 100
        
        print(f"Text {i}:")
        print(f"  Original: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"  Prediction: {label}")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Probabilities: Real={proba[0]:.2f}, Fake={proba[1]:.2f}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict fake news')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to classify')
    parser.add_argument('--file', type=str, default=None,
                       help='File containing texts to classify (one per line)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing trained model')
    
    args = parser.parse_args()
    main(args)
