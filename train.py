"""Training script for fake news detection model"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import TextPreprocessor
from src.model_trainer import FakeNewsClassifier, train_and_compare_models
import os

def load_sample_data():
    """
    Load or create sample data for demonstration
    
    Returns:
        pd.DataFrame: Sample dataset
    """
    # Create sample fake news dataset
    data = {
        'text': [
            'Scientists discover cure for all diseases in laboratory experiment',
            'Breaking: Aliens land on Earth and meet with world leaders',
            'Study shows drinking water is beneficial for health',
            'Research confirms exercise improves cardiovascular health',
            'President claims to have superpowers and can fly',
            'New study published in medical journal shows vaccine effectiveness',
            'Miracle pill will make you lose 50 pounds in one week',
            'Climate change effects observed by researchers worldwide',
            'Celebrity arrested for crimes they never committed',
            'Economic report shows unemployment rate declining',
            'Government secretly controls the weather with machines',
            'New smartphone model released with improved features',
            'Doctor warns about health risks of smoking cigarettes',
            'Moon landing was fake and filmed in Hollywood studio',
            'Scientists develop new renewable energy technology',
            'Eating this one food will cure cancer instantly',
            'University research reveals benefits of education',
            'Politician promises to eliminate all taxes forever',
            'Historical documents discovered in ancient library',
            'Drinking bleach cures coronavirus says fake doctor'
        ],
        'label': [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
        # 0 = Real, 1 = Fake
    }
    
    return pd.DataFrame(data)

def main(args):
    """Main training function"""
    print("=== Fake News Detection Model Training ===\n")
    
    # Load data
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading data from {args.data_path}...")
        df = pd.read_csv(args.data_path)
        text_column = args.text_column
        label_column = args.label_column
    else:
        print("Loading sample data...")
        df = load_sample_data()
        text_column = 'text'
        label_column = 'label'
    
    print(f"Dataset size: {len(df)} samples")
    print(f"Fake news: {sum(df[label_column])} samples")
    print(f"Real news: {len(df) - sum(df[label_column])} samples\n")
    
    # Preprocess data
    print("Preprocessing text data...")
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df, text_column, label_column)
    
    # Split data
    X = df['processed_text']
    y = df[label_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    if args.compare_models:
        # Train and compare multiple models
        print("Training and comparing multiple models...")
        results = train_and_compare_models(X_train, y_train, X_test, y_test)
        
        # Find best model
        best_model_type = max(results.keys(), 
                            key=lambda k: results[k]['eval_metrics']['accuracy'])
        best_classifier = results[best_model_type]['model']
        best_accuracy = results[best_model_type]['eval_metrics']['accuracy']
        
        print(f"\n=== Best Model: {best_model_type} ===")
        print(f"Test Accuracy: {best_accuracy:.4f}")
        print("\nClassification Report:")
        print(results[best_model_type]['eval_metrics']['classification_report'])
    else:
        # Train single model
        print(f"Training {args.model_type} model...")
        best_classifier = FakeNewsClassifier(model_type=args.model_type)
        
        train_metrics = best_classifier.train(X_train, y_train, X_test, y_test)
        print(f"Training Accuracy: {train_metrics['train_accuracy']:.4f}")
        print(f"Validation Accuracy: {train_metrics['val_accuracy']:.4f}")
        
        eval_metrics = best_classifier.evaluate(X_test, y_test)
        print(f"\nTest Accuracy: {eval_metrics['accuracy']:.4f}")
        print("\nClassification Report:")
        print(eval_metrics['classification_report'])
    
    # Save model
    if args.save_model:
        model_path = os.path.join(args.model_dir, 'model.pkl')
        vectorizer_path = os.path.join(args.model_dir, 'vectorizer.pkl')
        
        print(f"\nSaving model to {args.model_dir}...")
        best_classifier.save(model_path, vectorizer_path)
        print("Model saved successfully!")
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train fake news detection model')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to training data CSV file')
    parser.add_argument('--text-column', type=str, default='text',
                       help='Name of text column in CSV')
    parser.add_argument('--label-column', type=str, default='label',
                       help='Name of label column in CSV')
    parser.add_argument('--model-type', type=str, default='logistic',
                       choices=['logistic', 'naive_bayes', 'random_forest', 'svm'],
                       help='Type of model to train')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0-1)')
    parser.add_argument('--compare-models', action='store_true',
                       help='Train and compare multiple models')
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='Save trained model')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save model')
    
    args = parser.parse_args()
    main(args)
