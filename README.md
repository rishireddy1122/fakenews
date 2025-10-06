# Fake News Detection using Logistic Regression

This project implements a fake news detection system using logistic regression. The model is trained to classify news articles as either fake or real based on text content analysis.

## Features

- Data collection and preprocessing
- Text feature extraction using TF-IDF
- Logistic regression model training
- Model evaluation with accuracy metrics
- Prediction interface for new text

## Project Structure

```
fakenews/
├── collect_data.py      # Script to collect/generate fake news dataset
├── preprocess_data.py   # Text preprocessing and feature extraction
├── train_model.py       # Logistic regression model training
├── predict.py           # Make predictions on new text
├── run_pipeline.py      # Run complete pipeline
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rishireddy1122/fakenews.git
cd fakenews
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Complete Pipeline

Run the entire pipeline (data collection, preprocessing, training, and testing):

```bash
python run_pipeline.py
```

### Step-by-Step Execution

1. **Collect Data:**
```bash
python collect_data.py
```

2. **Preprocess Data:**
```bash
python preprocess_data.py
```

3. **Train Model:**
```bash
python train_model.py
```

4. **Make Predictions:**
```bash
python predict.py
```

## Model Details

- **Algorithm:** Logistic Regression
- **Features:** TF-IDF vectors with 1000 max features
- **N-grams:** Unigrams and bigrams
- **Train/Test Split:** 80/20

## Output

The trained model and preprocessing artifacts are saved in the `models/` directory:
- `logistic_regression_model.pkl` - Trained model
- `vectorizer.pkl` - TF-IDF vectorizer
- `train_data.pkl` - Processed training data
- `test_data.pkl` - Processed test data

## Performance

The model provides:
- Training and test accuracy metrics
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Confidence scores for predictions

## Example

```python
from predict import predict_news

text = "Scientists discover cure for all diseases overnight"
label, confidence = predict_news(text)
print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn

## License

This project is for educational purposes to practice machine learning models and prediction efficiency.
