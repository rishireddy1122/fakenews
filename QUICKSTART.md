# Quick Start Guide

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/rishireddy1122/fakenews.git
cd fakenews

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"
```

## 2. Train Your First Model

```bash
# Train with sample data (included)
python train.py --compare-models --save-model
```

This will:
- Load sample fake news data
- Train 4 different models
- Compare their performance
- Save the best model to `models/`

## 3. Make Predictions

```bash
# Predict from command line
python predict.py --text "Your news article here"
```

## 4. Start the Web API

```bash
# Start the Flask server
python deployment/app.py
```

Visit http://localhost:5000 in your browser to use the web interface!

## 5. Deploy with Docker

```bash
# Build and run with Docker
docker-compose up
```

## API Usage Examples

### Using curl:

```bash
# Single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news text here"}'

# Batch prediction
curl -X POST http://localhost:5000/api/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2"]}'
```

### Using Python:

```python
import requests

response = requests.post(
    'http://localhost:5000/api/predict',
    json={'text': 'Your news article here'}
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

## Using Your Own Data

Prepare a CSV file with columns: `text` and `label` (0=Real, 1=Fake)

```bash
python train.py \
  --data-path data/your_data.csv \
  --text-column text \
  --label-column label \
  --compare-models \
  --save-model
```

## Next Steps

1. Collect more training data for better accuracy
2. Try different model types and hyperparameters
3. Deploy to cloud (AWS, GCP, Azure, Heroku)
4. Integrate with your application via REST API

For more details, see the main README.md
