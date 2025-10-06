# Fake News Detection - End-to-End ML Model

A complete end-to-end machine learning project for detecting fake news using Natural Language Processing (NLP) and various classification algorithms. This project includes data preprocessing, model training, evaluation, and deployment capabilities.

## Features

- **Multiple ML Models**: Logistic Regression, Naive Bayes, Random Forest, and SVM
- **Text Preprocessing**: Advanced NLP preprocessing including cleaning, stopword removal, and stemming
- **Model Comparison**: Compare multiple models to find the best performer
- **REST API**: Flask-based API for real-time predictions
- **Web Interface**: User-friendly web interface for testing predictions
- **Docker Support**: Containerized deployment for easy scaling
- **Comprehensive Testing**: Unit tests for core components

## Project Structure

```
fakenews/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Text preprocessing utilities
│   └── model_trainer.py      # Model training and evaluation
├── deployment/
│   └── app.py               # Flask API application
├── tests/
│   ├── test_preprocessing.py
│   └── test_model_trainer.py
├── models/                  # Saved models (created after training)
├── data/                    # Dataset directory
├── train.py                 # Training script
├── predict.py               # Prediction script
├── notebook_example.ipynb   # Jupyter notebook example
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
└── README.md               # This file
```

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/rishireddy1122/fakenews.git
cd fakenews
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```bash
python -c "import nltk; nltk.download('stopwords')"
```

### Docker Installation

```bash
docker-compose up --build
```

## Quick Start

### 1. Train a Model

Train with sample data:
```bash
python train.py --compare-models --save-model
```

Train with your own data:
```bash
python train.py --data-path data/your_data.csv --text-column text --label-column label --save-model
```

Options:
- `--data-path`: Path to your CSV file
- `--text-column`: Name of the text column
- `--label-column`: Name of the label column (0=Real, 1=Fake)
- `--model-type`: Choose from `logistic`, `naive_bayes`, `random_forest`, `svm`
- `--compare-models`: Compare all models and save the best one
- `--test-size`: Test set proportion (default: 0.2)
- `--save-model`: Save the trained model
- `--model-dir`: Directory to save model (default: models)

### 2. Make Predictions

Predict a single text:
```bash
python predict.py --text "Breaking news: Scientists discover amazing new technology"
```

Predict from a file:
```bash
python predict.py --file data/test_texts.txt
```

### 3. Start the API Server

```bash
python deployment/app.py
```

The API will be available at `http://localhost:5000`

#### API Endpoints

- **Web Interface**: `GET /` - Interactive web interface
- **Health Check**: `GET /api/health` - Check API status
- **Single Prediction**: `POST /api/predict`
  ```json
  {
    "text": "Your news text here"
  }
  ```
  
- **Batch Prediction**: `POST /api/batch-predict`
  ```json
  {
    "texts": ["Text 1", "Text 2", "Text 3"]
  }
  ```

#### Example API Usage

```bash
# Single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Aliens land on Earth"}'

# Batch prediction
curl -X POST http://localhost:5000/api/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Real news text", "Fake news text"]}'
```

### 4. Using Docker

Build and run with Docker:
```bash
# Build the image
docker build -t fakenews-api .

# Run the container
docker run -p 5000:5000 -v $(pwd)/models:/app/models fakenews-api
```

Or use Docker Compose:
```bash
docker-compose up
```

## Data Format

Your dataset should be a CSV file with at least two columns:
- **Text column**: Contains the news article text
- **Label column**: Contains labels (0 for real news, 1 for fake news)

Example:
```csv
text,label
"Scientists discover new species in Amazon rainforest",0
"Miracle cure will solve all health problems instantly",1
```

## Model Performance

The project supports multiple models:

| Model | Typical Accuracy | Speed | Memory |
|-------|------------------|-------|--------|
| Logistic Regression | 85-92% | Fast | Low |
| Naive Bayes | 80-88% | Very Fast | Low |
| Random Forest | 85-90% | Medium | Medium |
| SVM | 86-93% | Medium | Medium |

*Note: Actual performance depends on your dataset*

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Using Jupyter Notebook

Open the example notebook:
```bash
jupyter notebook notebook_example.ipynb
```

The notebook demonstrates:
- Data loading and exploration
- Text preprocessing
- Model training
- Making predictions
- Model evaluation

## Deployment

### Production Deployment

For production deployment, consider:

1. **Using Gunicorn** (already configured in Dockerfile):
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 deployment.app:app
```

2. **Environment Variables**:
```bash
export FLASK_ENV=production
export MODEL_DIR=/path/to/models
```

3. **Reverse Proxy**: Use Nginx or similar for handling SSL and load balancing

4. **Cloud Deployment**: Deploy to AWS, GCP, Azure, or Heroku

Example for Heroku:
```bash
heroku create your-app-name
git push heroku main
```

## Model Improvement Tips

1. **Use More Data**: The model improves with more training examples
2. **Feature Engineering**: Experiment with different text features (TF-IDF, word embeddings)
3. **Hyperparameter Tuning**: Use GridSearchCV for optimal parameters
4. **Ensemble Methods**: Combine multiple models
5. **Deep Learning**: Try LSTM or BERT models for better accuracy

## Troubleshooting

### Model Not Found Error
```bash
# Make sure you've trained a model first
python train.py --save-model
```

### NLTK Data Not Found
```bash
python -c "import nltk; nltk.download('stopwords')"
```

### Port Already in Use
```bash
# Use a different port
python deployment/app.py --port 5001
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- NLTK for NLP preprocessing
- Scikit-learn for ML algorithms
- Flask for web framework
- The open-source community

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a demonstration project. For production use with real news detection, you would need:
- A larger, high-quality labeled dataset
- More sophisticated features and models
- Regular retraining with new data
- Careful consideration of bias and ethical implications
