"""Flask API for fake news detection service"""
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import TextPreprocessor
from src.model_trainer import FakeNewsClassifier

app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessor
model = None
preprocessor = None

def load_model():
    """Load the trained model"""
    global model, preprocessor
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(model_dir, 'model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model files not found. Please train a model first.")
    
    model = FakeNewsClassifier()
    model.load(model_path, vectorizer_path)
    preprocessor = TextPreprocessor()
    
    print("Model loaded successfully!")

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            box-sizing: border-box;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 5px;
            display: none;
        }
        .fake {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
        }
        .real {
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
        }
        .result h2 {
            margin-top: 0;
        }
        .confidence {
            font-size: 18px;
            font-weight: bold;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Fake News Detector</h1>
        <p>Enter a news article or statement to check if it's likely to be fake or real:</p>
        
        <textarea id="newsText" placeholder="Enter news text here..."></textarea>
        <button onclick="detectFakeNews()">Analyze</button>
        
        <div class="loading" id="loading">Analyzing...</div>
        
        <div class="result" id="result">
            <h2 id="resultLabel"></h2>
            <p class="confidence" id="confidence"></p>
            <p id="probabilities"></p>
        </div>
    </div>
    
    <script>
        async function detectFakeNews() {
            const text = document.getElementById('newsText').value;
            
            if (!text.trim()) {
                alert('Please enter some text to analyze');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                if (data.success) {
                    // Show result
                    const resultDiv = document.getElementById('result');
                    const isFake = data.prediction === 'FAKE';
                    
                    resultDiv.className = 'result ' + (isFake ? 'fake' : 'real');
                    resultDiv.style.display = 'block';
                    
                    document.getElementById('resultLabel').textContent = 
                        isFake ? '⚠️ Likely FAKE News' : '✅ Likely REAL News';
                    document.getElementById('confidence').textContent = 
                        'Confidence: ' + data.confidence.toFixed(2) + '%';
                    document.getElementById('probabilities').textContent = 
                        'Probability of Real: ' + (data.probabilities.real * 100).toFixed(2) + '% | ' +
                        'Probability of Fake: ' + (data.probabilities.fake * 100).toFixed(2) + '%';
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                alert('Error analyzing text: ' + error);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render home page with web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({
                'success': False,
                'error': 'Empty text provided'
            }), 400
        
        # Preprocess text
        processed_text = preprocessor.preprocess(text)
        
        if not processed_text:
            return jsonify({
                'success': False,
                'error': 'Text became empty after preprocessing'
            }), 400
        
        # Make prediction
        prediction = model.predict([processed_text])[0]
        proba = model.predict_proba([processed_text])[0]
        
        label = "FAKE" if prediction == 1 else "REAL"
        confidence = proba[prediction] * 100
        
        return jsonify({
            'success': True,
            'prediction': label,
            'confidence': float(confidence),
            'probabilities': {
                'real': float(proba[0]),
                'fake': float(proba[1])
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'No texts provided'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'success': False,
                'error': 'texts must be a list'
            }), 400
        
        results = []
        
        for text in texts:
            if not text.strip():
                results.append({
                    'text': text,
                    'error': 'Empty text'
                })
                continue
            
            # Preprocess text
            processed_text = preprocessor.preprocess(text)
            
            if not processed_text:
                results.append({
                    'text': text,
                    'error': 'Text became empty after preprocessing'
                })
                continue
            
            # Make prediction
            prediction = model.predict([processed_text])[0]
            proba = model.predict_proba([processed_text])[0]
            
            label = "FAKE" if prediction == 1 else "REAL"
            confidence = proba[prediction] * 100
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'prediction': label,
                'confidence': float(confidence),
                'probabilities': {
                    'real': float(proba[0]),
                    'fake': float(proba[1])
                }
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Load model at startup
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Please train a model first using train.py")
    
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=True)
