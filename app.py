from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import json
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load sample training data from JSON file
with open("data/sentiment_dataset.json", "r", encoding="utf-8") as f:
    SAMPLE_DATA = json.load(f)

# Initialize and train model
class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression(max_iter=1000)
        self.trained = False
    
    def preprocess_text(self, text):
        # Clean text
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train(self, texts, labels):
        # Preprocess texts
        processed_texts = [self.preprocess_text(t) for t in texts]
        
        # Vectorize
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Train model
        self.model.fit(X, labels)
        self.trained = True
    
    def predict(self, text):
        if not self.trained:
            raise Exception("Model not trained yet")
        
        # Preprocess
        processed = self.preprocess_text(text)
        
        # Vectorize
        X = self.vectorizer.transform([processed])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get confidence scores
        labels = self.model.classes_
        scores = {label: float(prob) for label, prob in zip(labels, probabilities)}
        
        return prediction, scores

# Initialize model
sentiment_model = SentimentModel()

# Train on startup
texts = [item[0] for item in SAMPLE_DATA]
labels = [item[1] for item in SAMPLE_DATA]
sentiment_model.train(texts, labels)

@app.route('/')
def home():
    return jsonify({
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze sentiment of text",
            "/batch": "POST - Analyze multiple texts",
            "/health": "GET - Health check"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_trained": sentiment_model.trained
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        # Predict sentiment
        sentiment, scores = sentiment_model.predict(text)
        
        return jsonify({
            "text": text,
            "sentiment": sentiment,
            "confidence": scores[sentiment],
            "scores": scores
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/batch', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                "error": "Missing 'texts' field in request body"
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                "error": "'texts' must be an array"
            }), 400
        
        results = []
        for text in texts:
            if text and len(text.strip()) > 0:
                sentiment, scores = sentiment_model.predict(text)
                results.append({
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": scores[sentiment],
                    "scores": scores
                })
        
        return jsonify({
            "count": len(results),
            "results": results
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)