"""
CYBRON - AI-Powered Review Authenticity Analyzer
Advanced machine learning system for detecting genuine reviews
Made with ♥ by Jarvis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import re
import pickle
import os
from pathlib import Path
import logging
import threading
import time
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for training
training_progress = 0
training_completed = False
training_stats = None
feedback_buffer = deque(maxlen=1000)  # Store last 1000 feedback items
training_lock = threading.Lock()

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:3002"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "100 per minute"]  # More lenient limits for testing
)

# Ensure model directory exists
MODEL_DIR = Path("api/models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "review_analyzer.pkl"

# Download required NLTK data safely
def download_nltk_data():
    try:
        # Create NLTK data directory if it doesn't exist
        nltk_data_dir = Path.home() / 'nltk_data'
        nltk_data_dir.mkdir(exist_ok=True)
        
        # Set NLTK data path
        nltk.data.path.append(str(nltk_data_dir))
        
        # Check and download required resources
        resources = {
            'tokenizers/punkt': 'punkt',
            'corpora/stopwords': 'stopwords',
            'corpora/wordnet': 'wordnet',
            'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
        }
        
        for path, resource in resources.items():
            try:
                nltk.data.find(path)
                logger.info(f"Resource {resource} already downloaded")
            except LookupError:
                logger.info(f"Downloading {resource}...")
                nltk.download(resource, download_dir=str(nltk_data_dir), quiet=True)
                logger.info(f"Successfully downloaded {resource}")
                
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")
        raise

# Initialize NLTK data before creating the analyzer
download_nltk_data()

class ReviewAnalyzer:
    def __init__(self):
        # Ensure NLTK data is available
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            download_nltk_data()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        
        # Initialize vectorizer with proper tokenization settings
        self.vectorizer = TfidfVectorizer(
            max_features=50,  # Fixed number of features
            ngram_range=(1, 2),  # Use unigrams and bigrams only
            min_df=1,  # Include terms that appear at least once
            max_df=0.95,
            strip_accents='unicode',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            token_pattern=r'(?u)\b\w\w+\b',
            stop_words='english'
        )
        
        self.classifier = RandomForestClassifier(
            n_estimators=100,  # Reduced number of trees for faster training
            max_depth=10,  # Limit tree depth to prevent overfitting
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            warm_start=True
        )
        
        # Load or train model
        self.load_or_train_model()
    
    def load_or_train_model(self):
        if MODEL_PATH.exists():
            logger.info("Loading existing model...")
            try:
                with open(MODEL_PATH, 'rb') as f:
                    model_data = pickle.load(f)
                self.vectorizer = model_data['vectorizer']
                self.classifier = model_data['classifier']
                if not isinstance(self.classifier, RandomForestClassifier):
                    logger.warning("Loaded classifier is not RandomForestClassifier, retraining...")
                    self.train_model()
                else:
                    logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.train_model()
        else:
            logger.info("Training new model...")
            self.train_model()
    
    def save_model(self):
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def preprocess_text(self, text):
        """Preprocess text for vectorization"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def train_model(self):
        """Initialize model with enhanced training data"""
        reviews = [
            # Genuine reviews with different styles
            "This product exceeded my expectations. The build quality is excellent.",
            "While not perfect, this item offers good value for money.",
            "A solid 4/5 product. There are some minor issues but works great.",
            "I purchased this for my home office and am quite satisfied.",
            "The product does what it claims, nothing more, nothing less.",
            "After using this for several weeks, I can say it's decent.",
            "Good product with minor flaws. Serves its purpose well.",
            "Reasonable quality for the price point. Fast shipping.",
            
            # Suspicious/Fake reviews with various red flags
            "AMAZING PRODUCT!!!! BEST PURCHASE EVER!!!! MUST BUY NOW!!!!",
            "Just received this amazing product and it's absolutely perfect!!!",
            "WORST THING EVER!!!! DO NOT BUY!!!! COMPLETE SCAM!!!!",
            "Free shipping fast delivery good product good seller recommend!!!",
            "I got this product for free in exchange for my honest review!!!",
            "Best thing I ever bought!!! Can't believe the quality!!!",
            "Terrible product!!! Complete waste of money!!! Don't buy!!!",
            
            # Balanced reviews
            "Good product overall. Some pros and cons worth considering.",
            "3.5 stars. Decent product but has room for improvement.",
            "Mixed feelings about this purchase. Quality is good but pricey.",
            "Not bad, but not great either. Does the job adequately.",
            "Second time buying this. New version has improvements."
        ]
        
        # Labels: 1 for genuine, 0 for suspicious/fake
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        
        try:
            # Preprocess training data
            processed_reviews = [self.preprocess_text(review) for review in reviews]
            
            # Fit and transform the vectorizer
            X = self.vectorizer.fit_transform(processed_reviews)
            y = np.array(labels)
            
            # Train the classifier
            self.classifier.fit(X, y)
            
            # Store initial training data
            self._X_train = processed_reviews
            self._y_train = labels
            
            # Save the model
            self.save_model()
            logger.info(f"Model trained successfully. Feature dimension: {X.shape[1]}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def analyze_review(self, text):
        """Analyze a review text for authenticity"""
        try:
            if not text:
                raise ValueError("Empty text provided")
            
            # Preprocess the input text
            processed_text = self.preprocess_text(text)
            if not processed_text:
                raise ValueError("Text contains no valid words after preprocessing")
            
            # Transform text using the fitted vectorizer
            try:
                # If vectorizer is not fitted yet, use initial training data
                if not hasattr(self, '_X_train') or len(self._X_train) < 2:
                    self.train_model()
                features = self.vectorizer.transform([processed_text])
            except Exception as e:
                logger.error(f"Error in vectorization: {e}")
                # Try retraining if vectorization fails
                self.train_model()
                features = self.vectorizer.transform([processed_text])
            
            try:
                # Get prediction and probability
                prediction = self.classifier.predict(features)[0]
                probabilities = self.classifier.predict_proba(features)[0]
                genuine_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                raise ValueError(f"Failed to analyze text: {str(e)}")
            
            # Calculate additional metrics
            try:
                language_score = self.calculate_language_score(text)
                behavioral_score = self.calculate_behavioral_score(text)
            except Exception as e:
                logger.error(f"Error calculating scores: {e}")
                language_score = 0.5
                behavioral_score = 0.5
            
            # Calculate weighted confidence score
            confidence = (0.4 * genuine_prob + 0.3 * language_score + 0.3 * behavioral_score)
            
            # Determine review status based on multiple factors
            is_suspicious = False
            suspicion_reasons = []
            
            # Check for suspicious patterns
            if behavioral_score < 0.4:
                is_suspicious = True
                suspicion_reasons.append("Suspicious behavioral patterns detected")
            if language_score < 0.4:
                is_suspicious = True
                suspicion_reasons.append("Unnatural language patterns detected")
            if genuine_prob < 0.4:
                is_suspicious = True
                suspicion_reasons.append("Low authenticity probability")
            
            # Additional checks
            words = text.split()
            if words:
                # Check for excessive capitalization
                caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
                if caps_ratio > 0.3:
                    is_suspicious = True
                    suspicion_reasons.append("Excessive use of capital letters")
                
                # Check for repetitive content
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.5:
                    is_suspicious = True
                    suspicion_reasons.append("Repetitive content detected")
                
                # Check for very short or very long reviews
                if len(words) < 5:
                    is_suspicious = True
                    suspicion_reasons.append("Review is too short")
                elif len(words) > 200:
                    confidence *= 0.9  # Slightly reduce confidence for very long reviews
            
            # Calculate final authenticity score
            final_score = confidence if not is_suspicious else confidence * 0.7
            
            # Determine status and message
            if final_score >= 0.8 and not is_suspicious:
                status = "genuine"
                message = "Strong indicators of authentic human review"
            elif final_score >= 0.6 and not is_suspicious:
                status = "likely_genuine"
                message = "Likely to be a genuine review"
            elif final_score >= 0.4:
                status = "uncertain"
                message = "Unable to determine authenticity with confidence"
            else:
                status = "suspicious"
                message = "Multiple indicators of artificial or suspicious content"
            
            return {
                'score': float(final_score),
                'confidence': float(confidence),
                'status': status,
                'message': message,
                'details': {
                    'language_score': float(language_score),
                    'behavioral_score': float(behavioral_score),
                    'model_score': float(genuine_prob),
                    'suspicion_reasons': suspicion_reasons if is_suspicious else [],
                    'preprocessed_text': processed_text
                }
            }
        except ValueError as e:
            logger.error(f"Error analyzing review: {str(e)}")
            raise ValueError(str(e))
        except Exception as e:
            logger.error(f"Unexpected error analyzing review: {str(e)}")
            raise ValueError(f"Unexpected error during analysis: {str(e)}")

    def calculate_language_score(self, text):
        # Enhanced language scoring
        words = text.split()
        if not words:
            return 0.0
            
        # Calculate basic metrics    
        unique_words = len(set(words))
        total_words = len(words)
        avg_word_length = sum(len(word) for word in words) / total_words
        
        # Enhanced scoring metrics
        diversity_score = min(unique_words / total_words, 1)  # Vocabulary diversity
        length_score = min(total_words / 50, 1)  # Reward longer, more detailed reviews
        complexity_score = min(avg_word_length / 8, 1)  # Word complexity
        
        # Detect suspicious patterns
        caps_ratio = sum(1 for word in words if word.isupper()) / total_words
        exclamation_ratio = text.count('!') / total_words
        repetition_ratio = len(set(words)) / total_words
        
        # Calculate pattern penalties
        pattern_penalty = 0
        if caps_ratio > 0.3:  # Too many caps
            pattern_penalty += 0.2
        if exclamation_ratio > 0.2:  # Too many exclamations
            pattern_penalty += 0.2
        if repetition_ratio < 0.5:  # Too much repetition
            pattern_penalty += 0.2
            
        # Final score calculation with pattern penalties
        base_score = (diversity_score + length_score + complexity_score) / 3
        final_score = max(0, base_score - pattern_penalty)
        
        return final_score
    
    def calculate_behavioral_score(self, text):
        score = 1.0
        
        # Enhanced pattern detection
        patterns = {
            r'[!?]{2,}': 0.2,  # Multiple exclamation/question marks
            r'[A-Z]{5,}': 0.2,  # All caps words
            r'\b(\w+)\b.*\b\1\b': 0.1,  # Repeated words
            r'(!!+|!1+|!i+)': 0.15,  # Excessive punctuation variations
            r'(buy|purchase|order).*now': 0.15,  # Pushy sales language
            r'(free|discount|offer).*limited': 0.15,  # Marketing speak
            r'(\d+%|100%)': 0.1,  # Percentage claims
            r'(best|amazing|incredible|perfect)\W+.*(best|amazing|incredible|perfect)': 0.2  # Excessive praise
        }
        
        # Check each pattern and apply penalties
        for pattern, penalty in patterns.items():
            if re.search(pattern, text.lower()):
                score -= penalty
                
        # Additional checks
        words = text.split()
        if words:
            # Check for excessive capitalization
            caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
            if caps_ratio > 0.3:
                score -= 0.2
            
            # Check for repetitive structure
            unique_words_ratio = len(set(words)) / len(words)
            if unique_words_ratio < 0.5:
                score -= 0.15
        
        return max(0, min(1, score))  # Ensure score is between 0 and 1

    def update_model(self, text, prediction, is_correct):
        """Update model using reinforcement learning feedback"""
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            # Initialize or load training data
            if not hasattr(self, '_X_train') or not hasattr(self, '_y_train'):
                self._X_train = []
                self._y_train = []
            
            # Add new example to training data
            self._X_train.append(processed_text)
            self._y_train.append(1 if is_correct else 0)
            
            # Keep only last 1000 examples
            if len(self._X_train) > 1000:
                self._X_train = self._X_train[-1000:]
                self._y_train = self._y_train[-1000:]
            
            # Transform all training data
            if len(self._X_train) >= 2:  # Need at least 2 examples for fitting
                # Fit and transform with the vectorizer
                X = self.vectorizer.fit_transform(self._X_train)
                y = np.array(self._y_train)
                
                # Calculate sample weights based on prediction errors
                weights = np.ones(len(self._y_train))
                current_example_idx = len(self._y_train) - 1
                weights[current_example_idx] = 2.0  # Give higher weight to new example
                
                # Fit the model with sample weights
                self.classifier.fit(X, y, sample_weight=weights)
                
                # Save updated model
                self.save_model()
                
                logger.info(f"Model updated successfully with feedback. Training size: {len(self._X_train)}")
                return True
            else:
                logger.warning("Not enough training examples for update")
                return False
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return False

    def train_on_feedback(self, feedback_data):
        """Train model on collected feedback"""
        try:
            if not feedback_data:
                return 0.0
            
            # Process all feedback data
            processed_texts = []
            labels = []
            weights = []
            
            for item in feedback_data:
                text = item['text']
                is_correct = item['is_correct']
                confidence = item['prediction'].get('confidence', 0.5)
                
                # Process text
                processed_text = self.preprocess_text(text)
                processed_texts.append(processed_text)
                
                # Create label
                labels.append(1 if is_correct else 0)
                
                # Calculate weight based on prediction error
                error = abs(confidence - (1 if is_correct else 0))
                weights.append(1 + error)  # Higher weight for larger errors
            
            if len(processed_texts) >= 2:  # Need at least 2 examples for fitting
                # Convert to numpy arrays
                X = self.vectorizer.fit_transform(processed_texts)
                y = np.array(labels)
                sample_weights = np.array(weights)
                
                # Get initial accuracy
                initial_accuracy = self.classifier.score(X, y)
                
                # Create new classifier with enhanced parameters
                self.classifier = RandomForestClassifier(
                    n_estimators=300,  # Increased number of trees
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1,
                    warm_start=True
                )
                
                # Fit model with sample weights
                self.classifier.fit(X, y, sample_weight=sample_weights)
                
                # Calculate improvement
                final_accuracy = self.classifier.score(X, y)
                improvement = final_accuracy - initial_accuracy
                
                # Update training data
                self._X_train = processed_texts[-1000:]  # Keep last 1000 examples
                self._y_train = labels[-1000:]
                
                # Save model
                self.save_model()
                
                logger.info(f"Model retrained on {len(processed_texts)} samples. Improvement: {improvement:.4f}")
                return improvement
            else:
                logger.warning("Not enough feedback data for retraining")
                return 0.0
            
        except Exception as e:
            logger.error(f"Error training on feedback: {e}")
            return 0.0

# Initialize the analyzer
try:
    analyzer = ReviewAnalyzer()
    logger.info("Review analyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize review analyzer: {e}")
    analyzer = None

def validate_input(text):
    """Validate and sanitize input text"""
    if not text or not isinstance(text, str):
        return False, "Invalid input: text must be a non-empty string"
    if len(text.strip()) < 10:
        return False, "Text is too short. Minimum length is 10 characters."
    if len(text) > 5000:
        return False, "Text is too long. Maximum length is 5000 characters."
    return True, text.strip()

@app.route('/api/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze():
    """Analyze a review text for authenticity"""
    if analyzer is None:
        return jsonify({
            'error': 'Service not ready. Please try again later.',
            'details': 'Model initialization incomplete'
        }), 503
        
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Invalid request format',
                'details': 'Request must be JSON'
            }), 400
            
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data provided',
                'details': 'Request body is empty'
            }), 400
            
        text = data.get('text')
        if text is None:
            return jsonify({
                'error': 'Missing required field',
                'details': 'The "text" field is required in the request body'
            }), 400
            
        # Validate text content
        if not isinstance(text, str):
            return jsonify({
                'error': 'Invalid text format',
                'details': 'Text must be a string'
            }), 400
            
        text = text.strip()
        if len(text) == 0:
            return jsonify({
                'error': 'Empty text',
                'details': 'Text cannot be empty'
            }), 400
            
        if len(text) < 10:
            return jsonify({
                'error': 'Text too short',
                'details': 'Text must be at least 10 characters long'
            }), 400
            
        if len(text) > 5000:
            return jsonify({
                'error': 'Text too long',
                'details': 'Text must not exceed 5000 characters'
            }), 400
        
        # Analyze the review
        try:
            result = analyzer.analyze_review(text)
            logger.info(f"Successfully analyzed review of length {len(text)}")
            return jsonify(result)
            
        except ValueError as e:
            logger.warning(f"Validation error during analysis: {str(e)}")
            return jsonify({
                'error': 'Analysis failed',
                'details': str(e)
            }), 400
            
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {str(e)}")
            return jsonify({
                'error': 'Internal server error',
                'details': 'An unexpected error occurred during analysis'
            }), 500
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'Request processing failed',
            'details': str(e)
        }), 400

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Rate limit exceeded',
        'details': 'Too many requests. Please try again later.'
    }), 429

@app.errorhandler(500)
def internal_error_handler(e):
    return jsonify({
        'error': 'Internal server error',
        'details': 'An unexpected error occurred'
    }), 500

@app.errorhandler(405)
def method_not_allowed_handler(e):
    return jsonify({
        'error': 'Method not allowed',
        'details': f'The {request.method} method is not allowed for this endpoint'
    }), 405

def train_model_task():
    """Background task for model training"""
    global training_progress, training_completed, training_stats
    
    try:
        training_progress = 0
        training_completed = False
        training_stats = None
        
        # Convert feedback buffer to list for training
        feedback_data = list(feedback_buffer)
        total_samples = len(feedback_data)
        
        if total_samples == 0:
            training_completed = True
            training_stats = {
                'accuracy_improvement': 0.0,
                'samples_processed': 0
            }
            return
        
        # Train the model
        accuracy_improvement = analyzer.train_on_feedback(feedback_data)
        
        # Update training stats
        training_stats = {
            'accuracy_improvement': float(accuracy_improvement),
            'samples_processed': total_samples
        }
        
        training_progress = 100
        training_completed = True
        
        logger.info(f"Model training completed. Processed {total_samples} samples.")
    except Exception as e:
        logger.error(f"Error in training task: {e}")
        training_completed = True
        training_stats = {
            'error': str(e),
            'accuracy_improvement': 0.0,
            'samples_processed': 0
        }

@app.route('/api/feedback', methods=['POST'])
@limiter.limit("30 per minute")
def feedback():
    """Endpoint for collecting user feedback"""
    if analyzer is None:
        return jsonify({'error': 'Service not ready'}), 503
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        text = data.get('text')
        prediction = data.get('prediction')
        is_correct = data.get('is_correct')
        
        if not all([text, prediction, isinstance(is_correct, bool)]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Store feedback for later training
        feedback_buffer.append({
            'text': text,
            'prediction': prediction,
            'is_correct': is_correct
        })
        
        # Update model in real-time
        success = analyzer.update_model(text, prediction, is_correct)
        
        return jsonify({
            'success': success,
            'message': 'Feedback received and model updated'
        })
    
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({'error': 'Failed to process feedback'}), 500

@app.route('/api/train/start', methods=['POST'])
@limiter.limit("5 per hour")
def start_training():
    """Start model training process"""
    global training_progress, training_completed, training_stats
    
    if analyzer is None:
        return jsonify({'error': 'Service not ready'}), 503
        
    try:
        with training_lock:
            if training_progress > 0 and not training_completed:
                return jsonify({'error': 'Training already in progress'}), 400
            
            # Reset training state
            training_progress = 0
            training_completed = False
            training_stats = None
            
            # Start training in background thread
            thread = threading.Thread(target=train_model_task)
            thread.start()
            
            return jsonify({
                'message': 'Training started',
                'progress': 0
            })
    
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({'error': 'Failed to start training'}), 500

@app.route('/api/train/progress', methods=['GET'])
def get_training_progress():
    """Get current training progress"""
    try:
        return jsonify({
            'progress': training_progress,
            'completed': training_completed,
            'stats': training_stats
        })
    
    except Exception as e:
        logger.error(f"Error getting training progress: {e}")
        return jsonify({'error': 'Failed to get training progress'}), 500

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════╗
    ║             Welcome to CYBRON API              ║
    ║──────────────────────────────────────────────────║
    ║  🛡️  Advanced Review Authentication System     ║
    ║  🤖  Powered by State-of-the-Art AI           ║
    ║  📊  Real-time Analysis & Training            ║
    ║  ⚡  High-Performance Processing              ║
    ║──────────────────────────────────────────────────║
    ║  Made with ♥ by Jarvis                          ║
    ║  Server running on http://localhost:5000         ║
    ╚══════════════════════════════════════════════════╝
    
    🚀 CYBRON is ready to analyze and authenticate reviews!
    """)
    app.run(port=5000, debug=True)