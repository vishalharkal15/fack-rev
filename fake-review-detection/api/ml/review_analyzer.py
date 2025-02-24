import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import spacy
import gc
from textblob import TextBlob
from collections import Counter

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        if not isinstance(text, str):
            text = str(text)
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Handle special characters
        text = re.sub(r'[^\w\s!?.]', '', text)
        
        # Normalize repeated characters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Convert 'woooow' to 'woow'
        
        # Handle common abbreviations
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'m", " am", text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_text(self, text):
        # Tokenize
        tokens = word_tokenize(text)
        
        # Lemmatize and remove stopwords
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [self.preprocess_text(text) for text in X]

class ExtraFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def extract_features(self, text):
        if not isinstance(text, str):
            text = str(text)
            
        # Text statistics
        length = len(text)
        words = text.split()
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        word_count = len(words)
        
        # Sentiment features using VADER
        vader_sentiment = self.sia.polarity_scores(text)
        
        # TextBlob sentiment and subjectivity
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment
        
        # Punctuation features
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Capitalization features
        caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) or 1)
        
        # POS tag features
        pos_tags = [tag for _, tag in blob.tags]
        noun_ratio = pos_tags.count('NN') / (len(pos_tags) or 1)
        verb_ratio = sum(1 for tag in pos_tags if tag.startswith('VB')) / (len(pos_tags) or 1)
        adj_ratio = sum(1 for tag in pos_tags if tag.startswith('JJ')) / (len(pos_tags) or 1)
        
        return [
            length,
            avg_word_length,
            word_count,
            vader_sentiment['compound'],
            vader_sentiment['pos'],
            vader_sentiment['neg'],
            vader_sentiment['neu'],
            float(textblob_sentiment.polarity),
            float(textblob_sentiment.subjectivity),
            exclamation_count,
            question_count,
            caps_ratio,
            noun_ratio,
            verb_ratio,
            adj_ratio
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([self.extract_features(text) for text in X])

class ReviewClassifier:
    def __init__(self, dataset_path):
        # Load only required columns with optimized dtypes
        dtype_dict = {
            'text_': str,
            'label': str
        }
        
        # Read data in chunks to handle memory efficiently
        chunks = []
        for chunk in pd.read_csv(dataset_path, 
                               usecols=['text_', 'label'],
                               dtype=dtype_dict,
                               chunksize=10000):
            chunk = chunk.dropna()
            chunks.append(chunk)
        
        self.dataset = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.dataset['label'] = self.label_encoder.fit_transform(self.dataset['label'])
        
        # Initialize spaCy for advanced NLP
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
        
        # Create pipeline components
        self.text_preprocessor = TextPreprocessor()
        self.tfidf = TfidfVectorizer(
            tokenizer=self.text_preprocessor.tokenize_text,
            ngram_range=(1, 3),
            max_features=15000,
            min_df=2,
            max_df=0.95
        )
        self.feature_extractor = ExtraFeatureExtractor()
        self.scaler = StandardScaler()
        
        # Create ensemble of classifiers
        self.classifiers = {
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            'svm': LinearSVC(
                C=1.0,
                class_weight='balanced',
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                early_stopping=True,
                random_state=42
            )
        }
        
        self.classifier = VotingClassifier(
            estimators=[
                ('gb', self.classifiers['gb']),
                ('svm', self.classifiers['svm']),
                ('mlp', self.classifiers['mlp'])
            ],
            voting='hard'
        )
    
    def train(self):
        print("Starting training process...")
        print(f"Dataset size: {len(self.dataset)} reviews")
        print(f"Label distribution: {pd.Series(self.dataset['label']).value_counts().to_dict()}")
        
        print("Preprocessing text...")
        X_preprocessed = self.text_preprocessor.transform(self.dataset['text_'])
        
        print("Extracting text features...")
        X_text = self.tfidf.fit_transform(X_preprocessed)
        
        print("Extracting additional features...")
        X_extra = self.feature_extractor.transform(X_preprocessed)
        X_extra_scaled = self.scaler.fit_transform(X_extra)
        
        print("Combining features...")
        X_combined = np.hstack([X_text.toarray(), X_extra_scaled])
        y = self.dataset['label'].values
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, stratify=y, random_state=42)
        
        print("Training ensemble model...")
        self.classifier.fit(X_train, y_train)
        
        # Get predictions
        y_train_pred = self.classifier.predict(X_train)
        y_test_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        train_accuracy = np.mean(y_train_pred == y_train)
        test_accuracy = np.mean(y_test_pred == y_test)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        
        # Get original labels
        test_true_labels = self.label_encoder.inverse_transform(y_test)
        test_pred_labels = self.label_encoder.inverse_transform(y_test_pred)
        unique_labels = self.label_encoder.classes_
        
        # Generate classification report
        class_report = classification_report(test_true_labels, test_pred_labels)
        conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)
        
        print("Saving model...")
        model_data = {
            'text_preprocessor': self.text_preprocessor,
            'tfidf': self.tfidf,
            'feature_extractor': self.feature_extractor,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, 'api/ml/review_classifier.pkl')
        print("Training complete!")
        
        # Get test texts for error analysis
        test_texts = self.dataset['text_'].iloc[y_test.index].values
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'labels': unique_labels,
            'test_texts': test_texts,
            'test_true_labels': test_true_labels,
            'test_pred_labels': test_pred_labels
        }
    
    def predict(self, text):
        # Load model components
        model_data = joblib.load('api/ml/review_classifier.pkl')
        text_preprocessor = model_data['text_preprocessor']
        tfidf = model_data['tfidf']
        feature_extractor = model_data['feature_extractor']
        scaler = model_data['scaler']
        classifier = model_data['classifier']
        label_encoder = model_data['label_encoder']
        
        # Preprocess text
        preprocessed_text = text_preprocessor.transform([text])[0]
        
        # Extract features
        X_text = tfidf.transform([preprocessed_text])
        X_extra = feature_extractor.transform([preprocessed_text])
        X_extra_scaled = scaler.transform(X_extra)
        
        # Combine features
        X_combined = np.hstack([X_text.toarray(), X_extra_scaled])
        
        # Get prediction probability and class
        predicted_class = label_encoder.inverse_transform(classifier.predict(X_combined))[0]
        
        # Get detailed analysis
        doc = self.nlp(text)
        sentiment = feature_extractor.sia.polarity_scores(text)
        blob = TextBlob(text)
        
        # Extract key phrases and entities
        key_phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Only multi-word phrases
                key_phrases.append(chunk.text)
        
        # Get word frequencies
        words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
        word_freq = Counter(words).most_common(5)
        
        analysis = {
            'predicted_class': predicted_class,
            'confidence': 'High' if sentiment['compound'] > 0.5 else 'Medium' if sentiment['compound'] > 0 else 'Low',
            'sentiment': {
                'vader': sentiment,
                'textblob': {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            },
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'key_phrases': key_phrases,
            'details': {
                'length': len(text),
                'word_count': len(text.split()),
                'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'caps_ratio': sum(1 for c in text if c.isupper()) / (len(text) or 1),
                'most_common_words': dict(word_freq),
                'pos_distribution': {pos: count/len(doc) for pos, count in Counter([token.pos_ for token in doc]).items()}
            }
        }
        
        return analysis
