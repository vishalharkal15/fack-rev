# Review Authenticity Analyzer

![CYBRON Logo](https://img.shields.io/badge/CYBRON-AI%20Powered-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![React](https://img.shields.io/badge/react-18.2.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An advanced AI-powered system for detecting and analyzing the authenticity of reviews. Built with Flask, React, and state-of-the-art machine learning techniques.

## 🌟 Features

- **Real-time Analysis**: Instant feedback on review authenticity
- **Advanced AI Model**: Uses RandomForest classifier with TF-IDF vectorization
- **Comprehensive Metrics**: Language analysis, behavioral patterns, and authenticity scores
- **Interactive UI**: Modern, responsive interface with real-time feedback
- **Continuous Learning**: Model improves through user feedback
- **Rate Limiting**: Built-in protection against abuse
- **Dark/Light Mode**: Customizable UI theme

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 14.0 or higher
- npm or yarn
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/review-authenticity-analyzer.git
   cd review-authenticity-analyzer
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python download_nltk_data.py
   ```

3. **Install frontend dependencies**
   ```bash
   npm install
   ```

4. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Application

1. **Start the Flask backend**
   ```bash
   flask run
   ```

2. **Start the React frontend**
   ```bash
   npm run dev
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - API: http://localhost:5000

## 🧪 Testing

### Backend Tests
```bash
python test_api.py  # API tests
python test_model.py  # Model tests
```

### Frontend Tests
```bash
npm test
```

## 🛠️ API Endpoints

### POST /api/analyze
Analyzes a review text for authenticity.

**Request Body:**
```json
{
  "text": "Your review text here"
}
```

**Response:**
```json
{
  "score": 0.85,
  "confidence": 0.9,
  "status": "genuine",
  "message": "Strong indicators of authentic human review",
  "details": {
    "language_score": 0.8,
    "behavioral_score": 0.9,
    "model_score": 0.85,
    "suspicion_reasons": []
  }
}
```

### POST /api/feedback
Provides feedback to improve the model.

### POST /api/train/start
Initiates model retraining.

### GET /api/train/progress
Checks training progress.

## 🎯 Features in Detail

### AI Analysis Components
- Language Pattern Analysis
- Behavioral Pattern Detection
- Sentiment Analysis
- Text Complexity Evaluation
- Suspicious Pattern Recognition

### User Interface
- Real-time Analysis
- Interactive Feedback System
- Analysis History
- Advanced Metrics Display
- Training Progress Monitoring
- Sample Reviews
- Copy & Share Functionality

## 🔒 Security Features

- Rate Limiting
- Input Validation
- CORS Protection
- Error Handling
- Request Size Limits

## 🎨 UI Customization

The application supports both dark and light modes, with smooth transitions and modern design elements:

- Gradient Backgrounds
- Animated Components
- Responsive Layout
- Interactive Elements
- Loading States
- Error Boundaries

## 📊 Performance

- Optimized Model Loading
- Efficient Text Processing
- Caching Mechanisms
- Rate Limiting
- Background Processing

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NLTK for natural language processing
- scikit-learn for machine learning capabilities
- React and Tailwind CSS for the frontend
- Flask for the backend framework

## 📧 Contact

CYBRON Team - cybron@example.com

Project Link: [https://github.com/maneomkar.369/review-authenticity-analyzer](https://github.com/maneomkar.369/review-authenticity-analyzer)

---

Made with ♥ by CYBRON Group 