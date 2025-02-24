import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { 
  ShieldCheckIcon, 
  ShieldExclamationIcon, 
  ArrowPathIcon,
  DocumentTextIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XMarkIcon,
  LightBulbIcon,
  ClockIcon,
  BookmarkIcon,
  ShareIcon,
  DocumentDuplicateIcon,
  AdjustmentsHorizontalIcon,
  BeakerIcon,
  SparklesIcon,
  HashtagIcon,
  ChatBubbleBottomCenterTextIcon,
  EyeIcon,
  BoltIcon,
  HandThumbUpIcon,
  HandThumbDownIcon,
  AcademicCapIcon,
  ArrowPathRoundedSquareIcon,
  CpuChipIcon
} from '@heroicons/react/24/solid';

const SAMPLE_REVIEWS = [
  {
    text: "This product exceeded my expectations! The quality is outstanding, and customer service was very helpful when I had questions. Highly recommended for anyone looking for a reliable solution.",
    label: "Genuine Review"
  },
  {
    text: "AMAZING PRODUCT!!!! MUST BUY NOW!!!! BEST THING EVER!!! LIFE CHANGING!!!! 100% RECOMMENDED!!!",
    label: "Suspicious Review"
  },
  {
    text: "The product arrived on time and works as described. Good value for money, though there's room for improvement in the packaging.",
    label: "Genuine Review"
  }
];

function App() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [charCount, setCharCount] = useState(0);
  const [showTips, setShowTips] = useState(false);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [showSamples, setShowSamples] = useState(false);
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState(false);
  const [modelTraining, setModelTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStats, setTrainingStats] = useState(null);
  const [feedbackSent, setFeedbackSent] = useState(false);

  useEffect(() => {
    setCharCount(review.length);
    // Load history from localStorage
    const savedHistory = localStorage.getItem('reviewHistory');
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, [review]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.ctrlKey && e.key === 'Enter') {
        analyzeReview();
      }
      if (e.ctrlKey && e.key === 'l') {
        clearForm();
      }
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [review]);

  const analyzeReview = async () => {
    if (!review.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }
    
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:5000/api/analyze', {
        text: review
      });
      setResult(response.data);
      setShowTips(true);
      
      // Add to history
      const newHistory = [
        { text: review, result: response.data, timestamp: new Date().toISOString() },
        ...history
      ].slice(0, 10); // Keep last 10 entries
      setHistory(newHistory);
      localStorage.setItem('reviewHistory', JSON.stringify(newHistory));
    } catch (error) {
      console.error('Error analyzing review:', error);
      let errorMessage;
      if (error.response) {
        // The server responded with an error
        errorMessage = error.response.data.error || 
                      'Server returned an error. Please try again.';
      } else if (error.request) {
        // The request was made but no response was received
        errorMessage = 'No response from server. Please check your connection.';
      } else {
        // Something happened in setting up the request
        errorMessage = 'Failed to send request. Please try again.';
      }
      setError(errorMessage);
      setResult(null);
    }
    setLoading(false);
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      // Show toast notification (you can add a toast library)
      alert('Copied to clipboard!');
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const shareResult = () => {
    if (!result) return;
    const shareText = `Review Analysis Result:
Score: ${Math.round(result.score * 100)}%
Status: ${result.status}
${getScoreDescription(result.score)}`;
    
    if (navigator.share) {
      navigator.share({
        title: 'Review Analysis Result',
        text: shareText,
      });
    } else {
      copyToClipboard(shareText);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 0.7) return 'text-green-500';
    if (score >= 0.4) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getScoreGradient = (score) => {
    if (score >= 0.7) return 'from-green-500 to-emerald-600';
    if (score >= 0.4) return 'from-yellow-500 to-orange-600';
    return 'from-red-500 to-rose-600';
  };

  const getScoreDescription = (score) => {
    if (score >= 0.7) return 'This review appears to be genuine.';
    if (score >= 0.4) return 'This review shows some suspicious patterns.';
    return 'This review is likely to be fake or spam.';
  };

  const getTips = (result) => {
    const tips = [];
    if (result.details.behavioral_score < 0.5) {
      tips.push('Avoid excessive punctuation and all-caps text');
      tips.push('Try to use more natural language');
    }
    if (result.details.language_score < 0.5) {
      tips.push('Use more diverse vocabulary');
      tips.push('Write longer, more detailed reviews');
    }
    return tips;
  };

  const getAIConfidenceLevel = (confidence) => {
    if (confidence >= 0.8) return 'Very High';
    if (confidence >= 0.6) return 'High';
    if (confidence >= 0.4) return 'Moderate';
    if (confidence >= 0.2) return 'Low';
    return 'Very Low';
  };

  const getAIInsights = (result) => {
    const insights = [];
    const { language_score, behavioral_score } = result.details;
    
    // Language Analysis
    if (language_score < 0.3) {
      insights.push({
        type: 'warning',
        message: 'Extremely simple or repetitive language detected',
        icon: ChatBubbleBottomCenterTextIcon
      });
    } else if (language_score > 0.8) {
      insights.push({
        type: 'positive',
        message: 'Natural and sophisticated language patterns',
        icon: SparklesIcon
      });
    }

    // Behavioral Analysis
    if (behavioral_score < 0.3) {
      insights.push({
        type: 'warning',
        message: 'Suspicious behavioral patterns detected',
        icon: EyeIcon
      });
    } else if (behavioral_score > 0.8) {
      insights.push({
        type: 'positive',
        message: 'Authentic behavioral characteristics',
        icon: CheckCircleIcon
      });
    }

    // Overall Analysis
    if (result.score < 0.4) {
      insights.push({
        type: 'critical',
        message: 'Multiple indicators of artificial or spam content',
        icon: ExclamationTriangleIcon
      });
    } else if (result.score > 0.8) {
      insights.push({
        type: 'excellent',
        message: 'Strong indicators of authentic human review',
        icon: ShieldCheckIcon
      });
    }

    return insights;
  };

  const getMetricColor = (score) => {
    if (score >= 0.8) return 'text-emerald-500';
    if (score >= 0.6) return 'text-blue-500';
    if (score >= 0.4) return 'text-yellow-500';
    if (score >= 0.2) return 'text-orange-500';
    return 'text-red-500';
  };

  const clearForm = () => {
    setReview('');
    setResult(null);
    setError(null);
    setShowTips(false);
  };

  const provideFeedback = async (isCorrect) => {
    if (!result || feedbackSent) return;
    
    setFeedbackSent(true);
    try {
      await axios.post('http://localhost:5000/api/feedback', {
        text: review,
        prediction: result,
        is_correct: isCorrect
      });
      
      // Show success message
      setError(null);
    } catch (error) {
      console.error('Error sending feedback:', error);
      setError('Failed to send feedback. Please try again.');
      setFeedbackSent(false);
    }
  };

  const startModelTraining = async () => {
    setModelTraining(true);
    setTrainingProgress(0);
    setTrainingStats(null);
    
    try {
      // Start training
      const response = await axios.post('http://localhost:5000/api/train/start');
      
      // Poll training progress
      const pollInterval = setInterval(async () => {
        try {
          const progressResponse = await axios.get('http://localhost:5000/api/train/progress');
          const { progress, completed, stats } = progressResponse.data;
          
          setTrainingProgress(progress);
          
          if (completed) {
            clearInterval(pollInterval);
            setTrainingStats(stats);
            setModelTraining(false);
          }
        } catch (error) {
          console.error('Error polling training progress:', error);
          clearInterval(pollInterval);
          setModelTraining(false);
          setError('Training interrupted. Please try again.');
        }
      }, 1000);
    } catch (error) {
      console.error('Error starting model training:', error);
      setError('Failed to start training. Please try again.');
      setModelTraining(false);
    }
  };

  return (
    <div className={`min-h-screen ${darkMode ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900' : 'bg-gradient-radial from-blue-50 via-white to-blue-50'}`}>
      {/* Header */}
      <header className={`${darkMode ? 'bg-black/30' : 'bg-white/70'} backdrop-blur-lg border-b ${darkMode ? 'border-white/10' : 'border-gray-200'} sticky top-0 z-10 shadow-glow-sm`}>
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <ShieldCheckIcon className="h-8 w-8 text-blue-500 animate-rotate-pulse" />
              <h1 className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'} animate-slide-in-left`}>
                Review Authenticity Analyzer
              </h1>
            </div>
            <div className="flex items-center space-x-4 animate-slide-in-right">
              <button
                onClick={() => setDarkMode(!darkMode)}
                className="p-2 rounded-lg hover:bg-gray-700/20 transition-all duration-300 hover:shadow-glow-sm"
              >
                <AdjustmentsHorizontalIcon className="h-5 w-5 text-blue-500 hover:animate-spin" />
              </button>
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="p-2 rounded-lg hover:bg-gray-700/20 transition-all duration-300 hover:shadow-glow-sm"
              >
                <ClockIcon className="h-5 w-5 text-blue-500 hover:animate-spin" />
              </button>
            </div>
          </div>
          <p className={`mt-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'} max-w-3xl animate-fade-in`}>
            Detect fake or spam reviews using advanced AI analysis. Enter a review below to check its authenticity score.
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Analysis Section */}
          <div className="lg:col-span-2">
            <div className={`${darkMode ? 'bg-white/10' : 'bg-white'} backdrop-blur-xl rounded-2xl shadow-xl border ${darkMode ? 'border-white/20' : 'border-gray-200'} p-8 transition-all duration-300 hover:shadow-glow-md animate-bounce-in`}>
              {/* Input Section */}
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <DocumentTextIcon className="h-5 w-5 text-blue-500 animate-float" />
                    <label htmlFor="review" className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      Review Text
                    </label>
                  </div>
                  <div className="flex items-center space-x-4">
                    <button
                      onClick={() => setShowSamples(!showSamples)}
                      className={`text-sm ${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-900'} flex items-center space-x-1 transition-all duration-300 hover:scale-105`}
                    >
                      <BeakerIcon className="h-4 w-4" />
                      <span>Sample Reviews</span>
                    </button>
                    <span className={`text-sm ${charCount > 5000 ? 'text-red-500 animate-pulse' : darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      {charCount}/5000
                    </span>
                  </div>
                </div>

                {/* Sample Reviews Dropdown */}
                {showSamples && (
                  <div className="grid grid-cols-1 gap-4 animate-slide-up">
                    {SAMPLE_REVIEWS.map((sample, index) => (
                      <button
                        key={index}
                        onClick={() => setReview(sample.text)}
                        className={`p-4 rounded-lg ${darkMode ? 'bg-black/30 hover:bg-black/40' : 'bg-gray-50 hover:bg-gray-100'} transition-all duration-300 text-left group hover:shadow-glow-sm`}
                      >
                        <div className="flex justify-between items-start mb-2">
                          <span className={`text-sm font-medium ${darkMode ? 'text-blue-400' : 'text-blue-600'}`}>
                            {sample.label}
                          </span>
                          <DocumentDuplicateIcon className="h-4 w-4 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                        </div>
                        <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                          {sample.text}
                        </p>
                      </button>
                    ))}
                  </div>
                )}

                <div className="relative group">
                  <textarea
                    id="review"
                    rows={4}
                    className={`block w-full rounded-xl ${darkMode ? 'bg-black/30 border-white/20' : 'bg-white border-gray-200'} border
                             focus:ring-2 focus:ring-blue-500 focus:border-transparent
                             ${darkMode ? 'text-white' : 'text-gray-900'} placeholder-gray-400 resize-none p-4 
                             transition-all duration-300 group-hover:shadow-glow-sm`}
                    placeholder="Enter the review text to analyze..."
                    value={review}
                    onChange={(e) => setReview(e.target.value)}
                    maxLength={5000}
                  />
                  <div className="absolute bottom-4 right-4 flex space-x-2">
                    <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'} opacity-50 group-hover:opacity-100 transition-opacity`}>
                      Press Ctrl + Enter to analyze
                    </span>
                  </div>
                </div>

                <div className="flex space-x-4">
                  <button
                    onClick={analyzeReview}
                    disabled={loading || !review.trim() || charCount > 5000}
                    className={`flex-1 inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium 
                           rounded-xl shadow-lg bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 
                           focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 
                           disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-glow-md
                           text-white transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98]`}
                  >
                    {loading ? (
                      <>
                        <ArrowPathIcon className="animate-spin h-5 w-5 mr-2" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <ChartBarIcon className="h-5 w-5 mr-2" />
                        Analyze Review
                      </>
                    )}
                  </button>
                  <button
                    onClick={clearForm}
                    className={`inline-flex items-center px-4 py-2 border ${darkMode ? 'border-white/20' : 'border-gray-200'} text-sm font-medium 
                             rounded-xl hover:bg-white/5 focus:outline-none focus:ring-2 
                             focus:ring-offset-2 focus:ring-blue-500 ${darkMode ? 'text-white' : 'text-gray-900'} 
                             transition-all duration-300 hover:shadow-glow-sm transform hover:scale-105 active:scale-95`}
                  >
                    <XMarkIcon className="h-5 w-5 mr-2" />
                    Clear
                  </button>
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <div className="mt-6 p-4 bg-red-500/20 border border-red-500/40 rounded-xl animate-shake">
                  <div className="flex items-center space-x-2">
                    <ExclamationTriangleIcon className="h-5 w-5 text-red-500 animate-pulse" />
                    <p className="text-red-500">{error}</p>
                  </div>
                </div>
              )}

              {/* Results Section */}
              {result && !error && (
                <div className="mt-8 space-y-6 animate-slide-up">
                  <div className="flex items-center justify-between">
                    <h2 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'} flex items-center space-x-2`}>
                      <BoltIcon className="h-6 w-6 text-blue-500 animate-pulse" />
                      <span>AI Analysis Results</span>
                    </h2>
                    <div className="flex items-center space-x-4">
                      <button
                        onClick={() => setShowAdvancedMetrics(!showAdvancedMetrics)}
                        className={`text-sm ${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-900'} 
                                 flex items-center space-x-1 transition-all duration-300 hover:scale-105`}
                      >
                        <HashtagIcon className="h-4 w-4" />
                        <span>{showAdvancedMetrics ? 'Hide' : 'Show'} Advanced Metrics</span>
                      </button>
                      <div className="flex space-x-2">
                        <button
                          onClick={() => copyToClipboard(JSON.stringify(result, null, 2))}
                          className="p-2 rounded-lg hover:bg-gray-700/20 transition-all duration-300 hover:shadow-glow-sm transform hover:scale-110"
                          title="Copy raw data"
                        >
                          <DocumentDuplicateIcon className="h-5 w-5 text-blue-500" />
                        </button>
                        <button
                          onClick={shareResult}
                          className="p-2 rounded-lg hover:bg-gray-700/20 transition-all duration-300 hover:shadow-glow-sm transform hover:scale-110"
                          title="Share results"
                        >
                          <ShareIcon className="h-5 w-5 text-blue-500" />
                        </button>
                      </div>
                    </div>
                  </div>
                  
                  {/* AI Insights */}
                  <div className={`${darkMode ? 'bg-blue-500/5' : 'bg-blue-50'} rounded-xl p-6 border ${darkMode ? 'border-blue-500/20' : 'border-blue-200'} 
                                transition-all duration-300 hover:shadow-glow-md animate-fade-in`}>
                    <h4 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'} mb-4 flex items-center space-x-2`}>
                      <SparklesIcon className="h-5 w-5 text-blue-500 animate-pulse" />
                      <span>AI Insights</span>
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {getAIInsights(result).map((insight, index) => (
                        <div
                          key={index}
                          className={`p-4 rounded-lg ${darkMode ? 'bg-black/20' : 'bg-white'} 
                                   border ${darkMode ? 'border-white/10' : 'border-gray-200'}
                                   transition-all duration-300 hover:shadow-glow-sm animate-fade-in`}
                          style={{ animationDelay: `${index * 100}ms` }}
                        >
                          <div className="flex items-start space-x-3">
                            <insight.icon className={`h-5 w-5 mt-0.5 ${
                              insight.type === 'positive' ? 'text-green-500' :
                              insight.type === 'warning' ? 'text-yellow-500' :
                              insight.type === 'critical' ? 'text-red-500' :
                              'text-blue-500'
                            } animate-pulse`} />
                            <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                              {insight.message}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  {/* Feedback Section */}
                  <div className={`${darkMode ? 'bg-green-500/10' : 'bg-green-50'} rounded-xl p-6 border 
                                ${darkMode ? 'border-green-500/20' : 'border-green-200'} 
                                transition-all duration-300 hover:shadow-glow-md animate-fade-in`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <BeakerIcon className="h-5 w-5 text-green-500 animate-pulse" />
                        <h4 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          Help Improve the AI
                        </h4>
                      </div>
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => provideFeedback(true)}
                          disabled={feedbackSent}
                          className={`p-2 rounded-lg hover:bg-gray-700/20 transition-all duration-300 
                                   ${feedbackSent ? 'opacity-50 cursor-not-allowed' : 'hover:scale-110'}`}
                          title="Correct Analysis"
                        >
                          <HandThumbUpIcon className="h-5 w-5 text-green-500" />
                        </button>
                        <button
                          onClick={() => provideFeedback(false)}
                          disabled={feedbackSent}
                          className={`p-2 rounded-lg hover:bg-gray-700/20 transition-all duration-300 
                                   ${feedbackSent ? 'opacity-50 cursor-not-allowed' : 'hover:scale-110'}`}
                          title="Incorrect Analysis"
                        >
                          <HandThumbDownIcon className="h-5 w-5 text-red-500" />
                        </button>
                      </div>
                    </div>
                    <p className={`mt-2 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      {feedbackSent 
                        ? 'Thank you for your feedback! This helps improve the AI model.'
                        : 'Was this analysis correct? Your feedback helps train the AI model.'}
                    </p>
                  </div>

                  {/* Score Card */}
                  <div className={`${darkMode ? 'bg-black/40' : 'bg-white'} rounded-xl p-6 border ${darkMode ? 'border-white/10' : 'border-gray-200'} 
                                hover:border-blue-500/20 transition-all duration-300 hover:shadow-glow-md group animate-scale-in`}>
                    <div className="space-y-4">
                      {/* Score Indicator */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          {result.score >= 0.7 ? (
                            <CheckCircleIcon className="h-6 w-6 text-green-500 animate-bounce" />
                          ) : (
                            <ShieldExclamationIcon className="h-6 w-6 text-yellow-500 animate-pulse" />
                          )}
                          <div>
                            <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                              Authenticity Score
                            </h3>
                            <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                              AI Confidence: {getAIConfidenceLevel(result.confidence)}
                            </p>
                          </div>
                        </div>
                        <span className={`text-3xl font-bold ${getScoreColor(result.score)} transition-all duration-300 group-hover:scale-110`}>
                          {Math.round(result.score * 100)}%
                        </span>
                      </div>

                      {/* Progress Bar */}
                      <div className="h-2 w-full bg-gray-700 rounded-full overflow-hidden">
                        <div 
                          className={`h-full bg-gradient-to-r ${getScoreGradient(result.score)} transition-all duration-1000 ease-out`}
                          style={{ width: `${result.score * 100}%` }}
                        />
                      </div>

                      {/* Description */}
                      <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} mt-2 animate-fade-in`}>
                        {getScoreDescription(result.score)}
                      </p>

                      {/* Advanced Metrics */}
                      {showAdvancedMetrics && (
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6 animate-slide-up">
                          <div className={`${darkMode ? 'bg-white/5' : 'bg-gray-50'} rounded-lg p-4 transition-all duration-300 hover:shadow-glow-sm group`}>
                            <div className="flex items-center space-x-2 mb-2">
                              <LightBulbIcon className="h-5 w-5 text-blue-500 group-hover:animate-spin" />
                              <h4 className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                                Language Score
                              </h4>
                            </div>
                            <div className="flex items-center justify-between">
                              <div className="w-full mr-4">
                                <div className="h-2 w-full bg-gray-700 rounded-full overflow-hidden">
                                  <div 
                                    className="h-full bg-gradient-to-r from-blue-400 to-blue-600 transition-all duration-1000 ease-out"
                                    style={{ width: `${result.details.language_score * 100}%` }}
                                  />
                                </div>
                              </div>
                              <span className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'} group-hover:text-blue-500 transition-colors`}>
                                {Math.round(result.details.language_score * 100)}%
                              </span>
                            </div>
                          </div>
                          <div className={`${darkMode ? 'bg-white/5' : 'bg-gray-50'} rounded-lg p-4 transition-all duration-300 hover:shadow-glow-sm group`}>
                            <div className="flex items-center space-x-2 mb-2">
                              <ClockIcon className="h-5 w-5 text-blue-500 group-hover:animate-spin" />
                              <h4 className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                                Behavioral Score
                              </h4>
                            </div>
                            <div className="flex items-center justify-between">
                              <div className="w-full mr-4">
                                <div className="h-2 w-full bg-gray-700 rounded-full overflow-hidden">
                                  <div 
                                    className="h-full bg-gradient-to-r from-blue-400 to-blue-600 transition-all duration-1000 ease-out"
                                    style={{ width: `${result.details.behavioral_score * 100}%` }}
                                  />
                                </div>
                              </div>
                              <span className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'} group-hover:text-blue-500 transition-colors`}>
                                {Math.round(result.details.behavioral_score * 100)}%
                              </span>
                            </div>
                          </div>
                          <div className={`${darkMode ? 'bg-white/5' : 'bg-gray-50'} rounded-lg p-4 transition-all duration-300 hover:shadow-glow-sm group`}>
                            <div className="flex items-center space-x-2 mb-2">
                              <EyeIcon className="h-5 w-5 text-blue-500 group-hover:animate-spin" />
                              <h4 className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                                AI Confidence
                              </h4>
                            </div>
                            <div className="flex items-center justify-between">
                              <div className="w-full mr-4">
                                <div className="h-2 w-full bg-gray-700 rounded-full overflow-hidden">
                                  <div 
                                    className="h-full bg-gradient-to-r from-purple-400 to-purple-600 transition-all duration-1000 ease-out"
                                    style={{ width: `${result.confidence * 100}%` }}
                                  />
                                </div>
                              </div>
                              <span className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'} group-hover:text-purple-500 transition-colors`}>
                                {Math.round(result.confidence * 100)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Training Section */}
                  <div className={`${darkMode ? 'bg-purple-500/10' : 'bg-purple-50'} rounded-xl p-6 border 
                                ${darkMode ? 'border-purple-500/20' : 'border-purple-200'} 
                                transition-all duration-300 hover:shadow-glow-md animate-slide-up`}>
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-2">
                        <CpuChipIcon className="h-5 w-5 text-purple-500 animate-pulse" />
                        <h4 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          AI Model Training
                        </h4>
                      </div>
                      <button
                        onClick={startModelTraining}
                        disabled={modelTraining}
                        className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium 
                                 rounded-lg shadow-sm bg-purple-500 hover:bg-purple-600 
                                 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 
                                 disabled:opacity-50 disabled:cursor-not-allowed
                                 text-white transition-all duration-200`}
                      >
                        {modelTraining ? (
                          <>
                            <ArrowPathRoundedSquareIcon className="animate-spin h-4 w-4 mr-2" />
                            Training...
                          </>
                        ) : (
                          <>
                            <AcademicCapIcon className="h-4 w-4 mr-2" />
                            Train Model
                          </>
                        )}
                      </button>
                    </div>

                    {modelTraining && (
                      <div className="space-y-4 animate-fade-in">
                        <div className="h-2 w-full bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-purple-400 to-purple-600 transition-all duration-300"
                            style={{ width: `${trainingProgress}%` }}
                          />
                        </div>
                        <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          Training in progress... {trainingProgress}%
                        </p>
                      </div>
                    )}

                    {trainingStats && (
                      <div className="mt-4 grid grid-cols-2 gap-4 animate-fade-in">
                        <div className={`p-4 rounded-lg ${darkMode ? 'bg-black/20' : 'bg-white'} border ${darkMode ? 'border-white/10' : 'border-gray-200'}`}>
                          <p className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                            Accuracy Improvement
                          </p>
                          <p className={`text-lg font-bold ${getMetricColor(trainingStats.accuracy_improvement)}`}>
                            +{(trainingStats.accuracy_improvement * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className={`p-4 rounded-lg ${darkMode ? 'bg-black/20' : 'bg-white'} border ${darkMode ? 'border-white/10' : 'border-gray-200'}`}>
                          <p className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                            Training Samples
                          </p>
                          <p className="text-lg font-bold text-purple-500">
                            {trainingStats.samples_processed}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Writing Tips */}
                  {showTips && getTips(result).length > 0 && (
                    <div className={`${darkMode ? 'bg-blue-500/10' : 'bg-blue-50'} rounded-xl p-6 border ${darkMode ? 'border-blue-500/20' : 'border-blue-200'} 
                                  transition-all duration-300 hover:shadow-glow-md animate-slide-up`}>
                      <h4 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'} mb-4 flex items-center space-x-2`}>
                        <LightBulbIcon className="h-5 w-5 text-blue-500 animate-float" />
                        <span>AI Writing Recommendations</span>
                      </h4>
                      <ul className="space-y-2">
                        {getTips(result).map((tip, index) => (
                          <li key={index} className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} flex items-start space-x-2 animate-fade-in`} 
                              style={{ animationDelay: `${index * 100}ms` }}>
                            <span className="text-blue-500">•</span>
                            <span>{tip}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* History Sidebar */}
          <div className={`lg:block ${showHistory ? 'block animate-slide-in-right' : 'hidden'}`}>
            <div className={`${darkMode ? 'bg-white/10' : 'bg-white'} backdrop-blur-xl rounded-2xl shadow-xl border ${darkMode ? 'border-white/20' : 'border-gray-200'} p-6 transition-all duration-300 hover:shadow-glow-md`}>
              <div className="flex items-center justify-between mb-4">
                <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'} flex items-center space-x-2`}>
                  <BookmarkIcon className="h-5 w-5 text-blue-500" />
                  <span>Analysis History</span>
                </h3>
                <button
                  onClick={() => {
                    setHistory([]);
                    localStorage.removeItem('reviewHistory');
                  }}
                  className="text-sm text-blue-500 hover:text-blue-400 transition-colors hover:scale-105 transform"
                >
                  Clear History
                </button>
              </div>
              <div className="space-y-4">
                {history.length === 0 ? (
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'} animate-pulse`}>
                    No analysis history yet
                  </p>
                ) : (
                  history.map((item, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg ${darkMode ? 'bg-black/30' : 'bg-gray-50'} cursor-pointer 
                               hover:bg-opacity-80 transition-all duration-300 hover:shadow-glow-sm transform hover:scale-[1.02] 
                               animate-fade-in`}
                      style={{ animationDelay: `${index * 50}ms` }}
                      onClick={() => {
                        setReview(item.text);
                        setResult(item.result);
                      }}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className={`text-sm font-medium ${getScoreColor(item.result.score)}`}>
                          {Math.round(item.result.score * 100)}% Authentic
                        </span>
                        <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                          {new Date(item.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'} line-clamp-2`}>
                        {item.text}
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Signature Footer */}
      <footer className={`${darkMode ? 'bg-black/30' : 'bg-white/70'} backdrop-blur-lg border-t ${darkMode ? 'border-white/10' : 'border-gray-200'} mt-auto py-6`}>
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="flex items-center space-x-2 group">
              <ShieldCheckIcon className="h-8 w-8 text-blue-500 group-hover:animate-bounce" />
              <h3 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'} group-hover:text-blue-500 transition-colors`}>
                CYBRON
              </h3>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className={`px-4 py-2 rounded-lg ${darkMode ? 'bg-blue-500/10' : 'bg-blue-50'} 
                           border ${darkMode ? 'border-blue-500/20' : 'border-blue-200'} 
                           transition-all duration-300 hover:shadow-glow-sm group cursor-pointer transform hover:scale-105`}>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Powered by{' '}
                  <span className="inline-flex items-center space-x-1">
                    <SparklesIcon className="h-4 w-4 text-blue-500 group-hover:animate-spin" />
                    <span className="font-semibold bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">Advance AI</span>
                  </span>
                </p>
              </div>
              
              <div className={`px-4 py-2 rounded-lg ${darkMode ? 'bg-purple-500/10' : 'bg-purple-50'} 
                           border ${darkMode ? 'border-purple-500/20' : 'border-purple-200'} 
                           transition-all duration-300 hover:shadow-glow-sm group cursor-pointer transform hover:scale-105`}>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Created with{' '}
                  <span className="inline-flex items-center space-x-1">
                    <span className="text-red-500 animate-pulse">♥</span>
                    <span className="font-semibold bg-gradient-to-r from-purple-400 via-pink-500 to-blue-500 bg-clip-text text-transparent group-hover:from-blue-500 group-hover:via-purple-500 group-hover:to-pink-500 transition-all duration-500">
                      by CYBRON Group
                    </span>
                  </span>
                </p>
              </div>
            </div>
            
            <div className="flex flex-col items-center space-y-2">
              <div className="flex items-center space-x-2 text-sm">
                <BeakerIcon className="h-4 w-4 text-green-500 animate-pulse" />
                <span className={`${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Empowering authentic voices since 2024
                </span>
              </div>
              <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'} text-center max-w-md`}>
                CYBRON: Where AI meets authenticity to create a more trustworthy digital world
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
