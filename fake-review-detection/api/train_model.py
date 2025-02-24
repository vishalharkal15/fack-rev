from ml.review_analyzer import ReviewClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def analyze_errors(analyzer, texts, true_labels, pred_labels):
    errors = []
    for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
        if true_label != pred_label:
            analysis = analyzer.predict(text)
            errors.append({
                'text': text[:200] + '...' if len(text) > 200 else text,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': analysis['confidence'],
                'sentiment': analysis['sentiment']['compound']
            })
    return pd.DataFrame(errors)

if __name__ == '__main__':
    print("Loading and training model...")
    analyzer = ReviewClassifier('data/fake reviews dataset.csv')
    
    # Train the model and get predictions
    metrics = analyzer.train()
    
    print("\nDetailed Performance Metrics:")
    print("-" * 50)
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Testing Accuracy: {metrics['test_accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        metrics['labels'],
        'Review Classification Confusion Matrix'
    )
    
    # Analyze errors
    error_analysis = analyze_errors(
        analyzer,
        metrics['test_texts'],
        metrics['test_true_labels'],
        metrics['test_pred_labels']
    )
    
    print("\nError Analysis:")
    print("-" * 50)
    print(f"Number of misclassified samples: {len(error_analysis)}")
    
    if not error_analysis.empty:
        print("\nSample of misclassified reviews:")
        pd.set_option('display.max_colwidth', 100)
        print(error_analysis.head())
