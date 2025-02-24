import nltk
import ssl
import os
from pathlib import Path

def download_nltk_data():
    """Download required NLTK data to a specific directory"""
    # Create NLTK data directory if it doesn't exist
    nltk_data_dir = Path.home() / 'nltk_data'
    nltk_data_dir.mkdir(exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.append(str(nltk_data_dir))
    
    # Handle SSL certificate verification
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Required NLTK resources
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'taggers/averaged_perceptron_tagger'
    ]
    
    # Download each resource
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource, download_dir=str(nltk_data_dir), quiet=True)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

if __name__ == "__main__":
    download_nltk_data()
