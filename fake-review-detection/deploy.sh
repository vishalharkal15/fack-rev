#!/bin/bash

# Exit on error
set -e

echo "Starting deployment process..."

# Create necessary directories
mkdir -p logs
mkdir -p api/models

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python download_nltk_data.py

# Set environment variables
export FLASK_ENV=production
export FLASK_APP=wsgi.py

# Create or update .env file
echo "Setting up environment variables..."
cat > .env << EOL
FLASK_APP=wsgi.py
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=https://verivoicecybron.netlify.app/
EOL

# Start Gunicorn
echo "Starting Gunicorn server..."
gunicorn --config gunicorn_config.py wsgi:app 