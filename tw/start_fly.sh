#!/bin/bash
# Startup script for Fly.io deployment

# Create necessary directories
mkdir -p local_cache
mkdir -p public/articles
mkdir -p models

echo "Starting application on Fly.io..."

# Start the Flask server
echo "Starting Flask server on port 8080..."
python server.py --port 8080 --host 0.0.0.0 