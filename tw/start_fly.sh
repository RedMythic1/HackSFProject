#!/bin/bash
# Startup script for Fly.io deployment

# Create necessary directories
mkdir -p local_cache
mkdir -p public/articles
mkdir -p models
mkdir -p stockbt/testing_bs/data_folder

# Make sure we have sample data for stock prediction
if [ ! -f "stockbt/testing_bs/data_folder/sample_data.csv" ] && [ -f "stockbt/testing_bs/data_folder/stock_data_1.csv" ]; then
    cp stockbt/testing_bs/data_folder/stock_data_1.csv stockbt/testing_bs/data_folder/sample_data.csv
fi

# If running on Fly.io with a volume, ensure proper paths
if [ -d "/data" ]; then
    mkdir -p /data/stockbt/testing_bs/data_folder
    mkdir -p /data/article_cache
    mkdir -p /data/models
    
    # Copy CSV files if they don't exist
    if [ ! "$(ls -A /data/stockbt/testing_bs/data_folder)" ]; then
        echo "Copying stock data files to volume..."
        cp -r stockbt/testing_bs/data_folder/* /data/stockbt/testing_bs/data_folder/ 2>/dev/null || true
    fi
    
    # Link the volume directories
    ln -sf /data/article_cache /app/article_cache
    ln -sf /data/models /app/models
    ln -sf /data/stockbt /app/stockbt
fi

echo "Starting application on Fly.io..."
echo "Python version: $(python --version)"
echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)')"

# Start the Flask server
echo "Starting Flask server on port 8080..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
python server.py --port 8080 --host 0.0.0.0 