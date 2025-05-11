#!/bin/bash
# Start both frontend and backend servers

# Create necessary directories if they don't exist
echo "Creating necessary directories..."
mkdir -p local_cache
mkdir -p public/articles

# Kill any processes using the ports we need
echo "Checking if ports are already in use..."
if lsof -i:5001 &> /dev/null; then
    echo "Port 5001 is in use. Stopping process..."
    kill $(lsof -t -i:5001) 2>/dev/null || true
fi

if lsof -i:9001 &> /dev/null; then
    echo "Port 9001 is in use. Stopping process..."
    kill $(lsof -t -i:9001) 2>/dev/null || true
fi

# Check if models directory exists and create if needed
if [ ! -d "models" ]; then
    mkdir -p models
    echo "Created models directory. Note: You may need to add model files manually."
fi

# Check if the required files exist
if [ ! -f "ansys_local.py" ]; then
    echo "ERROR: ansys_local.py not found in the current directory."
    echo "Please ensure the file exists before running this script."
    exit 1
fi

if [ ! -f "hackernews_summarizer_local.py" ]; then
    echo "ERROR: hackernews_summarizer_local.py not found in the current directory."
    echo "Please ensure the file exists before running this script."
    exit 1
fi

echo "Starting backend server on port 5001..."
python server.py --port 5001 > server.log 2>&1 &
BACKEND_PID=$!
echo "Backend server started with PID: $BACKEND_PID"

echo "Starting frontend server on port 9001..."
PORT=9001 npm start

# When npm start is terminated, also kill the backend
kill $BACKEND_PID
echo "Backend server stopped" 