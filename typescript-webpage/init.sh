#!/bin/bash

# Print colorful status messages
print_status() {
    echo -e "\033[1;34m==>\033[0m $1"
}

print_error() {
    echo -e "\033[1;31mError:\033[0m $1"
}

print_success() {
    echo -e "\033[1;32mSuccess:\033[0m $1"
}

# Get absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"
print_status "Changed to directory: $SCRIPT_DIR"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p .cache
mkdir -p user_data
mkdir -p dist

# Start the server
print_status "Starting the server in simplified mode..."
python3 server.py > server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start (up to 10 seconds)
print_status "Waiting for server to start..."
for i in {1..10}; do
    if curl -s http://localhost:5001/check-cache > /dev/null 2>&1; then
        print_success "Server is running"
        break
    fi
    if [ $i -eq 10 ]; then
        print_error "Server failed to start within 10 seconds"
        echo "Server logs:"
        cat server.log
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Start the frontend
print_status "Starting the frontend..."
npm start > frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start (up to 10 seconds)
print_status "Waiting for frontend to start..."
for i in {1..10}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        print_success "Frontend is running"
        break
    fi
    if [ $i -eq 10 ]; then
        print_error "Frontend failed to start within 10 seconds"
        echo "Frontend logs:"
        cat frontend.log
        kill $FRONTEND_PID 2>/dev/null
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

print_success "Initialization complete!"
print_status "Server is running on http://localhost:5001"
print_status "Frontend is running on http://localhost:3000"

# Function to handle script termination
cleanup() {
    print_status "Shutting down..."
    kill $SERVER_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up trap for script termination
trap cleanup SIGINT SIGTERM

# Keep script running
wait 