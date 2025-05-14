#!/bin/bash

# Debugging script to start all services and set up environment
echo "=== Starting debugging script ==="

# Kill any processes using ports 9000 and 3000
echo "Killing any processes using ports 9000 and 3000..."
lsof -ti:9000 | xargs kill -9 2>/dev/null || echo "No process found on port 9000"
lsof -ti:3000 | xargs kill -9 2>/dev/null || echo "No process found on port 3000"

# Set up environment variables
echo "Setting up environment variables..."
source setup-env.sh || echo "Warning: setup-env.sh failed"

# Build the project
echo "Building the project..."
npm run build

# Start the backend server
echo "Starting backend server..."
node api/server.js > server.log 2>&1 &
BACKEND_PID=$!
echo "Backend server started with PID: $BACKEND_PID"

# Start the frontend server
echo "Starting frontend server..."
npm start > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend server started with PID: $FRONTEND_PID"

echo "=== Servers are starting ==="
echo "Backend should be available at: http://localhost:3000"
echo "Frontend should be available at: http://localhost:9000"
echo "Server logs are in server.log and frontend.log"
echo ""
echo "Press Enter to stop servers and clean up"
read

# Kill the servers
kill $BACKEND_PID $FRONTEND_PID
echo "Servers stopped" 