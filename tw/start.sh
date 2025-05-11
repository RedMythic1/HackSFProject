#!/bin/bash

# start.sh - Script to start both frontend and backend services

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Starting Services ===${NC}"

# Check for required dependencies
check_dependency() {
  if ! command -v $1 &> /dev/null; then
    echo -e "${RED}Error: $1 is required but not installed.${NC}"
    echo -e "${YELLOW}Please install $1 and try again.${NC}"
    exit 1
  fi
}

check_dependency "node"
check_dependency "npm"
check_dependency "python3"

# Ensure all npm dependencies are installed
echo -e "${BLUE}Installing npm dependencies...${NC}"
npm install

# Function to start the backend server
start_backend() {
  echo -e "${GREEN}Starting Python Flask backend server...${NC}"
  python3 server.py &
  BACKEND_PID=$!
  echo -e "${GREEN}Backend server started with PID: ${BACKEND_PID}${NC}"
}

# Function to start the frontend
start_frontend() {
  echo -e "${GREEN}Starting frontend webpack development server...${NC}"
  npm start &
  FRONTEND_PID=$!
  echo -e "${GREEN}Frontend server started with PID: ${FRONTEND_PID}${NC}"
}

# Function to handle graceful shutdown
cleanup() {
  echo -e "${YELLOW}Shutting down services...${NC}"
  if [ ! -z "$BACKEND_PID" ]; then
    echo -e "${YELLOW}Stopping backend server (PID: ${BACKEND_PID})${NC}"
    kill $BACKEND_PID 2>/dev/null
  fi
  if [ ! -z "$FRONTEND_PID" ]; then
    echo -e "${YELLOW}Stopping frontend server (PID: ${FRONTEND_PID})${NC}"
    kill $FRONTEND_PID 2>/dev/null
  fi
  echo -e "${GREEN}Services stopped.${NC}"
  exit 0
}

# Set up the cleanup trap
trap cleanup SIGINT SIGTERM

# Start the services
start_backend
start_frontend

echo -e "${BLUE}All services started. Press Ctrl+C to stop.${NC}"
echo -e "${BLUE}Backend server: http://localhost:5001${NC}"
echo -e "${BLUE}Frontend server: http://localhost:3000${NC}"

# Keep the script running to hold the processes
while true; do
  sleep 1
done 