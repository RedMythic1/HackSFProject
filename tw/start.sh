#!/bin/bash

# start.sh - Script to start both frontend and backend services

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Starting Services ===${NC}"

# Set up environment variables for Vercel Blob Storage
if [ -f ".env" ]; then
  echo -e "${BLUE}Loading environment variables from .env file...${NC}"
  export $(grep -v '^#' .env | xargs)
else
  echo -e "${YELLOW}No .env file found, setting default Vercel Blob Storage variables...${NC}"
  export BLOB_READ_WRITE_TOKEN="vercel_blob_rw_MzCMzRmJaiRlp3km_L5RVXS9InB9rTT1Aov2ZI4kzQFoT5S"
  export BLOB_URL="https://mzcmzrmjairlp3km.public.blob.vercel-storage.com"
fi

echo -e "${GREEN}Vercel Blob Storage configured:${NC}"
echo -e "${GREEN}- BLOB_URL: ${BLOB_URL}${NC}"
echo -e "${GREEN}- BLOB_READ_WRITE_TOKEN: [Secret]${NC}"

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
  # Pass the environment variables to the frontend process
  BLOB_READ_WRITE_TOKEN="${BLOB_READ_WRITE_TOKEN}" BLOB_URL="${BLOB_URL}" npm start &
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
echo -e "${BLUE}Using Vercel Blob Storage: ${BLOB_URL}${NC}"

# Keep the script running to hold the processes
while true; do
  sleep 1
done 