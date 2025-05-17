#!/bin/bash

# Build script for Vercel deployment
echo "Starting build process for Vercel deployment..."

# Make sure we're in the correct directory
if [[ $(basename $(pwd)) == "tw" ]]; then
    PROJECT_DIR=$(pwd)
    echo "Current directory: $PROJECT_DIR"
else
    echo "Error: This script should be run from the 'tw' directory"
    exit 1
fi

# Step 1: Run the preparation script first
echo "Running preparation script to clean up conflicting files..."
./prepare-vercel-deploy.sh
if [ $? -ne 0 ]; then
    echo "Error: Preparation script failed."
    exit 1
fi

# Step 2: Install npm dependencies
echo "Installing npm dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo "Error: npm install failed."
    exit 1
fi

# Step 3: Build the frontend
echo "Building frontend..."
npm run build
if [ $? -ne 0 ]; then
    echo "Error: Frontend build failed."
    exit 1
fi

# Step 4: Verify the build succeeded
if [ ! -d "./dist" ]; then
    echo "Error: Build directory './dist' not found after build."
    exit 1
fi

echo "Build process completed successfully!"
exit 0 