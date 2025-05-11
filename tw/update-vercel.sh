#!/bin/bash

# Update Vercel deployment with correct Blob token configuration
echo "Updating Vercel deployment..."

# Make sure we're in the correct directory
if [[ $(basename $(pwd)) == "tw" ]]; then
    PROJECT_DIR=$(pwd)
    echo "Current directory: $PROJECT_DIR"
else
    echo "Error: This script should be run from the 'tw' directory"
    exit 1
fi

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "Error: Vercel CLI not found. Please install it with 'npm install -g vercel'"
    exit 1
fi

# Source the environment variables
echo "Loading environment variables..."
source setup-env.sh

# Verify the environment variables
echo "Verifying environment variables..."
if [ -z "$BLOB_READ_WRITE_TOKEN" ]; then
    echo "Error: BLOB_READ_WRITE_TOKEN is not set"
    exit 1
fi

if [ -z "$BLOB_URL" ]; then
    echo "Error: BLOB_URL is not set"
    exit 1
fi

# Test local access to Vercel Blob
echo "Testing Vercel Blob access..."
node tools/test-blob.js
if [ $? -ne 0 ]; then
    echo "Error: Vercel Blob access test failed. Please check your configuration."
    exit 1
fi

# Run the fix-article-issues script
echo "Checking for article issues..."
node tools/fix-article-issues.js
if [ $? -ne 0 ]; then
    echo "Warning: Article issue check failed. Proceeding with deployment anyway."
fi

# Fix any duplicated prefixes
echo "Checking for duplicated prefixes in blob keys..."
node tools/fix-duplicate-prefixes.js
if [ $? -ne 0 ]; then
    echo "Warning: Failed to fix duplicated prefixes. Proceeding with deployment anyway."
fi

# Deploy to Vercel with environment variables
echo "Deploying to Vercel..."
# Run deployment command from the current directory
vercel --prod \
  --env BLOB_READ_WRITE_TOKEN="$BLOB_READ_WRITE_TOKEN" \
  --env BLOB_URL="$BLOB_URL"

echo "Deployment complete!" 