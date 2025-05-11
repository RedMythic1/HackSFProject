#!/bin/bash

# Script to fix duplicate prefixes in blob storage

# Ensure environment is set up
if [ ! -f "setup-env.sh" ]; then
  echo "Error: setup-env.sh not found in current directory"
  exit 1
fi

# Load environment variables
source setup-env.sh

# Check if BLOB_READ_WRITE_TOKEN is set
if [ -z "$BLOB_READ_WRITE_TOKEN" ]; then
  echo "Error: BLOB_READ_WRITE_TOKEN not set. Please update setup-env.sh"
  exit 1
fi

echo "Running fix-duplicate-prefixes.js..."
node tools/fix-duplicate-prefixes.js

if [ $? -eq 0 ]; then
  echo "Successfully fixed duplicate prefixes in blob storage"
  exit 0
else
  echo "Error: Failed to fix duplicate prefixes in blob storage"
  exit 1
fi 