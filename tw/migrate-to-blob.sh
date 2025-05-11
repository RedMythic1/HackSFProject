#!/bin/bash

# migrate-to-blob.sh - Script to migrate local cache to Vercel Blob storage

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Migrating Local Cache to Vercel Blob Storage ===${NC}"

# Make sure we're in the right directory
cd "$(dirname "$0")" || { echo -e "${RED}Failed to change directory${NC}"; exit 1; }

# Source environment variables from setup-env.sh if it exists
if [ -f "./setup-env.sh" ]; then
  echo -e "${GREEN}Setting up environment variables from setup-env.sh...${NC}"
  source ./setup-env.sh
else
  # Check if environment variables are set
  if [ -z "$BLOB_READ_WRITE_TOKEN" ]; then
    echo -e "${RED}Error: BLOB_READ_WRITE_TOKEN environment variable is not set!${NC}"
    echo -e "Please set it with: export BLOB_READ_WRITE_TOKEN=\"your_token_here\""
    echo -e "Or run setup-env.sh if available."
    exit 1
  fi
  
  if [ -z "$BLOB_URL" ]; then
    echo -e "${YELLOW}Warning: BLOB_URL environment variable is not set. Using default.${NC}"
    export BLOB_URL="https://mzcmzrmjairlp3km.public.blob.vercel-storage.com"
  fi
fi

# Ensure we have the required packages installed
if ! command -v node &> /dev/null; then
  echo -e "${RED}Error: Node.js is not installed!${NC}"
  echo -e "Please install Node.js before running this script."
  exit 1
fi

# Check if @vercel/blob is installed
if ! npm list @vercel/blob &> /dev/null; then
  echo -e "${YELLOW}@vercel/blob not found, installing...${NC}"
  npm install @vercel/blob
fi

# Check if tools directory exists
if [ ! -d "./tools" ]; then
  echo -e "${YELLOW}Creating tools directory...${NC}"
  mkdir -p tools
fi

# Make migration script executable
if [ -f "./tools/migrate-local-cache.js" ]; then
  chmod +x ./tools/migrate-local-cache.js
else
  echo -e "${RED}Error: Migration script not found!${NC}"
  echo -e "Please make sure tools/migrate-local-cache.js exists."
  exit 1
fi

# Echo environment variables (without showing the token)
echo -e "${GREEN}Using the following configuration:${NC}"
echo -e "  BLOB_READ_WRITE_TOKEN: [HIDDEN]"
echo -e "  BLOB_URL: ${BLOB_URL}"

# Run the migration script
echo -e "\n${BLUE}Starting migration process...${NC}"
node ./tools/migrate-local-cache.js

# Check the result
if [ $? -eq 0 ]; then
  echo -e "\n${GREEN}Migration completed successfully!${NC}"
else
  echo -e "\n${RED}Migration failed!${NC}"
  exit 1
fi

echo -e "\n${BLUE}You can now use Vercel Blob Storage for article vectorization and interest matching.${NC}"
echo -e "${BLUE}Your previous local cache files have been migrated to the cloud.${NC}" 