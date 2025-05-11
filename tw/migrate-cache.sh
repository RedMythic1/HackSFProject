#!/bin/bash

# migrate-cache.sh - Migrate local cache files to Vercel Blob Storage

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo -e "${BLUE}=== Migrating Local Cache to Vercel Blob Storage ===${NC}"

# Make sure environment variables are set
if [ -f ".env" ]; then
  echo -e "${GREEN}Loading environment variables from .env file...${NC}"
  export $(grep -v '^#' .env | xargs)
elif [ -z "$BLOB_READ_WRITE_TOKEN" ]; then
  echo -e "${YELLOW}No .env file found and BLOB_READ_WRITE_TOKEN is not set.${NC}"
  echo -e "${YELLOW}Setting default Vercel Blob token...${NC}"
  export BLOB_READ_WRITE_TOKEN="vercel_blob_rw_MzCMzRmJaiRlp3km_L5RVXS9InB9rTT1Aov2ZI4kzQFoT5S"
  export BLOB_URL="https://mzcmzrmjairlp3km.public.blob.vercel-storage.com"
fi

echo -e "${GREEN}Using Blob Storage URL: ${BLOB_URL}${NC}"
echo -e "${GREEN}Using Blob Storage Token: [Secret]${NC}"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
  echo -e "${RED}Error: Node.js is required but not installed.${NC}"
  echo -e "${YELLOW}Please install Node.js and try again.${NC}"
  exit 1
fi

# Check for required dependencies
if [ ! -f "package.json" ]; then
  echo -e "${RED}Error: package.json not found. Are you in the correct directory?${NC}"
  exit 1
fi

# Install required packages if not already installed
if ! npm list @vercel/blob &> /dev/null; then
  echo -e "${YELLOW}Installing @vercel/blob package...${NC}"
  npm install --save @vercel/blob
fi

if ! npm list glob &> /dev/null; then
  echo -e "${YELLOW}Installing glob package...${NC}"
  npm install --save glob
fi

# Check if the migration script exists
if [ ! -f "tools/migrate-to-blob.js" ]; then
  echo -e "${RED}Error: Migration script not found at tools/migrate-to-blob.js${NC}"
  exit 1
fi

# Run the migration script
echo -e "${BLUE}Starting migration process...${NC}"
node tools/migrate-to-blob.js

# Check if the migration was successful
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Migration completed successfully!${NC}"
  echo -e "${BLUE}Your local cache files have been migrated to Vercel Blob Storage.${NC}"
  echo -e "${BLUE}You can now use Vercel Blob Storage for your cache files.${NC}"
else
  echo -e "${RED}Migration failed. Please check the error messages above.${NC}"
  exit 1
fi 