#!/bin/bash

# upload-cache.sh - Upload local cache files to Vercel Blob Storage

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo -e "${BLUE}=== Uploading Local Cache to Vercel Blob Storage ===${NC}"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
  echo -e "${RED}Error: Node.js is required but not installed.${NC}"
  echo -e "${YELLOW}Please install Node.js and try again.${NC}"
  exit 1
fi

# Check if the upload script exists
if [ ! -f "tools/upload-local-cache.js" ]; then
  echo -e "${RED}Error: Upload script not found at tools/upload-local-cache.js${NC}"
  exit 1
fi

# Check if the local_cache directory exists
if [ ! -d "local_cache" ]; then
  echo -e "${YELLOW}Warning: local_cache directory not found${NC}"
  echo -e "Do you want to create it? (y/n)"
  read -r response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    mkdir -p local_cache
    echo -e "${GREEN}Created local_cache directory${NC}"
  else
    echo -e "${RED}Aborting: local_cache directory is required${NC}"
    exit 1
  fi
fi

# Check if BLOB_READ_WRITE_TOKEN is set
if [ -z "$BLOB_READ_WRITE_TOKEN" ]; then
  echo -e "${YELLOW}Warning: BLOB_READ_WRITE_TOKEN environment variable is not set${NC}"
  echo -e "This token is required for uploading to Vercel Blob Storage."
  echo -e "You can set it with: export BLOB_READ_WRITE_TOKEN=\"your-token-here\""
  echo -e "Do you want to continue anyway? (y/n)"
  read -r response
  if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${RED}Aborting upload process${NC}"
    exit 1
  fi
fi

# Run the upload script
echo -e "${BLUE}Starting upload process...${NC}"
node tools/upload-local-cache.js

# Check if the upload was successful
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Upload process completed${NC}"
  echo -e "${BLUE}You can view detailed results in upload-results.json${NC}"
else
  echo -e "${RED}Upload process failed${NC}"
  exit 1
fi

echo -e "${GREEN}Done!${NC}" 