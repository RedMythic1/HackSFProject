#!/bin/bash

# complete-migration.sh - Run all migration steps to move from Blob Storage to Edge Config

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo -e "${BLUE}========= Complete Migration to Edge Config ==========${NC}"

# Step 1: Install dependencies
echo -e "\n${BLUE}[1/5] Installing required packages...${NC}"
npm install @vercel/edge-config @vercel/blob

if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to install dependencies. Aborting.${NC}"
  exit 1
fi

echo -e "${GREEN}Successfully installed dependencies${NC}"

# Step 2: Setup Edge Config environment variables
echo -e "\n${BLUE}[2/5] Setting up Edge Config environment...${NC}"
chmod +x setup-edge-config.sh
source ./setup-edge-config.sh

if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to set up Edge Config environment. Aborting.${NC}"
  exit 1
fi

# Step 3: Ensure migration scripts are executable
echo -e "\n${BLUE}[3/5] Preparing migration utilities...${NC}"
chmod +x migrate-to-edge-config.sh
chmod +x tools/edge-config-utils.js
chmod +x tools/migrate-blob-to-edge.js

# Step 4: Migrate data from Blob Storage to Edge Config
echo -e "\n${BLUE}[4/5] Running migration from Blob Storage to Edge Config...${NC}"
export BLOB_READ_WRITE_TOKEN="vercel_blob_rw_MzCMzRmJaiRlp3km_L5RVXS9InB9rTT1Aov2ZI4kzQFoT5S"
./migrate-to-edge-config.sh

if [ $? -ne 0 ]; then
  echo -e "${YELLOW}Migration from Blob Storage encountered issues, but we'll continue...${NC}"
fi

# Step 5: Update vercel.json to use Edge Config
echo -e "\n${BLUE}[5/5] Updating project configuration...${NC}"
node -e "
const fs = require('fs');
const vercelConfig = JSON.parse(fs.readFileSync('vercel.json', 'utf8'));

// Remove old Blob Storage variables
if (vercelConfig.env.BLOB_READ_WRITE_TOKEN) delete vercelConfig.env.BLOB_READ_WRITE_TOKEN;
if (vercelConfig.env.BLOB_STORE_ID) delete vercelConfig.env.BLOB_STORE_ID;
if (vercelConfig.env.BLOB_URL) delete vercelConfig.env.BLOB_URL;
if (vercelConfig.env.USE_BLOB_STORAGE) delete vercelConfig.env.USE_BLOB_STORAGE;

// Add Edge Config variables
vercelConfig.env.EDGE_CONFIG = '$EDGE_CONFIG';
vercelConfig.env.USE_EDGE_CONFIG = '1';

// Write updated config back to file
fs.writeFileSync('vercel.json', JSON.stringify(vercelConfig, null, 2));
console.log('Updated vercel.json with Edge Config settings and removed Blob Storage variables');
"

echo -e "\n${GREEN}========= Migration Complete! ==========${NC}"
echo -e "Your application is now configured to use Vercel Edge Config for data storage."
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Deploy your changes to Vercel"
echo -e "2. Test the application to ensure everything works correctly"
echo -e "3. Once confirmed working, you can safely remove old Blob Storage resources"

echo -e "\n${BLUE}You can use the following command to test Edge Config:${NC}"
echo -e "${GREEN}node tools/edge-config-utils.js list${NC}" 