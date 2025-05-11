#!/bin/bash

# migrate-to-edge-config.sh - Migrate data from Vercel Blob Storage to Edge Config

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo -e "${BLUE}=== Migrating from Vercel Blob Storage to Edge Config ===${NC}"

# Load environment variables
source "$SCRIPT_DIR/setup-edge-config.sh" >/dev/null 2>&1 || true

# Ensure BLOB_READ_WRITE_TOKEN is set for Blob Storage access
if [ -z "$BLOB_READ_WRITE_TOKEN" ]; then
  echo -e "${YELLOW}BLOB_READ_WRITE_TOKEN is not set. Please provide it now:${NC}"
  read -r BLOB_READ_WRITE_TOKEN
  export BLOB_READ_WRITE_TOKEN
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
  echo -e "${RED}Error: Node.js is required but not installed.${NC}"
  echo -e "${YELLOW}Please install Node.js and try again.${NC}"
  exit 1
fi

# Check if @vercel/blob and @vercel/edge-config are installed
if ! node -e "require('@vercel/blob'); require('@vercel/edge-config');" &> /dev/null; then
  echo -e "${YELLOW}Installing required packages...${NC}"
  npm install @vercel/blob @vercel/edge-config
fi

# Make the migration tool executable
chmod +x tools/migrate-blob-to-edge.js

# Run the migration
echo -e "${BLUE}Starting migration process...${NC}"
node tools/migrate-blob-to-edge.js all

if [ $? -eq 0 ]; then
  echo -e "\n${GREEN}Migration completed successfully!${NC}"
  
  # Ask if user wants to test Edge Config
  echo -e "\n${YELLOW}Do you want to test Edge Config by reading some data? (y/n)${NC}"
  read -r response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    # Make the edge-config-utils.js executable
    chmod +x tools/edge-config-utils.js
    
    echo -e "${BLUE}Testing Edge Config by listing article keys...${NC}"
    node tools/edge-config-utils.js list articles/final_article_
    
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}Edge Config test successful${NC}"
    else
      echo -e "${RED}Edge Config test failed${NC}"
    fi
  fi
  
  # Update vercel.json with Edge Config
  echo -e "\n${BLUE}Updating vercel.json with Edge Config settings...${NC}"
  if [ -f "vercel.json" ]; then
    # Backup original file
    cp vercel.json vercel.json.bak
    
    # Update vercel.json to use Edge Config
    node -e "
    const fs = require('fs');
    const vercelConfig = JSON.parse(fs.readFileSync('vercel.json', 'utf8'));
    
    // Add or update Edge Config env variable
    vercelConfig.env = {
      ...vercelConfig.env,
      EDGE_CONFIG: '$EDGE_CONFIG',
      USE_EDGE_CONFIG: '1'
    };
    
    // Write updated config back to file
    fs.writeFileSync('vercel.json', JSON.stringify(vercelConfig, null, 2));
    console.log('Updated vercel.json with Edge Config settings');
    "
    
    echo -e "${GREEN}Successfully updated vercel.json${NC}"
    echo -e "${YELLOW}Original file backed up as vercel.json.bak${NC}"
  else
    echo -e "${RED}vercel.json not found${NC}"
  fi
  
  echo -e "\n${BLUE}Migration to Edge Config is complete.${NC}"
  echo -e "${GREEN}Your application is now configured to use Edge Config for storage.${NC}"
  echo -e "${YELLOW}Remember to deploy your changes to Vercel for the new configuration to take effect.${NC}"
else
  echo -e "\n${RED}Migration failed${NC}"
  exit 1
fi 