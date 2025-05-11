#!/bin/bash

# cleanup.sh - Script to clean up duplicates and maintain consistency

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Project Cleanup Script ===${NC}"

# Ensure .cache directory exists in both parent and tw directories
if [ ! -d ".cache" ]; then
    mkdir -p .cache
    echo -e "${GREEN}Created parent .cache directory${NC}"
fi

if [ ! -d "tw/.cache" ]; then
    mkdir -p tw/.cache
    echo -e "${GREEN}Created tw/.cache directory${NC}"
fi

# Synchronize cache files (parent to tw)
echo -e "${BLUE}Synchronizing cache files...${NC}"
rsync -av --update .cache/ tw/.cache/
echo -e "${GREEN}Cache directories synchronized${NC}"

# Ensure final_articles directory exists
if [ ! -d "final_articles" ]; then
    mkdir -p final_articles/html
    mkdir -p final_articles/markdown
    echo -e "${GREEN}Created final_articles directories${NC}"
fi

# Ensure public/articles directory exists
if [ ! -d "tw/public/articles" ]; then
    mkdir -p tw/public/articles
    echo -e "${GREEN}Created public/articles directory${NC}"
fi

# Copy HTML articles to public directory
echo -e "${BLUE}Copying HTML articles to public directory...${NC}"
if [ -d "final_articles/html" ]; then
    cp -f final_articles/html/*.html tw/public/articles/ 2>/dev/null || :
    echo -e "${GREEN}HTML articles copied to public directory${NC}"
fi

# Check for and remove duplicate/backup files
echo -e "${BLUE}Checking for duplicate/backup files...${NC}"

# Check for server.py.backup
if [ -f "tw/server.py.backup" ]; then
    rm tw/server.py.backup
    echo -e "${GREEN}Removed tw/server.py.backup${NC}"
fi

# Check for backend.js (should use server.py instead)
if [ -f "tw/backend.js" ]; then
    rm tw/backend.js
    echo -e "${GREEN}Removed tw/backend.js${NC}"
fi

echo -e "${GREEN}Cleanup complete!${NC}"
echo -e "${BLUE}To start the web application, run: cd tw && ./start.sh${NC}" 