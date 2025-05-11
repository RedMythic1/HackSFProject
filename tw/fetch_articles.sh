#!/bin/bash

# fetch_articles.sh - Script to fetch articles from HackerNews using ansys_local.py
# This script uses the local version of ansys.py to avoid external dependencies

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CACHE_DIR="$(pwd)/local_cache"
HTML_DIR="$(pwd)/public/articles"
ANSYS_PATH="$(pwd)/ansys_local.py"
PROCESSED_IDS_FILE="${CACHE_DIR}/processed_article_ids.json"

echo -e "${BLUE}=== HackerNews Article Fetcher ===${NC}"
echo -e "${BLUE}Cache directory: ${CACHE_DIR}${NC}"
echo -e "${BLUE}HTML directory: ${HTML_DIR}${NC}"
echo -e "${BLUE}Ansys path: ${ANSYS_PATH}${NC}"

# Create directories if they don't exist
mkdir -p "${CACHE_DIR}"
mkdir -p "${HTML_DIR}"

# Check if ansys_local.py exists
if [ ! -f "${ANSYS_PATH}" ]; then
    echo -e "${RED}Error: ansys_local.py not found at ${ANSYS_PATH}${NC}"
    echo -e "${YELLOW}Please make sure ansys_local.py is in the current directory.${NC}"
    exit 1
fi

# Function to run ansys_local.py with caching
run_ansys_cache() {
    echo -e "${GREEN}Starting article caching process...${NC}"
    
    # Set environment variables for ansys.py
    export ANSYS_NO_SCORE=1
    export ANSYS_PROCESSED_IDS_FILE="${PROCESSED_IDS_FILE}"
    
    # Run ansys_local.py with --cache-only flag
    python3 "${ANSYS_PATH}" --cache-only
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Article caching completed successfully${NC}"
    else
        echo -e "${RED}Error: Article caching failed${NC}"
        exit 1
    fi
}

# Function to run ansys_local.py with full processing
run_ansys_full() {
    echo -e "${GREEN}Starting full article processing...${NC}"
    
    # Create a temporary file for user interests
    INTERESTS_FILE=$(mktemp)
    
    # Check if interests were provided as arguments
    if [ $# -gt 0 ]; then
        echo "$@" > "${INTERESTS_FILE}"
    else
        # Default interests if none provided
        echo "technology, programming, AI, machine learning" > "${INTERESTS_FILE}"
    fi
    
    echo -e "${BLUE}Using interests: $(cat ${INTERESTS_FILE})${NC}"
    
    # Set environment variables for ansys_local.py
    export ANSYS_PROCESSED_IDS_FILE="${PROCESSED_IDS_FILE}"
    
    # Run ansys_local.py with the interests file
    cat "${INTERESTS_FILE}" | python3 "${ANSYS_PATH}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Full article processing completed successfully${NC}"
        
        # Copy HTML files to public directory
        echo -e "${BLUE}Copying HTML files to public directory...${NC}"
        for html_file in tech_deep_dive_*.html; do
            if [ -f "$html_file" ]; then
                cp "$html_file" "${HTML_DIR}/"
                echo -e "${GREEN}Copied ${html_file} to ${HTML_DIR}/${NC}"
            fi
        done
    else
        echo -e "${RED}Error: Full article processing failed${NC}"
        exit 1
    fi
    
    # Clean up
    rm "${INTERESTS_FILE}"
}

# Function to show help
show_help() {
    echo -e "${BLUE}Usage: $0 [command] [options]${NC}"
    echo
    echo -e "${GREEN}Commands:${NC}"
    echo "  cache              Cache articles only (no processing)"
    echo "  process [interests] Process articles with optional interests"
    echo "  help               Show this help message"
    echo
    echo -e "${GREEN}Examples:${NC}"
    echo "  $0 cache"
    echo "  $0 process \"AI, machine learning, web development\""
    echo
}

# Process arguments
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    cache)
        run_ansys_cache
        ;;
    process)
        shift
        run_ansys_full "$@"
        ;;
    help)
        show_help
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$1'${NC}"
        show_help
        exit 1
        ;;
esac

echo -e "${BLUE}=== Complete ===${NC}"
exit 0 