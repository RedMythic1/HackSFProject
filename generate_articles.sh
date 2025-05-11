#!/bin/bash
#bombolcato       w w w wq  w wqw wq  w w  qw qw   qw q
# generate_articles.sh - Command-line utility for generating tech deep dive articles
# 
# This script provides a simple way to generate articles using ansys.py
# without going through the web interface.

# Set variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANSYS_PATH="${SCRIPT_DIR}/ansys.py"
OUTPUT_DIR="${SCRIPT_DIR}"
CACHE_DIR="${SCRIPT_DIR}/.cache"

# Create cache directory if it doesn't exist
mkdir -p "$CACHE_DIR"

# Display help information
show_help() {
  echo "Usage: $0 [options]"
  echo
  echo "Generate tech deep dive articles using ansys.py"
  echo
  echo "Options:"
  echo "  -h, --help             Show this help message"
  echo "  -c, --cache-only       Only cache articles, don't generate content"
  echo "  -i, --interests TEXT   Specify interests (comma-separated)"
  echo "  -s, --skip-scoring     Skip article scoring step (faster)"
  echo "  -p, --process          Process existing cached articles (runs with default interests)"
  echo "  -f, --force            Force processing of all articles, even if already in cache"
  echo
  echo "Examples:"
  echo "  $0 --interests \"AI, machine learning, programming\""
  echo "  $0 --cache-only"
  echo "  $0 --skip-scoring"
}

# Default values
INTERESTS="technology, programming, science, AI, machine learning, data science"
CACHE_ONLY=0
SKIP_SCORING=0
PROCESS_ONLY=0
FORCE_PROCESSING=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -c|--cache-only)
      CACHE_ONLY=1
      shift
      ;;
    -i|--interests)
      INTERESTS="$2"
      shift 2
      ;;
    -s|--skip-scoring)
      SKIP_SCORING=1
      shift
      ;;
    -p|--process)
      PROCESS_ONLY=1
      shift
      ;;
    -f|--force)
      FORCE_PROCESSING=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if ansys.py exists
if [[ ! -f "$ANSYS_PATH" ]]; then
  echo "Error: ansys.py not found at $ANSYS_PATH"
  exit 1
fi

# Function to check for existing final articles
check_existing_articles() {
  local count=0
  
  # Look for final_article_*.json files in cache
  if [[ -d "$CACHE_DIR" ]]; then
    count=$(find "$CACHE_DIR" -name "final_article_*.json" | wc -l | tr -d '[:space:]')
    
    # List the first few articles if they exist
    if [[ $count -gt 0 ]]; then
      echo "Found $count existing final articles in cache."
      echo "Examples:"
      find "$CACHE_DIR" -name "final_article_*.json" | head -3 | while read file; do
        echo "  - $(basename "$file")"
      done
      
      if [[ $FORCE_PROCESSING -eq 0 ]]; then
        echo "Will skip processing for these articles. Use --force to reprocess them."
      else
        echo "Force processing enabled - will reprocess all articles."
      fi
    else
      echo "No existing final articles found in cache."
    fi
  else
    echo "Cache directory not found."
  fi
  
  return $count
}

# Create processed IDs file to pass to ansys.py
create_processed_ids_file() {
  local ids_file="/tmp/processed_article_ids_$(date +%s).json"
  
  if [[ $FORCE_PROCESSING -eq 1 ]]; then
    # Empty array when forcing reprocessing
    echo '{"processed_ids": []}' > "$ids_file"
    echo "Created empty processed IDs file (force mode)"
  else
    # Extract IDs from existing final articles
    echo '{"processed_ids": [' > "$ids_file"
    
    # Find all final article files and extract their IDs
    local first=1
    find "$CACHE_DIR" -name "final_article_*.json" | while read file; do
      local id=$(basename "$file" | sed -E 's/final_article_([0-9]+).*/\1/')
      if [[ -n "$id" ]]; then
        if [[ $first -eq 1 ]]; then
          echo "\"$id\"" >> "$ids_file"
          first=0
        else
          echo ", \"$id\"" >> "$ids_file"
        fi
      fi
    done
    
    echo ']}' >> "$ids_file"
    echo "Created processed IDs file with existing article IDs"
  fi
  
  echo "$ids_file"
}

# Display current settings
echo "========================================"
echo "Article Generation Settings:"
echo "========================================"
echo "Using ansys.py at: $ANSYS_PATH"
echo "Cache directory: $CACHE_DIR"
echo "Interests: $INTERESTS"
echo "Cache only mode: $([[ $CACHE_ONLY -eq 1 ]] && echo "Yes" || echo "No")"
echo "Skip scoring: $([[ $SKIP_SCORING -eq 1 ]] && echo "Yes" || echo "No")"
echo "Process cache only: $([[ $PROCESS_ONLY -eq 1 ]] && echo "Yes" || echo "No")"
echo "Force processing: $([[ $FORCE_PROCESSING -eq 1 ]] && echo "Yes" || echo "No")"
echo "========================================"

# Check for existing articles
if [[ $CACHE_ONLY -eq 0 ]]; then
  echo "Checking for existing final articles..."
  check_existing_articles
  
  # Create processed IDs file
  if [[ $FORCE_PROCESSING -eq 0 ]]; then
    PROCESSED_IDS_FILE=$(create_processed_ids_file)
    # Pass to ansys.py through environment variable
    export ANSYS_PROCESSED_IDS_FILE="$PROCESSED_IDS_FILE"
    echo "Using processed IDs file: $PROCESSED_IDS_FILE"
  else
    echo "Skipping processed IDs file (force processing enabled)"
  fi
fi

# Set up command based on options
CMD="python $ANSYS_PATH"

if [[ $CACHE_ONLY -eq 1 ]]; then
  CMD="$CMD --cache-only"
  echo "Running in cache-only mode"
elif [[ $PROCESS_ONLY -eq 1 ]]; then
  # For process-only mode, we use default interests and normal processing
  echo "Processing existing cached articles with default interests"
  INTERESTS="technology, programming, science"
fi

if [[ $SKIP_SCORING -eq 1 ]]; then
  # Use environment variable to skip scoring
  export ANSYS_NO_SCORE=1
  echo "Skipping article scoring"
fi

# Run the command
echo "Starting article generation..."
echo "Command: $CMD"
echo

if [[ $CACHE_ONLY -eq 1 ]]; then
  # No input needed for cache-only mode
  $CMD
else
  # Feed interests to ansys.py
  echo "$INTERESTS" | $CMD
fi

# Check result
STATUS=$?
if [[ $STATUS -eq 0 ]]; then
  echo "Article generation completed successfully!"
  # List generated articles
  HTML_FILES=$(find "$OUTPUT_DIR" -name "tech_deep_dive_*.html" -type f -mtime -1 | wc -l | tr -d '[:space:]')
  echo "Generated $HTML_FILES new article(s) in the last 24 hours"
  echo "Articles can be found in: $OUTPUT_DIR"
else
  echo "Article generation failed with status code $STATUS"
fi

# Clean up
if [[ -n "$PROCESSED_IDS_FILE" && -f "$PROCESSED_IDS_FILE" ]]; then
  rm -f "$PROCESSED_IDS_FILE"
  echo "Cleaned up temporary files"
fi

echo "Done!" 