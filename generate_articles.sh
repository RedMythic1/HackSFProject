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
echo "========================================"

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
  HTML_FILES=$(find "$OUTPUT_DIR" -name "tech_deep_dive_*.html" -type f -mtime -1 | wc -l)
  echo "Generated $HTML_FILES new article(s) in the last 24 hours"
  echo "Articles can be found in: $OUTPUT_DIR"
else
  echo "Article generation failed with status code $STATUS"
fi

echo "Done!" 