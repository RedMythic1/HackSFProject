#!/bin/bash

# Print colorful status messages
print_status() {
    echo -e "\033[1;34m==>\033[0m $1"
}

print_error() {
    echo -e "\033[1;31mError:\033[0m $1"
}

print_success() {
    echo -e "\033[1;32mSuccess:\033[0m $1"
}

# First, check if the server is running
print_status "Checking if server is running..."

if curl -s http://localhost:5001/check-cache > /dev/null; then
    print_success "Server is running"
else
    print_error "Server is not running. Please start it with ./init.sh"
    exit 1
fi

# Check cache status
print_status "Checking current cache status..."
# Save response to a file first
curl -s http://localhost:5001/check-cache > /tmp/cache_response.json

# Extract article count with Python (more reliable for parsing JSON)
ARTICLE_COUNT=$(python3 -c 'import json, sys; print(json.load(open("/tmp/cache_response.json")).get("article_count", 0))' 2>/dev/null)

# Default to 0 if extraction failed
if [ -z "$ARTICLE_COUNT" ]; then
    print_error "Failed to extract article count. Check server logs for details."
    ARTICLE_COUNT="0"
fi

print_status "Current cached articles: $ARTICLE_COUNT"

# Run cache-articles endpoint
print_status "Starting article caching and question generation..."
# Save response to a file first
curl -s http://localhost:5001/cache-articles > /tmp/cache_start_response.json

# Extract message with Python
MESSAGE=$(python3 -c 'import json, sys; print(json.load(open("/tmp/cache_start_response.json")).get("message", "Process started"))' 2>/dev/null)

# Default message if extraction failed
if [ -z "$MESSAGE" ]; then
    MESSAGE="Process started"
fi

print_success "$MESSAGE"
print_status "This process will continue in the background."
print_status "You can check the status by running this script again later to see if the article count has increased."

print_status "Test complete!" 