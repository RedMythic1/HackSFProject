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

# Check cache status first
print_status "Checking current cache status..."
# Save response to a file first
curl -s http://localhost:5001/check-cache > /tmp/cache_response.json

# Extract article count with Python
ARTICLE_COUNT=$(python3 -c 'import json, sys; print(json.load(open("/tmp/cache_response.json")).get("article_count", 0))' 2>/dev/null)

print_status "Current cached articles: $ARTICLE_COUNT"

# Start question generation
print_status "Starting full question and answer generation..."
# Save response to a file first
curl -s http://localhost:5001/generate-questions > /tmp/questions_response.json

# Extract message with Python
MESSAGE=$(python3 -c 'import json, sys; print(json.load(open("/tmp/questions_response.json")).get("message", "Process started"))' 2>/dev/null)

print_success "$MESSAGE"
print_status "This process will run in the background and will take several minutes."
print_status "It will generate questions and answers for all cached articles."
print_status "You can check server.log for progress updates."

print_status "Test complete!" 