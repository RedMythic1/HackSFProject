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
CACHE_RESPONSE=$(curl -s http://localhost:5001/check-cache)
ARTICLE_COUNT=$(echo $CACHE_RESPONSE | grep -o '"article_count":[0-9]*' | cut -d ':' -f 2)

print_status "Current cached articles: $ARTICLE_COUNT"

# Run cache-articles endpoint
print_status "Starting article caching and question generation..."
CACHE_START_RESPONSE=$(curl -s http://localhost:5001/cache-articles)
MESSAGE=$(echo $CACHE_START_RESPONSE | grep -o '"message":"[^"]*"' | cut -d '"' -f 4)

print_success "$MESSAGE"
print_status "This process will continue in the background."
print_status "You can check the status by running this script again later to see if the article count has increased."

print_status "Test complete!" 