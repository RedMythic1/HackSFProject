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

print_help() {
    echo "Tech Deep Dive Article Manager"
    echo ""
    echo "Usage: ./manage.sh COMMAND"
    echo ""
    echo "Commands:"
    echo "  start       Start both server and frontend"
    echo "  stop        Stop running server and frontend processes"
    echo "  setup       Set up the environment (copy ansys.py, create cache dirs)"
    echo "  cache       Cache articles from Hacker News"
    echo "  questions   Generate questions for cached articles"
    echo "  status      Check the status of the cache and server"
    echo "  help        Show this help message"
    echo ""
}

# Get absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Check if server is running
check_server() {
    if curl -s http://localhost:5001/check-cache > /dev/null; then
        return 0  # Server is running
    else
        return 1  # Server is not running
    fi
}

# Setup environment and dependencies
setup_env() {
    print_status "Setting up environment..."

    # Create necessary directories
    mkdir -p "$SCRIPT_DIR/.cache"
    mkdir -p "$SCRIPT_DIR/user_data"
    mkdir -p "$SCRIPT_DIR/dist"
    
    # Check if ansys.py exists in the parent directory
    if [ -f "$PROJECT_ROOT/ansys.py" ]; then
        print_success "Found ansys.py in parent directory"
    else
        print_error "ansys.py not found in $PROJECT_ROOT"
        
        # Look for ansys.py in common locations
        print_status "Looking for ansys.py in other locations..."
        
        if [ -f "$HOME/Code/HackSFProject/ansys.py" ]; then
            print_success "Found ansys.py in $HOME/Code/HackSFProject"
        else
            print_error "Could not find ansys.py in common locations."
            print_status "Please ensure ansys.py is available in the parent directory."
            return 1
        fi
    fi
    
    print_success "Setup complete!"
    return 0
}

# Start server and frontend
start_services() {
    print_status "Starting services..."
    
    # Check if already running
    if check_server; then
        print_status "Server is already running"
    else
        # Start the server
        print_status "Starting the server in simplified mode..."
        python3 server.py > server.log 2>&1 &
        SERVER_PID=$!
        
        # Wait for server to start (up to 10 seconds)
        print_status "Waiting for server to start..."
        for i in {1..10}; do
            if check_server; then
                print_success "Server is running"
                break
            fi
            if [ $i -eq 10 ]; then
                print_error "Server failed to start within 10 seconds"
                echo "Server logs:"
                cat server.log
                kill $SERVER_PID 2>/dev/null
                return 1
            fi
            sleep 1
        done
    fi
    
    # Start the frontend
    print_status "Starting the frontend..."
    npm start > frontend.log 2>&1 &
    FRONTEND_PID=$!
    
    # Wait for frontend to start (up to 10 seconds)
    print_status "Waiting for frontend to start..."
    for i in {1..10}; do
        if curl -s http://localhost:9000 > /dev/null 2>&1; then
            print_success "Frontend is running"
            break
        fi
        if [ $i -eq 10 ]; then
            print_error "Frontend failed to start within 10 seconds"
            echo "Frontend logs:"
            cat frontend.log
            kill $FRONTEND_PID 2>/dev/null
            return 1
        fi
        sleep 1
    done
    
    print_success "All services started successfully!"
    print_status "Server is running on http://localhost:5001"
    print_status "Frontend is running on http://localhost:9000"
    return 0
}

# Stop services
stop_services() {
    print_status "Stopping services..."
    
    # Find and kill Python server process
    pkill -f "python3 server.py" || print_status "No server process found"
    
    # Find and kill Node frontend process (npm)
    pkill -f "node.*tw/dist" || print_status "No frontend process found"
    
    print_success "Services stopped"
    return 0
}

# Cache articles
cache_articles() {
    # First, check if the server is running
    if ! check_server; then
        print_error "Server is not running. Please start it with: ./manage.sh start"
        return 1
    fi
    
    # Check cache status
    print_status "Checking current cache status..."
    # Save response to a file first
    curl -s http://localhost:5001/check-cache > /tmp/cache_response.json
    
    # Extract article count with Python
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
    print_status "You can check the status later with: ./manage.sh status"
    return 0
}

# Generate questions
generate_questions() {
    # First, check if the server is running
    if ! check_server; then
        print_error "Server is not running. Please start it with: ./manage.sh start"
        return 1
    fi
    
    # Check cache status first
    print_status "Checking current cache status..."
    # Save response to a file first
    curl -s http://localhost:5001/check-cache > /tmp/cache_response.json
    
    # Extract article count with Python
    ARTICLE_COUNT=$(python3 -c 'import json, sys; print(json.load(open("/tmp/cache_response.json")).get("article_count", 0))' 2>/dev/null)
    
    print_status "Current cached articles: $ARTICLE_COUNT"
    
    if [ "$ARTICLE_COUNT" -eq 0 ]; then
        print_error "No cached articles found. Please run './manage.sh cache' first."
        return 1
    fi
    
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
    return 0
}

# Check status of cache and server
check_status() {
    # Check if server is running
    if check_server; then
        print_success "Server is running"
        
        # Check cache status
        print_status "Checking cache status..."
        # Save response to a file first
        curl -s http://localhost:5001/check-cache > /tmp/cache_response.json
        
        # Extract with Python
        ARTICLE_COUNT=$(python3 -c 'import json, sys; print(json.load(open("/tmp/cache_response.json")).get("article_count", 0))' 2>/dev/null)
        FINAL_COUNT=$(python3 -c 'import json, sys; print(json.load(open("/tmp/cache_response.json")).get("final_article_count", 0))' 2>/dev/null)
        VALID_COUNT=$(python3 -c 'import json, sys; print(json.load(open("/tmp/cache_response.json")).get("valid_article_count", 0))' 2>/dev/null)
        
        print_status "Cached articles: $ARTICLE_COUNT"
        print_status "Final articles: $FINAL_COUNT"
        print_status "Valid articles: $VALID_COUNT"
        
        # Check if frontend is running
        if curl -s http://localhost:9000 > /dev/null 2>&1; then
            print_success "Frontend is running at http://localhost:9000"
        else
            print_error "Frontend is not running"
        fi
    else
        print_error "Server is not running"
    fi
    return 0
}

# Main command processor
if [ $# -eq 0 ]; then
    print_help
    exit 1
fi

case "$1" in
    setup)
        setup_env
        ;;
    start)
        setup_env && start_services
        ;;
    stop)
        stop_services
        ;;
    cache)
        cache_articles
        ;;
    questions)
        generate_questions
        ;;
    status)
        check_status
        ;;
    help)
        print_help
        ;;
    *)
        print_error "Unknown command: $1"
        print_help
        exit 1
        ;;
esac

exit $? 