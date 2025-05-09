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

# Get absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TW_DIR="$SCRIPT_DIR/tw"

# Check Python installation
print_status "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Found $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.8+ and try again."
    exit 1
fi

# Check Node.js installation
print_status "Checking Node.js installation..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Found Node.js $NODE_VERSION"
else
    print_error "Node.js not found. Please install Node.js 14+ and try again."
    exit 1
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
if ! pip install -r "$SCRIPT_DIR/requirements.txt"; then
    print_error "Failed to install Python dependencies."
    exit 1
fi
print_success "Python dependencies installed successfully."

# Install Node.js dependencies for web application
if [ -d "$TW_DIR" ]; then
    print_status "Installing Node.js dependencies for web application..."
    cd "$TW_DIR"
    if ! npm install; then
        print_error "Failed to install Node.js dependencies."
        exit 1
    fi
    print_success "Node.js dependencies installed successfully."
else
    print_error "Web application directory not found at: $TW_DIR"
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p "$SCRIPT_DIR/.cache"
mkdir -p "$SCRIPT_DIR/final_articles/markdown"
mkdir -p "$SCRIPT_DIR/final_articles/html"
mkdir -p "$TW_DIR/.cache"
mkdir -p "$TW_DIR/user_data"
mkdir -p "$TW_DIR/dist"
print_success "Directories created successfully."

# Verify Llama model availability
print_status "Checking Llama model availability..."
MODEL_PATH="$HOME/llama-models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
if [ -f "$MODEL_PATH" ]; then
    print_success "Found Llama model at: $MODEL_PATH"
else
    print_error "Llama model not found at: $MODEL_PATH"
    print_status "You will need to download a Llama model and update the path in ansys.py"
    print_status "Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main"
fi

# Make the web application management script executable
print_status "Making web application management script executable..."
chmod +x "$TW_DIR/manage.sh"
print_success "Script permissions set."

print_success "Installation complete!"
print_status "To start the web application, run:"
print_status "cd tw && ./manage.sh start"
print_status "Then open http://localhost:9000 in your browser."

print_status "For the command-line version, run:"
print_status "python ansys.py" 