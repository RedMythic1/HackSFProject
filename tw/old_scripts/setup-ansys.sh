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
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Check if ansys.py exists in the parent directory
if [ -f "$PROJECT_ROOT/ansys.py" ]; then
    print_status "Found ansys.py in parent directory. Copying to current directory..."
    cp "$PROJECT_ROOT/ansys.py" "$SCRIPT_DIR/"
    print_success "Copied ansys.py to $SCRIPT_DIR"
else
    print_error "ansys.py not found in $PROJECT_ROOT"
    
    # Look for ansys.py in common locations
    print_status "Looking for ansys.py in other locations..."
    
    if [ -f "$HOME/Code/HackSFProject/ansys.py" ]; then
        print_status "Found ansys.py in $HOME/Code/HackSFProject. Copying..."
        cp "$HOME/Code/HackSFProject/ansys.py" "$SCRIPT_DIR/"
        print_success "Copied ansys.py to $SCRIPT_DIR"
    else
        print_error "Could not find ansys.py in common locations."
        print_status "Please copy ansys.py to $SCRIPT_DIR manually."
        exit 1
    fi
fi

# Create a .cache directory if it doesn't exist
print_status "Creating cache directory..."
mkdir -p "$SCRIPT_DIR/.cache"
print_success "Cache directory created at $SCRIPT_DIR/.cache"

print_success "Setup complete! You can now run the server which will be able to find ansys.py."
print_status "Run ./init.sh to start the server and frontend." 