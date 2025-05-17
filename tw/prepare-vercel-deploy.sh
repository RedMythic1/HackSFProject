#!/bin/bash

# Script to prepare for Vercel deployment by removing conflicting files
# from the Python packages directory

echo "Starting preparation for Vercel deployment..."

# Step 1: Install npm dependencies
echo "Ensuring npm dependencies are installed..."
if [ ! -d "./node_modules" ] || [ ! -f "./node_modules/.bin/webpack" ]; then
    echo "Installing npm dependencies..."
    npm install
    if [ ! -f "./node_modules/.bin/webpack" ]; then
        echo "Webpack not found after npm install, installing manually..."
        npm install --save-dev webpack webpack-cli
    fi
else
    echo "npm dependencies already installed"
fi

# Step 2: Create Python runtime config for Vercel
echo "Creating Python runtime configuration for Vercel..."
cat > runtime.txt << EOF
python-3.9
EOF
echo "Created runtime.txt specifying Python 3.9"

# Step 3: Remove .pyi files
echo "Removing .pyi files that can conflict with .py files..."
PYI_COUNT=$(find ./api/python_packages -name "*.pyi" | wc -l)
echo "Found $PYI_COUNT .pyi files to remove"
find ./api/python_packages -name "*.pyi" -type f -delete
NEW_PYI_COUNT=$(find ./api/python_packages -name "*.pyi" | wc -l)
echo "Removed $(($PYI_COUNT - $NEW_PYI_COUNT)) .pyi files"

# Step 4: Remove .pxd files (Cython definition files)
echo "Removing .pxd files..."
PXD_COUNT=$(find ./api/python_packages -name "*.pxd" | wc -l)
echo "Found $PXD_COUNT .pxd files to remove"
find ./api/python_packages -name "*.pxd" -type f -delete
NEW_PXD_COUNT=$(find ./api/python_packages -name "*.pxd" | wc -l)
echo "Removed $(($PXD_COUNT - $NEW_PXD_COUNT)) .pxd files"

# Step 5: Remove .hash directories and files (not needed for runtime)
echo "Removing .hash directories and hash files..."
HASH_DIR_COUNT=$(find ./api/python_packages -type d -name ".hash" | wc -l)
echo "Found $HASH_DIR_COUNT .hash directories to remove"
find ./api/python_packages -type d -name ".hash" -exec rm -rf {} \; 2>/dev/null || true
NEW_HASH_DIR_COUNT=$(find ./api/python_packages -type d -name ".hash" | wc -l)
echo "Removed $(($HASH_DIR_COUNT - $NEW_HASH_DIR_COUNT)) .hash directories"

# Step 6: Check for any remaining potential filename conflicts
echo "Checking for remaining potential filename conflicts..."

# Find all Python files
PY_FILES=$(find ./api/python_packages -name "*.py" | sort)

# Known extensions that can coexist with .py files
SAFE_EXTENSIONS="-not -name *.py -not -name *.pyc -not -name *.pyo -not -name *.pyd -not -name *.so"

# Check each Python file for a matching filename with a different extension
CONFLICTS_FOUND=0
for pyfile in $PY_FILES; do
    base=$(basename "$pyfile" .py)
    dir=$(dirname "$pyfile")
    
    # Check for files with the same name but different extensions in the same directory
    conflicts=$(find "$dir" -type f $SAFE_EXTENSIONS -name "$base.*" | wc -l)
    
    if [ $conflicts -gt 0 ]; then
        CONFLICTS_FOUND=1
        echo "Warning: Potential conflict for $pyfile"
        find "$dir" -type f $SAFE_EXTENSIONS -name "$base.*" -exec echo "  - Conflicting file: {}" \;
    fi
done

if [ $CONFLICTS_FOUND -eq 0 ]; then
    echo "No remaining conflicts found!"
else
    echo "Warning: Some potential conflicts remain. Deployment may still fail."
fi

# Create a .vercel/project.json file to specify no pip usage
mkdir -p .vercel
cat > .vercel/project.json << EOF
{
  "buildCommand": null,
  "framework": null,
  "installCommand": null,
  "devCommand": null
}
EOF
echo "Created .vercel/project.json to prevent automatic pip installation"

echo "Preparation complete. You can now deploy to Vercel." 