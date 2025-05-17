#!/bin/bash
# Final cleanup and restructuring for Fly.io deployment

echo "Starting final cleanup and restructuring..."

# Create a clean structure
mkdir -p clean_project/{dist,models,api,src}

# Copy only essential files to the clean structure
echo "Copying essential files to clean structure..."

# Copy configuration files
cp Dockerfile clean_project/
cp fly.toml clean_project/
cp .dockerignore clean_project/
cp start_fly.sh clean_project/
cp server.py clean_project/
cp requirements-fly.txt clean_project/
cp README.md clean_project/

# Copy package.json files
cp package.json clean_project/
cp tw/package.json clean_project/tw-package.json

# Copy source files
cp -r tw/src/* clean_project/src/
cp tw/webpack.config.js clean_project/
cp tw/tsconfig.json clean_project/

# Copy build output if it exists
if [ -d "tw/dist" ]; then
  cp -r tw/dist/* clean_project/dist/
else
  echo "Warning: No dist directory found. You may need to build the project."
fi

# Copy important API files
mkdir -p clean_project/api/python_packages
if [ -d "tw/api/python_packages" ]; then
  echo "Copying Python packages..."
  cp -r tw/api/python_packages/* clean_project/api/python_packages/
fi

if [ -d "tw/api" ]; then
  cp tw/api/*.js clean_project/api/ 2>/dev/null || echo "No JS files in api directory"
  cp tw/api/*.py clean_project/api/ 2>/dev/null || echo "No Python files in api directory"
fi

# Rename the old project directory
echo "Moving directories..."
mv tw tw.old
mv clean_project tw

echo "Final cleanup complete!"
echo "The old files are saved in tw.old directory. You can delete it if everything works correctly." 