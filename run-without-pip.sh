#!/bin/bash

# Script to deploy to Vercel without using pip
echo "Deploying to Vercel without using pip..."

# Make sure we're in the correct directory
if [[ $(basename $(pwd)) != "tw" ]]; then
    echo "Error: This script should be run from the 'tw' directory"
    exit 1
fi

# Set up environment variables
source setup-env.sh

# Prepare files for deployment
echo "Preparing for deployment without pip..."

# Create required directories
mkdir -p .vercel
mkdir -p api/.vercel

# Create a stamp file to indicate Python packages are installed
echo "packages_installed" > api/.vercel/python_stamp
echo "3.9" > api/.vercel/python_version

# Create project.json that skips pip installation
cat > .vercel/project.json << EOF
{
  "buildCommand": "bash -c './build.sh'",
  "outputDirectory": "dist",
  "framework": null,
  "installCommand": "echo 'Skipping pip installation'",
  "devCommand": null,
  "public": true
}
EOF

# Create a local vercel.json override
cat > .vercel/vercel.json << EOF
{
  "version": 2,
  "builds": [
    {
      "src": "build.sh",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "dist"
      }
    }
  ],
  "env": {
    "PYTHON_VERSION": "3.9",
    "PYTHONPATH": "/var/task/api/python_packages:/var/task",
    "PIP_NO_DEPS": "1",
    "PIP_NO_INSTALL": "1"
  }
}
EOF

# Run the build script
echo "Running build script..."
./build.sh

# Deploy to Vercel with --force to override any previous settings
echo "Deploying to Vercel..."
vercel --prod --force \
  --env PYTHONPATH="/var/task/api/python_packages:/var/task" \
  --env PIP_NO_DEPS="1" \
  --env PIP_NO_INSTALL="1" \
  --build-env PYTHONPATH="/var/task/api/python_packages:/var/task" \
  --build-env PIP_NO_DEPS="1" \
  --build-env PIP_NO_INSTALL="1"

echo "Deployment command completed." 