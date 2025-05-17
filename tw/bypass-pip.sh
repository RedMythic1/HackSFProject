#!/bin/bash

# Script to deploy to Vercel with pip completely bypassed
echo "Deploying to Vercel with pip completely bypassed..."

# Ensure we're in the right directory
if [[ $(basename $(pwd)) != "tw" ]]; then
    echo "Error: This script should be run from the 'tw' directory"
    exit 1
fi

# Set up environment variables
source setup-env.sh

# Prepare the deployment
echo "Preparing for deployment..."

# Create required directories
mkdir -p .vercel
mkdir -p api/.vercel

# Create Python version marker for Vercel
echo "3.9" > api/.vercel/python_version

# Create dummy pip scripts that return success but do nothing
mkdir -p bin
cat > bin/pip3.9 << 'EOF'
#!/bin/bash
echo "Dummy pip3.9: Skipping package installation"
exit 0
EOF

cat > bin/pip << 'EOF'
#!/bin/bash
echo "Dummy pip: Skipping package installation"
exit 0
EOF

# Make the scripts executable
chmod +x bin/pip3.9
chmod +x bin/pip

# Add to .gitignore and .vercelignore
echo "/bin/pip*" >> .gitignore
echo "!/bin/pip*" >> .vercelignore

# Create project.json with special settings
cat > .vercel/project.json << 'EOF'
{
  "buildCommand": "export PATH=\"$PWD/bin:$PATH\" && bash -c 'echo \"Using custom pip wrapper\" && ./build.sh'",
  "outputDirectory": "dist",
  "framework": null,
  "installCommand": "echo 'Skipping pip installation'",
  "devCommand": null,
  "public": true
}
EOF

# Create a zero-sized requirements.txt
echo "# Packages pre-installed in api/python_packages" > requirements.txt

# Create a custom package.json to control the build process
cat > package.json.deploy << 'EOF'
{
  "name": "typescript-webpage",
  "version": "1.0.0",
  "description": "TypeScript-based web page",
  "main": "src/index.ts",
  "scripts": {
    "start": "./start.sh",
    "build": "./node_modules/.bin/webpack --mode production",
    "vercel-build": "export PATH=\"$PWD/bin:$PATH\" && echo \"Using custom PATH for build: $PATH\" && ./build.sh"
  },
  "dependencies": {
    "express": "^4.18.2",
    "webpack": "^5.88.2",
    "webpack-cli": "^5.1.4",
    "style-loader": "^3.3.3",
    "css-loader": "^6.8.1",
    "ts-loader": "^9.4.4",
    "typescript": "^5.2.2",
    "@vercel/blob": "^0.16.1"
  },
  "vercel": {
    "installCommand": "echo 'Skipping pip installation'",
    "buildCommand": "export PATH=\"$PWD/bin:$PATH\" && bash -c 'echo \"Using custom pip wrapper\" && ./build.sh'",
    "outputDirectory": "dist"
  }
}
EOF

# Run the build script 
echo "Running build script..."
./build.sh

# Replace package.json with the deployment version
cp package.json package.json.orig
cp package.json.deploy package.json

# Deploy to Vercel with custom env vars
echo "Deploying to Vercel..."
PATH="$PWD/bin:$PATH" vercel deploy --prod --force \
  --env PATH="$PWD/bin:/usr/local/bin:/usr/bin:/bin" \
  --env PYTHONPATH="/var/task/api/python_packages:/var/task" \
  --env PIP_NO_DEPS="1" \
  --env PIP_NO_INSTALL="1" \
  --build-env PATH="$PWD/bin:/usr/local/bin:/usr/bin:/bin" \
  --build-env PYTHONPATH="/var/task/api/python_packages:/var/task" \
  --build-env PIP_NO_DEPS="1" \
  --build-env PIP_NO_INSTALL="1"

# Restore the original package.json
mv package.json.orig package.json

echo "Deployment command completed." 