FROM node:16-slim as base

WORKDIR /app

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    lsof \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy the primary package.json (typescript-webpage)
COPY tw/tw-package.json ./package.json
# If an associated lock file for tw-package.json exists (e.g., tw/tw-package-lock.json), copy it:
# COPY tw/tw-package-lock.json ./package-lock.json

# Install ALL npm dependencies, including devDependencies for 'concurrently'
# NODE_ENV=production is set in fly.toml, which would skip devDependencies.
# --production=false ensures they are installed.
RUN npm install --production=false

# Copy Python requirements separately
# Ensure this path is correct, relative to project root.
COPY tw/requirements-fly.txt ./requirements.txt
# Install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt

# Copy the rest of the 'tw' application files which are used by tw-package.json
# This will now also include tw/.cache and tw/datasets if you move them into the local tw/ folder.
COPY tw/ ./

# Copy other necessary root files if any (e.g., the Python server, startup script)
# These might be used by a different process or if CMD is respected.
COPY server.py ./server.py
COPY tw/start_fly.sh ./start_fly.sh
RUN chmod +x ./start_fly.sh

# Build the frontend (uses 'build' script from the new /app/package.json)
RUN npm run build

# Expose the ports
EXPOSE 5001 8080

# Start the application
# This CMD might be overridden by Fly.io if it forces 'npm run start' from package.json.
# If 'npm run start' (which runs 'concurrently') is the goal, this CMD is secondary.
CMD ["./start_fly.sh"] 