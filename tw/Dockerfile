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

# Copy package files and install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy the Python requirements separately
COPY requirements-fly.txt ./requirements.txt

# Install Python dependencies with optimized settings
RUN pip3 install --no-cache-dir --disable-pip-version-check -r requirements.txt

# Copy the rest of the application
COPY . .

# Build the frontend
RUN npm run build

# Make scripts executable
RUN chmod +x start_fly.sh

# Expose the ports
EXPOSE 8080

# Start the application
CMD ["./start_fly.sh"] 