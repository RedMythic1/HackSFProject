FROM node:16-slim as base

WORKDIR /app

# Install Python 3.9 using Debian's backports repository
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    lsof \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.9 from source
RUN wget https://www.python.org/ftp/python/3.9.16/Python-3.9.16.tgz \
    && tar xzf Python-3.9.16.tgz \
    && cd Python-3.9.16 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.9.16 Python-3.9.16.tgz

# Create symlinks for python
RUN ln -sf /usr/local/bin/python3.9 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip3.9 /usr/local/bin/pip3 && \
    ln -sf /usr/local/bin/pip3 /usr/local/bin/pip

# Copy package files and install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy the Python requirements separately
COPY requirements-fly.txt ./requirements.txt

# Install Python dependencies with optimized settings
RUN pip3 install --no-cache-dir --disable-pip-version-check -r requirements.txt

# Install g4f and pycryptodome directly to ensure they are compiled for this environment
RUN pip3 install --no-cache-dir pycryptodome==3.22.0 g4f==0.5.2.4

# Copy the rest of the application
COPY . .

# Create a .env file to specify we want to use the system-installed packages
RUN echo "USE_SYSTEM_PACKAGES=true" > .env

# Build the frontend
RUN npm run build

# Make scripts executable
RUN chmod +x start_fly.sh

# Expose the ports
EXPOSE 8080

# Start the application
CMD ["./start_fly.sh"] 