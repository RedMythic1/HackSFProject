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

# Upgrade pip and install essential tools
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Copy package files and install Node.js dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy the Python requirements separately
COPY requirements-fly.txt ./requirements.txt

# Install numpy FIRST to ensure all other packages are built against the same version
RUN pip3 install --no-cache-dir numpy==1.21.6

# Install torch separately as it's a large package
RUN pip3 install --no-cache-dir torch==1.13.1

# Now install the rest of the dependencies excluding numpy and torch which we already installed
RUN grep -v "numpy\|torch" requirements.txt > other_requirements.txt && \
    pip3 install --no-cache-dir -r other_requirements.txt && \
    rm other_requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories for stock prediction
RUN mkdir -p /app/stockbt/testing_bs/data_folder

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