# Tech Deep Dive

A tech article processing system that fetches, analyzes, and generates deep-dive content based on technical articles.

## Project Structure

This project is organized as follows:

- **Root Directory** - Core article processing engine
  - `ansys.py` - Main processing engine for fetching and analyzing articles
  - `hackernews_summarizer.py` - Extracts and summarizes content from Hacker News
  - `.cache/` - Cache for article processing data

- **Web Application** - Located in the `tw/` directory
  - `server.py` - Flask server providing API access to processed articles
  - `src/` - TypeScript frontend source
  - `.cache/` - Synchronized with parent directory's cache

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   cd tw
   npm install
   ```

2. **Run the cleanup script to ensure consistency**:
   ```bash
   ./cleanup.sh
   ```

3. **Start the web application**:
   ```bash
   cd tw
   ./start.sh
   ```

4. **Access the web interface**: 
   - Open http://localhost:3000 in your browser

## Deployment on Fly.io

1. **Install Flyctl CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Authenticate with Fly.io**:
   ```bash
   flyctl auth login
   ```

3. **Deploy the application**:
   ```bash
   flyctl deploy
   ```

4. **Open the deployed application**:
   ```bash
   flyctl open
   ```

## Development Process

1. **Article Processing**:
   - Core processing is handled by `ansys.py`
   - Articles are cached in `.cache/` directory
   - HTML files are generated in `final_articles/html/`

2. **Web Interface**:
   - Frontend developed in TypeScript (in `tw/src/`)
   - Backend API provided by Flask (in `tw/server.py`)
   - Frontend interacts with backend via REST API

## Maintenance

The `cleanup.sh` script helps maintain consistency by:
- Synchronizing cache directories
- Copying HTML files to the public directory
- Removing duplicate/backup files

Run it periodically to ensure the system remains consistent.

## License

MIT License 