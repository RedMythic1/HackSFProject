# Tech Deep Dive

A self-contained tool for extracting, analyzing, and summarizing tech articles from the web.

## About

This directory contains a self-contained version of the Tech Deep Dive tool, which can:
1. Fetch and analyze articles from Hacker News
2. Generate summaries using local AI models
3. Create deep dive explorations of tech subjects
4. Present the information in nicely formatted HTML pages

## Self-Contained Design

All the functionality in this directory is self-contained - it doesn't require any external files or dependencies from the parent directory. This makes it easier to deploy and use independently.

## Requirements

- Python 3.9+
- Node.js and npm
- Required Python packages (install with `pip install -r requirements.txt`):
  - flask
  - flask-cors
  - requests
  - bs4 (BeautifulSoup)
  - llama-cpp-python
  - transformers (optional, for advanced summarization)
  - sentence-transformers (optional, for embedding generation)
  - duckduckgo-search
- Required Node.js packages:
  - @vercel/edge-config

## Files and Structure

- `server.py` - Flask server for the backend
- `ansys_local.py` - Article analysis functionality
- `hackernews_summarizer_local.py` - Helper for article summarization
- Shell scripts:
  - `start_all.sh` - Start both frontend and backend
  - `fetch_articles.sh` - Fetch and cache articles from Hacker News
  - `process_articles.sh` - Process articles with user interests
  - `manage.sh` - Combined interface for common operations
- Directories:
  - `local_cache/` - Cached articles and summaries (for local fallback)
  - `public/articles/` - Generated HTML files for viewing
  - `models/` - Directory for AI model files (you may need to add these manually)

## Vercel Edge Config Integration

This application now uses Vercel Edge Config for storing cache data, configuration, and article data. This provides several benefits over the previous Blob Storage approach:

1. Better performance with lower latency (data is stored at the network edge)
2. Simpler API for key-value storage
3. Reduced cost for small data items
4. Built-in integration with Vercel deployments
5. Improved global distribution of data

## Edge Config Setup

All cache data has been migrated to Vercel Edge Config with the following key prefixes:

- `articles/final_article_*.json` - Final processed articles
- `articles/summary_*.json` - Article summaries
- `articles/search_*.json` - Search results
- `articles/processed_articles.json` - List of processed article IDs

To use Edge Config for your local development, you must set up the required environment variables:

```bash
# Required environment variables for Edge Config
export EDGE_CONFIG="https://edge-config.vercel.com/ecfg_xsczamr0q3eodjuagxzwjiznqxxs?token=854495e2-1208-47c1-84a6-213468e23510"
export USE_EDGE_CONFIG="1"
```

You can run the `setup-edge-config.sh` script to set these variables automatically:

```bash
source setup-edge-config.sh
```

## Vercel Edge Config Tools

### Migrating from Blob Storage to Edge Config

If you were previously using Vercel Blob Storage, you can migrate your data to Edge Config using the included script:

```bash
# Set your Vercel Blob Storage token (required for reading from Blob Storage)
export BLOB_READ_WRITE_TOKEN="your_vercel_blob_token"

# Run the migration script
./migrate-to-edge-config.sh
```

This will migrate all your data from Blob Storage to Edge Config, update your environment variables, and configure your application to use Edge Config for storage.

### Accessing Edge Config Directly

You can view and manage data stored in Edge Config using the edge-config-utils.js tool:

```bash
# List all keys with a specific prefix
node tools/edge-config-utils.js list articles/final_article_

# Get data for a specific key
node tools/edge-config-utils.js get articles/final_article_12345.json

# Upload a file to Edge Config
node tools/edge-config-utils.js put articles/my-file.json path/to/local/file.json

# Delete a key
node tools/edge-config-utils.js delete articles/my-file.json

# Upload a directory to Edge Config
node tools/edge-config-utils.js upload-dir local_cache articles/

# Download items to a local directory
node tools/edge-config-utils.js download articles/ downloaded-files/
```

## Command Line Interface

The command line interface for article processing has been updated to use Edge Config:

```bash
# Cache articles without processing (only subjectizes)
./process_articles.sh cache

# Process articles with specified interests
./process_articles.sh process "technology, AI, science"

# List all cached articles
./process_articles.sh list
```

When EDGE_CONFIG is set, these commands will use Edge Config for storage. Otherwise, they will fall back to using the local cache.

## API Endpoints

The API now reads and writes directly to Edge Config:

```
GET /api/articles - List all articles
GET /api/article/:id - Get a specific article
GET /api/get-summary?id=:id - Get summary for a specific article
GET /api/get-search?query=:query - Get search results
```

For a full list of available API endpoints, see the API documentation.

## Troubleshooting

If you encounter issues with Edge Config, try the following:

1. Check that your EDGE_CONFIG environment variable is set correctly
2. Verify that the @vercel/edge-config package is installed: `npm install @vercel/edge-config`
3. Run the setup-edge-config.sh script to ensure all environment variables are set correctly
4. Check the application logs for any error messages related to Edge Config
5. Clear and rebuild your local cache if necessary

## Getting Started

1. Make sure you have all the required dependencies installed.

2. You may need to download the LLama model file. The default path is:
   ```
   models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
   ```
   You can set an alternative path using the `LLAMA_MODEL_PATH` environment variable.

3. Start the application:
   ```bash
   ./start_all.sh
   ```
   This will start both the backend server and frontend.

4. Fetch articles for analysis:
   ```bash
   ./fetch_articles.sh cache
   ```

5. Process articles with your interests:
   ```bash
   ./fetch_articles.sh process "AI, machine learning, web development"
   ```

## Using the CLI Tools

The `process_articles.sh` script provides a convenient command-line interface:

- Cache articles: `./process_articles.sh cache`
- Process with interests: `./process_articles.sh process "technology, AI, science"`
- List cached articles: `./process_articles.sh list`
- Show help: `./process_articles.sh help`

## Troubleshooting

- If you encounter model loading issues, make sure the LLama model file is in the correct location.
- Check the `server.log` and `frontend.log` files for detailed error information.
- Make sure the `local_cache` directory exists and is writable. 