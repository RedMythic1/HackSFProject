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
  - `local_cache/` - Cached articles and summaries
  - `public/articles/` - Generated HTML files for viewing
  - `models/` - Directory for AI model files (you may need to add these manually)

## Storage

This application now uses Vercel Blob Storage for all cache operations instead of the local filesystem. This provides several benefits:

1. Persistent storage across serverless function invocations
2. No dependency on ephemeral filesystem in Vercel's serverless environment
3. Data is accessible from any region or instance

All local cache files have been migrated to Vercel Blob Storage under the following prefixes:

- `articles/final_article_*` - Processed articles
- `articles/summary_*` - Article summaries
- `articles/search_*` - Search results

### Local Development

When running in a local environment, the application will still use Vercel Blob Storage through the API rather than accessing local file systems. Make sure to set up the BLOB_READ_WRITE_TOKEN environment variable for local development.

```sh
export BLOB_READ_WRITE_TOKEN="your_token_here"
```

### Production Deployment

For production, the Vercel configuration automatically includes the blob token from the project environment variables.

## Vercel Blob Storage Tools

### Migrating Local Cache to Blob Storage

To upload local cache files to Vercel Blob Storage, you can use the included script:

```bash
# Set your Vercel Blob Storage token
export BLOB_READ_WRITE_TOKEN="your_vercel_blob_token"

# Run the upload script
./upload-cache.sh
```

This will upload all JSON files from the `local_cache` directory to Vercel Blob Storage.

### Accessing Blob Storage Directly

You can view and access all files stored in Vercel Blob Storage using these endpoints:

- List all blobs: `/api/list-blobs`
- View a specific blob: `/api/blob/articles/final_article_1234567890.json`

Example blob URLs will look like:
```
https://yourproject.public.blob.vercel-storage.com/articles/final_article_1234567890-randomstring.json
```

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