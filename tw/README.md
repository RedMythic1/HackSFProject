# Tech Deep Dive

A self-contained tool for extracting, analyzing, and summarizing tech articles from the web.

## About

This directory contains a self-contained version of the Tech Deep Dive tool, which can:
1. Fetch and analyze articles from Hacker News
2. Generate summaries using AI models
3. Create deep dive explorations of tech subjects
4. Present the information in nicely formatted HTML pages

## Self-Contained Design

All the functionality in this directory is self-contained with a fully JavaScript-based implementation. This makes it easier to deploy to serverless environments like Vercel.

## Requirements

- Node.js 16+ and npm
- Required Node packages are installed automatically with `npm install`

## Files and Structure

- `api/server.js` - Main server implementation for the backend
- `api/index.js` - API routes and endpoints
- `src/index.ts` - Frontend TypeScript implementation
- Shell scripts:
  - `start_all.sh` - Start both frontend and backend
  - `fetch_articles.sh` - Fetch and cache articles from Hacker News
  - `migrate-to-blob.sh` - Migrate local cache to Vercel Blob storage
  - `manage.sh` - Combined interface for common operations
- Directories:
  - `.cache/` - Deprecated: Use Vercel Blob Storage instead
  - `dist/` - Compiled frontend files
  - `src/` - Frontend source code

## Storage

This application uses Vercel Blob Storage for all cache operations instead of the local filesystem. This provides several benefits:

1. Persistent storage across serverless function invocations
2. No dependency on ephemeral filesystem in Vercel's serverless environment
3. Data is accessible from any region or instance

All article data, summaries, and search results are stored in Vercel Blob Storage under the following prefixes:

- `articles/final_article_*` - Processed articles
- `articles/summary_*` - Article summaries
- `articles/search_*` - Search results

### Local Development

When running in a local environment, the application will still use Vercel Blob Storage through the API rather than accessing local file systems. Make sure to set up the BLOB_READ_WRITE_TOKEN environment variable for local development.

```sh
export BLOB_READ_WRITE_TOKEN="vercel_blob_rw_MzCMzRmJaiRlp3km_L5RVXS9InB9rTT1Aov2ZI4kzQFoT5S"
export BLOB_URL="https://mzcmzrmjairlp3km.public.blob.vercel-storage.com"
```

Or simply run:

```sh
source setup-env.sh
```

### Production Deployment

For production, the Vercel configuration automatically includes the blob token from the project environment variables.

## Vercel Blob Storage Tools

### Migrating Local Cache to Blob Storage

To upload local cache files to Vercel Blob Storage, you can use the included script:

```bash
# Run the migration script
./migrate-to-blob.sh
```

This will upload all JSON files from the `.cache` directory to Vercel Blob Storage.

## Getting Started

1. Make sure you have Node.js installed.

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   ```bash
   source setup-env.sh
   ```

4. Start the application:
   ```bash
   ./start_all.sh
   ```
   This will start both the backend server and frontend.

5. Fetch articles for analysis:
   ```bash
   ./fetch_articles.sh cache
   ```

6. Process articles with your interests:
   ```bash
   ./fetch_articles.sh process "AI, machine learning, web development"
   ```

## Troubleshooting

- If you encounter storage issues, make sure your BLOB_READ_WRITE_TOKEN is set correctly.
- Check the `server.log` and `frontend.log` files for detailed error information.
- Verify that @vercel/blob package is installed with `npm list @vercel/blob`.

# Client-Side Chart Rendering

To optimize deployment size for Vercel, chart generation has been moved from the server to the client:

1. Chart.js is used for rendering interactive price and balance charts in the browser
2. The backend API returns raw data (prices, buy/sell points, balances) instead of generating images
3. No matplotlib dependency on the server, reducing deployment size
4. Responsive charts that work across all devices
5. Interactive tooltips for better user experience

This approach reduces server load and Vercel deployment size while providing a better user experience. 