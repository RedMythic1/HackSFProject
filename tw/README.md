# Tech Interests Analyzer

This application has been restructured to clearly separate the article processing from the web interface. Now the system uses:

1. Shell commands for all article processing
2. A shared cache for storing article data (in `/Users/avneh/Code/HackSFProject/tw/.cache`)
3. Web interface for viewing and accessing processed articles

## Architecture

The system is composed of two separate components:

1. **Article Processing** - Uses `ansys.py` (located in parent directory) to:
   - Cache articles from tech sources (subjectizing only)
   - Generate questions and answers
   - Process content for final articles
   - Store all results in the cache directory

2. **Web Interface** - Uses Node.js/Flask backend to:
   - Read from the shared cache
   - Display processed articles
   - Handle interest-based article scoring/matching
   - Provide a user interface for viewing content

## Usage

### Article Processing (Command Line)

All article processing is now done through the command line using the `process_articles.sh` script:

```bash
# Cache and subjectize articles only (first step) - does NOT generate questions/answers
./process_articles.sh cache

# Process articles with specific interests
./process_articles.sh process "technology, AI, programming, science"

# List all cached articles
./process_articles.sh list

# Show help
./process_articles.sh help
```

The process will:
1. Store cached articles in the `tw/.cache` directory
2. Generate processed articles
3. Create HTML files for the website

**Important note**: The `cache` command only subjectizes articles and caches the subject lines. It does NOT generate questions or final articles.

### Web Interface

The web interface reads from the cache and displays the content. It also handles interest-based scoring and matching.

To start the web server:

```bash
# Start the server
./start.sh
```

Then navigate to http://localhost:5001 in your browser.

## Workflow

The recommended workflow is:

1. Run `./process_articles.sh cache` to subjectize and cache articles
2. Start the web server with `./start.sh`
3. Access the web interface to view articles and enter your interests
4. The web interface will score and match articles based on your interests

## Directory Structure

- `tw/.cache/` - Shared cache directory containing all processed data
- `public/articles/` - HTML versions of final articles for web viewing
- `process_articles.sh` - CLI tool for article processing
- `server.py` - Backend server for web interface
- `src/` - Frontend source code

## Requirements

- Python 3.6+
- Node.js 14+
- Web browser

## Notes

- The article processing may take several minutes depending on the number of articles
- The web interface automatically refreshes when new articles are added to the cache
- All files are stored in the `tw/.cache` directory, not in the parent directory's cache 