# Tech Deep Dive

Tech Deep Dive is a tool that analyzes technology articles, generates insightful questions and deep dive content about the subjects covered in these articles.

## Project Structure

The project is organized into these main components:

- `ansys.py` - Core analysis engine for article processing and question generation
- `hackernews_summarizer.py` - Module for summarizing Hacker News articles
- `tw/` - Web application for interacting with the analysis engine
  - `server.py` - Flask server providing APIs for article management
  - `src/` - Frontend source code
  - `manage.sh` - All-in-one management script for the web application

## Quick Start

The easiest way to get started is through the web application:

1. Navigate to the `tw` directory:
   ```
   cd tw
   ```

2. Use the management script to set up and start the application:
   ```
   ./manage.sh setup  # Set up dependencies and environment
   ./manage.sh start  # Start the server and frontend
   ```

3. Open the application in your browser:
   ```
   http://localhost:9000
   ```

4. Cache articles and generate deep dive content:
   ```
   ./manage.sh cache      # Cache articles from Hacker News
   ./manage.sh questions  # Generate questions and deep dive content
   ```

## Using the Core Analysis Engine Directly

If you want to use the core analysis engine directly:

```python
python ansys.py
```

This will:
1. Prompt you for your interests
2. Fetch articles from Hacker News
3. Score them based on your interests
4. Process and analyze the most relevant articles
5. Generate deep dive content with questions and answers

## Requirements

- Python 3.8+
- Node.js 14+
- Required Python packages are listed in `requirements.txt`

## Development

For development, the most important files are:

- `ansys.py` - Core functionality and algorithms
- `tw/server.py` - Backend API server
- `tw/src/index.ts` - Frontend interface

See the README in the `tw/` directory for more details on the web application development. 